"""Batched ESM-2 protein featurization for PSICHIC.

Replaces the serial per-sequence loop in PSICHIC/utils/protein_init.py
with length-binned batch processing. Target: 4-6x preprocessing speedup.

Design decisions:
- fp32 only: contact_map() hard-thresholds probabilities at 0.5 to build
  edge_index. fp16 drift produces discrete graph topology changes.
- repr_layers=[33] only: approach='last' uses only the final layer, so
  requesting all 33 layers wastes ~32x memory.
- Sequences >700 aa: preserve the original sliding-window merge logic
  exactly (cannot batch naively due to overlapping-window averaging).
- Adaptive batch sizes by length bin to avoid OOM on A100 80GB.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

# Length bins → max batch sizes (A100 80GB with ESM-2 650M ~2.5GB)
LENGTH_BIN_BATCH_SIZES: list[tuple[int, int]] = [
    (300, 8),    # <=300 residues
    (500, 4),    # 301-500
    (700, 2),    # 501-700
]
SLIDING_WINDOW_THRESHOLD = 700


def _load_esm_model(device: torch.device) -> tuple[Any, Any, Any]:
    """Load ESM-2 650M model, alphabet, and batch_converter."""
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def _get_batch_size_for_length(seq_len: int) -> int:
    """Return adaptive batch size based on sequence length."""
    for max_len, bs in LENGTH_BIN_BATCH_SIZES:
        if seq_len <= max_len:
            return bs
    return 1


def _bin_sequences(
    seqs: list[str],
) -> dict[str, list[str]]:
    """Group sequences into length bins for efficient batching.

    Returns dict mapping bin label to list of sequences, sorted by length
    within each bin to minimize padding waste.
    """
    bins: dict[str, list[str]] = {
        "short_300": [],
        "medium_500": [],
        "long_700": [],
        "sliding": [],
    }
    for seq in seqs:
        n = len(seq)
        if n <= 300:
            bins["short_300"].append(seq)
        elif n <= 500:
            bins["medium_500"].append(seq)
        elif n <= 700:
            bins["long_700"].append(seq)
        else:
            bins["sliding"].append(seq)

    # Sort within each bin by length (shortest first = less padding)
    for key in bins:
        bins[key].sort(key=len)

    return bins


def _process_short_batch(
    model: Any,
    batch_converter: Any,
    seqs: list[str],
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Process a batch of short sequences (<=700 aa) through ESM-2.

    Returns list of (token_representation, contact_prob_map) tuples,
    one per sequence, in the same order as input seqs.
    """
    # Prepare batch for ESM
    data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs)]
    _labels, _strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device, non_blocking=True)

    with torch.inference_mode():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # Extract per-sequence results
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, seq in enumerate(seqs):
        seq_len = len(seq)

        # Token representation: layer 33 only (approach='last')
        # Shape: (batch, seq_len+2, 1280) — +2 for BOS/EOS tokens
        token_repr = results["representations"][33][i].cpu().numpy()
        # Strip BOS/EOS tokens: keep positions [1 : seq_len+1]
        token_repr = token_repr[1: seq_len + 1]

        # Contact map: (batch, padded_len, padded_len)
        # ESM contact_head already removes BOS/EOS, so shape is (padded_len-2, padded_len-2)
        # But we need only the first seq_len x seq_len submatrix
        contact_map_proba = results["contacts"][i].cpu().numpy()
        contact_map_proba = contact_map_proba[:seq_len, :seq_len]

        outputs.append((token_repr, contact_map_proba))

    return outputs


def _process_sliding_window(
    model: Any,
    batch_converter: Any,
    seq: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a long sequence (>700 aa) using the original sliding-window merge.

    This is an exact copy of the logic from PSICHIC/utils/protein_init.py:200-262,
    with two changes: repr_layers=[33] instead of [1..33], and torch.inference_mode().
    """
    layer = 33
    dim = 1280
    pro_id = "A"

    contact_prob_map = np.zeros((len(seq), len(seq)), dtype=np.float32)
    token_representation = np.zeros((len(seq), dim), dtype=np.float32)
    interval = 350
    num_windows = math.ceil(len(seq) / interval)

    for s in range(num_windows):
        start = s * interval
        end = min((s + 2) * interval, len(seq))

        temp_seq = seq[start:end]
        temp_data = [(pro_id, temp_seq)]
        _labels, _strs, batch_tokens = batch_converter(temp_data)
        batch_tokens = batch_tokens.to(device, non_blocking=True)

        with torch.inference_mode():
            results = model(
                batch_tokens,
                repr_layers=[layer],
                return_contacts=True,
            )

        # --- Contact map merging (averaging overlaps) ---
        row, col = np.where(contact_prob_map[start:end, start:end] != 0)
        row = row + start
        col = col + start
        contact_prob_map[start:end, start:end] = (
            contact_prob_map[start:end, start:end]
            + results["contacts"][0].cpu().numpy()
        )
        contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0

        # --- Token representation merging (averaging overlaps) ---
        # With repr_layers=[33], results['representations'][33] has shape (1, L+2, dim)
        subtoken_repr = results["representations"][layer][0].cpu().numpy()
        subtoken_repr = subtoken_repr[1: len(temp_seq) + 1]

        trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
        trow = trow + start
        token_representation[start:end] = token_representation[start:end] + subtoken_repr
        token_representation[trow] = token_representation[trow] / 2.0

        if end == len(seq):
            break

    return token_representation, contact_prob_map


_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _replace_non_standard_residues(seq: str) -> str:
    """Replace any non-standard amino acid with X for ESM compatibility.

    Handles U (selenocysteine), B (Asx), O (pyrrolysine), J (Xle), Z (Glx),
    and any other character not in the standard 20.
    """
    return "".join(c if c in _STANDARD_AA else "X" for c in seq)


def batched_protein_init(
    seqs: list[str],
    device: torch.device | None = None,
    seq_feature_fn: Any = None,
    contact_map_fn: Any = None,
) -> dict[str, dict[str, Any]]:
    """Batch-process protein sequences through ESM-2 and build PSICHIC features.

    Drop-in replacement for PSICHIC/utils/protein_init.py:protein_init().
    Returns the same dict format keyed by sequence.

    Args:
        seqs: List of unique protein sequences.
        device: CUDA device. Defaults to cuda:0.
        seq_feature_fn: Function to compute residue features (default: import from PSICHIC).
        contact_map_fn: Function to build contact graph (default: import from PSICHIC).

    Returns:
        Dict[seq] → {seq, seq_feat, token_representation, num_nodes,
                      node_pos, edge_index, edge_weight}
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Lazy import PSICHIC utilities
    if seq_feature_fn is None or contact_map_fn is None:
        import sys as _sys
        from pathlib import Path as _Path

        _psichic_dir = str(_Path(__file__).resolve().parents[1] / "PSICHIC")
        if _psichic_dir not in _sys.path:
            _sys.path.insert(0, _psichic_dir)
        from utils.protein_init import contact_map as _contact_map
        from utils.protein_init import seq_feature as _seq_feature

        if seq_feature_fn is None:
            seq_feature_fn = _seq_feature
        if contact_map_fn is None:
            contact_map_fn = _contact_map

    # Clean sequences
    cleaned_seqs = [_replace_non_standard_residues(s) for s in seqs]
    # Map original → cleaned for output keying
    seq_map = dict(zip(seqs, cleaned_seqs))

    # Deduplicate
    unique_cleaned = list(dict.fromkeys(cleaned_seqs))
    log.info("Processing %d unique proteins (%d input)", len(unique_cleaned), len(seqs))

    if not unique_cleaned:
        return {}

    # Load ESM model
    t0 = time.perf_counter()
    model, _alphabet, batch_converter = _load_esm_model(device)
    log.info("ESM-2 loaded in %.1fs", time.perf_counter() - t0)

    # Bin sequences by length
    bins = _bin_sequences(unique_cleaned)
    bin_counts = {k: len(v) for k, v in bins.items() if v}
    log.info("Length bins: %s", bin_counts)

    # Process all sequences, collecting (token_repr, contact_prob_map) per sequence
    esm_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    t_start = time.perf_counter()

    # Process short/medium/long bins (<=700) in batches
    for bin_name in ["short_300", "medium_500", "long_700"]:
        bin_seqs = bins[bin_name]
        if not bin_seqs:
            continue

        max_len = max(len(s) for s in bin_seqs)
        batch_size = _get_batch_size_for_length(max_len)
        log.info(
            "Processing %s: %d seqs, bs=%d",
            bin_name, len(bin_seqs), batch_size,
        )

        for batch_start in range(0, len(bin_seqs), batch_size):
            batch = bin_seqs[batch_start: batch_start + batch_size]
            batch_outputs = _process_short_batch(model, batch_converter, batch, device)
            for seq, (token_repr, contact_proba) in zip(batch, batch_outputs):
                esm_results[seq] = (token_repr, contact_proba)

    # Process sliding window sequences (>700) one at a time
    sliding_seqs = bins["sliding"]
    if sliding_seqs:
        log.info("Processing sliding window: %d seqs (>700 aa)", len(sliding_seqs))
        for i, seq in enumerate(sliding_seqs):
            token_repr, contact_proba = _process_sliding_window(
                model, batch_converter, seq, device,
            )
            esm_results[seq] = (token_repr, contact_proba)
            if (i + 1) % 100 == 0:
                log.info("  Sliding window: %d/%d done", i + 1, len(sliding_seqs))

    esm_time = time.perf_counter() - t_start
    log.info(
        "ESM-2 processing complete: %.1fs (%.3f s/protein)",
        esm_time, esm_time / max(len(unique_cleaned), 1),
    )

    # Free ESM model
    del model
    torch.cuda.empty_cache()

    # Build PSICHIC feature dicts, keyed by ORIGINAL input sequence so that
    # callers (precompute_proteins, cache lookups) can use the same key they
    # passed in.  Internally we deduplicate on the cleaned sequence so that
    # U/B-containing variants are only computed once.
    result_dict: dict[str, dict[str, Any]] = {}
    cleaned_features: dict[str, dict[str, Any]] = {}  # cleaned seq → features (dedup)

    for orig_seq in seqs:
        cleaned = seq_map[orig_seq]
        if orig_seq in result_dict:
            # Exact duplicate input — skip
            continue

        if cleaned not in cleaned_features:
            token_repr_np, contact_proba_np = esm_results[cleaned]
            contact_proba_tensor = torch.from_numpy(contact_proba_np)

            if len(contact_proba_tensor) != len(cleaned):
                raise RuntimeError(
                    f"Contact map length {len(contact_proba_tensor)} != seq length {len(cleaned)}"
                )

            edge_index, edge_weight = contact_map_fn(contact_proba_tensor)
            sf = seq_feature_fn(cleaned)

            cleaned_features[cleaned] = {
                "seq": cleaned,
                "seq_feat": torch.from_numpy(sf),
                "token_representation": torch.from_numpy(token_repr_np).half(),
                "num_nodes": len(cleaned),
                "node_pos": torch.arange(len(cleaned)).reshape(-1, 1),
                "edge_index": edge_index,
                "edge_weight": edge_weight,
            }

        result_dict[orig_seq] = cleaned_features[cleaned]

    log.info("Built feature dicts for %d proteins", len(result_dict))
    return result_dict
