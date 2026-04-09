#!/usr/bin/env python3
"""PSICHIC optimized screening pipeline.

Takes any FASTA file (protein sequences) + any SMILES file (CSV, SMI, or
tar.gz-wrapped CSV) and screens all protein-ligand pairs using the
optimized PSICHIC model with batched ESM-2 preprocessing.

Pipeline:
  1. Parse inputs (FASTA + SMILES)
  2. Precompute ligand features (LMDB cache + multiprocessing, CPU-only)
  3. Precompute protein embeddings (batched ESM-2, cached to disk)
  4. Load PSICHIC model
  5. Stream scoring in ligand chunks (AMP + bs=128)
  6. Output ranked CSV + optional interpretation

Usage:
    # Basic screening
    python inference.py --fasta proteins.fasta --smiles ligands.csv

    # With a tar.gz SMILES file
    python inference.py --fasta proteins.fasta --smiles zinc20_smiles.csv.tar.gz

    # Custom output and batch size
    python inference.py --fasta proteins.fasta --smiles ligands.smi \\
        --output results.csv --batch-size 256

    # Large-scale screening with top-K
    python inference.py --fasta proteins.fasta --smiles zinc.csv \\
        --top-k 100 --save-interpret

    # Multitask model (adds classification columns)
    python inference.py --fasta proteins.fasta --smiles ligands.csv \\
        --model multitask_PSICHIC

    # Skip protein cache (recompute every time)
    python inference.py --fasta proteins.fasta --smiles ligands.csv --no-cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import itertools
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import multiprocessing

import numpy as np
import torch
from torch_geometric.data import Batch

# spawn start method is set in main() to avoid side effects on import.

# ---------------------------------------------------------------------------
# Path setup — BEFORE any CUDA init
# ---------------------------------------------------------------------------
PROD_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROD_DIR.parent  # huankoh/PSICHIC repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROD_DIR))

from utils.dataset import MultiGraphData  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_PROTEIN_CACHE_DIR = PROD_DIR / "cache" / "proteins"
DEFAULT_LIGAND_CACHE_DIR = PROD_DIR / "cache"
LIGAND_LMDB_NAME = "ligands_v1.lmdb"

# Scale guard threshold
MAX_PAIRS_WITHOUT_TOPK = 10_000_000

# Streaming chunk size (number of original SMILES per chunk)
LIGAND_CHUNK_SIZE = 10_000

# Multiprocessing chunk size (SMILES per worker task).
# Large chunks reduce the number of futures (and shared-memory FDs) in flight.
MP_CHUNK_SIZE = 2000


def _batch_iter(it, n: int):
    """Yield successive n-sized lists from an iterator without materializing all at once."""
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into {header_id: sequence}.

    Handles multi-line sequences. The header ID is the first whitespace-
    delimited token after '>'.
    """
    proteins: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    proteins[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        proteins[current_id] = "".join(current_seq)

    log.info("Parsed %d protein(s) from %s", len(proteins), path.name)
    return proteins


def parse_smiles(path: Path) -> list[str]:
    """Parse SMILES strings from CSV, SMI, or tar.gz-wrapped CSV.

    Supported formats:
    - .csv: expects a column named 'smiles' or 'SMILES' (case-insensitive)
    - .smi: one SMILES per line (optionally followed by whitespace + name)
    - .tar.gz / .tgz: extracts the first CSV inside, then parses as CSV

    Returns deduplicated list of SMILES strings.
    """
    actual_path = path
    tmp_dir = None

    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        log.info("Extracting tar.gz archive: %s", path.name)
        tmp_dir = tempfile.mkdtemp(prefix="psichic_smiles_")
        with tarfile.open(path, "r:gz") as tar:
            tmp_dir_real = os.path.realpath(tmp_dir)
            members = []
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                # Reject absolute paths and path traversal components.
                # os.path.join with an absolute member name would escape tmp_dir,
                # so we must check before constructing the destination path.
                dest = os.path.realpath(os.path.join(tmp_dir_real, m.name))
                if not dest.startswith(tmp_dir_real + os.sep):
                    log.warning("Skipping unsafe tar member: %r", m.name)
                    continue
                members.append(m)
            # Filter out macOS resource fork files (._* prefix)
            members = [m for m in members if not os.path.basename(m.name).startswith("._")]
            csv_members = [m for m in members if m.name.endswith(".csv")]
            if not csv_members:
                smi_members = [m for m in members if m.name.endswith(".smi")]
                target = smi_members[0] if smi_members else members[0]
            else:
                target = csv_members[0]
            tar.extract(target, path=tmp_dir, filter="data")
            actual_path = Path(tmp_dir) / target.name
            log.info("Extracted: %s", actual_path.name)

    try:
        smiles_list = _parse_smiles_file(actual_path)
    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return smiles_list


def _parse_smiles_file(path: Path) -> list[str]:
    """Parse SMILES from a plain CSV or SMI file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _parse_smiles_csv(path)
    elif suffix in (".smi", ".smiles"):
        return _parse_smiles_smi(path)
    else:
        try:
            return _parse_smiles_csv(path)
        except (KeyError, csv.Error):
            return _parse_smiles_smi(path)


def _parse_smiles_csv(path: Path) -> list[str]:
    """Parse SMILES from a CSV file with a 'smiles' column.

    Uses streaming csv.reader to avoid loading the entire file into memory,
    which matters for ZINC-scale files (tens of millions of rows).
    """
    smiles_col_idx: int | None = None
    smiles_col_name: str = ""
    seen: set[str] = set()
    unique: list[str] = []

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV file: {path}") from None

        for i, col in enumerate(header):
            if col.strip().lower() in ("smiles", "smi", "canonical_smiles", "compound_smiles"):
                smiles_col_idx = i
                smiles_col_name = col.strip()
                break
        if smiles_col_idx is None:
            smiles_col_idx = 0
            smiles_col_name = header[0].strip()
            log.warning("No 'smiles' column found, using first column: '%s'", smiles_col_name)

        for row in reader:
            if smiles_col_idx >= len(row):
                continue
            s = row[smiles_col_idx].strip()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)

    log.info("Parsed %d unique SMILES from %s (column '%s')", len(unique), path.name, smiles_col_name)
    return unique


def _parse_smiles_smi(path: Path) -> list[str]:
    """Parse SMILES from a .smi file (one per line, optional name after space)."""
    seen: set[str] = set()
    unique: list[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            if smi and smi not in seen:
                seen.add(smi)
                unique.append(smi)

    log.info("Parsed %d unique SMILES from %s", len(unique), path.name)
    return unique


# ---------------------------------------------------------------------------
# Canonical SMILES utilities
# ---------------------------------------------------------------------------


def _build_canonical_map(smiles_list: list[str]) -> tuple[dict[str, str], list[str]]:
    """Build mapping from original SMILES to canonical form.

    Returns:
        canonical_map: {original_smiles: canonical_smiles}
        failed: list of unparseable SMILES
    """
    from rdkit import Chem
    canonical_map: dict[str, str] = {}
    failed: list[str] = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(smi)
            continue
        canonical_map[smi] = Chem.MolToSmiles(mol)

    if failed:
        log.warning("Failed to canonicalize %d SMILES (first 10): %s", len(failed), failed[:10])

    unique_canonical = len(set(canonical_map.values()))
    log.info(
        "Canonical map: %d input → %d unique canonical (%d deduped)",
        len(canonical_map), unique_canonical, len(canonical_map) - unique_canonical,
    )
    return canonical_map, failed


# ---------------------------------------------------------------------------
# Protein preprocessing (batched ESM-2 with disk cache)
# ---------------------------------------------------------------------------


def _seq_hash(seq: str) -> str:
    """SHA-256 hash of a protein sequence for cache keying."""
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def precompute_proteins(
    proteins: dict[str, str],
    device: torch.device,
    cache_dir: Path | None = DEFAULT_PROTEIN_CACHE_DIR,
) -> dict[str, dict[str, Any]]:
    """Precompute protein features using batched ESM-2.

    Checks disk cache first; only computes missing proteins. Saves new
    results to cache for future runs.

    Returns dict keyed by protein sequence -> feature dict.
    """
    unique_seqs = list(set(proteins.values()))
    log.info("Total unique protein sequences: %d", len(unique_seqs))

    cached: dict[str, dict] = {}
    missing_seqs: list[str] = []

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for seq in unique_seqs:
            h = _seq_hash(seq)
            shard_path = cache_dir / f"{h}.pt"
            if shard_path.exists():
                features = torch.load(shard_path, weights_only=True, map_location="cpu")
                features["seq_feat"] = features["seq_feat"].float()
                features["token_representation"] = features["token_representation"].float()
                features["num_nodes"] = len(features["seq"])
                features["node_pos"] = torch.arange(len(features["seq"])).reshape(-1, 1)
                features["edge_weight"] = features["edge_weight"].float()
                cached[seq] = features
            else:
                missing_seqs.append(seq)
        log.info("Protein cache: %d hit, %d miss", len(cached), len(missing_seqs))
    else:
        missing_seqs = unique_seqs
        log.info("Cache disabled, computing all %d proteins", len(missing_seqs))

    if missing_seqs:
        from esm_batched.esm_batched import batched_protein_init

        t0 = time.perf_counter()
        new_features = batched_protein_init(missing_seqs, device=device)
        elapsed = time.perf_counter() - t0
        log.info(
            "Batched ESM-2: %d proteins in %.1fs (%.2f proteins/s)",
            len(missing_seqs), elapsed, len(missing_seqs) / elapsed if elapsed > 0 else 0,
        )

        if cache_dir is not None:
            for seq, feat in new_features.items():
                h = _seq_hash(seq)
                shard_path = cache_dir / f"{h}.pt"
                fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".pt.tmp")
                os.close(fd)
                try:
                    torch.save(feat, tmp_path)
                    os.rename(tmp_path, str(shard_path))
                except Exception:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
            log.info("Saved %d protein shards to cache", len(new_features))

        for seq, feat in new_features.items():
            feat["seq_feat"] = feat["seq_feat"].float()
            feat["token_representation"] = feat["token_representation"].float()
            feat["num_nodes"] = len(feat["seq"])
            feat["node_pos"] = torch.arange(len(feat["seq"])).reshape(-1, 1)
            feat["edge_weight"] = feat["edge_weight"].float()

        cached.update(new_features)

    return cached


# ---------------------------------------------------------------------------
# Ligand preprocessing — post-transform helper
# ---------------------------------------------------------------------------


def _post_transform_ligand(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply cache_transform to raw ligand_init output (same as ProteinMoleculeDataset)."""
    v = dict(raw)  # shallow copy to avoid mutating input
    v["atom_idx"] = v["atom_idx"].long().view(-1, 1)
    v["atom_feature"] = v["atom_feature"].float()
    adj = v["bond_feature"].long()
    mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    v["atom_edge_index"] = mol_edge_index
    v["atom_edge_attr"] = adj[mol_edge_index[0], mol_edge_index[1]].long()
    v["atom_num_nodes"] = v["atom_idx"].shape[0]
    v["x_clique"] = v["x_clique"].long().view(-1, 1)
    v["clique_num_nodes"] = v["x_clique"].shape[0]
    v["tree_edge_index"] = v["tree_edge_index"].long()
    v["atom2clique_index"] = v["atom2clique_index"].long()
    return v


# ---------------------------------------------------------------------------
# LMDB ligand cache
# ---------------------------------------------------------------------------


def _open_lmdb(path: Path, map_size: int = 10 * 1024**3) -> Any:
    """Open or create an LMDB environment with secure permissions."""
    import lmdb
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Enforce permissions even if directories already existed with looser perms
    os.chmod(path.parent, 0o700)
    env = lmdb.open(str(path), map_size=map_size)
    if path.exists():
        os.chmod(path, 0o700)
    return env


def _lmdb_put_batch(env: Any, items: list[tuple[bytes, bytes]]) -> int:
    """Write a batch of (key, value) pairs to LMDB, auto-resizing on MapFullError.

    Returns the number of items that failed to write (skipped, not crashed).
    """
    import lmdb
    skipped = 0
    while True:
        try:
            with env.begin(write=True) as txn:
                for key, value in items:
                    try:
                        txn.put(key, value)
                    except (lmdb.BadValsizeError, lmdb.Error) as exc:
                        log.warning("LMDB put skipped (key=%d B, val=%d B): %s",
                                    len(key), len(value), exc)
                        skipped += 1
            return skipped
        except lmdb.MapFullError:
            new_size = env.info()["map_size"] * 2
            log.info("LMDB MapFullError — resizing to %d GB", new_size // (1024**3))
            env.set_mapsize(new_size)


# Top-level function for multiprocessing (must be picklable)
def _process_ligand_chunk(smiles_chunk: list[str]) -> list[tuple[str, bytes | None, str | None]]:
    """Process a chunk of SMILES into serialized feature dicts.

    Returns serialized bytes (not raw tensors) to avoid shared-memory FD
    exhaustion when sending results through multiprocessing queues.

    Each entry: (canonical_smiles, serialized_bytes_or_None, error_or_None)
    Must be top-level and import inside body for spawn worker compatibility.
    """
    # Import within worker to avoid CUDA fork issues
    # Add repo root (parent of PSICHIC-prod/) so utils/ is importable
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.ligand_init import smiles2graph
    import torch as _torch

    results: list[tuple[str, bytes | None, str | None]] = []
    for smi in smiles_chunk:
        try:
            raw = smiles2graph(smi)
            transformed = _post_transform_ligand(raw)
            # Serialize to bytes inside worker to avoid shared-memory FDs
            buf = io.BytesIO()
            _torch.save(transformed, buf)
            results.append((smi, buf.getvalue(), None))
        except Exception as exc:
            results.append((smi, None, str(exc)))
    return results


def precompute_ligands_lmdb(
    canonical_smiles_set: set[str],
    cache_dir: Path,
    num_workers: int,
) -> tuple[Path, int, int]:
    """Precompute ligand features into LMDB, using multiprocessing.

    Must be called BEFORE any CUDA initialization.

    Returns:
        lmdb_path: Path to LMDB directory
        hits: number of cache hits
        misses: number computed
    """
    lmdb_path = cache_dir / LIGAND_LMDB_NAME
    env = _open_lmdb(lmdb_path)

    try:
        # Check cache hits
        missing: list[str] = []
        hits = 0
        with env.begin() as txn:
            for csmi in canonical_smiles_set:
                if txn.get(_smiles_hash(csmi).encode("utf-8")) is not None:
                    hits += 1
                else:
                    missing.append(csmi)

        log.info("LMDB ligand cache: %d hit, %d miss", hits, len(missing))

        if not missing:
            return lmdb_path, hits, 0

        # Process missing SMILES with multiprocessing
        if num_workers <= 0:
            num_workers = max(1, (os.cpu_count() or 1) - 2)

        log.info(
            "Computing %d ligand features (%d workers, chunk_size=%d)...",
            len(missing), num_workers, MP_CHUNK_SIZE,
        )

        chunks = [missing[i:i + MP_CHUNK_SIZE] for i in range(0, len(missing), MP_CHUNK_SIZE)]
        computed = 0
        failed_count = 0
        t0 = time.perf_counter()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_ligand_chunk, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                results = future.result()
                # Collect items for batch LMDB write
                write_items: list[tuple[bytes, bytes]] = []
                for csmi, feat_bytes, error in results:
                    if error is not None:
                        failed_count += 1
                    else:
                        key = _smiles_hash(csmi).encode("utf-8")
                        write_items.append((key, feat_bytes))
                        computed += 1

                if write_items:
                    _lmdb_put_batch(env, write_items)

                # Progress
                done_chunks = sum(1 for f in futures if f.done())
                if done_chunks % max(1, len(chunks) // 10) == 0 or done_chunks == len(chunks):
                    elapsed = time.perf_counter() - t0
                    rate = computed / elapsed if elapsed > 0 else 0
                    log.info(
                        "  Ligands: %d/%d computed (%.0f/s), %d failed",
                        computed, len(missing), rate, failed_count,
                    )
        elapsed = time.perf_counter() - t0
        log.info(
            "Ligand preprocessing: %d computed, %d failed in %.1fs (%.0f/s)",
            computed, failed_count, elapsed, computed / elapsed if elapsed > 0 else 0,
        )
    finally:
        env.close()

    return lmdb_path, hits, computed


def _lmdb_batch_get(lmdb_path: Path, keys: list[str]) -> dict[str, dict[str, Any]]:
    """Batch-read post-transformed ligand features from LMDB.

    Values are deserialized via torch.load(weights_only=True).

    NOTE: lock=False is used because reads only happen after all writes are
    complete (Step 5 runs after Step 2). Do NOT call this concurrently with
    precompute_ligands_lmdb or another writer — torn pages may result.
    """
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    try:
        result: dict[str, dict[str, Any]] = {}
        with env.begin() as txn:
            for key in keys:
                raw = txn.get(_smiles_hash(key).encode("utf-8"))
                if raw is not None:
                    result[key] = torch.load(
                        io.BytesIO(raw), weights_only=True, map_location="cpu",
                    )
    finally:
        env.close()
    return result


def precompute_ligands_memory(
    canonical_smiles_set: set[str],
    num_workers: int,
) -> dict[str, dict[str, Any]]:
    """In-memory ligand featurization (no LMDB). For small runs or --no-ligand-cache."""
    if len(canonical_smiles_set) > 50_000:
        raise ValueError(
            f"--no-ligand-cache with {len(canonical_smiles_set)} unique ligands exceeds "
            f"50K limit. Remove --no-ligand-cache to use LMDB caching."
        )

    smiles_list = list(canonical_smiles_set)
    if num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 1) - 2)

    log.info("Computing %d ligand features in-memory (%d workers)...", len(smiles_list), num_workers)

    chunks = [smiles_list[i:i + MP_CHUNK_SIZE] for i in range(0, len(smiles_list), MP_CHUNK_SIZE)]
    result: dict[str, dict[str, Any]] = {}
    failed_count = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_ligand_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            for csmi, feat_bytes, error in future.result():
                if error is not None:
                    failed_count += 1
                else:
                    result[csmi] = torch.load(
                        io.BytesIO(feat_bytes), weights_only=True, map_location="cpu",
                    )

    elapsed = time.perf_counter() - t0
    log.info(
        "In-memory ligand features: %d OK, %d failed in %.1fs",
        len(result), failed_count, elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load PSICHIC model with pretrained weights.

    Args:
        model_name: one of 'PDBv2020_PSICHIC' or 'multitask_PSICHIC'
        device: CUDA device
    """
    from psichic_model import PSICHIC  # noqa: E402

    weights_dir = REPO_ROOT / "trained_weights" / model_name

    with open(weights_dir / "config.json") as f:
        config = json.load(f)
    degree_dict = torch.load(
        weights_dir / "degree.pt", map_location="cpu", weights_only=True,
    )
    params = config["params"]
    tasks = config["tasks"]
    model = PSICHIC(
        mol_deg=degree_dict["ligand_deg"],
        prot_deg=degree_dict["protein_deg"],
        mol_in_channels=params["mol_in_channels"],
        prot_in_channels=params["prot_in_channels"],
        prot_evo_channels=params["prot_evo_channels"],
        hidden_channels=params["hidden_channels"],
        pre_layers=params["pre_layers"],
        post_layers=params["post_layers"],
        aggregators=params["aggregators"],
        scalers=params["scalers"],
        total_layer=params["total_layer"],
        K=params["K"],
        heads=params["heads"],
        dropout=params["dropout"],
        dropout_attn_score=params["dropout_attn_score"],
        regression_head=tasks["regression_task"],
        classification_head=tasks["classification_task"],
        multiclassification_head=tasks["mclassification_task"],
        device=device,
    ).to(device)

    state_dict = torch.load(
        weights_dir / "model.pt", map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    log.info(
        "PSICHIC model '%s' loaded (%.1fM params, tasks: reg=%s cls=%s mcls=%s)",
        model_name,
        sum(p.numel() for p in model.parameters()) / 1e6,
        tasks["regression_task"],
        tasks["classification_task"],
        tasks["mclassification_task"],
    )
    return model


def _model_has_classification(model_name: str) -> tuple[bool, bool]:
    """Check if model has classification/multiclassification heads."""
    weights_dir = REPO_ROOT / "trained_weights" / model_name
    with open(weights_dir / "config.json") as f:
        config = json.load(f)
    tasks = config["tasks"]
    has_cls = bool(tasks.get("classification_task", False))
    has_mcls = bool(tasks.get("mclassification_task", 0))
    return has_cls, has_mcls


# ---------------------------------------------------------------------------
# Pair building + Data object construction
# ---------------------------------------------------------------------------


def build_data_object(mol: dict, prot: dict) -> MultiGraphData:
    """Build a single MultiGraphData object from preprocessed mol/prot dicts."""
    return MultiGraphData(
        mol_x=mol["atom_idx"],
        mol_x_feat=mol["atom_feature"],
        mol_edge_index=mol["atom_edge_index"],
        mol_edge_attr=mol["atom_edge_attr"],
        mol_num_nodes=mol["atom_num_nodes"],
        clique_x=mol["x_clique"],
        clique_edge_index=mol["tree_edge_index"],
        atom2clique_index=mol["atom2clique_index"],
        clique_num_nodes=mol["clique_num_nodes"],
        prot_node_aa=prot["seq_feat"],
        prot_node_evo=prot["token_representation"],
        prot_node_pos=prot["node_pos"],
        prot_seq=prot["seq"],
        prot_edge_index=prot["edge_index"],
        prot_edge_weight=prot["edge_weight"],
        prot_num_nodes=prot["num_nodes"],
        reg_y=None,
        cls_y=None,
        mcls_y=None,
        mol_key="",
        prot_key="",
    )


def _run_forward(
    model: torch.nn.Module,
    batch: Batch,
    use_amp: bool,
    save_cluster: bool = False,
) -> tuple:
    """Run model forward on a batch. Returns (reg_pred, cls_pred, mcls_pred, attention_dict)."""
    kwargs = dict(
        mol_x=batch.mol_x,
        mol_x_feat=batch.mol_x_feat,
        bond_x=batch.mol_edge_attr,
        atom_edge_index=batch.mol_edge_index,
        clique_x=batch.clique_x,
        clique_edge_index=batch.clique_edge_index,
        atom2clique_index=batch.atom2clique_index,
        residue_x=batch.prot_node_aa,
        residue_evo_x=batch.prot_node_evo,
        residue_edge_index=batch.prot_edge_index,
        residue_edge_weight=batch.prot_edge_weight,
        mol_batch=batch.mol_x_batch,
        prot_batch=batch.prot_node_aa_batch,
        clique_batch=batch.clique_x_batch,
        save_cluster=save_cluster,
    )

    if use_amp:
        with torch.amp.autocast(device_type="cuda"):
            reg_pred, cls_pred, mcls_pred, _sp, _o, _cl, attention_dict = model(**kwargs)
    else:
        reg_pred, cls_pred, mcls_pred, _sp, _o, _cl, attention_dict = model(**kwargs)

    return reg_pred, cls_pred, mcls_pred, attention_dict


# ---------------------------------------------------------------------------
# Streaming screening
# ---------------------------------------------------------------------------


def _load_chunk_ligands(
    canonical_smiles: list[str],
    lmdb_path: Path | None,
    memory_ligands: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Load ligand features for a chunk from LMDB or memory cache."""
    unique_canonical = list(set(canonical_smiles))
    if lmdb_path is not None:
        return _lmdb_batch_get(lmdb_path, unique_canonical)

    chunk_ligands: dict[str, dict[str, Any]] = {}
    for k in unique_canonical:
        if k in memory_ligands:
            chunk_ligands[k] = memory_ligands[k]
        else:
            log.warning("Canonical SMILES missing from memory cache: %s", k)
    return chunk_ligands


def _score_batch(
    batch_pairs: list[tuple[str, str, str, str]],
    chunk_ligands: dict[str, dict[str, Any]],
    protein_features: dict[str, dict],
    model: torch.nn.Module,
    use_amp: bool,
    has_cls: bool,
    has_mcls: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build batch, run forward pass, extract predictions.

    Returns (reg_np, cls_np_or_None, mcls_np_or_None).
    """
    data_objects: list[MultiGraphData] = []
    for _pid, prot_seq, can_smi, _orig_smi in batch_pairs:
        prot = protein_features[prot_seq]
        mol = chunk_ligands[can_smi]
        data_objects.append(build_data_object(mol, prot))

    batch = Batch.from_data_list(
        data_objects,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
    ).to(device)

    reg_pred, cls_pred, mcls_pred, _ = _run_forward(
        model, batch, use_amp, save_cluster=False,
    )

    # atleast_1d guards against squeeze on batch_size=1
    reg_np = (
        np.atleast_1d(reg_pred.squeeze().detach().cpu().float().numpy())
        if reg_pred is not None
        else np.zeros(len(batch_pairs))
    )

    cls_np = None
    if cls_pred is not None and has_cls:
        cls_np = np.atleast_1d(
            torch.sigmoid(cls_pred).squeeze().detach().cpu().float().numpy()
        )

    mcls_np = None
    if mcls_pred is not None and has_mcls:
        mcls_np = torch.softmax(mcls_pred, dim=-1).detach().cpu().float().numpy()

    return reg_np, cls_np, mcls_np


def _safe_protein_id(protein_id: str) -> str:
    """Sanitize protein_id for filesystem use."""
    safe = re.sub(r'[/\\:*?"<>|]', "_", protein_id)
    # Re-check after substitution: ".." cannot survive the regex above (/ and \
    # are replaced), but guard anyway for clarity.
    if ".." in safe:
        raise ValueError(f"Unsafe protein_id (contains '..'): {protein_id!r}")
    # Reject names that resolve to the current or parent directory.
    if not safe or safe.strip(".") == "":
        raise ValueError(f"Unsafe protein_id (resolves to dot path): {protein_id!r}")
    return safe


def _smiles_hash(smiles: str) -> str:
    """SHA-256 hash of SMILES string (for pair_id construction)."""
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def screen_streaming_topk(
    model: torch.nn.Module,
    protein_features: dict[str, dict],
    proteins: dict[str, str],
    canonical_map: dict[str, str],
    valid_smiles: list[str],
    lmdb_path: Path | None,
    memory_ligands: dict[str, dict[str, Any]] | None,
    top_k: int,
    batch_size: int,
    use_amp: bool,
    has_cls: bool,
    has_mcls: bool,
    device: torch.device,
) -> dict[str, list[tuple[float, str, float | None, tuple[float, ...] | None]]]:
    """Stream scoring with per-protein top-K heaps.

    Returns:
        heaps: {protein_id: [(neg_score, original_smiles, cls_pred, mcls_pred_tuple)]}
        Heap entries use negative scores for max-heap via heapq (min-heap).
    """
    num_proteins = len(proteins)
    num_ligands = len(valid_smiles)
    total_pairs = num_proteins * num_ligands
    log.info(
        "Streaming top-%d: %d proteins x %d ligands = %d pairs",
        top_k, num_proteins, num_ligands, total_pairs,
    )

    # Per-protein min-heaps. Entries: (score, counter, orig_smi, cls_val, mcls_val).
    # Counter is a monotonic tiebreaker so heapq never compares heterogeneous
    # cls_val/mcls_val fields (float vs None) when scores tie.
    heaps: dict[str, list] = {pid: [] for pid in proteins}
    heap_counter = itertools.count()
    scored = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for chunk_start in range(0, num_ligands, LIGAND_CHUNK_SIZE):
            chunk_smiles = valid_smiles[chunk_start:chunk_start + LIGAND_CHUNK_SIZE]
            chunk_canonical = [canonical_map[s] for s in chunk_smiles]
            chunk_ligands = _load_chunk_ligands(chunk_canonical, lmdb_path, memory_ligands)

            pair_iter = (
                (prot_id, prot_seq, can_smi, orig_smi)
                for prot_id, prot_seq in proteins.items()
                for orig_smi, can_smi in zip(chunk_smiles, chunk_canonical)
                if can_smi in chunk_ligands
            )

            for batch_pairs in _batch_iter(pair_iter, batch_size):
                reg_np, cls_np, mcls_np = _score_batch(
                    batch_pairs, chunk_ligands, protein_features,
                    model, use_amp, has_cls, has_mcls, device,
                )

                for i, (prot_id, _seq, _csmi, orig_smi) in enumerate(batch_pairs):
                    score = float(reg_np[i])
                    cls_val = float(cls_np[i]) if cls_np is not None else None
                    mcls_val = tuple(mcls_np[i].tolist()) if mcls_np is not None else None

                    heap = heaps[prot_id]
                    entry = (score, next(heap_counter), orig_smi, cls_val, mcls_val)
                    if len(heap) < top_k:
                        heapq.heappush(heap, entry)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, entry)

                scored += len(batch_pairs)

            elapsed = time.perf_counter() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            log.info("  Scored %d / %d pairs (%.0f pairs/s)", scored, total_pairs, rate)

    total_time = time.perf_counter() - t0
    log.info(
        "Streaming complete: %d pairs in %.1fs (%.0f pairs/s)",
        scored, total_time, scored / total_time if total_time > 0 else 0,
    )
    return heaps


def screen_all(
    model: torch.nn.Module,
    protein_features: dict[str, dict],
    proteins: dict[str, str],
    canonical_map: dict[str, str],
    valid_smiles: list[str],
    lmdb_path: Path | None,
    memory_ligands: dict[str, dict[str, Any]] | None,
    batch_size: int,
    use_amp: bool,
    has_cls: bool,
    has_mcls: bool,
    device: torch.device,
    output_path: Path,
) -> int:
    """Score all pairs, write results to CSV as we go. For ≤10M pairs without --top-k.

    Returns total rows written.
    """
    num_proteins = len(proteins)
    num_ligands = len(valid_smiles)
    total_pairs = num_proteins * num_ligands

    log.info("Scoring all %d pairs (streaming to CSV)...", total_pairs)

    # Determine CSV columns
    fieldnames = ["protein_id", "smiles", "prediction", "protein_length"]
    if has_cls:
        fieldnames.append("predicted_binary_interaction")
    if has_mcls:
        fieldnames.extend(["predicted_agonist", "predicted_antagonist", "predicted_nonbinder"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored = 0
    t0 = time.perf_counter()

    with open(output_path, "w", newline="") as fh, torch.inference_mode():
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for chunk_start in range(0, num_ligands, LIGAND_CHUNK_SIZE):
            chunk_smiles = valid_smiles[chunk_start:chunk_start + LIGAND_CHUNK_SIZE]
            chunk_canonical = [canonical_map[s] for s in chunk_smiles]
            chunk_ligands = _load_chunk_ligands(chunk_canonical, lmdb_path, memory_ligands)

            pair_iter = (
                (prot_id, prot_seq, can_smi, orig_smi)
                for prot_id, prot_seq in proteins.items()
                for orig_smi, can_smi in zip(chunk_smiles, chunk_canonical)
                if can_smi in chunk_ligands
            )

            for batch_pairs in _batch_iter(pair_iter, batch_size):
                reg_np, cls_np, mcls_np = _score_batch(
                    batch_pairs, chunk_ligands, protein_features,
                    model, use_amp, has_cls, has_mcls, device,
                )

                for i, (prot_id, prot_seq, _csmi, orig_smi) in enumerate(batch_pairs):
                    row: dict[str, Any] = {
                        "protein_id": prot_id,
                        "smiles": orig_smi,
                        "prediction": round(float(reg_np[i]), 4),
                        "protein_length": len(prot_seq),
                    }
                    if has_cls and cls_np is not None:
                        row["predicted_binary_interaction"] = round(float(cls_np[i]), 4)
                    if has_mcls and mcls_np is not None:
                        row["predicted_agonist"] = round(float(mcls_np[i][2]), 4)
                        row["predicted_antagonist"] = round(float(mcls_np[i][0]), 4)
                        row["predicted_nonbinder"] = round(float(mcls_np[i][1]), 4)
                    writer.writerow(row)

                scored += len(batch_pairs)

            elapsed = time.perf_counter() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            log.info("  Scored %d / %d pairs (%.0f pairs/s)", scored, total_pairs, rate)

    total_time = time.perf_counter() - t0
    log.info(
        "Scoring complete: %d pairs in %.1fs (%.0f pairs/s)",
        scored, total_time, scored / total_time if total_time > 0 else 0,
    )
    return scored


# ---------------------------------------------------------------------------
# Interpretation outputs
# ---------------------------------------------------------------------------


def _unbatch_nodes(data_tensor: torch.Tensor, index_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Unbatch a data tensor based on a batch index tensor."""
    return [data_tensor[index_tensor == i] for i in index_tensor.unique()]


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize. Returns zeros if span is zero."""
    span = arr.max() - arr.min()
    if span == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Percentile rank within array. Returns [1.0] for single-element."""
    if len(arr) <= 1:
        return np.ones_like(arr)
    return np.argsort(np.argsort(arr)) / (len(arr) - 1)


def write_interpretation(
    model: torch.nn.Module,
    protein_features: dict[str, dict],
    proteins: dict[str, str],
    canonical_map: dict[str, str],
    lmdb_path: Path | None,
    memory_ligands: dict[str, dict[str, Any]] | None,
    heaps: dict[str, list],
    batch_size: int,
    use_amp: bool,
    interpret_dir: Path,
    device: torch.device,
) -> None:
    """Re-run model on top-K pairs with save_cluster=True and write interpretation CSVs."""
    import pandas as pd

    # Collect all top-K pairs to re-score
    interpret_pairs: list[tuple[str, str, str]] = []  # (prot_id, prot_seq, original_smiles)
    for prot_id, heap in heaps.items():
        prot_seq = proteins[prot_id]
        for _score, _counter, orig_smi, _cls, _mcls in heap:
            interpret_pairs.append((prot_id, prot_seq, orig_smi))

    if not interpret_pairs:
        log.info("No pairs to interpret")
        return

    log.info("Running interpretation pass on %d top-K pairs...", len(interpret_pairs))

    # Collect all needed canonical SMILES
    needed_canonical = set()
    for _pid, _seq, orig_smi in interpret_pairs:
        if orig_smi in canonical_map:
            needed_canonical.add(canonical_map[orig_smi])

    if lmdb_path is not None:
        ligand_features = _lmdb_batch_get(lmdb_path, list(needed_canonical))
    else:
        ligand_features = {}
        for k in needed_canonical:
            if k in memory_ligands:
                ligand_features[k] = memory_ligands[k]
            else:
                log.warning("Canonical SMILES missing from memory cache: %s", k)

    interpret_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch_start in range(0, len(interpret_pairs), batch_size):
            batch_pairs = interpret_pairs[batch_start:batch_start + batch_size]

            data_objects = []
            valid_batch_pairs = []
            for _pid, prot_seq, orig_smi in batch_pairs:
                can_smi = canonical_map.get(orig_smi)
                if can_smi is None:
                    log.warning("Missing canonical mapping for %s — skipping", orig_smi)
                    continue
                mol = ligand_features.get(can_smi)
                if mol is None:
                    log.warning("Missing ligand features for %s — skipping", orig_smi)
                    continue
                prot = protein_features[prot_seq]
                data_objects.append(build_data_object(mol, prot))
                valid_batch_pairs.append((_pid, prot_seq, orig_smi))

            if not data_objects:
                continue

            batch_pairs = valid_batch_pairs

            batch = Batch.from_data_list(
                data_objects,
                follow_batch=["mol_x", "clique_x", "prot_node_aa"],
            ).to(device)

            _, _, _, attention_dict = _run_forward(
                model, batch, use_amp, save_cluster=True,
            )

            # Extract per-pair interpretation
            prot_batch_idx = attention_dict["protein_residue_index"]
            residue_scores_unbatched = _unbatch_nodes(
                attention_dict["residue_final_score"], prot_batch_idx,
            )

            # Cluster assignments (if available)
            cluster_s0 = cluster_s1 = cluster_s2 = None
            has_clusters = all(idx in attention_dict.get("cluster_s", {}) for idx in range(3))
            if has_clusters:
                cluster_s0 = _unbatch_nodes(
                    attention_dict["cluster_s"][0].softmax(dim=-1), prot_batch_idx,
                )
                cluster_s1 = _unbatch_nodes(
                    attention_dict["cluster_s"][1].softmax(dim=-1), prot_batch_idx,
                )
                cluster_s2 = _unbatch_nodes(
                    attention_dict["cluster_s"][2].softmax(dim=-1), prot_batch_idx,
                )

            for i, (prot_id, _prot_seq, orig_smi) in enumerate(batch_pairs):
                safe_pid = _safe_protein_id(prot_id)
                pair_id = f"{safe_pid}__{_smiles_hash(orig_smi)}"
                pair_dir = interpret_dir / safe_pid / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)

                # Residue scores
                scores_np = residue_scores_unbatched[i].cpu().flatten().numpy()
                normed = _minmax_norm(scores_np)
                pctile = _percentile_rank(normed)

                # Get protein sequence for residue types
                prot_seq = _prot_seq
                residue_types = list(prot_seq) if prot_seq else ["?"] * len(scores_np)

                df_data: dict[str, Any] = {
                    "Residue_ID": list(range(1, len(scores_np) + 1)),
                    "Residue_Type": residue_types[:len(scores_np)],
                    "PSICHIC_Residue_Score": normed,
                    "PSICHIC_Residue_Percentile": pctile,
                }

                if has_clusters:
                    s0 = cluster_s0[i].cpu().numpy()
                    for ci in range(s0.shape[1]):
                        df_data[f"Layer0_Cluster{ci}"] = s0[:, ci]

                    s1 = cluster_s1[i].cpu().numpy()
                    for ci in range(s1.shape[1]):
                        df_data[f"Layer1_Cluster{ci}"] = s1[:, ci]

                    s2 = cluster_s2[i].cpu().numpy()
                    for ci in range(s2.shape[1]):
                        df_data[f"Layer2_Cluster{ci}"] = s2[:, ci]

                protein_df = pd.DataFrame(df_data)
                protein_df.to_csv(pair_dir / "protein.csv", index=False)

    log.info("Interpretation CSVs written to %s", interpret_dir)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_topk_csv(
    heaps: dict[str, list],
    proteins: dict[str, str],
    has_cls: bool,
    has_mcls: bool,
    output_path: Path,
) -> int:
    """Write top-K results from heaps to sorted CSV.

    Returns number of rows written.
    """
    fieldnames = ["protein_id", "smiles", "prediction", "protein_length"]
    if has_cls:
        fieldnames.append("predicted_binary_interaction")
    if has_mcls:
        fieldnames.extend(["predicted_agonist", "predicted_antagonist", "predicted_nonbinder"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for prot_id in sorted(proteins.keys()):
            heap = heaps.get(prot_id, [])
            # Sort descending by score
            sorted_entries = sorted(heap, key=lambda e: -e[0])
            prot_seq = proteins[prot_id]

            for score, _counter, orig_smi, cls_val, mcls_val in sorted_entries:
                row: dict[str, Any] = {
                    "protein_id": prot_id,
                    "smiles": orig_smi,
                    "prediction": round(score, 4),
                    "protein_length": len(prot_seq),
                }
                if has_cls and cls_val is not None:
                    row["predicted_binary_interaction"] = round(cls_val, 4)
                if has_mcls and mcls_val is not None:
                    row["predicted_agonist"] = round(mcls_val[2], 4)
                    row["predicted_antagonist"] = round(mcls_val[0], 4)
                    row["predicted_nonbinder"] = round(mcls_val[1], 4)
                writer.writerow(row)
                rows_written += 1

    return rows_written


def sort_csv_by_score(output_path: Path) -> None:
    """Sort an existing CSV by (protein_id asc, prediction desc).

    Uses stdlib csv to avoid loading the entire file into a pandas DataFrame.
    """
    with open(output_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    pid_idx = header.index("protein_id")
    pred_idx = header.index("prediction")
    rows.sort(key=lambda r: (r[pid_idx], -float(r[pred_idx])))

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    log.info("Sorted %d rows in %s", len(rows), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSICHIC optimized screening pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fasta", type=Path, required=True,
        help="Path to FASTA file with protein sequences",
    )
    parser.add_argument(
        "--smiles", type=Path, required=True,
        help="Path to SMILES file (CSV, SMI, or tar.gz)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("screening_results.csv"),
        help="Output CSV path (default: screening_results.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP (default: on)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_PROTEIN_CACHE_DIR,
        help="Protein embedding cache directory",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable protein cache")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Only output top-K predictions per protein (required for >10M pairs)",
    )
    # Ligand cache flags
    parser.add_argument(
        "--ligand-cache-dir", type=Path, default=DEFAULT_LIGAND_CACHE_DIR,
        help="LMDB ligand cache directory (default: cache/)",
    )
    parser.add_argument(
        "--no-ligand-cache", action="store_true",
        help="Skip LMDB, hold ligands in memory (max 50K unique ligands)",
    )
    parser.add_argument(
        "--ligand-workers", type=int, default=0,
        help="Number of ligand featurization workers (0=auto, max 8)",
    )
    # Interpretation flags
    parser.add_argument(
        "--save-interpret", action="store_true",
        help="Save per-residue interpretation CSVs (requires --top-k)",
    )
    parser.add_argument(
        "--interpret-dir", type=Path, default=Path("interpretation_results"),
        help="Directory for interpretation outputs",
    )
    # Model selection
    parser.add_argument(
        "--model", default="PDBv2020_PSICHIC",
        choices=["PDBv2020_PSICHIC", "multitask_PSICHIC"],
        help="Model weights to use (default: PDBv2020_PSICHIC)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Set spawn start method before any ProcessPoolExecutor is created.
    # Must be inside main() to avoid side effects when inference.py is imported as a library.
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")

    args = parse_args()

    # Validate flag combinations
    if args.save_interpret and args.top_k is None:
        log.error("--save-interpret requires --top-k to limit output volume")
        sys.exit(1)

    # Determine model capabilities
    has_cls, has_mcls = _model_has_classification(args.model)

    t_total = time.perf_counter()

    # ---------------------------------------------------------------
    # STEP 1: Parse inputs (CPU — no CUDA yet)
    # ---------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1: Parsing inputs")
    proteins = parse_fasta(args.fasta)
    smiles_list = parse_smiles(args.smiles)

    if not proteins:
        log.error("No proteins found in %s", args.fasta)
        sys.exit(1)
    if not smiles_list:
        log.error("No SMILES found in %s", args.smiles)
        sys.exit(1)

    # Build canonical mapping
    canonical_map, failed_smiles = _build_canonical_map(smiles_list)
    valid_smiles = [s for s in smiles_list if s in canonical_map]
    unique_canonical = set(canonical_map.values())

    total_pairs = len(proteins) * len(valid_smiles)
    log.info(
        "Screening matrix: %d proteins x %d ligands = %d pairs",
        len(proteins), len(valid_smiles), total_pairs,
    )

    # Scale guard
    if total_pairs > MAX_PAIRS_WITHOUT_TOPK and args.top_k is None:
        log.error(
            "Screening %d pairs (>%dM) without --top-k. "
            "Use --top-k N to limit output volume (e.g. --top-k 100).",
            total_pairs, MAX_PAIRS_WITHOUT_TOPK // 1_000_000,
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # STEP 2: Precompute ligand features (CPU, multiprocessing — BEFORE CUDA)
    # ---------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 2: Precomputing ligand features")

    lmdb_path: Path | None = None
    memory_ligands: dict[str, dict[str, Any]] | None = None
    MAX_LIGAND_WORKERS = 8
    num_workers = args.ligand_workers if args.ligand_workers > 0 else max(1, (os.cpu_count() or 1) - 2)
    num_workers = min(num_workers, MAX_LIGAND_WORKERS)

    if args.no_ligand_cache:
        memory_ligands = precompute_ligands_memory(unique_canonical, num_workers)
    else:
        lmdb_path, _hits, _computed = precompute_ligands_lmdb(
            unique_canonical, args.ligand_cache_dir, num_workers,
        )

    # ---------------------------------------------------------------
    # STEP 3: Precompute protein embeddings (CUDA — ESM-2)
    # ---------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 3: Precomputing protein embeddings (batched ESM-2)")
    device = torch.device(args.device)
    protein_cache_dir = None if args.no_cache else args.cache_dir
    protein_features = precompute_proteins(proteins, device, protein_cache_dir)

    missing_prots = [pid for pid, seq in proteins.items() if seq not in protein_features]
    if missing_prots:
        log.error("Failed to process proteins: %s", missing_prots)
        sys.exit(1)

    # ---------------------------------------------------------------
    # STEP 4: Load PSICHIC model
    # ---------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 4: Loading PSICHIC model (%s)", args.model)
    model = load_model(args.model, device)

    # ---------------------------------------------------------------
    # STEP 5: Screening
    # ---------------------------------------------------------------
    log.info("=" * 60)

    if args.top_k is not None:
        # Top-K streaming mode
        log.info("STEP 5: Streaming top-%d scoring", args.top_k)
        heaps = screen_streaming_topk(
            model=model,
            protein_features=protein_features,
            proteins=proteins,
            canonical_map=canonical_map,
            valid_smiles=valid_smiles,
            lmdb_path=lmdb_path,
            memory_ligands=memory_ligands,
            top_k=args.top_k,
            batch_size=args.batch_size,
            use_amp=args.amp,
            has_cls=has_cls,
            has_mcls=has_mcls,
            device=device,
        )

        # Write top-K CSV
        log.info("STEP 6: Writing top-%d results to %s", args.top_k, args.output)
        rows_written = write_topk_csv(heaps, proteins, has_cls, has_mcls, args.output)

        # Optional interpretation pass
        if args.save_interpret:
            log.info("STEP 7: Running interpretation pass")
            write_interpretation(
                model=model,
                protein_features=protein_features,
                proteins=proteins,
                canonical_map=canonical_map,
                lmdb_path=lmdb_path,
                memory_ligands=memory_ligands,
                heaps=heaps,
                batch_size=args.batch_size,
                use_amp=args.amp,
                interpret_dir=args.interpret_dir,
                device=device,
            )
    else:
        # Full scoring mode (≤10M pairs)
        log.info("STEP 5: Scoring all pairs (streaming to CSV)")
        rows_written = screen_all(
            model=model,
            protein_features=protein_features,
            proteins=proteins,
            canonical_map=canonical_map,
            valid_smiles=valid_smiles,
            lmdb_path=lmdb_path,
            memory_ligands=memory_ligands,
            batch_size=args.batch_size,
            use_amp=args.amp,
            has_cls=has_cls,
            has_mcls=has_mcls,
            device=device,
            output_path=args.output,
        )
        # Sort the CSV
        log.info("STEP 6: Sorting results")
        sort_csv_by_score(args.output)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    total_time = time.perf_counter() - t_total
    log.info("=" * 60)
    log.info("DONE: %d results written to %s", rows_written, args.output)
    log.info("Total wall time: %.1fs", total_time)
    if torch.cuda.is_available():
        log.info(
            "Peak GPU memory: %.1f GB",
            torch.cuda.max_memory_allocated(device) / (1024**3),
        )


if __name__ == "__main__":
    main()
