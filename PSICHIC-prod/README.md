# PSICHIC-prod

Optimized production inference pipeline for PSICHIC drug-protein interaction screening on NVIDIA GPUs.

This directory is **additive** -- the original `screening.py` and `models/` are untouched. Use `PSICHIC-prod/` when you need fast, large-scale screening.

## Quick start

```bash
cd PSICHIC-prod

# Basic screening
python inference.py --fasta proteins.fasta --smiles ligands.csv

# Large-scale with top-K (ZINC-scale)
python inference.py --fasta proteins.fasta --smiles zinc.csv \
    --top-k 100 --batch-size 128

# Multitask model (adds agonist/antagonist/nonbinder columns)
python inference.py --fasta proteins.fasta --smiles ligands.csv \
    --model multitask_PSICHIC

# With interpretation outputs
python inference.py --fasta proteins.fasta --smiles ligands.csv \
    --top-k 50 --save-interpret
```

## What's different from `screening.py`

| Feature | `screening.py` | `PSICHIC-prod/inference.py` |
|---------|----------------|----------------------------|
| Model | `models/net.py` (original) | `psichic_model.py` (compile-friendly, no `torch_scatter`) |
| ESM-2 | Serial, one sequence at a time | Batched by length bin (`esm_batched/`) |
| Ligand featurization | In dataloader, per-item | Multiprocessing + LMDB cache |
| Scoring | Loads all pairs into memory | Streaming chunks (10K ligands/chunk) |
| AMP | No | Yes (Tensor Core acceleration) |
| Top-K | No | Per-protein heapq, memory-bounded |
| Interpretation | `--interpret` flag | `--save-interpret --top-k N` (two-pass) |
| Scale | ~1K ligands | Millions (ZINC-scale) |

## Performance (A100 80GB)

| Metric | Original | PSICHIC-prod |
|--------|----------|--------------|
| Model scoring | 55 pairs/s | **223 pairs/s** (4x, AMP + bs=128) |
| Protein preprocessing | ~21 min (12.6K proteins) | **18s** with cache (68x) |
| Ligand featurization | ~335/s serial | **~2,500/s** (8 workers) or **~100K/s** (cache hit) |

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--fasta` | required | Protein FASTA file |
| `--smiles` | required | SMILES file (CSV, SMI, or tar.gz) |
| `--output` | `results.csv` | Output CSV path |
| `--model` | `PDBv2020_PSICHIC` | Model weights (`PDBv2020_PSICHIC` or `multitask_PSICHIC`) |
| `--batch-size` | `128` | Inference batch size |
| `--device` | `cuda:0` | CUDA device |
| `--top-k` | None | Keep top-K ligands per protein |
| `--save-interpret` | off | Save per-residue interpretation CSVs (requires `--top-k`) |
| `--interpret-dir` | `interpretation_results/` | Interpretation output directory |
| `--no-cache` | off | Skip protein cache |
| `--ligand-cache-dir` | `cache/` | LMDB ligand cache directory |
| `--no-ligand-cache` | off | In-memory mode (max 50K ligands) |
| `--ligand-workers` | auto | Number of ligand preprocessing workers (0=auto, max 8) |
| `--amp` | on | Mixed precision inference |

## File structure

```
PSICHIC-prod/
  inference.py          # Production entry point
  psichic_model.py      # Self-contained model (no torch_scatter dependency)
  esm_batched/          # Batched ESM-2 protein preprocessing
  cache/                # Created at runtime
    proteins/           # Per-protein .pt shards
    ligands_v1.lmdb/    # LMDB ligand feature cache
```

Shared resources from parent repo (not duplicated):
- `../trained_weights/` -- model checkpoints
- `../utils/` -- ligand/protein featurization, dataset utilities

## Requirements

Same as PSICHIC, plus:
- `lmdb` (`pip install lmdb`)
- `torch_scatter` is **not** required (replaced with `torch_geometric.utils.scatter`)
