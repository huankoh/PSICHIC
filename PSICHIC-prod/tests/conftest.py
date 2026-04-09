"""Pre-stub heavy dependencies so tests can import inference.py utility functions.

inference.py imports utils.dataset → utils.__init__ → utils.metrics/trainer/etc.,
pulling in torch_geometric, lifelines, sklearn, scipy, reprint, esm.
We only test pure Python utilities (parse_fasta, _smiles_hash, etc.) that
don't need any of those, so we stub them out before collection.
"""

from __future__ import annotations

import sys
from unittest import mock

_STUBS = [
    # torch_geometric
    "torch_geometric",
    "torch_geometric.data",
    "torch_geometric.loader",
    "torch_geometric.nn",
    "torch_geometric.nn.aggr",
    "torch_geometric.nn.conv",
    "torch_geometric.nn.dense",
    "torch_geometric.nn.dense.linear",
    "torch_geometric.nn.inits",
    "torch_geometric.nn.norm",
    "torch_geometric.nn.resolver",
    "torch_geometric.typing",
    "torch_geometric.utils",
    # sklearn
    "sklearn",
    "sklearn.metrics",
    "sklearn.linear_model",
    # scipy
    "scipy",
    "scipy.stats",
    "scipy.sparse",
    "scipy.sparse.csgraph",
    # lifelines
    "lifelines",
    "lifelines.utils",
    # reprint
    "reprint",
    # esm (used by protein_init)
    "esm",
    "esm.pretrained",
    # BioPython (used by protein_init)
    "Bio",
    "Bio.PDB",
]

for mod_name in _STUBS:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock.MagicMock()
