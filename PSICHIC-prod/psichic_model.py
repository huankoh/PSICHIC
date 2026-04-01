"""PSICHIC model — consolidated single-file implementation.

Functionally identical to PSICHIC/models/{net,layers,pna,scaler,drug_pool,protein_pool}.py.
All class/attribute names preserved for state_dict compatibility with pretrained weights.

Verify idempotency with: python verify_model.py
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver as aggr_resolver,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    degree,
    softmax,
    subgraph,
    to_dense_adj,
    to_dense_batch,
)
from torch_geometric.utils import scatter

EPS = 1e-15


# ---------------------------------------------------------------------------
# Aggregation (from scaler.py)
# ---------------------------------------------------------------------------


class DegreeScalerAggregation(Aggregation):
    """PNA degree-scaler aggregation."""

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(f"Invalid aggregation type: {type(aggr)}")

        self.scaler = [scaler] if isinstance(aggr, str) else scaler
        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel(), device=deg.device)
        self.avg_deg: Dict[str, float] = {
            "lin": float((bin_degrees * deg).sum()) / num_nodes,
            "log": float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            "exp": float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        self.assert_index_present(index)
        out = self.aggr(x, index, ptr, dim_size, dim)
        assert index is not None
        deg = degree(index, num_nodes=dim_size, dtype=out.dtype).clamp_(1)
        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(size)

        outs = []
        for s in self.scaler:
            if s == "identity":
                outs.append(out)
            elif s == "amplification":
                outs.append(out * (torch.log(deg + 1) / self.avg_deg["log"]))
            elif s == "attenuation":
                outs.append(out * (self.avg_deg["log"] / torch.log(deg + 1)))
            elif s == "exponential":
                outs.append(out * (torch.exp(deg) / self.avg_deg["exp"]))
            elif s == "linear":
                outs.append(out * (deg / self.avg_deg["lin"]))
            elif s == "inverse_linear":
                outs.append(out * (self.avg_deg["lin"] / deg))
            else:
                raise ValueError(f"Unknown scaler '{s}'")
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


# ---------------------------------------------------------------------------
# PNAConv (from pna.py)
# ---------------------------------------------------------------------------


class PNAConv(MessagePassing):
    """Principal Neighbourhood Aggregation convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        divide_input: bool = False,
        **kwargs,
    ):
        aggr = DegreeScalerAggregation(aggregators, scalers, deg)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = out_channels // towers

        if self.edge_dim is not None:
            self.edge_encoder = PyGLinear(edge_dim, self.F_in)

        self.pre_nns = nn.ModuleList()
        self.post_nns = nn.ModuleList()
        for _ in range(towers):
            modules = [PyGLinear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [PyGLinear(self.F_in, self.F_in)]
            self.pre_nns.append(nn.Sequential(*modules))

            in_ch = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [PyGLinear(in_ch, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [PyGLinear(self.F_out, self.F_out)]
            self.post_nns.append(nn.Sequential(*modules))

        self.lin = PyGLinear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for n in self.pre_nns:
            reset(n)
        for n in self.post_nns:
            reset(n)
        self.lin.reset_parameters()

    def forward(
        self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)
        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return torch.stack([nn(h[:, i]) for i, nn in enumerate(self.pre_nns)], dim=1)


# ---------------------------------------------------------------------------
# Helper modules (from layers.py)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Multi-layer perceptron with optional input/output layer norm."""

    def __init__(
        self,
        dims: List[int],
        out_norm: bool = False,
        in_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.FC_layers = nn.ModuleList(
            [nn.Linear(dims[i - 1], dims[i], bias=bias) for i in range(1, len(dims))]
        )
        self.hidden_layers = len(dims) - 2
        self.out_norm = out_norm
        self.in_norm = in_norm
        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for layer in self.FC_layers:
            layer.reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        y = self.in_ln(x) if self.in_norm else x
        for i in range(self.hidden_layers):
            y = F.relu(self.FC_layers[i](y))
        y = self.FC_layers[-1](y)
        if self.out_norm:
            y = self.out_ln(y)
        return y


class PosLinear(nn.Module):
    """Linear layer with positive weights (stored as log, applied as exp)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_value: float = 0.2,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = nn.init.uniform_(
            torch.empty((out_features, in_features), **factory_kwargs),
            a=init_value / 2,
            b=init_value,
        )
        weight = torch.abs(weight)
        self.weight = nn.Parameter(weight.log())
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # Upcast to float32 before exp() to prevent fp16 overflow
        w = self.weight.float().exp()
        return F.linear(input, w.to(input.dtype), self.bias)


class GCNCluster(nn.Module):
    """Multi-layer GCN for cluster assignment."""

    def __init__(
        self, dims: List[int], out_norm: bool = False, in_norm: bool = False
    ):
        super().__init__()
        self.Conv_layers = nn.ModuleList(
            [GCNConv(dims[i - 1], dims[i]) for i in range(1, len(dims))]
        )
        self.hidden_layers = len(dims) - 2
        self.out_norm = out_norm
        self.in_norm = in_norm
        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for layer in self.Conv_layers:
            layer.reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        y = self.in_ln(x) if self.in_norm else x
        for i in range(self.hidden_layers):
            y = F.relu(self.Conv_layers[i](y, edge_index))
        y = self.Conv_layers[-1](y, edge_index)
        if self.out_norm:
            y = self.out_ln(y)
        return y


class Drug_PNAConv(nn.Module):
    """PNA convolution for drug atom graphs with bond encoding."""

    def __init__(
        self,
        mol_deg: Tensor,
        hidden_channels: int,
        edge_channels: int,
        pre_layers: int = 2,
        post_layers: int = 2,
        aggregators: List[str] = None,
        scalers: List[str] = None,
        num_towers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if aggregators is None:
            aggregators = ["sum", "mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]

        self.bond_encoder = nn.Embedding(5, hidden_channels)
        self.atom_conv = PNAConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=mol_deg,
            pre_layers=pre_layers,
            post_layers=post_layers,
            towers=num_towers,
            divide_input=True,
        )
        self.atom_norm = nn.LayerNorm(hidden_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.atom_conv.reset_parameters()
        self.atom_norm.reset_parameters()

    def forward(
        self, atom_x: Tensor, bond_x: Tensor, atom_edge_index: Tensor
    ) -> Tensor:
        bond_x = self.bond_encoder(bond_x.squeeze())
        atom_x = atom_x + F.relu(
            self.atom_norm(self.atom_conv(atom_x, atom_edge_index, bond_x))
        )
        atom_x = F.dropout(atom_x, self.dropout, training=self.training)
        return atom_x


class Protein_PNAConv(nn.Module):
    """PNA convolution for protein residue graphs."""

    def __init__(
        self,
        prot_deg: Tensor,
        hidden_channels: int,
        edge_channels: int,
        pre_layers: int = 2,
        post_layers: int = 2,
        aggregators: List[str] = None,
        scalers: List[str] = None,
        num_towers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if aggregators is None:
            aggregators = ["sum", "mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]

        self.conv = PNAConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=prot_deg,
            pre_layers=pre_layers,
            post_layers=post_layers,
            towers=num_towers,
            divide_input=True,
        )
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(
        self, x: Tensor, prot_edge_index: Tensor, prot_edge_attr: Tensor
    ) -> Tensor:
        x = x + F.relu(self.norm(self.conv(x, prot_edge_index, prot_edge_attr)))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class DrugProteinConv(MessagePassing):
    """Cross-modal attention between drug cliques and protein clusters."""

    _alpha: OptTensor

    def __init__(
        self,
        atom_channels: int,
        residue_channels: int,
        heads: int = 1,
        t: float = 0.2,
        dropout_attn_score: float = 0.2,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        assert residue_channels % heads == 0
        assert atom_channels % heads == 0

        self.residue_out_channels = residue_channels // heads
        self.atom_out_channels = atom_channels // heads
        self.heads = heads
        self.edge_dim = edge_dim
        self._alpha = None
        self.t = t
        self.dropout_attn_score = dropout_attn_score

        H_a = heads * self.atom_out_channels
        H_r = heads * self.residue_out_channels

        # Protein → Drug attention
        self.lin_key = nn.Linear(residue_channels, H_a, bias=False)
        self.lin_query = nn.Linear(atom_channels, H_a, bias=False)
        self.lin_value = nn.Linear(residue_channels, H_a, bias=False)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, H_a, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        # Drug → Protein
        self.lin_atom_value = nn.Linear(atom_channels, H_r, bias=False)

        # Normalization
        self.drug_in_norm = nn.LayerNorm(atom_channels)
        self.residue_in_norm = nn.LayerNorm(residue_channels)
        self.drug_out_norm = nn.LayerNorm(H_a)
        self.residue_out_norm = nn.LayerNorm(H_r)

        # MLP updates
        self.clique_mlp = MLP(
            [atom_channels * 2, atom_channels * 2, atom_channels], out_norm=True
        )
        self.residue_mlp = MLP(
            [residue_channels * 2, residue_channels * 2, residue_channels], out_norm=True
        )

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_atom_value.reset_parameters()
        self.drug_in_norm.reset_parameters()
        self.residue_in_norm.reset_parameters()
        self.drug_out_norm.reset_parameters()
        self.residue_out_norm.reset_parameters()
        self.clique_mlp.reset_parameters()
        self.residue_mlp.reset_parameters()

    def forward(
        self,
        drug_x: Tensor,
        clique_x: Tensor,
        clique_batch: Tensor,
        residue_x: Tensor,
        edge_index: Adj,
    ):
        H, aC = self.heads, self.atom_out_channels

        # Protein residue → Drug atom
        residue_hx = self.residue_in_norm(residue_x)
        query = self.lin_query(drug_x).view(-1, H, aC)
        key = self.lin_key(residue_hx).view(-1, H, aC)
        value = self.lin_value(residue_hx).view(-1, H, aC)

        drug_out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=None, size=None
        )
        alpha = self._alpha
        self._alpha = None

        drug_out = drug_out.view(-1, H * aC)
        drug_out = self.drug_out_norm(drug_out)
        clique_out = torch.cat([clique_x, drug_out[clique_batch]], dim=-1)
        clique_out = self.clique_mlp(clique_out)

        # Drug atom → Protein residue
        H, rC = self.heads, self.residue_out_channels
        drug_hx = self.drug_in_norm(drug_x)
        residue_value = self.lin_atom_value(drug_hx).view(-1, H, rC)[edge_index[1]]
        residue_out = residue_value * alpha.view(-1, H, 1)
        residue_out = residue_out.view(-1, H * rC)
        residue_out = self.residue_out_norm(residue_out)
        residue_out = torch.cat([residue_out, residue_x], dim=-1)
        residue_out = self.residue_mlp(residue_out)

        return clique_out, residue_out, (edge_index, alpha)

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.atom_out_channels)
        alpha = alpha / self.t
        alpha = F.dropout(alpha, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        return value_j * alpha.view(-1, self.heads, 1)


# ---------------------------------------------------------------------------
# Drug pooling (from drug_pool.py)
# ---------------------------------------------------------------------------


class MotifPool(nn.Module):
    """Multi-head motif-based drug pooling with attention scoring."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout_attn_score: float = 0,
        dropout_node_proba: float = 0,
    ):
        super().__init__()
        assert hidden_dim % heads == 0

        self.lin_proj = nn.Linear(hidden_dim, hidden_dim)
        head_dim = hidden_dim // heads

        self.score_proj = nn.ModuleList()
        for _ in range(heads):
            self.score_proj.append(MLP([head_dim, head_dim * 2, 1]))

        self.heads = heads
        self.hidden_dim = head_dim
        self.dropout_node_proba = dropout_node_proba
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        for m in self.score_proj:
            m.reset_parameters()

    def forward(
        self,
        x: Tensor,
        x_clique: Tensor,
        atom2clique_index: Tensor,
        clique_batch: Tensor,
        clique_edge_index: Tensor,
    ):
        row, col = atom2clique_index
        H, C = self.heads, self.hidden_dim

        hx_clique = scatter(
            x[row], col, dim=0, dim_size=x_clique.size(0), reduce="mean"
        )
        x_clique = x_clique + F.relu(self.lin_proj(hx_clique))

        score_clique = x_clique.view(-1, H, C)
        score = torch.cat(
            [mlp(score_clique[:, i]) for i, mlp in enumerate(self.score_proj)], dim=-1
        )
        score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(score, clique_batch)

        _, _, clique_drop_mask = _dropout_node(
            clique_edge_index,
            self.dropout_node_proba,
            x_clique.size(0),
            clique_batch,
            self.training,
        )
        scaling_factor = 1.0 / (1.0 - self.dropout_node_proba)

        drug_feat = x_clique.view(-1, H, C) * alpha.view(-1, H, 1)
        drug_feat = drug_feat.view(-1, H * C) * clique_drop_mask.view(-1, 1)
        drug_feat = global_add_pool(drug_feat, clique_batch) * scaling_factor

        return drug_feat, x_clique, alpha


# ---------------------------------------------------------------------------
# Protein pooling (from protein_pool.py)
# ---------------------------------------------------------------------------


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum("ijj->i", x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    return eye * x.unsqueeze(2).expand(*x.size(), x.size(1))


def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    cluster_drop_node: Optional[Tensor] = None,
):
    """MinCut pooling: soft cluster assignment with spectral regularization."""
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)
    s = torch.softmax(s, dim=-1)

    if mask is not None:
        s = s * mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x_mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        if cluster_drop_node is not None:
            x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)
        x = x * x_mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum("ijk->ij", adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = torch.mean(-(mincut_num / mincut_den))

    # Orthogonality regularization
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.mean(
        torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
            dim=(-1, -2),
        )
    )

    # Normalize coarsened adjacency
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d.float())[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return s, out, out_adj, mincut_loss, ortho_loss


# ---------------------------------------------------------------------------
# Utility functions (from layers.py / net.py)
# ---------------------------------------------------------------------------


def _unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> Tuple:
    """Split tensor by batch vector."""
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def _dropout_edge(
    edge_index: Tensor,
    p: float = 0.5,
    force_undirected: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Randomly drop edges with probability p."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
    if not training or p == 0.0:
        return edge_index, edge_index.new_ones(edge_index.size(1), dtype=torch.bool)

    row, col = edge_index
    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p
    if force_undirected:
        edge_mask[row > col] = False
    edge_index = edge_index[:, edge_mask]
    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()
    return edge_index, edge_mask


def _dropout_node(
    edge_index: Tensor,
    p: float,
    num_nodes: int,
    batch: Tensor,
    training: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Randomly drop nodes, ensuring no graph is fully dropped."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    batch_tf = global_add_pool(node_mask.view(-1, 1), batch).flatten()
    unbatched = _unbatch(node_mask, batch)
    parts = []
    for has_nodes, sub_mask in zip(batch_tf, unbatched):
        if has_nodes.item():
            parts.append(sub_mask)
        else:
            idx = torch.randperm(sub_mask.size(0))[:1]
            sub_mask[idx] = True
            parts.append(sub_mask)
    node_mask = torch.cat(parts)

    edge_index, _, edge_mask = subgraph(
        node_mask, edge_index, num_nodes=num_nodes, return_edge_mask=True
    )
    return edge_index, edge_mask, node_mask


def _rbf(
    D: Tensor,
    D_min: float = 0.0,
    D_max: float = 1.0,
    D_count: int = 16,
) -> Tensor:
    """Radial basis function embedding of distances."""
    D = torch.clamp(D, max=D_max)
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


# ---------------------------------------------------------------------------
# PSICHIC network (from net.py)
# ---------------------------------------------------------------------------


class PSICHIC(nn.Module):
    """PSICHIC drug-protein interaction model.

    Class renamed from `net` for clarity, but all submodule attribute names
    are preserved for state_dict compatibility.
    """

    def __init__(
        self,
        mol_deg: Tensor,
        prot_deg: Tensor,
        mol_in_channels: int = 43,
        prot_in_channels: int = 33,
        prot_evo_channels: int = 1280,
        hidden_channels: int = 200,
        pre_layers: int = 2,
        post_layers: int = 1,
        aggregators: List[str] = None,
        scalers: List[str] = None,
        total_layer: int = 3,
        K: Union[int, List[int]] = None,
        t: float = 1,
        heads: int = 5,
        dropout: float = 0,
        dropout_attn_score: float = 0.2,
        drop_atom: float = 0,
        drop_residue: float = 0,
        dropout_cluster_edge: float = 0,
        gaussian_noise: float = 0,
        regression_head: bool = True,
        classification_head: bool = False,
        multiclassification_head: int = 0,
        device: str = "cuda:0",
    ):
        super().__init__()

        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "linear"]
        if K is None:
            K = [5, 10, 20]

        if not (regression_head or classification_head):
            raise ValueError("Must have at least one objective head")

        self.total_layer = total_layer
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head

        if isinstance(K, int):
            K = [K] * total_layer
        self.num_cluster = K

        self.t = t
        self.dropout = dropout
        self.drop_atom = drop_atom
        self.drop_residue = drop_residue
        self.dropout_cluster_edge = dropout_cluster_edge
        self.gaussian_noise = gaussian_noise
        self.prot_edge_dim = hidden_channels
        # NOTE: self.device removed — derive from input tensors in forward()
        # Kept as init arg for API compat but not stored.

        # Input encoders
        self.atom_type_encoder = nn.Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP(
            [mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )
        self.clique_encoder = nn.Embedding(4, hidden_channels)
        self.prot_evo = MLP(
            [prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )
        self.prot_aa = MLP(
            [prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )

        # Per-layer modules
        self.mol_convs = nn.ModuleList()
        self.prot_convs = nn.ModuleList()
        self.mol_gn2 = nn.ModuleList()
        self.prot_gn2 = nn.ModuleList()
        self.inter_convs = nn.ModuleList()
        self.cluster = nn.ModuleList()
        self.mol_pools = nn.ModuleList()
        self.mol_norms = nn.ModuleList()
        self.prot_norms = nn.ModuleList()
        self.atom_lins = nn.ModuleList()
        self.residue_lins = nn.ModuleList()
        self.c2a_mlps = nn.ModuleList()
        self.c2r_mlps = nn.ModuleList()

        for idx in range(total_layer):
            self.mol_convs.append(
                Drug_PNAConv(
                    mol_deg,
                    hidden_channels,
                    edge_channels=hidden_channels,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    aggregators=aggregators,
                    scalers=scalers,
                    num_towers=heads,
                    dropout=dropout,
                )
            )
            self.prot_convs.append(
                Protein_PNAConv(
                    prot_deg,
                    hidden_channels,
                    edge_channels=hidden_channels,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    aggregators=aggregators,
                    scalers=scalers,
                    num_towers=heads,
                    dropout=dropout,
                )
            )
            self.cluster.append(
                GCNCluster(
                    [hidden_channels, hidden_channels * 2, K[idx]], in_norm=True
                )
            )
            self.inter_convs.append(
                DrugProteinConv(
                    atom_channels=hidden_channels,
                    residue_channels=hidden_channels,
                    heads=heads,
                    t=t,
                    dropout_attn_score=dropout_attn_score,
                )
            )
            self.mol_pools.append(
                MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom)
            )
            self.mol_norms.append(nn.LayerNorm(hidden_channels))
            self.prot_norms.append(nn.LayerNorm(hidden_channels))
            self.atom_lins.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
            self.residue_lins.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
            self.c2a_mlps.append(
                MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False)
            )
            self.c2r_mlps.append(
                MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False)
            )
            self.mol_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))

        # Attention scoring
        self.atom_attn_lin = PosLinear(
            heads * total_layer, 1, bias=False, init_value=1 / heads
        )
        self.residue_attn_lin = PosLinear(
            heads * total_layer, 1, bias=False, init_value=1 / heads
        )

        # Output heads
        self.mol_out = MLP(
            [hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )
        self.prot_out = MLP(
            [hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True
        )
        if self.regression_head:
            self.reg_out = MLP([hidden_channels * 2, hidden_channels, 1])
        if self.classification_head:
            self.cls_out = MLP([hidden_channels * 2, hidden_channels, 1])
        if self.multiclassification_head:
            self.mcls_out = MLP(
                [hidden_channels * 2, hidden_channels, multiclassification_head]
            )

    def reset_parameters(self):
        self.atom_feat_encoder.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()
        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()
            self.prot_convs[idx].reset_parameters()
            self.mol_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()
            self.cluster[idx].reset_parameters()
            self.mol_pools[idx].reset_parameters()
            self.mol_norms[idx].reset_parameters()
            self.prot_norms[idx].reset_parameters()
            self.inter_convs[idx].reset_parameters()
            self.atom_lins[idx].reset_parameters()
            self.residue_lins[idx].reset_parameters()
            self.c2a_mlps[idx].reset_parameters()
            self.c2r_mlps[idx].reset_parameters()
        self.atom_attn_lin.reset_parameters()
        self.residue_attn_lin.reset_parameters()
        self.mol_out.reset_parameters()
        self.prot_out.reset_parameters()
        if self.regression_head:
            self.reg_out.reset_parameters()
        if self.classification_head:
            self.cls_out.reset_parameters()
        if self.multiclassification_head:
            self.mcls_out.reset_parameters()

    def forward(
        self,
        # Molecule
        mol_x: Tensor,
        mol_x_feat: Tensor,
        bond_x: Tensor,
        atom_edge_index: Tensor,
        clique_x: Tensor,
        clique_edge_index: Tensor,
        atom2clique_index: Tensor,
        # Protein
        residue_x: Tensor,
        residue_evo_x: Tensor,
        residue_edge_index: Tensor,
        residue_edge_weight: Tensor,
        # Batch indices
        mol_batch: Optional[Tensor] = None,
        prot_batch: Optional[Tensor] = None,
        clique_batch: Optional[Tensor] = None,
        # Debug
        save_cluster: bool = False,
    ):
        reg_pred = None
        cls_pred = None
        mcls_pred = None
        device = mol_x.device

        residue_edge_attr = _rbf(
            residue_edge_weight,
            D_max=1.0,
            D_count=self.prot_edge_dim,
        )

        # Encode inputs
        residue_x = self.prot_aa(residue_x) + self.prot_evo(residue_evo_x)
        atom_x = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(
            mol_x_feat
        )
        clique_x = self.clique_encoder(clique_x.squeeze())

        spectral_loss = torch.zeros((), device=device, dtype=torch.float32)
        ortho_loss = torch.zeros((), device=device, dtype=torch.float32)
        cluster_loss = torch.zeros((), device=device, dtype=torch.float32)
        clique_scores = []
        residue_scores = []
        layer_s = {}

        for idx in range(self.total_layer):
            atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)
            residue_x = self.prot_convs[idx](
                residue_x, residue_edge_index, residue_edge_attr
            )

            # Pool drug via motif attention
            drug_x, clique_x, clique_score = self.mol_pools[idx](
                atom_x, clique_x, atom2clique_index, clique_batch, clique_edge_index
            )
            drug_x = self.mol_norms[idx](drug_x)
            clique_scores.append(clique_score)

            # Cluster protein residues
            dropped_edge_index, _ = _dropout_edge(
                residue_edge_index,
                p=self.dropout_cluster_edge,
                force_undirected=True,
                training=self.training,
            )
            s = self.cluster[idx](residue_x, dropped_edge_index)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)

            if save_cluster:
                layer_s[idx] = s

            s, _ = to_dense_batch(s, prot_batch)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            cluster_mask = residue_mask

            cluster_drop_mask = None
            if self.drop_residue != 0 and self.training:
                _, _, residue_drop_mask = _dropout_node(
                    residue_edge_index,
                    self.drop_residue,
                    residue_x.size(0),
                    prot_batch,
                    self.training,
                )
                residue_drop_mask, _ = to_dense_batch(
                    residue_drop_mask.reshape(-1, 1), prot_batch
                )
                residue_drop_mask = residue_drop_mask.squeeze()
                cluster_drop_mask = residue_mask * residue_drop_mask.squeeze()

            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(
                residue_hx, residue_adj, s, cluster_mask, cluster_drop_mask
            )
            ortho_loss += o_loss
            cluster_loss += cl_loss
            cluster_x = self.prot_norms[idx](cluster_x)

            # Connect drug and protein cluster
            batch_size = s.size(0)
            num_k = self.num_cluster[idx]
            cluster_residue_batch = torch.arange(
                batch_size, device=device
            ).repeat_interleave(num_k)
            cluster_x = cluster_x.reshape(batch_size * num_k, -1)
            p2m_edge_index = torch.stack([
                torch.arange(batch_size * num_k, device=device),
                cluster_residue_batch,
            ])

            # Cross-modal interaction
            clique_x, cluster_x, inter_attn = self.inter_convs[idx](
                drug_x, clique_x, clique_batch, cluster_x, p2m_edge_index
            )
            inter_attn = inter_attn[1]

            # Residual: clique → atom
            row, col = atom2clique_index
            atom_x = atom_x + F.relu(
                self.atom_lins[idx](
                    scatter(
                        clique_x[col],
                        row,
                        dim=0,
                        dim_size=atom_x.size(0),
                        reduce="mean",
                    )
                )
            )
            atom_x = atom_x + self.c2a_mlps[idx](atom_x)
            atom_x = F.dropout(atom_x, self.dropout, training=self.training)

            # Residual: cluster → residue
            residue_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)
            inter_attn, _ = to_dense_batch(inter_attn, cluster_residue_batch)
            residue_x = residue_x + F.relu(
                self.residue_lins[idx]((s @ residue_hx)[residue_mask])
            )
            residue_x = residue_x + self.c2r_mlps[idx](residue_x)
            residue_x = F.dropout(residue_x, self.dropout, training=self.training)
            inter_attn = (s @ inter_attn)[residue_mask]
            residue_scores.append(inter_attn)

            # Graph normalization
            atom_x = self.mol_gn2[idx](atom_x, mol_batch)
            residue_x = self.prot_gn2[idx](residue_x, prot_batch)

        # Final pooling with learned attention
        row, col = atom2clique_index
        clique_scores = torch.cat(clique_scores, dim=-1)
        atom_scores = scatter(
            clique_scores[col], row, dim=0, dim_size=atom_x.size(0), reduce="mean"
        )
        atom_score = self.atom_attn_lin(atom_scores)
        atom_score = softmax(atom_score, mol_batch)
        mol_pool_feat = global_add_pool(atom_x * atom_score, mol_batch)

        residue_scores = torch.cat(residue_scores, dim=-1)
        residue_score = softmax(
            self.residue_attn_lin(residue_scores), prot_batch
        )
        prot_pool_feat = global_add_pool(residue_x * residue_score, prot_batch)

        mol_pool_feat = self.mol_out(mol_pool_feat)
        prot_pool_feat = self.prot_out(prot_pool_feat)
        mol_prot_feat = torch.cat([mol_pool_feat, prot_pool_feat], dim=-1)

        if self.regression_head:
            reg_pred = self.reg_out(mol_prot_feat)
        if self.classification_head:
            cls_pred = self.cls_out(mol_prot_feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(mol_prot_feat)

        attention_dict = {
            "residue_final_score": residue_score,
            "atom_final_score": atom_score,
            "clique_layer_scores": clique_scores,
            "residue_layer_scores": residue_scores,
            "drug_atom_index": mol_batch,
            "drug_clique_index": clique_batch,
            "protein_residue_index": prot_batch,
            "mol_feature": mol_pool_feat,
            "prot_feature": prot_pool_feat,
            "interaction_fingerprint": mol_prot_feat,
            "cluster_s": layer_s,
        }

        return (
            reg_pred,
            cls_pred,
            mcls_pred,
            spectral_loss,
            ortho_loss,
            cluster_loss,
            attention_dict,
        )
