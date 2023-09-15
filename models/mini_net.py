import torch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch.nn import Embedding, Linear
from torch_geometric.utils import degree, to_scipy_sparse_matrix, segregate_self_loops
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
import scipy.sparse as sp
from models.layers import MLP, AtomEncoder, Drug_PNAConv, Protein_PNAConv, DrugProteinConv, PosLinear, GCNCluster, SAGECluster, SGCluster, APPNPCluster, dropout_edge
from copy import deepcopy
## for drug pooling
from models.drug_pool import MotifPool
## for cluster
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch, dropout_adj, degree, subgraph, softmax, add_remaining_self_loops
from models.protein_pool import dense_mincut_pool , dense_dmon_pool, simplify_pool
## for cluster
from torch_geometric.nn.norm import GraphNorm
import torch_geometric


EPS = 1e-15
import math

class net(torch.nn.Module):
    def __init__(self, mol_deg, prot_deg,
                 # MOLECULE
                 mol_in_channels=43,  prot_in_channels=33, prot_evo_channels=1280,
                 hidden_channels=512,
                 pre_layers=2, post_layers=2,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 # interaction
                 total_layer=3,                
                 K = 10,
                 t = 0.1,
                 # training
                 heads=2, 
                 dropout=0.1,
                 drop_atom=0.1,
                 drop_residue=0.1,
                 dropout_attn_score=0.1,
                 dropout_cluster_edge=0.1,
                 gaussian_noise=0.1,
                 # objective
                 regression_head = True,
                 classification_head = False,
                 device='cuda:0'):
        super(net, self).__init__()
        self.total_layer = total_layer

        if (regression_head or classification_head) == False:
            raise Exception('must have one objective')
        
        self.regression_head = regression_head
        self.classification_head = classification_head

        if isinstance(K, int):
            K = [K]*total_layer 

        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20,hidden_channels)
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True) 

        ### MOLECULE and PROTEIN
        self.mol_convs = torch.nn.ModuleList()

        self.mol_gn2 = torch.nn.ModuleList()

        for idx in range(total_layer):
            self.mol_convs.append(Drug_PNAConv(
                mol_deg, hidden_channels,edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=dropout
            ))

            self.mol_gn2.append(GraphNorm(hidden_channels))
        

        if self.regression_head:
            self.reg_out = MLP([hidden_channels, hidden_channels, 1]) 
        if self.classification_head:
            self.cls_out = MLP([hidden_channels, hidden_channels, 1]) 

        self.device = device

    def reset_parameters(self):
        self.atom_feat_encoder.reset_parameters()

        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()

            self.mol_gn2[idx].reset_parameters()

        if self.regression_head:
            self.reg_out.reset_parameters()
        if self.classification_head:
            self.cls_out.reset_parameters()

    def forward(self,
                # Molecule
                mol_x, mol_x_feat, bond_x, atom_edge_index, 
                clique_x, clique_edge_index, atom2clique_index, # drug cliques
                # Protein
                residue_x, residue_evo_x, residue_edge_index, residue_edge_weight, 
                # Mol-Protein Interaction batch
                mol_batch=None, prot_batch=None, clique_batch=None):
        # Init variables        
        reg_pred = None
        cls_pred = None
        # MOLECULE Featurize
        atom_x = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat)

        spectral_loss = torch.tensor(0.).to(self.device)
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device) 
        attention_dict = {}
        # MOLECULE-PROTEIN Layers
        for idx in range(self.total_layer):
            atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)

            ## Graph Normalization
            atom_x = self.mol_gn2[idx](atom_x, mol_batch)

        mol_pool_feat = global_mean_pool(atom_x, mol_batch)
    
        if self.regression_head:
            reg_pred = self.reg_out(mol_pool_feat)
        if self.classification_head:
            cls_pred = self.cls_out(mol_pool_feat)

        return reg_pred, cls_pred, spectral_loss, ortho_loss, cluster_loss, attention_dict
    
    def temperature_clamp(self):
        pass

    
    def connect_mol_prot(self, mol_batch, prot_batch):
        mol_num_nodes = mol_batch.size(0)
        prot_num_nodes = prot_batch.size(0)
        mol_adj = mol_batch.reshape(-1, 1).repeat(1, prot_num_nodes)
        pro_adj = prot_batch.repeat(mol_num_nodes, 1)

        m2p_edge_index = (mol_adj == pro_adj).nonzero(as_tuple=False).t().contiguous()

        return m2p_edge_index
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()  
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias') or pn.endswith('mean_scale'):# or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)
        return optimizer
    

def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)



def unbatch_edge_index(edge_index, batch):
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def compute_connectivity(edge_index, batch):  ## for numerical stability (i.e. we cap inv_con at 100)

    edges_by_batch = unbatch_edge_index(edge_index, batch)

    nodes_counts = torch.unique(batch, return_counts=True)[1]

    connectivity = torch.tensor([nodes_in_largest_graph(e, n) for e, n in zip(edges_by_batch, nodes_counts)])
    isolation = torch.tensor([isolated_nodes(e, n) for e, n in zip(edges_by_batch, nodes_counts)])

    return connectivity, isolation


def nodes_in_largest_graph(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    num_components, component = sp.csgraph.connected_components(adj)

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return subset.sum() / num_nodes


def isolated_nodes(edge_index, num_nodes):
    r"""Find isolate nodes """
    edge_attr = None

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[edge_index.view(-1)] = 0

    return mask.sum() / num_nodes

def dropout_node(edge_index, p, num_nodes, batch, training):
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    
    ## ensure no graph is totally dropped out
    batch_tf = global_add_pool(node_mask.view(-1,1),batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []
    
    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)
            
    node_mask = torch.cat(node_mask_list)
    
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask


