import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax

from torch_scatter import scatter
from torch_geometric.nn import global_add_pool
from models.layers import MLP, dropout_node

class MotifPool(torch.nn.Module):
    def __init__(self, hidden_dim, heads, dropout_attn_score=0, dropout_node_proba=0): 
        super().__init__()
        assert hidden_dim % heads == 0 

        self.lin_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        hidden_dim = hidden_dim // heads
        
        self.score_proj = torch.nn.ModuleList()
        for _ in range(heads): 
            self.score_proj.append( MLP([ hidden_dim, hidden_dim*2, 1]) )
        
        self.heads = heads
        self.hidden_dim = hidden_dim 
        self.dropout_node_proba = dropout_node_proba
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        for m in self.score_proj:
            m.reset_parameters()

    def forward(self, x, x_clique, atom2clique_index, clique_batch, clique_edge_index):
        row, col = atom2clique_index
        H = self.heads
        C = self.hidden_dim
        ## residual connection + atom2clique
        hx_clique = scatter(x[row], col, dim=0, dim_size=x_clique.size(0), reduce='mean')
        x_clique = x_clique + F.relu(self.lin_proj(hx_clique))
        ## GNN scoring
        score_clique = x_clique.view(-1, H, C)
        score = torch.cat([ mlp(score_clique[:, i]) for i, mlp in enumerate(self.score_proj) ], dim=-1)
        score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(score, clique_batch)    

        ## multihead aggregation of drug feature
        scaling_factor = 1. 
        _, _, clique_drop_mask = dropout_node(clique_edge_index, self.dropout_node_proba, x_clique.size(0), clique_batch, self.training)
        scaling_factor = 1. / (1. - self.dropout_node_proba)

        drug_feat = x_clique.view(-1, H, C) * alpha.view(-1, H, 1) 
        drug_feat = drug_feat.view(-1, H * C) * clique_drop_mask.view(-1,1)
        drug_feat = global_add_pool(drug_feat, clique_batch) * scaling_factor

        return drug_feat, x_clique, alpha