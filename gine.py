"""
GINE baseline for multi-label molecular property prediction (+ masked loss).

GINE = GIN that also uses edge (bond) features, so the 10-d bond features are not thrown away.
Architecture (kept deliberately small): Linear(34->H) -> 3 x [GINEConv -> BatchNorm -> ReLU -> Dropout]
-> mean pooling over atoms -> MLP head -> n_tasks logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class GINEModel(nn.Module):
    def __init__(self, in_dim=34, edge_dim=10, hidden=128, n_layers=3, n_tasks=25, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_tasks))
        self.n_tasks = n_tasks

    def forward(self, data):
        h = self.in_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, data.edge_index, data.edge_attr)
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, data.batch)   # [n_graphs, hidden]
        return self.head(g)                    # [n_graphs, n_tasks] logits


def masked_bce_with_logits(logits, y):
    """BCE over observed labels only (NaN = missing). Returns the mean over observed entries."""
    if logits.shape != y.shape:
        raise ValueError(f"Dimension mismatch: logits {tuple(logits.shape)} vs labels {tuple(y.shape)}")
    mask = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y, nan=0.0), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)
