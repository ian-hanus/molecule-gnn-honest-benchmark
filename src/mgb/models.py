"""Message-passing GNNs in plain PyTorch (no PyG dependency).

Three architectures share a readout:
  * gcn   - Kipf & Welling graph convolution (ignores bond features)
  * gin   - GIN with bond features added to messages (GINE)
  * dmpnn - directed-edge message passing, as in Chemprop
"""

from __future__ import annotations

import torch
from torch import nn

from .featurize import ATOM_DIM, BOND_DIM, GraphBatch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros((n, src.shape[1]))
    return out.index_add_(0, index, src)


def mean_pool(x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    total = scatter_sum(x, batch, n_graphs)
    count = scatter_sum(torch.ones_like(x[:, :1]), batch, n_graphs).clamp(min=1)
    return total / count


class GCNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, h, edge_index, edge_attr):
        n = h.shape[0]
        src, dst = edge_index
        deg = scatter_sum(torch.ones(src.shape[0], 1, device=h.device), dst, n) + 1  # + self loop
        norm = deg.rsqrt()
        h = self.lin(h)
        msg = h[src] * norm[src] * norm[dst]
        return scatter_sum(msg, dst, n) + h * norm * norm


class GINLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.edge_lin = nn.Linear(BOND_DIM, dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        msg = torch.relu(h[src] + self.edge_lin(edge_attr))
        agg = scatter_sum(msg, dst, h.shape[0])
        return self.mlp((1 + self.eps) * h + agg)


class NodeGNN(nn.Module):
    """Stack of node-centred layers (gcn / gin) with residuals, norm and dropout."""

    def __init__(self, layer_cls, dim: int, depth: int, dropout: float):
        super().__init__()
        self.embed = nn.Linear(ATOM_DIM, dim)
        self.layers = nn.ModuleList(layer_cls(dim) for _ in range(depth))
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: GraphBatch) -> torch.Tensor:
        h = self.embed(g.x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + self.dropout(torch.relu(norm(layer(h, g.edge_index, g.edge_attr))))
        return h


class DMPNNEncoder(nn.Module):
    """Directed-edge message passing.

    Edges come in reverse pairs (2k, 2k+1), so the reverse of edge ``e`` is
    ``e ^ 1``. A message on u->v aggregates every edge into u *except* v->u,
    which stops information from immediately bouncing back along a bond.
    """

    def __init__(self, dim: int, depth: int, dropout: float):
        super().__init__()
        self.w_in = nn.Linear(ATOM_DIM + BOND_DIM, dim)
        self.w_h = nn.Linear(dim, dim)
        self.w_out = nn.Linear(ATOM_DIM + dim, dim)
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: GraphBatch) -> torch.Tensor:
        src, dst = g.edge_index
        n = g.x.shape[0]
        if src.numel() == 0:
            return torch.relu(self.w_out(torch.cat([g.x, g.x.new_zeros(n, self.w_h.in_features)], 1)))
        rev = torch.arange(src.shape[0], device=src.device) ^ 1
        h0 = torch.relu(self.w_in(torch.cat([g.x[src], g.edge_attr], dim=1)))
        h = h0
        for _ in range(self.depth - 1):
            node_in = scatter_sum(h, dst, n)
            m = node_in[src] - h[rev]
            h = self.dropout(torch.relu(h0 + self.w_h(m)))
        node_in = scatter_sum(h, dst, n)
        return torch.relu(self.w_out(torch.cat([g.x, node_in], dim=1)))


class MoleculeGNN(nn.Module):
    def __init__(self, arch: str, n_tasks: int, dim: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        if arch == "gcn":
            self.encoder = NodeGNN(GCNLayer, dim, depth, dropout)
        elif arch == "gin":
            self.encoder = NodeGNN(GINLayer, dim, depth, dropout)
        elif arch == "dmpnn":
            self.encoder = DMPNNEncoder(dim, depth, dropout)
        else:
            raise ValueError(f"unknown arch: {arch}")
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, n_tasks),
        )

    def forward(self, g: GraphBatch) -> torch.Tensor:
        h = self.encoder(g)
        return self.head(mean_pool(h, g.batch, g.n_graphs))
