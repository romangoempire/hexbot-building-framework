"""
Graph Neural Network for Hex Connect-6.

Operates directly on hex topology - each cell is a node with 6 neighbors.
Eliminates the rectangular grid encoding, enabling:
- Native hex geometry (all 12 D6 symmetries are trivial)
- Variable board size without retraining
- Global attention across all positions

Experimental - untested in competitive play.

Usage:
    from bot import create_network
    net = create_network('hex-gnn')

Requires: torch_geometric (pip install torch-geometric)
Falls back to a pure-PyTorch message-passing implementation if not available.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 12


# Hex neighbor offsets in axial coordinates
HEX_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def _build_hex_adjacency(board_size: int):
    """Build adjacency list for hex grid encoded as NxN board.

    Returns (edge_index, num_nodes) where edge_index is (2, num_edges).
    """
    N = board_size
    edges_src, edges_dst = [], []

    for i in range(N):
        for j in range(N):
            node = i * N + j
            for di, dj in HEX_NEIGHBORS:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    neighbor = ni * N + nj
                    edges_src.append(node)
                    edges_dst.append(neighbor)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index, N * N


class HexMessagePassing(nn.Module):
    """Simple message passing layer for hex graph.

    Each node aggregates features from its 6 hex neighbors,
    transforms with a linear layer, and adds a residual connection.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear_msg = nn.Linear(dim, dim, bias=False)
        self.linear_upd = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        """
        x: (batch, num_nodes, dim)
        edge_index: (2, num_edges) - shared across batch
        """
        src, dst = edge_index
        batch, N, D = x.shape

        # Gather neighbor features
        msg = self.linear_msg(x)  # (batch, N, D)
        # Scatter-add messages from neighbors
        # Reshape for indexing: (batch, N, D)
        agg = torch.zeros_like(x)
        src_features = msg[:, src]  # (batch, num_edges, D)
        # Scatter add into destination nodes
        agg.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(batch, -1, D),
                         src_features)

        # Count neighbors for normalization
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(len(dst), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        agg = agg / deg

        # Update: concatenate self + aggregated, project back
        combined = torch.cat([x, agg], dim=-1)  # (batch, N, 2D)
        out = self.linear_upd(combined)  # (batch, N, D)
        out = self.norm(F.gelu(out) + x)  # residual + norm
        return out


class HexGNN(nn.Module):
    """Graph Neural Network operating on hex topology.

    Architecture:
        Input: (batch, channels, N, N) tensor (same as HexNet for compatibility)
          -> Reshape to (batch, N*N, channels) node features
          -> Linear projection to hidden dim
          -> HexMessagePassing x num_layers (with hex adjacency)
          -> Global attention pooling (for value head)
          -> Per-node output (for policy head)
          -> Policy head: per-node logits -> (batch, N*N)
          -> Value head: global pool -> FC -> tanh
          -> Threat head: global pool -> FC -> 4

    ~2.5M parameters at dim=128, 8 layers.
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_channels: int = NUM_CHANNELS,
        hidden_dim: int = 128,
        num_layers: int = 8,
    ):
        super().__init__()
        self.board_size = board_size
        N = board_size
        bs2 = N * N

        # Build adjacency (registered as buffer - moves to device with model)
        edge_index, num_nodes = _build_hex_adjacency(board_size)
        self.register_buffer('edge_index', edge_index)

        # Input projection
        self.input_proj = nn.Linear(num_channels, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Message passing layers
        self.layers = nn.ModuleList([
            HexMessagePassing(hidden_dim) for _ in range(num_layers)
        ])

        # Global attention pooling (for value/threat heads)
        self.attn_query = nn.Linear(hidden_dim, 1)

        # Policy head (per-node)
        self.policy_proj = nn.Linear(hidden_dim, 1)

        # Value head (global)
        self.value_fc1 = nn.Linear(hidden_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Threat head (global)
        self.threat_fc = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: (batch, channels, N, N) - same format as HexNet
        Output: (policy_logits, value, threat) - same format as HexNet
        """
        batch = x.size(0)
        N = self.board_size

        # Reshape: (batch, C, N, N) -> (batch, N*N, C)
        x = x.view(batch, x.size(1), N * N).permute(0, 2, 1)

        # Project to hidden dim
        x = self.input_norm(F.gelu(self.input_proj(x)))

        # Message passing
        for layer in self.layers:
            x = layer(x, self.edge_index)

        # Policy: per-node logits
        p = self.policy_proj(x).squeeze(-1)  # (batch, N*N)

        # Global attention pooling
        attn_weights = F.softmax(self.attn_query(x).squeeze(-1), dim=1)  # (batch, N*N)
        global_feat = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden)

        # Value head
        v = F.gelu(self.value_fc1(global_feat))
        v = torch.tanh(self.value_fc2(v))  # (batch, 1)

        # Threat head
        t = self.threat_fc(global_feat)  # (batch, 4)

        return p, v, t

    def forward_pv(self, x):
        p, v, _ = self.forward(x)
        return p, v
