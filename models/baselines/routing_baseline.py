from typing import Any

import torch
import torch.nn as nn


def _get_field(batch: Any, key: str):
    if isinstance(batch, dict):
        return batch[key]
    if hasattr(batch, key):
        return getattr(batch, key)
    raise KeyError(f"Batch has no field `{key}`")


class RoutingBaselineModel(nn.Module):
    """
    A minimal baseline model for RoutingDataset:
    - temporal encoder: GRU over x(node, window, features)
    - static fusion: concatenate node_attr
    - outlet readout: gather outlet node predictions
    """

    def __init__(
        self,
        input_dim: int = 2,
        node_attr_dim: int = 3,
        hidden_dim: int = 64,
        pred_len: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.node_attr_dim = int(node_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.pred_len = int(pred_len)

        self.temporal_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.node_attr_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.node_out = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, batch: Any):
        x = _get_field(batch, "x")
        node_attr = _get_field(batch, "node_attr")
        outlet_index = _get_field(batch, "outlet_index").long()

        is_single_sample = False
        if x.dim() == 3:
            # Single sample: [N, W, F]
            is_single_sample = True
            x = x.unsqueeze(0)
            node_attr = node_attr.unsqueeze(0)
            outlet_index = outlet_index.unsqueeze(0)

        if node_attr.dim() == 2:
            node_attr = node_attr.unsqueeze(0)
        if outlet_index.dim() == 1:
            outlet_index = outlet_index.unsqueeze(0)

        if x.dim() != 4:
            raise ValueError(f"`x` must be [B,N,W,F] or [N,W,F], got {tuple(x.shape)}")

        batch_size, num_nodes, window_size, in_dim = x.shape
        if in_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, but x.shape[-1]={in_dim}")

        if node_attr.shape[0] != batch_size or node_attr.shape[1] != num_nodes:
            raise ValueError(
                "node_attr shape mismatch: "
                f"x={tuple(x.shape)}, node_attr={tuple(node_attr.shape)}"
            )
        if node_attr.shape[-1] != self.node_attr_dim:
            raise ValueError(
                f"Expected node_attr_dim={self.node_attr_dim}, got node_attr.shape[-1]={node_attr.shape[-1]}"
            )

        x_seq = x.reshape(batch_size * num_nodes, window_size, in_dim)
        _, h_n = self.temporal_gru(x_seq)
        temporal_feat = h_n[-1]  # [B*N, H]

        node_attr_flat = node_attr.reshape(batch_size * num_nodes, self.node_attr_dim)
        node_feat = torch.cat([temporal_feat, node_attr_flat], dim=-1)
        node_feat = self.node_mlp(node_feat)
        node_pred = self.node_out(node_feat)  # [B*N, P]

        # [B, N, P] -> [B, P, N]
        node_pred = node_pred.view(batch_size, num_nodes, self.pred_len).transpose(1, 2).contiguous()

        if outlet_index.shape[0] == 1 and batch_size > 1:
            outlet_index = outlet_index.expand(batch_size, -1)
        if outlet_index.shape[0] != batch_size:
            raise ValueError(
                "outlet_index batch dimension mismatch: "
                f"batch_size={batch_size}, outlet_index.shape={tuple(outlet_index.shape)}"
            )

        gather_index = outlet_index.unsqueeze(1).expand(-1, self.pred_len, -1)
        y_pred = torch.gather(node_pred, dim=2, index=gather_index)

        if is_single_sample:
            y_pred = y_pred.squeeze(0)
        return y_pred
