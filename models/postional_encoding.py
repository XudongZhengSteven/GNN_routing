import torch
from torch import nn


def _as_batch_matrix(x: torch.Tensor, ndim: int):
    if x.dim() == ndim - 1:
        return x.unsqueeze(0), True
    if x.dim() == ndim:
        return x, False
    raise ValueError(f"Unexpected tensor dim={x.dim()}, expected {ndim-1} or {ndim}.")


class SpatialPositionalEncoding(nn.Module):
    """
    Build graph edge_index + edge_weight from path-level edge attributes and masks.
    """

    def __init__(
        self,
        edge_attr_dim: int,
        hidden_dim: int,
        threshold: float = 1e-6,
        cache_topology: bool = True,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.cache_topology = bool(cache_topology)
        self.edge_mlp = nn.Sequential(
            nn.Linear(int(edge_attr_dim), int(hidden_dim)),
            nn.LeakyReLU(0.1),
            nn.Linear(int(hidden_dim), 1),
        )
        self._topology_cache = {}

    def forward(
        self,
        mask_downstream_adj: torch.Tensor,
        mask_khop_up_adj: torch.Tensor,
        full_path_edge_attr_adj: torch.Tensor,
    ):
        """
        Inputs:
        - mask_downstream_adj: [N, N] or [B, N, N]
        - mask_khop_up_adj: [N, N] or [B, N, N]
        - full_path_edge_attr_adj: [N, N, F] or [B, N, N, F]

        Outputs:
        - edge_index: [2, E] (shared topology for a batch)
        - edge_weight: [E] if single sample else [B, E]
        """
        path_attr, squeeze_b = _as_batch_matrix(full_path_edge_attr_adj, 4)
        mask_down, _ = _as_batch_matrix(mask_downstream_adj, 3)
        mask_up, _ = _as_batch_matrix(mask_khop_up_adj, 3)

        if mask_down.shape != mask_up.shape:
            raise ValueError(f"Mask shape mismatch: {tuple(mask_down.shape)} vs {tuple(mask_up.shape)}")
        if path_attr.shape[:3] != mask_down.shape:
            raise ValueError(
                "Path-attr and mask shape mismatch: "
                f"path={tuple(path_attr.shape)}, mask={tuple(mask_down.shape)}"
            )

        batch_size, n_nodes, _, _ = path_attr.shape

        score = self.edge_mlp(path_attr).squeeze(-1)  # [B, N, N]
        mask = (mask_down > 0) & (mask_up > 0)       # [B, N, N]

        score = score.masked_fill(~mask, -1e9)
        w = torch.softmax(score, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        w = torch.where(mask, w, torch.zeros_like(w))
        w = torch.where(w >= self.threshold, w, torch.zeros_like(w))

        # Assume same topology across batches for RoutingDataset. Cache src/tgt for speed.
        mask0 = mask[0]
        edge_count = int(mask0.sum().item())
        cache_key = (str(mask.device), int(n_nodes), edge_count)

        use_cache = False
        if self.cache_topology and cache_key in self._topology_cache:
            src_cached, tgt_cached = self._topology_cache[cache_key]
            if src_cached.numel() == edge_count and bool(mask0[src_cached, tgt_cached].all()):
                src, tgt = src_cached, tgt_cached
                use_cache = True

        if not use_cache:
            src, tgt = torch.nonzero(mask0, as_tuple=True)
            if self.cache_topology:
                self._topology_cache[cache_key] = (src, tgt)
        edge_index = torch.stack([src, tgt], dim=0).long()

        edge_weight = w[:, src, tgt]  # [B, E]
        if squeeze_b and batch_size == 1:
            edge_weight = edge_weight.squeeze(0)  # [E]

        return edge_index, edge_weight
