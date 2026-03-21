import torch
from torch import nn
from typing import Callable, Dict, List


def _apply_dynamic_edge_factor(
    edge_weight: torch.Tensor,
    msg: torch.Tensor,
    edge_mlp: nn.Module,
    edge_weight_max: float,
):
    if edge_weight.dim() != 1:
        edge_weight = edge_weight.view(-1)
    w = edge_weight.view(-1, 1)
    if edge_mlp is not None:
        w_dynamic = torch.sigmoid(edge_mlp(msg))
        w = w * torch.nn.functional.softplus(4.0 * (w_dynamic - 0.5))
    w = torch.clamp(w, min=0.0, max=float(edge_weight_max))
    return w


def _edge_softmax_by_dst(logits: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Softmax over incoming edges for each destination node.
    logits: [E] or [E, H]
    dst: [E]
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False

    if logits.numel() == 0:
        out = logits
    else:
        dst = dst.long()
        num_dst = int(dst.max().item()) + 1
        idx = dst.view(-1, 1).expand(-1, logits.shape[1])

        if hasattr(torch.Tensor, "scatter_reduce_"):
            neg_inf = torch.finfo(logits.dtype).min
            max_by_dst = torch.full(
                (num_dst, logits.shape[1]),
                fill_value=neg_inf,
                dtype=logits.dtype,
                device=logits.device,
            )
            max_by_dst.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)

            shifted = logits - max_by_dst[dst]
            exp_shifted = torch.exp(shifted)
            denom = torch.zeros(
                (num_dst, logits.shape[1]),
                dtype=logits.dtype,
                device=logits.device,
            )
            denom.scatter_add_(0, idx, exp_shifted)
            out = exp_shifted / (denom[dst] + 1e-12)
        else:
            # Fallback for older torch versions without scatter_reduce_.
            out = torch.zeros_like(logits)
            unique_dst = torch.unique(dst)
            for d in unique_dst:
                mask = dst == d
                out[mask] = torch.softmax(logits[mask], dim=0)

    if squeeze:
        out = out.squeeze(-1)
    return out


class GraphConvPosEnc(nn.Module):
    """
    Custom GCN-like spatial layer with (optional) dynamic edge reweighting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        use_dynamic_edge_weight: bool = True,
        normalize_by_in_degree: bool = True,
        edge_weight_max: float = 5.0,
        edge_dropout: float = 0.0,
        message_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_dynamic_edge_weight = bool(use_dynamic_edge_weight)
        self.normalize_by_in_degree = bool(normalize_by_in_degree)
        self.edge_weight_max = float(edge_weight_max)
        self.edge_dropout = nn.Dropout(float(edge_dropout))
        self.message_dropout = nn.Dropout(float(message_dropout))
        self.eps = float(eps)

        self.linear_in = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(float(dropout))

        if self.use_dynamic_edge_weight:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.output_dim, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 1),
            )
        else:
            self.edge_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 2 or state.dim() != 2:
            raise ValueError(f"`x` and `state` must be [BN, F], got {tuple(x.shape)}, {tuple(state.shape)}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"`edge_index` must be [2, E], got {tuple(edge_index.shape)}")

        x_and_state = torch.cat([x, state], dim=-1)  # [BN, Fx + H]
        x_proj = self.linear_in(x_and_state)  # [BN, O]

        src = edge_index[0].long()
        dst = edge_index[1].long()
        msg = x_proj[src]  # [E, O]

        if edge_weight.dim() != 1:
            edge_weight = edge_weight.view(-1)
        if edge_weight.numel() != msg.shape[0]:
            raise ValueError(
                f"`edge_weight` length mismatch: expected {msg.shape[0]}, got {edge_weight.numel()}."
            )

        w = _apply_dynamic_edge_factor(
            edge_weight=edge_weight,
            msg=msg,
            edge_mlp=self.edge_mlp,
            edge_weight_max=self.edge_weight_max,
        )
        w = self.edge_dropout(w)
        w = torch.clamp(w, min=0.0, max=float(self.edge_weight_max))
        msg = msg * w
        msg = self.message_dropout(msg)

        out = torch.zeros_like(x_proj)
        out.index_add_(0, dst, msg)
        if self.normalize_by_in_degree:
            deg = torch.zeros((x_proj.shape[0], 1), dtype=x_proj.dtype, device=x_proj.device)
            deg.index_add_(0, dst, w.abs())
            out = out / (deg + self.eps)

        if out.shape[-1] == x_proj.shape[-1]:
            out = out + x_proj

        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        return out


class GraphAttentionPosEnc(nn.Module):
    """
    GAT-like spatial layer using attention coefficients as edge weights.
    Input `edge_weight` is ignored by design to keep pure attention behavior.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ELU(),
        use_dynamic_edge_weight: bool = True,
        edge_weight_max: float = 5.0,
        num_heads: int = 1,
        attn_dropout: float = 0.0,
        edge_dropout: float = 0.0,
        message_dropout: float = 0.0,
        negative_slope: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.attn_dropout = nn.Dropout(float(attn_dropout))
        self.edge_dropout = nn.Dropout(float(edge_dropout))
        self.message_dropout = nn.Dropout(float(message_dropout))
        self.negative_slope = float(negative_slope)

        if self.num_heads <= 0:
            raise ValueError("`num_heads` must be positive.")
        if self.output_dim % self.num_heads != 0:
            raise ValueError(
                f"`output_dim` ({self.output_dim}) must be divisible by `num_heads` ({self.num_heads})."
            )
        self.head_dim = self.output_dim // self.num_heads

        self.linear_in = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.num_heads, self.head_dim))

        self.activation = activation
        self.dropout = nn.Dropout(float(dropout))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 2 or state.dim() != 2:
            raise ValueError(f"`x` and `state` must be [BN, F], got {tuple(x.shape)}, {tuple(state.shape)}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"`edge_index` must be [2, E], got {tuple(edge_index.shape)}")
        _ = edge_weight  # kept in signature for interface compatibility, intentionally unused.

        x_and_state = torch.cat([x, state], dim=-1)  # [BN, Fx + H]
        h_flat = self.linear_in(x_and_state)  # [BN, O]
        h = h_flat.view(h_flat.shape[0], self.num_heads, self.head_dim)  # [BN, H, D]

        src = edge_index[0].long()
        dst = edge_index[1].long()
        h_src = h[src]  # [E, H, D]
        h_dst = h[dst]  # [E, H, D]

        logits = (h_src * self.attn_src.unsqueeze(0)).sum(dim=-1) + (h_dst * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        logits = torch.nn.functional.leaky_relu(logits, negative_slope=self.negative_slope)  # [E, H]

        alpha = _edge_softmax_by_dst(logits=logits, dst=dst)  # [E, H]
        alpha = self.attn_dropout(alpha)
        alpha = self.edge_dropout(alpha)

        msg = h_src * alpha.unsqueeze(-1)  # [E, H, D]
        msg_flat = msg.reshape(msg.shape[0], self.output_dim)  # [E, O]
        msg_flat = self.message_dropout(msg_flat)

        out_flat = torch.zeros_like(h_flat)
        out_flat.index_add_(0, dst, msg_flat)

        if out_flat.shape[-1] == h_flat.shape[-1]:
            out_flat = out_flat + h_flat

        if self.activation is not None:
            out_flat = self.activation(out_flat)
        out_flat = self.dropout(out_flat)
        return out_flat


SpatialLayerBuilderFn = Callable[..., nn.Module]

_SPATIAL_LAYER_BUILDERS: Dict[str, SpatialLayerBuilderFn] = {}


def register_spatial_layer(*names: str):
    normalized = [str(n).strip().lower() for n in names if str(n).strip()]
    if len(normalized) == 0:
        raise ValueError("At least one non-empty spatial layer name must be provided.")

    def _decorator(fn):
        for name in normalized:
            _SPATIAL_LAYER_BUILDERS[name] = fn
        return fn

    return _decorator


def available_spatial_layers():
    return sorted(_SPATIAL_LAYER_BUILDERS.keys())


@register_spatial_layer("gcn", "graphconv", "custom_gcn", "graph_conv", "conv")
def _build_spatial_gcn(
    input_dim: int,
    output_dim: int,
    dropout: float = 0.1,
    activation: nn.Module = None,
    use_dynamic_edge_weight: bool = True,
    normalize_by_in_degree: bool = True,
    edge_weight_max: float = 5.0,
    edge_dropout: float = 0.0,
    message_dropout: float = 0.0,
    **kwargs,
):
    if activation is None:
        activation = nn.GELU()
    return GraphConvPosEnc(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
        use_dynamic_edge_weight=use_dynamic_edge_weight,
        normalize_by_in_degree=normalize_by_in_degree,
        edge_weight_max=edge_weight_max,
        edge_dropout=edge_dropout,
        message_dropout=message_dropout,
    )


@register_spatial_layer("gat", "graph_attention", "attention")
def _build_spatial_gat(
    input_dim: int,
    output_dim: int,
    dropout: float = 0.1,
    activation: nn.Module = None,
    use_dynamic_edge_weight: bool = True,
    edge_weight_max: float = 5.0,
    gat_num_heads: int = 1,
    gat_attn_dropout: float = 0.0,
    edge_dropout: float = 0.0,
    message_dropout: float = 0.0,
    gat_negative_slope: float = 0.2,
    **kwargs,
):
    if activation is None:
        activation = nn.ELU()
    # `use_dynamic_edge_weight` / `edge_weight_max` are kept for config compatibility,
    # but GAT now relies on attention coefficients only.
    return GraphAttentionPosEnc(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
        use_dynamic_edge_weight=use_dynamic_edge_weight,
        edge_weight_max=edge_weight_max,
        num_heads=gat_num_heads,
        attn_dropout=gat_attn_dropout,
        edge_dropout=edge_dropout,
        message_dropout=message_dropout,
        negative_slope=gat_negative_slope,
    )


def build_spatial_layer(
    kind: str,
    input_dim: int,
    output_dim: int,
    dropout: float = 0.1,
    activation: nn.Module = None,
    use_dynamic_edge_weight: bool = True,
    normalize_by_in_degree: bool = True,
    edge_weight_max: float = 5.0,
    edge_dropout: float = 0.0,
    message_dropout: float = 0.0,
    gat_num_heads: int = 1,
    gat_attn_dropout: float = 0.0,
    gat_negative_slope: float = 0.2,
    **kwargs,
):
    key = str(kind).strip().lower()
    if key not in _SPATIAL_LAYER_BUILDERS:
        available = ", ".join(available_spatial_layers())
        raise ValueError(f"Unsupported spatial model type: `{kind}`. Available: {available}")

    return _SPATIAL_LAYER_BUILDERS[key](
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
        use_dynamic_edge_weight=use_dynamic_edge_weight,
        normalize_by_in_degree=normalize_by_in_degree,
        edge_weight_max=edge_weight_max,
        edge_dropout=edge_dropout,
        message_dropout=message_dropout,
        gat_num_heads=gat_num_heads,
        gat_attn_dropout=gat_attn_dropout,
        gat_negative_slope=gat_negative_slope,
        **kwargs,
    )
