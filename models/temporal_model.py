import torch
import torch.nn as nn
from typing import Callable, Dict, Optional

from .spatial_model import build_spatial_layer


class _RMSNormFallback(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = x / rms
        if self.weight is not None:
            y = y * self.weight
        return y


def _build_hidden_norm(
    kind: str,
    dim: int,
    eps: float = 1e-5,
    affine: bool = True,
    batchnorm_momentum: float = 0.1,
):
    key = str(kind).strip().lower()
    if key in {"none", "identity", ""}:
        return nn.Identity()
    if key in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(int(dim), eps=float(eps), elementwise_affine=bool(affine))
    if key in {"batchnorm", "batch_norm", "bn", "batchnorm1d"}:
        return nn.BatchNorm1d(
            int(dim),
            eps=float(eps),
            momentum=float(batchnorm_momentum),
            affine=bool(affine),
            track_running_stats=True,
        )
    if key in {"rmsnorm", "rms_norm", "rms"}:
        if hasattr(nn, "RMSNorm"):
            return nn.RMSNorm(int(dim), eps=float(eps), elementwise_affine=bool(affine))
        return _RMSNormFallback(int(dim), eps=float(eps), affine=bool(affine))
    raise ValueError(
        f"Unsupported cell norm type: `{kind}`. Supported: none, layernorm, batchnorm, rmsnorm."
    )


class TailMeanTemporalReadout(nn.Module):
    """
    Baseline temporal readout:
    project hidden states on the last `input_freq_per_day` steps, then average.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        input_freq_per_day: int = 8,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.input_freq_per_day = int(input_freq_per_day)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        # hidden_seq: [BN, T, H]
        if hidden_seq.dim() != 3:
            raise ValueError(f"`hidden_seq` must be [BN,T,H], got {tuple(hidden_seq.shape)}")

        tail_steps = min(self.input_freq_per_day, hidden_seq.shape[1])
        tail_hidden = hidden_seq[:, -tail_steps:, :]  # [BN, S, H]
        tail_out = self.fc_out(tail_hidden)           # [BN, S, O]
        return tail_out.mean(dim=1)                   # [BN, O]


class TemporalAttentionReadout(nn.Module):
    """
    Temporal attention readout:
    use the last hidden state as query to aggregate sequence features.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        input_freq_per_day: int = 8,
        attn_use_tail_window: bool = True,
        attn_dropout: float = 0.0,
        attn_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.input_freq_per_day = int(input_freq_per_day)
        self.attn_use_tail_window = bool(attn_use_tail_window)
        self.attn_dropout = nn.Dropout(float(attn_dropout))
        self.attn_temperature = float(attn_temperature)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.last_attention_weights = None

    def forward(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        # hidden_seq: [BN, T, H]
        if hidden_seq.dim() != 3:
            raise ValueError(f"`hidden_seq` must be [BN,T,H], got {tuple(hidden_seq.shape)}")

        if self.attn_use_tail_window:
            tail_steps = min(self.input_freq_per_day, hidden_seq.shape[1])
            hidden_seq = hidden_seq[:, -tail_steps:, :]

        query = hidden_seq[:, -1, :]  # [BN, H]
        scores = torch.einsum("bth,bh->bt", hidden_seq, query)
        scores = scores / (self.hidden_dim ** 0.5)
        scores = scores / max(self.attn_temperature, 1e-6)

        alpha = torch.softmax(scores, dim=1)
        alpha = self.attn_dropout(alpha)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)

        context = torch.einsum("bt,bth->bh", alpha, hidden_seq)  # [BN, H]
        self.last_attention_weights = alpha.detach()
        return self.fc_out(context)  # [BN, O]


TemporalReadoutBuilderFn = Callable[..., nn.Module]
_TEMPORAL_READOUT_BUILDERS: Dict[str, TemporalReadoutBuilderFn] = {}


def register_temporal_readout(*names: str):
    normalized = [str(n).strip().lower() for n in names if str(n).strip()]
    if len(normalized) == 0:
        raise ValueError("At least one non-empty temporal readout name must be provided.")

    def _decorator(fn):
        for name in normalized:
            _TEMPORAL_READOUT_BUILDERS[name] = fn
        return fn

    return _decorator


def available_temporal_readouts():
    return sorted(_TEMPORAL_READOUT_BUILDERS.keys())


@register_temporal_readout("tail_mean", "mean", "gru")
def _build_tail_mean_temporal_readout(
    hidden_dim: int,
    output_dim: int,
    input_freq_per_day: int = 8,
    **kwargs,
):
    return TailMeanTemporalReadout(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        input_freq_per_day=input_freq_per_day,
    )


@register_temporal_readout("attention", "attn", "temporal_attention", "time_attention")
def _build_temporal_attention_readout(
    hidden_dim: int,
    output_dim: int,
    input_freq_per_day: int = 8,
    attn_use_tail_window: bool = True,
    attn_dropout: float = 0.0,
    attn_temperature: float = 1.0,
    **kwargs,
):
    return TemporalAttentionReadout(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        input_freq_per_day=input_freq_per_day,
        attn_use_tail_window=attn_use_tail_window,
        attn_dropout=attn_dropout,
        attn_temperature=attn_temperature,
    )


def build_temporal_readout(
    kind: str,
    hidden_dim: int,
    output_dim: int,
    input_freq_per_day: int = 8,
    **kwargs,
):
    key = str(kind).strip().lower()
    if key not in _TEMPORAL_READOUT_BUILDERS:
        available = ", ".join(available_temporal_readouts())
        raise ValueError(f"Unsupported temporal readout type: `{kind}`. Available: {available}")

    return _TEMPORAL_READOUT_BUILDERS[key](
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        input_freq_per_day=input_freq_per_day,
        **kwargs,
    )


class GraphGRUCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_proj: int = None,
        act: nn.Module = nn.Tanh(),
        dropout: float = 0.1,
        spatial_model_type: str = "gcn",
        spatial_layer_kwargs: Optional[Dict] = None,
        norm_type: str = "none",
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        batchnorm_momentum: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.num_proj = None if num_proj is None else int(num_proj)
        self.activation = act

        spatial_kwargs = {} if spatial_layer_kwargs is None else dict(spatial_layer_kwargs)
        self.gconv_gates = build_spatial_layer(
            kind=spatial_model_type,
            input_dim=self.input_dim + self.hidden_dim,
            output_dim=2 * self.hidden_dim,
            dropout=dropout,
            **spatial_kwargs,
        )
        self.gconv_candidate = build_spatial_layer(
            kind=spatial_model_type,
            input_dim=self.input_dim + self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=dropout,
            **spatial_kwargs,
        )
        self.project = nn.Linear(self.hidden_dim, self.num_proj) if self.num_proj is not None else None
        self.state_norm = _build_hidden_norm(
            kind=norm_type,
            dim=self.hidden_dim,
            eps=norm_eps,
            affine=norm_affine,
            batchnorm_momentum=batchnorm_momentum,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        # x: [BN, F], h_prev: [BN, H]
        if x.dim() != 2 or h_prev.dim() != 2:
            raise ValueError(f"`x` and `h_prev` must be [BN,F], [BN,H], got {tuple(x.shape)}, {tuple(h_prev.shape)}")

        batch_size = int(x.shape[0] // self.num_nodes)
        gates = self.gconv_gates(x, h_prev, edge_index, edge_weight)  # [BN, 2H]
        gates = gates.view(batch_size, self.num_nodes, 2 * self.hidden_dim)
        r, z = torch.split(gates, self.hidden_dim, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        h_prev_3d = h_prev.view(batch_size, self.num_nodes, self.hidden_dim)
        h_reset = (r * h_prev_3d).view(batch_size * self.num_nodes, self.hidden_dim)

        n_tilde = self.gconv_candidate(x, h_reset, edge_index, edge_weight)
        n_tilde = n_tilde.view(batch_size, self.num_nodes, self.hidden_dim)
        if self.activation is not None:
            n_tilde = self.activation(n_tilde)

        h_new = (1.0 - z) * n_tilde + z * h_prev_3d
        new_h = h_new.view(batch_size * self.num_nodes, self.hidden_dim)
        new_h = self.state_norm(new_h)

        if self.project is not None:
            output = self.project(new_h)
        else:
            output = new_h

        return output, new_h

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(int(batch_size) * self.num_nodes, self.hidden_dim, device=device)


class GraphGRUBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.1,
        act: nn.Module = nn.ReLU(),
        input_freq_per_day: int = 8,
        spatial_model_type: str = "gcn",
        spatial_layer_kwargs: Optional[Dict] = None,
        temporal_readout_type: str = "tail_mean",
        temporal_readout_kwargs: Optional[Dict] = None,
        use_layer_residual: bool = False,
        layer_residual_dropout: float = 0.0,
        cell_norm_type: str = "none",
        cell_norm_eps: float = 1e-5,
        cell_norm_affine: bool = True,
        cell_batchnorm_momentum: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self._input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.input_freq_per_day = int(input_freq_per_day)
        self.use_layer_residual = bool(use_layer_residual)
        self.layer_residual_dropout = nn.Dropout(float(layer_residual_dropout))

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            self.cells.append(
                GraphGRUCell(
                    input_dim=self._input_dim if i == 0 else self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    num_nodes=self.num_nodes,
                    act=act,
                    num_proj=None,
                    dropout=dropout,
                    spatial_model_type=spatial_model_type,
                    spatial_layer_kwargs=spatial_layer_kwargs,
                    norm_type=cell_norm_type,
                    norm_eps=cell_norm_eps,
                    norm_affine=cell_norm_affine,
                    batchnorm_momentum=cell_batchnorm_momentum,
                )
            )

        readout_kwargs = {} if temporal_readout_kwargs is None else dict(temporal_readout_kwargs)
        self.temporal_readout = build_temporal_readout(
            kind=temporal_readout_type,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            input_freq_per_day=self.input_freq_per_day,
            **readout_kwargs,
        )

    @property
    def input_dim(self):
        return self._input_dim

    def forward(
        self,
        x: torch.Tensor,
        init_hidden_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        # x: [BN, T, F]
        if x.dim() != 3:
            raise ValueError(f"`x` must be [BN,T,F], got {tuple(x.shape)}")
        if init_hidden_state.dim() != 3:
            raise ValueError(f"`init_hidden_state` must be [L,BN,H], got {tuple(init_hidden_state.shape)}")

        _, t_steps, _ = x.shape
        current_inputs = x
        output_hidden = []

        for i_layer in range(self.num_layers):
            residual_input = current_inputs
            hidden_state = init_hidden_state[i_layer]
            output_inner = []
            for t in range(t_steps):
                out_t, hidden_state = self.cells[i_layer](
                    current_inputs[:, t, :],
                    hidden_state,
                    edge_index,
                    edge_weight,
                )
                output_inner.append(out_t.unsqueeze(1))
            output_inner = torch.cat(output_inner, dim=1)  # [BN, T, H]
            if (
                self.use_layer_residual
                and i_layer > 0
                and output_inner.shape == residual_input.shape
            ):
                output_inner = output_inner + self.layer_residual_dropout(residual_input)
            output_hidden.append(hidden_state)
            current_inputs = output_inner

        node_pred = self.temporal_readout(current_inputs)  # [BN, O]
        return node_pred, torch.stack(output_hidden, dim=0), current_inputs

    def init_hidden(self, batch_size: int, device: torch.device):
        states = []
        for cell in self.cells:
            states.append(cell.init_hidden(batch_size, device=device))
        return torch.stack(states, dim=0)
