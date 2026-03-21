from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .slope_routing_model import SlopeRouting


class LinearMLPProjector(nn.Module):
    """
    Lightweight alternative to SlopeRouting:
    - project dynamic features per time step
    - project static features per node
    - fuse with MLP
    """

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dyn_dim = int(dyn_dim)
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        self.dynamic_proj = nn.Sequential(
            nn.Linear(self.dyn_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(self.static_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, dyn_x: torch.Tensor, static_feat: torch.Tensor):
        # dyn_x: [BN, T, F], static_feat: [BN, D]
        if dyn_x.dim() != 3:
            raise ValueError(f"`dyn_x` must be [BN,T,F], got {tuple(dyn_x.shape)}")
        if static_feat.dim() != 2:
            raise ValueError(f"`static_feat` must be [BN,D], got {tuple(static_feat.shape)}")
        if dyn_x.shape[0] != static_feat.shape[0]:
            raise ValueError(
                f"BN mismatch between dyn_x and static_feat: {dyn_x.shape[0]} vs {static_feat.shape[0]}"
            )

        dyn_h = self.dynamic_proj(dyn_x)  # [BN, T, H]
        static_h = self.static_proj(static_feat).unsqueeze(1).expand(-1, dyn_x.shape[1], -1)  # [BN, T, H]
        out = self.fuse(torch.cat([dyn_h, static_h], dim=-1))
        return out


InputProjectorBuilderFn = Callable[..., nn.Module]

_INPUT_PROJECTOR_BUILDERS: Dict[str, InputProjectorBuilderFn] = {}


def register_input_projector(*names: str):
    normalized = [str(n).strip().lower() for n in names if str(n).strip()]
    if len(normalized) == 0:
        raise ValueError("At least one non-empty input projector name must be provided.")

    def _decorator(fn: InputProjectorBuilderFn):
        for name in normalized:
            _INPUT_PROJECTOR_BUILDERS[name] = fn
        return fn

    return _decorator


def available_input_projectors() -> List[str]:
    return sorted(_INPUT_PROJECTOR_BUILDERS.keys())


def build_input_projector(name: str, **kwargs) -> nn.Module:
    key = str(name).strip().lower()
    if key not in _INPUT_PROJECTOR_BUILDERS:
        available = ", ".join(available_input_projectors())
        raise ValueError(f"Unsupported input projector: `{name}`. Available: {available}")
    return _INPUT_PROJECTOR_BUILDERS[key](**kwargs)


@register_input_projector("slope", "slope_routing", "film_gru")
def _build_slope_projector(
    dyn_dim: int,
    static_dim: int,
    seq_len: int,
    hidden_dim: int,
    num_nodes: int,
    use_film_lag: bool = True,
    **kwargs,
):
    return SlopeRouting(
        dyn_dim=int(dyn_dim),
        static_dim=int(static_dim),
        seq_len=int(seq_len),
        lstm_units=int(hidden_dim),
        num_nodes=int(num_nodes),
        num_lstm_layers=1,
        use_film_lstm=bool(use_film_lag),
    )


@register_input_projector("mlp", "linear_mlp", "linear")
def _build_mlp_projector(
    dyn_dim: int,
    static_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    **kwargs,
):
    return LinearMLPProjector(
        dyn_dim=int(dyn_dim),
        static_dim=int(static_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )


class NodeInputProjector(nn.Module):
    """
    Switchable input projector for GR2N:
    - name='slope' -> SlopeRouting
    - name='mlp'   -> LinearMLPProjector
    """

    def __init__(
        self,
        name: str,
        dyn_dim: int,
        static_dim: int,
        seq_len: int,
        hidden_dim: int,
        num_nodes: int,
        use_film_lag: bool = True,
        dropout: float = 0.1,
        projector_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.name = str(name).strip().lower()
        kwargs = {} if projector_kwargs is None else dict(projector_kwargs)
        kwargs.setdefault("use_film_lag", bool(use_film_lag))
        kwargs.setdefault("dropout", float(dropout))
        self.projector = build_input_projector(
            name=self.name,
            dyn_dim=int(dyn_dim),
            static_dim=int(static_dim),
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_nodes=int(num_nodes),
            **kwargs,
        )

    def forward(self, dyn_x: torch.Tensor, static_feat: torch.Tensor):
        return self.projector(dyn_x, static_feat)
