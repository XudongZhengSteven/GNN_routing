from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from .baselines import RoutingBaselineModel
from .routing_model import GR2N, GR2NSeq2Seq


def _infer_dims_from_dataset(dataset=None):
    if dataset is None:
        return {}
    if len(dataset) == 0:
        return {}

    sample = dataset[0]
    return {
        "input_dim": int(sample.x.shape[-1]),
        "node_attr_dim": int(sample.node_attr.shape[-1]),
        "pred_len": int(sample.y.shape[0]),
        "num_nodes": int(sample.x.shape[0]),
        "seq_len": int(sample.x.shape[1]),
        "edge_attr_dim": int(sample.edge_attr.shape[-1]),
        "input_freq_per_day": int(getattr(dataset, "_input_freq_per_day", 1)),
    }


def _resolve_model_name(name_raw: str) -> str:
    key = str(name_raw).strip().lower()
    alias = {
        "routing_direct": "gr2n",
        "direct": "gr2n",
        "grn": "gr2n",
        "routing_seq": "gr2n_seq2seq",
        "seq": "gr2n_seq2seq",
        "seq2seq": "gr2n_seq2seq",
    }
    return alias.get(key, key)


def _resolve_input_projector(cfg: Dict[str, Any]) -> bool:
    # True -> SlopeRouting, False -> LinearMLPProjector
    if "input_projector" not in cfg:
        return bool(cfg.get("use_slope_routing", True))

    key = str(cfg.get("input_projector")).strip().lower()
    if key in {"slope", "slope_routing", "film_gru"}:
        return True
    if key in {"mlp", "linear_mlp", "linear"}:
        return False
    raise ValueError(
        f"Unsupported `input_projector`: `{cfg.get('input_projector')}`. "
        "Supported: slope, mlp."
    )


def _resolve_spatial_type(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("spatial", cfg.get("spatial_model_type", "gcn"))
    if isinstance(raw, dict):
        raw = raw.get("name", raw.get("type", "gcn"))
    key = str(raw).strip().lower()
    if key in {"gcn", "graphconv", "conv", "custom_gcn", "graph_conv"}:
        return "gcn"
    if key in {"gat", "attention", "graph_attention"}:
        return "gat"
    raise ValueError(
        f"Unsupported spatial model type: `{raw}`. "
        "Supported: gcn, gat."
    )


def _resolve_edge_weight_mode(cfg: Dict[str, Any]) -> bool:
    # True -> dynamic edge reweighting, False -> static only.
    if "edge_weight_mode" not in cfg:
        return bool(cfg.get("spatial_use_dynamic_edge_weight", True))

    key = str(cfg.get("edge_weight_mode")).strip().lower()
    if key in {"dynamic", "dyn"}:
        return True
    if key in {"static"}:
        return False
    raise ValueError(
        f"Unsupported `edge_weight_mode`: `{cfg.get('edge_weight_mode')}`. "
        "Supported: dynamic, static."
    )


def _resolve_input_projector_name(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("input_projector", None)
    if isinstance(raw, dict):
        raw = raw.get("name", "slope")

    if raw is not None:
        key = str(raw).strip().lower()
        if key in {"slope", "slope_routing", "film_gru"}:
            return "slope"
        if key in {"mlp", "linear_mlp", "linear"}:
            return "mlp"
        raise ValueError(
            f"Unsupported `input_projector`: `{raw}`. "
            "Supported: slope, mlp."
        )

    return "slope" if _resolve_input_projector(cfg) else "mlp"


def _build_input_projector_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    projector_cfg = cfg.get("input_projector", None)
    if isinstance(projector_cfg, dict):
        kwargs.update({k: v for k, v in projector_cfg.items() if k not in {"name", "type"}})
    extra_cfg = cfg.get("input_projector_kwargs", None)
    if isinstance(extra_cfg, dict):
        kwargs.update(extra_cfg)
    return kwargs


def _build_spatial_layer_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    spatial_cfg: Dict[str, Any] = {}
    spatial_raw = cfg.get("spatial", None)
    if isinstance(spatial_raw, dict):
        spatial_cfg.update({k: v for k, v in spatial_raw.items() if k not in {"name", "type"}})

    extra_cfg = cfg.get("spatial_layer_kwargs", None)
    if isinstance(extra_cfg, dict):
        spatial_cfg.update(extra_cfg)

    # Backward compatibility: flat keys -> kwargs.
    spatial_cfg.setdefault("use_dynamic_edge_weight", _resolve_edge_weight_mode(cfg))
    spatial_cfg.setdefault(
        "normalize_by_in_degree",
        bool(cfg.get("spatial_normalize_by_in_degree", cfg.get("in_degree_norm", True))),
    )
    spatial_cfg.setdefault(
        "edge_weight_max",
        float(cfg.get("spatial_edge_weight_max", cfg.get("edge_weight_max", 5.0))),
    )
    spatial_cfg.setdefault(
        "edge_dropout",
        float(cfg.get("spatial_edge_dropout", cfg.get("edge_dropout", 0.0))),
    )
    spatial_cfg.setdefault(
        "message_dropout",
        float(cfg.get("spatial_message_dropout", cfg.get("message_dropout", 0.0))),
    )
    spatial_cfg.setdefault("gat_num_heads", int(cfg.get("gat_num_heads", cfg.get("heads", 1))))
    spatial_cfg.setdefault("gat_attn_dropout", float(cfg.get("gat_attn_dropout", cfg.get("attn_dropout", 0.0))))
    spatial_cfg.setdefault(
        "gat_negative_slope",
        float(cfg.get("gat_negative_slope", cfg.get("attn_negative_slope", 0.2))),
    )
    return spatial_cfg


def _resolve_temporal_readout_type(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("temporal", cfg.get("temporal_readout_type", "tail_mean"))
    if isinstance(raw, dict):
        raw = raw.get("name", raw.get("type", "tail_mean"))

    key = str(raw).strip().lower()
    if key in {"tail_mean", "mean", "gru", "graph_gru", "default"}:
        return "tail_mean"
    if key in {"attention", "attn", "temporal_attention", "time_attention", "gru_attn", "temporal_attn"}:
        return "attention"

    raise ValueError(
        f"Unsupported temporal readout type: `{raw}`. "
        "Supported: tail_mean, attention."
    )


def _build_temporal_readout_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    temporal_cfg: Dict[str, Any] = {}
    temporal_raw = cfg.get("temporal", None)
    if isinstance(temporal_raw, dict):
        temporal_cfg.update({k: v for k, v in temporal_raw.items() if k not in {"name", "type"}})

    extra_cfg = cfg.get("temporal_readout_kwargs", None)
    if isinstance(extra_cfg, dict):
        temporal_cfg.update(extra_cfg)

    # Backward compatibility: flat keys -> kwargs.
    # Only set when explicitly provided so each module can keep its own default.
    if "attn_use_tail_window" in cfg and "attn_use_tail_window" not in temporal_cfg:
        temporal_cfg["attn_use_tail_window"] = bool(cfg.get("attn_use_tail_window"))

    if "temporal_attn_dropout" in cfg and "attn_dropout" not in temporal_cfg:
        temporal_cfg["attn_dropout"] = float(cfg.get("temporal_attn_dropout"))
    if "time_attn_dropout" in cfg and "attn_dropout" not in temporal_cfg:
        temporal_cfg["attn_dropout"] = float(cfg.get("time_attn_dropout"))

    if "attn_temperature" in cfg and "attn_temperature" not in temporal_cfg:
        temporal_cfg["attn_temperature"] = float(cfg.get("attn_temperature"))

    if "attn_tail_steps" in cfg and "attn_tail_steps" not in temporal_cfg:
        temporal_cfg["attn_tail_steps"] = int(cfg.get("attn_tail_steps"))
    if "temporal_attn_tail_steps" in cfg and "attn_tail_steps" not in temporal_cfg:
        temporal_cfg["attn_tail_steps"] = int(cfg.get("temporal_attn_tail_steps"))

    return temporal_cfg


def _build_temporal_block_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block_cfg: Dict[str, Any] = {}
    block_raw = cfg.get("temporal_block", None)
    if isinstance(block_raw, dict):
        block_cfg.update({k: v for k, v in block_raw.items() if k not in {"name", "type"}})

    extra_cfg = cfg.get("temporal_block_kwargs", None)
    if isinstance(extra_cfg, dict):
        block_cfg.update(extra_cfg)

    block_cfg.setdefault("use_layer_residual", bool(cfg.get("use_layer_residual", False)))
    block_cfg.setdefault("layer_residual_dropout", float(cfg.get("layer_residual_dropout", 0.0)))
    block_cfg.setdefault("cell_norm_type", str(cfg.get("cell_norm_type", "none")))
    block_cfg.setdefault("cell_norm_eps", float(cfg.get("cell_norm_eps", 1e-5)))
    block_cfg.setdefault("cell_norm_affine", bool(cfg.get("cell_norm_affine", True)))
    block_cfg.setdefault("cell_batchnorm_momentum", float(cfg.get("cell_batchnorm_momentum", 0.1)))
    return block_cfg


@dataclass
class ModelBuildConfig:
    name: str
    input_dim: int
    node_attr_dim: int
    pred_len: int
    num_nodes: int
    seq_len: int
    edge_attr_dim: int
    input_freq_per_day: int
    hidden_dim: int
    dropout: float
    num_layers: int
    pos_hidden_dim: int
    use_film_lag: bool
    input_projector_name: str
    input_projector_kwargs: Dict[str, Any]
    spatial_model_type: str
    spatial_layer_kwargs: Dict[str, Any]
    temporal_readout_type: str
    temporal_readout_kwargs: Dict[str, Any]
    temporal_block_kwargs: Dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], dataset_dims: Dict[str, Any]):
        cfg = {} if cfg is None else dict(cfg)
        ds = {} if dataset_dims is None else dict(dataset_dims)

        return cls(
            name=_resolve_model_name(cfg.get("name", "routing_baseline")),
            input_dim=int(cfg.get("input_dim", ds.get("input_dim", 2))),
            node_attr_dim=int(cfg.get("node_attr_dim", ds.get("node_attr_dim", 3))),
            pred_len=int(cfg.get("pred_len", ds.get("pred_len", 1))),
            num_nodes=int(cfg.get("num_nodes", ds.get("num_nodes", 0))),
            seq_len=int(cfg.get("seq_len", ds.get("seq_len", 0))),
            edge_attr_dim=int(cfg.get("edge_attr_dim", ds.get("edge_attr_dim", 4))),
            input_freq_per_day=int(cfg.get("input_freq_per_day", ds.get("input_freq_per_day", 1))),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            dropout=float(cfg.get("dropout", 0.1)),
            num_layers=int(cfg.get("num_layers", 2)),
            pos_hidden_dim=int(cfg.get("pos_hidden_dim", 32)),
            use_film_lag=bool(cfg.get("use_film_lag", True)),
            input_projector_name=_resolve_input_projector_name(cfg),
            input_projector_kwargs=_build_input_projector_kwargs(cfg),
            spatial_model_type=_resolve_spatial_type(cfg),
            spatial_layer_kwargs=_build_spatial_layer_kwargs(cfg),
            temporal_readout_type=_resolve_temporal_readout_type(cfg),
            temporal_readout_kwargs=_build_temporal_readout_kwargs(cfg),
            temporal_block_kwargs=_build_temporal_block_kwargs(cfg),
        )


ModelBuilderFn = Callable[[ModelBuildConfig], nn.Module]


class ModelVariantRegistry:
    def __init__(self):
        self._builders: Dict[str, ModelBuilderFn] = {}

    def register(self, names: List[str]):
        normalized = [str(n).strip().lower() for n in names if str(n).strip()]
        if len(normalized) == 0:
            raise ValueError("At least one non-empty model name must be provided.")

        def _decorator(fn: ModelBuilderFn):
            for n in normalized:
                self._builders[n] = fn
            return fn

        return _decorator

    def get(self, name: str) -> ModelBuilderFn:
        key = str(name).strip().lower()
        if key not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise ValueError(f"Unsupported model name: `{name}`. Available: {available}")
        return self._builders[key]

    def names(self) -> List[str]:
        return sorted(self._builders.keys())


MODEL_VARIANT_REGISTRY = ModelVariantRegistry()


def register_model_variant(*names: str):
    return MODEL_VARIANT_REGISTRY.register(list(names))


@register_model_variant("routing_baseline", "baseline", "gru_baseline")
def _build_routing_baseline(cfg: ModelBuildConfig) -> nn.Module:
    return RoutingBaselineModel(
        input_dim=cfg.input_dim,
        node_attr_dim=cfg.node_attr_dim,
        hidden_dim=cfg.hidden_dim,
        pred_len=cfg.pred_len,
        dropout=cfg.dropout,
    )


@register_model_variant("gr2n", "routing_gr2n", "graph_routing_recurrent_network")
def _build_gr2n(cfg: ModelBuildConfig) -> nn.Module:
    if cfg.num_nodes <= 0 or cfg.seq_len <= 0:
        raise ValueError(
            "For `gr2n`, `num_nodes` and `seq_len` must be positive. "
            "Provide them in model config or pass a non-empty dataset to build_model()."
        )
    return GR2N(
        seq_len=cfg.seq_len,
        input_freq_per_day=cfg.input_freq_per_day,
        num_nodes=cfg.num_nodes,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        pred_len=cfg.pred_len,
        num_layers=cfg.num_layers,
        edge_attr_dim=cfg.edge_attr_dim,
        node_attr_dim=cfg.node_attr_dim,
        pos_hidden_dim=cfg.pos_hidden_dim,
        dropout=cfg.dropout,
        use_film_lag=cfg.use_film_lag,
        input_projector_name=cfg.input_projector_name,
        input_projector_kwargs=cfg.input_projector_kwargs,
        spatial_model_type=cfg.spatial_model_type,
        spatial_layer_kwargs=cfg.spatial_layer_kwargs,
        temporal_readout_type=cfg.temporal_readout_type,
        temporal_readout_kwargs=cfg.temporal_readout_kwargs,
        temporal_block_kwargs=cfg.temporal_block_kwargs,
    )


@register_model_variant("gr2n_seq2seq", "routing_gr2n_seq2seq", "graph_gru_seq2seq")
def _build_gr2n_seq2seq(cfg: ModelBuildConfig) -> nn.Module:
    if cfg.num_nodes <= 0 or cfg.seq_len <= 0:
        raise ValueError(
            "For `gr2n_seq2seq`, `num_nodes` and `seq_len` must be positive. "
            "Provide them in model config or pass a non-empty dataset to build_model()."
        )
    return GR2NSeq2Seq(
        seq_len=cfg.seq_len,
        num_nodes=cfg.num_nodes,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        pred_len=cfg.pred_len,
        num_layers=cfg.num_layers,
        edge_attr_dim=cfg.edge_attr_dim,
        node_attr_dim=cfg.node_attr_dim,
        pos_hidden_dim=cfg.pos_hidden_dim,
        dropout=cfg.dropout,
        use_film_lag=cfg.use_film_lag,
        input_projector_name=cfg.input_projector_name,
        input_projector_kwargs=cfg.input_projector_kwargs,
        spatial_model_type=cfg.spatial_model_type,
        spatial_layer_kwargs=cfg.spatial_layer_kwargs,
        temporal_readout_type=cfg.temporal_readout_type,
        temporal_readout_kwargs=cfg.temporal_readout_kwargs,
        temporal_block_kwargs=cfg.temporal_block_kwargs,
    )


def available_model_variants() -> List[str]:
    return MODEL_VARIANT_REGISTRY.names()


def build_model(model_cfg: Optional[Dict[str, Any]] = None, dataset=None) -> nn.Module:
    cfg_raw = {} if model_cfg is None else dict(model_cfg)
    dataset_dims = _infer_dims_from_dataset(dataset)
    cfg = ModelBuildConfig.from_dict(cfg_raw, dataset_dims)
    builder = MODEL_VARIANT_REGISTRY.get(cfg.name)
    return builder(cfg)
