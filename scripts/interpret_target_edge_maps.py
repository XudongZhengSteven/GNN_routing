import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.collections import LineCollection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import build_dataloader, build_dataset
from models import build_model
from models.spatial_model import GraphAttentionPosEnc, GraphConvPosEnc
from trainers import select_device, set_seed


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def merge_model_cfg(model_cfg_file: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {}
    model_section = model_cfg_file.get("model", None)
    if isinstance(model_section, dict):
        cfg.update(model_section)

    train_model_section = train_cfg.get("model", None)
    if isinstance(train_model_section, dict):
        cfg.update(train_model_section)
    return cfg


def _move_to_device(batch: Any, device: torch.device):
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except TypeError:
            pass
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device) for v in batch)
    return batch


def _get_field(batch: Any, key: str):
    if isinstance(batch, dict):
        return batch[key]
    if hasattr(batch, key):
        return getattr(batch, key)
    raise KeyError(f"Batch has no field `{key}`")


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _safe_name(name: str):
    return (
        str(name)
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


class DynamicEdgeWeightRecorder:
    """
    Capture effective dynamic edge weights in GraphConvPosEnc modules:
      w_eff = clamp(w_base * softplus(4 * (sigmoid(edge_mlp(msg)) - 0.5)))
    """

    def __init__(self, model: torch.nn.Module, base_edge_count: int):
        self.model = model
        self.base_edge_count = int(base_edge_count)
        self.handles = []
        self.global_sum: Optional[torch.Tensor] = None
        self.global_count = 0

    def _reduce_to_base_edges(self, w: torch.Tensor) -> torch.Tensor:
        w = w.reshape(-1)
        if self.base_edge_count <= 0:
            return w
        if w.numel() % self.base_edge_count != 0:
            raise ValueError(
                f"Dynamic edge weight length {w.numel()} is not divisible by base_edge_count={self.base_edge_count}."
            )
        return w.view(-1, self.base_edge_count).mean(dim=0)

    def _make_hook(self):
        def _hook(module: GraphConvPosEnc, inputs):
            x, state, edge_index, edge_weight = inputs
            with torch.no_grad():
                x_and_state = torch.cat([x, state], dim=-1)
                x_proj = module.linear_in(x_and_state)
                src = edge_index[0].long()
                msg = x_proj[src]

                w = edge_weight.reshape(-1, 1)
                if module.edge_mlp is not None:
                    w_dynamic = torch.sigmoid(module.edge_mlp(msg))
                    w = w * torch.nn.functional.softplus(4.0 * (w_dynamic - 0.5))
                w = torch.clamp(w, min=0.0, max=module.edge_weight_max)
                w_mean = self._reduce_to_base_edges(w.squeeze(-1).detach()).to(torch.float32).cpu()

                if self.global_sum is None:
                    self.global_sum = torch.zeros_like(w_mean)
                self.global_sum += w_mean
                self.global_count += 1

        return _hook

    def attach(self):
        hook = self._make_hook()
        for _, module in self.model.named_modules():
            if isinstance(module, (GraphConvPosEnc, GraphAttentionPosEnc)):
                self.handles.append(module.register_forward_pre_hook(hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def mean_global(self) -> torch.Tensor:
        if self.global_sum is None or self.global_count <= 0:
            raise RuntimeError("No dynamic edge weights were recorded.")
        return self.global_sum / float(self.global_count)


def _extract_base_topology(model, dataset, device: torch.device):
    if not hasattr(model, "positional_encoding"):
        raise ValueError("Model has no `positional_encoding`; this script currently targets GR2N-like models.")

    sample = dataset[0]
    mask_downstream_adj = sample.mask_downstream_adj.unsqueeze(0).to(device)
    mask_khop_up_adj = sample.mask_khop_up_adj.unsqueeze(0).to(device)
    full_path_edge_attr_adj = sample.full_path_edge_attr_adj.unsqueeze(0).to(device)

    with torch.no_grad():
        edge_index, edge_weight = model.positional_encoding(
            mask_downstream_adj,
            mask_khop_up_adj,
            full_path_edge_attr_adj,
        )
    if edge_weight.dim() == 2:
        edge_weight = edge_weight.mean(dim=0)
    return edge_index.detach().cpu(), edge_weight.detach().to(torch.float32).cpu()


def _build_node_positions(dataset):
    pos = {}
    row_col = {}
    for idx in range(int(dataset.num_nodes)):
        node_name = dataset.id_node_map[int(idx)]
        row, col = dataset.node_pos_map[node_name]
        row_col[int(idx)] = (int(row), int(col))
        pos[int(idx)] = (float(col), -float(row))
    return pos, row_col


def _build_segments(edge_index: np.ndarray, node_pos: Dict[int, tuple]):
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    segs = []
    valid = []
    for i, (s, d) in enumerate(zip(src, dst)):
        if s not in node_pos or d not in node_pos:
            continue
        segs.append([node_pos[s], node_pos[d]])
        valid.append(i)
    return segs, np.asarray(valid, dtype=int)


def _plot_target_edge_map(
    edge_index_np: np.ndarray,
    edge_weight_np: np.ndarray,
    node_pos: Dict[int, tuple],
    target_idx: int,
    target_name: str,
    upstream_nodes: set,
    outlet_indices: list,
    save_path: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    src = edge_index_np[0].astype(int)
    dst = edge_index_np[1].astype(int)
    mask_target = np.asarray([(int(s) in upstream_nodes and int(d) in upstream_nodes) for s, d in zip(src, dst)], dtype=bool)

    segments_all, valid_all = _build_segments(edge_index_np, node_pos)
    if len(valid_all) == 0:
        raise RuntimeError("No valid drawable segments.")

    fig, ax = plt.subplots(figsize=(8.0, 6.8), dpi=240)

    # Background river network
    lc_bg = LineCollection([segments_all[i] for i in range(len(segments_all))], colors="#bdbdbd", linewidths=0.6, alpha=0.35)
    ax.add_collection(lc_bg)

    target_valid_mask = mask_target[valid_all]
    idx_sel = np.where(target_valid_mask)[0]
    if idx_sel.size > 0:
        segments_sel = [segments_all[i] for i in idx_sel]
        values_sel = edge_weight_np[valid_all[idx_sel]]
        lc_sel = LineCollection(segments_sel, cmap="viridis", linewidths=1.25, alpha=0.95)
        lc_sel.set_array(values_sel.astype(float))
        ax.add_collection(lc_sel)
        cbar = fig.colorbar(lc_sel, ax=ax, fraction=0.04, pad=0.01)
        cbar.set_label("dynamic edge weight")

    node_xy = np.asarray([node_pos[k] for k in sorted(node_pos.keys())], dtype=float)
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=4, c="#111111", alpha=0.3, zorder=2)

    # Outlets
    out_pts = [node_pos[int(i)] for i in outlet_indices if int(i) in node_pos]
    if len(out_pts) > 0:
        out_arr = np.asarray(out_pts, dtype=float)
        ax.scatter(out_arr[:, 0], out_arr[:, 1], s=16, c="#ff7f0e", alpha=0.9, zorder=3, label="outlets")

    # Target outlet highlight
    if int(target_idx) in node_pos:
        tx, ty = node_pos[int(target_idx)]
        ax.scatter([tx], [ty], s=40, c="#d62728", marker="*", zorder=4, label=f"target: {target_name}")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Target Edge Map - {target_name}")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return save_path, mask_target


def _plot_2d_heatmap(
    heat_arr: np.ndarray,
    outlet_indices: list,
    outlet_names: list,
    row_col_map: Dict[int, tuple],
    title: str,
    save_path: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    arr = heat_arr.copy()
    arr_plot = np.where(np.isfinite(arr), arr, np.nan)

    fig, ax = plt.subplots(figsize=(8.0, 6.8), dpi=240)
    im = ax.imshow(arr_plot, cmap="hot", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label("aggregated target-relevant edge intensity")

    for idx, name in zip(outlet_indices, outlet_names):
        idx = int(idx)
        if idx not in row_col_map:
            continue
        r, c = row_col_map[idx]
        ax.scatter([c], [r], s=28, c="cyan", edgecolors="black", linewidths=0.5)
        ax.text(c + 0.8, r + 0.8, str(name), fontsize=7, color="white")

    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot target-wise spatial edge-weight maps and aggregate 2D heatmap for GR2N."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (best.ckpt/last.ckpt).")
    parser.add_argument(
        "--split",
        default="verify",
        choices=["train", "val", "test", "warmup", "calibrate", "verify", "full"],
        help="Dataset split for interpretation.",
    )
    parser.add_argument("--train-cfg", default="configs/train.yaml", help="Training config YAML.")
    parser.add_argument("--data-cfg", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--model-cfg", default="configs/model.yaml", help="Model config YAML.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0/cpu/auto.")
    parser.add_argument("--batch-size", type=int, default=1, help="Interpretation batch size (recommend 1).")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples used to aggregate dynamic weights.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Default: <ckpt_dir>/target_edge_interpret/<split>")
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg = load_yaml(args.train_cfg)
    model_cfg_file = load_yaml(args.model_cfg)
    set_seed(int(train_cfg.get("seed", 42)))

    dataset_cfg = dict(train_cfg.get("dataset", {}))
    dataset_cfg.setdefault("data_cfg_path", args.data_cfg)
    dataset_cfg.setdefault("model_cfg_path", args.model_cfg)

    train_ds = build_dataset("train", dataset_kwargs=dataset_cfg)
    eval_kwargs = dict(dataset_cfg)
    eval_kwargs["normalizers"] = train_ds.normalizers
    eval_ds = build_dataset(args.split, dataset_kwargs=eval_kwargs)

    loader_cfg = dict(train_cfg.get("dataloader", {}).get("val", {}))
    loader_cfg["shuffle"] = False
    loader_cfg["use_balance_sampler"] = False
    loader_cfg["batch_size"] = int(args.batch_size)
    eval_loader = build_dataloader(eval_ds, **loader_cfg)

    model_cfg = merge_model_cfg(model_cfg_file=model_cfg_file, train_cfg=train_cfg)
    if "pred_len" not in model_cfg and "n_pred" in dataset_cfg:
        model_cfg["pred_len"] = dataset_cfg["n_pred"]
    model = build_model(model_cfg=model_cfg, dataset=train_ds)

    device = select_device(args.device if args.device is not None else train_cfg.get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    edge_index_static, edge_weight_static = _extract_base_topology(model, eval_ds, device)
    base_edge_count = int(edge_index_static.shape[1])

    recorder = DynamicEdgeWeightRecorder(model=model, base_edge_count=base_edge_count)
    recorder.attach()
    used_samples = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = _move_to_device(batch, device)
            _ = model(batch)

            x = _get_field(batch, "x")
            bs = int(x.shape[0]) if x.dim() >= 4 else 1
            used_samples += bs
            if used_samples >= int(args.num_samples):
                break
    recorder.detach()

    edge_weight_dynamic = recorder.mean_global()

    edge_index_np = edge_index_static.numpy().astype(int)
    edge_weight_np = edge_weight_dynamic.numpy().astype(float).reshape(-1)

    node_pos, row_col_map = _build_node_positions(eval_ds)
    outlet_indices = [int(x) for x in getattr(eval_ds, "outlet_indices", [])]
    outlet_names = list(getattr(eval_ds, "outlet_names", [f"outlet_{i}" for i in range(len(outlet_indices))]))

    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(eval_ds.num_nodes)))
    for i in range(base_edge_count):
        u = int(edge_index_np[0, i])
        v = int(edge_index_np[1, i])
        graph.add_edge(u, v, weight=float(edge_weight_np[i]))

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(ckpt_dir, "target_edge_interpret", str(args.split))
    os.makedirs(save_dir, exist_ok=True)

    per_target_rows = []
    per_target_plots = {}

    node_score_total = np.zeros(int(eval_ds.num_nodes), dtype=float)
    node_score_by_target = {}

    for outlet_idx, outlet_name in zip(outlet_indices, outlet_names):
        upstream_nodes = set(nx.ancestors(graph, int(outlet_idx)))
        upstream_nodes.add(int(outlet_idx))

        target_plot_path = os.path.join(save_dir, f"target_edge_map_{_safe_name(outlet_name)}.png")
        plot_ret = _plot_target_edge_map(
            edge_index_np=edge_index_np,
            edge_weight_np=edge_weight_np,
            node_pos=node_pos,
            target_idx=int(outlet_idx),
            target_name=str(outlet_name),
            upstream_nodes=upstream_nodes,
            outlet_indices=outlet_indices,
            save_path=target_plot_path,
        )
        if plot_ret is None:
            mask_target = np.zeros(base_edge_count, dtype=bool)
            plot_file = None
        else:
            plot_file, mask_target = plot_ret
        per_target_plots[str(outlet_name)] = plot_file

        w_target = np.where(mask_target, edge_weight_np, 0.0)
        selected_edge_count = int(mask_target.sum())

        node_score = np.zeros(int(eval_ds.num_nodes), dtype=float)
        src = edge_index_np[0]
        dst = edge_index_np[1]
        for i in np.where(mask_target)[0]:
            w = float(w_target[i])
            u = int(src[i])
            v = int(dst[i])
            node_score[u] += 0.5 * w
            node_score[v] += 0.5 * w
        node_score_by_target[str(outlet_name)] = node_score
        node_score_total += node_score

        per_target_rows.append(
            {
                "target_name": str(outlet_name),
                "target_node_idx": int(outlet_idx),
                "upstream_node_count": int(len(upstream_nodes)),
                "selected_edge_count": selected_edge_count,
                "selected_edge_weight_sum": float(w_target.sum()),
                "selected_edge_weight_mean": float(w_target[mask_target].mean()) if selected_edge_count > 0 else 0.0,
            }
        )

    # Build 2D aggregated heatmap over grid coordinates.
    max_row = max(r for r, _ in row_col_map.values()) if len(row_col_map) > 0 else 0
    max_col = max(c for _, c in row_col_map.values()) if len(row_col_map) > 0 else 0
    heat = np.full((max_row + 1, max_col + 1), np.nan, dtype=float)
    for idx in range(int(eval_ds.num_nodes)):
        if idx not in row_col_map:
            continue
        r, c = row_col_map[idx]
        heat[r, c] = node_score_total[idx]

    heatmap_path = _plot_2d_heatmap(
        heat_arr=heat,
        outlet_indices=outlet_indices,
        outlet_names=outlet_names,
        row_col_map=row_col_map,
        title=f"Aggregated 2D Heatmap ({args.split})",
        save_path=os.path.join(save_dir, "target_edge_aggregate_heatmap_2d.png"),
    )

    # Save tables.
    target_summary_csv = os.path.join(save_dir, "target_edge_summary.csv")
    pd.DataFrame(per_target_rows).to_csv(target_summary_csv, index=False)

    node_table = {
        "node_idx": np.arange(int(eval_ds.num_nodes), dtype=int),
    }
    for idx in range(int(eval_ds.num_nodes)):
        node_name = eval_ds.id_node_map[int(idx)]
        row, col = eval_ds.node_pos_map[node_name]
        if idx == 0:
            node_table["node_name"] = []
            node_table["row"] = []
            node_table["col"] = []
        node_table["node_name"].append(node_name)
        node_table["row"].append(int(row))
        node_table["col"].append(int(col))

    node_table["score_aggregate"] = node_score_total
    for name in outlet_names:
        node_table[f"score_{_safe_name(name)}"] = node_score_by_target[str(name)]

    node_score_csv = os.path.join(save_dir, "target_edge_node_scores.csv")
    pd.DataFrame(node_table).to_csv(node_score_csv, index=False)

    # Edge-level export with dynamic/static.
    src_idx = edge_index_np[0].astype(int)
    dst_idx = edge_index_np[1].astype(int)
    edge_df = pd.DataFrame(
        {
            "edge_id": np.arange(base_edge_count, dtype=int),
            "src_idx": src_idx,
            "dst_idx": dst_idx,
            "src_node": [eval_ds.id_node_map[int(i)] for i in src_idx],
            "dst_node": [eval_ds.id_node_map[int(i)] for i in dst_idx],
            "weight_static": edge_weight_static.numpy().astype(float).reshape(-1),
            "weight_dynamic_mean": edge_weight_np,
            "weight_dynamic_minus_static": edge_weight_np - edge_weight_static.numpy().astype(float).reshape(-1),
        }
    )
    edge_csv = os.path.join(save_dir, "target_edge_weights.csv")
    edge_df.to_csv(edge_csv, index=False)

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "num_samples_used": int(used_samples),
        "model_name": model_cfg.get("name", None),
        "graph_num_nodes": int(eval_ds.num_nodes),
        "graph_num_edges": int(base_edge_count),
        "outlet_names": outlet_names,
        "outlet_indices": outlet_indices,
        "plots": {
            "per_target": per_target_plots,
            "aggregate_heatmap_2d": heatmap_path,
        },
        "tables": {
            "target_summary_csv": target_summary_csv,
            "node_score_csv": node_score_csv,
            "edge_weight_csv": edge_csv,
        },
    }
    summary_json = os.path.join(save_dir, "target_edge_interpretation.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    payload["summary_json"] = summary_json

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
