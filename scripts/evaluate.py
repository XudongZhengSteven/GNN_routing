import argparse
import json
import math
import os
import sys
from typing import Any, Dict

import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import build_dataloader, build_dataset
from models import build_model
from trainers import Trainer, build_loss, compute_kge_summary, set_seed


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate routing model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test", "warmup", "calibrate", "verify", "full"],
        help="Evaluation split.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated splits, e.g. calibrate,verify. If set, it overrides --split.",
    )
    parser.add_argument("--train-cfg", default="configs/train.yaml", help="Training config YAML.")
    parser.add_argument("--data-cfg", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--model-cfg", default="configs/model.yaml", help="Model graph config YAML.")
    parser.add_argument("--device", default=None, help="Device override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch size override.")
    parser.add_argument("--save-dir", default=None, help="Directory to save metrics, simulation csv, and plots.")
    parser.add_argument("--plot", action="store_true", help="Whether to save plots.")
    parser.add_argument("--max-plot-points", type=int, default=1200, help="Max points for timeseries plots per outlet.")
    return parser.parse_args()


def merge_model_cfg(model_cfg_file: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {}
    model_section = model_cfg_file.get("model", None)
    if isinstance(model_section, dict):
        cfg.update(model_section)

    # Backward compatibility: allow overriding in train.yaml
    train_model_section = train_cfg.get("model", None)
    if isinstance(train_model_section, dict):
        cfg.update(train_model_section)
    return cfg


def merge_dataset_cfg(model_cfg_file: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge dataset config with optional overrides from model.yaml.
    Priority: model.yaml:dataset > train.yaml:dataset
    """
    cfg = {}
    train_dataset_section = train_cfg.get("dataset", None)
    if isinstance(train_dataset_section, dict):
        cfg.update(train_dataset_section)

    model_dataset_section = model_cfg_file.get("dataset", None)
    if isinstance(model_dataset_section, dict):
        cfg.update(model_dataset_section)

    return cfg


def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    err = pred - target
    mse = torch.mean(err * err)
    mae = torch.mean(torch.abs(err))
    rmse = torch.sqrt(mse + 1e-12)
    denom = torch.sum((target - torch.mean(target)) ** 2)
    nse = 1.0 - torch.sum(err ** 2) / (denom + 1e-12)
    return {
        "loss": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "nse": float(nse.detach().cpu()),
    }


def _metrics_with_kge(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    out = compute_regression_metrics(pred, target)
    out.update(compute_kge_summary(pred, target))
    return out


def _get_prediction_times(eval_ds, n_samples: int, pred_len: int):
    obs_times = pd.to_datetime(eval_ds.obs_time_index[eval_ds._time_slice_obs])
    times = []
    for i in range(n_samples):
        start = i + eval_ds._windowsize_in_days - eval_ds.predict_current
        for h in range(pred_len):
            idx = start + h
            if 0 <= idx < len(obs_times):
                times.append(obs_times[idx])
            else:
                times.append(pd.NaT)
    return pd.to_datetime(times)


def _build_per_outlet_metrics(pred: torch.Tensor, target: torch.Tensor, outlet_names):
    pred_2d = pred.reshape(-1, pred.shape[-1])
    target_2d = target.reshape(-1, target.shape[-1])
    rows = []
    for i in range(pred_2d.shape[-1]):
        row = {"outlet": outlet_names[i] if outlet_names is not None and i < len(outlet_names) else f"outlet_{i}"}
        row.update(_metrics_with_kge(pred_2d[:, i], target_2d[:, i]))
        rows.append(row)
    return rows


def _save_simulation_csv(save_dir: str, split: str, eval_ds, pred_denorm: torch.Tensor, target_denorm: torch.Tensor):
    os.makedirs(save_dir, exist_ok=True)
    pred_2d = pred_denorm.reshape(-1, pred_denorm.shape[-1]).detach().cpu().numpy()
    target_2d = target_denorm.reshape(-1, target_denorm.shape[-1]).detach().cpu().numpy()

    times = _get_prediction_times(eval_ds, n_samples=pred_denorm.shape[0], pred_len=pred_denorm.shape[1])
    df = pd.DataFrame({"time": times})
    names = getattr(eval_ds, "outlet_names", [f"outlet_{i}" for i in range(pred_2d.shape[-1])])
    for i, name in enumerate(names):
        safe_name = str(name).replace(" ", "_")
        df[f"obs_{safe_name}"] = target_2d[:, i]
        df[f"pred_{safe_name}"] = pred_2d[:, i]
        df[f"err_{safe_name}"] = pred_2d[:, i] - target_2d[:, i]

    csv_path = os.path.join(save_dir, f"simulation_{split}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_kge_bar(save_dir: str, split: str, per_outlet_denorm):
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib is unavailable; skip plotting.")
        return None

    names = [x["outlet"] for x in per_outlet_denorm]
    values = [x["kge"] for x in per_outlet_denorm]

    fig_w = max(8.0, 1.2 * len(names))
    plt.figure(figsize=(fig_w, 4.5))
    plt.bar(names, values)
    plt.axhline(y=0.0, color="k", linewidth=1.0, linestyle="--")
    plt.ylim(min(-1.0, min(values) - 0.1), max(1.0, max(values) + 0.1))
    plt.title(f"KGE by Outlet ({split})")
    plt.ylabel("KGE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, f"kge_bar_{split}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_timeseries(save_dir: str, split: str, eval_ds, pred_denorm: torch.Tensor, target_denorm: torch.Tensor, max_points: int):
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib is unavailable; skip plotting.")
        return []

    names = getattr(eval_ds, "outlet_names", [f"outlet_{i}" for i in range(pred_denorm.shape[-1])])
    times = _get_prediction_times(eval_ds, n_samples=pred_denorm.shape[0], pred_len=pred_denorm.shape[1])
    pred_2d = pred_denorm.reshape(-1, pred_denorm.shape[-1]).detach().cpu().numpy()
    target_2d = target_denorm.reshape(-1, target_denorm.shape[-1]).detach().cpu().numpy()

    paths = []
    n_rows = pred_2d.shape[0]
    step = max(1, int(math.ceil(n_rows / max(1, int(max_points)))))
    sl = slice(0, None, step)

    # Build per-outlet KGE for legend.
    kge_by_outlet = {}
    for i, name in enumerate(names):
        kge = compute_kge_summary(torch.tensor(pred_2d[:, i]), torch.tensor(target_2d[:, i]))["kge"]
        kge_by_outlet[str(name)] = kge

    # Combined panel: N rows x 1 col. For five outlets it is exactly 5x1.
    n_outlets = len(names)
    fig_h = max(10.0, 2.6 * n_outlets)
    fig, axes = plt.subplots(n_outlets, 1, figsize=(10.0, fig_h), squeeze=False)
    axes = axes[:, 0]

    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(times[sl], target_2d[sl, i], label="obs", linewidth=1.3)
        kge_text = f"{kge_by_outlet[str(name)]:.3f}" if math.isfinite(kge_by_outlet[str(name)]) else "nan"
        ax.plot(times[sl], pred_2d[sl, i], label=f"pred (KGE={kge_text})", linewidth=1.2)
        ax.set_title(str(name))
        ax.set_xlabel("time")
        if i == 0:
            ax.set_ylabel("streamflow (denorm)")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{split} simulation by outlet", y=1.02)
    fig.tight_layout()
    panel_path = os.path.join(save_dir, f"timeseries_panel_{split}.png")
    fig.savefig(panel_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(panel_path)

    return paths


def _evaluate_one_split(
    split: str,
    trainer: Trainer,
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    train_ds,
    batch_size_override: int = None,
    save_dir: str = None,
    do_plot: bool = False,
    max_plot_points: int = 1200,
):
    eval_kwargs = dict(dataset_cfg)
    eval_kwargs["normalizers"] = train_ds.normalizers
    eval_ds = build_dataset(split, dataset_kwargs=eval_kwargs)

    eval_loader_cfg = dict(train_cfg.get("dataloader", {}).get("val", {}))
    eval_loader_cfg["shuffle"] = False
    eval_loader_cfg["use_balance_sampler"] = False
    if batch_size_override is not None:
        eval_loader_cfg["batch_size"] = int(batch_size_override)
    eval_loader = build_dataloader(eval_ds, **eval_loader_cfg)

    metrics_norm_from_trainer = trainer.evaluate(eval_loader, split=split)
    pred_norm, target_norm = trainer.predict(eval_loader, return_target=True)
    pred_denorm = eval_ds.inverse_transform_streamflow_tensor(pred_norm)
    target_denorm = eval_ds.inverse_transform_streamflow_tensor(target_norm)

    # Overall metrics
    metrics_norm = dict(metrics_norm_from_trainer)
    metrics_norm.update(compute_kge_summary(pred_norm, target_norm))
    metrics_denorm = _metrics_with_kge(pred_denorm, target_denorm)

    outlet_names = getattr(eval_ds, "outlet_names", None)
    per_outlet_norm = _build_per_outlet_metrics(pred_norm, target_norm, outlet_names)
    per_outlet_denorm = _build_per_outlet_metrics(pred_denorm, target_denorm, outlet_names)

    result = {
        "split": split,
        "metrics_norm": metrics_norm,
        "metrics_denorm": metrics_denorm,
        "per_outlet_norm": per_outlet_norm,
        "per_outlet_denorm": per_outlet_denorm,
        "pred_shape": list(pred_norm.shape),
        "target_shape": list(target_norm.shape),
        "outlet_names": outlet_names,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = _save_simulation_csv(save_dir, split, eval_ds, pred_denorm, target_denorm)
        result["simulation_csv"] = csv_path

        metrics_path = os.path.join(save_dir, f"metrics_{split}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result["metrics_json"] = metrics_path

        if do_plot:
            kge_plot = _plot_kge_bar(save_dir, split, per_outlet_denorm)
            ts_plots = _plot_timeseries(
                save_dir=save_dir,
                split=split,
                eval_ds=eval_ds,
                pred_denorm=pred_denorm,
                target_denorm=target_denorm,
                max_points=max_plot_points,
            )
            result["kge_plot"] = kge_plot
            result["timeseries_plots"] = ts_plots

    return result


def main():
    args = parse_args()
    train_cfg = load_yaml(args.train_cfg)
    model_cfg_file = load_yaml(args.model_cfg)
    seed = int(train_cfg.get("seed", 42))
    reproducibility_cfg = dict(train_cfg.get("reproducibility", {}))
    deterministic = bool(reproducibility_cfg.get("deterministic", train_cfg.get("deterministic", False)))
    cudnn_benchmark = reproducibility_cfg.get("cudnn_benchmark", None)
    set_seed(seed, deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)

    dataset_cfg = merge_dataset_cfg(model_cfg_file=model_cfg_file, train_cfg=train_cfg)
    dataset_cfg.setdefault("data_cfg_path", args.data_cfg)
    dataset_cfg.setdefault("model_cfg_path", args.model_cfg)

    train_ds = build_dataset("train", dataset_kwargs=dataset_cfg)

    model_cfg = merge_model_cfg(model_cfg_file=model_cfg_file, train_cfg=train_cfg)
    if "pred_len" not in model_cfg and "n_pred" in dataset_cfg:
        model_cfg["pred_len"] = dataset_cfg["n_pred"]
    model = build_model(model_cfg=model_cfg, dataset=train_ds)
    criterion = build_loss(train_cfg.get("loss", {}), dataset=train_ds)

    trainer_cfg = dict(train_cfg.get("trainer", {}))
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=None,
        scheduler=None,
        device=args.device if args.device is not None else train_cfg.get("device", "auto"),
        checkpoint_dir=str(trainer_cfg.get("checkpoint_dir", "checkpoints/default")),
        monitor=str(trainer_cfg.get("monitor", "val_loss")),
        monitor_mode=str(trainer_cfg.get("monitor_mode", "min")),
    )

    trainer.load_checkpoint(args.checkpoint, strict=True)

    if args.splits:
        splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    else:
        splits = [args.split]

    results = {}
    for split in splits:
        split_save_dir = None
        if args.save_dir is not None:
            split_save_dir = os.path.join(args.save_dir, split) if len(splits) > 1 else args.save_dir
        res = _evaluate_one_split(
            split=split,
            trainer=trainer,
            train_cfg=train_cfg,
            dataset_cfg=dataset_cfg,
            train_ds=train_ds,
            batch_size_override=args.batch_size,
            save_dir=split_save_dir,
            do_plot=bool(args.plot),
            max_plot_points=int(args.max_plot_points),
        )
        results[split] = res

    payload = results[splits[0]] if len(splits) == 1 else {"splits": results}
    if args.save_dir is not None and len(splits) > 1:
        os.makedirs(args.save_dir, exist_ok=True)
        all_path = os.path.join(args.save_dir, "metrics_all_splits.json")
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        payload["metrics_all_splits_json"] = all_path

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
