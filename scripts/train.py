import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import build_train_val_dataloaders
from models import build_model
from trainers import (
    Trainer,
    build_loss,
    build_optimizer,
    build_scheduler,
    compute_mean_kge,
    set_seed,
)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Train routing model.")
    parser.add_argument("--train-cfg", default="configs/train.yaml", help="Path to training config YAML.")
    parser.add_argument("--data-cfg", default="configs/data.yaml", help="Path to data split config YAML.")
    parser.add_argument("--model-cfg", default="configs/model.yaml", help="Path to model graph config YAML.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0/cpu/auto.")
    parser.add_argument("--resume", default=None, help="Checkpoint path for resume.")
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


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_objective_vs_epoch(history, save_path: str):
    """
    Plot objective (loss) versus epoch from trainer history.
    """
    if history is None or len(history) == 0:
        return None

    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib is unavailable; skip objective curve plotting.")
        return None

    epochs = [int(row["epoch"]) for row in history if "epoch" in row]
    train_loss = [float(row.get("train_loss", float("nan"))) for row in history]
    val_loss = [float(row.get("val_loss", float("nan"))) for row in history]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(epochs, train_loss, marker="o", linewidth=1.5, label="train_loss")
    if any(torch.isfinite(torch.tensor(val_loss)).tolist()):
        ax.plot(epochs, val_loss, marker="s", linewidth=1.5, label="val_loss")

    ax.set_xlabel("iteration (epoch)")
    ax.set_ylabel("objective (loss)")
    ax.set_title("Objective vs Iteration")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return save_path


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

    dataloader_cfg = train_cfg.get("dataloader", {})
    train_loader_cfg = dict(dataloader_cfg.get("train", {}))
    val_loader_cfg = dict(dataloader_cfg.get("val", {}))

    train_ds, val_ds, train_loader, val_loader = build_train_val_dataloaders(
        dataset_kwargs=dataset_cfg,
        train_loader_kwargs=train_loader_cfg,
        val_loader_kwargs=val_loader_cfg,
    )

    print("[Dataset] train runtime schema:")
    train_ds.print_runtime_schema()

    model_cfg = merge_model_cfg(model_cfg_file=model_cfg_file, train_cfg=train_cfg)
    if "pred_len" not in model_cfg and "n_pred" in dataset_cfg:
        model_cfg["pred_len"] = dataset_cfg["n_pred"]
    model = build_model(model_cfg=model_cfg, dataset=train_ds)

    loss_cfg = dict(train_cfg.get("loss", {}))
    criterion = build_loss(loss_cfg, dataset=train_ds)

    optimizer_cfg = dict(train_cfg.get("optimizer", {}))
    optimizer = build_optimizer(model.parameters(), optimizer_cfg)

    trainer_cfg = dict(train_cfg.get("trainer", {}))
    epochs = int(trainer_cfg.get("epochs", 20))
    scheduler_cfg = dict(train_cfg.get("scheduler", {}))
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=scheduler_cfg,
        total_epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    def extra_eval_denorm_metrics(tr: Trainer, loader, split: str):
        if split != "val":
            return {}
        pred_norm, target_norm = tr.predict(loader, return_target=True)
        pred_denorm = val_ds.inverse_transform_streamflow_tensor(pred_norm)
        target_denorm = val_ds.inverse_transform_streamflow_tensor(target_norm)
        return {"kge_denorm": float(compute_mean_kge(pred_denorm, target_denorm).detach().cpu())}

    checkpoint_dir_cfg = trainer_cfg.get("checkpoint_dir", None)
    if checkpoint_dir_cfg in {None, ""}:
        model_name = str(model_cfg.get("name", "model")).strip().lower()
        checkpoint_dir_cfg = os.path.join("checkpoints", model_name)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device if args.device is not None else train_cfg.get("device", "auto"),
        use_amp=bool(trainer_cfg.get("use_amp", False)),
        amp_dtype=str(trainer_cfg.get("amp_dtype", "float16")),
        grad_clip_norm=trainer_cfg.get("grad_clip_norm", None),
        log_interval=int(trainer_cfg.get("log_interval", 50)),
        checkpoint_dir=str(checkpoint_dir_cfg),
        monitor=str(trainer_cfg.get("monitor", "val_loss")),
        monitor_mode=str(trainer_cfg.get("monitor_mode", "min")),
        early_stopping_patience=trainer_cfg.get("early_stopping_patience", None),
        keep_last_k=int(trainer_cfg.get("keep_last_k", 3)),
        extra_eval_fn=extra_eval_denorm_metrics,
    )

    os.makedirs(trainer.checkpoint_dir, exist_ok=True)
    resolved_cfg = {
        "train_cfg_path": os.path.abspath(args.train_cfg),
        "data_cfg_path": os.path.abspath(args.data_cfg),
        "model_cfg_path": os.path.abspath(args.model_cfg),
        "seed": seed,
        "dataset": dataset_cfg,
        "dataloader": {
            "train": train_loader_cfg,
            "val": val_loader_cfg,
        },
        "model": model_cfg,
        "model_from_model_yaml": model_cfg_file.get("model", {}),
        "model_from_train_yaml": train_cfg.get("model", {}),
        "loss": loss_cfg,
        "optimizer": optimizer_cfg,
        "scheduler": scheduler_cfg,
        "trainer": trainer_cfg,
    }
    with open(os.path.join(trainer.checkpoint_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, ensure_ascii=False)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        resume_path=args.resume if args.resume is not None else trainer_cfg.get("resume_path", None),
    )

    print(f"Training finished. epochs_run={len(history)}")
    print(f"best_epoch={trainer.best_epoch}, best_metric={trainer.best_metric:.6f}")
    print(f"best_checkpoint={trainer.get_best_checkpoint_path()}")
    print(f"last_checkpoint={trainer.get_last_checkpoint_path()}")
    print(f"history_csv={trainer.history_csv_path}")

    # Save objective-vs-epoch figure for every training run by default.
    plot_train_curve = bool(trainer_cfg.get("plot_train_curve", True))
    if plot_train_curve:
        plot_file = str(trainer_cfg.get("plot_train_curve_file", "objective_vs_epoch.png"))
        plot_path = os.path.join(trainer.checkpoint_dir, plot_file)
        saved = plot_objective_vs_epoch(history=history, save_path=plot_path)
        if saved is not None:
            print(f"objective_curve={saved}")

    # Optional: run full-period simulation/plotting after training.
    post_train_simulate = bool(trainer_cfg.get("post_train_simulate", True))
    if post_train_simulate:
        splits = trainer_cfg.get("post_train_splits", ["calibrate", "verify"])
        if isinstance(splits, str):
            splits = [x.strip() for x in splits.split(",") if x.strip()]
        if not isinstance(splits, list) or len(splits) == 0:
            splits = ["calibrate", "verify"]

        post_save_root = trainer_cfg.get("post_train_save_dir", "post_train_simulation")
        post_save_root = os.path.join(trainer.checkpoint_dir, post_save_root)
        os.makedirs(post_save_root, exist_ok=True)

        device_for_eval = args.device if args.device is not None else str(train_cfg.get("device", "auto"))
        post_batch_size = trainer_cfg.get("post_train_batch_size", None)
        post_plot = bool(trainer_cfg.get("post_train_plot", True))

        print(f"Post-train simulation enabled. splits={splits}")
        for split in splits:
            split_save_dir = os.path.join(post_save_root, split)
            cmd = [
                sys.executable,
                os.path.join("scripts", "evaluate.py"),
                "--checkpoint",
                trainer.get_best_checkpoint_path(),
                "--split",
                str(split),
                "--train-cfg",
                args.train_cfg,
                "--data-cfg",
                args.data_cfg,
                "--model-cfg",
                args.model_cfg,
                "--device",
                str(device_for_eval),
                "--save-dir",
                split_save_dir,
            ]
            if post_batch_size is not None:
                cmd.extend(["--batch-size", str(int(post_batch_size))])
            if post_plot:
                cmd.append("--plot")

            print("RUN:", " ".join(cmd))
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

