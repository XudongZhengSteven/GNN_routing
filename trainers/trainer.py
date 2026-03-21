import csv
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from .metrics import compute_mean_kge


def set_seed(
    seed: int = 42,
    deterministic: bool = False,
    cudnn_benchmark: Optional[bool] = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = bool(deterministic)
    if cudnn_benchmark is None:
        cudnn_benchmark = not deterministic

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(deterministic)


def select_device(device: str = "auto") -> torch.device:
    text = str(device).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _normalize_station_name(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[\s_\-]+", "", value)
    return value


def _resolve_station_index(
    loss_cfg: Dict[str, Any],
    dataset: Any = None,
) -> Tuple[int, Optional[str]]:
    if "station_index" in loss_cfg and loss_cfg.get("station_index", None) is not None:
        station_index = int(loss_cfg["station_index"])
        station_name = None
        if dataset is not None and hasattr(dataset, "outlet_names"):
            names = list(getattr(dataset, "outlet_names", []))
            if len(names) > 0 and not (0 <= station_index < len(names)):
                raise ValueError(
                    f"`station_index` out of range: index={station_index}, available=[0, {len(names)-1}]"
                )
            if 0 <= station_index < len(names):
                station_name = str(names[station_index])
        return station_index, station_name

    station_name = loss_cfg.get("station_name", None)
    if station_name in {None, ""}:
        raise ValueError(
            "`single_station_kge` requires either `station_index` or `station_name` in loss config."
        )
    if dataset is None or not hasattr(dataset, "outlet_names"):
        raise ValueError(
            "`single_station_kge` with `station_name` requires a dataset that provides `outlet_names`."
        )

    names = [str(x) for x in list(getattr(dataset, "outlet_names", []))]
    if len(names) == 0:
        raise ValueError("Dataset has empty `outlet_names`; cannot resolve `station_name`.")

    query = _normalize_station_name(str(station_name))
    normalized = [_normalize_station_name(x) for x in names]

    if query in normalized:
        idx = normalized.index(query)
        return int(idx), names[idx]

    raise ValueError(
        f"`station_name` `{station_name}` is not found in outlet_names={names}."
    )


def build_loss(loss_cfg: Optional[Dict[str, Any]] = None, dataset: Any = None) -> nn.Module:
    cfg = {} if loss_cfg is None else dict(loss_cfg)
    name = str(cfg.get("name", "mse")).lower()

    if name in {"mse", "mse_loss", "l2"}:
        return nn.MSELoss()
    if name in {"mae", "l1", "l1_loss"}:
        return nn.L1Loss()
    if name in {"huber", "smooth_l1"}:
        beta = float(cfg.get("beta", cfg.get("delta", 1.0)))
        return nn.SmoothL1Loss(beta=beta)
    if name in {"mse_kge", "kge_mse", "hybrid_kge"}:
        return MSEKGELoss(
            lambda_mse=float(cfg.get("lambda_mse", 1.0)),
            lambda_kge=float(cfg.get("lambda_kge", 0.2)),
            eps=float(cfg.get("eps", 1e-6)),
        )
    if name in {"mean_kge", "kge_mean", "station_mean_kge"}:
        return MeanKGELoss(eps=float(cfg.get("eps", 1e-6)))
    if name in {"single_station_kge", "station_kge", "single_kge", "target_station_kge"}:
        station_index, station_name = _resolve_station_index(cfg, dataset=dataset)
        return SingleStationKGELoss(
            station_index=station_index,
            station_name=station_name,
            eps=float(cfg.get("eps", 1e-6)),
        )

    raise ValueError(f"Unsupported loss type: `{name}`")


class MSEKGELoss(nn.Module):
    def __init__(self, lambda_mse: float = 1.0, lambda_kge: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_kge = float(lambda_kge)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = torch.mean((pred - target) ** 2)
        kge = compute_mean_kge(pred, target, eps=self.eps)
        return self.lambda_mse * mse + self.lambda_kge * (1.0 - kge)

class MeanKGELoss(nn.Module):
    """
    Loss = 1 - mean(KGE_i), where i indexes outlet stations.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mean_kge = compute_mean_kge(pred, target, eps=self.eps)
        return 1.0 - mean_kge


class SingleStationKGELoss(nn.Module):
    """
    Loss = 1 - KGE(target_station)
    """

    def __init__(self, station_index: int, station_name: Optional[str] = None, eps: float = 1e-6):
        super().__init__()
        self.station_index = int(station_index)
        self.station_name = None if station_name is None else str(station_name)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_2d = pred.reshape(-1, pred.shape[-1]).to(torch.float32)
        target_2d = target.reshape(-1, target.shape[-1]).to(torch.float32)
        if pred_2d.shape != target_2d.shape:
            raise ValueError(
                f"Shape mismatch for SingleStationKGELoss: pred={tuple(pred_2d.shape)}, target={tuple(target_2d.shape)}"
            )
        if self.station_index < 0 or self.station_index >= pred_2d.shape[-1]:
            raise IndexError(
                f"`station_index` out of bounds: index={self.station_index}, num_outlets={pred_2d.shape[-1]}"
            )

        kge_station = compute_mean_kge(
            pred_2d[:, self.station_index],
            target_2d[:, self.station_index],
            eps=self.eps,
        )
        return 1.0 - kge_station


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_cfg: Optional[Dict[str, Any]] = None,
) -> Optimizer:
    cfg = {} if optimizer_cfg is None else dict(optimizer_cfg)
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(cfg.get("betas", [0.9, 0.999])),
            eps=float(cfg.get("eps", 1e-8)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(cfg.get("betas", [0.9, 0.999])),
            eps=float(cfg.get("eps", 1e-8)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )

    raise ValueError(f"Unsupported optimizer type: `{name}`")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Optional[Dict[str, Any]] = None,
    total_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
):
    cfg = {} if scheduler_cfg is None else dict(scheduler_cfg)
    name = str(cfg.get("name", "none")).lower()
    if name in {"none", "", "null"}:
        return None

    if name in {"plateau", "reduce_on_plateau", "reducelronplateau"}:
        return ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.get("mode", "min")),
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 3)),
            threshold=float(cfg.get("threshold", 1e-4)),
            min_lr=float(cfg.get("min_lr", 0.0)),
        )

    if name in {"step", "steplr"}:
        return StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.5)),
        )

    if name in {"cosine", "cosineannealing", "cosineannealinglr"}:
        t_max = int(cfg.get("t_max", total_epochs if total_epochs is not None else 50))
        return CosineAnnealingLR(optimizer, T_max=max(1, t_max))

    if name in {"onecycle", "onecyclelr"}:
        if total_epochs is None or steps_per_epoch is None:
            raise ValueError("OneCycleLR requires both `total_epochs` and `steps_per_epoch`.")
        max_lr = float(cfg.get("max_lr", cfg.get("lr", 1e-3)))
        pct_start = float(cfg.get("pct_start", 0.3))
        div_factor = float(cfg.get("div_factor", 25.0))
        final_div_factor = float(cfg.get("final_div_factor", 1e4))
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=int(total_epochs),
            steps_per_epoch=int(steps_per_epoch),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    raise ValueError(f"Unsupported scheduler type: `{name}`")


def _extract_field(batch: Any, key: str):
    if isinstance(batch, Mapping):
        return batch[key]
    if hasattr(batch, key):
        return getattr(batch, key)
    raise KeyError(f"Batch has no field `{key}`")


def _extract_prediction(output: Any):
    if isinstance(output, Mapping):
        for key in ("pred", "y_pred", "output", "logits"):
            if key in output:
                return output[key]
        raise KeyError("Model output dict has no `pred`/`y_pred`/`output`/`logits` key.")
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("Model output list/tuple is empty.")
        return output[0]
    return output


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


def _infer_batch_size(batch: Any, target: torch.Tensor) -> int:
    try:
        x = _extract_field(batch, "x")
        if torch.is_tensor(x) and x.dim() >= 4:
            return int(x.shape[0])
    except Exception:
        pass

    if target.dim() >= 3:
        return int(target.shape[0])
    return 1


def _compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    err = pred - target
    mae = torch.mean(torch.abs(err))
    mse = torch.mean(err * err)
    rmse = torch.sqrt(mse + 1e-12)

    denom = torch.sum((target - torch.mean(target)) ** 2)
    nse = 1.0 - torch.sum(err ** 2) / (denom + 1e-12)
    kge = _compute_kge_scalar(pred, target)

    return {
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "nse": float(nse.detach().cpu()),
        "kge": float(kge.detach().cpu()),
    }


def _compute_kge_scalar(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    return compute_mean_kge(pred, target, eps=eps)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        device: str = "auto",
        use_amp: bool = False,
        amp_dtype: str = "float16",
        grad_clip_norm: Optional[float] = None,
        log_interval: int = 50,
        checkpoint_dir: str = "checkpoints/default",
        monitor: str = "val_loss",
        monitor_mode: str = "min",
        early_stopping_patience: Optional[int] = None,
        keep_last_k: int = 3,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        loss_fn: Optional[Callable[[torch.Tensor, Any, nn.Module], Tuple[torch.Tensor, torch.Tensor]]] = None,
        extra_eval_fn: Optional[Callable[["Trainer", Any, str], Dict[str, float]]] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = select_device(device)
        self.model.to(self.device)

        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16 if str(amp_dtype).lower() in {"fp16", "float16", "half"} else torch.bfloat16
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.log_interval = max(1, int(log_interval))
        self.monitor = str(monitor)
        self.monitor_mode = str(monitor_mode).lower()
        self.early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)
        self.keep_last_k = max(1, int(keep_last_k))
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.extra_eval_fn = extra_eval_fn

        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_csv_path = os.path.join(self.checkpoint_dir, "history.csv")
        self.summary_json_path = os.path.join(self.checkpoint_dir, "summary.json")

        scaler_enabled = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.history: List[Dict[str, Any]] = []
        self.best_metric = math.inf if self.monitor_mode == "min" else -math.inf
        self.best_epoch: Optional[int] = None
        self.last_epoch = 0
        self._saved_epoch_ckpts: List[str] = []

    @staticmethod
    def _default_forward(model: nn.Module, batch: Any):
        return model(batch)

    def _default_loss(self, pred: torch.Tensor, batch: Any):
        target = _extract_field(batch, "y")
        return self.criterion(pred, target), target

    def _autocast_context(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=self.amp_dtype)
        return nullcontext()

    def _is_better(self, value: float) -> bool:
        if self.monitor_mode == "min":
            return value < self.best_metric
        if self.monitor_mode == "max":
            return value > self.best_metric
        raise ValueError(f"Unsupported monitor mode: `{self.monitor_mode}`")

    def _save_history_csv(self):
        if not self.history:
            return
        all_keys = sorted({k for row in self.history for k in row.keys()})
        with open(self.history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def _save_summary_json(self):
        payload = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "last_epoch": self.last_epoch,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "history_csv": self.history_csv_path,
        }
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        ckpt = {
            "epoch": int(epoch),
            "model_state": self.model.state_dict(),
            "best_metric": float(self.best_metric),
            "best_epoch": self.best_epoch,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "history": self.history,
        }
        if self.optimizer is not None:
            ckpt["optimizer_state"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            ckpt["scaler_state"] = self.scaler.state_dict()

        last_path = os.path.join(self.checkpoint_dir, "last.ckpt")
        torch.save(ckpt, last_path)

        epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.ckpt")
        torch.save(ckpt, epoch_path)
        self._saved_epoch_ckpts.append(epoch_path)
        while len(self._saved_epoch_ckpts) > self.keep_last_k:
            old = self._saved_epoch_ckpts.pop(0)
            if os.path.exists(old):
                os.remove(old)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.ckpt")
            torch.save(ckpt, best_path)

    def get_best_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best.ckpt")

    def get_last_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "last.ckpt")

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> int:
        path = os.path.abspath(checkpoint_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=strict)

        if self.optimizer is not None and "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler.is_enabled() and "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])

        self.best_metric = float(ckpt.get("best_metric", self.best_metric))
        self.best_epoch = ckpt.get("best_epoch", self.best_epoch)
        self.history = list(ckpt.get("history", self.history))
        self.last_epoch = int(ckpt.get("epoch", 0))
        return self.last_epoch

    def _run_epoch(self, loader, training: bool, epoch: int, split: str) -> Dict[str, float]:
        if training and self.optimizer is None:
            raise RuntimeError("Optimizer is required for training mode.")

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_count = 0
        metric_sum = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "nse": 0.0, "kge": 0.0}

        start_time = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = _move_to_device(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with self._autocast_context():
                    forward = self._default_forward if self.forward_fn is None else self.forward_fn
                    out = forward(self.model, batch)
                    pred = _extract_prediction(out)
                    if self.loss_fn is None:
                        loss, target = self._default_loss(pred, batch)
                    else:
                        loss, target = self.loss_fn(pred, batch, self.criterion)

                if training:
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.optimizer.step()

                    if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step()

            batch_size = _infer_batch_size(batch, target)
            batch_metrics = _compute_regression_metrics(pred.detach(), target.detach())
            batch_metrics["loss"] = float(loss.detach().cpu())

            total_count += batch_size
            for k in metric_sum:
                metric_sum[k] += batch_metrics[k] * batch_size

            if training and step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch:03d}] {split} step {step:04d}/{len(loader):04d} "
                    f"loss={batch_metrics['loss']:.6f} mae={batch_metrics['mae']:.6f} "
                    f"rmse={batch_metrics['rmse']:.6f} nse={batch_metrics['nse']:.6f} "
                    f"kge={batch_metrics['kge']:.6f} "
                    f"time={elapsed:.1f}s"
                )

        if total_count == 0:
            raise RuntimeError(f"DataLoader for split `{split}` is empty.")

        return {k: metric_sum[k] / total_count for k in metric_sum}

    def train_one_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        return self._run_epoch(train_loader, training=True, epoch=epoch, split="train")

    @torch.no_grad()
    def evaluate(self, loader, split: str = "val") -> Dict[str, float]:
        return self._run_epoch(loader, training=False, epoch=self.last_epoch, split=split)

    @torch.no_grad()
    def predict(self, loader, return_target: bool = True):
        self.model.eval()
        preds = []
        targets = []
        for batch in loader:
            batch = _move_to_device(batch, self.device)
            out = self._default_forward(self.model, batch) if self.forward_fn is None else self.forward_fn(self.model, batch)
            pred = _extract_prediction(out)
            preds.append(pred.detach().cpu())
            if return_target:
                targets.append(_extract_field(batch, "y").detach().cpu())

        pred_all = torch.cat(preds, dim=0) if preds else torch.empty(0)
        if return_target:
            target_all = torch.cat(targets, dim=0) if targets else torch.empty(0)
            return pred_all, target_all
        return pred_all

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 20,
        resume_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("`epochs` must be positive.")

        start_epoch = 1
        if resume_path:
            resumed = self.load_checkpoint(resume_path)
            start_epoch = resumed + 1
            print(f"Resumed from `{resume_path}` at epoch={resumed}.")

        bad_epochs = 0
        for epoch in range(start_epoch, epochs + 1):
            epoch_t0 = time.time()
            train_metrics = self.train_one_epoch(train_loader, epoch=epoch)
            record: Dict[str, Any] = {"epoch": epoch}
            record.update({f"train_{k}": v for k, v in train_metrics.items()})

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, split="val")
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
                if self.extra_eval_fn is not None:
                    extra = self.extra_eval_fn(self, val_loader, "val")
                    if extra is not None:
                        for k, v in extra.items():
                            key = k if str(k).startswith("val_") else f"val_{k}"
                            record[key] = v

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_name = self.monitor
                    monitored = record.get(metric_name, record.get("val_loss", record.get("train_loss")))
                    if monitored is not None:
                        self.scheduler.step(monitored)
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            monitor_value = record.get(self.monitor)
            if monitor_value is None:
                monitor_value = record.get("val_loss", record.get("train_loss"))
            if monitor_value is None:
                raise RuntimeError(f"Monitor metric `{self.monitor}` is not available in epoch record.")

            improved = self._is_better(float(monitor_value))
            if improved:
                self.best_metric = float(monitor_value)
                self.best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            self.last_epoch = epoch
            record["best_metric"] = self.best_metric
            record["lr"] = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer is not None else None
            record["seconds"] = time.time() - epoch_t0
            self.history.append(record)

            self._save_checkpoint(epoch=epoch, is_best=improved)
            self._save_history_csv()
            self._save_summary_json()

            lr_value = record.get("lr")
            lr_text = "NA" if lr_value is None else f"{float(lr_value):.3e}"
            train_kge = record.get("train_kge", float("nan"))
            val_kge = record.get("val_kge", float("nan"))
            print(
                f"[Epoch {epoch:03d}] train_loss={record.get('train_loss', float('nan')):.6f} "
                f"val_loss={record.get('val_loss', float('nan')):.6f} "
                f"train_kge={train_kge:.6f} val_kge={val_kge:.6f} "
                f"best({self.monitor})={self.best_metric:.6f} "
                f"lr={lr_text}"
            )

            if self.early_stopping_patience is not None and bad_epochs >= self.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch={epoch} "
                    f"(patience={self.early_stopping_patience})."
                )
                break

        return self.history
