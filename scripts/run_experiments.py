import argparse
import copy
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_run_state(run_dir: str, payload: Dict[str, Any]):
    state = dict(payload)
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_json(os.path.join(run_dir, "run_state.json"), state)


def as_abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def normalize_name(text: str) -> str:
    value = str(text).strip().lower()
    return re.sub(r"[\s_\-]+", "", value)


def to_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def metric_from_nested(payload: Dict[str, Any], *keys) -> float:
    cur = payload
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    return to_float_or_nan(cur)


def find_station_metric(per_outlet_rows: Any, station_name: str, metric_key: str = "kge") -> float:
    if not isinstance(per_outlet_rows, list):
        return float("nan")
    q = normalize_name(station_name)
    for row in per_outlet_rows:
        if not isinstance(row, dict):
            continue
        outlet = row.get("outlet", None)
        if outlet is None:
            continue
        if normalize_name(outlet) == q:
            return to_float_or_nan(row.get(metric_key, float("nan")))
    return float("nan")


def sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(text))


def build_loss_id(loss_cfg: Dict[str, Any]) -> str:
    name = str(loss_cfg.get("name", "loss")).lower()
    if name == "single_station_kge":
        station_name = loss_cfg.get("station_name", None)
        station_index = loss_cfg.get("station_index", None)
        if station_name not in {None, ""}:
            return f"single_station_kge_{sanitize_token(station_name)}"
        if station_index is not None:
            return f"single_station_kge_idx{int(station_index)}"
    return sanitize_token(name)


def build_variant_id(variant_cfg: Dict[str, Any], idx: int) -> str:
    v_id = variant_cfg.get("id", None)
    if v_id not in {None, ""}:
        return str(v_id)
    return f"variant_{idx:02d}"


@dataclass
class Setting:
    setting_id: str
    variant_id: str
    windowsize: int
    khop: int
    loss_cfg: Dict[str, Any]
    loss_id: str
    variant_model_cfg: Dict[str, Any]
    variant_dataset_cfg: Dict[str, Any]


def build_settings(cfg: Dict[str, Any]) -> List[Setting]:
    exp = cfg.get("experiment", {})
    search = exp.get("search", {})

    windowsizes = [int(x) for x in list(search.get("windowsize", []))]
    khops = [int(x) for x in list(search.get("khop", []))]
    variants = list(search.get("variants", []))
    losses = list(exp.get("losses", []))

    if len(windowsizes) == 0:
        raise ValueError("`experiment.search.windowsize` is empty.")
    if len(khops) == 0:
        raise ValueError("`experiment.search.khop` is empty.")
    if len(variants) == 0:
        raise ValueError("`experiment.search.variants` is empty.")
    if len(losses) == 0:
        raise ValueError("`experiment.losses` is empty.")

    settings: List[Setting] = []
    sid = 0
    for v_idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError("Each variant must be a mapping.")
        variant_id = build_variant_id(variant, v_idx)
        variant_model_cfg = dict(variant.get("model", {}))
        variant_dataset_cfg = dict(variant.get("dataset", {}))
        for ws in windowsizes:
            for kh in khops:
                for loss_cfg in losses:
                    sid += 1
                    loss_cfg_dict = dict(loss_cfg)
                    loss_id = build_loss_id(loss_cfg_dict)
                    setting_id = f"S{sid:04d}_{variant_id}_w{ws}_k{kh}_{loss_id}"
                    settings.append(
                        Setting(
                            setting_id=setting_id,
                            variant_id=variant_id,
                            windowsize=int(ws),
                            khop=int(kh),
                            loss_cfg=loss_cfg_dict,
                            loss_id=loss_id,
                            variant_model_cfg=variant_model_cfg,
                            variant_dataset_cfg=variant_dataset_cfg,
                        )
                    )
    return settings


def setup_run_configs(
    setting: Setting,
    seed: int,
    run_dir: str,
    base_train_cfg: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
    base_data_cfg_path: str,
    device: Optional[str],
    train_overrides: Dict[str, Any],
    model_overrides: Dict[str, Any],
) -> Tuple[str, str, str]:
    train_cfg = deep_update(base_train_cfg, train_overrides)
    model_cfg = deep_update(base_model_cfg, model_overrides)

    train_cfg["seed"] = int(seed)
    if device not in {None, ""}:
        train_cfg["device"] = str(device)

    train_cfg["loss"] = dict(setting.loss_cfg)

    train_ds_cfg = dict(train_cfg.get("dataset", {}))
    train_ds_cfg["windowsize"] = int(setting.windowsize)
    train_ds_cfg["khop"] = int(setting.khop)
    train_cfg["dataset"] = deep_update(train_ds_cfg, setting.variant_dataset_cfg)

    model_ds_cfg = dict(model_cfg.get("dataset", {}))
    model_ds_cfg["windowsize"] = int(setting.windowsize)
    model_ds_cfg["khop"] = int(setting.khop)
    model_cfg["dataset"] = deep_update(model_ds_cfg, setting.variant_dataset_cfg)

    model_section = dict(model_cfg.get("model", {}))
    model_section = deep_update(model_section, setting.variant_model_cfg)
    model_cfg["model"] = model_section

    trainer_cfg = dict(train_cfg.get("trainer", {}))
    trainer_cfg["checkpoint_dir"] = run_dir
    trainer_cfg["plot_train_curve"] = True
    trainer_cfg.setdefault("plot_train_curve_file", "objective_vs_epoch.png")
    trainer_cfg["post_train_simulate"] = True
    trainer_cfg["post_train_splits"] = ["calibrate", "verify"]
    trainer_cfg["post_train_plot"] = True
    trainer_cfg.setdefault("post_train_batch_size", 128)
    trainer_cfg.setdefault("post_train_save_dir", "post_train_simulation")
    train_cfg["trainer"] = trainer_cfg

    run_cfg_dir = os.path.join(run_dir, "configs")
    train_cfg_path = os.path.join(run_cfg_dir, "train.yaml")
    model_cfg_path = os.path.join(run_cfg_dir, "model.yaml")
    data_cfg_path = os.path.join(run_cfg_dir, "data.yaml")

    save_yaml(train_cfg_path, train_cfg)
    save_yaml(model_cfg_path, model_cfg)
    # data cfg just points to the chosen shared file content for reproducibility
    save_yaml(data_cfg_path, load_yaml(base_data_cfg_path))
    return train_cfg_path, model_cfg_path, data_cfg_path


def run_subprocess(cmd: List[str], log_path: str, cwd: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
        proc.wait()
        return int(proc.returncode)


def run_output_exists(run_dir: str) -> bool:
    summary_path = os.path.join(run_dir, "summary.json")
    verify_metrics = os.path.join(run_dir, "post_train_simulation", "verify", "metrics_verify.json")
    calibrate_metrics = os.path.join(run_dir, "post_train_simulation", "calibrate", "metrics_calibrate.json")
    return os.path.exists(summary_path) and os.path.exists(verify_metrics) and os.path.exists(calibrate_metrics)


def collect_run_metrics(run_dir: str, target_station_name: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "status": "failed",
        "best_epoch": float("nan"),
        "best_metric": float("nan"),
        "train_best_val_loss": float("nan"),
        "train_best_val_kge": float("nan"),
        "train_best_val_kge_denorm": float("nan"),
        "calibrate_kge_denorm": float("nan"),
        "calibrate_nse_denorm": float("nan"),
        "calibrate_rmse_denorm": float("nan"),
        "verify_kge_denorm": float("nan"),
        "verify_nse_denorm": float("nan"),
        "verify_rmse_denorm": float("nan"),
        "verify_shiquan_kge_denorm": float("nan"),
        "verify_target_station_kge_denorm": float("nan"),
    }

    summary = load_json(os.path.join(run_dir, "summary.json"))
    if isinstance(summary, dict):
        row["best_epoch"] = to_float_or_nan(summary.get("best_epoch", float("nan")))
        row["best_metric"] = to_float_or_nan(summary.get("best_metric", float("nan")))

    history_path = os.path.join(run_dir, "history.csv")
    if os.path.exists(history_path):
        try:
            hist = pd.read_csv(history_path)
            if "val_loss" in hist.columns:
                row["train_best_val_loss"] = to_float_or_nan(hist["val_loss"].min())
            if "val_kge" in hist.columns:
                row["train_best_val_kge"] = to_float_or_nan(hist["val_kge"].max())
            if "val_kge_denorm" in hist.columns:
                row["train_best_val_kge_denorm"] = to_float_or_nan(hist["val_kge_denorm"].max())
        except Exception:
            pass

    cali_payload = load_json(os.path.join(run_dir, "post_train_simulation", "calibrate", "metrics_calibrate.json"))
    if isinstance(cali_payload, dict):
        row["calibrate_kge_denorm"] = metric_from_nested(cali_payload, "metrics_denorm", "kge")
        row["calibrate_nse_denorm"] = metric_from_nested(cali_payload, "metrics_denorm", "nse")
        row["calibrate_rmse_denorm"] = metric_from_nested(cali_payload, "metrics_denorm", "rmse")

    verify_payload = load_json(os.path.join(run_dir, "post_train_simulation", "verify", "metrics_verify.json"))
    if isinstance(verify_payload, dict):
        row["verify_kge_denorm"] = metric_from_nested(verify_payload, "metrics_denorm", "kge")
        row["verify_nse_denorm"] = metric_from_nested(verify_payload, "metrics_denorm", "nse")
        row["verify_rmse_denorm"] = metric_from_nested(verify_payload, "metrics_denorm", "rmse")
        per_outlet = verify_payload.get("per_outlet_denorm", [])
        row["verify_shiquan_kge_denorm"] = find_station_metric(per_outlet, "Shiquan", "kge")
        if target_station_name not in {None, ""}:
            row["verify_target_station_kge_denorm"] = find_station_metric(per_outlet, target_station_name, "kge")

    # success requires summary + both split metrics
    if (
        os.path.exists(os.path.join(run_dir, "summary.json"))
        and os.path.exists(os.path.join(run_dir, "post_train_simulation", "calibrate", "metrics_calibrate.json"))
        and os.path.exists(os.path.join(run_dir, "post_train_simulation", "verify", "metrics_verify.json"))
    ):
        row["status"] = "success"
    return row


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_by_setting(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    success = df[df["status"] == "success"].copy()
    if len(success) == 0:
        return pd.DataFrame()

    static_cols = [
        "setting_id",
        "variant_id",
        "windowsize",
        "khop",
        "loss_name",
        "loss_id",
        "target_station_name",
        "model_name",
        "input_projector",
        "spatial",
        "temporal",
    ]
    metric_cols = [
        "train_best_val_loss",
        "train_best_val_kge",
        "train_best_val_kge_denorm",
        "calibrate_kge_denorm",
        "calibrate_nse_denorm",
        "calibrate_rmse_denorm",
        "verify_kge_denorm",
        "verify_nse_denorm",
        "verify_rmse_denorm",
        "verify_shiquan_kge_denorm",
        "verify_target_station_kge_denorm",
    ]

    rows = []
    for setting_id, group in success.groupby("setting_id"):
        first = group.iloc[0]
        row = {k: first.get(k, None) for k in static_cols}
        row["seed_count"] = int(len(group))
        for c in metric_cols:
            vals = pd.to_numeric(group[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{c}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    if "verify_kge_denorm_mean" in out.columns:
        out = out.sort_values(by="verify_kge_denorm_mean", ascending=False).reset_index(drop=True)
    return out


def dataframe_to_markdown_table(df: pd.DataFrame, columns: List[str], max_rows: int = 20) -> str:
    if len(df) == 0:
        return "No rows.\n"
    show = df.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |\n"
    lines = [header, sep]
    for _, r in show.iterrows():
        vals = []
        for c in columns:
            v = r[c]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |\n")
    return "".join(lines)


def select_top_settings_staged(
    run_df: pd.DataFrame,
    screen_seed: int,
    top_k_per_loss: int,
    ranking_metric: str,
) -> List[str]:
    if len(run_df) == 0:
        return []
    screen = run_df[(run_df["seed"] == int(screen_seed)) & (run_df["status"] == "success")].copy()
    if len(screen) == 0:
        return []
    if ranking_metric not in screen.columns:
        raise ValueError(f"Staged ranking metric `{ranking_metric}` not found in run table columns.")

    selected: List[str] = []
    for _, group in screen.groupby("loss_id"):
        g = group.copy()
        g[ranking_metric] = pd.to_numeric(g[ranking_metric], errors="coerce")
        g = g[g[ranking_metric].notna()].sort_values(by=ranking_metric, ascending=False)
        selected.extend(list(g["setting_id"].head(int(top_k_per_loss)).astype(str).values))
    return sorted(set(selected))


def parse_args():
    parser = argparse.ArgumentParser(description="Run routing model experiment matrix and summarize results.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Experiment config YAML.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "full", "staged"],
        help="Execution mode: full factorial or staged (screen + top-k).",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional max runs for this invocation.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate plan and run manifests, no training.")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_cfg = load_yaml(as_abs(args.config))
    exp = exp_cfg.get("experiment", {})
    if not isinstance(exp, dict):
        raise ValueError("`experiment` section missing in experiment config.")

    exp_name = str(exp.get("name", "experiment")).strip()
    checkpoint_root = as_abs(str(exp.get("checkpoint_root", "checkpoints/experiments")))
    exp_root = os.path.join(checkpoint_root, exp_name)
    design_dir = os.path.join(exp_root, "design")
    results_dir = os.path.join(exp_root, "results")
    os.makedirs(design_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    base_train_cfg_path = as_abs(str(exp.get("base_train_cfg", "configs/train.yaml")))
    base_model_cfg_path = as_abs(str(exp.get("base_model_cfg", "configs/model.yaml")))
    base_data_cfg_path = as_abs(str(exp.get("base_data_cfg", "configs/data.yaml")))

    base_train_cfg = load_yaml(base_train_cfg_path)
    base_model_cfg = load_yaml(base_model_cfg_path)

    settings = build_settings(exp_cfg)
    settings_rows = []
    for s in settings:
        row = {
            "setting_id": s.setting_id,
            "variant_id": s.variant_id,
            "windowsize": s.windowsize,
            "khop": s.khop,
            "loss_name": s.loss_cfg.get("name", None),
            "loss_id": s.loss_id,
            "target_station_name": s.loss_cfg.get("station_name", None),
            "model_name": s.variant_model_cfg.get("name", None),
            "input_projector": s.variant_model_cfg.get("input_projector", None),
            "spatial": s.variant_model_cfg.get("spatial", None),
            "temporal": s.variant_model_cfg.get("temporal", None),
        }
        settings_rows.append(row)
    write_csv(os.path.join(design_dir, "settings_manifest.csv"), settings_rows)
    manifest_path = os.path.join(design_dir, "experiment_manifest.json")
    save_json(
        manifest_path,
        {
            "config_path": as_abs(args.config),
            "base_train_cfg": base_train_cfg_path,
            "base_model_cfg": base_model_cfg_path,
            "base_data_cfg": base_data_cfg_path,
            "total_settings": len(settings),
        },
    )

    seeds = [int(x) for x in list(exp.get("seeds", [42]))]
    device = exp.get("device", None)
    continue_on_error = bool(exp.get("continue_on_error", True))
    skip_existing = bool(exp.get("skip_existing", True))
    train_overrides = dict(exp.get("train_overrides", {}))
    model_overrides = dict(exp.get("model_overrides", {}))

    staged_cfg = dict(exp.get("staged", {}))
    staged_enabled = bool(staged_cfg.get("enabled", False))
    mode = args.mode
    if mode == "auto":
        mode = "staged" if staged_enabled else "full"
    if mode == "staged" and not staged_enabled:
        raise ValueError("`--mode staged` requested but `experiment.staged.enabled` is false.")

    screen_seed = int(staged_cfg.get("screen_seed", seeds[0]))
    top_k_per_loss = int(staged_cfg.get("top_k_per_loss", 5))
    ranking_metric = str(staged_cfg.get("ranking_metric", "verify_kge_denorm"))

    save_json(
        manifest_path,
        {
            "config_path": as_abs(args.config),
            "base_train_cfg": base_train_cfg_path,
            "base_model_cfg": base_model_cfg_path,
            "base_data_cfg": base_data_cfg_path,
            "total_settings": len(settings),
            "seeds": seeds,
            "mode": mode,
            "staged": {
                "enabled": staged_enabled,
                "screen_seed": screen_seed,
                "top_k_per_loss": top_k_per_loss,
                "ranking_metric": ranking_metric,
            },
        },
    )

    print(f"[Experiment] name={exp_name}")
    print(f"[Experiment] settings={len(settings)}, seeds={seeds}, mode={mode}")
    print(f"[Experiment] output_root={exp_root}")

    # Build run plan.
    plan: List[Tuple[Setting, int]] = []
    if mode == "full":
        for s in settings:
            for seed in seeds:
                plan.append((s, int(seed)))
    else:
        # Stage 1 screen runs
        for s in settings:
            plan.append((s, int(screen_seed)))

    if args.max_runs is not None:
        plan = plan[: int(args.max_runs)]
    print(f"[Plan] runs_in_current_phase={len(plan)}")

    run_rows: List[Dict[str, Any]] = []
    run_level_csv_live = os.path.join(results_dir, "run_level_metrics_live.csv")

    def _run_one(setting: Setting, seed: int):
        setting_dir = os.path.join(exp_root, setting.setting_id)
        run_dir = os.path.join(setting_dir, f"seed_{int(seed):04d}")
        os.makedirs(run_dir, exist_ok=True)

        row = {
            "setting_id": setting.setting_id,
            "variant_id": setting.variant_id,
            "seed": int(seed),
            "windowsize": int(setting.windowsize),
            "khop": int(setting.khop),
            "loss_name": setting.loss_cfg.get("name", None),
            "loss_id": setting.loss_id,
            "target_station_name": setting.loss_cfg.get("station_name", None),
            "model_name": setting.variant_model_cfg.get("name", None),
            "input_projector": setting.variant_model_cfg.get("input_projector", None),
            "spatial": setting.variant_model_cfg.get("spatial", None),
            "temporal": setting.variant_model_cfg.get("temporal", None),
            "run_dir": run_dir,
            "train_log": os.path.join(run_dir, "train.log"),
        }
        write_run_state(
            run_dir,
            {
                "state": "prepared",
                "setting_id": setting.setting_id,
                "seed": int(seed),
                "execution": "pending",
            },
        )

        if not args.dry_run:
            if skip_existing and run_output_exists(run_dir):
                row["execution"] = "skipped_existing"
                write_run_state(
                    run_dir,
                    {
                        "state": "skipped_existing",
                        "setting_id": setting.setting_id,
                        "seed": int(seed),
                        "execution": row["execution"],
                    },
                )
            else:
                train_cfg_path, model_cfg_path, data_cfg_path = setup_run_configs(
                    setting=setting,
                    seed=int(seed),
                    run_dir=run_dir,
                    base_train_cfg=base_train_cfg,
                    base_model_cfg=base_model_cfg,
                    base_data_cfg_path=base_data_cfg_path,
                    device=device,
                    train_overrides=train_overrides,
                    model_overrides=model_overrides,
                )

                cmd = [
                    sys.executable,
                    os.path.join("scripts", "train.py"),
                    "--train-cfg",
                    train_cfg_path,
                    "--model-cfg",
                    model_cfg_path,
                    "--data-cfg",
                    data_cfg_path,
                ]
                if device not in {None, ""}:
                    cmd.extend(["--device", str(device)])

                print(f"[Run] setting={setting.setting_id} seed={seed}")
                write_run_state(
                    run_dir,
                    {
                        "state": "running",
                        "setting_id": setting.setting_id,
                        "seed": int(seed),
                        "execution": "running",
                        "command": cmd,
                    },
                )
                code = run_subprocess(cmd=cmd, log_path=row["train_log"], cwd=PROJECT_ROOT)
                row["return_code"] = int(code)
                row["execution"] = "finished" if code == 0 else "failed"
                write_run_state(
                    run_dir,
                    {
                        "state": "finished" if code == 0 else "failed",
                        "setting_id": setting.setting_id,
                        "seed": int(seed),
                        "execution": row["execution"],
                        "return_code": int(code),
                    },
                )
                if code != 0 and not continue_on_error:
                    extra = collect_run_metrics(run_dir, target_station_name=setting.loss_cfg.get("station_name", None))
                    row.update(extra)
                    run_rows.append(row)
                    write_csv(os.path.join(results_dir, "run_level_metrics.csv"), run_rows)
                    write_csv(run_level_csv_live, run_rows)
                    raise RuntimeError(f"Training failed for setting={setting.setting_id}, seed={seed}.")
        else:
            row["execution"] = "dry_run"
            row["status"] = "dry_run"
            row["best_epoch"] = float("nan")
            row["best_metric"] = float("nan")
            write_run_state(
                run_dir,
                {
                    "state": "dry_run",
                    "setting_id": setting.setting_id,
                    "seed": int(seed),
                    "execution": row["execution"],
                    "status": row["status"],
                },
            )

        if not args.dry_run:
            extra = collect_run_metrics(run_dir, target_station_name=setting.loss_cfg.get("station_name", None))
            row.update(extra)
            write_run_state(
                run_dir,
                {
                    "state": str(row.get("status", "unknown")),
                    "setting_id": setting.setting_id,
                    "seed": int(seed),
                    "execution": row.get("execution", None),
                    "status": row.get("status", None),
                    "return_code": row.get("return_code", None),
                    "best_epoch": row.get("best_epoch", None),
                    "best_metric": row.get("best_metric", None),
                    "verify_kge_denorm": row.get("verify_kge_denorm", None),
                },
            )
        run_rows.append(row)
        write_csv(run_level_csv_live, run_rows)

    for setting, seed in plan:
        _run_one(setting, seed)

    # Stage 2 (staged mode): select top-k settings from screen results and run full seeds.
    if mode == "staged" and not args.dry_run:
        run_df_screen = pd.DataFrame(run_rows)
        selected_setting_ids = select_top_settings_staged(
            run_df=run_df_screen,
            screen_seed=screen_seed,
            top_k_per_loss=top_k_per_loss,
            ranking_metric=ranking_metric,
        )
        print(
            f"[Stage2] selected_settings={len(selected_setting_ids)} "
            f"(top_k_per_loss={top_k_per_loss}, metric={ranking_metric})"
        )

        stage2_plan: List[Tuple[Setting, int]] = []
        settings_by_id = {s.setting_id: s for s in settings}
        for sid in selected_setting_ids:
            s = settings_by_id[sid]
            for seed in seeds:
                stage2_plan.append((s, int(seed)))
        if args.max_runs is not None:
            stage2_plan = stage2_plan[: int(args.max_runs)]
        print(f"[Stage2] runs_in_current_phase={len(stage2_plan)}")

        for setting, seed in stage2_plan:
            # Avoid duplicate re-run rows when screen seed already exists and output completed.
            if any((r["setting_id"] == setting.setting_id and int(r["seed"]) == int(seed)) for r in run_rows):
                setting_dir = os.path.join(exp_root, setting.setting_id)
                run_dir = os.path.join(setting_dir, f"seed_{int(seed):04d}")
                if skip_existing and run_output_exists(run_dir):
                    # keep existing row as-is
                    continue
            _run_one(setting, seed)

    # Final tables
    write_csv(os.path.join(results_dir, "run_level_metrics.csv"), run_rows)
    run_df = pd.DataFrame(run_rows)
    if len(run_df) > 0:
        run_df.to_csv(os.path.join(results_dir, "run_level_metrics_pandas.csv"), index=False)

    agg_df = aggregate_by_setting(run_df)
    if len(agg_df) > 0:
        agg_path = os.path.join(results_dir, "setting_aggregate_metrics.csv")
        agg_df.to_csv(agg_path, index=False)
        by_loss_rows = []
        for loss_id, g in agg_df.groupby("loss_id"):
            g_sorted = g.sort_values(by="verify_kge_denorm_mean", ascending=False)
            out_path = os.path.join(results_dir, f"leaderboard_{loss_id}.csv")
            g_sorted.to_csv(out_path, index=False)
            by_loss_rows.append({"loss_id": loss_id, "path": out_path, "count": int(len(g_sorted))})
        write_csv(os.path.join(results_dir, "leaderboard_index.csv"), by_loss_rows)

        report_lines = []
        report_lines.append(f"# Experiment Report: {exp_name}\n")
        report_lines.append(f"- Total run rows: {len(run_df)}\n")
        report_lines.append(f"- Successful runs: {int((run_df['status'] == 'success').sum())}\n")
        report_lines.append(f"- Aggregated settings: {len(agg_df)}\n")
        report_lines.append("\n## Top Settings (Overall, by verify_kge_denorm_mean)\n")
        report_lines.append(
            dataframe_to_markdown_table(
                agg_df,
                columns=[
                    "setting_id",
                    "loss_id",
                    "seed_count",
                    "train_best_val_kge_denorm_mean",
                    "verify_kge_denorm_mean",
                    "verify_shiquan_kge_denorm_mean",
                ],
                max_rows=20,
            )
        )

        for loss_id, g in agg_df.groupby("loss_id"):
            report_lines.append(f"\n## Top Settings for `{loss_id}`\n")
            g_sorted = g.sort_values(by="verify_kge_denorm_mean", ascending=False).reset_index(drop=True)
            report_lines.append(
                dataframe_to_markdown_table(
                    g_sorted,
                    columns=[
                        "setting_id",
                        "seed_count",
                        "train_best_val_kge_denorm_mean",
                        "verify_kge_denorm_mean",
                        "verify_nse_denorm_mean",
                        "verify_shiquan_kge_denorm_mean",
                    ],
                    max_rows=10,
                )
            )

        report_path = os.path.join(results_dir, "experiment_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("".join(report_lines))
        print(f"[Done] report={report_path}")
        print(f"[Done] run_table={os.path.join(results_dir, 'run_level_metrics.csv')}")
        print(f"[Done] agg_table={os.path.join(results_dir, 'setting_aggregate_metrics.csv')}")
    else:
        print("[Done] No successful runs yet. Check run logs and rerun with skip_existing=true.")


if __name__ == "__main__":
    main()
