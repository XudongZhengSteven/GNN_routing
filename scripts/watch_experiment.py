import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def as_abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def parse_args():
    parser = argparse.ArgumentParser(description="Watch experiment progress in terminal.")
    parser.add_argument("--config", default=None, help="Experiment config YAML. Optional if --exp-name is provided.")
    parser.add_argument("--exp-name", default=None, help="Experiment name under checkpoints/experiments.")
    parser.add_argument("--root", default="checkpoints/experiments", help="Experiment root directory.")
    parser.add_argument("--interval", type=int, default=20, help="Refresh interval in seconds.")
    parser.add_argument("--stale-seconds", type=int, default=300, help="No-log-update threshold for stale state.")
    parser.add_argument("--top-k", type=int, default=8, help="Top-K completed runs by verify KGE.")
    parser.add_argument(
        "--history-tail",
        type=int,
        default=6,
        help="Show last N epochs trend for active/stale runs (0 to disable).",
    )
    parser.add_argument(
        "--history-metric",
        default="val_kge_denorm",
        help="Preferred metric column for epoch trend in history.csv.",
    )
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear terminal between refreshes.")
    return parser.parse_args()


def resolve_exp_name(args) -> str:
    if args.exp_name not in {None, ""}:
        return str(args.exp_name)
    if args.config in {None, ""}:
        raise ValueError("Either --exp-name or --config must be provided.")
    cfg = load_yaml(as_abs(args.config))
    exp = cfg.get("experiment", {})
    name = exp.get("name", None)
    if name in {None, ""}:
        raise ValueError("`experiment.name` not found in config.")
    return str(name)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if v != v:
            return float("nan")
        return v
    except Exception:
        return float("nan")


def tail_text(path: str, max_bytes: int = 65536) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek = max(0, size - int(max_bytes))
            f.seek(seek, os.SEEK_SET)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_log_progress(log_path: str) -> Dict[str, Any]:
    text = tail_text(log_path, max_bytes=131072)
    if text == "":
        return {
            "last_epoch": None,
            "train_loss": float("nan"),
            "val_loss": float("nan"),
            "best_metric": float("nan"),
        }

    epoch_matches = re.findall(r"\[Epoch\s+(\d+)\]", text)
    last_epoch = int(epoch_matches[-1]) if len(epoch_matches) > 0 else None

    summary_matches = re.findall(
        r"\[Epoch\s+(\d+)\]\s+train_loss=([\-+eE0-9\.]+)\s+val_loss=([\-+eE0-9\.]+).*best\([^)]+\)=([\-+eE0-9\.]+)",
        text,
    )
    if len(summary_matches) > 0:
        _, train_loss, val_loss, best_metric = summary_matches[-1]
        return {
            "last_epoch": last_epoch,
            "train_loss": safe_float(train_loss),
            "val_loss": safe_float(val_loss),
            "best_metric": safe_float(best_metric),
        }

    best_matches = re.findall(r"best\([^)]+\)=([\-+eE0-9\.]+)", text)
    best_metric = safe_float(best_matches[-1]) if len(best_matches) > 0 else float("nan")
    return {
        "last_epoch": last_epoch,
        "train_loss": float("nan"),
        "val_loss": float("nan"),
        "best_metric": best_metric,
    }


def parse_history_progress(history_path: str, summary_path: str) -> Dict[str, Any]:
    out = {
        "last_epoch": None,
        "train_loss": float("nan"),
        "val_loss": float("nan"),
        "best_metric": float("nan"),
    }
    if os.path.exists(history_path):
        try:
            # Read last non-empty row from history.csv
            with open(history_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                last = None
                for row in reader:
                    if row:
                        last = row
            if isinstance(last, dict):
                if "epoch" in last and str(last["epoch"]).strip() != "":
                    out["last_epoch"] = int(float(last["epoch"]))
                if "train_loss" in last:
                    out["train_loss"] = safe_float(last["train_loss"])
                if "val_loss" in last:
                    out["val_loss"] = safe_float(last["val_loss"])
                # Prefer explicit `best_metric` recorded by trainer.
                if "best_metric" in last:
                    out["best_metric"] = safe_float(last["best_metric"])
        except Exception:
            pass

    summary = load_json(summary_path)
    if isinstance(summary, dict):
        if out["last_epoch"] is None and summary.get("last_epoch", None) not in {None, ""}:
            try:
                out["last_epoch"] = int(summary.get("last_epoch"))
            except Exception:
                pass
        if summary.get("best_metric", None) not in {None, ""}:
            out["best_metric"] = safe_float(summary.get("best_metric"))
    return out


def parse_history_tail(
    history_path: str,
    preferred_metric: str = "val_kge_denorm",
    tail_n: int = 6,
) -> Dict[str, Any]:
    out = {
        "metric": None,
        "pairs": [],  # List[Tuple[int, float]]
    }
    if tail_n <= 0 or not os.path.exists(history_path):
        return out

    try:
        rows = []
        with open(history_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    rows.append(row)
        if len(rows) == 0:
            return out

        last = rows[-1]
        preferred = str(preferred_metric).strip()
        candidates = [
            preferred,
            "val_kge_denorm",
            "val_kge",
            "val_loss",
            "train_loss",
        ]
        metric_used = None
        for c in candidates:
            if c in last and str(last.get(c, "")).strip() != "":
                metric_used = c
                break
        if metric_used is None:
            return out

        out["metric"] = metric_used
        tail_rows = rows[-int(tail_n) :]
        pairs = []
        for r in tail_rows:
            if "epoch" not in r:
                continue
            try:
                ep = int(float(r.get("epoch")))
            except Exception:
                continue
            val = safe_float(r.get(metric_used))
            pairs.append((ep, val))
        out["pairs"] = pairs
        return out
    except Exception:
        return out


def metric_from_json(path: str, *keys) -> float:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return float("nan")
    cur: Any = payload
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    return safe_float(cur)


def discover_settings(exp_dir: str) -> List[str]:
    settings_manifest = os.path.join(exp_dir, "design", "settings_manifest.csv")
    if os.path.exists(settings_manifest):
        try:
            df = pd.read_csv(settings_manifest)
            if "setting_id" in df.columns:
                return [str(x) for x in df["setting_id"].dropna().tolist()]
        except Exception:
            pass

    out = []
    for name in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, name)
        if not os.path.isdir(path):
            continue
        if name in {"design", "results"}:
            continue
        out.append(name)
    return out


def discover_seeds(exp_dir: str, config_path: Optional[str]) -> List[int]:
    if config_path not in {None, ""}:
        try:
            cfg = load_yaml(as_abs(config_path))
            seeds = cfg.get("experiment", {}).get("seeds", None)
            if isinstance(seeds, list) and len(seeds) > 0:
                return sorted({int(x) for x in seeds})
        except Exception:
            pass

    manifest = load_json(os.path.join(exp_dir, "design", "experiment_manifest.json"))
    if isinstance(manifest, dict):
        seeds = manifest.get("seeds", None)
        if isinstance(seeds, list) and len(seeds) > 0:
            try:
                return sorted({int(x) for x in seeds})
            except Exception:
                pass

    found = set()
    for setting in discover_settings(exp_dir):
        setting_dir = os.path.join(exp_dir, setting)
        if not os.path.isdir(setting_dir):
            continue
        for name in os.listdir(setting_dir):
            m = re.match(r"seed_(\d+)$", name)
            if m:
                found.add(int(m.group(1)))
    return sorted(found)


def build_run_record(
    setting_id: str,
    seed: int,
    run_dir: str,
    stale_seconds: int,
    now_ts: float,
    history_tail: int = 6,
    history_metric: str = "val_kge_denorm",
) -> Dict[str, Any]:
    log_path = os.path.join(run_dir, "train.log")
    history_path = os.path.join(run_dir, "history.csv")
    state_path = os.path.join(run_dir, "run_state.json")
    summary_path = os.path.join(run_dir, "summary.json")
    cali_path = os.path.join(run_dir, "post_train_simulation", "calibrate", "metrics_calibrate.json")
    verify_path = os.path.join(run_dir, "post_train_simulation", "verify", "metrics_verify.json")

    exists = os.path.isdir(run_dir)
    has_log = os.path.exists(log_path)
    has_history = os.path.exists(history_path)
    has_success = os.path.exists(summary_path) and os.path.exists(cali_path) and os.path.exists(verify_path)

    last_mtime = None
    for p in [log_path, history_path, summary_path, cali_path, verify_path, state_path]:
        if os.path.exists(p):
            m = os.path.getmtime(p)
            last_mtime = m if last_mtime is None else max(last_mtime, m)
    age_sec = None if last_mtime is None else max(0.0, now_ts - last_mtime)

    run_state = load_json(state_path)
    state_str = str(run_state.get("state", "")).lower() if isinstance(run_state, dict) else ""

    if not exists:
        status = "pending"
    elif has_success:
        status = "success"
    elif state_str in {"running"}:
        if age_sec is not None and age_sec > float(stale_seconds):
            status = "stale"
        else:
            status = "running"
    elif state_str in {"failed"}:
        status = "failed"
    elif state_str in {"dry_run"}:
        status = "dry_run"
    elif has_log or has_history:
        if age_sec is not None and age_sec <= float(stale_seconds):
            status = "running"
        else:
            status = "stale"
    else:
        status = "pending"

    prog_log = parse_log_progress(log_path) if has_log else {
        "last_epoch": None,
        "train_loss": float("nan"),
        "val_loss": float("nan"),
        "best_metric": float("nan"),
    }
    prog_hist = parse_history_progress(history_path=history_path, summary_path=summary_path)
    trend = parse_history_tail(
        history_path=history_path,
        preferred_metric=history_metric,
        tail_n=int(history_tail),
    )
    prog = prog_hist if prog_hist["last_epoch"] is not None else prog_log

    record = {
        "setting_id": setting_id,
        "seed": int(seed),
        "run_dir": run_dir,
        "status": status,
        "age_sec": age_sec if age_sec is not None else float("nan"),
        "last_epoch": prog["last_epoch"],
        "train_loss": prog["train_loss"],
        "val_loss": prog["val_loss"],
        "best_metric_log": prog["best_metric"],
        "trend_metric": trend.get("metric", None),
        "trend_pairs": trend.get("pairs", []),
        "verify_kge_denorm": metric_from_json(verify_path, "metrics_denorm", "kge"),
        "calibrate_kge_denorm": metric_from_json(cali_path, "metrics_denorm", "kge"),
    }
    return record


def build_snapshot(
    exp_dir: str,
    config_path: Optional[str],
    stale_seconds: int,
    history_tail: int = 6,
    history_metric: str = "val_kge_denorm",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    settings = discover_settings(exp_dir)
    seeds = discover_seeds(exp_dir, config_path=config_path)
    now_ts = time.time()

    rows: List[Dict[str, Any]] = []
    if len(settings) == 0:
        return pd.DataFrame(), {"settings": 0, "seeds": len(seeds), "expected_runs": 0}

    if len(seeds) == 0:
        # If seeds are unknown, use existing seed dirs only.
        for setting_id in settings:
            setting_dir = os.path.join(exp_dir, setting_id)
            if not os.path.isdir(setting_dir):
                continue
            for name in os.listdir(setting_dir):
                m = re.match(r"seed_(\d+)$", name)
                if not m:
                    continue
                seed = int(m.group(1))
                run_dir = os.path.join(setting_dir, name)
                rows.append(
                    build_run_record(
                        setting_id,
                        seed,
                        run_dir,
                        stale_seconds=stale_seconds,
                        now_ts=now_ts,
                        history_tail=history_tail,
                        history_metric=history_metric,
                    )
                )
        expected_runs = len(rows)
    else:
        expected_runs = len(settings) * len(seeds)
        for setting_id in settings:
            setting_dir = os.path.join(exp_dir, setting_id)
            for seed in seeds:
                run_dir = os.path.join(setting_dir, f"seed_{int(seed):04d}")
                rows.append(
                    build_run_record(
                        setting_id,
                        seed,
                        run_dir,
                        stale_seconds=stale_seconds,
                        now_ts=now_ts,
                        history_tail=history_tail,
                        history_metric=history_metric,
                    )
                )

    df = pd.DataFrame(rows)
    meta = {
        "settings": len(settings),
        "seeds": len(seeds),
        "expected_runs": int(expected_runs),
    }
    return df, meta


def short_setting_id(text: str, max_len: int = 50) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def render_snapshot(exp_name: str, exp_dir: str, df: pd.DataFrame, meta: Dict[str, Any], top_k: int):
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_text}] Experiment Watch: {exp_name}")
    print(f"path: {exp_dir}")

    if len(df) == 0:
        print("No run data found yet.")
        return

    counts = df["status"].value_counts().to_dict()
    success = int(counts.get("success", 0))
    running = int(counts.get("running", 0))
    stale = int(counts.get("stale", 0))
    failed = int(counts.get("failed", 0))
    pending = int(counts.get("pending", 0))
    dry_run = int(counts.get("dry_run", 0))
    expected_runs = int(meta.get("expected_runs", len(df)))
    progress = (100.0 * success / expected_runs) if expected_runs > 0 else 0.0

    print(
        f"settings={meta.get('settings', '?')} seeds={meta.get('seeds', '?')} "
        f"expected_runs={expected_runs}"
    )
    print(
        "status: "
        f"success={success} running={running} stale={stale} failed={failed} "
        f"pending={pending} dry_run={dry_run} | progress={progress:.1f}%"
    )

    done = df[df["status"] == "success"].copy()
    if len(done) > 0:
        done["verify_kge_denorm"] = pd.to_numeric(done["verify_kge_denorm"], errors="coerce")
        done = done.sort_values(by="verify_kge_denorm", ascending=False)
        print("\nTop completed runs (by verify_kge_denorm):")
        print("  setting_id                                         seed   verify_kge  cali_kge")
        for _, r in done.head(int(top_k)).iterrows():
            print(
                f"  {short_setting_id(r['setting_id'], 48):48s} "
                f"{int(r['seed']):4d}   "
                f"{safe_float(r['verify_kge_denorm']):9.4f}  "
                f"{safe_float(r['calibrate_kge_denorm']):8.4f}"
            )

    active = df[df["status"].isin(["running", "stale", "failed"])].copy()
    if len(active) > 0:
        active["age_sec"] = pd.to_numeric(active["age_sec"], errors="coerce")
        active = active.sort_values(by="age_sec", ascending=True)
        print("\nActive / problematic runs:")
        print("  setting_id                                         seed   status   epoch      best    age(min)")
        for _, r in active.head(max(20, int(top_k) * 2)).iterrows():
            epoch = "-" if pd.isna(r["last_epoch"]) else str(int(r["last_epoch"]))
            age_min = safe_float(r["age_sec"]) / 60.0 if not pd.isna(r["age_sec"]) else float("nan")
            print(
                f"  {short_setting_id(r['setting_id'], 48):48s} "
                f"{int(r['seed']):4d}   "
                f"{str(r['status']):7s} "
                f"{epoch:>5s}  "
                f"{safe_float(r['best_metric_log']):9.4f}  "
                f"{age_min:8.1f}"
            )
            trend_pairs = r.get("trend_pairs", [])
            trend_metric = r.get("trend_metric", None)
            if isinstance(trend_pairs, list) and len(trend_pairs) > 0 and trend_metric not in {None, ""}:
                pieces = []
                for ep, val in trend_pairs:
                    ep_text = str(ep)
                    if isinstance(val, float) and (val != val):
                        val_text = "nan"
                    else:
                        try:
                            val_text = f"{float(val):.4f}"
                        except Exception:
                            val_text = "nan"
                    pieces.append(f"{ep_text}:{val_text}")
                print(f"    trend[{trend_metric}]: " + " | ".join(pieces))


def main():
    args = parse_args()
    exp_name = resolve_exp_name(args)
    root_abs = as_abs(args.root)
    exp_dir = os.path.join(root_abs, exp_name)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    while True:
        df, meta = build_snapshot(
            exp_dir,
            config_path=args.config,
            stale_seconds=int(args.stale_seconds),
            history_tail=int(args.history_tail),
            history_metric=str(args.history_metric),
        )
        if not args.no_clear:
            os.system("cls" if os.name == "nt" else "clear")
        render_snapshot(exp_name=exp_name, exp_dir=exp_dir, df=df, meta=meta, top_k=int(args.top_k))
        if args.once:
            break
        time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    main()
