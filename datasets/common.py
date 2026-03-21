import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from netCDF4 import num2date


def infer_scenario(dataset_type: str) -> str:
    if dataset_type in {"train", "warmup", "calibrate"}:
        return "cali0"
    if dataset_type in {"val", "test", "verify"}:
        return "vali0"
    return "cali0"


def as_numpy(x):
    if np.ma.isMaskedArray(x):
        return x.filled(np.nan)
    return np.asarray(x)


def decode_time(time_var) -> pd.DatetimeIndex:
    values = time_var[:]
    units = getattr(time_var, "units", None)
    if units is None:
        return pd.date_range("1970-01-01", periods=len(values), freq="D")

    calendar = getattr(time_var, "calendar", "standard")
    dt = num2date(values, units=units, calendar=calendar)
    date_text = [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dt]
    return pd.to_datetime(date_text)


def load_yaml_cfg(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_file(data_home: str, patterns: List[str]) -> str:
    for pattern in patterns:
        full_pattern = os.path.join(data_home, pattern)
        if any(token in pattern for token in ["*", "?", "["]):
            matches = sorted(glob.glob(full_pattern))
            if matches:
                return matches[0]
        elif os.path.exists(full_pattern):
            return full_pattern

    joined = " | ".join(patterns)
    raise FileNotFoundError(f"No file matched in `{data_home}`: {joined}")


def dataset_type_to_split(dataset_type: str) -> str:
    mapping = {
        "train": "calibrate",
        "calibrate": "calibrate",
        "val": "verify",
        "verify": "verify",
        "test": "verify",
        "warmup": "warmup",
        "full": "full",
        "all": "full",
    }
    if dataset_type not in mapping:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return mapping[dataset_type]


def build_time_slice(date_index: pd.DatetimeIndex, split_name: str, time_cfg: Optional[Dict]) -> slice:
    if split_name == "full" or time_cfg is None:
        return slice(0, len(date_index))

    split_cfg = time_cfg.get("time", {}).get("split", {}).get(split_name)
    if split_cfg is None:
        return slice(0, len(date_index))

    start = pd.to_datetime(split_cfg["start"]).normalize()
    end = pd.to_datetime(split_cfg["end"]).normalize()
    dates = pd.to_datetime(date_index).normalize()

    idx = np.where((dates >= start) & (dates <= end))[0]
    if len(idx) == 0:
        raise ValueError(
            f"No overlap between data time index and split `{split_name}`: "
            f"{start.date()} to {end.date()}."
        )

    return slice(int(idx[0]), int(idx[-1]) + 1)
