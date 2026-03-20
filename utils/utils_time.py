import pandas as pd


def build_time_dict(cfg):
    """
    build time dict based on data.yaml
    """

    time_cfg = cfg["time"]
    
    full_start = time_cfg["full_period"]["start"]
    full_end = time_cfg["full_period"]["end"]

    freq_input = time_cfg["freq"]["input"]
    freq_eval = time_cfg["freq"]["evaluate"]

    # ===== 2. full period =====
    date_input = pd.date_range(full_start, full_end, freq=freq_input)
    date_eval = pd.date_range(full_start, full_end, freq=freq_eval)

    # ===== 3. split =====
    def build_split(split_cfg, freq):
        return pd.date_range(
            split_cfg["start"],
            split_cfg["end"],
            freq=freq
        )

    warmup_input = build_split(time_cfg["split"]["warmup"], freq_input)
    calibrate_input = build_split(time_cfg["split"]["calibrate"], freq_input)
    verify_input = build_split(time_cfg["split"]["verify"], freq_input)

    warmup_eval = build_split(time_cfg["split"]["warmup"], freq_eval)
    calibrate_eval = build_split(time_cfg["split"]["calibrate"], freq_eval)
    verify_eval = build_split(time_cfg["split"]["verify"], freq_eval)

    return {
        "input": {
            "full": date_input,
            "warmup": warmup_input,
            "calibrate": calibrate_input,
            "verify": verify_input,
        },
        "eval": {
            "full": date_eval,
            "warmup": warmup_eval,
            "calibrate": calibrate_eval,
            "verify": verify_eval,
        }
    }