# Experiment Runner

## Goal
Run model-variant/hyperparameter experiments with:
- separate checkpoint folders per run
- two objective functions (`mean_kge`, `single_station_kge`)
- multiple seeds for initialization uncertainty
- automatic post-train full-period simulation/plotting (`calibrate`, `verify`)
- final comparison tables

## Config
Recommended phased workflow:
- Phase-1 (architecture selection): `configs/experiments/phase1_architecture.yaml`
- Phase-2 (hyperparameter analysis): `configs/experiments/phase2_hyperparams.yaml`
- Phase-3 (objective-function comparison): `configs/experiments/phase3_objective.yaml`

Default config `configs/experiment.yaml` is Phase-1.

## Run
Dry-run (only generate plan/manifests):

```bash
python scripts/run_experiments.py --config configs/experiments/phase1_architecture.yaml --dry-run
```

Actual run:

```bash
python scripts/run_experiments.py --config configs/experiments/phase1_architecture.yaml
python scripts/run_experiments.py --config configs/experiments/phase2_hyperparams.yaml
python scripts/run_experiments.py --config configs/experiments/phase3_objective.yaml
```

Optional:

```bash
python scripts/run_experiments.py --config configs/experiments/phase2_hyperparams.yaml --mode full
python scripts/run_experiments.py --config configs/experiments/phase2_hyperparams.yaml --max-runs 20
```

## Terminal Watch
Real-time terminal dashboard (progress/state/top results):

```bash
python scripts/watch_experiment.py --config configs/experiments/phase1_architecture.yaml
```

One-time snapshot:

```bash
python scripts/watch_experiment.py --config configs/experiments/phase1_architecture.yaml --once
```

Show recent epoch trend (example: last 8 points of `val_kge_denorm`):

```bash
python scripts/watch_experiment.py --config configs/experiments/phase1_architecture.yaml --history-tail 8 --history-metric val_kge_denorm
```

## Outputs
All outputs are under:

`checkpoints/experiments/<experiment_name>/`

Important files:
- `design/settings_manifest.csv`: all generated settings
- `results/run_level_metrics.csv`: per-run metrics (per seed)
- `results/setting_aggregate_metrics.csv`: seed-mean/std by setting
- `results/leaderboard_*.csv`: per-objective leaderboard
- `results/experiment_report.md`: summary report
