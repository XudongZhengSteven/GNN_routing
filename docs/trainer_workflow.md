# Trainer Workflow

## 1. Configure
Edit:
- `configs/model.yaml` (graph threshold + model architecture params under `model:`)
- `configs/data.yaml` (time split)
- `configs/train.yaml` (dataset/dataloader/loss/optimizer/scheduler/trainer; `model` is optional override)

## 2. Train
```bash
python scripts/train.py --train-cfg configs/train.yaml --data-cfg configs/data.yaml --model-cfg configs/model.yaml
```

Outputs are saved under `trainer.checkpoint_dir`:
- `best.ckpt`
- `last.ckpt`
- `epoch_XXXX.ckpt` (keep last K)
- `history.csv`
- `summary.json`
- `resolved_config.json`

## 3. Resume
```bash
python scripts/train.py --train-cfg configs/train.yaml --resume checkpoints/routing_baseline/last.ckpt
```

## 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/routing_baseline/best.ckpt --split val
python scripts/test.py --checkpoint checkpoints/routing_baseline/best.ckpt
```

`scripts/evaluate.py` now reports both:
- `metrics_norm`: metrics in normalized space
- `metrics_denorm`: metrics after inverse-transform back to physical streamflow units

It also supports full-period simulation + KGE + plotting:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/routing_baseline/best.ckpt \
  --splits calibrate,verify \
  --save-dir checkpoints/routing_baseline/manual_eval \
  --plot
```

Saved outputs for each split:
- `simulation_<split>.csv` (full-period simulation: obs/pred/error by outlet)
- `metrics_<split>.json` (overall + per-outlet metrics, including KGE)
- `kge_bar_<split>.png`
- `timeseries_<split>_<outlet>.png`

## 5. Post-Train Simulation
`configs/train.yaml -> trainer` supports automatic simulation after training:
- `post_train_simulate: true`
- `post_train_splits: [\"calibrate\", \"verify\"]`
- `post_train_plot: true`
- `post_train_batch_size: 128`
- `post_train_save_dir: \"post_train_simulation\"`

When enabled, `scripts/train.py` automatically calls `scripts/evaluate.py` using `best.ckpt` and saves results under:
- `<checkpoint_dir>/<post_train_save_dir>/<split>/...`
