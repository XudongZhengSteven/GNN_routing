# Notebooks

## 1. Train + Monitor
- Notebook: `01_train_and_monitor.ipynb`
- Purpose:
1. Build `RoutingDataset` train/val loaders
2. Train model (default from `configs/model.yaml`)
3. Show training curves (`history.csv`)
4. Show normalized and de-normalized validation metrics
5. Plot outlet time series

## 2. Result Showcase
- Notebook: `02_result_showcase.ipynb`
- Purpose:
1. Load existing `best.ckpt`
2. Evaluate on test split
3. Show normalized and de-normalized metrics
4. Plot scatter / time series / error histogram per outlet

## Notes
- Both notebooks auto-detect project root and import from the repo.
- To run quickly, set `quick_run = True` in notebook 1.
- Default checkpoint path in notebook 2 is derived from `configs/train.yaml -> trainer.checkpoint_dir`.

