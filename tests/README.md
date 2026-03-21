# Tests

## Dataset Build Tests

Run:

```bash
python -m unittest tests.test_dataset_build -v
python -m unittest tests.test_dataset_build_real -v
python -m unittest tests.test_trainer_smoke -v
python -m unittest tests.test_model_gr2n_forward -v
```

Both tests print:
- `RoutingDataset/runtime schema` (current dataset content, feature channels, tensor shapes)
- one sample summary (`x/y/edge/node_attr` shapes and outlet names)

If you use VSCode Test Explorer, check `Output -> Python Test Log` to view printed schema.
