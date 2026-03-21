# GR2N Module Mapping (for current RoutingDataset)

This project now includes a dataset-aligned implementation of the six reference model files:

- `models/spatial_model.py`: `GraphConvPosEnc`
- `models/temporal_model.py`: `GraphGRUCell`, `GraphGRUBlock`
- `models/encoder_decoder.py`: `GraphGRUEncoder`, `GraphGRUDecoder`
- `models/postional_encoding.py`: `SpatialPositionalEncoding`
- `models/slope_routing_model.py`: `SlopeRouting` (+ FiLM-GRU helpers)
- `models/routing_model.py`: `GR2N`, `GR2NSeq2Seq`

## Current Dataset Interface

Input fields from `RoutingDataset` sample:

- `x`: `[N, W, F_dyn]` or `[B, N, W, F_dyn]`
- `node_attr`: `[N, D_s]` or `[B, N, D_s]`
- `outlet_index`: `[O]` or `[B, O]`
- `mask_downstream_adj`: `[N, N]` or `[B, N, N]`
- `mask_khop_up_adj`: `[N, N]` or `[B, N, N]`
- `full_path_edge_attr_adj`: `[N, N, F_e]` or `[B, N, N, F_e]`

Output target:

- `y`: `[P, O]` or `[B, P, O]`

Both `GR2N` and `GR2NSeq2Seq` output shape aligned to `y`.

## Build Model

Use `models.build_model()` with:

- `name: gr2n`
- or `name: gr2n_seq2seq`

`build_model()` auto-infers `num_nodes`, `seq_len`, `pred_len`, `edge_attr_dim`, and feature dims from the dataset when dataset is provided.

