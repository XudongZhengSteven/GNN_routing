# RoutingDataset Tensor Schema

Symbols:
- `N`: number of graph nodes
- `E`: number of graph edges
- `W`: input window size (time steps)
- `P`: prediction horizon (`n_pred`)
- `O`: number of outlets
- `K`: upstream hop count (`khop`)

## `x`
- Meaning: dynamic node input features
- Shape: `[N, W, 2]`
- Channels:
1. `runoff` (from `OUT_RUNOFF`, normalized by `log1p(clamp>=0) + minmax`)
2. `baseflow` (from `OUT_BASEFLOW`, normalized by `log1p(clamp>=0) + minmax`)

## `node_attr`
- Meaning: static node features
- Shape: `[N, 3]`
- Channels:
1. `elev` (`minmax`)
2. `slope` (`minmax`)
3. `flow_distance` (`minmax`)

## `y`
- Meaning: outlet streamflow targets
- Shape: `[P, O]`
- Channel:
1. `streamflow_obs` (from `streamflow`, normalized per outlet by `log1p(clamp>=0) + minmax-11`)

## `edge_index`
- Meaning: directed edge list
- Shape: `[2, E]`

## `edge_attr`
- Meaning: edge static features
- Shape: `[E, 4]`
- Channels:
1. `elev_diff`
2. `slope_mean`
3. `flow_distance_mean`
4. `flow_angle_diff`

## `outlet_index`
- Meaning: outlet node IDs in graph index space
- Shape: `[O]`

## `mask_downstream_adj`
- Meaning: downstream reachability mask
- Shape: `[N, N]`

## `mask_khop_up_adj`
- Meaning: `K`-hop upstream reachability mask
- Shape: `[N, N]`

## `full_path_edge_attr_adj`
- Meaning: accumulated edge-attribute matrices over downstream paths
- Shape: `[N, N, 4]`
- Channels:
1. `elev_diff_acc`
2. `slope_mean_acc`
3. `flow_distance_acc`
4. `flow_angle_acc`
- Normalization: `log1p(clamp>=0) + minmax` for each channel matrix

## Runtime View
You can print the same schema with runtime dimensions from Python:

```python
from datasets.dataset import RoutingDataset

ds = RoutingDataset(dataset_type="train")
ds.print_tensor_schema()
```

You can also print a runtime-focused schema for the currently built dataset instance:

```python
from datasets.dataset import RoutingDataset

ds = RoutingDataset(dataset_type="train")
ds.print_runtime_schema()  # includes current tensor shapes and feature channels
```
