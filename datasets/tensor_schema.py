import json
from copy import deepcopy


ROUTING_DATASET_TENSOR_SCHEMA = {
    "symbols": {
        "N": "num_nodes (river network nodes)",
        "E": "num_edges (graph edges)",
        "W": "windowsize (input time steps)",
        "P": "n_pred (prediction horizon)",
        "O": "outlet_nums (observed outlets)",
        "K": "khop for upstream mask",
    },
    "x": {
        "meaning": "Dynamic node features used as model input",
        "shape": ["N", "W", 2],
        "channels": [
            {
                "index": 0,
                "name": "runoff",
                "source": "OUT_RUNOFF",
                "normalization": "log1p(clamp>=0) + minmax",
            },
            {
                "index": 1,
                "name": "baseflow",
                "source": "OUT_BASEFLOW",
                "normalization": "log1p(clamp>=0) + minmax",
            },
        ],
    },
    "node_attr": {
        "meaning": "Static node features",
        "shape": ["N", 3],
        "channels": [
            {"index": 0, "name": "elev", "normalization": "minmax"},
            {"index": 1, "name": "slope", "normalization": "minmax"},
            {"index": 2, "name": "flow_distance", "normalization": "minmax"},
        ],
    },
    "y": {
        "meaning": "Observed outlet streamflow targets",
        "shape": ["P", "O"],
        "channels": [
            {
                "name": "streamflow_obs",
                "source": "streamflow",
                "normalization": "per-outlet log1p(clamp>=0) + minmax-11",
            }
        ],
    },
    "edge_index": {
        "meaning": "Directed graph connectivity (source, target)",
        "shape": [2, "E"],
        "dtype": "torch.long",
    },
    "edge_attr": {
        "meaning": "Per-edge static features",
        "shape": ["E", 4],
        "channels": [
            {"index": 0, "name": "elev_diff", "normalization": "minmax"},
            {"index": 1, "name": "slope_mean", "normalization": "minmax"},
            {"index": 2, "name": "flow_distance_mean", "normalization": "minmax"},
            {"index": 3, "name": "flow_angle_diff", "normalization": "minmax"},
        ],
    },
    "outlet_index": {
        "meaning": "Node indices corresponding to outlet stations",
        "shape": ["O"],
        "dtype": "torch.long",
    },
    "mask_downstream_adj": {
        "meaning": "Downstream reachability mask",
        "shape": ["N", "N"],
    },
    "mask_khop_up_adj": {
        "meaning": "k-hop upstream reachability mask",
        "shape": ["N", "N"],
        "depends_on": "K",
    },
    "full_path_edge_attr_adj": {
        "meaning": "Accumulated edge-attribute matrices along downstream paths",
        "shape": ["N", "N", 4],
        "channels": [
            {"index": 0, "name": "elev_diff_acc"},
            {"index": 1, "name": "slope_mean_acc"},
            {"index": 2, "name": "flow_distance_acc"},
            {"index": 3, "name": "flow_angle_acc"},
        ],
        "normalization": "log1p(clamp>=0) + minmax (applied to each channel matrix)",
    },
}


def get_routing_dataset_tensor_schema(runtime_dims=None):
    schema = deepcopy(ROUTING_DATASET_TENSOR_SCHEMA)
    if runtime_dims:
        schema["runtime_dims"] = runtime_dims
    return schema


def format_routing_dataset_tensor_schema(runtime_dims=None):
    schema = get_routing_dataset_tensor_schema(runtime_dims=runtime_dims)
    return json.dumps(schema, ensure_ascii=False, indent=2)

