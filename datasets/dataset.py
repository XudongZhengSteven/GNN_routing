# code: utf-8
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import torch
from netCDF4 import Dataset as ncDataset
from netCDF4 import chartostring
from torch.utils.data import WeightedRandomSampler

if __package__ is None or __package__ == "":
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _CURRENT_DIR not in sys.path:
        sys.path.insert(0, _CURRENT_DIR)

    from compat import Data, InMemoryDataset, to_dense_adj
    from common import (
        as_numpy,
        build_time_slice,
        dataset_type_to_split,
        decode_time,
        infer_scenario,
        load_yaml_cfg,
        resolve_file,
    )
    from feature_builder import (
        build_adjs,
        build_dynamic_features,
        build_edge_features,
        build_full_path_edge_attr_adj,
        build_khop_upstream_edges,
        build_node_features,
        build_static_features,
        normalize_feature,
    )
    from river_network import (
        build_rivernetwork_for_threshold,
        find_rivernetwork_graph_file,
        get_rivernetwork_threshold,
        normalize_threshold_value,
        resolve_rivernetwork_graph_path,
    )
    from tensor_schema import (
        ROUTING_DATASET_TENSOR_SCHEMA,
        format_routing_dataset_tensor_schema,
        get_routing_dataset_tensor_schema,
    )
else:
    from .compat import Data, InMemoryDataset, to_dense_adj
    from .common import (
        as_numpy,
        build_time_slice,
        dataset_type_to_split,
        decode_time,
        infer_scenario,
        load_yaml_cfg,
        resolve_file,
    )
    from .feature_builder import (
        build_adjs,
        build_dynamic_features,
        build_edge_features,
        build_full_path_edge_attr_adj,
        build_khop_upstream_edges,
        build_node_features,
        build_static_features,
        normalize_feature,
    )
    from .river_network import (
        build_rivernetwork_for_threshold,
        find_rivernetwork_graph_file,
        get_rivernetwork_threshold,
        normalize_threshold_value,
        resolve_rivernetwork_graph_path,
    )
    from .tensor_schema import (
        ROUTING_DATASET_TENSOR_SCHEMA,
        format_routing_dataset_tensor_schema,
        get_routing_dataset_tensor_schema,
    )


class RoutingDataset(InMemoryDataset):
    TENSOR_SCHEMA = ROUTING_DATASET_TENSOR_SCHEMA

    def __init__(
        self,
        windowsize: int = 16,
        input_freq_per_day: int = 1,
        n_pred: int = 1,
        khop: int = 3,
        normalizers: Optional[Dict] = None,
        predict_current: int = 1,
        dataset_type: str = "train",
        self_loop: bool = False,
        data_home: Optional[str] = None,
        data_cfg_path: Optional[str] = None,
        model_cfg_path: Optional[str] = None,
        scenario: Optional[str] = None,
    ):
        try:
            super().__init__(root=None)
        except TypeError:
            super().__init__()

        if windowsize <= 0:
            raise ValueError("`windowsize` must be positive.")
        if input_freq_per_day <= 0:
            raise ValueError("`input_freq_per_day` must be positive.")
        if n_pred <= 0:
            raise ValueError("`n_pred` must be positive.")

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._data_home = data_home or os.path.join(root_dir, "data", "raw", "case5")
        self._data_cfg_path = data_cfg_path or os.path.join(root_dir, "configs", "data.yaml")
        self._model_cfg_path = model_cfg_path or os.path.join(root_dir, "configs", "model.yaml")

        self._self_loop = bool(self_loop)
        self._windowsize = int(windowsize)
        self._input_freq_per_day = int(input_freq_per_day)
        self._windowsize_in_days = max(1, int((self._windowsize + self._input_freq_per_day - 1) / self._input_freq_per_day))
        self._n_pred = int(n_pred)
        self._khop = int(khop)
        self._dataset_type = dataset_type.lower()

        # [x1, ..., xt] -> [y_t]  : predict_current = 1
        # [x1, ..., xt] -> [y_t+1]: predict_current = 0
        self.predict_current = int(predict_current)

        self._scenario = (scenario or self._infer_scenario(self._dataset_type)).lower()
        self.normalizers = {} if normalizers is None else normalizers

        self._time_cfg = self._load_time_cfg(self._data_cfg_path)
        self._model_cfg = self._load_model_cfg(self._model_cfg_path)

        self.load()
        self._time_slice_obs, self._time_slice_input = self.getTimeIndex()
        self.preprocess(self.normalizers)

    @staticmethod
    def _infer_scenario(dataset_type: str) -> str:
        return infer_scenario(dataset_type)

    @staticmethod
    def _as_numpy(x):
        return as_numpy(x)

    @staticmethod
    def _decode_time(time_var):
        return decode_time(time_var)

    @staticmethod
    def _load_time_cfg(path: str) -> Optional[Dict]:
        return load_yaml_cfg(path)

    @staticmethod
    def _load_model_cfg(path: str) -> Optional[Dict]:
        return load_yaml_cfg(path)

    @staticmethod
    def _normalize_threshold_value(value) -> Optional[str]:
        return normalize_threshold_value(value)

    def _get_rivernetwork_threshold(self) -> Optional[str]:
        return get_rivernetwork_threshold(self._model_cfg)

    def _find_rivernetwork_graph_file(self, threshold: str) -> Optional[str]:
        return find_rivernetwork_graph_file(self._data_home, threshold)

    def _build_rivernetwork_for_threshold(self, threshold: str):
        return build_rivernetwork_for_threshold(self._data_home, threshold)

    def _resolve_rivernetwork_graph_path(self) -> str:
        return resolve_rivernetwork_graph_path(
            data_home=self._data_home,
            model_cfg=self._model_cfg,
            resolve_file_fn=self._resolve_file,
            build_fn=self._build_rivernetwork_for_threshold,
        )

    def _resolve_file(self, patterns: List[str]) -> str:
        return resolve_file(self._data_home, patterns)

    @staticmethod
    def _dataset_type_to_split(dataset_type: str) -> str:
        return dataset_type_to_split(dataset_type)

    def _build_time_slice(self, date_index, split_name: str) -> slice:
        return build_time_slice(date_index, split_name, self._time_cfg)

    def getTimeIndex(self) -> Tuple[slice, slice]:
        split_name = self._dataset_type_to_split(self._dataset_type)
        slice_obs = self._build_time_slice(self.obs_time_index, split_name)
        slice_input = self._build_time_slice(self.input_time_index, split_name)
        return slice_obs, slice_input

    def load(self):
        if not os.path.isdir(self._data_home):
            raise FileNotFoundError(f"Data directory does not exist: {self._data_home}")

        graph_path = self._resolve_rivernetwork_graph_path()
        self._graph_path = graph_path
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)

        if self._self_loop:
            self.G.add_edges_from([(n, n) for n in self.G.nodes() if not self.G.has_edge(n, n)])

        self.node_id_map = {n: i for i, n in enumerate(self.G.nodes)}
        self.id_node_map = {v: k for k, v in self.node_id_map.items()}
        self.node_pos_map = {n: self.G.nodes[n]["matrix_pos"] for n in self.G.nodes}
        self.num_nodes = len(self.G.nodes)

        edge_list = [[self.node_id_map[u], self.node_id_map[v]] for u, v in self.G.edges]
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.adj = to_dense_adj(self.edge_index, max_num_nodes=self.num_nodes).squeeze(0)

        flux_path = self._resolve_file(
            [
                f"fluxes.2003-01-01_{self._scenario}.nc",
                "fluxes.2003-01-01_total.nc",
                "fluxes.2003-01-01_cali0.nc",
                "fluxes*.nc",
            ]
        )
        with ncDataset(flux_path, "r") as ds:
            self.runoff = self._as_numpy(ds.variables["OUT_RUNOFF"][:, :, :])
            self.baseflow = self._as_numpy(ds.variables["OUT_BASEFLOW"][:, :, :])
            self.lon = self._as_numpy(ds.variables["lon"][:])
            self.lat = self._as_numpy(ds.variables["lat"][:])
            self.input_time_index = self._decode_time(ds.variables["time"])

        params_path = self._resolve_file(
            [
                f"params_level1_{self._scenario}.nc",
                "params_level1.nc",
                "params_level1_cali0.nc",
                "params_level1*.nc",
            ]
        )
        with ncDataset(params_path, "r") as ds:
            self.elev = self._as_numpy(ds.variables["elev"][:, :])
            self.slope = self._as_numpy(ds.variables["slope"][:, :])

        flow_path = self._resolve_file(["flow_direction_file.nc", "*flow*direction*.nc"])
        with ncDataset(flow_path, "r") as ds:
            self.flow_distance = self._as_numpy(ds.variables["Flow_Distance"][:, :])
            self.flow_direction = self._as_numpy(ds.variables["Flow_Direction"][:, :])

        rvic_path = self._resolve_file([f"*rvic*{self._scenario}.nc", "*rvic*cali0.nc", "*rvic*.nc"])
        with ncDataset(rvic_path, "r") as ds:
            streamflow = self._as_numpy(ds.variables["streamflow"][:, :])
            outlet_x_ind = self._as_numpy(ds.variables["outlet_x_ind"][:]).astype(int)
            outlet_y_ind = self._as_numpy(ds.variables["outlet_y_ind"][:]).astype(int)
            outlet_name_raw = ds.variables["outlet_name"][:]
            self.obs_time_index = self._decode_time(ds.variables["time"])

        try:
            outlet_names = [str(x).replace("\x00", "").strip() for x in chartostring(outlet_name_raw)]
        except Exception:
            outlet_names = [f"outlet_{i}" for i in range(streamflow.shape[1])]

        valid_cols: List[int] = []
        self.outlet_nodes: List[str] = []
        self.outlet_indices: List[int] = []
        filtered_names: List[str] = []

        for i, (x_idx, y_idx, name) in enumerate(zip(outlet_x_ind, outlet_y_ind, outlet_names)):
            node_name = f"cell_{int(y_idx)}_{int(x_idx)}"
            if node_name in self.node_id_map:
                valid_cols.append(i)
                self.outlet_nodes.append(node_name)
                self.outlet_indices.append(self.node_id_map[node_name])
                filtered_names.append(name)

        if not valid_cols:
            raise ValueError("No RVIC outlet can be mapped to graph nodes.")

        self.streamflow_obs = streamflow[:, valid_cols]
        self.outlet_names = filtered_names
        self.outlet_nums = len(self.outlet_names)

    def preprocess(self, normalizers: Optional[Dict] = None):
        self.normalizers = {} if normalizers is None else normalizers

        self.build_dynamic_features()
        self.build_static_features()
        self.build_node_features()
        self.normed_dynamic = torch.stack([self.normed_runoff_nodes, self.normed_baseflow_nodes], dim=2)

        self.build_edge_features()
        self.build_adjs()

    def build_dynamic_features(self):
        build_dynamic_features(self)

    def build_static_features(self):
        build_static_features(self)

    def build_node_features(self):
        build_node_features(self)

    def build_edge_features(self):
        build_edge_features(self)

    @staticmethod
    def build_khop_upstream_edges(G, elev, slope, flow_distance, flow_angle, node_id_map, k: int = 2):
        return build_khop_upstream_edges(G, elev, slope, flow_distance, flow_angle, node_id_map, k)

    def build_adjs(self):
        build_adjs(self)

    @staticmethod
    def build_full_path_edge_attr_adj(
        edge_mat,
        reach_down,
        accumulate_fn=None,
        self_loop_fn=None,
        use_count_for_mean: bool = False,
        self_loop: bool = False,
    ):
        return build_full_path_edge_attr_adj(
            edge_mat=edge_mat,
            reach_down=reach_down,
            accumulate_fn=accumulate_fn,
            self_loop_fn=self_loop_fn,
            use_count_for_mean=use_count_for_mean,
            self_loop=self_loop,
        )

    @staticmethod
    def normalize(data, normalizer=None, use_log: bool = False, method: str = "zscore"):
        return normalize_feature(data, normalizer=normalizer, use_log=use_log, method=method)

    @staticmethod
    def inverse_transform_streamflow(x, normalizer):
        return normalizer.inverse_transform(x)

    def inverse_transform_streamflow_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform streamflow tensor using per-outlet normalizers.
        Expected shape: [..., O], where O == outlet_nums.
        """
        if "streamflow" not in self.normalizers:
            raise ValueError("`streamflow` normalizers not found. Build train dataset first.")

        normalizers = self.normalizers["streamflow"]
        if not isinstance(normalizers, (list, tuple)) or len(normalizers) == 0:
            raise ValueError("Invalid `streamflow` normalizers.")

        x_t = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
        if x_t.shape[-1] != len(normalizers):
            raise ValueError(
                "Last dim of streamflow tensor must match number of outlet normalizers: "
                f"x.shape={tuple(x_t.shape)}, normalizers={len(normalizers)}"
            )

        x_cpu = x_t.detach().to(torch.float32).cpu()
        y_cpu = torch.empty_like(x_cpu)
        for i, normalizer in enumerate(normalizers):
            y_cpu[..., i] = normalizer.inverse_transform(x_cpu[..., i])

        return y_cpu.to(device=x_t.device, dtype=x_t.dtype)

    def len(self):
        return self.num_samples

    def get(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index out of range: {idx} not in [0, {self.num_samples})")

        x = self.normed_dynamic[:, idx : idx + self._windowsize, :]

        y_start = idx + self._windowsize_in_days - self.predict_current
        y_end = y_start + self._n_pred
        y = self.normed_streamflow_obs[y_start:y_end, :]

        data = Data(
            x=x,
            y=y,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_attr=self.node_attr,
            outlet_index=torch.tensor(self.outlet_indices, dtype=torch.long),
            mask_downstream_adj=self.mask_downstream_adj,
            mask_khop_up_adj=self.mask_khop_up_adj,
            full_path_edge_attr_adj=self.full_path_edge_attr_adj,
        )
        return data

    def get_balance_weight(self):
        start = self._windowsize_in_days - self.predict_current
        end = start + self.num_samples
        targets = self.normed_streamflow_obs[start:end, :]

        scores = targets.mean(dim=1)
        hist, bin_edges = torch.histogram(scores, bins=10)
        hist = hist.float()
        hist[hist == 0] = 1.0

        bin_indices = torch.bucketize(scores, bin_edges[1:-1], right=False)
        weights = 1.0 / hist[bin_indices]
        weights = weights / (weights.sum() + 1e-12)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return sampler, weights

    def get_tensor_schema(self, include_runtime_dims: bool = True):
        runtime_dims = None
        if include_runtime_dims:
            runtime_dims = {
                "N": getattr(self, "num_nodes", None),
                "E": int(self.edge_index.shape[1]) if hasattr(self, "edge_index") else None,
                "W": int(self._windowsize),
                "P": int(self._n_pred),
                "O": getattr(self, "outlet_nums", None),
                "K": int(self._khop),
            }
        return get_routing_dataset_tensor_schema(runtime_dims=runtime_dims)

    def format_tensor_schema(self, include_runtime_dims: bool = True) -> str:
        runtime_dims = None
        if include_runtime_dims:
            runtime_dims = {
                "N": getattr(self, "num_nodes", None),
                "E": int(self.edge_index.shape[1]) if hasattr(self, "edge_index") else None,
                "W": int(self._windowsize),
                "P": int(self._n_pred),
                "O": getattr(self, "outlet_nums", None),
                "K": int(self._khop),
            }
        return format_routing_dataset_tensor_schema(runtime_dims=runtime_dims)

    def print_tensor_schema(self, include_runtime_dims: bool = True):
        print(self.format_tensor_schema(include_runtime_dims=include_runtime_dims))

    @staticmethod
    def _shape_list(x):
        if hasattr(x, "shape"):
            return [int(v) for v in x.shape]
        return None

    def get_runtime_schema(self, sample_idx: int = 0, include_template: bool = False):
        if self.num_samples <= 0:
            raise ValueError("Dataset has no samples; runtime schema cannot be built.")

        sample_idx = max(0, min(int(sample_idx), self.num_samples - 1))
        sample = self.get(sample_idx)

        runtime_schema = {
            "dataset": {
                "dataset_type": self._dataset_type,
                "scenario": self._scenario,
                "num_samples": int(self.num_samples),
                "sample_index": int(sample_idx),
            },
            "config": {
                "windowsize": int(self._windowsize),
                "input_freq_per_day": int(self._input_freq_per_day),
                "n_pred": int(self._n_pred),
                "khop": int(self._khop),
                "predict_current": int(self.predict_current),
                "self_loop": bool(self._self_loop),
            },
            "paths": {
                "data_home": self._data_home,
                "data_cfg_path": self._data_cfg_path,
                "model_cfg_path": self._model_cfg_path,
                "graph_path": getattr(self, "_graph_path", None),
            },
            "runtime_dims": {
                "N": int(self.num_nodes),
                "E": int(self.edge_index.shape[1]),
                "W": int(self._windowsize),
                "P": int(self._n_pred),
                "O": int(self.outlet_nums),
                "K": int(self._khop),
            },
            "features": {
                "x_channels": [c["name"] for c in self.TENSOR_SCHEMA["x"]["channels"]],
                "node_attr_channels": [c["name"] for c in self.TENSOR_SCHEMA["node_attr"]["channels"]],
                "y_channels": [c["name"] for c in self.TENSOR_SCHEMA["y"]["channels"]],
                "edge_attr_channels": [c["name"] for c in self.TENSOR_SCHEMA["edge_attr"]["channels"]],
            },
            "tensor_shapes": {
                "x": self._shape_list(sample.x),
                "node_attr": self._shape_list(sample.node_attr),
                "y": self._shape_list(sample.y),
                "edge_index": self._shape_list(sample.edge_index),
                "edge_attr": self._shape_list(sample.edge_attr),
                "outlet_index": self._shape_list(sample.outlet_index),
                "mask_downstream_adj": self._shape_list(sample.mask_downstream_adj),
                "mask_khop_up_adj": self._shape_list(sample.mask_khop_up_adj),
                "full_path_edge_attr_adj": self._shape_list(sample.full_path_edge_attr_adj),
            },
            "outlet_names_head": self.outlet_names[: min(5, len(self.outlet_names))],
        }

        if include_template:
            runtime_schema["template"] = self.get_tensor_schema(include_runtime_dims=False)

        return runtime_schema

    def format_runtime_schema(self, sample_idx: int = 0, include_template: bool = False) -> str:
        schema = self.get_runtime_schema(sample_idx=sample_idx, include_template=include_template)
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def print_runtime_schema(self, sample_idx: int = 0, include_template: bool = False):
        print(self.format_runtime_schema(sample_idx=sample_idx, include_template=include_template))
