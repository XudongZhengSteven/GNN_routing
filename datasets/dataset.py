# code: utf-8
import glob
import os
import pickle
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from netCDF4 import Dataset as ncDataset
from netCDF4 import chartostring, num2date
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler
try:
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.utils import to_dense_adj
except ImportError:
    class InMemoryDataset(TorchDataset):
        def __len__(self):
            return self.len()

        def __getitem__(self, idx):
            return self.get(idx)

    class Data(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__ = self

    def to_dense_adj(edge_index, max_num_nodes=None):
        if max_num_nodes is None:
            max_num_nodes = int(edge_index.max().item()) + 1
        adj = torch.zeros((max_num_nodes, max_num_nodes), dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        return adj.unsqueeze(0)

try:
    from .normalizer import FeatureNormalizer
except ImportError:
    from normalizer import FeatureNormalizer


direction_to_angle = {
    1: 0.0,
    2: 45.0,
    4: 90.0,
    8: 135.0,
    16: 180.0,
    32: 225.0,
    64: 270.0,
    128: 315.0,
}


class RoutingDataset(InMemoryDataset):
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

        self._self_loop = bool(self_loop)
        self._windowsize = int(windowsize)
        self._input_freq_per_day = int(input_freq_per_day)
        self._windowsize_in_days = max(1, int(np.ceil(self._windowsize / self._input_freq_per_day)))
        self._n_pred = int(n_pred)
        self._khop = int(khop)
        self._dataset_type = dataset_type.lower()

        # [x1, ..., xt] -> [y_t]  : predict_current = 1
        # [x1, ..., xt] -> [y_t+1]: predict_current = 0
        self.predict_current = int(predict_current)

        self._scenario = (scenario or self._infer_scenario(self._dataset_type)).lower()
        self.normalizers = {} if normalizers is None else normalizers

        self._time_cfg = self._load_time_cfg(self._data_cfg_path)

        self.load()
        self._time_slice_obs, self._time_slice_input = self.getTimeIndex()
        self.preprocess(self.normalizers)

    @staticmethod
    def _infer_scenario(dataset_type: str) -> str:
        if dataset_type in {"train", "warmup", "calibrate"}:
            return "cali0"
        if dataset_type in {"val", "test", "verify"}:
            return "vali0"
        return "cali0"

    @staticmethod
    def _as_numpy(x):
        if np.ma.isMaskedArray(x):
            return x.filled(np.nan)
        return np.asarray(x)

    @staticmethod
    def _decode_time(time_var) -> pd.DatetimeIndex:
        values = time_var[:]
        units = getattr(time_var, "units", None)
        if units is None:
            return pd.date_range("1970-01-01", periods=len(values), freq="D")

        calendar = getattr(time_var, "calendar", "standard")
        dt = num2date(values, units=units, calendar=calendar)
        # Use YYYY-MM-DD for robust conversion from cftime objects.
        date_text = [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dt]
        return pd.to_datetime(date_text)

    @staticmethod
    def _load_time_cfg(path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _resolve_file(self, patterns: List[str]) -> str:
        for pattern in patterns:
            full_pattern = os.path.join(self._data_home, pattern)
            if any(token in pattern for token in ["*", "?", "["]):
                matches = sorted(glob.glob(full_pattern))
                if matches:
                    return matches[0]
            elif os.path.exists(full_pattern):
                return full_pattern

        joined = " | ".join(patterns)
        raise FileNotFoundError(f"No file matched in `{self._data_home}`: {joined}")

    @staticmethod
    def _dataset_type_to_split(dataset_type: str) -> str:
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

    def _build_time_slice(self, date_index: pd.DatetimeIndex, split_name: str) -> slice:
        if split_name == "full" or self._time_cfg is None:
            return slice(0, len(date_index))

        split_cfg = self._time_cfg.get("time", {}).get("split", {}).get(split_name)
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

    def getTimeIndex(self) -> Tuple[slice, slice]:
        split_name = self._dataset_type_to_split(self._dataset_type)
        slice_obs = self._build_time_slice(self.obs_time_index, split_name)
        slice_input = self._build_time_slice(self.input_time_index, split_name)
        return slice_obs, slice_input

    def load(self):
        if not os.path.isdir(self._data_home):
            raise FileNotFoundError(f"Data directory does not exist: {self._data_home}")

        # load graph topo: nodes, edges
        graph_path = self._resolve_file(["river_network_graph_connected.pkl", "river_network_graph*.pkl"])
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

        # load dynamic inputs: runoff, baseflow
        flux_path = self._resolve_file(
            [
                f"fluxes.2003-01-01_{self._scenario}.nc",
                "fluxes.2003-01-01_total.nc",
                "fluxes.2003-01-01_cali0.nc",
                "fluxes*.nc",
            ]
        )

        with ncDataset(flux_path, "r") as ds:
            runoff = self._as_numpy(ds.variables["OUT_RUNOFF"][:, :, :])
            baseflow = self._as_numpy(ds.variables["OUT_BASEFLOW"][:, :, :])
            lon = self._as_numpy(ds.variables["lon"][:])
            lat = self._as_numpy(ds.variables["lat"][:])
            input_time_index = self._decode_time(ds.variables["time"])

        self.runoff = runoff
        self.baseflow = baseflow
        self.lon = lon
        self.lat = lat
        self.input_time_index = input_time_index

        # load static features: elev, slope
        params_path = self._resolve_file(
            [
                f"params_level1_{self._scenario}.nc",
                "params_level1.nc",
                "params_level1_cali0.nc",
                "params_level1*.nc",
            ]
        )
        with ncDataset(params_path, "r") as ds:
            elev = self._as_numpy(ds.variables["elev"][:, :])
            slope = self._as_numpy(ds.variables["slope"][:, :])

        self.elev = elev
        self.slope = slope

        # load flow geometry
        flow_path = self._resolve_file(["flow_direction_file.nc", "*flow*direction*.nc"])
        with ncDataset(flow_path, "r") as ds:
            flow_distance = self._as_numpy(ds.variables["Flow_Distance"][:, :])
            flow_direction = self._as_numpy(ds.variables["Flow_Direction"][:, :])

        self.flow_distance = flow_distance
        self.flow_direction = flow_direction

        # load target streamflow and outlets
        rvic_path = self._resolve_file([f"*rvic*{self._scenario}.nc", "*rvic*cali0.nc", "*rvic*.nc"])
        with ncDataset(rvic_path, "r") as ds:
            streamflow = self._as_numpy(ds.variables["streamflow"][:, :])
            outlet_x_ind = self._as_numpy(ds.variables["outlet_x_ind"][:]).astype(int)
            outlet_y_ind = self._as_numpy(ds.variables["outlet_y_ind"][:]).astype(int)
            outlet_name_raw = ds.variables["outlet_name"][:]
            obs_time_index = self._decode_time(ds.variables["time"])

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
        self.obs_time_index = obs_time_index

    def preprocess(self, normalizers: Optional[Dict] = None):
        self.normalizers = {} if normalizers is None else normalizers

        # build dynamic features
        self.build_dynamic_features()

        # build static features
        self.build_static_features()

        # build node features
        self.build_node_features()
        self.normed_dynamic = torch.stack(
            [
                self.normed_runoff_nodes,
                self.normed_baseflow_nodes,
            ],
            dim=2,
        )

        # build edge features
        self.build_edge_features()

        # build adjacency-based masks and path features
        self.build_adjs()

    def build_dynamic_features(self):
        self.runoff = torch.tensor(self.runoff[self._time_slice_input, :, :], dtype=torch.float32)
        self.baseflow = torch.tensor(self.baseflow[self._time_slice_input, :, :], dtype=torch.float32)
        self.streamflow_obs = torch.tensor(self.streamflow_obs[self._time_slice_obs, :], dtype=torch.float32)

        num_input_steps = self.runoff.shape[0]
        num_obs_steps = self.streamflow_obs.shape[0]

        num_samples_input = num_input_steps - self._windowsize + 1
        num_samples_obs = num_obs_steps - self._windowsize_in_days - self._n_pred + 1 + self.predict_current
        self.num_samples = int(min(num_samples_input, num_samples_obs))
        if self.num_samples <= 0:
            raise ValueError(
                "No valid sample can be generated. "
                f"num_input={num_input_steps}, num_obs={num_obs_steps}, "
                f"windowsize={self._windowsize}, n_pred={self._n_pred}."
            )

        # normalization for basin-wide dynamic features
        self.normed_runoff, normalizer_runoff = self.normalize(
            self.runoff,
            normalizer=self.normalizers.get("runoff", None),
            use_log=True,
            method="minmax",
        )
        self.normed_baseflow, normalizer_baseflow = self.normalize(
            self.baseflow,
            normalizer=self.normalizers.get("baseflow", None),
            use_log=True,
            method="minmax",
        )

        normed_streamflow_obs_list = []
        normalizer_streamflow_obs = []
        existing_streamflow_normalizers = self.normalizers.get("streamflow", None)

        for i in range(self.streamflow_obs.shape[1]):
            obs_i = self.streamflow_obs[:, i]

            normalizer_i = None
            if isinstance(existing_streamflow_normalizers, (list, tuple)) and i < len(existing_streamflow_normalizers):
                normalizer_i = existing_streamflow_normalizers[i]

            normed_obs, normalizer_streamflow = self.normalize(
                obs_i,
                normalizer=normalizer_i,
                use_log=True,
                method="minmax-11",
            )

            normed_streamflow_obs_list.append(normed_obs)
            normalizer_streamflow_obs.append(normalizer_streamflow)

        self.normed_streamflow_obs = torch.stack(normed_streamflow_obs_list, dim=1)

        # save fitted normalizers
        self.normalizers["runoff"] = normalizer_runoff
        self.normalizers["baseflow"] = normalizer_baseflow
        self.normalizers["streamflow"] = normalizer_streamflow_obs

    def build_static_features(self):
        self.elev = torch.tensor(self.elev[:, :], dtype=torch.float32)
        self.slope = torch.tensor(self.slope[:, :], dtype=torch.float32)
        self.flow_distance = torch.tensor(self.flow_distance[:, :], dtype=torch.float32)
        self.flow_direction = torch.tensor(self.flow_direction[:, :], dtype=torch.long)

        # flow_direction -> angle
        self.flow_angle = torch.zeros_like(self.flow_direction, dtype=torch.float32)
        for direction, angle in direction_to_angle.items():
            self.flow_angle = torch.where(
                self.flow_direction == int(direction),
                torch.tensor(float(angle), dtype=torch.float32),
                self.flow_angle,
            )

        self.normed_elev, _ = self.normalize(self.elev, normalizer=None, use_log=False, method="minmax")
        self.normed_slope, _ = self.normalize(self.slope, normalizer=None, use_log=False, method="minmax")
        self.normed_flow_distance, _ = self.normalize(
            self.flow_distance,
            normalizer=None,
            use_log=False,
            method="minmax",
        )
        self.normed_flow_angle, _ = self.normalize(self.flow_angle, normalizer=None, use_log=False, method="minmax")

    def build_node_features(self):
        normed_runoff_nodes = []
        normed_baseflow_nodes = []

        normed_elev_nodes = []
        normed_slope_nodes = []
        normed_flow_distance_nodes = []

        for node_name in self.G.nodes:
            i, j = self.G.nodes[node_name]["matrix_pos"]

            normed_runoff_nodes.append(self.normed_runoff[:, i, j])
            normed_baseflow_nodes.append(self.normed_baseflow[:, i, j])

            normed_elev_nodes.append(self.normed_elev[i, j])
            normed_slope_nodes.append(self.normed_slope[i, j])
            normed_flow_distance_nodes.append(self.normed_flow_distance[i, j])

        self.normed_runoff_nodes = torch.stack(normed_runoff_nodes, dim=0)  # [N, T]
        self.normed_baseflow_nodes = torch.stack(normed_baseflow_nodes, dim=0)  # [N, T]

        self.node_attr = torch.stack(
            [
                torch.tensor(normed_elev_nodes, dtype=torch.float32),
                torch.tensor(normed_slope_nodes, dtype=torch.float32),
                torch.tensor(normed_flow_distance_nodes, dtype=torch.float32),
            ],
            dim=1,
        )

    def build_edge_features(self):
        u_nodes = self.edge_index[0].tolist()
        v_nodes = self.edge_index[1].tolist()

        elev_diff_edges = []
        slope_mean_edges = []
        flow_distance_mean_edges = []
        flow_angle_diff_edges = []

        for u_id, v_id in zip(u_nodes, v_nodes):
            u = self.id_node_map[u_id]
            v = self.id_node_map[v_id]

            u_i, u_j = self.G.nodes[u]["matrix_pos"]
            v_i, v_j = self.G.nodes[v]["matrix_pos"]

            if u_id == v_id:
                elev_diff = torch.tensor(0.0, dtype=torch.float32)
                slope_mean = self.slope[u_i, u_j]
                flow_distance_mean = self.flow_distance[u_i, u_j]
                flow_angle_diff = torch.tensor(0.0, dtype=torch.float32)
            else:
                elev_diff = torch.abs(self.elev[u_i, u_j] - self.elev[v_i, v_j])
                slope_mean = (self.slope[u_i, u_j] + self.slope[v_i, v_j]) / 2.0
                flow_distance_mean = (self.flow_distance[u_i, u_j] + self.flow_distance[v_i, v_j]) / 2.0
                flow_angle_diff = torch.abs(self.flow_angle[u_i, u_j] - self.flow_angle[v_i, v_j])

            elev_diff_edges.append(elev_diff)
            slope_mean_edges.append(slope_mean)
            flow_distance_mean_edges.append(flow_distance_mean)
            flow_angle_diff_edges.append(flow_angle_diff)

        elev_diff_edges = torch.stack(elev_diff_edges)
        slope_mean_edges = torch.stack(slope_mean_edges)
        flow_distance_mean_edges = torch.stack(flow_distance_mean_edges)
        flow_angle_diff_edges = torch.stack(flow_angle_diff_edges)

        normed_elev_diff_edges, _ = self.normalize(elev_diff_edges, normalizer=None, use_log=False, method="minmax")
        normed_slope_mean_edges, _ = self.normalize(slope_mean_edges, normalizer=None, use_log=False, method="minmax")
        normed_flow_distance_mean_edges, _ = self.normalize(
            flow_distance_mean_edges,
            normalizer=None,
            use_log=False,
            method="minmax",
        )
        normed_flow_angle_diff_edges, _ = self.normalize(
            flow_angle_diff_edges,
            normalizer=None,
            use_log=False,
            method="minmax",
        )

        self.edge_attr = torch.stack(
            [
                normed_elev_diff_edges,
                normed_slope_mean_edges,
                normed_flow_distance_mean_edges,
                normed_flow_angle_diff_edges,
            ],
            dim=1,
        )

    @staticmethod
    def build_khop_upstream_edges(
        G,
        elev,
        slope,
        flow_distance,
        flow_angle,
        node_id_map,
        k: int = 2,
    ):
        new_edges = []
        elev_diff_edges = []
        slope_mean_edges = []
        flow_distance_sum_edges = []
        flow_angle_diff_edges = []

        for target in G.nodes:
            visited = set()
            queue = deque()
            queue.append((target, 0, 0.0, 0.0, 0.0, 0.0))

            while queue:
                node, hop, elev_acc, slope_acc, dist_acc, dir_acc = queue.popleft()
                if hop == k:
                    continue

                for pred in G.predecessors(node):
                    if (pred, node) in visited:
                        continue
                    visited.add((pred, node))

                    u_idx, u_jdx = G.nodes[pred]["matrix_pos"]
                    v_idx, v_jdx = G.nodes[node]["matrix_pos"]

                    if pred == node:
                        elev_acc_new = 0.0
                        slope_acc_new = slope[u_idx, u_jdx]
                        dist_acc_new = flow_distance[u_idx, u_jdx]
                        dir_acc_new = 0.0
                    else:
                        elev_acc_new = elev_acc + abs(elev[u_idx, u_jdx] - elev[v_idx, v_jdx])
                        slope_acc_new = slope_acc + (slope[u_idx, u_jdx] + slope[v_idx, v_jdx]) / 2
                        dist_acc_new = dist_acc + (flow_distance[u_idx, u_jdx] + flow_distance[v_idx, v_jdx])
                        dir_acc_new = dir_acc + abs(flow_angle[u_idx, u_jdx] - flow_angle[v_idx, v_jdx])

                    if hop + 1 <= k:
                        new_edges.append((pred, target))
                        elev_diff_edges.append(elev_acc_new)
                        slope_mean_edges.append(slope_acc_new)
                        flow_distance_sum_edges.append(dist_acc_new)
                        flow_angle_diff_edges.append(dir_acc_new)

                    queue.append((pred, hop + 1, elev_acc_new, slope_acc_new, dist_acc_new, dir_acc_new))

        new_edges_int = [(node_id_map[u], node_id_map[v]) for u, v in new_edges]
        new_edge_index = torch.tensor(new_edges_int, dtype=torch.long).t().contiguous()

        elev_diff_edges = torch.tensor(np.array(elev_diff_edges), dtype=torch.float32)
        slope_mean_edges = torch.tensor(np.array(slope_mean_edges), dtype=torch.float32)
        flow_distance_sum_edges = torch.tensor(np.array(flow_distance_sum_edges), dtype=torch.float32)
        flow_angle_diff_edges = torch.tensor(np.array(flow_angle_diff_edges), dtype=torch.float32)

        normed_elev_diff_edges, _ = RoutingDataset.normalize(
            elev_diff_edges,
            normalizer=None,
            use_log=False,
            method="minmax",
        )
        normed_slope_mean_edges, _ = RoutingDataset.normalize(
            slope_mean_edges,
            normalizer=None,
            use_log=False,
            method="minmax",
        )
        normed_flow_distance_sum_edges, _ = RoutingDataset.normalize(
            flow_distance_sum_edges,
            normalizer=None,
            use_log=False,
            method="minmax",
        )
        normed_flow_angle_diff_edges, _ = RoutingDataset.normalize(
            flow_angle_diff_edges,
            normalizer=None,
            use_log=False,
            method="minmax",
        )

        new_G = nx.DiGraph()
        new_G.add_nodes_from(G.nodes)
        new_G.add_edges_from(new_edges)
        for n, attr in G.nodes(data=True):
            new_G.nodes[n].update(attr)

        return (
            new_G,
            new_edges,
            new_edge_index,
            normed_elev_diff_edges,
            normed_slope_mean_edges,
            normed_flow_distance_sum_edges,
            normed_flow_angle_diff_edges,
        )

    def build_adjs(self):
        # [adjacent matrix]: adj[i, j] == 1 means i -> j has edge
        # [reachability matrix]: reach_down[i, j] == 1 means j is downstream of i
        adj = self.adj.clone()
        reach_down = torch.zeros_like(adj)
        adj_power = adj.clone()
        for _ in range(adj.shape[0]):
            reach_down += (adj_power > 0).float()
            adj_power = adj_power @ adj

        self.reach_down = (reach_down > 0).float()
        self.mask_downstream_adj = self.reach_down.clone()
        self.num_downstream = self.reach_down.sum(dim=1)

        # k-hop upstream mask
        adj_t = adj.t().clone()
        reach_up_khop = torch.zeros_like(adj)
        adj_power = adj_t.clone()

        for _ in range(max(self._khop, 0)):
            reach_up_khop += (adj_power > 0).float()
            adj_power = adj_power @ adj_t

        reach_up_khop = (reach_up_khop > 0).float()
        if self._self_loop:
            reach_up_khop.fill_diagonal_(1.0)

        self.mask_khop_up_adj = reach_up_khop.clone().t()
        self.num_up_khop = reach_up_khop.sum(dim=1)

        # build full-path edge attribute adjacency [N, N, F]
        edge_mat = torch.zeros(
            (self.num_nodes, self.num_nodes, self.edge_attr.shape[1]),
            dtype=self.edge_attr.dtype,
            device=self.edge_attr.device,
        )
        source = self.edge_index[0]
        target = self.edge_index[1]
        edge_mat[source, target, :] = self.edge_attr

        elev_diff_acc_adj = self.build_full_path_edge_attr_adj(
            edge_mat[:, :, 0],
            self.reach_down,
            accumulate_fn=lambda a, b, _: a + b,
            self_loop_fn=lambda _mat, _i: 0.0,
            self_loop=self._self_loop,
        )
        slope_mean_acc_adj = self.build_full_path_edge_attr_adj(
            edge_mat[:, :, 1],
            self.reach_down,
            accumulate_fn=lambda a, b, _: a + b,
            self_loop_fn=lambda mat, i: mat[i, i],
            use_count_for_mean=True,
            self_loop=self._self_loop,
        )
        flow_distance_acc_adj = self.build_full_path_edge_attr_adj(
            edge_mat[:, :, 2],
            self.reach_down,
            accumulate_fn=lambda a, b, _: a + b,
            self_loop_fn=lambda mat, i: mat[i, i],
            self_loop=self._self_loop,
        )
        flow_angle_acc_adj = self.build_full_path_edge_attr_adj(
            edge_mat[:, :, 3],
            self.reach_down,
            accumulate_fn=lambda a, b, _: a + b,
            self_loop_fn=lambda _mat, _i: 0.0,
            self_loop=self._self_loop,
        )

        normed_elev_diff_acc_adj = self.normalize(
            elev_diff_acc_adj,
            normalizer=None,
            use_log=True,
            method="minmax",
        )[0]
        normed_slope_mean_acc_adj = self.normalize(
            slope_mean_acc_adj,
            normalizer=None,
            use_log=True,
            method="minmax",
        )[0]
        normed_flow_distance_acc_adj = self.normalize(
            flow_distance_acc_adj,
            normalizer=None,
            use_log=True,
            method="minmax",
        )[0]
        normed_flow_angle_acc_adj = self.normalize(
            flow_angle_acc_adj,
            normalizer=None,
            use_log=True,
            method="minmax",
        )[0]

        self.full_path_edge_attr_adj = torch.stack(
            [
                normed_elev_diff_acc_adj,
                normed_slope_mean_acc_adj,
                normed_flow_distance_acc_adj,
                normed_flow_angle_acc_adj,
            ],
            dim=-1,
        )

    @staticmethod
    def build_full_path_edge_attr_adj(
        edge_mat,
        reach_down,
        accumulate_fn=None,
        self_loop_fn=None,
        use_count_for_mean: bool = False,
        self_loop: bool = False,
    ):
        path_edge_mat = edge_mat.clone()
        reach_down = reach_down.clone()
        num_nodes = edge_mat.shape[0]

        if accumulate_fn is None:
            accumulate_fn = lambda a, b, _: a + b

        if self_loop_fn is None:
            self_loop_fn = lambda _mat, _i: 0.0

        count_mat = (edge_mat != 0).float() if use_count_for_mean else None

        # Floyd-Warshall style accumulation
        for k in range(num_nodes):
            a = path_edge_mat[:, k].unsqueeze(1)
            b = path_edge_mat[k, :].unsqueeze(0)

            mask = (reach_down[:, k].unsqueeze(1) * reach_down[k, :].unsqueeze(0)) > 0
            updated = accumulate_fn(a, b, path_edge_mat)
            path_edge_mat = torch.where(mask, updated, path_edge_mat)

            if use_count_for_mean:
                a_count = count_mat[:, k].unsqueeze(1)
                b_count = count_mat[k, :].unsqueeze(0)
                count_updated = a_count + b_count
                count_mat = torch.where(mask, count_updated, count_mat)

        if self_loop:
            diag_idx = torch.arange(num_nodes)
            path_edge_mat[diag_idx, diag_idx] = torch.tensor(
                [self_loop_fn(edge_mat, i) for i in range(num_nodes)],
                dtype=edge_mat.dtype,
            )

        if use_count_for_mean:
            path_edge_mat = path_edge_mat / (count_mat + 1e-6)

        return path_edge_mat

    @staticmethod
    def normalize(data, normalizer=None, use_log: bool = False, method: str = "zscore"):
        if normalizer is None:
            normalizer = FeatureNormalizer(use_log=use_log, method=method)
            normalizer.fit(data)

        norm_data = normalizer.transform(data)
        if not torch.is_tensor(norm_data):
            norm_data = torch.tensor(norm_data, dtype=torch.float32)
        return norm_data, normalizer

    @staticmethod
    def inverse_transform_streamflow(x, normalizer):
        return normalizer.inverse_transform(x)

    def len(self):
        return self.num_samples

    def get(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index out of range: {idx} not in [0, {self.num_samples})")

        # x: [N, T, F]
        x = self.normed_dynamic[:, idx : idx + self._windowsize, :]

        # y: [n_pred, outlets]
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

        # use inner edges as thresholds
        bin_indices = torch.bucketize(scores, bin_edges[1:-1], right=False)
        weights = 1.0 / hist[bin_indices]
        weights = weights / (weights.sum() + 1e-12)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return sampler, weights
