from collections import deque

import networkx as nx
import numpy as np
import torch

try:
    from .normalizer import FeatureNormalizer
except ImportError:
    from normalizer import FeatureNormalizer


DIRECTION_TO_ANGLE = {
    1: 0.0,
    2: 45.0,
    4: 90.0,
    8: 135.0,
    16: 180.0,
    32: 225.0,
    64: 270.0,
    128: 315.0,
}


def normalize_feature(data, normalizer=None, use_log: bool = False, method: str = "zscore"):
    if normalizer is None:
        normalizer = FeatureNormalizer(use_log=use_log, method=method)
        normalizer.fit(data)

    norm_data = normalizer.transform(data)
    if not torch.is_tensor(norm_data):
        norm_data = torch.tensor(norm_data, dtype=torch.float32)
    return norm_data, normalizer


def build_dynamic_features(dataset):
    dataset.runoff = torch.tensor(dataset.runoff[dataset._time_slice_input, :, :], dtype=torch.float32)
    dataset.baseflow = torch.tensor(dataset.baseflow[dataset._time_slice_input, :, :], dtype=torch.float32)
    dataset.streamflow_obs = torch.tensor(dataset.streamflow_obs[dataset._time_slice_obs, :], dtype=torch.float32)

    num_input_steps = dataset.runoff.shape[0]
    num_obs_steps = dataset.streamflow_obs.shape[0]

    num_samples_input = num_input_steps - dataset._windowsize + 1
    num_samples_obs = (
        num_obs_steps
        - dataset._windowsize_in_days
        - dataset._n_pred
        + 1
        + dataset.predict_current
    )
    dataset.num_samples = int(min(num_samples_input, num_samples_obs))

    if dataset.num_samples <= 0:
        raise ValueError(
            "No valid sample can be generated. "
            f"num_input={num_input_steps}, num_obs={num_obs_steps}, "
            f"windowsize={dataset._windowsize}, n_pred={dataset._n_pred}."
        )

    dataset.normed_runoff, normalizer_runoff = normalize_feature(
        dataset.runoff,
        normalizer=dataset.normalizers.get("runoff", None),
        use_log=True,
        method="minmax",
    )
    dataset.normed_baseflow, normalizer_baseflow = normalize_feature(
        dataset.baseflow,
        normalizer=dataset.normalizers.get("baseflow", None),
        use_log=True,
        method="minmax",
    )

    normed_streamflow_obs_list = []
    normalizer_streamflow_obs = []
    existing_streamflow_normalizers = dataset.normalizers.get("streamflow", None)

    for i in range(dataset.streamflow_obs.shape[1]):
        obs_i = dataset.streamflow_obs[:, i]

        normalizer_i = None
        if isinstance(existing_streamflow_normalizers, (list, tuple)) and i < len(existing_streamflow_normalizers):
            normalizer_i = existing_streamflow_normalizers[i]

        normed_obs, normalizer_streamflow = normalize_feature(
            obs_i,
            normalizer=normalizer_i,
            use_log=True,
            method="minmax-11",
        )

        normed_streamflow_obs_list.append(normed_obs)
        normalizer_streamflow_obs.append(normalizer_streamflow)

    dataset.normed_streamflow_obs = torch.stack(normed_streamflow_obs_list, dim=1)

    dataset.normalizers["runoff"] = normalizer_runoff
    dataset.normalizers["baseflow"] = normalizer_baseflow
    dataset.normalizers["streamflow"] = normalizer_streamflow_obs


def build_static_features(dataset):
    dataset.elev = torch.tensor(dataset.elev[:, :], dtype=torch.float32)
    dataset.slope = torch.tensor(dataset.slope[:, :], dtype=torch.float32)
    dataset.flow_distance = torch.tensor(dataset.flow_distance[:, :], dtype=torch.float32)
    dataset.flow_direction = torch.tensor(dataset.flow_direction[:, :], dtype=torch.long)

    dataset.flow_angle = torch.zeros_like(dataset.flow_direction, dtype=torch.float32)
    for direction, angle in DIRECTION_TO_ANGLE.items():
        dataset.flow_angle = torch.where(
            dataset.flow_direction == int(direction),
            torch.tensor(float(angle), dtype=torch.float32),
            dataset.flow_angle,
        )

    dataset.normed_elev, _ = normalize_feature(dataset.elev, normalizer=None, use_log=False, method="minmax")
    dataset.normed_slope, _ = normalize_feature(dataset.slope, normalizer=None, use_log=False, method="minmax")
    dataset.normed_flow_distance, _ = normalize_feature(
        dataset.flow_distance,
        normalizer=None,
        use_log=False,
        method="minmax",
    )
    dataset.normed_flow_angle, _ = normalize_feature(
        dataset.flow_angle,
        normalizer=None,
        use_log=False,
        method="minmax",
    )


def build_node_features(dataset):
    normed_runoff_nodes = []
    normed_baseflow_nodes = []

    normed_elev_nodes = []
    normed_slope_nodes = []
    normed_flow_distance_nodes = []

    for node_name in dataset.G.nodes:
        i, j = dataset.G.nodes[node_name]["matrix_pos"]

        normed_runoff_nodes.append(dataset.normed_runoff[:, i, j])
        normed_baseflow_nodes.append(dataset.normed_baseflow[:, i, j])

        normed_elev_nodes.append(dataset.normed_elev[i, j])
        normed_slope_nodes.append(dataset.normed_slope[i, j])
        normed_flow_distance_nodes.append(dataset.normed_flow_distance[i, j])

    dataset.normed_runoff_nodes = torch.stack(normed_runoff_nodes, dim=0)
    dataset.normed_baseflow_nodes = torch.stack(normed_baseflow_nodes, dim=0)

    dataset.node_attr = torch.stack(
        [
            torch.tensor(normed_elev_nodes, dtype=torch.float32),
            torch.tensor(normed_slope_nodes, dtype=torch.float32),
            torch.tensor(normed_flow_distance_nodes, dtype=torch.float32),
        ],
        dim=1,
    )


def build_edge_features(dataset):
    u_nodes = dataset.edge_index[0].tolist()
    v_nodes = dataset.edge_index[1].tolist()

    elev_diff_edges = []
    slope_mean_edges = []
    flow_distance_mean_edges = []
    flow_angle_diff_edges = []

    for u_id, v_id in zip(u_nodes, v_nodes):
        u = dataset.id_node_map[u_id]
        v = dataset.id_node_map[v_id]

        u_i, u_j = dataset.G.nodes[u]["matrix_pos"]
        v_i, v_j = dataset.G.nodes[v]["matrix_pos"]

        if u_id == v_id:
            elev_diff = torch.tensor(0.0, dtype=torch.float32)
            slope_mean = dataset.slope[u_i, u_j]
            flow_distance_mean = dataset.flow_distance[u_i, u_j]
            flow_angle_diff = torch.tensor(0.0, dtype=torch.float32)
        else:
            elev_diff = torch.abs(dataset.elev[u_i, u_j] - dataset.elev[v_i, v_j])
            slope_mean = (dataset.slope[u_i, u_j] + dataset.slope[v_i, v_j]) / 2.0
            flow_distance_mean = (dataset.flow_distance[u_i, u_j] + dataset.flow_distance[v_i, v_j]) / 2.0
            flow_angle_diff = torch.abs(dataset.flow_angle[u_i, u_j] - dataset.flow_angle[v_i, v_j])

        elev_diff_edges.append(elev_diff)
        slope_mean_edges.append(slope_mean)
        flow_distance_mean_edges.append(flow_distance_mean)
        flow_angle_diff_edges.append(flow_angle_diff)

    elev_diff_edges = torch.stack(elev_diff_edges)
    slope_mean_edges = torch.stack(slope_mean_edges)
    flow_distance_mean_edges = torch.stack(flow_distance_mean_edges)
    flow_angle_diff_edges = torch.stack(flow_angle_diff_edges)

    normed_elev_diff_edges, _ = normalize_feature(elev_diff_edges, normalizer=None, use_log=False, method="minmax")
    normed_slope_mean_edges, _ = normalize_feature(slope_mean_edges, normalizer=None, use_log=False, method="minmax")
    normed_flow_distance_mean_edges, _ = normalize_feature(
        flow_distance_mean_edges,
        normalizer=None,
        use_log=False,
        method="minmax",
    )
    normed_flow_angle_diff_edges, _ = normalize_feature(
        flow_angle_diff_edges,
        normalizer=None,
        use_log=False,
        method="minmax",
    )

    dataset.edge_attr = torch.stack(
        [
            normed_elev_diff_edges,
            normed_slope_mean_edges,
            normed_flow_distance_mean_edges,
            normed_flow_angle_diff_edges,
        ],
        dim=1,
    )


def build_khop_upstream_edges(G, elev, slope, flow_distance, flow_angle, node_id_map, k: int = 2):
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

    normed_elev_diff_edges, _ = normalize_feature(
        elev_diff_edges,
        normalizer=None,
        use_log=False,
        method="minmax",
    )
    normed_slope_mean_edges, _ = normalize_feature(
        slope_mean_edges,
        normalizer=None,
        use_log=False,
        method="minmax",
    )
    normed_flow_distance_sum_edges, _ = normalize_feature(
        flow_distance_sum_edges,
        normalizer=None,
        use_log=False,
        method="minmax",
    )
    normed_flow_angle_diff_edges, _ = normalize_feature(
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


def build_adjs(dataset):
    adj = dataset.adj.clone()
    reach_down = torch.zeros_like(adj)
    adj_power = adj.clone()
    for _ in range(adj.shape[0]):
        reach_down += (adj_power > 0).float()
        adj_power = adj_power @ adj

    dataset.reach_down = (reach_down > 0).float()
    dataset.mask_downstream_adj = dataset.reach_down.clone()
    dataset.num_downstream = dataset.reach_down.sum(dim=1)

    adj_t = adj.t().clone()
    reach_up_khop = torch.zeros_like(adj)
    adj_power = adj_t.clone()

    for _ in range(max(dataset._khop, 0)):
        reach_up_khop += (adj_power > 0).float()
        adj_power = adj_power @ adj_t

    reach_up_khop = (reach_up_khop > 0).float()
    if dataset._self_loop:
        reach_up_khop.fill_diagonal_(1.0)

    dataset.mask_khop_up_adj = reach_up_khop.clone().t()
    dataset.num_up_khop = reach_up_khop.sum(dim=1)

    edge_mat = torch.zeros(
        (dataset.num_nodes, dataset.num_nodes, dataset.edge_attr.shape[1]),
        dtype=dataset.edge_attr.dtype,
        device=dataset.edge_attr.device,
    )
    source = dataset.edge_index[0]
    target = dataset.edge_index[1]
    edge_mat[source, target, :] = dataset.edge_attr

    elev_diff_acc_adj = build_full_path_edge_attr_adj(
        edge_mat[:, :, 0],
        dataset.reach_down,
        accumulate_fn=lambda a, b, _: a + b,
        self_loop_fn=lambda _mat, _i: 0.0,
        self_loop=dataset._self_loop,
    )
    slope_mean_acc_adj = build_full_path_edge_attr_adj(
        edge_mat[:, :, 1],
        dataset.reach_down,
        accumulate_fn=lambda a, b, _: a + b,
        self_loop_fn=lambda mat, i: mat[i, i],
        use_count_for_mean=True,
        self_loop=dataset._self_loop,
    )
    flow_distance_acc_adj = build_full_path_edge_attr_adj(
        edge_mat[:, :, 2],
        dataset.reach_down,
        accumulate_fn=lambda a, b, _: a + b,
        self_loop_fn=lambda mat, i: mat[i, i],
        self_loop=dataset._self_loop,
    )
    flow_angle_acc_adj = build_full_path_edge_attr_adj(
        edge_mat[:, :, 3],
        dataset.reach_down,
        accumulate_fn=lambda a, b, _: a + b,
        self_loop_fn=lambda _mat, _i: 0.0,
        self_loop=dataset._self_loop,
    )

    normed_elev_diff_acc_adj = normalize_feature(
        elev_diff_acc_adj,
        normalizer=None,
        use_log=True,
        method="minmax",
    )[0]
    normed_slope_mean_acc_adj = normalize_feature(
        slope_mean_acc_adj,
        normalizer=None,
        use_log=True,
        method="minmax",
    )[0]
    normed_flow_distance_acc_adj = normalize_feature(
        flow_distance_acc_adj,
        normalizer=None,
        use_log=True,
        method="minmax",
    )[0]
    normed_flow_angle_acc_adj = normalize_feature(
        flow_angle_acc_adj,
        normalizer=None,
        use_log=True,
        method="minmax",
    )[0]

    dataset.full_path_edge_attr_adj = torch.stack(
        [
            normed_elev_diff_acc_adj,
            normed_slope_mean_acc_adj,
            normed_flow_distance_acc_adj,
            normed_flow_angle_acc_adj,
        ],
        dim=-1,
    )
