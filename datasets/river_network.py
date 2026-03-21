import glob
import os
import pickle
import re
import shutil
import sys
import warnings
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
from netCDF4 import Dataset as ncDataset
from netCDF4 import chartostring


def _import_real_build_dependencies():
    try:
        from easy_vic_build.Evb_dir_class import Evb_dir
        from easy_vic_build.build_hydroanalysis import buildRivernetwork_level1
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Real river-network build requires easy_vic_build dependencies that are not importable. "
            f"Missing module: `{exc.name}`. "
            "Please ensure `easy_vic_build` is available on PYTHONPATH."
        ) from exc

    return (
        Evb_dir,
        buildRivernetwork_level1,
    )


def normalize_threshold_value(value) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return format(number, "g")
    except ValueError:
        return text


def get_rivernetwork_threshold(model_cfg: Optional[Dict]) -> Optional[str]:
    graph_cfg = (model_cfg or {}).get("graph", {})
    threshold = graph_cfg.get("rivernetwork_threshold", None)
    return normalize_threshold_value(threshold)


def find_rivernetwork_graph_file(data_home: str, threshold: Optional[str]) -> Optional[str]:
    if threshold is None:
        return None

    threshold_dir = os.path.join(data_home, f"threshold_{threshold}")
    threshold_dir_patterns = [
        "river_network_graph_connected_threshold_*.pkl",
        "river_network_graph_connected*.pkl",
        "river_network_graphthreshold_*.pkl",
        "river_network_graph*_threshold_*.pkl",
        "river_network_graph*.pkl",
    ]
    threshold_dir_candidates = []
    if os.path.isdir(threshold_dir):
        for pattern in threshold_dir_patterns:
            threshold_dir_candidates.extend(glob.glob(os.path.join(threshold_dir, pattern)))

    candidate_patterns = [
        "river_network_graph_connected_threshold_*.pkl",
        "river_network_graphthreshold_*.pkl",
        "river_network_graph*_threshold_*.pkl",
    ]

    all_candidates = []
    for pattern in candidate_patterns:
        all_candidates.extend(glob.glob(os.path.join(data_home, pattern)))

    if not threshold_dir_candidates and not all_candidates:
        return None

    # Prefer files under `threshold_<n>/`, then fallback to data_home root legacy files.
    unique_candidates = sorted(set(threshold_dir_candidates + all_candidates))
    matched = []

    for path in unique_candidates:
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]
        found = re.search(r"threshold_([A-Za-z0-9\.\-]+)$", stem)

        if found is not None:
            found_threshold = normalize_threshold_value(found.group(1))
            if found_threshold == threshold:
                matched.append(path)
                continue

        # If file is already in threshold-specific directory and filename has no threshold suffix,
        # accept it as a match.
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(threshold_dir):
            matched.append(path)

    if not matched:
        return None

    def _priority(path: str):
        name = os.path.basename(path).lower()
        return (
            0 if "river_network_graph_connected" in name else 1,
            0 if "connected" in name else 1,
            0 if "river_network_graph" in name else 1,
            name,
        )

    matched.sort(key=_priority)
    return matched[0]


def organize_threshold_graph_files(data_home: str):
    # Move legacy threshold-tagged files in data_home root into `threshold_<n>/`.
    root_patterns = [
        "river_network_graph*_threshold_*.*",
        "river_network_graphthreshold_*.*",
        "fig_river_network*_threshold_*.*",
    ]
    moved = 0
    for pattern in root_patterns:
        for src_path in glob.glob(os.path.join(data_home, pattern)):
            if os.path.isdir(src_path):
                continue

            filename = os.path.basename(src_path)
            stem = os.path.splitext(filename)[0]
            found = re.search(r"threshold_([A-Za-z0-9\.\-]+)$", stem)
            if found is None:
                continue

            threshold = normalize_threshold_value(found.group(1))
            dst_dir = os.path.join(data_home, f"threshold_{threshold}")
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, filename)

            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue
            if os.path.exists(dst_path):
                continue

            shutil.move(src_path, dst_path)
            moved += 1
    return moved


def build_rivernetwork_for_threshold(data_home: str, threshold: str):
    try:
        _build_rivernetwork_with_evb(data_home, threshold)
    except Exception as exc:
        warnings.warn(
            "EVB river-network build failed; fallback graph build will be used. "
            f"threshold={threshold}, reason={type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        _build_rivernetwork_from_existing_graphs(data_home, threshold)


def _build_rivernetwork_with_evb(data_home: str, threshold: str):
    (
        Evb_dir,
        buildRivernetwork_level1,
    ) = _import_real_build_dependencies()

    evb_dir = Evb_dir(cases_home=data_home)
    evb_dir.builddir(case_name="build_river_network")
    _copy_flow_tifs_to_evb_hydroanalysis_dir(evb_dir, data_home)

    domain_dataset = _build_domain_dataset_like(data_home)
    outlets = _read_labeled_nodes_from_rvic(data_home)

    river_network = buildRivernetwork_level1(
        evb_dir,
        threshold=int(float(threshold)),
        domain_dataset=domain_dataset,
        plot_bool=True,
        labeled_nodes=outlets,
    )

    threshold_dir = os.path.join(data_home, f"threshold_{threshold}")
    os.makedirs(threshold_dir, exist_ok=True)

    save_path_river_network = os.path.join(threshold_dir, f"river_network_graph_threshold_{threshold}.pkl")
    save_path_river_network_full = os.path.join(threshold_dir, f"river_network_graph_full_threshold_{threshold}.pkl")
    save_path_river_network_connected = os.path.join(threshold_dir, f"river_network_graph_connected_threshold_{threshold}.pkl")

    with open(save_path_river_network, "wb") as f:
        pickle.dump(river_network["river_network_graph"], f)

    with open(save_path_river_network_full, "wb") as f:
        pickle.dump(river_network["river_network_graph_full"], f)

    with open(save_path_river_network_connected, "wb") as f:
        pickle.dump(river_network["river_network_graph_connected"], f)

    river_network["figs"]["fig_river_network"].savefig(
        os.path.join(threshold_dir, f"fig_river_network_threshold_{threshold}.png")
    )
    river_network["figs"]["fig_river_network_full"].savefig(
        os.path.join(threshold_dir, f"fig_river_network_full_threshold_{threshold}.png")
    )
    river_network["figs"]["fig_river_network_connected"].savefig(
        os.path.join(threshold_dir, f"fig_river_network_connected_threshold_{threshold}.png")
    )


class _ArrayVar:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class _DomainDatasetLike:
    def __init__(self, mask_array):
        self.variables = {"mask": _ArrayVar(mask_array)}

    def close(self):
        return None


def _build_domain_dataset_like(data_home: str):
    flow_nc_path = os.path.join(data_home, "flow_direction_file.nc")
    if not os.path.exists(flow_nc_path):
        raise FileNotFoundError(f"`flow_direction_file.nc` not found under `{data_home}`.")

    with ncDataset(flow_nc_path, "r") as ds:
        if "Basin_ID" in ds.variables:
            basin_id = ds.variables["Basin_ID"][:, :]
            basin_id = basin_id.filled(0) if hasattr(basin_id, "filled") else np.asarray(basin_id)
            mask = (basin_id > 0).astype(np.int32)
        else:
            flow_direction = ds.variables["Flow_Direction"][:, :]
            flow_direction = flow_direction.filled(0) if hasattr(flow_direction, "filled") else np.asarray(flow_direction)
            mask = (flow_direction > 0).astype(np.int32)

    return _DomainDatasetLike(mask)


def _read_labeled_nodes_from_rvic(data_home: str):
    rvic_candidates = sorted(glob.glob(os.path.join(data_home, "*rvic*.nc")))
    if not rvic_candidates:
        return None

    rvic_path = rvic_candidates[0]
    with ncDataset(rvic_path, "r") as ds:
        outlet_x = ds.variables["outlet_x_ind"][:]
        outlet_y = ds.variables["outlet_y_ind"][:]
        if "outlet_name" in ds.variables:
            _ = chartostring(ds.variables["outlet_name"][:])  # keep consistent with RVIC parsing

    outlet_x = outlet_x.filled(np.nan) if hasattr(outlet_x, "filled") else np.asarray(outlet_x)
    outlet_y = outlet_y.filled(np.nan) if hasattr(outlet_y, "filled") else np.asarray(outlet_y)

    labeled_nodes = []
    for x_idx, y_idx in zip(outlet_x, outlet_y):
        if np.isnan(x_idx) or np.isnan(y_idx):
            continue
        labeled_nodes.append(f"cell_{int(y_idx)}_{int(x_idx)}")

    return labeled_nodes if labeled_nodes else None


def _copy_flow_tifs_to_evb_hydroanalysis_dir(evb_dir, data_home: str):
    hydroanalysis_dir = getattr(evb_dir, "Hydroanalysis_dir", None)
    if not hydroanalysis_dir:
        raise AttributeError("evb_dir has no `Hydroanalysis_dir` attribute.")

    os.makedirs(hydroanalysis_dir, exist_ok=True)

    for filename in ["flow_direction.tif", "flow_acc.tif"]:
        src_path = os.path.join(data_home, filename)
        dst_path = os.path.join(hydroanalysis_dir, filename)

        if os.path.exists(src_path):
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)
            continue

        # Keep existing files if already available in Hydroanalysis_dir.
        if os.path.exists(dst_path):
            continue

        raise FileNotFoundError(
            f"Required file not found for EVB hydroanalysis setup: `{filename}`. "
            f"Expected at `{src_path}`."
        )


def _extract_threshold_from_filename(path: str) -> Optional[float]:
    stem = os.path.splitext(os.path.basename(path))[0]
    found = re.search(r"threshold_([A-Za-z0-9\.\-]+)$", stem)
    if found is None:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def _pick_fallback_source_graph(data_home: str, threshold_value: float) -> str:
    patterns = [
        "river_network_graph_full_threshold_*.pkl",
        "river_network_graphthreshold_*.pkl",
        "river_network_graph_connected_threshold_*.pkl",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(data_home, pattern)))
        candidates.extend(glob.glob(os.path.join(data_home, "**", pattern), recursive=True))

    if not candidates:
        raise FileNotFoundError(
            f"No existing threshold graph files found under `{data_home}` for fallback build."
        )

    rows = []
    for path in sorted(set(candidates)):
        name = os.path.basename(path).lower()
        src_threshold = _extract_threshold_from_filename(path)
        kind_priority = 0
        if "connected" in name:
            kind_priority = 1
        rows.append((path, src_threshold, kind_priority))

    # Prefer source threshold <= target so filtering can shrink the network consistently.
    valid = [row for row in rows if row[1] is not None and row[1] <= threshold_value]
    if valid:
        valid.sort(key=lambda x: (-(x[1]), x[2], x[0]))  # highest threshold first
        return valid[0][0]

    # If no threshold <= target exists, pick the least-threshold file available.
    rows.sort(key=lambda x: (x[1] if x[1] is not None else float("inf"), x[2], x[0]))
    return rows[0][0]


def _build_rivernetwork_from_existing_graphs(data_home: str, threshold: str):
    try:
        threshold_value = float(threshold)
    except ValueError as exc:
        raise ValueError(f"Threshold `{threshold}` is not numeric; fallback build requires numeric threshold.") from exc

    source_path = _pick_fallback_source_graph(data_home, threshold_value)
    with open(source_path, "rb") as f:
        source_graph = pickle.load(f)

    nodes_to_keep = []
    for node_name, attr in source_graph.nodes(data=True):
        flow_acc = attr.get("flow_acc", None)
        if flow_acc is None:
            nodes_to_keep.append(node_name)
            continue

        try:
            keep = float(flow_acc) >= threshold_value
        except (TypeError, ValueError):
            keep = True
        if keep:
            nodes_to_keep.append(node_name)

    if not nodes_to_keep:
        raise RuntimeError(
            f"Fallback build produced empty graph for threshold={threshold} from source `{source_path}`."
        )

    graph_filtered = source_graph.subgraph(nodes_to_keep).copy()
    if graph_filtered.number_of_nodes() == 0:
        raise RuntimeError(
            f"Fallback build produced empty graph for threshold={threshold} from source `{source_path}`."
        )

    if isinstance(graph_filtered, nx.DiGraph):
        components = list(nx.weakly_connected_components(graph_filtered))
    else:
        components = list(nx.connected_components(graph_filtered))

    if components:
        largest = max(components, key=len)
        graph_connected = graph_filtered.subgraph(largest).copy()
    else:
        graph_connected = graph_filtered

    threshold_dir = os.path.join(data_home, f"threshold_{threshold}")
    os.makedirs(threshold_dir, exist_ok=True)

    save_path_river_network = os.path.join(threshold_dir, f"river_network_graph_threshold_{threshold}.pkl")
    save_path_river_network_full = os.path.join(threshold_dir, f"river_network_graph_full_threshold_{threshold}.pkl")
    save_path_river_network_connected = os.path.join(threshold_dir, f"river_network_graph_connected_threshold_{threshold}.pkl")

    with open(save_path_river_network, "wb") as f:
        pickle.dump(graph_filtered, f)

    with open(save_path_river_network_full, "wb") as f:
        pickle.dump(graph_filtered, f)

    with open(save_path_river_network_connected, "wb") as f:
        pickle.dump(graph_connected, f)

    _save_fallback_graph_figures(
        graph_filtered=graph_filtered,
        graph_connected=graph_connected,
        threshold_dir=threshold_dir,
        threshold=threshold,
    )


def _save_fallback_graph_figures(graph_filtered, graph_connected, threshold_dir: str, threshold: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _draw_graph(ax, graph, title: str):
        ax.set_title(title)
        if graph.number_of_nodes() == 0:
            ax.axis("off")
            return

        # Prefer geographic-like matrix positions when available.
        has_matrix_pos = all("matrix_pos" in attr for _, attr in graph.nodes(data=True))
        if has_matrix_pos:
            pos = {
                node: (float(attr["matrix_pos"][1]), -float(attr["matrix_pos"][0]))
                for node, attr in graph.nodes(data=True)
            }
        else:
            pos = nx.spring_layout(graph, seed=42)

        nx.draw_networkx_edges(graph, pos=pos, ax=ax, width=0.3, alpha=0.6, arrows=False)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=6, alpha=0.85)
        ax.axis("off")

    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=200)
    _draw_graph(ax1, graph_filtered, f"river_network threshold={threshold}")
    fig1.tight_layout()
    fig1.savefig(os.path.join(threshold_dir, f"fig_river_network_threshold_{threshold}.png"))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=200)
    _draw_graph(ax2, graph_filtered, f"river_network_full threshold={threshold}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(threshold_dir, f"fig_river_network_full_threshold_{threshold}.png"))
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=200)
    _draw_graph(ax3, graph_connected, f"river_network_connected threshold={threshold}")
    fig3.tight_layout()
    fig3.savefig(os.path.join(threshold_dir, f"fig_river_network_connected_threshold_{threshold}.png"))
    plt.close(fig3)


def resolve_rivernetwork_graph_path(
    data_home: str,
    model_cfg: Optional[Dict],
    resolve_file_fn: Callable,
    build_fn: Optional[Callable[[str], None]] = None,
) -> str:
    organize_threshold_graph_files(data_home)

    threshold = get_rivernetwork_threshold(model_cfg)
    if threshold is None:
        try:
            return resolve_file_fn(["river_network_graph_connected.pkl", "river_network_graph*.pkl"])
        except FileNotFoundError:
            candidates = sorted(
                glob.glob(os.path.join(data_home, "threshold_*", "river_network_graph_connected_threshold_*.pkl"))
            )
            if not candidates:
                candidates = sorted(
                    glob.glob(os.path.join(data_home, "threshold_*", "river_network_graph*.pkl"))
                )
            if candidates:
                return candidates[0]
            raise

    graph_path = find_rivernetwork_graph_file(data_home, threshold)
    if graph_path is not None:
        return graph_path

    if build_fn is not None:
        build_fn(threshold)

    graph_path = find_rivernetwork_graph_file(data_home, threshold)
    if graph_path is not None:
        return graph_path

    raise FileNotFoundError(
        "No river network graph matched threshold "
        f"`{threshold}` in `{data_home}`, and build function has not generated one yet."
    )
