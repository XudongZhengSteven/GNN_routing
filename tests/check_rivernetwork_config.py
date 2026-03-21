import argparse
import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.common import load_yaml_cfg, resolve_file
from datasets.river_network import (
    build_rivernetwork_for_threshold,
    find_rivernetwork_graph_file,
    get_rivernetwork_threshold,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate river network selection from model.yaml threshold."
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/model.yaml",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--data-home",
        default="data/raw/case5",
        help="Directory containing river network graph files.",
    )
    parser.add_argument(
        "--build-missing",
        action="store_true",
        help="If no graph matches threshold, build it with build_rivernetwork_for_threshold().",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = load_yaml_cfg(args.model_cfg)
    threshold = get_rivernetwork_threshold(model_cfg)

    print(f"model_cfg: {os.path.abspath(args.model_cfg)}")
    print(f"data_home: {os.path.abspath(args.data_home)}")
    print(f"rivernetwork_threshold: {threshold}")

    if threshold is None:
        graph_path = resolve_file(args.data_home, ["river_network_graph_connected.pkl", "river_network_graph*.pkl"])
        print("threshold is empty, fallback to default graph file pattern")
    else:
        graph_path = find_rivernetwork_graph_file(args.data_home, threshold)
        if graph_path is None and args.build_missing:
            print(f"no existing graph for threshold={threshold}, building...")
            build_rivernetwork_for_threshold(args.data_home, threshold)
            graph_path = find_rivernetwork_graph_file(args.data_home, threshold)

        if graph_path is None:
            raise FileNotFoundError(
                f"No river network graph matched threshold={threshold} in {args.data_home}. "
                "Use --build-missing to build automatically."
            )

    print(f"selected_graph: {graph_path}")

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    num_nodes = getattr(graph, "number_of_nodes", lambda: None)()
    num_edges = getattr(graph, "number_of_edges", lambda: None)()
    print(f"graph_nodes: {num_nodes}")
    print(f"graph_edges: {num_edges}")


if __name__ == "__main__":
    main()
