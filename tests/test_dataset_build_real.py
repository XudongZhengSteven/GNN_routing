import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataset import RoutingDataset


class TestRoutingDatasetBuildReal(unittest.TestCase):
    @staticmethod
    def _print_sample_summary(ds, sample, tag: str):
        print(f"\n[{tag}]")
        print(f"len(ds)={len(ds)}, num_nodes={ds.num_nodes}, outlet_nums={ds.outlet_nums}")
        print(f"x.shape={tuple(sample.x.shape)}, y.shape={tuple(sample.y.shape)}")
        print(
            f"edge_index.shape={tuple(sample.edge_index.shape)}, "
            f"edge_attr.shape={tuple(sample.edge_attr.shape)}"
        )
        print(f"node_attr.shape={tuple(sample.node_attr.shape)}")
        print(f"outlet_names(head)={ds.outlet_names[: min(5, len(ds.outlet_names))]}")

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.data_home = cls.project_root / "data" / "raw" / "case5"
        cls.data_cfg = cls.project_root / "configs" / "data.yaml"
        cls.model_cfg = cls.project_root / "configs" / "model.yaml"

    def setUp(self):
        if not self.data_home.exists():
            self.skipTest(f"data directory not found: {self.data_home}")

    def test_build_routing_dataset_with_real_config(self):
        ds = RoutingDataset(
            dataset_type="train",
            windowsize=16,
            input_freq_per_day=1,
            n_pred=1,
            khop=3,
            data_home=str(self.data_home),
            data_cfg_path=str(self.data_cfg),
            model_cfg_path=str(self.model_cfg),
        )

        self.assertGreater(len(ds), 0)
        print("\n[RoutingDataset/runtime schema]")
        ds.print_runtime_schema()
        sample = ds[0]
        self._print_sample_summary(ds, sample, "RoutingDataset/real train sample")

        self.assertTrue(hasattr(sample, "x"))
        self.assertTrue(hasattr(sample, "y"))
        self.assertTrue(hasattr(sample, "edge_index"))
        self.assertEqual(sample.edge_index.shape[0], 2)
        self.assertEqual(sample.x.ndim, 3)
        self.assertEqual(sample.y.ndim, 2)

        y_denorm = ds.inverse_transform_streamflow_tensor(sample.y)
        self.assertEqual(tuple(y_denorm.shape), tuple(sample.y.shape))
        self.assertTrue(bool(torch.isfinite(y_denorm).all()))

        threshold = ds._get_rivernetwork_threshold()
        graph_path = ds._resolve_rivernetwork_graph_path()
        if threshold is not None:
            self.assertIn(f"threshold_{threshold}", os.path.basename(graph_path))


if __name__ == "__main__":
    unittest.main(exit=False)
