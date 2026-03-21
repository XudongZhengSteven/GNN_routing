import os
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataset import RoutingDataset


class TestRoutingDatasetBuild(unittest.TestCase):
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

        cls._tmpdir = TemporaryDirectory()
        cls.model_cfg = Path(cls._tmpdir.name) / "model_test.yaml"

        cls._can_run = True
        cls._skip_reason = ""

        if not cls.data_home.exists():
            cls._can_run = False
            cls._skip_reason = f"data directory not found: {cls.data_home}"
            return

        threshold = cls._find_existing_threshold(cls.data_home)
        if threshold is None:
            # No threshold file found; keep config valid and let test decide if fallback graph exists.
            cls.model_cfg.write_text("graph:\n  rivernetwork_threshold: null\n", encoding="utf-8")
        else:
            cls.model_cfg.write_text(
                f'graph:\n  rivernetwork_threshold: "{threshold}"\n',
                encoding="utf-8",
            )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    @staticmethod
    def _find_existing_threshold(data_home: Path):
        candidates = sorted(data_home.rglob("river_network_graph*_threshold_*.pkl"))
        for path in candidates:
            found = re.search(r"threshold_([A-Za-z0-9\.\-]+)$", path.stem)
            if found:
                return found.group(1)
        return None

    def setUp(self):
        if not self._can_run:
            self.skipTest(self._skip_reason)

    def test_can_build_train_dataset_and_get_sample(self):
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
        self._print_sample_summary(ds, sample, "RoutingDataset/train sample")
        self.assertTrue(hasattr(sample, "x"))
        self.assertTrue(hasattr(sample, "y"))
        self.assertTrue(hasattr(sample, "edge_index"))
        self.assertTrue(hasattr(sample, "edge_attr"))
        self.assertTrue(hasattr(sample, "node_attr"))

        self.assertEqual(sample.x.ndim, 3)
        self.assertEqual(sample.y.ndim, 2)
        self.assertEqual(sample.edge_index.shape[0], 2)
        self.assertEqual(sample.node_attr.shape[0], ds.num_nodes)

    def test_can_build_val_dataset_with_train_normalizers(self):
        train_ds = RoutingDataset(
            dataset_type="train",
            windowsize=16,
            input_freq_per_day=1,
            n_pred=1,
            khop=3,
            data_home=str(self.data_home),
            data_cfg_path=str(self.data_cfg),
            model_cfg_path=str(self.model_cfg),
        )
        val_ds = RoutingDataset(
            dataset_type="val",
            windowsize=16,
            input_freq_per_day=1,
            n_pred=1,
            khop=3,
            normalizers=train_ds.normalizers,
            data_home=str(self.data_home),
            data_cfg_path=str(self.data_cfg),
            model_cfg_path=str(self.model_cfg),
        )

        self.assertGreater(len(val_ds), 0)
        self.assertIs(train_ds.normalizers["runoff"], val_ds.normalizers["runoff"])
        self.assertEqual(train_ds.outlet_nums, val_ds.outlet_nums)


if __name__ == "__main__":
    unittest.main(exit=False)
