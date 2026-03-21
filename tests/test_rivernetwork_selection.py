import os
import sys
import types
import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


def _register_module(name, attrs=None):
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if attrs:
        for key, value in attrs.items():
            setattr(mod, key, value)
    return mod


def _load_routing_dataset_class():
    project_root = Path(__file__).resolve().parents[1]

    # Provide `normalizer` module alias expected by current dataset.py.
    if "normalizer" not in sys.modules:
        norm_path = project_root / "datasets" / "normalizer.py"
        norm_spec = importlib.util.spec_from_file_location("normalizer", norm_path)
        norm_module = importlib.util.module_from_spec(norm_spec)
        assert norm_spec.loader is not None
        norm_spec.loader.exec_module(norm_module)
        sys.modules["normalizer"] = norm_module

    # Stub optional heavy dependencies so the module can be imported.
    _register_module("torch_geometric")
    _register_module(
        "torch_geometric.data",
        {
            "Data": type("Data", (dict,), {}),
            "InMemoryDataset": type("InMemoryDataset", (object,), {}),
        },
    )
    _register_module("torch_geometric.utils", {"to_dense_adj": lambda *args, **kwargs: None})

    _register_module("easy_vic_build")
    _register_module("easy_vic_build.Evb_dir_class", {"Evb_dir": object})
    _register_module("easy_vic_build.tools")
    _register_module("easy_vic_build.tools.utilities", {"readDomain": lambda *args, **kwargs: None})
    _register_module("easy_vic_build.build_hydroanalysis", {"buildRivernetwork_level1": lambda *args, **kwargs: None})

    _register_module("utils")
    _register_module("utils.HRB_utils")
    _register_module("utils.HRB_utils.HRB_build_dpc", {"dataProcess_VIC_level3_HRB": object})
    _register_module("utils.HRB_utils.general_info", {"station_names": []})

    dataset_path = project_root / "datasets" / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_under_test", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dataset_module)
    return dataset_module.RoutingDataset


RoutingDataset = _load_routing_dataset_class()


class TestRiverNetworkSelection(unittest.TestCase):
    def _make_dataset_stub(self, data_home, threshold="4"):
        ds = RoutingDataset.__new__(RoutingDataset)
        ds._data_home = data_home
        ds._model_cfg = {"graph": {"rivernetwork_threshold": threshold}}
        return ds

    def test_find_rivernetwork_graph_file_matches_threshold(self):
        with TemporaryDirectory() as tmp:
            expected = os.path.join(tmp, "river_network_graph_connected_threshold_4.pkl")
            other = os.path.join(tmp, "river_network_graphthreshold_5.pkl")

            open(expected, "wb").close()
            open(other, "wb").close()

            ds = self._make_dataset_stub(tmp, threshold="4")
            found = ds._find_rivernetwork_graph_file("4")

            self.assertEqual(found, expected)

    def test_resolve_rivernetwork_graph_path_triggers_build_when_missing(self):
        with TemporaryDirectory() as tmp:
            ds = self._make_dataset_stub(tmp, threshold="4")
            called = {"threshold": None}

            def fake_build(self, threshold):
                called["threshold"] = threshold
                created = os.path.join(
                    self._data_home,
                    f"river_network_graph_connected_threshold_{threshold}.pkl",
                )
                open(created, "wb").close()

            ds._build_rivernetwork_for_threshold = types.MethodType(fake_build, ds)
            graph_path = ds._resolve_rivernetwork_graph_path()

            self.assertEqual(called["threshold"], "4")
            self.assertTrue(graph_path.endswith("river_network_graph_connected_threshold_4.pkl"))
            self.assertTrue(os.path.exists(graph_path))

    def test_resolve_rivernetwork_graph_path_raises_if_build_does_not_generate(self):
        with TemporaryDirectory() as tmp:
            ds = self._make_dataset_stub(tmp, threshold="4")
            ds._build_rivernetwork_for_threshold = types.MethodType(lambda self, _: None, ds)

            with self.assertRaises(FileNotFoundError):
                ds._resolve_rivernetwork_graph_path()


if __name__ == "__main__":
    unittest.main(exit=False)
