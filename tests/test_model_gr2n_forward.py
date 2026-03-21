import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import build_dataloader
from datasets.dataset import RoutingDataset
from models import build_model


class TestGR2NForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.data_home = cls.project_root / "data" / "raw" / "case5"
        cls.data_cfg = cls.project_root / "configs" / "data.yaml"
        cls.model_cfg = cls.project_root / "configs" / "model.yaml"

    def setUp(self):
        if not self.data_home.exists():
            self.skipTest(f"data directory not found: {self.data_home}")

    def _build_dataset(self):
        return RoutingDataset(
            dataset_type="train",
            windowsize=16,
            input_freq_per_day=1,
            n_pred=1,
            khop=3,
            data_home=str(self.data_home),
            data_cfg_path=str(self.data_cfg),
            model_cfg_path=str(self.model_cfg),
        )

    def test_gr2n_single_and_batch_forward(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred_single = model(sample)
        self.assertEqual(tuple(pred_single.shape), tuple(sample.y.shape))

        loader = build_dataloader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        with torch.no_grad():
            pred_batch = model(batch)
        self.assertEqual(pred_batch.dim(), 3)
        self.assertEqual(pred_batch.shape[0], 2)
        self.assertEqual(pred_batch.shape[1], sample.y.shape[0])
        self.assertEqual(pred_batch.shape[2], sample.y.shape[1])

    def test_gr2n_seq2seq_single_forward(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n_seq2seq",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred = model(sample)
        self.assertEqual(tuple(pred.shape), tuple(sample.y.shape))

    def test_gr2n_variant_gat_static_no_slope(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
                "input_projector": "mlp",
                "spatial": "gat",
                "edge_weight_mode": "static",
                "heads": 1,
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred = model(sample)
        self.assertEqual(tuple(pred.shape), tuple(sample.y.shape))

    def test_gr2n_variant_nested_component_config(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
                "input_projector": {"name": "mlp", "dropout": 0.0},
                "spatial": {
                    "name": "gat",
                    "use_dynamic_edge_weight": True,
                    "normalize_by_in_degree": True,
                    "edge_weight_max": 5.0,
                    "gat_num_heads": 1,
                    "gat_attn_dropout": 0.0,
                    "gat_negative_slope": 0.2,
                },
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred = model(sample)
        self.assertEqual(tuple(pred.shape), tuple(sample.y.shape))

    def test_gr2n_variant_temporal_attention(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
                "spatial": "gat",
                "temporal": {
                    "name": "attention",
                    "attn_use_tail_window": True,
                    "attn_dropout": 0.0,
                    "attn_temperature": 1.0,
                },
                "heads": 1,
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred = model(sample)
        self.assertEqual(tuple(pred.shape), tuple(sample.y.shape))

    def test_gr2n_seq2seq_variant_temporal_attention(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n_seq2seq",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
                "spatial": "gat",
                "temporal": {
                    "name": "attention",
                    "attn_use_tail_window": True,
                    "attn_tail_steps": 4,
                    "attn_dropout": 0.0,
                    "attn_temperature": 1.0,
                },
                "heads": 1,
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred = model(sample)
        self.assertEqual(tuple(pred.shape), tuple(sample.y.shape))

    def test_gr2n_variant_norm_residual_and_spatial_dropouts(self):
        ds = self._build_dataset()
        model = build_model(
            {
                "name": "gr2n",
                "hidden_dim": 32,
                "num_layers": 2,
                "dropout": 0.0,
                "pos_hidden_dim": 16,
                "spatial": {
                    "name": "gcn",
                    "edge_dropout": 0.1,
                    "message_dropout": 0.1,
                },
                "temporal_block": {
                    "cell_norm_type": "layernorm",
                    "use_layer_residual": True,
                    "layer_residual_dropout": 0.1,
                },
            },
            dataset=ds,
        )
        model.eval()

        sample = ds[0]
        with torch.no_grad():
            pred = model(sample)
        self.assertEqual(tuple(pred.shape), tuple(sample.y.shape))


if __name__ == "__main__":
    unittest.main(exit=False)
