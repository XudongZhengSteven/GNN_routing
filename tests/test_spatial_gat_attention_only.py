import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.spatial_model import GraphAttentionPosEnc


class TestSpatialGATAttentionOnly(unittest.TestCase):
    def test_gat_output_is_independent_of_input_edge_weight(self):
        torch.manual_seed(42)
        layer = GraphAttentionPosEnc(
            input_dim=5,
            output_dim=6,
            dropout=0.0,
            activation=None,
            num_heads=2,
            attn_dropout=0.0,
        )
        layer.eval()

        num_nodes = 4
        x = torch.randn(num_nodes, 2)
        state = torch.randn(num_nodes, 3)
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 1, 2],
                [1, 2, 3, 0, 0, 1],
            ],
            dtype=torch.long,
        )

        edge_weight_a = torch.ones(edge_index.shape[1], dtype=torch.float32)
        edge_weight_b = torch.linspace(0.1, 2.0, steps=edge_index.shape[1], dtype=torch.float32)

        with torch.no_grad():
            out_a = layer(x, state, edge_index, edge_weight_a)
            out_b = layer(x, state, edge_index, edge_weight_b)

        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-7, rtol=1e-6))


if __name__ == "__main__":
    unittest.main(exit=False)
