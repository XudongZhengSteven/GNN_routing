import os
import sys
import unittest

import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataloader import build_dataloader


class _DummyData:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, node_attr: torch.Tensor, outlet_index: torch.Tensor):
        self.x = x
        self.y = y
        self.node_attr = node_attr
        self.outlet_index = outlet_index

    def keys(self):
        return ["x", "y", "node_attr", "outlet_index"]


class _DummyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        base = float(idx)
        x = torch.full((3, 5, 2), fill_value=base, dtype=torch.float32)  # [N,W,F]
        y = torch.full((1, 2), fill_value=base, dtype=torch.float32)      # [P,O]
        node_attr = torch.full((3, 3), fill_value=base, dtype=torch.float32)
        outlet_index = torch.tensor([0, 2], dtype=torch.long)
        return _DummyData(x=x, y=y, node_attr=node_attr, outlet_index=outlet_index)


class TestDataloaderCollate(unittest.TestCase):
    def test_collate_data_like_objects_to_batched_dict(self):
        ds = _DummyDataset()
        loader = build_dataloader(
            dataset=ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            use_balance_sampler=False,
        )
        batch = next(iter(loader))

        self.assertIsInstance(batch, dict)
        self.assertEqual(tuple(batch["x"].shape), (2, 3, 5, 2))
        self.assertEqual(tuple(batch["y"].shape), (2, 1, 2))
        self.assertEqual(tuple(batch["node_attr"].shape), (2, 3, 3))
        self.assertEqual(tuple(batch["outlet_index"].shape), (2, 2))


if __name__ == "__main__":
    unittest.main(exit=False)
