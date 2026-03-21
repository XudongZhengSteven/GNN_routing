import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainers import build_loss


class _DummyDataset:
    outlet_names = ["Zhangjiaba", "Shiquan", "Ankang"]


class TestSingleStationKGELoss(unittest.TestCase):
    def test_single_station_kge_by_name(self):
        dataset = _DummyDataset()
        criterion = build_loss(
            {
                "name": "single_station_kge",
                "station_name": "Shi_quan",
                "eps": 1.0e-6,
            },
            dataset=dataset,
        )

        # Shape: [B, P, O] where O=3 stations.
        target = torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[2.0, 3.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [
                [[100.0, 2.0, -50.0]],   # large errors on non-target stations
                [[-99.0, 3.0, 999.0]],
            ],
            dtype=torch.float32,
        )

        loss = criterion(pred, target)
        self.assertLess(float(loss.detach().cpu()), 1e-4)
        self.assertEqual(int(criterion.station_index), 1)
        self.assertEqual(str(criterion.station_name), "Shiquan")

    def test_single_station_kge_by_index(self):
        criterion = build_loss(
            {
                "name": "single_station_kge",
                "station_index": 2,
            }
        )

        target = torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[2.0, 3.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[2.0, 3.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        loss = criterion(pred, target)
        self.assertLess(float(loss.detach().cpu()), 1e-4)

    def test_single_station_kge_missing_station(self):
        dataset = _DummyDataset()
        with self.assertRaises(ValueError):
            build_loss(
                {
                    "name": "single_station_kge",
                    "station_name": "NotExists",
                },
                dataset=dataset,
            )


if __name__ == "__main__":
    unittest.main(exit=False)
