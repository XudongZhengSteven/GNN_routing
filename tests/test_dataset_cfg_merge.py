import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.evaluate import merge_dataset_cfg as merge_dataset_cfg_eval
from scripts.train import merge_dataset_cfg as merge_dataset_cfg_train


class TestDatasetConfigMerge(unittest.TestCase):
    def test_model_yaml_dataset_overrides_train_yaml_dataset(self):
        train_cfg = {
            "dataset": {
                "windowsize": 16,
                "input_freq_per_day": 1,
                "n_pred": 1,
            }
        }
        model_cfg = {
            "dataset": {
                "windowsize": 24,
            }
        }

        merged_train = merge_dataset_cfg_train(model_cfg, train_cfg)
        merged_eval = merge_dataset_cfg_eval(model_cfg, train_cfg)

        self.assertEqual(merged_train["windowsize"], 24)
        self.assertEqual(merged_eval["windowsize"], 24)
        self.assertEqual(merged_train["input_freq_per_day"], 1)
        self.assertEqual(merged_eval["n_pred"], 1)


if __name__ == "__main__":
    unittest.main(exit=False)
