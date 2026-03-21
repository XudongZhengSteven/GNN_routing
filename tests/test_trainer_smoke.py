import os
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_model
from trainers import Trainer, build_loss, build_optimizer, build_scheduler, set_seed


class TinyRoutingDataset(Dataset):
    def __init__(self, num_samples=24, num_nodes=12, window=8, n_pred=1, num_outlets=4):
        self.num_samples = int(num_samples)
        self.num_nodes = int(num_nodes)
        self.window = int(window)
        self.n_pred = int(n_pred)
        self.num_outlets = int(num_outlets)
        self.outlet_index = torch.tensor([0, 3, 7, 10][: self.num_outlets], dtype=torch.long)

        edges = [[i, i + 1] for i in range(self.num_nodes - 1)]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
        self.edge_attr = torch.ones(self.edge_index.shape[1], 4, dtype=torch.float32)
        self.node_attr_base = torch.randn(self.num_nodes, 3, dtype=torch.float32)

        self.mask_downstream_adj = torch.eye(self.num_nodes, dtype=torch.float32)
        self.mask_khop_up_adj = torch.eye(self.num_nodes, dtype=torch.float32)
        self.full_path_edge_attr_adj = torch.zeros(self.num_nodes, self.num_nodes, 4, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(int(idx) + 123)
        x = torch.randn(self.num_nodes, self.window, 2, generator=g)
        node_attr = self.node_attr_base + 0.01 * torch.randn(self.num_nodes, 3, generator=g)

        # Construct target from last-step node signal at outlet nodes.
        y_now = x[:, -1, 0][self.outlet_index]
        y = y_now.unsqueeze(0).repeat(self.n_pred, 1)

        return {
            "x": x,
            "y": y,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "node_attr": node_attr,
            "outlet_index": self.outlet_index,
            "mask_downstream_adj": self.mask_downstream_adj,
            "mask_khop_up_adj": self.mask_khop_up_adj,
            "full_path_edge_attr_adj": self.full_path_edge_attr_adj,
        }


class TestTrainerSmoke(unittest.TestCase):
    def test_train_and_eval_smoke(self):
        set_seed(42)
        train_ds = TinyRoutingDataset(num_samples=20)
        val_ds = TinyRoutingDataset(num_samples=8)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        model = build_model(
            {
                "name": "routing_baseline",
                "input_dim": 2,
                "node_attr_dim": 3,
                "hidden_dim": 32,
                "pred_len": 1,
                "dropout": 0.0,
            }
        )
        criterion = build_loss({"name": "mse"})
        optimizer = build_optimizer(model.parameters(), {"name": "adam", "lr": 1e-3})
        scheduler = build_scheduler(optimizer, {"name": "step", "step_size": 1, "gamma": 0.9}, total_epochs=2, steps_per_epoch=len(train_loader))

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
                checkpoint_dir=tmpdir,
                monitor="val_loss",
                monitor_mode="min",
                early_stopping_patience=5,
                log_interval=1000,
                keep_last_k=2,
            )
            history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=2)
            self.assertGreaterEqual(len(history), 1)
            self.assertTrue(os.path.exists(trainer.get_last_checkpoint_path()))
            self.assertTrue(os.path.exists(trainer.get_best_checkpoint_path()))

            metrics = trainer.evaluate(val_loader, split="val")
            self.assertIn("loss", metrics)
            self.assertIn("mae", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("nse", metrics)


if __name__ == "__main__":
    unittest.main(exit=False)
