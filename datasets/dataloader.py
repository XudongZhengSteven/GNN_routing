from typing import Dict, Optional, Tuple

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch.utils.data import DataLoader

from .dataset import RoutingDataset


def build_dataset(split: str, dataset_kwargs: Optional[Dict] = None) -> RoutingDataset:
    kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    kwargs["dataset_type"] = split
    return RoutingDataset(**kwargs)


def build_dataloader(
    dataset: RoutingDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    use_balance_sampler: bool = False,
):
    sampler = None
    if use_balance_sampler:
        sampler, _ = dataset.get_balance_weight()
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_train_val_dataloaders(
    dataset_kwargs: Optional[Dict] = None,
    train_loader_kwargs: Optional[Dict] = None,
    val_loader_kwargs: Optional[Dict] = None,
) -> Tuple[RoutingDataset, RoutingDataset, DataLoader, DataLoader]:
    train_dataset = build_dataset("train", dataset_kwargs=dataset_kwargs)
    val_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    val_kwargs["normalizers"] = train_dataset.normalizers
    val_dataset = build_dataset("val", dataset_kwargs=val_kwargs)

    train_loader_cfg = {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
        "use_balance_sampler": False,
    }
    if train_loader_kwargs:
        train_loader_cfg.update(train_loader_kwargs)

    val_loader_cfg = {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
        "use_balance_sampler": False,
    }
    if val_loader_kwargs:
        val_loader_cfg.update(val_loader_kwargs)

    train_loader = build_dataloader(train_dataset, **train_loader_cfg)
    val_loader = build_dataloader(val_dataset, **val_loader_cfg)

    return train_dataset, val_dataset, train_loader, val_loader
