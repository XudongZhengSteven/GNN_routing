from collections.abc import Mapping
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import RoutingDataset


def build_dataset(split: str, dataset_kwargs: Optional[Dict] = None) -> RoutingDataset:
    kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    kwargs["dataset_type"] = split
    return RoutingDataset(**kwargs)


def _sample_to_mapping(sample):
    if isinstance(sample, Mapping):
        return dict(sample)

    if hasattr(sample, "to_dict"):
        try:
            out = sample.to_dict()
            if isinstance(out, Mapping):
                return dict(out)
        except Exception:
            pass

    keys = None
    if hasattr(sample, "keys"):
        try:
            keys_attr = sample.keys
            keys = list(keys_attr()) if callable(keys_attr) else list(keys_attr)
        except Exception:
            keys = None

    if keys:
        out = {}
        for key in keys:
            if isinstance(sample, Mapping):
                out[key] = sample[key]
                continue
            if hasattr(sample, key):
                out[key] = getattr(sample, key)
                continue
            out[key] = sample[key]
        return out

    if hasattr(sample, "__dict__"):
        return {k: v for k, v in vars(sample).items() if not str(k).startswith("_")}

    raise TypeError(f"Unsupported sample type for collation: {type(sample)!r}")


def _collate_values(values):
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)

    if isinstance(first, Mapping):
        keys = list(first.keys())
        return {k: _collate_values([v[k] for v in values]) for k in keys}

    if isinstance(first, (list, tuple)):
        transposed = list(zip(*values))
        collated = [_collate_values(list(v)) for v in transposed]
        return tuple(collated) if isinstance(first, tuple) else collated

    if isinstance(first, (int, float, bool)):
        return torch.tensor(values)

    return values


def routing_collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("Received empty batch in routing_collate_fn.")

    mappings = [_sample_to_mapping(sample) for sample in batch]
    keys = list(mappings[0].keys())
    return {k: _collate_values([m[k] for m in mappings]) for k in keys}


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
        collate_fn=routing_collate_fn,
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
