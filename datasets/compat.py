import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.utils import to_dense_adj
except ImportError:
    class InMemoryDataset(TorchDataset):
        def __len__(self):
            return self.len()

        def __getitem__(self, idx):
            return self.get(idx)

    class Data(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__ = self

    def to_dense_adj(edge_index, max_num_nodes=None):
        if max_num_nodes is None:
            max_num_nodes = int(edge_index.max().item()) + 1

        adj = torch.zeros((max_num_nodes, max_num_nodes), dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        return adj.unsqueeze(0)
