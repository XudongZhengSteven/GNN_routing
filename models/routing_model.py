from typing import Any

import torch
from torch import nn

from .encoder_decoder import GraphGRUDecoder, GraphGRUEncoder
from .input_projector import NodeInputProjector
from .postional_encoding import SpatialPositionalEncoding
from .temporal_model import GraphGRUBlock


def _get_field(batch: Any, key: str):
    if isinstance(batch, dict):
        return batch[key]
    if hasattr(batch, key):
        return getattr(batch, key)
    raise KeyError(f"Batch has no field `{key}`")


def _as_batched(x: torch.Tensor, target_dim: int):
    if x.dim() == target_dim - 1:
        return x.unsqueeze(0), True
    if x.dim() == target_dim:
        return x, False
    raise ValueError(f"Tensor dim mismatch: got {x.dim()}, expected {target_dim-1} or {target_dim}")


def _build_batched_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_size: int, num_nodes: int):
    # edge_index: [2, E], edge_weight: [E] or [B, E]
    edge_index = edge_index.long()
    if edge_weight.dim() == 1:
        edge_weight = edge_weight.unsqueeze(0).expand(batch_size, -1)
    elif edge_weight.dim() == 2:
        if edge_weight.shape[0] == 1 and batch_size > 1:
            edge_weight = edge_weight.expand(batch_size, -1)
        if edge_weight.shape[0] != batch_size:
            raise ValueError(
                f"edge_weight batch mismatch: expected B={batch_size}, got {edge_weight.shape[0]}"
            )
    else:
        raise ValueError(f"Unexpected edge_weight shape: {tuple(edge_weight.shape)}")

    e = edge_index.shape[1]
    device = edge_index.device
    offsets = (torch.arange(batch_size, device=device, dtype=edge_index.dtype) * int(num_nodes)).view(1, batch_size, 1)
    edge_index_batched = edge_index.view(2, 1, e).expand(2, batch_size, e) + offsets
    edge_index_batched = edge_index_batched.reshape(2, batch_size * e).contiguous()
    edge_weight_batched = edge_weight.reshape(batch_size * e)
    return edge_index_batched, edge_weight_batched


class GR2N(nn.Module):
    """
    Graph Routing Recurrent Network aligned with current RoutingDataset.
    Input batch fields:
    - x: [B, N, W, F] or [N, W, F]
    - node_attr: [B, N, D_s] or [N, D_s]
    - mask_downstream_adj, mask_khop_up_adj: [B, N, N] or [N, N]
    - full_path_edge_attr_adj: [B, N, N, F_e] or [N, N, F_e]
    - outlet_index: [B, O] or [O]
    Output:
    - y_pred: [B, P, O] or [P, O]
    """

    def __init__(
        self,
        seq_len: int,
        input_freq_per_day: int,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        num_layers: int,
        edge_attr_dim: int,
        node_attr_dim: int,
        pos_hidden_dim: int = 32,
        dropout: float = 0.1,
        use_film_lag: bool = True,
        input_projector_name: str = "slope",
        input_projector_kwargs: dict = None,
        spatial_model_type: str = "gcn",
        spatial_layer_kwargs: dict = None,
        temporal_readout_type: str = "tail_mean",
        temporal_readout_kwargs: dict = None,
        temporal_block_kwargs: dict = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_freq_per_day = int(input_freq_per_day)
        self.num_nodes = int(num_nodes)
        self.pred_len = int(pred_len)
        self.edge_attr_dim = int(edge_attr_dim)
        self.node_attr_dim = int(node_attr_dim)

        self.input_projector = NodeInputProjector(
            name=str(input_projector_name),
            dyn_dim=int(input_dim),
            static_dim=self.node_attr_dim,
            seq_len=self.seq_len,
            hidden_dim=int(hidden_dim),
            num_nodes=self.num_nodes,
            use_film_lag=bool(use_film_lag),
            dropout=float(dropout),
            projector_kwargs=input_projector_kwargs,
        )
        self.positional_encoding = SpatialPositionalEncoding(
            edge_attr_dim=self.edge_attr_dim,
            hidden_dim=int(pos_hidden_dim),
        )
        self.blocks = GraphGRUBlock(
            input_dim=int(hidden_dim),
            hidden_dim=int(hidden_dim),
            num_nodes=self.num_nodes,
            num_layers=int(num_layers),
            output_dim=self.pred_len,
            dropout=float(dropout),
            input_freq_per_day=self.input_freq_per_day,
            spatial_model_type=spatial_model_type,
            spatial_layer_kwargs=spatial_layer_kwargs,
            temporal_readout_type=temporal_readout_type,
            temporal_readout_kwargs=temporal_readout_kwargs,
            **({} if temporal_block_kwargs is None else dict(temporal_block_kwargs)),
        )

    def forward(self, batch: Any):
        x = _get_field(batch, "x")
        node_attr = _get_field(batch, "node_attr")
        outlet_index = _get_field(batch, "outlet_index").long()
        mask_downstream_adj = _get_field(batch, "mask_downstream_adj")
        mask_khop_up_adj = _get_field(batch, "mask_khop_up_adj")
        full_path_edge_attr_adj = _get_field(batch, "full_path_edge_attr_adj")

        x, squeeze_b = _as_batched(x, 4)                              # [B,N,W,F]
        node_attr, _ = _as_batched(node_attr, 3)                      # [B,N,D]
        outlet_index, _ = _as_batched(outlet_index, 2)                # [B,O]
        mask_downstream_adj, _ = _as_batched(mask_downstream_adj, 3)  # [B,N,N]
        mask_khop_up_adj, _ = _as_batched(mask_khop_up_adj, 3)        # [B,N,N]
        full_path_edge_attr_adj, _ = _as_batched(full_path_edge_attr_adj, 4)  # [B,N,N,Fe]

        batch_size, num_nodes, t_enc, f_dyn = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: model={self.num_nodes}, data={num_nodes}")

        edge_index, edge_weight = self.positional_encoding(
            mask_downstream_adj,
            mask_khop_up_adj,
            full_path_edge_attr_adj,
        )
        edge_index_batched, edge_weight_batched = _build_batched_edges(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch_size=batch_size,
            num_nodes=num_nodes,
        )

        x_flat = x.reshape(batch_size * num_nodes, t_enc, f_dyn)
        node_attr_flat = node_attr.reshape(batch_size * num_nodes, node_attr.shape[-1])
        x_proj = self.input_projector(x_flat, node_attr_flat)

        init_hidden_state = self.blocks.init_hidden(batch_size=batch_size, device=x.device)
        node_pred_flat, _, _ = self.blocks(
            x_proj,
            init_hidden_state,
            edge_index_batched,
            edge_weight_batched,
        )  # [BN, P]

        node_pred = node_pred_flat.view(batch_size, num_nodes, self.pred_len).transpose(1, 2).contiguous()  # [B,P,N]

        if outlet_index.shape[0] == 1 and batch_size > 1:
            outlet_index = outlet_index.expand(batch_size, -1)
        gather_index = outlet_index.unsqueeze(1).expand(-1, self.pred_len, -1)
        y_pred = torch.gather(node_pred, dim=2, index=gather_index)  # [B,P,O]

        if squeeze_b:
            y_pred = y_pred.squeeze(0)
        return y_pred


class GR2NSeq2Seq(nn.Module):
    """
    A seq2seq variant using GraphGRUEncoder + GraphGRUDecoder.
    """

    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        num_layers: int,
        edge_attr_dim: int,
        node_attr_dim: int,
        pos_hidden_dim: int = 32,
        dropout: float = 0.1,
        use_film_lag: bool = True,
        input_projector_name: str = "slope",
        input_projector_kwargs: dict = None,
        spatial_model_type: str = "gcn",
        spatial_layer_kwargs: dict = None,
        temporal_readout_type: str = "tail_mean",
        temporal_readout_kwargs: dict = None,
        temporal_block_kwargs: dict = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_nodes = int(num_nodes)
        self.pred_len = int(pred_len)
        self.edge_attr_dim = int(edge_attr_dim)
        self.node_attr_dim = int(node_attr_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_projector = NodeInputProjector(
            name=str(input_projector_name),
            dyn_dim=int(input_dim),
            static_dim=self.node_attr_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            use_film_lag=bool(use_film_lag),
            dropout=float(dropout),
            projector_kwargs=input_projector_kwargs,
        )
        self.positional_encoding = SpatialPositionalEncoding(
            edge_attr_dim=self.edge_attr_dim,
            hidden_dim=int(pos_hidden_dim),
        )
        block_kwargs = {} if temporal_block_kwargs is None else dict(temporal_block_kwargs)

        self.encoder = GraphGRUEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=int(num_layers),
            spatial_model_type=spatial_model_type,
            spatial_layer_kwargs=spatial_layer_kwargs,
            **block_kwargs,
        )
        self.decoder = GraphGRUDecoder(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=int(num_layers),
            output_dim=1,
            pred_steps=self.pred_len,
            spatial_model_type=spatial_model_type,
            spatial_layer_kwargs=spatial_layer_kwargs,
            temporal_attention_type=temporal_readout_type,
            temporal_attention_kwargs=temporal_readout_kwargs,
            **block_kwargs,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, batch: Any):
        x = _get_field(batch, "x")
        node_attr = _get_field(batch, "node_attr")
        outlet_index = _get_field(batch, "outlet_index").long()
        mask_downstream_adj = _get_field(batch, "mask_downstream_adj")
        mask_khop_up_adj = _get_field(batch, "mask_khop_up_adj")
        full_path_edge_attr_adj = _get_field(batch, "full_path_edge_attr_adj")

        x, squeeze_b = _as_batched(x, 4)
        node_attr, _ = _as_batched(node_attr, 3)
        outlet_index, _ = _as_batched(outlet_index, 2)
        mask_downstream_adj, _ = _as_batched(mask_downstream_adj, 3)
        mask_khop_up_adj, _ = _as_batched(mask_khop_up_adj, 3)
        full_path_edge_attr_adj, _ = _as_batched(full_path_edge_attr_adj, 4)

        batch_size, num_nodes, t_enc, f_dyn = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: model={self.num_nodes}, data={num_nodes}")

        edge_index, edge_weight = self.positional_encoding(
            mask_downstream_adj,
            mask_khop_up_adj,
            full_path_edge_attr_adj,
        )
        edge_index_batched, edge_weight_batched = _build_batched_edges(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch_size=batch_size,
            num_nodes=num_nodes,
        )

        x_flat = x.reshape(batch_size * num_nodes, t_enc, f_dyn)
        node_attr_flat = node_attr.reshape(batch_size * num_nodes, node_attr.shape[-1])
        x_proj = self.dropout(self.input_projector(x_flat, node_attr_flat))

        init_hidden = self.encoder.init_hidden(batch_size=batch_size, device=x.device)
        encoder_hidden, encoder_outputs = self.encoder(
            x_proj, init_hidden, edge_index_batched, edge_weight_batched
        )
        node_seq_pred, _ = self.decoder(
            encoder_hidden, encoder_outputs, edge_index_batched, edge_weight_batched
        )  # [BN, P, 1]

        node_seq_pred = node_seq_pred.squeeze(-1)  # [BN, P]
        node_pred = node_seq_pred.view(batch_size, num_nodes, self.pred_len).transpose(1, 2).contiguous()  # [B,P,N]

        if outlet_index.shape[0] == 1 and batch_size > 1:
            outlet_index = outlet_index.expand(batch_size, -1)
        gather_index = outlet_index.unsqueeze(1).expand(-1, self.pred_len, -1)
        y_pred = torch.gather(node_pred, dim=2, index=gather_index)

        if squeeze_b:
            y_pred = y_pred.squeeze(0)
        return y_pred
