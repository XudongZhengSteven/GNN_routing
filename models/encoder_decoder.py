import torch
from torch import nn
from typing import Dict, Optional

from .temporal_model import GraphGRUCell


class GraphGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_layers: int,
        spatial_model_type: str = "gcn",
        spatial_layer_kwargs: Optional[Dict] = None,
        use_layer_residual: bool = False,
        layer_residual_dropout: float = 0.0,
        cell_norm_type: str = "none",
        cell_norm_eps: float = 1e-5,
        cell_norm_affine: bool = True,
        cell_batchnorm_momentum: float = 0.1,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.use_layer_residual = bool(use_layer_residual)
        self.layer_residual_dropout = nn.Dropout(float(layer_residual_dropout))

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            cell_input_dim = int(input_dim) if i == 0 else self.hidden_dim
            self.cells.append(
                GraphGRUCell(
                    input_dim=cell_input_dim,
                    hidden_dim=self.hidden_dim,
                    num_nodes=self.num_nodes,
                    spatial_model_type=spatial_model_type,
                    spatial_layer_kwargs=spatial_layer_kwargs,
                    norm_type=cell_norm_type,
                    norm_eps=cell_norm_eps,
                    norm_affine=cell_norm_affine,
                    batchnorm_momentum=cell_batchnorm_momentum,
                )
            )

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        # x: [BN, T, F]
        if x.dim() != 3:
            raise ValueError(f"`x` must be [BN,T,F], got {tuple(x.shape)}")

        current_inputs = x
        output_hidden = []

        for i_layer in range(self.num_layers):
            residual_input = current_inputs
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(current_inputs.shape[1]):
                out_t, hidden_state = self.cells[i_layer](
                    current_inputs[:, t, :],
                    hidden_state,
                    edge_index,
                    edge_weight,
                )
                output_inner.append(out_t.unsqueeze(1))
            output_inner = torch.cat(output_inner, dim=1)
            if (
                self.use_layer_residual
                and i_layer > 0
                and output_inner.shape == residual_input.shape
            ):
                output_inner = output_inner + self.layer_residual_dropout(residual_input)
            output_hidden.append(hidden_state)
            current_inputs = output_inner

        return output_hidden, current_inputs

    def init_hidden(self, batch_size: int, device: torch.device):
        init_states = []
        for cell in self.cells:
            init_states.append(cell.init_hidden(batch_size, device=device))
        return torch.stack(init_states, dim=0)


class GraphGRUDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_layers: int,
        output_dim: int,
        pred_steps: int,
        spatial_model_type: str = "gcn",
        spatial_layer_kwargs: Optional[Dict] = None,
        temporal_attention_type: str = "none",
        temporal_attention_kwargs: Optional[Dict] = None,
        use_layer_residual: bool = False,
        layer_residual_dropout: float = 0.0,
        cell_norm_type: str = "none",
        cell_norm_eps: float = 1e-5,
        cell_norm_affine: bool = True,
        cell_batchnorm_momentum: float = 0.1,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.pred_steps = int(pred_steps)
        self.use_layer_residual = bool(use_layer_residual)
        self.layer_residual_dropout = nn.Dropout(float(layer_residual_dropout))

        self.cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.cells.append(
                GraphGRUCell(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    num_nodes=self.num_nodes,
                    spatial_model_type=spatial_model_type,
                    spatial_layer_kwargs=spatial_layer_kwargs,
                    norm_type=cell_norm_type,
                    norm_eps=cell_norm_eps,
                    norm_affine=cell_norm_affine,
                    batchnorm_momentum=cell_batchnorm_momentum,
                )
            )
        self.proj_out = nn.Linear(self.hidden_dim, self.output_dim)

        attn_type = str(temporal_attention_type).strip().lower()
        self.use_temporal_attention = attn_type in {
            "attention",
            "attn",
            "temporal_attention",
            "time_attention",
            "gru_attn",
            "temporal_attn",
        }
        attn_cfg = {} if temporal_attention_kwargs is None else dict(temporal_attention_kwargs)
        self.attn_use_tail_window = bool(attn_cfg.get("attn_use_tail_window", False))
        attn_tail_steps = attn_cfg.get("attn_tail_steps", None)
        self.attn_tail_steps = None if attn_tail_steps is None else int(attn_tail_steps)
        self.attn_dropout = nn.Dropout(float(attn_cfg.get("attn_dropout", 0.0)))
        self.attn_temperature = float(attn_cfg.get("attn_temperature", 1.0))
        self.last_attention_weights = None

        if self.use_temporal_attention:
            self.attn_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attn_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attn_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.attn_fuse = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        else:
            self.attn_query = None
            self.attn_key = None
            self.attn_value = None
            self.attn_fuse = None

    def _prepare_attention_memory(self, encoder_outputs: torch.Tensor):
        # encoder_outputs: [BN, T, H]
        enc = encoder_outputs
        if self.attn_use_tail_window:
            if self.attn_tail_steps is None:
                tail_steps = enc.shape[1]
            else:
                tail_steps = min(max(1, self.attn_tail_steps), enc.shape[1])
            enc = enc[:, -tail_steps:, :]

        return self.attn_key(enc), self.attn_value(enc)  # [BN, T, H], [BN, T, H]

    def _build_context(self, query: torch.Tensor, key_memory: torch.Tensor, value_memory: torch.Tensor):
        # query: [BN, H], key/value memory: [BN, T, H]
        q = self.attn_query(query)  # [BN, H]

        scores = torch.einsum("bth,bh->bt", key_memory, q)
        scores = scores / (self.hidden_dim ** 0.5)
        scores = scores / max(self.attn_temperature, 1e-6)

        alpha = torch.softmax(scores, dim=1)
        alpha = self.attn_dropout(alpha)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)

        context = torch.einsum("bt,bth->bh", alpha, value_memory)  # [BN, H]
        return context, alpha

    def forward(
        self,
        encoder_hidden,
        current_inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        # current_inputs: [BN, T, H]
        if current_inputs.dim() != 3:
            raise ValueError(f"`current_inputs` must be [BN,T,H], got {tuple(current_inputs.shape)}")

        current_input = current_inputs[:, -1, :]  # [BN, H]
        hiddens = [h.clone() for h in encoder_hidden]
        outputs = []
        attn_history = []
        key_memory = None
        value_memory = None
        if self.use_temporal_attention:
            key_memory, value_memory = self._prepare_attention_memory(current_inputs)

        for _ in range(self.pred_steps):
            x_in = current_input
            next_hiddens = []
            for i_layer, cell in enumerate(self.cells):
                residual_input = x_in
                out_l, h_new = cell(x_in, hiddens[i_layer], edge_index, edge_weight)
                if (
                    self.use_layer_residual
                    and i_layer > 0
                    and out_l.shape == residual_input.shape
                ):
                    out_l = out_l + self.layer_residual_dropout(residual_input)
                next_hiddens.append(h_new)
                x_in = out_l

            hiddens = next_hiddens
            step_feature = hiddens[-1]
            if self.use_temporal_attention:
                context, alpha = self._build_context(step_feature, key_memory, value_memory)
                step_feature = torch.tanh(self.attn_fuse(torch.cat([step_feature, context], dim=-1)))
                attn_history.append(alpha)

            step_out = self.proj_out(step_feature)  # [BN, O]
            outputs.append(step_out)
            current_input = step_feature

        outputs = torch.stack(outputs, dim=1)  # [BN, pred_steps, O]
        if self.use_temporal_attention and len(attn_history) > 0:
            self.last_attention_weights = torch.stack(attn_history, dim=1).detach()  # [BN, P, T]
        else:
            self.last_attention_weights = None
        return outputs, hiddens
