import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    def __init__(self, static_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gammas = nn.ModuleList([nn.Linear(int(static_dim), int(hidden_dim)) for _ in range(int(num_layers))])
        self.betas = nn.ModuleList([nn.Linear(int(static_dim), int(hidden_dim)) for _ in range(int(num_layers))])
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, static_feat: torch.Tensor):
        film_gammas, film_betas = [], []
        for g_layer, b_layer in zip(self.gammas, self.betas):
            film_gammas.append(self.dropout(g_layer(static_feat)))
            film_betas.append(self.dropout(b_layer(static_feat)))
        return film_gammas, film_betas


class FiLMGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.w_x = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.w_h = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, gamma: torch.Tensor = None, beta: torch.Tensor = None):
        x_gates = self.w_x(x)
        h_gates = self.w_h(h_prev)

        x_z, x_r, x_n = torch.chunk(x_gates, 3, dim=-1)
        h_z, h_r, h_n = torch.chunk(h_gates, 3, dim=-1)

        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        n = torch.tanh(x_n + r * h_n)
        h_t = (1.0 - z) * n + z * h_prev

        if gamma is not None and beta is not None:
            h_t = gamma * h_t + beta

        return h_t

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(int(batch_size), self.hidden_dim, device=device)


class FiLMGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cells = nn.ModuleList(
            [
                FiLMGRUCell(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim)
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, initial_state, gammas=None, betas=None):
        # x: [T, BN, F]
        t_steps, batch_nodes, _ = x.shape
        layer_input = x
        final_states = []

        for i_layer, cell in enumerate(self.cells):
            h = initial_state[i_layer]
            outputs = []
            gamma_l = None if gammas is None else gammas[i_layer]
            beta_l = None if betas is None else betas[i_layer]

            for t in range(t_steps):
                h = cell(layer_input[t], h, gamma_l, beta_l)
                outputs.append(h)

            outputs = torch.stack(outputs, dim=0)  # [T, BN, H]
            final_states.append(h)
            layer_input = outputs

        return outputs, final_states

    def init_hidden(self, batch_size: int, device: torch.device):
        return [cell.init_hidden(batch_size, device=device) for cell in self.cells]


class SlopeRouting(nn.Module):
    """
    Capture temporal lag effects conditioned by static node attributes.
    """

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        seq_len: int,
        lstm_units: int,
        num_nodes: int,
        num_lstm_layers: int = 1,
        use_film_lstm: bool = True,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.dyn_dim = int(dyn_dim)
        self.static_dim = int(static_dim)
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.num_lstm_layers = int(num_lstm_layers)
        self.lstm_units = int(lstm_units)
        self.use_film_lstm = bool(use_film_lstm)
        self.act = act

        if not self.use_film_lstm:
            self.fc = nn.Linear(self.dyn_dim, self.lstm_units)

        self.film_gen = FiLMGenerator(self.static_dim, self.lstm_units, self.num_lstm_layers)
        self.film_gru = FiLMGRU(self.dyn_dim, self.lstm_units, num_layers=self.num_lstm_layers)
        self.lin_skip = nn.Linear(self.dyn_dim, self.lstm_units)
        self.layer_norm = nn.LayerNorm(self.lstm_units)

    def forward(self, dyn_x: torch.Tensor, static_feat: torch.Tensor):
        # dyn_x: [BN, T, F_dyn], static_feat: [BN, D_s]
        if dyn_x.dim() != 3 or static_feat.dim() != 2:
            raise ValueError(
                f"`dyn_x` must be [BN,T,F], `static_feat` must be [BN,D], got {tuple(dyn_x.shape)} and {tuple(static_feat.shape)}"
            )
        if dyn_x.shape[0] != static_feat.shape[0]:
            raise ValueError(
                f"Batch-node size mismatch: dyn_x BN={dyn_x.shape[0]}, static_feat BN={static_feat.shape[0]}"
            )

        # [BN, T, F] -> [T, BN, F]
        x_tbf = dyn_x.permute(1, 0, 2).contiguous()

        if self.use_film_lstm:
            gammas, betas = self.film_gen(static_feat)
            init_state = self.film_gru.init_hidden(batch_size=dyn_x.shape[0], device=dyn_x.device)
            outputs, _ = self.film_gru(x_tbf, init_state, gammas, betas)

            x_skip = self.lin_skip(x_tbf)
            outputs = self.layer_norm(outputs + x_skip)
        else:
            outputs = self.fc(x_tbf)

        outputs = self.act(outputs)
        return outputs.permute(1, 0, 2).contiguous()  # [BN, T, H]

