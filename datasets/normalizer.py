import torch


class FeatureNormalizer:
    def __init__(self, use_log: bool = False, method: str = "zscore", eps: float = 1e-6):
        self.use_log = use_log
        self.method = method
        self.eps = eps

        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().clone().to(torch.float32)
        return torch.tensor(x, dtype=torch.float32)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_log:
            # Clamp to avoid invalid log for negative values.
            x = torch.log1p(torch.clamp(x, min=0.0))
        return x

    @staticmethod
    def _valid_values(x: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x)
        if mask.any():
            return x[mask]
        return torch.tensor([0.0], dtype=torch.float32)

    def fit(self, x):
        x = self._preprocess(self._to_tensor(x))
        valid = self._valid_values(x)

        if self.method == "zscore":
            self.mean = valid.mean()
            self.std = valid.std(unbiased=False)
            if self.std < self.eps:
                self.std = torch.tensor(1.0, dtype=torch.float32)

        elif self.method in {"minmax", "minmax-11"}:
            self.min = valid.min()
            self.max = valid.max()
            if (self.max - self.min) < self.eps:
                self.max = self.min + 1.0

    def transform(self, x):
        x = self._preprocess(self._to_tensor(x))

        if self.method == "zscore":
            x_norm = (x - self.mean) / (self.std + self.eps)
        elif self.method == "minmax":
            x_norm = (x - self.min) / (self.max - self.min + self.eps)
        elif self.method == "minmax-11":
            x_norm = 2.0 * (x - self.min) / (self.max - self.min + self.eps) - 1.0
        else:
            x_norm = x

        return torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    def inverse_transform(self, x):
        x = self._to_tensor(x)

        if self.method == "zscore":
            x_orig = x * (self.std + self.eps) + self.mean
        elif self.method == "minmax":
            x_orig = x * (self.max - self.min + self.eps) + self.min
        elif self.method == "minmax-11":
            x_orig = (x + 1.0) / 2.0 * (self.max - self.min + self.eps) + self.min
        else:
            x_orig = x

        if self.use_log:
            x_orig = torch.expm1(x_orig)

        return x_orig
