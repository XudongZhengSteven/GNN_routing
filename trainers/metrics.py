from typing import Dict, List

import torch


def _as_station_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to [T, O] where O is outlet/station dim (last dim).
    """
    if x.dim() == 0:
        return x.reshape(1, 1).to(torch.float32)
    if x.dim() == 1:
        return x.reshape(-1, 1).to(torch.float32)
    return x.reshape(-1, x.shape[-1]).to(torch.float32)


def _nan_scalar_like(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(float("nan"), dtype=x.dtype, device=x.device)


def _compute_kge_components_for_series(pred_1d: torch.Tensor, target_1d: torch.Tensor, eps: float = 1e-12):
    mask = torch.isfinite(pred_1d) & torch.isfinite(target_1d)
    p = pred_1d[mask]
    t = target_1d[mask]
    if p.numel() < 2:
        nan = _nan_scalar_like(pred_1d)
        return nan, nan, nan, nan

    p_mean = p.mean()
    t_mean = t.mean()
    p_std = p.std(unbiased=False)
    t_std = t.std(unbiased=False)
    cov = torch.mean((p - p_mean) * (t - t_mean))

    r = cov / (p_std * t_std + eps)
    alpha = p_std / (t_std + eps)
    beta = p_mean / (t_mean + eps)
    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge, r, alpha, beta


def compute_kge_per_station(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> List[Dict[str, torch.Tensor]]:
    pred_2d = _as_station_matrix(pred)
    target_2d = _as_station_matrix(target)
    if pred_2d.shape != target_2d.shape:
        raise ValueError(f"Shape mismatch for KGE: pred={tuple(pred_2d.shape)}, target={tuple(target_2d.shape)}")

    rows: List[Dict[str, torch.Tensor]] = []
    for i in range(pred_2d.shape[-1]):
        kge, r, alpha, beta = _compute_kge_components_for_series(pred_2d[:, i], target_2d[:, i], eps=eps)
        rows.append(
            {
                "kge": kge,
                "r": r,
                "alpha": alpha,
                "beta": beta,
            }
        )
    return rows


def _nanmean(values: List[torch.Tensor], fallback_like: torch.Tensor) -> torch.Tensor:
    if len(values) == 0:
        return _nan_scalar_like(fallback_like)
    stacked = torch.stack(values)
    finite = torch.isfinite(stacked)
    if not finite.any():
        return _nan_scalar_like(stacked)
    return torch.nanmean(stacked)


def compute_mean_kge(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Mean KGE over station dimension (last dim).
    """
    rows = compute_kge_per_station(pred, target, eps=eps)
    if len(rows) == 0:
        return _nan_scalar_like(pred)
    return _nanmean([r["kge"] for r in rows], fallback_like=rows[0]["kge"])


def compute_kge_summary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    Summary over stations using station-mean aggregation.
    """
    rows = compute_kge_per_station(pred, target, eps=eps)
    if len(rows) == 0:
        return {"kge": float("nan"), "r": float("nan"), "alpha": float("nan"), "beta": float("nan")}

    kge = _nanmean([x["kge"] for x in rows], fallback_like=rows[0]["kge"])
    r = _nanmean([x["r"] for x in rows], fallback_like=rows[0]["r"])
    alpha = _nanmean([x["alpha"] for x in rows], fallback_like=rows[0]["alpha"])
    beta = _nanmean([x["beta"] for x in rows], fallback_like=rows[0]["beta"])
    return {
        "kge": float(kge.detach().cpu()),
        "r": float(r.detach().cpu()),
        "alpha": float(alpha.detach().cpu()),
        "beta": float(beta.detach().cpu()),
    }
