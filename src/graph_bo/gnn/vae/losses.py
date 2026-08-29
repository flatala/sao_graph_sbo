from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def full_adj_bce(A_logits: Tensor, A_target: Tensor) -> Tensor:
    # BCE over every off-diagonal entry of the padded adjacency, uniformly weighted.
    eye = torch.eye(A_logits.shape[1], dtype=A_logits.dtype, device=A_logits.device).unsqueeze(0)
    M = (1.0 - eye).expand_as(A_logits)
    bce = F.binary_cross_entropy_with_logits(A_logits, A_target, reduction="none")
    return (bce * M).sum() / M.sum().clamp(min=1.0)


def mixed_feature_loss(
    X_logits: Tensor,
    X_target: Tensor,
    binary_feature_indices: Sequence[int] | Tensor | None = None,
    continuous_feature_indices: Sequence[int] | Tensor | None = None,
) -> dict[str, Tensor]:
    if binary_feature_indices is None:
        binary_feature_indices = tuple(range(X_target.shape[-1]))
    if continuous_feature_indices is None:
        continuous_feature_indices = ()

    zero = X_logits.sum() * 0.0

    binary_idx = torch.as_tensor(binary_feature_indices, dtype=torch.long, device=X_logits.device)
    if binary_idx.numel() > 0:
        binary_logits = X_logits.index_select(-1, binary_idx)
        binary_target = X_target.index_select(-1, binary_idx)
        l_binary = F.binary_cross_entropy_with_logits(binary_logits, binary_target)
    else:
        l_binary = zero

    continuous_idx = torch.as_tensor(continuous_feature_indices, dtype=torch.long, device=X_logits.device)
    if continuous_idx.numel() > 0:
        continuous_pred = X_logits.index_select(-1, continuous_idx).sigmoid()
        continuous_target = X_target.index_select(-1, continuous_idx)
        l_continuous = F.mse_loss(continuous_pred, continuous_target)
    else:
        l_continuous = zero

    total = l_binary + l_continuous
    return {
        "loss_features": total,
        "loss_features_binary": l_binary,
        "loss_features_continuous": l_continuous,
    }


def kl_divergence(mu: Tensor, log_var: Tensor) -> Tensor:
    kl_nodes = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)
    return kl_nodes.mean()


def vae_loss_full(
    outputs: dict,
    A_target: Tensor,
    X_target: Tensor,
    beta: float = 1.0,
    binary_feature_indices: Sequence[int] | Tensor | None = None,
    continuous_feature_indices: Sequence[int] | Tensor | None = None,
) -> dict[str, Tensor]:
    # arch2vec-style: reconstruct the whole padded graph, inactive rows/cols included,
    # every entry weighted equally.
    l_adj = full_adj_bce(outputs["A_logits"], A_target)
    feature_losses = mixed_feature_loss(
        outputs["X_logits"],
        X_target,
        binary_feature_indices=binary_feature_indices,
        continuous_feature_indices=continuous_feature_indices,
    )
    l_features = feature_losses["loss_features"]
    l_kl = kl_divergence(outputs["mu"], outputs["log_var"])

    total = l_adj + l_features + beta * l_kl
    losses = {
        "loss": total,
        "loss_adj": l_adj,
        "loss_kl": l_kl,
    }
    losses.update(feature_losses)
    return losses