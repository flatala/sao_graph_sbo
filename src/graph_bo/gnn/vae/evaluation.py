from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from graph_bo.gnn.vae.model import GIN_VAE
from graph_bo.gnn.vae.training import adjacency_metrics, collate_dense_vae_batch, feature_metrics


def plot_vae_training_curves(history: pd.DataFrame, show: bool = True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, key, title in zip(
        axes[0],
        ["loss_recon", "loss_adj", "loss_features"],
        ["Reconstruction loss", "Adjacency BCE", "Feature loss"],
    ):
        if f"train_{key}" in history:
            ax.plot(history["epoch"], history[f"train_{key}"], label="train")
            ax.plot(history["epoch"], history[f"val_{key}"], label="val", linestyle="--")
        ax.set_title(title)
        ax.legend()

    for ax, key, title in zip(
        axes[1],
        ["val_adj_precision", "val_adj_recall", "val_adj_f1"],
        [
            "Validation adjacency precision",
            "Validation adjacency recall",
            "Validation adjacency F1",
        ],
    ):
        if key in history:
            ax.plot(history["epoch"], history[key], color="tab:green")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.0, 1.0)
    plt.suptitle("VAE reconstruction training curves")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def evaluate_vae_reconstruction(
    model: GIN_VAE,
    loader,
    device: torch.device | str,
    adj_cutoff: float,
    print_metrics: bool = True,
) -> dict[str, float]:
    model.eval()
    metric_values = {
        "adj_accuracy": [],
        "adj_precision": [],
        "adj_recall": [],
        "adj_f1": [],
        "adj_edge_rate": [],
        "feature_mae": [],
    }

    with torch.no_grad():
        for x, A, mask, A_target, X_target in loader:
            x = x.to(device)
            A = A.to(device)
            mask = mask.to(device)
            A_target = A_target.to(device)
            X_target = X_target.to(device)

            outputs = model(x, A)
            adj_metrics = adjacency_metrics(outputs["A_logits"], A_target, mask, adj_cutoff)
            feat_metrics = feature_metrics(outputs["X_logits"], X_target, mask)

            for key, values in adj_metrics.items():
                metric_values[key].extend(values.cpu().tolist())
            metric_values["feature_mae"].extend(feat_metrics["feature_mae"].cpu().tolist())

    summary = {}
    for key, values in metric_values.items():
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    if print_metrics:
        print_vae_reconstruction_metrics(summary, adj_cutoff)
    return summary


def print_vae_reconstruction_metrics(metrics: dict[str, float], adj_cutoff: float) -> None:
    print(f"Adjacency edge rate on active upper-triangle pairs: {metrics['adj_edge_rate_mean']:.3f} +/- {metrics['adj_edge_rate_std']:.3f}")
    print(f"Adjacency precision on active upper-triangle pairs @ {adj_cutoff:.2f}: {metrics['adj_precision_mean']:.3f} +/- {metrics['adj_precision_std']:.3f}")
    print(f"Adjacency recall on active upper-triangle pairs @ {adj_cutoff:.2f}: {metrics['adj_recall_mean']:.3f} +/- {metrics['adj_recall_std']:.3f}")
    print(f"Adjacency F1 on active upper-triangle pairs @ {adj_cutoff:.2f}: {metrics['adj_f1_mean']:.3f} +/- {metrics['adj_f1_std']:.3f}")
    print(f"Adjacency accuracy on active upper-triangle pairs @ {adj_cutoff:.2f}: {metrics['adj_accuracy_mean']:.3f} +/- {metrics['adj_accuracy_std']:.3f}")
    print(f"Feature MAE on active nodes: {metrics['feature_mae_mean']:.3f} +/- {metrics['feature_mae_std']:.3f}")


def plot_random_vae_reconstruction(
    model: GIN_VAE,
    dataset,
    device: torch.device | str,
    adj_cutoff: float,
    rng: np.random.Generator | None = None,
    sample_index: int | None = None,
    plot_scope: str = "active",
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if plot_scope not in {"active", "parent"}:
        raise ValueError(f"plot_scope must be 'active' or 'parent', got {plot_scope!r}")

    if rng is None:
        rng = np.random.default_rng()
    if sample_index is None:
        sample_index = int(rng.integers(len(dataset)))

    graph = dataset[sample_index]
    x, A, mask, A_target, X_target = collate_dense_vae_batch([graph])
    x = x.to(device)
    A = A.to(device)
    mask = mask.to(device)
    A_target = A_target.to(device)
    X_target = X_target.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x, A)

    A_true = A_target.squeeze(0).cpu().numpy()
    A_prob = outputs["A_logits"].sigmoid().squeeze(0).cpu().numpy()
    np.fill_diagonal(A_true, 0.0)
    np.fill_diagonal(A_prob, 0.0)
    A_pred = (A_prob >= adj_cutoff).astype(float)
    np.fill_diagonal(A_pred, 0.0)

    node_mask = mask.squeeze(0).cpu().numpy()
    active_node_idx = np.where(node_mask > 0)[0]
    sample_id = getattr(graph, "sample_id", sample_index)

    print(f"Held-out sample_id={sample_id}, dataset index={sample_index}, active nodes={len(active_node_idx)}")
    print(f"Adjacency cutoff: {adj_cutoff:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("white")
    if plot_scope == "active":
        display_idx = active_node_idx
        axis_label = "active node index"
        active_pair_display = None
    else:
        display_idx = np.arange(A_true.shape[0])
        axis_label = "parent ADSG node index"
        active_pair_display = np.outer(node_mask > 0, node_mask > 0)
        np.fill_diagonal(active_pair_display, False)

    plot_items = [
        (A_true, "True adjacency"),
        (A_prob, "Predicted adjacency probability"),
        (A_pred, "Predicted adjacency thresholded"),
    ]
    for plot_i, (ax, (matrix, title)) in enumerate(zip(axes, plot_items)):
        ax.set_facecolor("white")
        display_matrix = matrix[np.ix_(display_idx, display_idx)].copy()
        np.fill_diagonal(display_matrix, 0.0)
        if active_pair_display is not None and plot_i != 1:
            active_pair_matrix = active_pair_display[np.ix_(display_idx, display_idx)]
            display_matrix[~active_pair_matrix] = 0.0
        ax.imshow(display_matrix, cmap="Blues", vmin=0, vmax=1, interpolation="none")
        ax.set_title(title)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")
    plt.tight_layout()

    X_pred = outputs["X_logits"].sigmoid().squeeze(0).cpu().numpy()
    X_true = X_target.squeeze(0).cpu().numpy()
    feature_names = [f"feature_{i}" for i in range(X_true.shape[1])]
    active_node_labels = [f"node_{i}" for i in active_node_idx]
    true_features = pd.DataFrame(X_true[active_node_idx], index=active_node_labels, columns=feature_names)
    pred_features = pd.DataFrame(X_pred[active_node_idx], index=active_node_labels, columns=feature_names)
    feature_comparison = pd.concat({"true": true_features, "pred": pred_features}, axis=1)

    return {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "active_node_idx": active_node_idx,
        "figure": fig,
        "feature_comparison": feature_comparison,
        "A_true": A_true,
        "A_prob": A_prob,
        "A_pred": A_pred,
        "X_true": X_true,
        "X_pred": X_pred,
    }


def print_random_vae_feature_comparison(
    model: GIN_VAE,
    dataset,
    builder,
    device: torch.device | str,
    rng: np.random.Generator | None = None,
    sample_index: int | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    if sample_index is None:
        sample_index = int(rng.integers(len(dataset)))

    graph = dataset[sample_index]
    x, A, mask, A_target, X_target = collate_dense_vae_batch([graph])
    x = x.to(device)
    A = A.to(device)
    mask = mask.to(device)
    X_target = X_target.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x, A)

    active_node_idx = np.where(mask.squeeze(0).cpu().numpy() > 0)[0]
    node_row = int(rng.integers(len(active_node_idx)))
    node_idx = int(active_node_idx[node_row])

    true_vec = X_target.squeeze(0).cpu().numpy()[node_idx].copy()
    pred_vec = outputs["X_logits"].sigmoid().squeeze(0).cpu().numpy()[node_idx].copy()
    display_true = true_vec.copy()
    display_pred = pred_vec.copy()

    binary_idx = np.asarray(builder.binary_feature_indices, dtype=int)
    continuous_idx = np.asarray(builder.continuous_feature_indices, dtype=int)
    if binary_idx.size > 0:
        display_true[binary_idx] = np.rint(display_true[binary_idx])
        display_pred[binary_idx] = np.rint(display_pred[binary_idx])
    if continuous_idx.size > 0:
        display_true[continuous_idx] = np.round(display_true[continuous_idx], 3)
        display_pred[continuous_idx] = np.round(display_pred[continuous_idx], 3)

    sample_id = getattr(graph, "sample_id", sample_index)
    print(f"sample_id={sample_id}, dataset index={sample_index}, node_idx={node_idx}")
    print("true")
    print(np.array2string(display_true, precision=3, suppress_small=True))
    print("predicted")
    print(np.array2string(display_pred, precision=3, suppress_small=True))

    return {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "node_idx": node_idx,
        "true": true_vec,
        "predicted": pred_vec,
        "display_true": display_true,
        "display_predicted": display_pred,
    }
