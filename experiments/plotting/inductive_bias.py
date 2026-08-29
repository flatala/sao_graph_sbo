from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
INDUCTIVE_ROOT = REPO_ROOT / "experiments" / "kernels" / "inductive_bias_ablation"
OUTPUT_DIR = REPO_ROOT / "experiments" / "plots" / "inductive_bias_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROBLEM_ORDER = ("mdgnc", "mdgnc_edge_failures")
PROBLEM_LABELS = {
    "mdgnc": "MDGNC",
    "mdgnc_edge_failures": "MDGNC-EF",
}
MODEL_ORDER = ("Hierarchical", "BLR", "KPLS", "ESP", "MG LD-WLOA")
MODEL_DISPLAY_NAMES = {
    "Hierarchical": "HIER",
    "BLR": "BLR",
    "KPLS": "KPLS",
    "ESP": "SP-RBF",
    "MG LD-WLOA": "MG-LD-WLOA",
}
MODEL_COLORS = {
    "Hierarchical": "#0072B2",
    "BLR": "#666666",
    "KPLS": "#E69F00",
    "ESP": "#CC79A7",
    "MG LD-WLOA": "#009E73",
}
MODEL_MARKERS = {
    "Hierarchical": "o",
    "BLR": "v",
    "KPLS": "^",
    "ESP": "s",
    "MG LD-WLOA": "*",
}
MODEL_LINESTYLES = {
    "Hierarchical": (0, (4, 1.6)),
    "BLR": (0, (6, 2)),
    "KPLS": ":",
    "ESP": "--",
    "MG LD-WLOA": "-",
}
REGIME_ORDER = ("T", "A", "B")
REGIME_TITLES = {
    "T": "(a) Exact architecture",
    "A": "(b) Isomorphic architecture",
    "B": "(c) Novel structure",
}


def set_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.titlecolor": "#202020",
            "axes.edgecolor": "#303030",
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def plot_inductive_bias(summary: pd.DataFrame) -> None:
    budgets = tuple(sorted(summary["budget"].unique()))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15), sharex=True, sharey="row")
    for row_index, problem in enumerate(PROBLEM_ORDER):
        for column_index, regime in enumerate(REGIME_ORDER):
            ax = axes[row_index, column_index]
            subset = summary[(summary["problem"] == problem) & (summary["regime"] == regime)]
            for model in MODEL_ORDER:
                model_data = subset[subset["model"] == model].set_index("budget").reindex(budgets)
                ax.plot(
                    budgets,
                    model_data["rmse"],
                    label=MODEL_DISPLAY_NAMES[model],
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    markersize=5.2 if model == "MG LD-WLOA" else 3.5,
                    linestyle=MODEL_LINESTYLES[model],
                    linewidth=1.25,
                    alpha=0.95,
                )
            if row_index == 0:
                ax.set_title(REGIME_TITLES[regime], fontsize=9.5, pad=5)
            if column_index == 0:
                ax.set_ylabel(PROBLEM_LABELS[problem], fontsize=8.5)
            ax.set_xticks(range(50, 301, 50))
            ax.set_xlim(20, 305)
            ax.tick_params(labelsize=7.5, length=3)
            ax.grid(True, linewidth=0.6)

    fig.supxlabel("Training budget $n$", fontsize=8.5, y=0.055)
    fig.supylabel("Failure-probability RMSE", fontsize=8.5, x=0.025)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), h_pad=0.7, w_pad=0.55)
    for suffix in ("pdf", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"inductive_bias_f0_rmse_by_structure.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    set_plot_style()
    inductive_summary = pd.read_csv(INDUCTIVE_ROOT / "aggregate_results.csv")
    plot_inductive_bias(inductive_summary)
    print(f"Wrote inductive-bias plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
