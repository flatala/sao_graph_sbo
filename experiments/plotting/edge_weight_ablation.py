from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure
from experimenter.reporting import set_publication_plot_style


ROOT = RESULTS_ROOT / "kernels" / "edge_weight_ablation"
SPECS = [
    ("time", "runtime_sec", False),
    ("regret", "delta_hv.regret", False),
    ("HV gap ratio", "delta_hv.ratio", False),
    ("spread", "spread.delta", False),
]
EXPERIMENTS = {
    "mdgnc": {
        "HIER": ROOT / "mdgnc" / "bo_no_kpls_mdgnc_init_30_infill_10_200_evals",
        "LD-WLOA": ROOT / "mdgnc" / "adsg_ld_wloa_cutoff8_mdgnc_init_30_infill_10_200_evals",
    },
    "mdgnc_edge_failures": {
        "HIER": ROOT / "mdgnc_edge_failures" / "bo_no_kpls_mdgnc_edge_failures_init_30_infill_10_250_evals",
        "LD-WLOA": ROOT / "mdgnc_edge_failures" / "adsg_ld_wloa_cutoff8_mdgnc_edge_failures_init_30_infill_10_250_evals",
        "LD-WLOA + EW": ROOT / "mdgnc_edge_failures" / "adsg_ld_wloa_edge_weight_cutoff8_mdgnc_edge_failures_init_30_infill_10_250_evals",
    },
}
STYLES = {
    "LD-WLOA": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
    "HIER": dict(color="#009E73", linestyle=(0, (6, 2)), marker="s", linewidth=1.8),
    "LD-WLOA + EW": dict(color="#CC79A7", linestyle=(0, (6, 2, 1.5, 2)), marker="v", linewidth=2.0),
}


def ratio_curve(path):
    data = pd.read_csv(path / "aggregate_results.csv", usecols=["n_eval", "delta_hv.ratio", "run"])
    if data["run"].nunique() != 20:
        raise ValueError(f"Expected 20 runs in {path}")
    grouped = data.groupby("n_eval", sort=True)["delta_hv.ratio"]
    return grouped.mean(), grouped.quantile(0.125), grouped.quantile(0.875)


def main():
    set_publication_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), sharey=True, facecolor="white")
    panels = (
        ("(a) MDGNC", EXPERIMENTS["mdgnc"], [30, 50, 100, 150, 200]),
        ("(b) MDGNC-EF", EXPERIMENTS["mdgnc_edge_failures"], [30, 50, 100, 150, 200, 250]),
    )
    for ax, (title, experiments, ticks) in zip(axes, panels):
        for label, path in experiments.items():
            mean, lower, upper = ratio_curve(path)
            style = STYLES[label]
            ax.plot(
                mean.index, mean.values, drawstyle="steps-post", markevery=max(1, len(mean) // 8),
                markersize=4, markerfacecolor="white", markeredgewidth=0.9,
                zorder=4 if label.startswith("LD-WLOA") else 3, **style,
            )
            ax.fill_between(mean.index, lower.values, upper.values, step="post", color=style["color"], alpha=0.08, linewidth=0)
        ax.axhline(0, color="#333333", linewidth=0.7, alpha=0.55)
        ax.set_title(title, fontweight="normal", pad=8)
        ax.set_xlim(left=30)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.supxlabel("Function evaluations", y=0.035)
    fig.supylabel(r"$\Delta$HV ratio (lower is better)", x=0.02)
    handles = [Line2D([0], [0], label=name, markersize=4.5, markerfacecolor="white", **style) for name, style in STYLES.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.80, wspace=0.16)
    save_figure(fig, PLOTS_ROOT / "edge_weight_ablation", "mdgnc_edge_weight_convergence")


if __name__ == "__main__":
    main()
