from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.plotting._common import PLOTS_ROOT, save_figure


def synthetic_ratio(rng, n_eval, final_ratio, progress_power):
    increments = rng.gamma(shape=1.35, scale=1, size=len(n_eval) - 1)
    progress = np.concatenate(([0], np.cumsum(increments) / increments.sum()))
    return final_ratio ** (progress ** progress_power)


def main():
    rng = np.random.default_rng(12)
    runs = [
        {"title": "Run A: rapid early improvement", "final_ratio": 0.30, "power": 0.50, "color": "#0072B2", "fill": "#DCEEF7", "linestyle": "-", "marker": "o", "hatch": "/"},
        {"title": "Run B: slower improvement", "final_ratio": 0.38, "power": 1.25, "color": "#D55E00", "fill": "#F9E4D3", "linestyle": "--", "marker": "s", "hatch": "\\"},
    ]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True, sharey=True, facecolor="white")
    for ax, run in zip(axes, runs):
        n_eval = np.arange(30, 301, 15)
        ratio = synthetic_ratio(rng, n_eval, run["final_ratio"], run["power"])
        regret = np.sum(0.5 * (ratio[:-1] + ratio[1:]) * np.diff(n_eval))
        ax.fill_between(n_eval, ratio, 0, facecolor=run["fill"], edgecolor=run["color"], hatch=run["hatch"], linewidth=0.3)
        ax.plot(n_eval, ratio, color=run["color"], linestyle=run["linestyle"], marker=run["marker"], markersize=3.5, linewidth=1.5)
        ax.scatter([n_eval[-1]], [ratio[-1]], s=34, marker=run["marker"], facecolor="white", edgecolor=run["color"], linewidth=1.3, zorder=3)
        ax.text(0.96, 0.93, f"Final ratio: {ratio[-1]:.2f}\nRegret: {regret:.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none"})
        ax.legend(
            handles=[
                Line2D([0], [0], color=run["color"], linestyle=run["linestyle"], marker=run["marker"], markersize=3.5, linewidth=1.5, label=r"$\Delta$HV ratio"),
                Patch(facecolor=run["fill"], edgecolor=run["color"], hatch=run["hatch"], linewidth=0.3, label="Cumulative regret"),
            ],
            title=run["title"], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
        )
        ax.set_xlim(n_eval[0], n_eval[-1])
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.55)
    axes[0].set_ylabel(r"$\Delta$HV ratio")
    fig.supxlabel("Total function evaluations", fontsize=14)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.17, wspace=0.08)
    save_figure(fig, PLOTS_ROOT / "metric_explanation", "delta_hv_ratio_and_regret")


if __name__ == "__main__":
    main()
