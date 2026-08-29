from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure
from experimenter.reporting import REPORT_COLORS, set_publication_plot_style


FIXED_ROOT = RESULTS_ROOT / "kernels" / "wloa_depth_ablation"
LEARNED_ROOT = RESULTS_ROOT / "kernels" / "encoding_granularity_ablation"
SETTINGS = {
    "engine": {"n_init": 75, "max_evals": 300, "regret": "best_gap.regret", "ratio": "best_gap.ratio"},
    "mdgnc": {"n_init": 30, "max_evals": 200, "regret": "delta_hv.regret", "ratio": "delta_hv.ratio"},
}


def result_dir(root, problem, method):
    config = SETTINGS[problem]
    return root / problem / f"{method}_{problem}_init_{config['n_init']}_infill_10_{config['max_evals']}_evals"


def experiments(problem):
    items = {
        f"Fixed d={depth}": result_dir(FIXED_ROOT, problem, f"adsg_wloa_semantic_type_d{depth}_imputed_interaction")
        for depth in range(9)
    }
    items["LD-WLOA"] = result_dir(LEARNED_ROOT, problem, "adsg_ld_wloa_semantic_type_d8_imputed_interaction")
    return items


EXPERIMENTS = {problem: experiments(problem) for problem in SETTINGS}


def learned_weights(path):
    data = pd.read_csv(path / "aggregate_results.csv")
    objectives = ("f0_failure_rate", "f1_mass")
    depths = np.arange(9)
    normalized = {}
    for objective in objectives:
        columns = [f"kernel_diagnostics.{objective}.graph_0.ld_wloa_semantic_type.w_{depth}" for depth in depths]
        values = data[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        normalized[objective] = pd.DataFrame(values / values.sum(axis=1, keepdims=True), index=data.index, columns=depths)
    return data, normalized, objectives, depths


def main():
    data, normalized, objectives, depths = learned_weights(EXPERIMENTS["mdgnc"]["LD-WLOA"])
    centre = normalized[objectives[0]].to_numpy() @ depths
    grouped = pd.DataFrame({"n_eval": data["n_eval"], "centre": centre}).groupby("n_eval")["centre"]
    mean, lower, upper = grouped.mean(), grouped.quantile(0.125), grouped.quantile(0.875)

    set_publication_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.490684, 2.774474), gridspec_kw={"width_ratios": (1.15, 1, 1)})
    x = mean.index.to_numpy(dtype=float)
    axes[0].fill_between(x, lower.to_numpy(), upper.to_numpy(), color=REPORT_COLORS[0], alpha=0.18, linewidth=0)
    axes[0].plot(x, mean.to_numpy(), color=REPORT_COLORS[0], linewidth=1.8)
    axes[0].scatter(x, mean.to_numpy(), color=REPORT_COLORS[0], s=9)
    axes[0].set_xlabel("Evaluations")
    axes[0].set_ylabel(r"WL depth centre of mass $\bar{h}_w$")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("(a) Failure probability", fontsize=9, pad=5.3061)
    axes[0].grid(axis="y")

    for ax, objective, color, title in zip(axes[1:], objectives, REPORT_COLORS[:2], ("(b) Failure probability", "(c) Mass")):
        run_average = normalized[objective].copy()
        run_average["run"] = data["run"].to_numpy()
        run_average = run_average.groupby("run", sort=False).mean()
        upper_whiskers = []
        for depth in depths:
            values = run_average[depth].dropna().to_numpy()
            q1, q3 = np.quantile(values, (0.25, 0.75))
            upper_fence = q3 + 1.5 * (q3 - q1)
            upper_whiskers.append(values[values <= upper_fence].max())
        run_average = run_average / max(upper_whiskers)
        ax.boxplot(
            [run_average[depth].dropna().to_numpy() for depth in depths], positions=depths, widths=0.58,
            patch_artist=True, showfliers=False,
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.22},
            whiskerprops={"color": color}, capprops={"color": color}, medianprops={"color": "#222222", "linewidth": 1.25},
        )
        ticks = np.arange(0, 9, 2)
        ax.set_xticks(ticks, ticks)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r"WL refinement depth $h$")
        ax.set_title(title, fontsize=9, pad=5.3061)
        ax.grid(axis="y")
    axes[1].set_ylabel("Relative learned weight")
    axes[2].tick_params(labelleft=False)
    fig.tight_layout(w_pad=0.5588)
    save_figure(fig, PLOTS_ROOT / "wloa_depth_ablation", "ld_wloa_learned_depth_mdgnc")


if __name__ == "__main__":
    main()
