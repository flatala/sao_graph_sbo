from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure
from experimenter.reporting import REPORT_COLORS, set_publication_plot_style


ROOT = RESULTS_ROOT / "kernels" / "interaction_composition_ablation"
SETTINGS = {"mdgnc": (30, 200), "engine": (75, 300)}
SPECS = {
    "mdgnc": [("time", "runtime_sec", False), ("regret", "delta_hv.regret", False), ("HV gap ratio", "delta_hv.ratio", False), ("spread", "spread.delta", False)],
    "engine": [("time", "runtime_sec", False), ("regret", "best_gap.regret", False), ("gap ratio", "best_gap.ratio", False)],
}
RESPONSES = {
    "mdgnc": [("f0_failure_rate", "Failure\nprobability"), ("f1_mass", "Mass")],
    "engine": [("f0_tsfc", "TSFC"), ("g0_jet_mach", "Jet Mach"), ("g1_pr", r"$PR_1$"), ("g2_pr", r"$PR_2$"), ("g3_pr", r"$PR_3$"), ("g4_pr_perc_sum", "PR fraction\nsum")],
}
COMPONENTS = [
    ("graph_0.mg_ld_wloa_adsg_semantic_type", "Graph"),
    ("sizing", "Sizing"),
    ("interaction.graph_0_mg_ld_wloa_adsg_semantic_type_sizing", r"Graph $\times$ sizing"),
]


def result_dir(problem, method):
    n_init, max_evals = SETTINGS[problem]
    return ROOT / problem / f"{method}_{problem}_init_{n_init}_infill_10_{max_evals}_evals"


def experiments(problem):
    return {
        "MG-LD-WLOA — additive": result_dir(problem, "adsg_mg_ld_wloa_adsg_semantic_type_imputed_sizing_additive_d8"),
        "MG-LD-WLOA — multiplicative": result_dir(problem, "adsg_mg_ld_wloa_adsg_semantic_type_imputed_sizing_multiplicative_d8"),
        "MG-LD-WLOA — additive + product": result_dir(problem, "adsg_mg_ld_wloa_adsg_semantic_type_imputed_sizing_interaction_d8"),
        "SP-RBF — additive": result_dir(problem, "adsg_esp_semantic_type_imputed_sizing_additive"),
        "SP-RBF — multiplicative": result_dir(problem, "adsg_esp_semantic_type_imputed_sizing_multiplicative"),
        "SP-RBF — additive + product": result_dir(problem, "adsg_esp_semantic_type_imputed_sizing_interaction"),
    }


EXPERIMENTS = {problem: experiments(problem) for problem in SETTINGS}


def average_weights(problem):
    data = pd.read_csv(EXPERIMENTS[problem]["MG-LD-WLOA — additive + product"] / "aggregate_results.csv")
    columns = {
        response: [f"kernel_weight.{response}.{component}" for component, _ in COMPONENTS]
        for response, _ in RESPONSES[problem]
    }
    all_columns = [column for response_columns in columns.values() for column in response_columns]
    updates = data.dropna(subset=all_columns)
    averaged = updates.groupby("run", sort=False)[all_columns].mean().reset_index()
    if averaged["run"].nunique() != 20:
        raise ValueError(f"Expected 20 runs for {problem}")
    return averaged, columns


def main():
    weights = {problem: average_weights(problem) for problem in SETTINGS}
    set_publication_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), sharey=True, gridspec_kw={"width_ratios": (1, 2.25)}, facecolor="white")
    offsets = (-0.24, 0, 0.24)
    colors = REPORT_COLORS[:3]
    for ax, problem, title in zip(axes, ("mdgnc", "engine"), ("(a) MDGNC", "(b) ENGINE")):
        averaged, columns = weights[problem]
        centres = np.arange(len(RESPONSES[problem]), dtype=float)
        for component_index, (offset, color) in enumerate(zip(offsets, colors)):
            distributions = [averaged[columns[response][component_index]].to_numpy() for response, _ in RESPONSES[problem]]
            positions = centres + offset
            ax.boxplot(
                distributions, positions=positions, widths=0.19, patch_artist=True, showfliers=False, manage_ticks=False,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.20},
                whiskerprops={"color": color}, capprops={"color": color}, medianprops={"color": "#222222", "linewidth": 1.25},
            )
            for position, distribution in zip(positions, distributions):
                ax.scatter(position + np.linspace(-0.035, 0.035, len(distribution)), distribution, s=7, color=color, alpha=0.40, linewidths=0)
        ax.set_xticks(centres, [label for _, label in RESPONSES[problem]])
        ax.set_xlim(-0.55, len(centres) - 0.45)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontweight="normal")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[0].set_ylabel("Mean composition weight")
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor=color, alpha=0.28, label=label) for (_, label), color in zip(COMPONENTS, colors)],
        loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=0.9)
    save_figure(
        fig,
        PLOTS_ROOT / "interaction_composition_ablation",
        "mg_ld_wloa_interaction_weights_optimization_average",
    )


if __name__ == "__main__":
    main()
