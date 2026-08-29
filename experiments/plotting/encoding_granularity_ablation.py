from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure
from experimenter.reporting import REPORT_COLORS, set_publication_plot_style


ENCODING_ROOT = RESULTS_ROOT / "kernels" / "encoding_granularity_ablation"
COMPOSITION_ROOT = RESULTS_ROOT / "kernels" / "interaction_composition_ablation"
SETTINGS = {"engine": (75, 300), "mdgnc": (30, 200)}
SPECS = {
    "engine": [("time", "runtime_sec", False), ("regret", "best_gap.regret", False), ("gap ratio", "best_gap.ratio", False)],
    "mdgnc": [("time", "runtime_sec", False), ("regret", "delta_hv.regret", False), ("HV gap ratio", "delta_hv.ratio", False), ("spread", "spread.delta", False)],
}


def result_dir(root, problem, method):
    n_init, max_evals = SETTINGS[problem]
    return root / problem / f"{method}_{problem}_init_{n_init}_infill_10_{max_evals}_evals"


def experiments(problem):
    return {
        "DSG": result_dir(ENCODING_ROOT, problem, "adsg_ld_wloa_dsg_d8_imputed_interaction"),
        "ADSG": result_dir(ENCODING_ROOT, problem, "adsg_ld_wloa_adsg_d8_imputed_interaction"),
        "Semantic type": result_dir(ENCODING_ROOT, problem, "adsg_ld_wloa_semantic_type_d8_imputed_interaction"),
        "ADSG + semantic type": result_dir(COMPOSITION_ROOT, problem, "adsg_mg_ld_wloa_adsg_semantic_type_imputed_sizing_interaction_d8"),
    }


EXPERIMENTS = {problem: experiments(problem) for problem in SETTINGS}
RESPONSES = {
    "engine": [("f0_tsfc", "TSFC"), ("g0_jet_mach", "Jet Mach"), ("g1_pr", r"$PR_1$"), ("g2_pr", r"$PR_2$"), ("g3_pr", r"$PR_3$"), ("g4_pr_perc_sum", "PR fraction\nsum")],
    "mdgnc": [("f0_failure_rate", "Failure\nrate"), ("f1_mass", "Mass")],
}
KERNEL_NAME = "graph_0.mg_ld_wloa_adsg_semantic_type"


def learned_weights(problem):
    data = pd.read_csv(EXPERIMENTS[problem]["ADSG + semantic type"] / "aggregate_results.csv")
    distributions = []
    for response, _ in RESPONSES[problem]:
        prefix = f"kernel_diagnostics.{response}.{KERNEL_NAME}"
        columns = [f"{prefix}.adsg_weight", f"{prefix}.semantic_type_weight"]
        values = data[["run", "n_eval", *columns]].dropna()
        if not np.allclose(values[columns].sum(axis=1), 1):
            raise ValueError(f"Granularity weights do not sum to one for {problem}/{response}")
        run_means = values.groupby("run", sort=False)[columns].mean()
        if len(run_means) != 20:
            raise ValueError(f"Expected 20 runs for {problem}/{response}")
        distributions.append(run_means[columns[0]].to_numpy())
    return distributions


def main():
    values = {problem: learned_weights(problem) for problem in SETTINGS}
    set_publication_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3), sharey=True, gridspec_kw={"width_ratios": (3, 1.15)}, facecolor="white")
    styles = [("ADSG", "#0072B2", lambda x: x), ("Semantic", REPORT_COLORS[1], lambda x: 1 - x)]
    for ax, problem, title in zip(axes, ("engine", "mdgnc"), ("(a) ENGINE", "(b) MDGNC")):
        centres = np.arange(len(values[problem]))
        ax.axhline(0.5, color="#666666", linestyle=(0, (4, 2)), linewidth=0.8)
        for (_, color, transform), offset in zip(styles, (-0.16, 0.16)):
            distributions = [transform(item) for item in values[problem]]
            positions = centres + offset
            ax.boxplot(
                distributions, positions=positions, widths=0.27, patch_artist=True, showfliers=False, manage_ticks=False,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.22},
                whiskerprops={"color": color}, capprops={"color": color}, medianprops={"color": "#222222", "linewidth": 1.3},
            )
            for position, distribution in zip(positions, distributions):
                ax.scatter(position + np.linspace(-0.055, 0.055, len(distribution)), distribution, s=11, color=color, alpha=0.48, linewidths=0)
        ax.set_xticks(centres, [label for _, label in RESPONSES[problem]])
        ax.set_xlim(-0.55, len(centres) - 0.45)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontweight="normal")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axes[0].set_ylabel("Learned granularity weight")
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor=color, alpha=0.35, label=label) for label, color, _ in styles],
        loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1)
    save_figure(fig, PLOTS_ROOT / "encoding_granularity_ablation", "learned_granularity_weights")


if __name__ == "__main__":
    main()
