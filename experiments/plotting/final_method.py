from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure
from experimenter.reporting import set_publication_plot_style


KERNEL_RESULTS = RESULTS_ROOT / "kernels"
GNN_RESULTS = RESULTS_ROOT / "gnn"
FINAL_BASELINES = KERNEL_RESULTS / "final_method_baselines"
FINAL_SURROGATES = KERNEL_RESULTS / "final_method_surrogates"
FINAL_ROCKET = KERNEL_RESULTS / "final_method_rocket"
ARCH2VEC_BASELINE = GNN_RESULTS / "final_method_arch2vec_bo"
FINAL_DGBO = GNN_RESULTS / "final_method_dgbo"
FINAL_ROCKET_GNN = GNN_RESULTS / "final_method_rocket_gnn"

MO_SPECS = [
    ("time", "runtime_sec", False),
    ("HV regret", "delta_hv.regret", False),
    ("HV ratio", "delta_hv.ratio", False),
    ("spread", "spread.delta", False),
]
ENGINE_SPECS = [
    ("time", "runtime_sec", False),
    ("regret", "best_gap.regret", False),
    ("gap ratio", "best_gap.ratio", False),
]

PROBLEMS = {
    "engine": {
        "title": "ENGINE",
        "metric": "best_gap.ratio",
        "specs": ENGINE_SPECS,
        "xticks": [75, 100, 150, 200, 250, 300],
        "experiments": {
            "Random": FINAL_BASELINES / "engine" / "random_engine_init_75_pop_10_300_evals",
            "GA": FINAL_BASELINES / "engine" / "ga_engine_init_75_pop_10_300_evals",
            "HIER": FINAL_SURROGATES / "engine" / "bo_no_kpls_engine_init_75_infill_10_300_evals",
            "KPLS": FINAL_SURROGATES / "engine" / "bo_kpls_engine_init_75_infill_10_300_evals",
            "BLR": FINAL_SURROGATES / "engine" / "blr_activeness_engine_init_75_infill_10_300_evals",
            "SP-RBF": FINAL_SURROGATES / "engine" / "adsg_esp_semantic_type_imputed_interaction_engine_init_75_infill_10_300_evals",
            "MG-LD-WLOA": FINAL_SURROGATES / "engine" / "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction_engine_init_75_infill_10_300_evals",
            "Arch2Vec": ARCH2VEC_BASELINE / "engine" / "gnn_dngo_node_d5_b0p0025_z32_e050_engine_init_75_pop_10_300_evals",
            "DGBO": FINAL_DGBO / "engine" / "dgbo_engine_init_75_pop_10_300_evals",
        },
        "plot_methods": ("MG-LD-WLOA", "HIER", "BLR", "GA", "DGBO"),
    },
    "mdgnc": {
        "title": "MDGNC",
        "metric": "delta_hv.ratio",
        "specs": MO_SPECS,
        "xticks": [30, 50, 100, 150, 200],
        "experiments": {
            "Random": FINAL_BASELINES / "mdgnc" / "random_mdgnc_init_30_pop_10_200_evals",
            "GA": FINAL_BASELINES / "mdgnc" / "ga_mdgnc_init_30_pop_10_200_evals",
            "HIER": FINAL_SURROGATES / "mdgnc" / "bo_no_kpls_mdgnc_init_30_infill_10_200_evals",
            "KPLS": FINAL_SURROGATES / "mdgnc" / "bo_kpls_mdgnc_init_30_infill_10_200_evals",
            "BLR": FINAL_SURROGATES / "mdgnc" / "blr_activeness_mdgnc_init_30_infill_10_200_evals",
            "SP-RBF": FINAL_SURROGATES / "mdgnc" / "adsg_esp_semantic_type_imputed_interaction_mdgnc_init_30_infill_10_200_evals",
            "MG-LD-WLOA": FINAL_SURROGATES / "mdgnc" / "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction_mdgnc_init_30_infill_10_200_evals",
            "Arch2Vec": ARCH2VEC_BASELINE / "mdgnc" / "gnn_dngo_node_d5_b0p0025_z32_e050_mdgnc_init_30_pop_10_200_evals",
            "DGBO": FINAL_DGBO / "mdgnc" / "dgbo_mdgnc_init_30_pop_10_200_evals",
        },
        "plot_methods": ("MG-LD-WLOA", "HIER", "BLR", "GA", "DGBO"),
    },
    "mdgnc_edge_failures": {
        "title": "MDGNC-EF",
        "metric": "delta_hv.ratio",
        "specs": MO_SPECS,
        "xticks": [30, 50, 100, 150, 200, 250],
        "experiments": {
            "Random": FINAL_BASELINES / "mdgnc_edge_failures" / "random_mdgnc_edge_failures_init_30_pop_10_250_evals",
            "GA": FINAL_BASELINES / "mdgnc_edge_failures" / "ga_mdgnc_edge_failures_init_30_pop_10_250_evals",
            "HIER": FINAL_SURROGATES / "mdgnc_edge_failures" / "bo_no_kpls_mdgnc_edge_failures_init_30_infill_10_250_evals",
            "KPLS": FINAL_SURROGATES / "mdgnc_edge_failures" / "bo_kpls_mdgnc_edge_failures_init_30_infill_10_250_evals",
            "BLR": FINAL_SURROGATES / "mdgnc_edge_failures" / "blr_activeness_mdgnc_edge_failures_init_30_infill_10_250_evals",
            "SP-RBF": FINAL_SURROGATES / "mdgnc_edge_failures" / "adsg_esp_semantic_type_edge_weight_imputed_interaction_mdgnc_edge_failures_init_30_infill_10_250_evals",
            "MG-LD-WLOA": FINAL_SURROGATES / "mdgnc_edge_failures" / "adsg_mg_ld_wloa_adsg_semantic_type_d8_edge_weight_imputed_interaction_mdgnc_edge_failures_init_30_infill_10_250_evals",
            "Arch2Vec": ARCH2VEC_BASELINE / "mdgnc_edge_failures" / "gnn_dngo_node_d5_b0p0025_z32_e050_mdgnc_edge_failures_init_30_pop_10_250_evals",
            "DGBO": FINAL_DGBO / "mdgnc_edge_failures" / "dgbo_mdgnc_edge_failures_init_30_pop_10_250_evals",
        },
        "plot_methods": ("MG-LD-WLOA", "HIER", "KPLS", "GA", "DGBO"),
    },
    "rocket": {
        "title": "ROCKET",
        "metric": "delta_hv.ratio",
        "specs": MO_SPECS,
        "xticks": [100, 200, 300, 400, 500, 600, 700],
        "experiments": {
            "Random": FINAL_ROCKET / "rocket" / "random_hierarchical_rocket_init_100_pop_10_750_evals",
            "GA": FINAL_ROCKET / "rocket" / "ga_rocket_init_100_pop_10_750_evals",
            "HIER": FINAL_ROCKET / "rocket" / "bo_no_kpls_mean_lcb_rocket_init_100_infill_10_750_evals",
            "KPLS": FINAL_ROCKET / "rocket" / "bo_kpls_mean_lcb_rocket_init_100_infill_10_750_evals",
            "BLR": FINAL_ROCKET / "rocket" / "blr_activeness_mean_lcb_rocket_init_100_infill_10_750_evals",
            "SP-RBF": FINAL_ROCKET / "rocket" / "adsg_esp_semantic_type_imputed_interaction_mean_lcb_rocket_init_100_infill_10_750_evals",
            "MG-LD-WLOA": FINAL_ROCKET / "rocket" / "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction_mean_lcb_rocket_init_100_infill_10_750_evals",
            "Arch2Vec": FINAL_ROCKET_GNN / "rocket" / "gnn_dngo_node_d5_b0p0025_z32_e050_mean_lcb_rocket_init_100_pop_10_750_evals",
            "DGBO": FINAL_ROCKET_GNN / "rocket" / "dgbo_mean_lcb_rocket_init_100_pop_10_750_evals",
        },
        "plot_methods": ("MG-LD-WLOA", "HIER", "KPLS", "GA", "DGBO"),
    },
}

STYLES = {
    "MG-LD-WLOA": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
    "HIER": dict(color="#009E73", linestyle=(0, (6, 2)), marker="s", linewidth=1.8),
    "BLR": dict(color="#CC79A7", linestyle=(0, (6, 2, 1.5, 2)), marker="D", linewidth=1.8),
    "KPLS": dict(color="#CC79A7", linestyle=(0, (6, 2, 1.5, 2)), marker="D", linewidth=1.8),
    "GA": dict(color="#D55E00", linestyle=(0, (9, 3)), marker="^", linewidth=1.8),
    "DGBO": dict(color="#222222", linestyle=(0, (5, 1)), marker="X", linewidth=1.7),
}
GLOBAL_LEGEND_METHODS = ("MG-LD-WLOA", "HIER", "GA", "DGBO")


def ratio_curve(path, metric):
    data = pd.read_csv(path / "aggregate_results.csv", usecols=["n_eval", metric, "run"])
    if data["run"].nunique() < 2:
        raise ValueError(f"Expected multiple runs in {path}")
    return data.groupby("n_eval", sort=True)[metric].mean()


def main():
    set_publication_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.75), sharey=True, facecolor="white")
    for panel, (ax, config) in enumerate(zip(axes.flat, PROBLEMS.values())):
        for method in config["plot_methods"]:
            mean = ratio_curve(config["experiments"][method], config["metric"])
            style = STYLES[method]
            markevery = max(1, len(mean) // 8)
            ax.plot(
                mean.index, mean.values, label=method, drawstyle="steps-post",
                markevery=markevery, markersize=5.2, markerfacecolor="white",
                markeredgewidth=1.1, zorder=4 if method == "MG-LD-WLOA" else 3,
                **style,
            )
        ax.axhline(0, color="#333333", linewidth=0.7, alpha=0.55, zorder=2)
        ax.set_title(
            f"({chr(97 + panel)}) {config['title']}",
            fontfamily="sans-serif", fontweight="normal", fontsize=11, pad=8,
        )
        selected_encoding = config["plot_methods"][2]
        encoding_style = STYLES[selected_encoding]
        ax.legend(
            handles=[Line2D(
                [0], [0], label=selected_encoding, markersize=5.2,
                markerfacecolor="white", markeredgewidth=1.1,
                **encoding_style,
            )],
            loc="upper right", frameon=False, fontsize=7.3,
            handlelength=1.8, handletextpad=0.4, borderaxespad=0.35,
        )
        ax.set_xticks(config["xticks"])
        ax.set_xlim(left=config["xticks"][0])
        ax.set_ylim(-0.10, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        ax.tick_params(direction="out", length=3, width=0.7)
    fig.supxlabel("Function evaluations", y=0.025)
    fig.supylabel(r"$\Delta$HV ratio / gap ratio (lower is better)", x=0.015, fontsize=11.5)
    handles = [
        Line2D(
            [0], [0], label=method, markersize=5.2,
            markerfacecolor="white", **STYLES[method],
        )
        for method in GLOBAL_LEGEND_METHODS
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, frameon=False, fontsize=7.4, markerscale=1.15,
        columnspacing=1.25, handlelength=2.0, handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.88, wspace=0.16, hspace=0.36)
    save_figure(fig, PLOTS_ROOT / "final_method", "final_method_gap_ratio_curves")


if __name__ == "__main__":
    main()
