#!/usr/bin/env python3
"""Paper-ready figures for the ELSA stable feature-selection analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_style import apply_paper_style

apply_paper_style()


FAMILY_ORDER = ["disease", "adl"]
FAMILY_LABEL = {"disease": "Disease", "adl": "ADL"}
METRIC_SPECS = {
    "auc": {"label": "Mean AUC", "better": "higher"},
    "brier": {"label": "Mean Brier", "better": "lower"},
    "ece": {"label": "Mean ECE", "better": "lower"},
    "r2cal": {"label": "Mean Calibration $R^2$", "better": "higher"},
}
MODEL_STYLE = {
    ("LR", "full"): {"color": "#1f77b4", "marker": "o", "label": "LR full"},
    ("LR", "reduced"): {"color": "#0b559f", "marker": "s", "label": "LR reduced"},
    ("DNN", "full"): {"color": "#ff7f0e", "marker": "^", "label": "DNN full"},
    ("DNN", "reduced"): {"color": "#c45508", "marker": "D", "label": "DNN reduced"},
}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def plot_selection_curves(curve: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("ELSA | Stable mRMR feature selection", fontsize=18, y=1.02)
    metrics = ["auc", "brier", "ece", "r2cal"]
    for row_idx, family in enumerate(FAMILY_ORDER):
        fam_curve = curve[curve["family"] == family].sort_values("k")
        fam_summary = summary[summary["family"] == family].iloc[0]
        for col_idx, metric in enumerate(metrics[:2]):
            ax = axes[row_idx, col_idx]
            spec = METRIC_SPECS[metric]
            ax.errorbar(
                fam_curve["k"],
                fam_curve[f"mean_{metric}"],
                yerr=fam_curve[f"se_{metric}"],
                color="#1f4e79",
                marker="o",
                capsize=3,
                linewidth=2,
            )
            ax.axvline(float(fam_summary["k_one_se"]), color="#2ca02c", linestyle="--", linewidth=1.5, label="1-SE choice")
            ax.axvline(float(fam_summary["k_best_brier"]), color="#7f7f7f", linestyle=":", linewidth=1.3, label="Best Brier")
            ax.set_title(f"{FAMILY_LABEL[family]} | {spec['label']}")
            ax.set_xlabel("Number of selected features")
            ax.set_ylabel(spec["label"])
            ax.grid(alpha=0.25)
            if spec["better"] == "lower":
                ax.invert_yaxis()
        for col_idx, metric in enumerate(metrics[2:], start=0):
            ax = axes[row_idx, col_idx + 0]  # placeholder for mypy clarity
    # overwrite the right column with ECE and R2 plots
    for row_idx, family in enumerate(FAMILY_ORDER):
        fam_curve = curve[curve["family"] == family].sort_values("k")
        fam_summary = summary[summary["family"] == family].iloc[0]
        for ax, metric in zip(axes[row_idx, :], ["auc", "brier"]):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=9)

    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig2.suptitle("ELSA | Stable mRMR feature selection (calibration metrics)", fontsize=18, y=1.02)
    for row_idx, family in enumerate(FAMILY_ORDER):
        fam_curve = curve[curve["family"] == family].sort_values("k")
        fam_summary = summary[summary["family"] == family].iloc[0]
        for col_idx, metric in enumerate(["ece", "r2cal"]):
            ax = axes2[row_idx, col_idx]
            spec = METRIC_SPECS[metric]
            ax.errorbar(
                fam_curve["k"],
                fam_curve[f"mean_{metric}"],
                yerr=fam_curve[f"se_{metric}"],
                color="#6b5b95",
                marker="o",
                capsize=3,
                linewidth=2,
            )
            ax.axvline(float(fam_summary["k_one_se"]), color="#2ca02c", linestyle="--", linewidth=1.5, label="1-SE choice")
            ax.axvline(float(fam_summary["k_best_brier"]), color="#7f7f7f", linestyle=":", linewidth=1.3, label="Best Brier")
            ax.set_title(f"{FAMILY_LABEL[family]} | {spec['label']}")
            ax.set_xlabel("Number of selected features")
            ax.set_ylabel(spec["label"])
            ax.grid(alpha=0.25)
            if spec["better"] == "lower":
                ax.invert_yaxis()
            ax.legend(loc="best", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.parent / "selection_curves_auc_brier.png", dpi=220, bbox_inches="tight")
    fig2.savefig(out_path.parent / "selection_curves_ece_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig2)


def plot_feature_stability(rankings: dict[str, pd.DataFrame], out_path: Path, top_n: int = 20) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("ELSA | Stable selected features", fontsize=18, y=1.02)
    for ax, family in zip(axes, FAMILY_ORDER):
        df = rankings[family].sort_values(["selection_frequency_top_k", "rank"], ascending=[False, True]).head(top_n)
        df = df.sort_values("selection_frequency_top_k", ascending=True)
        colors = ["#2ca02c" if sel else "#9ecae1" for sel in df["selected_final"].astype(int)]
        ax.barh(df["feature"], df["selection_frequency_top_k"], color=colors)
        ax.set_title(FAMILY_LABEL[family])
        ax.set_xlabel("Selection frequency in top-K across leave-one-transition-out rankings")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reduced_vs_full(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("ELSA | Reduced vs full feature sets", fontsize=18, y=1.02)
    metric_order = ["auc", "brier", "ece", "r2cal"]
    x = np.arange(len(FAMILY_ORDER), dtype=float)
    offsets = {
        ("LR", "full"): -0.24,
        ("LR", "reduced"): -0.08,
        ("DNN", "full"): 0.08,
        ("DNN", "reduced"): 0.24,
    }

    for ax, metric in zip(axes.flatten(), metric_order):
        spec = METRIC_SPECS[metric]
        for key, style in MODEL_STYLE.items():
            model, feature_set = key
            values = []
            for family in FAMILY_ORDER:
                sub = summary[
                    summary["family"].eq(family)
                    & summary["model"].eq(model)
                    & summary["feature_set"].eq(feature_set)
                ]
                values.append(float(sub[f"mean_{metric}"].iloc[0]) if not sub.empty else np.nan)
            ax.plot(x + offsets[key], values, marker=style["marker"], color=style["color"], linewidth=1.8, label=style["label"])
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER])
        ax.set_title(spec["label"])
        ax.set_ylabel(spec["label"])
        ax.grid(alpha=0.25)
        if spec["better"] == "lower":
            ax.invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--metrics-root", default="analysis/ELSA/feature_selection/stable_mrmr")
    ap.add_argument("--out-root", default="analysis/ELSA/figures/feature_selection/stable_mrmr")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    metrics_root = Path(args.metrics_root) / run_tag
    out_root = Path(args.out_root) / run_tag

    curve = pd.read_csv(metrics_root / "selection_curve_summary.csv")
    summary = pd.read_csv(metrics_root / "selection_summary.csv")
    rankings = {
        family: pd.read_csv(metrics_root / f"ranking_full_{family}.csv")
        for family in FAMILY_ORDER
    }
    reduced_vs_full = pd.read_csv(metrics_root / "reduced_vs_full_summary.csv")

    plot_selection_curves(curve, summary, out_root / "selection_curves_auc_brier.png")
    plot_feature_stability(rankings, out_root / "feature_stability.png")
    plot_reduced_vs_full(reduced_vs_full, out_root / "reduced_vs_full_comparison.png")


if __name__ == "__main__":
    main()
