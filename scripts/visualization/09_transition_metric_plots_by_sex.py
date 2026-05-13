#!/usr/bin/env python3
"""Plot transition-level metrics by sex for standard or damage/repair analyses."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ("LR", "DNN")
SEX_ORDER = ("female", "male")
PANEL_ORDER = ("disease", "adl", "all")
PANEL_LABELS = {"disease": "DISEASE", "adl": "ADL", "all": "ALL"}
MODEL_STYLE = {
    "LR": {"color": "#1f77b4", "marker": "o"},
    "DNN": {"color": "#ff7f0e", "marker": "s"},
}
METRIC_SPECS = {
    "auc": {"label": "Mean AUC", "better": "higher", "filename": "mean_auc_by_transition.png"},
    "ece": {"label": "Mean ECE", "better": "lower", "filename": "mean_ece_by_transition.png"},
    "r2cal": {"label": "Mean Calibration R2", "better": "higher", "filename": "mean_r2cal_by_transition.png"},
    "brier": {"label": "Mean Brier", "better": "lower", "filename": "mean_brier_by_transition.png"},
}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def transition_sort_key(name: str) -> tuple[int, int]:
    m = re.match(r"^S(\d+)_S(\d+)$", str(name).strip())
    if not m:
        return (999, 999)
    return (int(m.group(1)), int(m.group(2)))


def read_metrics(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["model"] = model_name
    if "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.lower()
        df = df[status.eq("ok")].copy()
    df["transition"] = df["transition"].astype(str).str.strip()
    df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    return df


def summarize_panel(df: pd.DataFrame, metric: str, panel: str) -> pd.DataFrame:
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric, "transition", "model", "sex"])
    if panel != "all":
        work = work[work["target_type"].eq(panel)].copy()
    if work.empty:
        return work
    summary = (
        work.groupby(["model", "sex", "transition"], as_index=False)
        .agg(
            mean_value=(metric, "mean"),
            target_count=(metric, "count"),
            std_value=(metric, lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
        )
    )
    summary["se_value"] = summary["std_value"] / np.sqrt(summary["target_count"].clip(lower=1))
    return summary


def compute_limits(summary_by_panel: dict[str, pd.DataFrame], better: str) -> tuple[float, float]:
    vals = []
    for panel in summary_by_panel.values():
        if panel.empty:
            continue
        vals.extend(np.maximum(0.0, panel["mean_value"].to_numpy(dtype=float) - panel["se_value"].to_numpy(dtype=float)))
        vals.extend(panel["mean_value"].to_numpy(dtype=float) + panel["se_value"].to_numpy(dtype=float))
    if not vals:
        return (0.0, 1.0)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if better == "lower":
        return (0.0, min(1.0, hi * 1.10 if hi > 0 else 1.0))
    span = hi - lo
    pad = 0.02 if span <= 0 else 0.08 * span
    y_min = max(0.0, lo - pad)
    y_max = min(1.02, hi + pad)
    if y_max <= y_min:
        y_max = min(1.02, y_min + 0.05)
    return (y_min, y_max)


def approx_target_count(panel: pd.DataFrame, sex: str) -> str:
    block = panel[panel["sex"] == sex]
    if block.empty or "target_count" not in block.columns:
        return ""
    approx = int(round(block["target_count"].mean()))
    return f"avg over ~{approx} targets"


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, dataset_label: str, run_tag: str, title_suffix: str) -> None:
    spec = METRIC_SPECS[metric]
    transitions = sorted(df["transition"].dropna().unique().tolist(), key=transition_sort_key)
    summaries = {panel: summarize_panel(df, metric, panel) for panel in PANEL_ORDER}
    y_min, y_max = compute_limits(summaries, spec["better"])

    for sex in SEX_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=False)
        fig.suptitle(f"{dataset_label} | {run_tag} | {sex.title()} | {title_suffix} | {spec['label']} (LR vs DNN)", fontsize=17, y=0.98)

        for idx, panel_name in enumerate(PANEL_ORDER):
            ax = axes[idx]
            panel = summaries[panel_name]
            ax.set_title(PANEL_LABELS[panel_name], fontsize=16)
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)

            block = panel[panel["sex"] == sex].copy()
            if block.empty:
                ax.text(0.5, 0.5, "No evaluable targets", ha="center", va="center", transform=ax.transAxes, fontsize=13)
                ax.set_xticks([])
                continue

            x = np.arange(len(transitions), dtype=float)
            for model in MODEL_ORDER:
                style = MODEL_STYLE[model]
                sub = block[block["model"] == model].set_index("transition").reindex(transitions)
                mask = sub["mean_value"].notna().to_numpy()
                if not mask.any():
                    continue
                sub_valid = sub.iloc[np.flatnonzero(mask)]
                ax.errorbar(
                    x[mask],
                    sub_valid["mean_value"].to_numpy(dtype=float),
                    yerr=sub_valid["se_value"].to_numpy(dtype=float),
                    fmt=style["marker"] + "-",
                    color=style["color"],
                    ecolor=style["color"],
                    linewidth=2,
                    elinewidth=1.5,
                    capsize=3,
                    markersize=6.5,
                    label=model,
                )
            ax.text(0.03, 0.92, approx_target_count(panel, sex), transform=ax.transAxes, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(transitions, rotation=35, ha="right")
            ax.set_ylim(y_min, y_max)
            if spec["better"] == "lower":
                ax.invert_yaxis()

        axes[0].set_ylabel(f"{spec['label']} (+/- SE across targets)", fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=12)
        fig.tight_layout(rect=[0, 0.0, 1, 0.94])
        sex_out = out_path / sex / spec["filename"]
        sex_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sex_out, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dataset-label", default=None)
    ap.add_argument("--metrics-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--event-type", default=None, help="Optional event type for damage/repair metrics")
    ap.add_argument("--title-suffix", default="Standard")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    dataset_label = args.dataset_label or run_tag
    metrics_root = Path(args.metrics_root)
    out_root = Path(args.out_root) / run_tag

    lr = read_metrics(metrics_root / "lr" / run_tag / "metrics_by_target.csv", "LR")
    dnn = read_metrics(metrics_root / "dnn" / run_tag / "metrics_by_target.csv", "DNN")
    df = pd.concat([lr, dnn], ignore_index=True)
    if args.event_type:
        df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
        df = df[df["event_type"].eq(args.event_type)].copy()
        out_root = out_root / args.event_type

    for metric in METRIC_SPECS:
        plot_metric(df=df, metric=metric, out_path=out_root, dataset_label=dataset_label, run_tag=run_tag, title_suffix=args.title_suffix)


if __name__ == "__main__":
    main()
