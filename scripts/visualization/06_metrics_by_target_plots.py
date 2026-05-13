#!/usr/bin/env python3
"""Plot target-level mean +/- SE metrics across transitions for LR vs DNN.

This script summarizes evaluated metrics by target and target type
(`disease` / `adl`) and writes paired LR-vs-DNN figures for:
- standard next-wave prediction metrics
- damage event metrics
- repair event metrics

Expected inputs:
- <metrics-root>/lr/<RUN_TAG>/metrics_by_target.csv
- <metrics-root>/dnn/<RUN_TAG>/metrics_by_target.csv
- <metrics-root>/damage_repair/lr/<RUN_TAG>/metrics_by_target.csv
- <metrics-root>/damage_repair/dnn/<RUN_TAG>/metrics_by_target.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from paper_style import apply_paper_style

apply_paper_style()


MODEL_ORDER = ("LR", "DNN")
MODEL_STYLE = {
    "LR": {"color": "#1f77b4", "marker": "o", "offset": -0.12},
    "DNN": {"color": "#ff7f0e", "marker": "s", "offset": 0.12},
}
TARGET_TYPE_ORDER = ("disease", "adl")
TARGET_TYPE_LABELS = {"disease": "DISEASE", "adl": "ADL"}
METRIC_SPECS = {
    "ece": {"label": "Mean ECE", "better": "lower", "filename": "mean_ece_by_target.png"},
    "brier": {"label": "Mean Brier", "better": "lower", "filename": "mean_brier_by_target.png"},
    "auc": {"label": "Mean AUC", "better": "higher", "filename": "mean_auc_by_target.png"},
    "r2cal": {"label": "Mean Calibration R2", "better": "higher", "filename": "mean_r2cal_by_target.png"},
}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def read_metrics(path: Path, model_name: str, require_status_ok: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["model"] = model_name
    if require_status_ok and "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.lower()
        df = df[status.eq("ok")].copy()

    df["target"] = df["target"].astype(str).str.strip().str.upper()
    df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
    df["transition"] = df["transition"].astype(str).str.strip()
    return df


def summarize_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric, "target", "target_type", "transition", "model"])
    work = work[work["target_type"].isin(TARGET_TYPE_ORDER)].copy()

    if work.empty:
        return work

    grouped = (
        work.groupby(["model", "target_type", "target"], as_index=False)
        .agg(
            mean_value=(metric, "mean"),
            transition_count=(metric, "count"),
            std_value=(metric, lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
        )
    )
    grouped["se_value"] = grouped["std_value"] / np.sqrt(grouped["transition_count"].clip(lower=1))
    return grouped


def order_targets(panel: pd.DataFrame, better: str) -> list[str]:
    rank = (
        panel.pivot_table(index="target", columns="model", values="mean_value", aggfunc="mean")
        .reindex(columns=list(MODEL_ORDER))
    )
    if "DNN" in rank.columns:
        sort_key = rank["DNN"].fillna(rank.mean(axis=1))
    else:
        sort_key = rank.mean(axis=1)
    ascending = better == "higher"
    return sort_key.sort_values(ascending=ascending).index.tolist()


def compute_limits(panel: pd.DataFrame, better: str) -> tuple[float, float]:
    vals = np.concatenate(
        [
            np.maximum(0.0, panel["mean_value"].to_numpy(dtype=float) - panel["se_value"].to_numpy(dtype=float)),
            panel["mean_value"].to_numpy(dtype=float) + panel["se_value"].to_numpy(dtype=float),
        ]
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (0.0, 1.0)

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if better == "lower":
        return (0.0, min(1.0, hi * 1.10 if hi > 0 else 1.0))

    span = hi - lo
    pad = 0.03 if span <= 0 else 0.08 * span
    y_min = max(0.0, lo - pad)
    y_max = min(1.02, hi + pad)
    if y_max <= y_min:
        y_max = min(1.02, y_min + 0.05)
    return (y_min, y_max)


def title_text(dataset_label: str, run_tag: str, metric_label: str, event_type: str | None) -> str:
    left = dataset_label.strip()
    if run_tag.strip() and run_tag.strip() != dataset_label.strip():
        left = f"{left} | {run_tag}"
    if event_type:
        return f"{left} | {event_type.upper()} | {metric_label} by target"
    return f"{left} | {metric_label} by target"


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    out_path: Path,
    dataset_label: str,
    run_tag: str,
    event_type: str | None = None,
) -> None:
    spec = METRIC_SPECS[metric]
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=False)
    fig.suptitle(title_text(dataset_label, run_tag, spec["label"], event_type), fontsize=20, y=0.98)
    note = "avg over available transitions (error bars = SE across transitions)"
    if event_type:
        note += ", status = ok only"
    fig.text(0.06, 0.915, note, fontsize=11, ha="left")

    for idx, target_type in enumerate(TARGET_TYPE_ORDER):
        ax = axes[idx]
        panel = summary[summary["target_type"] == target_type].copy()
        ax.set_title(TARGET_TYPE_LABELS[target_type], fontsize=16, pad=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        if panel.empty:
            ax.text(0.5, 0.5, "No evaluable targets", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_ylabel(spec["label"])
            continue

        target_order = order_targets(panel, spec["better"])
        x = np.arange(len(target_order), dtype=float)

        for model in MODEL_ORDER:
            style = MODEL_STYLE[model]
            sub = (
                panel[panel["model"] == model]
                .set_index("target")
                .reindex(target_order)
            )
            mask = sub["mean_value"].notna().to_numpy()
            if not mask.any():
                continue
            sub_valid = sub.iloc[np.flatnonzero(mask)]
            ax.errorbar(
                x[mask] + style["offset"],
                sub_valid["mean_value"].to_numpy(dtype=float),
                yerr=sub_valid["se_value"].to_numpy(dtype=float),
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.4,
                capsize=3,
                markersize=6,
                linewidth=0,
                label=model,
            )

        y_min, y_max = compute_limits(panel, spec["better"])
        ax.set_ylim(y_min, y_max)
        if spec["better"] == "lower":
            ax.invert_yaxis()
        ax.set_ylabel(f"{spec['label']} (+/- SE across transitions)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(target_order, rotation=90, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", ncol=2, fontsize=12)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_family(
    lr_path: Path,
    dnn_path: Path,
    out_dir: Path,
    dataset_label: str,
    run_tag: str,
    require_status_ok: bool,
    event_type: str | None = None,
) -> None:
    lr = read_metrics(lr_path, "LR", require_status_ok=require_status_ok)
    dnn = read_metrics(dnn_path, "DNN", require_status_ok=require_status_ok)
    df = pd.concat([lr, dnn], ignore_index=True)
    if event_type:
        df = df[df["event_type"].astype(str).str.strip().str.lower().eq(event_type)].copy()

    for metric in METRIC_SPECS:
        summary = summarize_metric(df, metric)
        plot_metric(
            summary=summary,
            metric=metric,
            out_path=out_dir / METRIC_SPECS[metric]["filename"],
            dataset_label=dataset_label,
            run_tag=run_tag,
            event_type=event_type,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--dataset-label", default=None, help="Dataset label shown in figure titles, e.g. ELSA or HRS")
    ap.add_argument("--metrics-root", default="analysis/ELSA/metrics")
    ap.add_argument("--out-root", default="analysis/ELSA/figures/metrics_by_target")
    ap.add_argument("--include-damage-repair", action="store_true")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    dataset_label = args.dataset_label or run_tag

    metrics_root = Path(args.metrics_root)
    out_root = Path(args.out_root) / run_tag

    run_family(
        lr_path=metrics_root / "lr" / run_tag / "metrics_by_target.csv",
        dnn_path=metrics_root / "dnn" / run_tag / "metrics_by_target.csv",
        out_dir=out_root / "standard",
        dataset_label=dataset_label,
        run_tag=run_tag,
        require_status_ok=False,
    )

    if args.include_damage_repair:
        damage_root = metrics_root / "damage_repair"
        run_family(
            lr_path=damage_root / "lr" / run_tag / "metrics_by_target.csv",
            dnn_path=damage_root / "dnn" / run_tag / "metrics_by_target.csv",
            out_dir=out_root / "damage",
            dataset_label=dataset_label,
            run_tag=run_tag,
            require_status_ok=True,
            event_type="damage",
        )
        run_family(
            lr_path=damage_root / "lr" / run_tag / "metrics_by_target.csv",
            dnn_path=damage_root / "dnn" / run_tag / "metrics_by_target.csv",
            out_dir=out_root / "repair",
            dataset_label=dataset_label,
            run_tag=run_tag,
            require_status_ok=True,
            event_type="repair",
        )


if __name__ == "__main__":
    main()
