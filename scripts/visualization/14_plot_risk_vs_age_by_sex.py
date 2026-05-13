#!/usr/bin/env python3
"""Plot target-level standard risk vs age by sex."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-codex"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_style import apply_paper_style

apply_paper_style()


SEX_ORDER = ("female", "male")
FAMILY_ORDER = ("disease", "adl")
MODEL_ORDER = ("LR", "DNN")
MODEL_STYLE = {
    "LR": {"color": "#1f77b4", "marker": "o", "label": "LR mean predicted"},
    "DNN": {"color": "#ff7f0e", "marker": "s", "label": "DNN mean predicted"},
}
OBS_STYLE = {"color": "#222222", "marker": "D", "label": "Observed prevalence"}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def grid_dims(n_targets: int) -> tuple[int, int]:
    if n_targets <= 8:
        return (2, 4)
    if n_targets <= 12:
        return (3, 4)
    if n_targets <= 18:
        return (4, 5)
    return (5, 5)


def plot_family_sex(df: pd.DataFrame, out_path: Path, dataset_label: str, run_tag: str, family: str, sex: str) -> None:
    sub = df[(df["target_type"] == family) & (df["sex"] == sex)].copy()
    targets = sorted(sub["target"].dropna().unique().tolist())
    if not targets:
        return

    nrows, ncols = grid_dims(len(targets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), constrained_layout=False)
    axes = np.array(axes).reshape(-1)

    handles = []
    labels = []

    for i, target in enumerate(targets):
        ax = axes[i]
        dt = sub[sub["target"] == target].copy()

        obs = (
            dt.groupby(["age_mid"], as_index=False)
            .agg(observed_rate=("observed_rate", "mean"), n_obs=("n_obs", "sum"))
            .sort_values("age_mid")
        )
        if not obs.empty:
            h = ax.plot(
                obs["age_mid"],
                obs["observed_rate"],
                color=OBS_STYLE["color"],
                marker=OBS_STYLE["marker"],
                linestyle="--",
                linewidth=1.7,
                markersize=4.5,
                label=OBS_STYLE["label"],
            )[0]
            if OBS_STYLE["label"] not in labels:
                handles.append(h)
                labels.append(OBS_STYLE["label"])

        for model in MODEL_ORDER:
            dm = dt[dt["model"] == model].sort_values("age_mid")
            if dm.empty:
                continue
            style = MODEL_STYLE[model]
            h = ax.plot(
                dm["age_mid"],
                dm["mean_predicted"],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.8,
                markersize=4.5,
                label=style["label"],
            )[0]
            if style["label"] not in labels:
                handles.append(h)
                labels.append(style["label"])

        ymax = float(max(dt["mean_predicted"].max(), dt["observed_rate"].max()))
        ax.set_ylim(0.0, min(1.0, ymax * 1.18 + 0.01))
        ax.set_title(target, fontsize=10.5)
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        ax.tick_params(axis="y", labelsize=8)

        if i % ncols == 0:
            ax.set_ylabel("Risk")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Age")

    for j in range(len(targets), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{dataset_label} | {run_tag} | {family.upper()} | {sex.capitalize()} risk vs age",
        fontsize=18,
        y=0.99,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot target-level standard risk vs age by sex.")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dataset-label", required=True)
    ap.add_argument("--metrics-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    in_dir = Path(args.metrics_root) / run_tag
    out_dir = Path(args.out_root) / run_tag

    df = pd.read_csv(in_dir / "target_binned_risk.csv")
    for family in FAMILY_ORDER:
        for sex in SEX_ORDER:
            plot_family_sex(
                df=df,
                out_path=out_dir / f"{family}_{sex}_risk_vs_age.png",
                dataset_label=args.dataset_label,
                run_tag=run_tag,
                family=family,
                sex=sex,
            )
    print(f"[OK] Wrote target-level risk-vs-age figures under {out_dir}")


if __name__ == "__main__":
    main()
