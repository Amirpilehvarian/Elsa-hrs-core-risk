#!/usr/bin/env python3
"""Plot damage risk versus age from derived damage-event tables.

Outputs:
- one pooled all-transitions figure with LR/DNN x disease/ADL overlays
- one facet figure showing the same overlay for each transition
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_style import apply_paper_style

apply_paper_style()


SERIES_SPECS = {
    ("LR", "disease"): {"label": "LR diseases", "color": "#1f77b4", "marker": "o"},
    ("DNN", "disease"): {"label": "DNN diseases", "color": "#ff7f0e", "marker": "s"},
    ("LR", "adl"): {"label": "LR ADLs", "color": "#2ca02c", "marker": "^"},
    ("DNN", "adl"): {"label": "DNN ADLs", "color": "#d62728", "marker": "D"},
}
TARGET_TYPE_ORDER = ("disease", "adl")
MODEL_ORDER = ("LR", "DNN")


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def normalize_id_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    return s.astype(str).str.extract(r"(\d+)", expand=False)


def transition_sort_key(name: str) -> tuple[int, int]:
    m = re.match(r"^S(\d+)_S(\d+)$", str(name).strip())
    if not m:
        return (999, 999)
    return (int(m.group(1)), int(m.group(2)))


def age_column_for_transition(df: pd.DataFrame, transition: str) -> str:
    t = transition_sort_key(transition)[0]
    preferred = f"S{t}INDAGER"
    if preferred in df.columns:
        return preferred
    age_cols = [c for c in df.columns if "AGE" in c.upper()]
    if age_cols:
        return age_cols[0]
    raise KeyError(f"No age column found for transition {transition}")


def lowess_1d(x: np.ndarray, y: np.ndarray, frac: float = 0.45, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    n = len(x)
    if n < 5:
        return x, y

    k = max(3, int(math.ceil(frac * n)))
    yhat = np.zeros(n, float)

    for i in range(n):
        d = np.abs(x - x[i])
        idx = np.argsort(d)[:k]
        xx = x[idx]
        yy = y[idx]

        dmax = np.max(np.abs(xx - x[i])) + eps
        u = np.abs(xx - x[i]) / dmax
        w = (1 - u**3) ** 3
        w[u >= 1] = 0.0

        X = np.vstack([np.ones_like(xx), xx]).T
        W = np.diag(w)
        xtwx = X.T @ W @ X
        if np.linalg.cond(xtwx) > 1e12:
            yhat[i] = np.average(yy, weights=w + eps)
        else:
            beta = np.linalg.solve(xtwx, X.T @ W @ yy)
            yhat[i] = beta[0] + beta[1] * x[i]

    return x, yhat


def load_damage_events_with_age(
    events_root: Path,
    transitions_root: Path,
    run_tag: str,
    model: str,
) -> pd.DataFrame:
    events_dir = events_root / model / run_tag
    transitions_dir = transitions_root / run_tag
    if not events_dir.exists():
        raise FileNotFoundError(f"Missing events directory: {events_dir}")
    if not transitions_dir.exists():
        raise FileNotFoundError(f"Missing transitions directory: {transitions_dir}")

    frames: list[pd.DataFrame] = []
    for fp in sorted(events_dir.glob("events_S*_S*.csv"), key=lambda p: transition_sort_key(p.stem.replace("events_", ""))):
        transition = fp.stem.replace("events_", "")
        events = pd.read_csv(fp)
        if events.empty:
            continue
        events = events[events["event_type"].astype(str).str.strip().str.lower().eq("damage")].copy()
        if events.empty:
            continue

        trans_fp = transitions_dir / f"{transition}.csv"
        if not trans_fp.exists():
            continue
        dft = pd.read_csv(trans_fp, usecols=lambda c: True)
        age_col = age_column_for_transition(dft, transition)
        age_df = dft[["idauniq", age_col]].copy().rename(columns={age_col: "age"})

        events["idauniq"] = normalize_id_series(events["idauniq"])
        age_df["idauniq"] = normalize_id_series(age_df["idauniq"])

        merged = events.merge(age_df, on="idauniq", how="left")
        merged["transition"] = transition
        merged["model"] = model
        merged["target_type"] = merged["target_type"].astype(str).str.strip().str.lower()
        merged["p"] = pd.to_numeric(merged["p"], errors="coerce")
        merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
        merged = merged.dropna(subset=["age", "p"])
        merged = merged[merged["target_type"].isin(TARGET_TYPE_ORDER)].copy()
        frames.append(merged[["transition", "model", "target_type", "age", "p"]])

    if not frames:
        raise RuntimeError(f"No damage event rows with age found for {model} {run_tag}")
    return pd.concat(frames, ignore_index=True)


def binned_mean_risk(
    df: pd.DataFrame,
    bin_width: int,
    min_n: int,
    age_min: float | None = None,
    age_max: float | None = None,
) -> pd.DataFrame:
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["p"], errors="coerce").to_numpy(float)
    mask = np.isfinite(age) & np.isfinite(p)
    age = age[mask]
    p = p[mask]
    if len(age) == 0:
        return pd.DataFrame(columns=["age_mid", "n", "mean_risk"])

    lo = float(math.floor(np.nanmin(age) / bin_width) * bin_width) if age_min is None else float(age_min)
    hi = float(math.ceil(np.nanmax(age) / bin_width) * bin_width) if age_max is None else float(age_max)
    edges = np.arange(lo, hi + bin_width, bin_width, dtype=float)
    rows = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if i < len(edges) - 2:
            m = (age >= left) & (age < right)
        else:
            m = (age >= left) & (age <= right)
        n = int(m.sum())
        if n < min_n:
            continue
        pp = p[m]
        rows.append(
            {
                "age_left": left,
                "age_right": right,
                "age_mid": 0.5 * (left + right),
                "n": n,
                "mean_risk": float(np.mean(pp)),
            }
        )
    return pd.DataFrame(rows)


def style_ax(ax: plt.Axes, title: str, ymax: float) -> None:
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Age")
    ax.set_ylabel("Risk")
    ax.set_ylim(0.0, ymax)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def plot_series(ax: plt.Axes, dt: pd.DataFrame, model: str, target_type: str, lowess_frac: float) -> None:
    if dt.empty:
        return
    spec = SERIES_SPECS[(model, target_type)]
    ax.scatter(dt["age_mid"], dt["mean_risk"], s=55, marker=spec["marker"], color=spec["color"], alpha=0.85, label=spec["label"])
    if len(dt) >= 4:
        xs, ys = lowess_1d(dt["age_mid"].to_numpy(), dt["mean_risk"].to_numpy(), frac=lowess_frac)
        ax.plot(xs, ys, linewidth=2.2, color=spec["color"], alpha=0.9)


def pooled_plot(
    df: pd.DataFrame,
    out_path: Path,
    dataset_label: str,
    run_tag: str,
    bin_width: int,
    min_n: int,
    ymax: float,
    lowess_frac: float,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    age_min = float(math.floor(df["age"].min() / bin_width) * bin_width)
    age_max = float(math.ceil(df["age"].max() / bin_width) * bin_width)

    for model in MODEL_ORDER:
        for target_type in TARGET_TYPE_ORDER:
            sub = df[(df["model"] == model) & (df["target_type"] == target_type)].copy()
            dt = binned_mean_risk(sub, bin_width=bin_width, min_n=min_n, age_min=age_min, age_max=age_max)
            plot_series(ax, dt, model=model, target_type=target_type, lowess_frac=lowess_frac)

    style_ax(ax, title=f"{dataset_label} | {run_tag} | DAMAGE risk vs age (all transitions pooled)", ymax=ymax)
    ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def facet_plot(
    df: pd.DataFrame,
    out_path: Path,
    dataset_label: str,
    run_tag: str,
    bin_width: int,
    min_n: int,
    ymax: float,
    lowess_frac: float,
) -> None:
    transitions = sorted(df["transition"].dropna().unique().tolist(), key=transition_sort_key)
    n = len(transitions)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    age_min = float(math.floor(df["age"].min() / bin_width) * bin_width)
    age_max = float(math.ceil(df["age"].max() / bin_width) * bin_width)

    for i, transition in enumerate(transitions):
        ax = axes[i]
        sub_tr = df[df["transition"] == transition].copy()
        for model in MODEL_ORDER:
            for target_type in TARGET_TYPE_ORDER:
                sub = sub_tr[(sub_tr["model"] == model) & (sub_tr["target_type"] == target_type)].copy()
                dt = binned_mean_risk(sub, bin_width=bin_width, min_n=min_n, age_min=age_min, age_max=age_max)
                plot_series(ax, dt, model=model, target_type=target_type, lowess_frac=lowess_frac)
        style_ax(ax, title=transition, ymax=ymax)
        if i == 0:
            ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{dataset_label} | {run_tag} | DAMAGE risk vs age by transition", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--dataset-label", default=None, help="Dataset label for figure titles")
    ap.add_argument("--events-root", default="analysis/ELSA/derived/damage_repair")
    ap.add_argument("--transitions-root", default="Transition_data/01_transitions")
    ap.add_argument("--out-root", default="analysis/ELSA/figures/damage_risk_vs_age")
    ap.add_argument("--bin-width", type=int, default=5)
    ap.add_argument("--transition-min-n", type=int, default=50)
    ap.add_argument("--pooled-min-n", type=int, default=100)
    ap.add_argument("--ymax", type=float, default=0.5)
    ap.add_argument("--lowess-frac", type=float, default=0.45)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    dataset_label = args.dataset_label or run_tag
    events_root = Path(args.events_root)
    transitions_root = Path(args.transitions_root)
    out_root = Path(args.out_root) / run_tag

    frames = []
    for model in MODEL_ORDER:
        frames.append(load_damage_events_with_age(events_root, transitions_root, run_tag, model))
    df = pd.concat(frames, ignore_index=True)

    pooled_plot(
        df=df,
        out_path=out_root / "damage_risk_vs_age_pooled.png",
        dataset_label=dataset_label,
        run_tag=run_tag,
        bin_width=args.bin_width,
        min_n=args.pooled_min_n,
        ymax=args.ymax,
        lowess_frac=args.lowess_frac,
    )
    facet_plot(
        df=df,
        out_path=out_root / "damage_risk_vs_age_by_transition.png",
        dataset_label=dataset_label,
        run_tag=run_tag,
        bin_width=args.bin_width,
        min_n=args.transition_min_n,
        ymax=args.ymax,
        lowess_frac=args.lowess_frac,
    )


if __name__ == "__main__":
    main()
