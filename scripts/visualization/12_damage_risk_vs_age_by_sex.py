#!/usr/bin/env python3
"""Plot pooled damage risk vs age stratified by sex."""

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


MODEL_ORDER = ("LR", "DNN")
SEX_ORDER = ("female", "male")
TARGET_TYPE_ORDER = ("disease", "adl")
SERIES_SPECS = {
    ("LR", "disease"): {"label": "LR diseases", "color": "#1f77b4", "marker": "o"},
    ("DNN", "disease"): {"label": "DNN diseases", "color": "#ff7f0e", "marker": "s"},
    ("LR", "adl"): {"label": "LR ADLs", "color": "#2ca02c", "marker": "^"},
    ("DNN", "adl"): {"label": "DNN ADLs", "color": "#d62728", "marker": "D"},
}


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
    raise KeyError(f"No age column found for {transition}")


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


def binned_mean_risk(df: pd.DataFrame, bin_width: int, min_n: int, age_min: float, age_max: float) -> pd.DataFrame:
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["p"], errors="coerce").to_numpy(float)
    mask = np.isfinite(age) & np.isfinite(p)
    age = age[mask]
    p = p[mask]
    if len(age) == 0:
        return pd.DataFrame(columns=["age_mid", "n", "mean_risk"])
    edges = np.arange(age_min, age_max + bin_width, bin_width, dtype=float)
    rows = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        m = (age >= left) & (age < right) if i < len(edges) - 2 else (age >= left) & (age <= right)
        n = int(m.sum())
        if n < min_n:
            continue
        rows.append({"age_mid": 0.5 * (left + right), "n": n, "mean_risk": float(np.mean(p[m]))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dataset-label", default=None)
    ap.add_argument("--events-root", required=True)
    ap.add_argument("--transitions-root", required=True)
    ap.add_argument("--sex-lookup", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bin-width", type=int, default=5)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--ymax", type=float, default=0.5)
    ap.add_argument("--lowess-frac", type=float, default=0.45)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    dataset_label = args.dataset_label or run_tag
    sex_lookup = pd.read_csv(args.sex_lookup)[["idauniq", "sex"]].copy()
    sex_lookup["idauniq"] = normalize_id_series(sex_lookup["idauniq"])
    sex_lookup["sex"] = sex_lookup["sex"].astype(str).str.strip().str.lower()
    sex_lookup = sex_lookup[sex_lookup["sex"].isin(SEX_ORDER)].drop_duplicates()

    frames = []
    for model in MODEL_ORDER:
        events_dir = Path(args.events_root) / model / run_tag
        transitions_dir = Path(args.transitions_root) / run_tag
        for fp in sorted(events_dir.glob("events_S*_S*.csv"), key=lambda p: transition_sort_key(p.stem.replace("events_", ""))):
            transition = fp.stem.replace("events_", "")
            df = pd.read_csv(fp)
            df["idauniq"] = normalize_id_series(df["idauniq"])
            df["sex"] = df["idauniq"].map(sex_lookup.set_index("idauniq")["sex"])
            df = df[df["event_type"].astype(str).str.strip().str.lower().eq("damage")].copy()
            df = df[df["sex"].isin(SEX_ORDER)].copy()
            if df.empty:
                continue
            trans_fp = transitions_dir / f"{transition}.csv"
            dft = pd.read_csv(trans_fp)
            age_col = age_column_for_transition(dft, transition)
            age_df = dft[["idauniq", age_col]].copy().rename(columns={age_col: "age"})
            age_df["idauniq"] = normalize_id_series(age_df["idauniq"])
            df = df.merge(age_df, on="idauniq", how="left")
            df["model"] = model
            df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    if "age" not in df_all.columns or df_all["age"].isna().all():
        raise ValueError("No evaluable age column after merging transitions into damage-event tables.")

    age_min = float(math.floor(df_all["age"].min() / args.bin_width) * args.bin_width)
    age_max = float(math.ceil(df_all["age"].max() / args.bin_width) * args.bin_width)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, sex in enumerate(SEX_ORDER):
        ax = axes[idx]
        block = df_all[df_all["sex"] == sex].copy()
        for model in MODEL_ORDER:
            for target_type in TARGET_TYPE_ORDER:
                sub = block[(block["model"] == model) & (block["target_type"] == target_type)].copy()
                dt = binned_mean_risk(sub, bin_width=args.bin_width, min_n=args.min_n, age_min=age_min, age_max=age_max)
                if dt.empty:
                    continue
                spec = SERIES_SPECS[(model, target_type)]
                ax.scatter(dt["age_mid"], dt["mean_risk"], s=50, marker=spec["marker"], color=spec["color"], alpha=0.85, label=spec["label"])
                if len(dt) >= 4:
                    xs, ys = lowess_1d(dt["age_mid"].to_numpy(), dt["mean_risk"].to_numpy(), frac=args.lowess_frac)
                    ax.plot(xs, ys, color=spec["color"], linewidth=2.0, alpha=0.9)
        ax.set_title(sex.title())
        ax.set_xlabel("Age")
        ax.set_ylim(0, args.ymax)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Risk")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.suptitle(f"{dataset_label} | {run_tag} | DAMAGE risk vs age by sex", y=0.98, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_fp = Path(args.out_root) / run_tag / "damage_risk_vs_age_by_sex.png"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
