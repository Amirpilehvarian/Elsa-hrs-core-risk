#!/usr/bin/env python3
"""
scripts/visualization/05_damage_repair_calibration_plots.py

Reads:
analysis/.../damage_repair/{MODEL}/{RUN_TAG}/events_all.csv

Creates combined rank-decile calibration plots with LR and DNN on the same figure
for:
- event_type: damage, repair
- target_type: disease, adl
- per transition and per target

Output:
analysis/.../damage_repair_calibration/{RUN_TAG}/
  {transition}/{event_type}/{target_type}/cal_{target}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-6
MODEL_COLORS = {"LR": "tab:blue", "DNN": "tab:orange"}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def decile_calibration(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    d = df[["p", "y"]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("p").reset_index(drop=True)
    d["bin"] = pd.qcut(np.arange(len(d)), q=n_bins, labels=False, duplicates="drop")

    g = d.groupby("bin", as_index=False).agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        y_mean=("y", "mean"),
    )
    g["y_se"] = np.sqrt((g["y_mean"] * (1 - g["y_mean"])) / g["n"].clip(lower=1))
    return g


def maybe_log_axes(loglog: bool) -> None:
    if not loglog:
        return
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-3, 1)
    plt.ylim(1e-3, 1)


def plot_curves(
    out_fp: Path,
    title: str,
    curves: list[tuple[str, pd.DataFrame]],
    loglog: bool,
) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot([EPS, 1 - EPS], [EPS, 1 - EPS], linestyle="--", linewidth=1)

    for model, tab in curves:
        x = tab["p_mean"].to_numpy(dtype=float)
        y = tab["y_mean"].to_numpy(dtype=float)
        y_se = tab["y_se"].to_numpy(dtype=float)

        if loglog:
            x = np.clip(x, EPS, 1 - EPS)
            y = np.clip(y, EPS, 1 - EPS)
            y_low = np.clip(y - y_se, EPS, 1 - EPS)
            y_high = np.clip(y + y_se, EPS, 1 - EPS)
        else:
            y_low = np.clip(y - y_se, 0.0, 1.0)
            y_high = np.clip(y + y_se, 0.0, 1.0)

        yerr = np.vstack([
            np.maximum(y - y_low, 0.0),
            np.maximum(y_high - y, 0.0),
        ])

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=2,
            linewidth=1,
            elinewidth=1,
            markersize=5,
            color=MODEL_COLORS.get(model),
            label=model,
        )

    plt.xlabel("Mean predicted probability (decile)")
    plt.ylabel("Observed event rate (decile)")
    plt.title(title)
    maybe_log_axes(loglog)
    if len(curves) > 1:
        plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=200)
    plt.close()


def load_model_events(events_root: Path, model: str, run_tag: str) -> pd.DataFrame:
    fp = events_root / model / run_tag / "events_all.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing events_all.csv: {fp}")
    df = pd.read_csv(fp)
    if df.empty:
        raise ValueError(f"{fp} is empty")
    df["model"] = model
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, choices=["MAR", "MNAR"])
    ap.add_argument("--method", default=None, choices=["Cart", "Pmm"])
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--model", default=None, choices=["LR", "DNN"], help="Deprecated single-model alias")
    ap.add_argument("--models", nargs="+", default=["LR", "DNN"], choices=["LR", "DNN"])

    ap.add_argument("--events-root", default="analysis/derived/damage_repair")
    ap.add_argument("--out-root", default="analysis/figures/damage_repair_calibration")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=200, help="min rows required to plot a model/target")
    ap.add_argument("--loglog", action="store_true", help="Use log-log axes for calibration plots")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    models = [args.model] if args.model else list(dict.fromkeys(args.models))

    events_root = Path(args.events_root)
    out_dir = Path(args.out_root) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for model in models:
        df = load_model_events(events_root, model, run_tag)
        frames.append(df)
        print(f"[OK] loaded {model} {run_tag} events | rows={len(df):,}")
    df_all = pd.concat(frames, ignore_index=True)

    print(f"[OK] writing plots to {out_dir}")

    group_cols = ["transition", "event_type", "target_type"]
    for (transition, event_type, target_type), block in df_all.groupby(group_cols):
        block_dir = out_dir / transition / event_type / target_type
        block_dir.mkdir(parents=True, exist_ok=True)

        pooled_curves: list[tuple[str, pd.DataFrame]] = []
        for model in models:
            sub = block[block["model"] == model]
            if len(sub) < args.min_n:
                continue
            tab = decile_calibration(sub, n_bins=args.n_bins)
            if tab.empty or len(tab) < 3:
                continue
            pooled_curves.append((model, tab))
        if pooled_curves:
            plot_curves(
                block_dir / f"cal_POOLED_{target_type}.png",
                f"{run_tag} | {transition} | {event_type} | POOLED_{target_type}",
                pooled_curves,
                loglog=args.loglog,
            )

        targets = sorted(block["target"].dropna().astype(str).unique().tolist())
        for target in targets:
            curves: list[tuple[str, pd.DataFrame]] = []
            for model in models:
                sub = block[(block["model"] == model) & (block["target"] == target)]
                if len(sub) < args.min_n:
                    continue
                tab = decile_calibration(sub, n_bins=args.n_bins)
                if tab.empty or len(tab) < 3:
                    continue
                curves.append((model, tab))
            if not curves:
                continue
            plot_curves(
                block_dir / f"cal_{target}.png",
                f"{run_tag} | {transition} | {event_type} | {target}",
                curves,
                loglog=args.loglog,
            )

        print(f"[OK] {transition} | {event_type} | {target_type} -> {block_dir}")

    print("[DONE] combined damage/repair calibration plots created.")


if __name__ == "__main__":
    main()
