#!/usr/bin/env python3
"""
scripts/visualization/02_ece_plots.py

Creates ECE-focused visualizations comparing LR vs DNN.

Inputs (expected):
- analysis/metrics/lr/<RUN_TAG>/metrics_by_target.csv
- analysis/metrics/dnn/<RUN_TAG>/metrics_by_target.csv

Outputs:
- analysis/metrics/plots/<SCENARIO>_<METHOD>/
    01_ece_box_by_transition.png
    02_ece_vs_prevalence.png
    03_ece_heatmap_transition_by_type.png
    04_delta_ece_worst_targets.png
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



ADL_TARGETS = {
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA",
    "TOILTA","MEALSA","MAPA","BEDA","DIMEA","SHOPA","BATHA","ARMSA",
    "DRESSA","WALK100A","SITA","CLIM1A","HOUSEWKA","PUSHA","LIFTA",
    "CHAIRA","CLIMSA","STOOPA"
}

DISEASE_TARGETS = {
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE",
    "LUNGE","CANCRE","ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE",
    "ASTHMAE","CATRACTE","HCHOLE","ARTHRE","HIBPE"
}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")

def read_metrics(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty metrics file: {path}")

    # Force model label
    df["model"] = model_name

    # Make status robust (DNN files often differ)
    if "status" not in df.columns:
        df["status"] = "ok"
    else:
        df["status"] = (
            df["status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--lr-dir", default="analysis/lr")
    ap.add_argument("--dnn-dir", default="analysis/dnn")
    ap.add_argument("--metrics-root", default=None,
                    help="If provided, read evaluated metrics from <metrics-root>/<lr|dnn>/<RUN_TAG>/metrics_by_target.csv")
    ap.add_argument("--out-dir", default="analysis/figures/ece_plots")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    tag = resolve_run_tag(args.run_tag, args.scenario, args.method)

    if args.metrics_root:
        metrics_root = Path(args.metrics_root)
        lr_path = metrics_root / "lr" / tag / "metrics_by_target.csv"
        dnn_path = metrics_root / "dnn" / tag / "metrics_by_target.csv"
    else:
        lr_path = Path(args.lr_dir) / tag / "metrics_by_target.csv"
        dnn_path = Path(args.dnn_dir) / tag / "metrics_by_target.csv"

    if not lr_path.exists():
        raise FileNotFoundError(f"Missing LR metrics: {lr_path}")
    if not dnn_path.exists():
        raise FileNotFoundError(f"Missing DNN metrics: {dnn_path}")

    outdir = Path(args.out_dir) / tag
    ensure_outdir(outdir)

    lr  = read_metrics(lr_path,  "LR")
    dnn = read_metrics(dnn_path, "DNN")

    # Keep only successful rows with numeric ECE
    df = pd.concat([lr, dnn], ignore_index=True)
    df = df[df["status"].eq("ok")].copy()
    df["ece"] = pd.to_numeric(df["ece"], errors="coerce")
    df["prevalence"] = pd.to_numeric(df["prevalence"], errors="coerce")
    df = df.dropna(subset=["ece", "prevalence", "transition", "target", "model"])

    # Normalize key columns
    df["transition"] = df["transition"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip().str.upper()

    df["target_type"] = np.where(
    df["target"].isin(ADL_TARGETS),
    "adl",
    np.where(
        df["target"].isin(DISEASE_TARGETS),
        "disease",
        "other"
    )
)

    # Optional: drop anything not ADL/Disease
    df = df[df["target_type"].isin(["adl", "disease"])]
    # ---------------------------
    # 1) Boxplot: ECE across targets per transition (LR vs DNN)
    # ---------------------------
    transitions = sorted(df["transition"].unique(), key=lambda s: int(s.split("_")[0][1:]))
    fig, ax = plt.subplots(figsize=(12, 5))

    # For each transition, create two boxplots (LR, DNN) side-by-side
    positions = []
    data = []
    labels = []
    pos = 1

    for tr in transitions:
        for model in ["LR", "DNN"]:
            vals = df[(df["transition"] == tr) & (df["model"] == model)]["ece"].values
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(pos)
            labels.append(f"{tr}\n{model}")
            pos += 1
        pos += 1  # gap between transitions

    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("ECE")
    ax.set_title(f"ECE across targets per transition ({tag})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "01_ece_box_by_transition.png", dpi=200)
    plt.close(fig)

    # ---------------------------
    # 2) Scatter: ECE vs prevalence (log-x)
    # ---------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ["LR", "DNN"]:
        sub = df[df["model"] == model]
        ax.scatter(sub["prevalence"], sub["ece"], alpha=0.35, s=18, label=model)

    ax.set_xscale("log")
    ax.set_xlabel("Prevalence (log scale)")
    ax.set_ylabel("ECE")
    ax.set_title(f"ECE vs prevalence ({tag})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_ece_vs_prevalence.png", dpi=200)
    plt.close(fig)

    # ---------------------------
    # 3) Heatmap: mean ECE by transition x target_type (LR vs DNN separate)
    # ---------------------------
    # Compute mean ECE for each (model, transition, target_type)
    piv = (
        df.groupby(["model", "transition", "target_type"], as_index=False)
          .agg(mean_ece=("ece", "mean"), n=("ece", "count"))
    )

    for model in ["LR", "DNN"]:
        sub = piv[piv["model"] == model].copy()
        # Build a 2-column matrix (adl, disease)
        sub["target_type"] = sub["target_type"].astype(str)
        mat = sub.pivot(index="transition", columns="target_type", values="mean_ece").reindex(transitions)

        arr = mat.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        if finite.any():
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
        else:
            vmin = 0.0
            vmax = 1.0
            print(f"[WARN] No finite ECE values for heatmap ({model}) - {tag}")

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        masked = np.ma.array(arr, mask=~finite)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_title(f"Mean ECE heatmap ({model}) - {tag}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"03_ece_heatmap_transition_by_type_{model}.png", dpi=200)
        plt.close(fig)

    # ---------------------------
    # 4) Worst targets by ΔECE (DNN - LR): negative means DNN better
    # ---------------------------
    # Merge LR and DNN per (transition, target)
    lr2 = df[df["model"] == "LR"][["transition","target","ece","prevalence","target_type"]].copy()
    lr2["transition"] = lr2["transition"].astype(str).str.strip()
    lr2["target"] = lr2["target"].astype(str).str.strip().str.upper()
    lr2 = lr2.rename(columns={"ece":"ece_lr"})

    dn2 = df[df["model"] == "DNN"][["transition","target","ece"]].copy()
    dn2["transition"] = dn2["transition"].astype(str).str.strip()
    dn2["target"] = dn2["target"].astype(str).str.strip().str.upper()
    dn2 = dn2.rename(columns={"ece":"ece_dnn"})

    m = lr2.merge(dn2, on=["transition","target"], how="left")
    m["delta_ece"] = m["ece_dnn"] - m["ece_lr"]

    # Take the worst LR-calibrated targets (highest LR ECE), and show how DNN changes them
    worst = m.sort_values("ece_lr", ascending=False).head(args.topk).copy()
    worst = worst.sort_values("ece_lr", ascending=True)

    if worst.empty:
        print(f"[WARN] No rows available for worst-target plot - {tag}")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(worst))
        ax.barh(y - 0.15, worst["ece_lr"], height=0.3, label="LR")
        ax.barh(y + 0.15, worst["ece_dnn"], height=0.3, label="DNN")
        ax.set_yticks(y)
        yt = [f"{t}:{g}" for t,g in zip(worst["transition"], worst["target"])]
        ax.set_yticklabels(yt)
        ax.set_xlabel("ECE")
        ax.set_title(f"Worst {args.topk} targets by LR ECE (compare DNN) - {tag}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "04_delta_ece_worst_targets.png", dpi=200)
        plt.close(fig)

    print(f"[OK] Wrote ECE plots to: {outdir}")


if __name__ == "__main__":
    main()
