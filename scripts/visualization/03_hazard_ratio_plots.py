#!/usr/bin/env python3
"""Top-decile event-enrichment plots from precomputed HR tables.

This script is intentionally simple:
- It reads the per-(transition,target) enrichment table produced by the analysis step.
- It aggregates enrichment ratios across transitions for each target (mean + SEM).
- It writes two plots per model: disease and ADL event enrichment.

Definition (as in your paper):
  ER10 = (observed event rate in top predicted-risk decile) / (overall observed event rate)
So ER10 is bounded in [1, 10].

Expected input layout (default):
  analysis/hazard_ratio_tables/<MODEL>/<SCENARIO>_<METHOD>/HR_by_target.csv

Additionally, the script will look for:
  analysis/metrics/hazard_ratio/<MODEL>/<SCENARIO>_<METHOD>/HR_by_transition.csv
which should contain per-transition HR rows.

Each HR_by_target.csv can be either:
  A) Per-transition table with columns: transition, target, hr
  B) Already-aggregated table with columns like: target, type, HR (optionally HR_sd, n_transitions, etc.)

Example:
  python3 scripts/visualization/03_hazard_ratio_plots.py --scenario MAR --method Cart --models LR DNN
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

# --- Target sets (canonical labels) -------------------------------------------------

ADL_TARGETS = {
    "DANGERA", "EATA", "MEDSA", "COMMUNA", "PHONEA", "MONEYA", "WALKRA", "TOILTA",
    "MEALSA", "MAPA", "BEDA", "DIMEA", "SHOPA", "BATHA", "ARMSA", "DRESSA",
    "WALK100A", "SITA", "CLIM1A", "HOUSEWKA", "PUSHA", "LIFTA", "CHAIRA",
    "CLIMSA", "STOOPA",
}

DISEASE_TARGETS = {
    "PARKINE", "CONHRTFE", "HIPE", "HEARTE", "HRTMRE", "HRTATTE", "STROKE", "LUNGE",
    "CANCRE", "ANGINE", "OSTEOE", "HRTRHME", "PSYCHE", "DIABE", "ASTHMAE", "CATRACTE",
    "HCHOLE", "ARTHRE", "HIBPE",
}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


# --- Helpers ------------------------------------------------------------------------

def _safe_sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def load_hr_raw(base_dir: Path, model: str, run_tag: str) -> pd.DataFrame:
    """Load per-transition HR rows.

    Preferred file: HR_by_transition.csv

    Returns a DataFrame with columns:
      transition, target, hr

    Accepts common column variants:
      - transition,target,hr
      - transition,target,HR
      - transition,target,HR_val
      - transition,target,hr_mean
    If neither HR_by_transition.csv nor a per-transition HR_by_target.csv is present, returns empty.
    """
    # 1) Prefer explicit per-transition file
    fp = base_dir / model / run_tag / "HR_by_transition.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        cols_upper = {c: c.strip().upper() for c in df.columns}
        df = df.rename(columns={c: cols_upper[c] for c in df.columns})

        if not {"TRANSITION", "TARGET"}.issubset(df.columns):
            return pd.DataFrame(columns=["transition", "target", "hr"])

        # pick hr column
        hr_col = None
        for cand in ["HR", "HR_VAL", "HR_MEAN", "HRATIO", "HR_VALUE", "HRVAL", "HR_"]:
            if cand in df.columns:
                hr_col = cand
                break
        if hr_col is None:
            # also allow lowercase 'hr' after normalization
            if "HR" in df.columns:
                hr_col = "HR"
            else:
                return pd.DataFrame(columns=["transition", "target", "hr"])

        tmp = df.copy()
        tmp["TRANSITION"] = tmp["TRANSITION"].astype(str).str.strip().str.upper()
        tmp["TARGET"] = tmp["TARGET"].astype(str).str.strip().str.upper()
        tmp[hr_col] = pd.to_numeric(tmp[hr_col], errors="coerce")
        tmp = tmp.dropna(subset=[hr_col]).copy()

        return tmp[["TRANSITION", "TARGET", hr_col]].rename(
            columns={"TRANSITION": "transition", "TARGET": "target", hr_col: "hr"}
        )

    # 2) Fallback: HR_by_target.csv might already contain per-transition rows
    fp2 = base_dir / model / run_tag / "HR_by_target.csv"
    if not fp2.exists():
        return pd.DataFrame(columns=["transition", "target", "hr"])

    df2 = pd.read_csv(fp2)
    cols_upper = {c: c.strip().upper() for c in df2.columns}
    df2 = df2.rename(columns={c: cols_upper[c] for c in df2.columns})

    if "TRANSITION" not in df2.columns or "TARGET" not in df2.columns:
        return pd.DataFrame(columns=["transition", "target", "hr"])

    tmp = df2.copy()
    tmp["TRANSITION"] = tmp["TRANSITION"].astype(str).str.strip().str.upper()
    tmp["TARGET"] = tmp["TARGET"].astype(str).str.strip().str.upper()

    if "HR" in tmp.columns:
        hr_col = "HR"
    elif "HR_VAL" in tmp.columns:
        hr_col = "HR_VAL"
    else:
        return pd.DataFrame(columns=["transition", "target", "hr"])

    tmp[hr_col] = pd.to_numeric(tmp[hr_col], errors="coerce")
    tmp = tmp.dropna(subset=[hr_col]).copy()

    return tmp[["TRANSITION", "TARGET", hr_col]].rename(
        columns={"TRANSITION": "transition", "TARGET": "target", hr_col: "hr"}
    )


def load_hr_table(base_dir: Path, model: str, run_tag: str) -> pd.DataFrame:
    """Load HR_by_target.csv for one model/scenario/method.

    Supports two input formats:
      A) Per-transition table: transition, target, hr
      B) Summary table (already aggregated): target, type, HR (and optional HR_sd, n_transitions, ...)

    The function normalizes to a common schema with at least:
      model, scenario, method, target, group, hr_mean, hr_sem, n_transitions

    Notes:
      - If the input is per-transition, hr_mean/hr_sem are computed across transitions.
      - If the input is already summarized, we use HR as hr_mean and HR_sd as hr_sem (best effort).
    """
    fp = base_dir / model / run_tag / "HR_by_target.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing HR table: {fp}")

    df = pd.read_csv(fp)

    # Normalize column names for robust matching
    cols_upper = {c: c.strip().upper() for c in df.columns}
    df = df.rename(columns={c: cols_upper[c] for c in df.columns})

    # Helper to normalize target strings
    def _norm_target(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.upper()

    if {"TRANSITION", "TARGET"}.issubset(df.columns) and "HR" in df.columns:
        # Interpret HR as a legacy column name for the ER10 event-enrichment value.
        tmp = df[["TRANSITION", "TARGET", "HR"]].copy()
        tmp["TARGET"] = _norm_target(tmp["TARGET"])
        tmp["TRANSITION"] = tmp["TRANSITION"].astype(str).str.strip().str.upper()
        tmp["HR"] = pd.to_numeric(tmp["HR"], errors="coerce")
        tmp = tmp.dropna(subset=["HR"]).copy()

        g = (
            tmp.groupby(["TARGET"], as_index=False)
            .agg(
                hr_mean=("HR", "mean"),
                hr_sem=("HR", _safe_sem),
                n_transitions=("TRANSITION", "nunique"),
            )
        )
        g["group"] = np.where(g["TARGET"].isin(DISEASE_TARGETS), "Disease",
                              np.where(g["TARGET"].isin(ADL_TARGETS), "ADL", "Other"))
        g["model"] = model
        g["run_tag"] = run_tag
        g = g.rename(columns={"TARGET": "target"})
        return g

    # --- Case A: per-transition table with HR_VAL column -----------------------------
    if {"TRANSITION", "TARGET"}.issubset(df.columns) and "HR_VAL" in df.columns:
        df["TARGET"] = _norm_target(df["TARGET"])
        df["TRANSITION"] = df["TRANSITION"].astype(str).str.strip().str.upper()
        df["HR_VAL"] = pd.to_numeric(df["HR_VAL"], errors="coerce")
        df = df.dropna(subset=["HR_VAL"]).copy()

        # Aggregate across transitions per target
        g = (
            df.groupby(["TARGET"], as_index=False)
            .agg(
                hr_mean=("HR_VAL", "mean"),
                hr_sem=("HR_VAL", _safe_sem),
                n_transitions=("TRANSITION", "nunique"),
            )
        )

        # Attach group using canonical target sets
        g["group"] = np.where(g["TARGET"].isin(DISEASE_TARGETS), "Disease",
                              np.where(g["TARGET"].isin(ADL_TARGETS), "ADL", "Other"))

        g["model"] = model
        g["run_tag"] = run_tag
        g = g.rename(columns={"TARGET": "target"})
        return g

    # --- Case B: already aggregated summary ------------------------------------------
    # Expected columns like: TARGET, TYPE, HR, (optional) HR_SD, N_TRANSITIONS
    if "TARGET" in df.columns and "HR" in df.columns:
        out = df.copy()
        out["TARGET"] = _norm_target(out["TARGET"])
        out["HR"] = pd.to_numeric(out["HR"], errors="coerce")
        out = out.dropna(subset=["HR"]).copy()

        # Determine group
        if "TYPE" in out.columns:
            # normalize to our labels
            t = out["TYPE"].astype(str).str.strip().str.upper()
            out["group"] = np.where(t.str.contains("DISEASE"), "Disease",
                                    np.where(t.str.contains("ADL"), "ADL", t))
        else:
            out["group"] = np.where(out["TARGET"].isin(DISEASE_TARGETS), "Disease",
                                    np.where(out["TARGET"].isin(ADL_TARGETS), "ADL", "Other"))

        out = out.rename(columns={"TARGET": "target", "HR": "hr_mean"})

        # Best-effort uncertainty
        if "HR_SD" in out.columns:
            out["hr_sem"] = pd.to_numeric(out["HR_SD"], errors="coerce").fillna(0.0)
        elif "HR_SEM" in out.columns:
            out["hr_sem"] = pd.to_numeric(out["HR_SEM"], errors="coerce").fillna(0.0)
        else:
            out["hr_sem"] = 0.0

        if "N_TRANSITIONS" in out.columns:
            out["n_transitions"] = pd.to_numeric(out["N_TRANSITIONS"], errors="coerce").fillna(0).astype(int)
        else:
            # If not present, assume the summary already pooled across transitions
            out["n_transitions"] = 0

        out["model"] = model
        out["run_tag"] = run_tag

        keep_cols = ["model", "run_tag", "target", "group", "hr_mean", "hr_sem", "n_transitions"]
        return out[keep_cols].copy()

    raise ValueError(
        f"Unrecognized HR table format in {fp}. Columns found: {list(df.columns)}\n"
        "Expected either (transition,target,hr/HR) or (target,type,HR,...)."
    )


def summarize_hr(df: pd.DataFrame, target_set: set[str]) -> pd.DataFrame:
    """Filter an already-aggregated HR table to the requested target set and sort."""
    sub = df[df["target"].isin(target_set)].copy()
    if sub.empty:
        return sub

    # Ensure numeric
    sub["hr_mean"] = pd.to_numeric(sub["hr_mean"], errors="coerce")
    sub["hr_sem"] = pd.to_numeric(sub["hr_sem"], errors="coerce").fillna(0.0)
    sub = sub.dropna(subset=["hr_mean"]).copy()

    # Sort: highest HR first
    sub = sub.sort_values(["hr_mean", "target"], ascending=[False, True]).reset_index(drop=True)
    return sub


def summarize_hr_raw(df_raw: pd.DataFrame, target_set: set[str]) -> pd.DataFrame:
    """Filter raw per-transition HR rows to a target set and rank-order (highest first)."""
    if df_raw.empty:
        return df_raw.copy()

    sub = df_raw[df_raw["target"].isin(target_set)].copy()
    if sub.empty:
        return sub

    sub["hr"] = pd.to_numeric(sub["hr"], errors="coerce")
    sub = sub.dropna(subset=["hr"]).copy()
    sub = sub.sort_values(["hr", "target"], ascending=[False, True]).reset_index(drop=True)
    return sub


def plot_hr_points(summary: pd.DataFrame, out_png: Path, title: str) -> None:
    """Plot HR per target with SEM error bars."""
    if summary.empty:
        # write an empty placeholder plot (helps pipeline)
        fig = plt.figure(figsize=(10, 3))
        plt.axis("off")
        plt.title(title + " (no data)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    x = np.arange(len(summary))
    y = summary["hr_mean"].to_numpy(dtype=float)
    yerr = summary["hr_sem"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(max(10, 0.35 * len(summary)), 4.5))
    ax = plt.gca()

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=2, markersize=4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["target"].tolist(), rotation=90)
    ax.set_ylabel(r"Event enrichment $\mathrm{ER}_{10}$")
    ax.set_title(title)

    ax.set_ylim(1, 10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def plot_hr_points_noerr(summary: pd.DataFrame, out_png: Path, title: str) -> None:
    """Plot HR per target (no error bars), paper-style rank ordered."""
    if summary.empty:
        fig = plt.figure(figsize=(10, 3))
        plt.axis("off")
        plt.title(title + " (no data)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    x = np.arange(len(summary))
    y = summary["hr"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(max(10, 0.35 * len(summary)), 4.5))
    ax = plt.gca()
    ax.plot(x, y, "o", markersize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["target"].tolist(), rotation=90)
    ax.set_ylabel(r"Event enrichment $\mathrm{ER}_{10}$")
    ax.set_title(title)
    ax.set_ylim(1, 10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    p.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    p.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")

    p.add_argument(
        "--models",
        nargs="+",
        default=["LR", "DNN"],
        help="Which model subfolders to plot (default: LR DNN)",
    )

    p.add_argument(
        "--hr-root",
        default="analysis/metrics/hazard_ratio",
        help="Root folder containing HR tables (default: analysis/metrics/hazard_ratio)",
    )

    p.add_argument(
        "--out-root",
        default="analysis/figures/hazard_ratio",
        help="Where to write figures and summaries (default: analysis/figures/hazard_ratio)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    models = [str(m).strip() for m in args.models]

    hr_root = Path(args.hr_root)
    out_root = Path(args.out_root) / run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model in models:
        df = load_hr_table(hr_root, model=model, run_tag=run_tag)

        summ_dis = summarize_hr(df, DISEASE_TARGETS)
        summ_adl = summarize_hr(df, ADL_TARGETS)

        # Save summary tables
        model_dir = out_root / model
        model_dir.mkdir(parents=True, exist_ok=True)

        dis_csv = model_dir / "HR_summary_Disease.csv"
        adl_csv = model_dir / "HR_summary_ADL.csv"
        summ_dis.to_csv(dis_csv, index=False)
        summ_adl.to_csv(adl_csv, index=False)

        # Plots
        plot_hr_points(
            summ_dis,
            out_png=model_dir / "HR_Disease.png",
            title=fr"$\mathrm{{ER}}_{{10}}$ (Diseases) — {model} — {run_tag}",
        )
        plot_hr_points(
            summ_adl,
            out_png=model_dir / "HR_ADL.png",
            title=fr"$\mathrm{{ER}}_{{10}}$ (ADLs) — {model} — {run_tag}",
        )

        # --- Per-transition plots (if raw per-transition HR rows are available) ---
        raw = load_hr_raw(hr_root, model=model, run_tag=run_tag)
        if not raw.empty:
            print(f"[OK] {model}: per-transition HR rows: {len(raw):,} | transitions: {raw['transition'].nunique()}")
            per_dir = model_dir / "per_transition"
            per_dir.mkdir(parents=True, exist_ok=True)

            for trans, dft in raw.groupby("transition"):
                trans_dir = per_dir / trans
                trans_dir.mkdir(parents=True, exist_ok=True)

                dis_t = summarize_hr_raw(dft, DISEASE_TARGETS)
                adl_t = summarize_hr_raw(dft, ADL_TARGETS)

                # Save ranked tables
                dis_t.to_csv(trans_dir / "HR_ranked_Disease.csv", index=False)
                adl_t.to_csv(trans_dir / "HR_ranked_ADL.csv", index=False)

                # Plots (no error bars per transition)
                plot_hr_points_noerr(
                    dis_t,
                    out_png=trans_dir / "HR_Disease.png",
                    title=fr"$\mathrm{{ER}}_{{10}}$ (Diseases) — {model} — {run_tag} — {trans}",
                )
                plot_hr_points_noerr(
                    adl_t,
                    out_png=trans_dir / "HR_ADL.png",
                    title=fr"$\mathrm{{ER}}_{{10}}$ (ADLs) — {model} — {run_tag} — {trans}",
                )

            print(f"[OK] {model}: wrote per-transition HR plots -> {per_dir}")
        else:
            print(f"[INFO] {model}: HR_by_target.csv has no per-transition rows; skipping per-transition plots")

        all_summaries.append(
            pd.concat(
                [
                    summ_dis.assign(group="Disease"),
                    summ_adl.assign(group="ADL"),
                ],
                ignore_index=True,
            )
        )

        print(f"[OK] {model}: wrote plots + summaries -> {model_dir}")

    if all_summaries:
        all_df = pd.concat(all_summaries, ignore_index=True)
        all_df.to_csv(out_root / "HR_summary_all_models.csv", index=False)

    print(f"[DONE] Outputs in: {out_root}")


if __name__ == "__main__":
    main()
