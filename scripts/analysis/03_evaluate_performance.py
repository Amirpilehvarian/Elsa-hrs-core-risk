#!/usr/bin/env python3
"""
scripts/analysis/03_evaluate_performance.py

Purpose
-------
Evaluate predictive performance *per transition* (e.g., S1->S2) and *per target*
(ADL or Disease) using:
1) out-of-fold (OOF) predicted probabilities written by
   `02_LR_oof_probs_and_metrics.py` (or the DNN analog), and
2) the true outcomes stored in the transition CSVs written by
   `01_build_transitions.py`.

Inputs
------
- Transition datasets:
    Transition_data/01_transitions/<SCENARIO>_<METHOD>/S1_S2.csv, ...
  Each transition file contains target columns:
    Y_S<tp1>_<TARGET>
  and an identifier column `idauniq`.

- Probability files:
    <probs-dir>/<SCENARIO>_<METHOD>/probs_S1_S2.csv, ...
  Each probs file contains:
    idauniq
    P_S<tp1>_<TARGET>  (predicted probability for that target)

Key idea
--------
We join probabilities to ground-truth by `idauniq`, then compute metrics for
all targets that exist in both files.

Outputs
-------
- metrics_by_target.csv:
    One row per (scenario, method, transition, target) with:
    prevalence, AUC, PR-AUC, Brier, ECE, n_obs.

- metrics_summary.csv:
    Aggregated metrics per (scenario, method, transition, target_type).

Notes
-----
- Extremely rare targets may be skipped (not enough positives/negatives).
- Probabilities are clipped to (EPS, 1-EPS) to avoid numerical issues.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ---------- Canonical target lists (paper definitions) ----------
ADL_TARGETS = {
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA","TOILTA","MEALSA","MAPA",
    "BEDA","DIMEA","SHOPA","BATHA","ARMSA","DRESSA","WALK100A","SITA","CLIM1A","HOUSEWKA",
    "PUSHA","LIFTA","CHAIRA","CLIMSA","STOOPA"
}

DISEASE_TARGETS = {
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE","LUNGE","CANCRE",
    "ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE","ASTHMAE","CATRACTE","HCHOLE","ARTHRE","HIBPE"
}

ALL_TARGETS = ADL_TARGETS | DISEASE_TARGETS

# ---------- Constants ----------
EPS = 1e-6


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def quantile_edges(p: np.ndarray, n_bins: int = 10, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    edges[0] = max(edges[0], eps)
    edges[-1] = min(edges[-1], 1.0)
    return edges


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)

    edges = quantile_edges(p, n_bins=n_bins, eps=EPS)
    rows: list[tuple[float, float, int]] = []
    for k in range(n_bins):
        left, right = edges[k], edges[k + 1]
        if k < n_bins - 1:
            mask = (p >= left) & (p < right)
        else:
            mask = (p >= left) & (p <= right)

        nk = int(mask.sum())
        if nk == 0:
            continue
        rows.append((float(p[mask].mean()), float(y[mask].mean()), nk))

    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)


def r2_calibration_from_bins(tab: np.ndarray) -> float:
    if tab.shape[0] < 3:
        return np.nan
    reg = LinearRegression().fit(tab[:, 0].reshape(-1, 1), tab[:, 1])
    return float(reg.score(tab[:, 0].reshape(-1, 1), tab[:, 1]))


# ---------- Calibration metric: Expected Calibration Error (ECE) ----------
def expected_calibration_error(y, p, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and observed
    frequencies (accuracy) across bins of predictions. It quantifies how well
    calibrated the predicted probabilities are.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary outcomes (0 or 1).

    p : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    n_bins : int, default=10
        Number of equal-width bins to use between 0 and 1.

    Returns
    -------
    float
        The ECE value, a non-negative number where 0 indicates perfect calibration.

    Notes
    -----
    - The predictions are binned into uniform bins on [0,1].
    - Bins with no samples are ignored in the calculation.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1  # Bin indices for each prediction
    idx = np.clip(idx, 0, n_bins - 1)  # Ensure indices are valid

    ece = 0.0
    N = len(y)
    for k in range(n_bins):
        mask = idx == k
        if mask.sum() == 0:
            # Skip bins with no samples to avoid division by zero
            continue
        acc = y[mask].mean()  # Accuracy (observed frequency) in bin
        conf = p[mask].mean()  # Mean predicted probability in bin
        ece += (mask.sum() / N) * abs(acc - conf)  # Weighted absolute difference
    return float(ece)


# ---------- CLI entrypoint ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--model", default="LR", choices=["LR", "DNN"],
                    help="Which model's probability folder to read (LR or DNN).")
    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--probs-dir", default="analysis",
                    help="Root analysis directory that contains per-model prob folders.")
    ap.add_argument("--out-dir", default="analysis/metrics",
                    help="Root analysis directory to write per-model metrics outputs.")
    ap.add_argument("--calibration-bins", type=int, default=10,
                    help="Number of quantile bins for ECE and calibration R^2.")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""

    trans_dir = Path(args.transitions_dir) / run_tag

    # Support two directory layouts for probabilities/metrics:
    #   (A) legacy: <probs-dir>/<RUN_TAG>/probs_S1_S2.csv
    #   (B) model-specific: <probs-dir>/<model>/<RUN_TAG>/probs_S1_S2.csv
    probs_root = Path(args.probs_dir)
    legacy_probs = probs_root / run_tag
    model_probs = probs_root / args.model.lower() / run_tag
    probs_dir = legacy_probs if legacy_probs.exists() else model_probs

    # Metrics are written in a parallel model-specific folder:
    #   <out-dir>/<model>/<RUN_TAG>/
    out_root = Path(args.out_dir)
    out_dir = out_root / args.model.lower() / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Run tag:     {run_tag}")
    print(f"[OK] Transitions: {trans_dir}")
    print(f"[OK] Probs:       {probs_dir}")
    print(f"[OK] Out:         {out_dir}")

    rows = []

    # Find all probability files matching pattern probs_S*_S*.csv
    prob_files = sorted(probs_dir.glob("probs_S*_S*.csv"))
    if not prob_files:
        raise FileNotFoundError(f"No probability files found in {probs_dir}")

    for prob_fp in prob_files:
        # Extract transition name from filename, e.g. probs_S1_S2.csv -> S1_S2
        m = re.match(r"probs_(S\d+_S\d+)\.csv", prob_fp.name)
        if not m:
            continue
        trans = m.group(1)

        print(f"[INFO] Evaluating {trans}")

        # Load predicted probabilities and true outcomes for this transition
        df_p = pd.read_csv(prob_fp)
        df_t = pd.read_csv(trans_dir / f"{trans}.csv")

        # Merge on 'idauniq' to align predictions with true labels;
        # inner join ensures we only keep rows present in both datasets
        df = df_t.merge(df_p, on="idauniq", how="inner")

        for target in ALL_TARGETS:
            # Construct column names for true outcome and predicted probability
            # The outcome column is named Y_<tp2>_<target>, where tp2 is second state in transition
            ycol = f"Y_{trans.split('_')[1]}_{target}"
            pcol = f"P_{trans.split('_')[1]}_{target}"

            # Skip if either true or predicted columns are missing
            if ycol not in df.columns or pcol not in df.columns:
                continue

            # Convert columns to numeric, coercing errors to NaN
            y = pd.to_numeric(df[ycol], errors="coerce")
            p = pd.to_numeric(df[pcol], errors="coerce")

            # Mask to keep only rows with non-missing true and predicted values
            mask = y.notna() & p.notna()
            # Skip targets with fewer than 10 observations or only one class present
            if mask.sum() < 10 or y[mask].nunique() < 2:
                continue

            # Extract numpy arrays for true and predicted values
            yv = y[mask].astype(int).to_numpy()
            # Clip predicted probabilities to avoid numerical issues with metrics
            pv = np.clip(p[mask].to_numpy(), EPS, 1 - EPS)

            # Compute and store metrics for this target and transition
            rows.append({
                "model": args.model,
                "run_tag": run_tag,
                "scenario": scenario,
                "method": method,
                "transition": trans,
                "target": target,
                "target_type": "adl" if target in ADL_TARGETS else "disease",
                "n_obs": int(len(yv)),
                "prevalence": float(yv.mean()),  # Fraction positive
                "auc": float(roc_auc_score(yv, pv)),  # Area under ROC curve
                "pr_auc": float(average_precision_score(yv, pv)),  # Area under PR curve
                "brier": float(brier_score_loss(yv, pv)),  # Brier score (mean squared error)
                "ece": expected_calibration_error(yv, pv, n_bins=args.calibration_bins),
                "r2cal": r2_calibration_from_bins(calibration_table(yv, pv, n_bins=args.calibration_bins)),
            })

    # Create dataframe with metrics per target
    df_metrics = pd.DataFrame(rows)
    out_all = out_dir / "metrics_by_target.csv"
    df_metrics.to_csv(out_all, index=False)
    print(f"[OK] Saved {out_all}")

    # Summarize metrics by model, scenario, method, transition, and target type (adl/disease)
    summary = (
        df_metrics
        .groupby(["model","run_tag","scenario","method","transition","target_type"], as_index=False)
        .agg(
            n_targets=("target","count"),
            mean_auc=("auc","mean"),
            median_auc=("auc","median"),
            mean_brier=("brier","mean"),
            mean_ece=("ece","mean"),
            mean_pr_auc=("pr_auc","mean"),
            mean_r2cal=("r2cal","mean"),
        )
    )

    out_sum = out_dir / "metrics_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"[OK] Saved {out_sum}")


if __name__ == "__main__":
    main()
