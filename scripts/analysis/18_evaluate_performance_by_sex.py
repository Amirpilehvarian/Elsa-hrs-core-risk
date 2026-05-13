#!/usr/bin/env python3
"""Evaluate standard next-wave prediction metrics stratified by sex."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


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
SEX_ORDER = ("female", "male")
EPS = 1e-6


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
        mask = (p >= left) & (p < right) if k < n_bins - 1 else (p >= left) & (p <= right)
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


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for k in range(n_bins):
        mask = idx == k
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--model", default="LR", choices=["LR", "DNN"])
    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--probs-dir", default="analysis/ELSA")
    ap.add_argument("--sex-lookup", required=True)
    ap.add_argument("--out-dir", default="analysis/ELSA/metrics/by_sex")
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=10)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""

    sex_lookup = pd.read_csv(args.sex_lookup)
    sex_lookup["idauniq"] = normalize_id_series(sex_lookup["idauniq"])
    sex_lookup["sex"] = sex_lookup["sex"].astype(str).str.strip().str.lower()
    sex_lookup = sex_lookup[sex_lookup["sex"].isin(SEX_ORDER)][["idauniq", "sex"]].drop_duplicates()

    trans_dir = Path(args.transitions_dir) / run_tag
    probs_dir = Path(args.probs_dir) / args.model.lower() / run_tag
    out_dir = Path(args.out_dir) / args.model.lower() / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for prob_fp in sorted(probs_dir.glob("probs_S*_S*.csv")):
        m = re.match(r"probs_(S\d+_S\d+)\.csv", prob_fp.name)
        if not m:
            continue
        trans = m.group(1)
        df_p = pd.read_csv(prob_fp)
        df_t = pd.read_csv(trans_dir / f"{trans}.csv")
        df_p["idauniq"] = normalize_id_series(df_p["idauniq"])
        df_t["idauniq"] = normalize_id_series(df_t["idauniq"])
        df = df_t.merge(df_p, on="idauniq", how="inner").merge(sex_lookup, on="idauniq", how="inner")

        for target in ALL_TARGETS:
            ycol = f"Y_{trans.split('_')[1]}_{target}"
            pcol = f"P_{trans.split('_')[1]}_{target}"
            if ycol not in df.columns or pcol not in df.columns:
                continue
            for sex in SEX_ORDER:
                block = df[df["sex"] == sex].copy()
                if block.empty:
                    continue
                y = pd.to_numeric(block[ycol], errors="coerce")
                p = pd.to_numeric(block[pcol], errors="coerce")
                mask = y.notna() & p.notna()
                row = {
                    "model": args.model,
                    "run_tag": run_tag,
                    "scenario": scenario,
                    "method": method,
                    "sex": sex,
                    "transition": trans,
                    "target": target,
                    "target_type": "adl" if target in ADL_TARGETS else "disease",
                    "n_obs": int(mask.sum()),
                    "prevalence": np.nan,
                    "auc": np.nan,
                    "pr_auc": np.nan,
                    "brier": np.nan,
                    "ece": np.nan,
                    "r2cal": np.nan,
                    "status": "ok",
                }
                if mask.sum() < args.min_n:
                    row["status"] = "skipped_too_few_rows"
                    rows.append(row)
                    continue
                yv = y[mask].astype(int).to_numpy()
                pv = np.clip(p[mask].to_numpy(dtype=float), EPS, 1 - EPS)
                row["prevalence"] = float(yv.mean())
                if np.unique(yv).size < 2:
                    row["status"] = "skipped_one_class"
                    rows.append(row)
                    continue
                row["auc"] = float(roc_auc_score(yv, pv))
                row["pr_auc"] = float(average_precision_score(yv, pv))
                row["brier"] = float(brier_score_loss(yv, pv))
                row["ece"] = float(expected_calibration_error(yv, pv, n_bins=args.calibration_bins))
                row["r2cal"] = float(r2_calibration_from_bins(calibration_table(yv, pv, n_bins=args.calibration_bins)))
                rows.append(row)

    metrics_by_target = pd.DataFrame(rows)
    metrics_by_target.to_csv(out_dir / "metrics_by_target.csv", index=False)

    ok = metrics_by_target[metrics_by_target["status"] == "ok"].copy()
    summary = (
        ok.groupby(["model", "run_tag", "scenario", "method", "sex", "target_type"], as_index=False)
        .agg(
            n_target_transition_pairs=("target", "size"),
            n_unique_targets=("target", "nunique"),
            mean_auc=("auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
            mean_r2cal=("r2cal", "mean"),
            mean_prevalence=("prevalence", "mean"),
        )
        .sort_values(["sex", "target_type", "model"])
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    summary_by_transition = (
        ok.groupby(["model", "run_tag", "scenario", "method", "sex", "transition", "target_type"], as_index=False)
        .agg(
            n_target_transition_pairs=("target", "size"),
            n_unique_targets=("target", "nunique"),
            mean_auc=("auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
            mean_r2cal=("r2cal", "mean"),
            mean_prevalence=("prevalence", "mean"),
        )
        .sort_values(["sex", "transition", "target_type", "model"])
    )
    summary_by_transition.to_csv(out_dir / "metrics_summary_by_transition.csv", index=False)

    print(f"[OK] Saved {out_dir / 'metrics_by_target.csv'}")
    print(f"[OK] Saved {out_dir / 'metrics_summary.csv'}")
    print(f"[OK] Saved {out_dir / 'metrics_summary_by_transition.csv'}")


if __name__ == "__main__":
    main()
