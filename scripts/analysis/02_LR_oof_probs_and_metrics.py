#!/usr/bin/env python3
"""One-vs-rest logistic regression with out-of-fold probabilities.

This script is the LR counterpart to the DNN OOF pipeline:
- one independent LR model per target
- per-target complete-case rows
- stratified K-fold when feasible, with K-fold fallback
- wide probability output with one OOF probability column per target
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ADL_TARGETS = [
    "DANGERA", "EATA", "MEDSA", "COMMUNA", "PHONEA", "MONEYA", "WALKRA", "TOILTA", "MEALSA", "MAPA",
    "BEDA", "DIMEA", "SHOPA", "BATHA", "ARMSA", "DRESSA", "WALK100A", "SITA", "CLIM1A", "HOUSEWKA",
    "PUSHA", "LIFTA", "CHAIRA", "CLIMSA", "STOOPA",
]
DISEASE_TARGETS = [
    "PARKINE", "CONHRTFE", "HIPE", "HEARTE", "HRTMRE", "HRTATTE", "STROKE", "LUNGE", "CANCRE",
    "ANGINE", "OSTEOE", "HRTRHME", "PSYCHE", "DIABE", "ASTHMAE", "CATRACTE", "HCHOLE", "ARTHRE", "HIBPE",
]
ALL_TARGETS = ADL_TARGETS + DISEASE_TARGETS
EPS = 1e-6


def infer_transition_from_filename(fp: Path) -> tuple[int, int] | None:
    match = re.match(r"^S(\d+)_S(\d+)\.csv$", fp.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def build_lr_pipeline(max_iter: int = 2000, class_weight: str | None = None) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=max_iter, solver="lbfgs", class_weight=class_weight)),
        ]
    )


def coerce_numeric_X(df: pd.DataFrame, X_cols: list[str]) -> pd.DataFrame:
    return df[X_cols].apply(pd.to_numeric, errors="coerce")


def is_binary_vector(y: np.ndarray) -> bool:
    y = np.asarray(y)
    observed = y[~pd.isna(y)]
    if observed.size == 0:
        return False
    return set(np.unique(observed).tolist()).issubset({0, 1})


def choose_splitter(y: np.ndarray, requested_folds: int, seed: int):
    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts.min())
    n_splits = min(int(requested_folds), min_class)
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits, "stratified"
    return None, 0, "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions", help="Root transitions directory")
    ap.add_argument("--out-dir", default="analysis/lr", help="Where to write outputs")

    ap.add_argument("--cv-folds", type=int, default=5, help="Requested number of CV folds for OOF predictions")
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Deprecated and ignored. Kept only for backward-compatible CLI calls.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--include-age", action="store_true", help="Also include Age as predictor if present")

    ap.add_argument("--max-iter", type=int, default=2000, help="LogisticRegression max_iter")
    ap.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Use class_weight='balanced' or not",
    )

    ap.add_argument(
        "--export-probs",
        action="store_true",
        help="If set, write per-individual OOF probabilities per transition.",
    )
    ap.add_argument(
        "--probs-include-y",
        action="store_true",
        help="If set, include observed Y columns alongside probabilities in exported probs CSVs.",
    )
    ap.add_argument(
        "--probs-include-x",
        action="store_true",
        help="If set, include predictor X columns alongside probabilities in exported probs CSVs (can be large).",
    )

    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""

    trans_dir = Path(args.transitions_dir) / run_tag
    if not trans_dir.exists():
        raise FileNotFoundError(f"Transitions folder not found: {trans_dir}")

    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in trans_dir.glob("S*_S*.csv") if infer_transition_from_filename(p) is not None])
    if not files:
        raise FileNotFoundError(f"No transition CSV files found in: {trans_dir}")

    class_weight = None if args.class_weight == "none" else "balanced"

    print(f"[OK] Run tag: {run_tag}")
    print(f"[OK] Using transitions: {trans_dir}")
    print(f"[OK] Found {len(files)} transition files")
    print(f"[OK] OOF folds={args.cv_folds} | seed={args.seed} | class_weight={class_weight}")
    if args.test_size != 0.20:
        print("[WARN] --test-size is deprecated and ignored by the OOF LR pipeline")

    metrics_rows: list[dict] = []

    for fp in files:
        tt = infer_transition_from_filename(fp)
        if tt is None:
            continue
        t, tp1 = tt
        trans_name = f"S{t}_S{tp1}"

        print(f"\n=== LR OOF: {trans_name} ===")
        df = pd.read_csv(fp)

        X_cols = [c for c in df.columns if c.startswith(f"S{t}")]
        if args.include_age and "Age" in df.columns and "Age" not in X_cols:
            X_cols.append("Age")

        y_cols = [f"Y_S{tp1}_{target}" for target in ALL_TARGETS if f"Y_S{tp1}_{target}" in df.columns]

        print(f"Rows: {len(df)} | X: {len(X_cols)} | Y: {len(y_cols)}")
        if not X_cols or not y_cols:
            print(f"[WARN] {trans_name}: missing X or Y columns; skipping")
            continue

        X_num_full = coerce_numeric_X(df, X_cols)

        probs_out = None
        if args.export_probs:
            if "idauniq" in df.columns:
                probs_out = pd.DataFrame({"idauniq": df["idauniq"].to_numpy()})
            else:
                probs_out = pd.DataFrame({"row_index": np.arange(len(df), dtype=int)})
            probs_out["_note"] = "OOF probabilities; rows excluded for missing X or missing y remain NaN."

            if args.probs_include_y:
                for yc in y_cols:
                    probs_out[yc] = pd.to_numeric(df[yc], errors="coerce")
            if args.probs_include_x:
                for xc in X_cols:
                    probs_out[xc] = X_num_full[xc]

        n_ok = 0

        for yc in y_cols:
            target_name = yc.replace(f"Y_S{tp1}_", "")
            target_type = "adl" if target_name in ADL_TARGETS else "disease" if target_name in DISEASE_TARGETS else "unknown"
            y_num = pd.to_numeric(df[yc], errors="coerce")
            mask = X_num_full.notna().all(axis=1) & y_num.notna()

            base_row = {
                "scenario": scenario,
                "method": method,
                "run_tag": run_tag,
                "transition": trans_name,
                "t": t,
                "tp1": tp1,
                "target": target_name,
                "target_type": target_type,
                "n_obs": int(mask.sum()),
                "prevalence": float(y_num.loc[mask].mean()) if mask.any() else np.nan,
                "auc": np.nan,
                "pr_auc": np.nan,
                "brier": np.nan,
                "n_predictors": int(len(X_cols)),
                "class_weight": class_weight if class_weight is not None else "none",
                "cv_folds": int(args.cv_folds),
                "file": str(fp),
                "status": "ok",
            }

            if not mask.any():
                base_row["status"] = "skipped_no_complete_rows"
                metrics_rows.append(base_row)
                continue

            X_t = X_num_full.loc[mask, X_cols]
            y_t = y_num.loc[mask].to_numpy()

            if not is_binary_vector(y_t):
                base_row["status"] = "skipped_non_binary_target"
                metrics_rows.append(base_row)
                continue

            y_t = y_t.astype(int)
            if np.unique(y_t).size < 2:
                base_row["status"] = "skipped_one_class"
                metrics_rows.append(base_row)
                continue

            splitter, n_splits, split_mode = choose_splitter(y_t, args.cv_folds, args.seed)
            if splitter is None:
                base_row["status"] = "skipped_not_enough_rows_to_split"
                metrics_rows.append(base_row)
                continue

            oof_pred = np.full(len(y_t), np.nan, dtype=float)
            X_arr = X_t.to_numpy(dtype=float)

            for train_idx, test_idx in splitter.split(X_arr, y_t):
                model = build_lr_pipeline(max_iter=args.max_iter, class_weight=class_weight)
                model.fit(X_arr[train_idx], y_t[train_idx])
                oof_pred[test_idx] = model.predict_proba(X_arr[test_idx])[:, 1]

            observed_mask = np.isfinite(oof_pred)
            if observed_mask.sum() < 2 or np.unique(y_t[observed_mask]).size < 2:
                base_row["status"] = "skipped_invalid_oof_predictions"
                metrics_rows.append(base_row)
                continue

            oof_pred = np.clip(oof_pred, EPS, 1.0 - EPS)

            base_row["auc"] = float(roc_auc_score(y_t[observed_mask], oof_pred[observed_mask]))
            base_row["pr_auc"] = float(average_precision_score(y_t[observed_mask], oof_pred[observed_mask]))
            base_row["brier"] = float(brier_score_loss(y_t[observed_mask], oof_pred[observed_mask]))
            base_row["cv_folds"] = int(n_splits)
            base_row["cv_strategy"] = split_mode
            metrics_rows.append(base_row)

            if probs_out is not None:
                pcol = f"P_S{tp1}_{target_name}"
                if pcol not in probs_out.columns:
                    probs_out[pcol] = np.nan
                probs_out.loc[X_t.index.to_numpy(), pcol] = oof_pred

            n_ok += 1

        print(f"[OK] {trans_name}: targets scored: {n_ok}")

        if probs_out is not None:
            probs_path = out_dir / f"probs_{trans_name}.csv"
            probs_out.to_csv(probs_path, index=False)
            print(f"[OK] Wrote OOF probabilities: {probs_path.name} | rows: {len(probs_out):,}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = out_dir / "metrics_by_target.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n[OK] Saved metrics: {metrics_csv}")

    if not metrics_df.empty:
        summary = (
            metrics_df.groupby(["run_tag", "scenario", "method", "transition", "target_type"], as_index=False)
            .agg(
                n_targets=("target", "count"),
                n_ok=("status", lambda s: int((s == "ok").sum())),
                mean_auc=("auc", "mean"),
                median_auc=("auc", "median"),
                mean_pr_auc=("pr_auc", "mean"),
                mean_brier=("brier", "mean"),
            )
            .sort_values(["run_tag", "scenario", "method", "transition", "target_type"])
        )
        summary_csv = out_dir / "metrics_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"[OK] Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
