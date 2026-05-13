#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


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
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate damage/repair event tables.")
    ap.add_argument("--scenario", default=None, choices=["MAR", "MNAR"])
    ap.add_argument("--method", default=None, choices=["Cart", "Pmm"])
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--model", required=True, choices=["LR", "DNN"])
    ap.add_argument("--events-root", default="analysis/derived/damage_repair")
    ap.add_argument("--out-root", default="analysis/metrics/damage_repair")
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""

    events_fp = Path(args.events_root) / args.model / run_tag / "events_all.csv"
    if not events_fp.exists():
        raise FileNotFoundError(f"Missing events_all.csv: {events_fp}")

    out_dir = Path(args.out_root) / args.model.lower() / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(events_fp)
    if df.empty:
        raise ValueError(f"{events_fp} is empty")

    rows: list[dict] = []
    for (transition, target, event_type, target_type), block in df.groupby(
        ["transition", "target", "event_type", "target_type"], dropna=False
    ):
        d = block[["p", "y"]].dropna().copy()
        d["p"] = pd.to_numeric(d["p"], errors="coerce")
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
        d = d.dropna()
        d = d[d["y"].isin([0, 1])]

        row = {
            "model": args.model,
            "run_tag": run_tag,
            "scenario": scenario,
            "method": method,
            "transition": transition,
            "target": target,
            "event_type": event_type,
            "target_type": target_type,
            "n_obs": int(len(d)),
            "prevalence": float(d["y"].mean()) if len(d) else np.nan,
            "auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "ece": np.nan,
            "r2cal": np.nan,
            "status": "ok",
        }

        if len(d) < args.min_n:
            row["status"] = "skipped_too_few_rows"
            rows.append(row)
            continue

        y = d["y"].to_numpy(dtype=int)
        p = np.clip(d["p"].to_numpy(dtype=float), EPS, 1.0 - EPS)

        if np.unique(y).size < 2:
            row["status"] = "skipped_one_class"
            rows.append(row)
            continue

        row["auc"] = float(roc_auc_score(y, p))
        row["pr_auc"] = float(average_precision_score(y, p))
        row["brier"] = float(brier_score_loss(y, p))
        row["ece"] = float(expected_calibration_error(y, p, n_bins=args.calibration_bins))
        row["r2cal"] = float(r2_calibration_from_bins(calibration_table(y, p, n_bins=args.calibration_bins)))
        rows.append(row)

    metrics_by_target = pd.DataFrame(rows)
    metrics_by_target.to_csv(out_dir / "metrics_by_target.csv", index=False)

    ok = metrics_by_target[metrics_by_target["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No evaluable damage/repair targets found in {events_fp}")

    summary = (
        ok.groupby(["model", "run_tag", "scenario", "method", "event_type", "target_type"], as_index=False)
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
        .sort_values(["event_type", "target_type", "model"])
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    summary_by_transition = (
        ok.groupby(["model", "run_tag", "scenario", "method", "transition", "event_type", "target_type"], as_index=False)
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
        .sort_values(["transition", "event_type", "target_type", "model"])
    )
    summary_by_transition.to_csv(out_dir / "metrics_summary_by_transition.csv", index=False)

    print(f"[OK] Run tag: {run_tag}")
    print(f"[OK] Events:   {events_fp}")
    print(f"[OK] Out:      {out_dir}")
    print(f"[OK] Saved {out_dir / 'metrics_by_target.csv'}")
    print(f"[OK] Saved {out_dir / 'metrics_summary.csv'}")
    print(f"[OK] Saved {out_dir / 'metrics_summary_by_transition.csv'}")


if __name__ == "__main__":
    main()
