#!/usr/bin/env python3
"""Build decision-curve analysis tables for standard or derived event predictions.

This script supports two modes:

1. standard
   Reads saved OOF probability files from analysis/{lr,dnn}/{RUN_TAG}/probs_S*_S*.csv
   and the corresponding transition files to evaluate next-wave prediction utility.

2. derived
   Reads long-format event tables from analysis/.../derived/damage_repair/{MODEL}/{RUN_TAG}/events_all.csv
   and evaluates either damage or repair decision utility.

Outputs are written under:
  <out-root>/<RUN_TAG>/<analysis_kind>/
    curves_by_pair.csv
    pooled_curves.csv
    curves_by_transition.csv
    summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
EPS = 1e-9


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def infer_transition_from_filename(fp: Path) -> tuple[int, int] | None:
    match = re.match(r"^S(\d+)_S(\d+)\.csv$", fp.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def normalize_id_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").astype("Int64")
        return values.astype(str)
    return series.astype(str).str.extract(r"(\d+)", expand=False)


def build_thresholds(start: float, stop: float, step: float) -> np.ndarray:
    if start <= 0 or stop >= 1 or step <= 0 or start >= stop:
        raise ValueError("Thresholds must satisfy 0 < start < stop < 1 and step > 0.")
    thresholds = np.arange(start, stop + step / 2.0, step, dtype=float)
    return thresholds[(thresholds > 0) & (thresholds < 1)]


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    n = len(y)
    if n == 0:
        return np.nan
    pred_pos = p >= threshold
    tp = np.sum(pred_pos & (y == 1))
    fp = np.sum(pred_pos & (y == 0))
    odds = threshold / max(1.0 - threshold, EPS)
    return float(tp / n - fp / n * odds)


def treat_all_net_benefit(prevalence: float, threshold: float) -> float:
    odds = threshold / max(1.0 - threshold, EPS)
    return float(prevalence - (1.0 - prevalence) * odds)


def contiguous_range_string(thresholds: np.ndarray, mask: np.ndarray) -> str:
    if thresholds.size == 0 or mask.size == 0 or not mask.any():
        return ""
    chosen = thresholds[mask]
    return f"{chosen.min():.2f}-{chosen.max():.2f}"


def aggregate_curves(curves: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = curves.groupby(group_cols, as_index=False).agg(
        mean_net_benefit=("net_benefit", "mean"),
        sd_net_benefit=("net_benefit", "std"),
        n_pairs=("pair_id", "nunique"),
        mean_prevalence=("prevalence", "mean"),
        mean_n_obs=("n_obs", "mean"),
    )
    grouped["sd_net_benefit"] = grouped["sd_net_benefit"].fillna(0.0)
    grouped["se_net_benefit"] = grouped["sd_net_benefit"] / np.sqrt(grouped["n_pairs"].clip(lower=1))
    return grouped


def summarize_model_curves(pooled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target_type in sorted(pooled["target_type"].dropna().unique()):
        sub = pooled[pooled["target_type"] == target_type].copy()
        treat_all = sub[sub["strategy"] == "Treat all"].sort_values("threshold")
        treat_none = sub[sub["strategy"] == "Treat none"].sort_values("threshold")

        if treat_all.empty or treat_none.empty:
            continue

        ref_thresholds = treat_all["threshold"].to_numpy(dtype=float)
        ref_none = treat_none["mean_net_benefit"].to_numpy(dtype=float)
        ref_all = treat_all["mean_net_benefit"].to_numpy(dtype=float)
        default_ref = np.maximum(ref_all, ref_none)

        for model in ("LR", "DNN"):
            model_curve = sub[sub["strategy"] == model].sort_values("threshold")
            peer_curve = sub[sub["strategy"] == ("DNN" if model == "LR" else "LR")].sort_values("threshold")
            if model_curve.empty:
                continue

            thresholds = model_curve["threshold"].to_numpy(dtype=float)
            nb = model_curve["mean_net_benefit"].to_numpy(dtype=float)

            if not np.array_equal(thresholds, ref_thresholds):
                raise ValueError(f"Threshold mismatch while summarizing {target_type} {model}")

            peer_nb = peer_curve["mean_net_benefit"].to_numpy(dtype=float) if not peer_curve.empty else np.full_like(nb, np.nan)

            mean_nb = float(np.trapezoid(nb, thresholds) / (thresholds.max() - thresholds.min()))
            delta_default = nb - default_ref
            delta_peer = nb - peer_nb

            best_idx = int(np.nanargmax(nb))
            better_default = nb > default_ref
            better_peer = nb > peer_nb if np.isfinite(peer_nb).all() else np.zeros_like(nb, dtype=bool)

            rows.append(
                {
                    "target_type": target_type,
                    "model": model,
                    "n_pairs": int(model_curve["n_pairs"].iloc[0]),
                    "mean_prevalence": float(model_curve["mean_prevalence"].mean()),
                    "threshold_start": float(thresholds.min()),
                    "threshold_stop": float(thresholds.max()),
                    "integrated_net_benefit": mean_nb,
                    "max_net_benefit": float(nb[best_idx]),
                    "best_threshold": float(thresholds[best_idx]),
                    "integrated_delta_vs_default": float(np.trapezoid(delta_default, thresholds) / (thresholds.max() - thresholds.min())),
                    "integrated_delta_vs_peer": float(np.trapezoid(delta_peer, thresholds) / (thresholds.max() - thresholds.min())) if np.isfinite(delta_peer).all() else np.nan,
                    "share_thresholds_better_than_default": float(better_default.mean()),
                    "share_thresholds_better_than_peer": float(better_peer.mean()) if np.isfinite(peer_nb).all() else np.nan,
                    "beneficial_threshold_range": contiguous_range_string(thresholds, better_default),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_type", "model"]).reset_index(drop=True)


def build_standard_curves(
    dataset_label: str,
    run_tag: str,
    transitions_dir: Path,
    analysis_root: Path,
    thresholds: np.ndarray,
    min_n: int,
) -> pd.DataFrame:
    trans_dir = transitions_dir / run_tag
    if not trans_dir.exists():
        raise FileNotFoundError(f"Missing transitions directory: {trans_dir}")

    rows: list[dict] = []

    for model in ("LR", "DNN"):
        probs_dir = analysis_root / model.lower() / run_tag
        if not probs_dir.exists():
            raise FileNotFoundError(f"Missing probability directory: {probs_dir}")

        for prob_fp in sorted(probs_dir.glob("probs_S*_S*.csv")):
            match = re.match(r"^probs_(S\d+_S\d+)\.csv$", prob_fp.name)
            if not match:
                continue
            transition = match.group(1)
            trans_fp = trans_dir / f"{transition}.csv"
            if not trans_fp.exists():
                continue

            dft = pd.read_csv(trans_fp)
            dfp = pd.read_csv(prob_fp)

            if "idauniq" in dft.columns and "idauniq" in dfp.columns:
                dft["idauniq"] = normalize_id_series(dft["idauniq"])
                dfp["idauniq"] = normalize_id_series(dfp["idauniq"])
                df = dft.merge(dfp, on="idauniq", how="inner")
            elif "row_index" in dft.columns and "row_index" in dfp.columns:
                df = dft.merge(dfp, on="row_index", how="inner")
            else:
                raise KeyError(f"{transition}: could not find a common identifier between {trans_fp} and {prob_fp}")

            tp1 = transition.split("_")[1]
            for target in ALL_TARGETS:
                ycol = f"Y_{tp1}_{target}"
                pcol = f"P_{tp1}_{target}"
                if ycol not in df.columns or pcol not in df.columns:
                    continue

                y = pd.to_numeric(df[ycol], errors="coerce")
                p = pd.to_numeric(df[pcol], errors="coerce")
                mask = y.notna() & p.notna()
                if int(mask.sum()) < min_n:
                    continue

                yv = y.loc[mask].to_numpy(dtype=int)
                pv = np.clip(p.loc[mask].to_numpy(dtype=float), EPS, 1.0 - EPS)
                if np.unique(yv).size < 2:
                    continue

                target_type = "ADL" if target in ADL_TARGETS else "Disease"
                prevalence = float(yv.mean())
                pair_id = f"{transition}::{target}"

                for threshold in thresholds:
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "run_tag": run_tag,
                            "analysis_kind": "standard",
                            "event_type": "standard",
                            "transition": transition,
                            "target": target,
                            "target_type": target_type,
                            "model": model,
                            "strategy": model,
                            "threshold": threshold,
                            "net_benefit": net_benefit(yv, pv, threshold),
                            "n_obs": len(yv),
                            "prevalence": prevalence,
                            "pair_id": pair_id,
                        }
                    )
                    if model == "LR":
                        rows.append(
                            {
                                "dataset": dataset_label,
                                "run_tag": run_tag,
                                "analysis_kind": "standard",
                                "event_type": "standard",
                                "transition": transition,
                                "target": target,
                                "target_type": target_type,
                                "model": "",
                                "strategy": "Treat all",
                                "threshold": threshold,
                                "net_benefit": treat_all_net_benefit(prevalence, threshold),
                                "n_obs": len(yv),
                                "prevalence": prevalence,
                                "pair_id": pair_id,
                            }
                        )
                        rows.append(
                            {
                                "dataset": dataset_label,
                                "run_tag": run_tag,
                                "analysis_kind": "standard",
                                "event_type": "standard",
                                "transition": transition,
                                "target": target,
                                "target_type": target_type,
                                "model": "",
                                "strategy": "Treat none",
                                "threshold": threshold,
                                "net_benefit": 0.0,
                                "n_obs": len(yv),
                                "prevalence": prevalence,
                                "pair_id": pair_id,
                            }
                        )

    if not rows:
        raise ValueError(f"No standard decision-curve rows produced for {dataset_label} {run_tag}")
    return pd.DataFrame(rows)


def build_derived_curves(
    dataset_label: str,
    run_tag: str,
    event_type: str,
    events_root: Path,
    thresholds: np.ndarray,
    min_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for model in ("LR", "DNN"):
        events_fp = events_root / model / run_tag / "events_all.csv"
        if not events_fp.exists():
            raise FileNotFoundError(f"Missing derived event table: {events_fp}")

        df = pd.read_csv(events_fp)
        df = df[df["event_type"] == event_type].copy()

        for (transition, target, target_type), block in df.groupby(["transition", "target", "target_type"], dropna=False):
            d = block[["p", "y"]].dropna().copy()
            d["p"] = pd.to_numeric(d["p"], errors="coerce")
            d["y"] = pd.to_numeric(d["y"], errors="coerce")
            d = d.dropna()
            d = d[d["y"].isin([0, 1])]
            if len(d) < min_n:
                continue

            yv = d["y"].to_numpy(dtype=int)
            pv = np.clip(d["p"].to_numpy(dtype=float), EPS, 1.0 - EPS)
            if np.unique(yv).size < 2:
                continue

            pair_id = f"{transition}::{target}"
            target_type_fmt = "ADL" if str(target_type).lower() == "adl" else "Disease"
            prevalence = float(yv.mean())

            for threshold in thresholds:
                rows.append(
                    {
                        "dataset": dataset_label,
                        "run_tag": run_tag,
                        "analysis_kind": event_type,
                        "event_type": event_type,
                        "transition": transition,
                        "target": target,
                        "target_type": target_type_fmt,
                        "model": model,
                        "strategy": model,
                        "threshold": threshold,
                        "net_benefit": net_benefit(yv, pv, threshold),
                        "n_obs": len(yv),
                        "prevalence": prevalence,
                        "pair_id": pair_id,
                    }
                )
                if model == "LR":
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "run_tag": run_tag,
                            "analysis_kind": event_type,
                            "event_type": event_type,
                            "transition": transition,
                            "target": target,
                            "target_type": target_type_fmt,
                            "model": "",
                            "strategy": "Treat all",
                            "threshold": threshold,
                            "net_benefit": treat_all_net_benefit(prevalence, threshold),
                            "n_obs": len(yv),
                            "prevalence": prevalence,
                            "pair_id": pair_id,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "run_tag": run_tag,
                            "analysis_kind": event_type,
                            "event_type": event_type,
                            "transition": transition,
                            "target": target,
                            "target_type": target_type_fmt,
                            "model": "",
                            "strategy": "Treat none",
                            "threshold": threshold,
                            "net_benefit": 0.0,
                            "n_obs": len(yv),
                            "prevalence": prevalence,
                            "pair_id": pair_id,
                        }
                    )

    if not rows:
        raise ValueError(f"No derived decision-curve rows produced for {dataset_label} {run_tag} {event_type}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build decision-curve analysis tables.")
    ap.add_argument("--dataset-label", required=True, help="Display label, e.g. 'ELSA full' or 'HRS shared'")
    ap.add_argument("--scenario", default=None, help="ELSA-style scenario when not using --run-tag")
    ap.add_argument("--method", default=None, help="ELSA-style method when not using --run-tag")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. MNAR_Pmm or HRS_SHARED")

    ap.add_argument("--analysis-kind", required=True, choices=["standard", "damage", "repair"])
    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--analysis-root", default="analysis/ELSA")
    ap.add_argument("--events-root", default="analysis/ELSA/derived/damage_repair")
    ap.add_argument("--out-root", default="analysis/ELSA/metrics/decision_curve")

    ap.add_argument("--threshold-start", type=float, default=None)
    ap.add_argument("--threshold-stop", type=float, default=None)
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    if args.threshold_start is None:
        args.threshold_start = 0.01
    if args.threshold_stop is None:
        args.threshold_stop = 0.30 if args.analysis_kind == "damage" else 0.50
    thresholds = build_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    if args.analysis_kind == "standard":
        curves = build_standard_curves(
            dataset_label=args.dataset_label,
            run_tag=run_tag,
            transitions_dir=Path(args.transitions_dir),
            analysis_root=Path(args.analysis_root),
            thresholds=thresholds,
            min_n=args.min_n,
        )
    else:
        curves = build_derived_curves(
            dataset_label=args.dataset_label,
            run_tag=run_tag,
            event_type=args.analysis_kind,
            events_root=Path(args.events_root),
            thresholds=thresholds,
            min_n=args.min_n,
        )

    out_dir = Path(args.out_root) / run_tag / args.analysis_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = curves.sort_values(["target_type", "strategy", "transition", "target", "threshold"]).reset_index(drop=True)
    pooled = aggregate_curves(curves, ["dataset", "run_tag", "analysis_kind", "event_type", "target_type", "strategy", "threshold"])
    by_transition = aggregate_curves(curves, ["dataset", "run_tag", "analysis_kind", "event_type", "transition", "target_type", "strategy", "threshold"])
    summary = summarize_model_curves(pooled)
    summary.insert(0, "dataset", args.dataset_label)
    summary.insert(1, "run_tag", run_tag)
    summary.insert(2, "analysis_kind", args.analysis_kind)

    curves.to_csv(out_dir / "curves_by_pair.csv", index=False)
    pooled.to_csv(out_dir / "pooled_curves.csv", index=False)
    by_transition.to_csv(out_dir / "curves_by_transition.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"[OK] Dataset:        {args.dataset_label}")
    print(f"[OK] Run tag:        {run_tag}")
    print(f"[OK] Analysis kind:  {args.analysis_kind}")
    print(f"[OK] Thresholds:     {thresholds.min():.2f} to {thresholds.max():.2f} by {args.threshold_step:.2f}")
    print(f"[OK] Wrote {out_dir / 'curves_by_pair.csv'}")
    print(f"[OK] Wrote {out_dir / 'pooled_curves.csv'}")
    print(f"[OK] Wrote {out_dir / 'curves_by_transition.csv'}")
    print(f"[OK] Wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
