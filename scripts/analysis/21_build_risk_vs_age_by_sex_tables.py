#!/usr/bin/env python3
"""Build target-level standard next-wave risk-vs-age tables by sex.

This replaces the earlier wave-to-wave trajectory idea.

Outputs:
- target_binned_risk.csv
- by_transition_target_binned_risk.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ADL_TARGETS = {
    "DANGERA", "EATA", "MEDSA", "COMMUNA", "PHONEA", "MONEYA", "WALKRA", "TOILTA", "MEALSA", "MAPA",
    "BEDA", "DIMEA", "SHOPA", "BATHA", "ARMSA", "DRESSA", "WALK100A", "SITA", "CLIM1A", "HOUSEWKA",
    "PUSHA", "LIFTA", "CHAIRA", "CLIMSA", "STOOPA",
}
DISEASE_TARGETS = {
    "PARKINE", "CONHRTFE", "HIPE", "HEARTE", "HRTMRE", "HRTATTE", "STROKE", "LUNGE", "CANCRE",
    "ANGINE", "OSTEOE", "HRTRHME", "PSYCHE", "DIABE", "ASTHMAE", "CATRACTE", "HCHOLE", "ARTHRE", "HIBPE",
}
SEX_ORDER = ("female", "male")
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
    raise KeyError(f"No age column found for {transition}")


def binned_summary(
    df: pd.DataFrame,
    age_min: float,
    age_max: float,
    bin_width: int,
    min_n: int,
    group_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    edges = np.arange(age_min, age_max + bin_width, bin_width, dtype=float)
    rows: list[dict] = []
    for keys, block in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        age = pd.to_numeric(block["age"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(block["p"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(block["y"], errors="coerce").to_numpy(float)
        mask = np.isfinite(age) & np.isfinite(p) & np.isfinite(y)
        age = age[mask]
        p = p[mask]
        y = y[mask]
        if len(age) == 0:
            continue
        for i in range(len(edges) - 1):
            left = edges[i]
            right = edges[i + 1]
            m = (age >= left) & (age < right) if i < len(edges) - 2 else (age >= left) & (age <= right)
            n = int(m.sum())
            if n < min_n:
                continue
            rows.append(
                {
                    **base,
                    "age_left": left,
                    "age_right": right,
                    "age_mid": 0.5 * (left + right),
                    "n_obs": n,
                    "mean_predicted": float(np.mean(p[m])),
                    "observed_rate": float(np.mean(y[m])),
                    "mean_error": float(np.mean(p[m]) - np.mean(y[m])),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pooled standard risk-vs-age tables by sex.")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dataset-label", required=True)
    ap.add_argument("--transitions-dir", required=True)
    ap.add_argument("--analysis-root", required=True)
    ap.add_argument("--sex-lookup", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bin-width", type=int, default=5)
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    trans_dir = Path(args.transitions_dir) / run_tag
    analysis_root = Path(args.analysis_root)
    out_dir = Path(args.out_root) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sex_lookup = pd.read_csv(args.sex_lookup)
    sex_lookup["idauniq"] = normalize_id_series(sex_lookup["idauniq"])
    sex_lookup["sex"] = sex_lookup["sex"].astype(str).str.strip().str.lower()
    sex_lookup = sex_lookup[sex_lookup["sex"].isin(SEX_ORDER)][["idauniq", "sex"]].drop_duplicates()

    long_rows: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        probs_dir = analysis_root / model.lower() / run_tag
        for prob_fp in sorted(probs_dir.glob("probs_S*_S*.csv"), key=lambda p: transition_sort_key(p.stem.replace("probs_", ""))):
            m = re.match(r"^probs_(S\d+_S\d+)\.csv$", prob_fp.name)
            if not m:
                continue
            transition = m.group(1)
            trans_fp = trans_dir / f"{transition}.csv"
            if not trans_fp.exists():
                continue

            df_p = pd.read_csv(prob_fp)
            df_t = pd.read_csv(trans_fp)
            age_col = age_column_for_transition(df_t, transition)
            df_t = df_t[["idauniq", age_col] + [c for c in df_t.columns if c.startswith(f"Y_{transition.split('_')[1]}_")]].copy()

            df_p["idauniq"] = normalize_id_series(df_p["idauniq"])
            df_t["idauniq"] = normalize_id_series(df_t["idauniq"])
            df_t = df_t.merge(sex_lookup, on="idauniq", how="inner")
            merged = df_t.merge(df_p, on="idauniq", how="inner")
            merged["age"] = pd.to_numeric(merged[age_col], errors="coerce")

            tp1 = transition.split("_")[1]
            prob_cols = [c for c in merged.columns if c.startswith(f"P_{tp1}_")]
            for pcol in prob_cols:
                target = pcol.replace(f"P_{tp1}_", "")
                ycol = f"Y_{tp1}_{target}"
                if ycol not in merged.columns:
                    continue
                if target in ADL_TARGETS:
                    target_type = "adl"
                elif target in DISEASE_TARGETS:
                    target_type = "disease"
                else:
                    continue

                block = merged[["idauniq", "sex", "age", ycol, pcol]].copy()
                block = block.rename(columns={ycol: "y", pcol: "p"})
                block["dataset"] = args.dataset_label
                block["run_tag"] = run_tag
                block["transition"] = transition
                block["model"] = model
                block["target"] = target
                block["target_type"] = target_type
                block = block.dropna(subset=["sex", "age", "y", "p"])
                long_rows.append(block[["dataset", "run_tag", "transition", "model", "target", "target_type", "idauniq", "sex", "age", "p", "y"]])

    if not long_rows:
        raise ValueError(f"No standard risk rows produced for {args.dataset_label} {run_tag}")

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["age"] = pd.to_numeric(long_df["age"], errors="coerce")
    long_df["p"] = pd.to_numeric(long_df["p"], errors="coerce")
    long_df["y"] = pd.to_numeric(long_df["y"], errors="coerce")
    long_df = long_df.dropna(subset=["age", "p", "y", "sex", "target_type"])

    age_min = float(math.floor(long_df["age"].min() / args.bin_width) * args.bin_width)
    age_max = float(math.ceil(long_df["age"].max() / args.bin_width) * args.bin_width)

    pooled = binned_summary(
        df=long_df,
        age_min=age_min,
        age_max=age_max,
        bin_width=args.bin_width,
        min_n=args.min_n,
        group_cols=["dataset", "run_tag", "sex", "model", "target_type", "target"],
    )
    by_transition = binned_summary(
        df=long_df,
        age_min=age_min,
        age_max=age_max,
        bin_width=args.bin_width,
        min_n=args.min_n,
        group_cols=["dataset", "run_tag", "transition", "sex", "model", "target_type", "target"],
    )

    pooled = pooled.sort_values(["target_type", "target", "sex", "model", "age_mid"]).reset_index(drop=True)
    by_transition = by_transition.sort_values(["target_type", "target", "transition", "sex", "model", "age_mid"]).reset_index(drop=True)

    pooled.to_csv(out_dir / "target_binned_risk.csv", index=False)
    by_transition.to_csv(out_dir / "by_transition_target_binned_risk.csv", index=False)

    print(f"[OK] Saved {out_dir / 'target_binned_risk.csv'}")
    print(f"[OK] Saved {out_dir / 'by_transition_target_binned_risk.csv'}")


if __name__ == "__main__":
    main()
