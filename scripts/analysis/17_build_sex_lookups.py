#!/usr/bin/env python3
"""Build canonical sex lookup tables for ELSA and HRS."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ELSA_MAP = {1: "male", 2: "female"}
HRS_MAP = {1: "male", 2: "female"}


def normalize_id_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    return s.astype(str).str.extract(r"(\d+)", expand=False)


def resolve_mode(values: list[int]) -> tuple[int | None, bool]:
    clean = [int(v) for v in values if pd.notna(v)]
    if not clean:
        return None, False
    counts = Counter(clean)
    top_n = max(counts.values())
    top_vals = sorted(v for v, n in counts.items() if n == top_n)
    return top_vals[0], len(counts) > 1


def build_elsa_lookup(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for wave in range(2, 10):
        fp = raw_dir / f"wave_{wave}_core.sav"
        if not fp.exists():
            continue
        df = pd.read_spss(fp, convert_categoricals=False, usecols=["idauniq", "indsex"])
        df["idauniq"] = normalize_id_series(df["idauniq"])
        df["indsex"] = pd.to_numeric(df["indsex"], errors="coerce")
        df = df[df["indsex"].isin([1, 2])].copy()
        df["wave"] = wave
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No ELSA raw wave files with indsex found under {raw_dir}")

    df_all = pd.concat(rows, ignore_index=True)
    grouped = []
    for idauniq, block in df_all.groupby("idauniq", dropna=False):
        sex_code, had_conflict = resolve_mode(block["indsex"].tolist())
        grouped.append(
            {
                "idauniq": idauniq,
                "sex_code": sex_code,
                "sex": ELSA_MAP.get(sex_code, np.nan),
                "n_sources": int(len(block)),
                "n_unique_codes": int(block["indsex"].nunique()),
                "had_conflict": bool(had_conflict),
            }
        )
    out = pd.DataFrame(grouped).dropna(subset=["idauniq", "sex_code"])
    out["sex_code"] = out["sex_code"].astype(int)
    return out.sort_values("idauniq").reset_index(drop=True)


def build_hrs_lookup(hrs_fp: Path) -> pd.DataFrame:
    df = pd.read_csv(hrs_fp, usecols=["id", "ragender", "sex"])
    df["idauniq"] = normalize_id_series(df["id"])
    df["ragender"] = pd.to_numeric(df["ragender"], errors="coerce")
    df["sex_clean"] = df["sex"].astype(str).str.strip().str.lower()

    grouped = []
    for idauniq, block in df.groupby("idauniq", dropna=False):
        codes = [int(v) for v in block["ragender"].dropna().tolist() if int(v) in HRS_MAP]
        sex_code, had_conflict = resolve_mode(codes)
        sex_label = HRS_MAP.get(sex_code, np.nan)
        if pd.isna(sex_label):
            labels = [v for v in block["sex_clean"].tolist() if v in {"male", "female"}]
            if labels:
                counts = Counter(labels)
                sex_label = sorted([k for k, n in counts.items() if n == max(counts.values())])[0]
                sex_code = 1 if sex_label == "male" else 2
                had_conflict = len(counts) > 1
        if pd.isna(sex_label):
            continue
        grouped.append(
            {
                "idauniq": idauniq,
                "sex_code": int(sex_code),
                "sex": sex_label,
                "n_sources": int(len(block)),
                "n_unique_codes": int(block["ragender"].dropna().nunique()),
                "had_conflict": bool(had_conflict),
            }
        )
    out = pd.DataFrame(grouped).dropna(subset=["idauniq", "sex"])
    return out.sort_values("idauniq").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elsa-raw-dir", default="data/restricted/ELSA/raw")
    ap.add_argument("--hrs-file", default="data/restricted/HRS/hrs_rand_preproc.csv")
    ap.add_argument("--out-dir", default="data/schema/harmonized")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    elsa = build_elsa_lookup(Path(args.elsa_raw_dir))
    hrs = build_hrs_lookup(Path(args.hrs_file))

    elsa_fp = out_dir / "elsa_sex_lookup.csv"
    hrs_fp = out_dir / "hrs_sex_lookup.csv"
    summary_fp = out_dir / "sex_lookup_summary.csv"

    elsa.to_csv(elsa_fp, index=False)
    hrs.to_csv(hrs_fp, index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "ELSA",
                "n_ids": int(len(elsa)),
                "n_male": int((elsa["sex"] == "male").sum()),
                "n_female": int((elsa["sex"] == "female").sum()),
                "n_conflicts": int(elsa["had_conflict"].sum()),
            },
            {
                "dataset": "HRS",
                "n_ids": int(len(hrs)),
                "n_male": int((hrs["sex"] == "male").sum()),
                "n_female": int((hrs["sex"] == "female").sum()),
                "n_conflicts": int(hrs["had_conflict"].sum()),
            },
        ]
    )
    summary.to_csv(summary_fp, index=False)

    print(f"[OK] Saved {elsa_fp}")
    print(f"[OK] Saved {hrs_fp}")
    print(f"[OK] Saved {summary_fp}")


if __name__ == "__main__":
    main()
