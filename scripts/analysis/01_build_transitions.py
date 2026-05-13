#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ADL_TARGETS = [
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA","TOILTA","MEALSA","MAPA",
    "BEDA","DIMEA","SHOPA","BATHA","ARMSA","DRESSA","WALK100A","SITA","CLIM1A","HOUSEWKA",
    "PUSHA","LIFTA","CHAIRA","CLIMSA","STOOPA"
]

DISEASE_TARGETS = [
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE","LUNGE","CANCRE",
    "ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE","ASTHMAE","CATRACTE","HCHOLE","ARTHRE","HIBPE"
]


def infer_wave_num_from_filename(name: str) -> int | None:
    """
    Infer wave number from filenames like:
      w1_...   -> 1
      w09_...  -> 9
      w111_... -> 1
      w222_... -> 2
      ...
      w999_... -> 9
    """
    n = name.lower().strip()
    m = re.match(r"^w(\d+)", n)
    if not m:
        return None

    digits = m.group(1)  # e.g. "111", "222", "1", "09"

    # MNAR-style repeats: 111/222/.../999  -> 1/2/.../9
    if len(digits) >= 3 and digits == digits[0] * len(digits):
        return int(digits[0])

    # also tolerate rare 11/22 -> 1/2
    if len(digits) == 2 and digits[0] == digits[1]:
        return int(digits[0])

    # normal case
    try:
        wave = int(digits)
        return wave
    except ValueError:
        return None


def pick_id_col(cols: list[str]) -> str:
    for cand in ["idauniq", "IDAUNIQ"]:
        if cand in cols:
            return cand
    for c in cols:
        if "idauniq" in c.lower():
            return c
    raise ValueError("Could not find ID column (expected something like 'idauniq').")


def main():
    ap = argparse.ArgumentParser(description="Build S_t -> S_{t+1} transition datasets (one file per transition).")
    ap.add_argument("--scenario", default="MNAR", choices=["MNAR", "MAR"])
    ap.add_argument("--method", default="Cart", choices=["Cart", "Pmm"])
    ap.add_argument("--data-root", default="data/Imputed_data/Core/Imputed")
    ap.add_argument("--out-dir", default="Transition_data/01_transitions")
    ap.add_argument("--targets", default="all", choices=["all", "adl", "disease"])
    args = ap.parse_args()

    data_root = Path(args.data_root) / args.scenario / args.method
    if not data_root.exists():
        raise FileNotFoundError(f"Data folder not found: {data_root}")

    out_root = Path(args.out_dir) / f"{args.scenario}_{args.method}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.targets == "adl":
        targets = ADL_TARGETS
    elif args.targets == "disease":
        targets = DISEASE_TARGETS
    else:
        targets = ADL_TARGETS + DISEASE_TARGETS

    files = sorted(data_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_root}")

    wave_to_path: dict[int, Path] = {}
    for fp in files:
        w = infer_wave_num_from_filename(fp.name)
        if w is not None:
            wave_to_path[w] = fp

    available_waves = sorted(wave_to_path.keys())
    print(f"[OK] Using data: {data_root}")
    print(f"[OK] Waves found: {available_waves}")

    manifest_rows = []

    for t in available_waves:
        tp1 = t + 1
        if tp1 not in wave_to_path:
            continue

        fp_t = wave_to_path[t]
        fp_tp1 = wave_to_path[tp1]

        print(f"\n=== Transition S{t} -> S{tp1} ===")
        df_t = pd.read_csv(fp_t)
        df_tp1 = pd.read_csv(fp_tp1)

        id_col = pick_id_col(list(df_t.columns))
        id2 = id_col if id_col in df_tp1.columns else pick_id_col(list(df_tp1.columns))

        prefix_t = f"S{t}"
        X_cols = [c for c in df_t.columns if c.startswith(prefix_t)]
        if not X_cols:
            raise ValueError(f"No predictor columns found with prefix '{prefix_t}' in {fp_t.name}")

        prefix_tp1 = f"S{tp1}"
        y_cols = []
        y_rename = {}

        for targ in targets:
            col = f"{prefix_tp1}{targ}"
            if col in df_tp1.columns:
                y_cols.append(col)
                y_rename[col] = f"Y_{prefix_tp1}_{targ}"  # e.g., Y_S2_DIABE

        if not y_cols:
            print(f"[WARN] No target columns found for S{tp1} in {fp_tp1.name}. Skipping this transition.")
            continue

        dfX = df_t[[id_col] + X_cols].copy()
        dfY = df_tp1[[id2] + y_cols].copy()

        dfMerged = dfX.merge(dfY, left_on=id_col, right_on=id2, how="inner")
        if id2 != id_col:
            dfMerged = dfMerged.drop(columns=[id2])

        dfMerged = dfMerged.rename(columns=y_rename)

        out_fp = out_root / f"S{t}_S{tp1}.csv"
        dfMerged.to_csv(out_fp, index=False)

        manifest_rows.append({
            "scenario": args.scenario,
            "method": args.method,
            "t": t,
            "tp1": tp1,
            "file": str(out_fp),
            "n_rows": int(dfMerged.shape[0]),
            "n_predictors": int(len(X_cols)),
            "n_targets": int(len(y_cols)),
            "id_col": id_col
        })

        print(f"[OK] Wrote transition file: {out_fp.name} | rows={dfMerged.shape[0]:,} | X={len(X_cols):,} | Y={len(y_cols):,}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_fp = out_root / "manifest_transitions.csv"
    manifest.to_csv(manifest_fp, index=False)
    print(f"\n[OK] Manifest: {manifest_fp}")
    print(f"[OK] Total transition files: {len(manifest_rows)}")


if __name__ == "__main__":
    main()