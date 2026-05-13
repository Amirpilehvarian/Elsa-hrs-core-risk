#!/usr/bin/env python3
"""
Build hazard-ratio (HR) tables from OOF probability files + transition truth files.

HR definition (paper style):
  HR = (observed event rate in top predicted decile) / (overall observed event rate)
Computed per (transition, target), then averaged over transitions per target.

Inputs:
- probs: analysis/<lr|dnn>/<SCENARIO>_<METHOD>/probs_S*_S*.csv
  wide: idauniq + columns like P_S2_HIBPE, P_S2_EATA, ...
- manifest: Transition_data/01_transitions/<SCENARIO>_<METHOD>/manifest_transitions.csv
  must contain 'file' and either:
    - 'transition' column, OR
    - 't' and 'tp1' columns (waves), e.g. 1 and 2

Outputs:
analysis/metrics/hazard_ratio/<MODEL>/<SCENARIO>_<METHOD>/
- HR_by_transition.csv
- HR_by_target.csv
- HR_ranked_Disease.csv
- HR_ranked_ADL.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ADL_TARGETS = [
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA","TOILTA",
    "MEALSA","MAPA","BEDA","DIMEA","SHOPA","BATHA","ARMSA","DRESSA",
    "WALK100A","SITA","CLIM1A","HOUSEWKA","PUSHA","LIFTA","CHAIRA","CLIMSA","STOOPA"
]
DISEASE_TARGETS = [
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE","LUNGE",
    "CANCRE","ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE","ASTHMAE","CATRACTE",
    "HCHOLE","ARTHRE","HIBPE"
]

ADL_SET = set(ADL_TARGETS)
DIS_SET = set(DISEASE_TARGETS)
CANON_SET = ADL_SET | DIS_SET


def model_dir(model: str) -> str:
    m = model.strip().upper()
    return {"LR": "lr", "DNN": "dnn"}[m]


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def parse_transition_from_probs_filename(fp: Path) -> str:
    m = re.search(r"S\d+_S\d+", fp.stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse transition from filename: {fp.name}")
    return m.group(0).upper()


def to_wave_from_transition(tr: str) -> int:
    # "S1_S2" -> 2
    return int(tr.split("_")[1].replace("S", ""))


def normalize_target_from_probcol(col: str) -> str:
    # P_S2_HIBPE -> HIBPE
    s = str(col).strip()
    s = re.sub(r"^(P_)?S\d{1,2}_", "", s, flags=re.IGNORECASE)
    return s.upper()


def build_transition_col(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in manifest.columns}

    if "file" not in cols:
        raise ValueError(f"manifest missing 'file' column. Columns={list(manifest.columns)}")

    # standardize file column
    manifest = manifest.rename(columns={cols["file"]: "file"}).copy()

    if "transition" in cols:
        manifest = manifest.rename(columns={cols["transition"]: "transition"})
        manifest["transition"] = manifest["transition"].astype(str).str.upper()
        return manifest

    if "t" in cols and "tp1" in cols:
        tcol, tp1col = cols["t"], cols["tp1"]

        def _to_int(x):
            s = str(x).strip().upper().replace("S", "")
            return int(float(s))

        manifest["transition"] = "S" + manifest[tcol].map(_to_int).astype(str) + "_S" + manifest[tp1col].map(_to_int).astype(str)
        return manifest

    raise ValueError(
        f"manifest must have either ('transition','file') or ('t','tp1','file'). Columns={list(manifest.columns)}"
    )


def observed_hr(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)

    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return np.nan

    base = float(np.mean(y))
    if base <= 0:
        return np.nan

    q90 = np.quantile(p, 0.9)
    top = y[p >= q90]
    if top.size == 0:
        return np.nan

    return float(np.mean(top) / base)


def load_truth(manifest: pd.DataFrame, tr: str, targets: list[str]) -> pd.DataFrame:
    """
    Return truth dataframe with columns: idauniq + each target as 0/1.

    Supports naming styles:
      - Y_S2_EATA   (your case)
      - Y_S2EATA
      - S2_EATA
      - S2EATA
    """
    row = manifest.loc[manifest["transition"].str.upper() == tr.upper()]
    if row.empty:
        raise FileNotFoundError(f"Transition {tr} not found in manifest")

    fp = Path(row.iloc[0]["file"]).expanduser()
    df = pd.read_csv(fp)

    to_w = to_wave_from_transition(tr)  # e.g. 2 for S1_S2

    found_cols = []
    rename_map = {}

    # build uppercase lookup for case-insensitive match
    cols_upper = {c.upper(): c for c in df.columns}

    for t in targets:
        candidates = [
            f"Y_S{to_w}_{t}",   # <-- your main truth naming
            f"Y_S{to_w}{t}",
            f"S{to_w}_{t}",
            f"S{to_w}{t}",
        ]

        real_col = None
        for cand in candidates:
            if cand in df.columns:
                real_col = cand
                break
            if cand.upper() in cols_upper:
                real_col = cols_upper[cand.upper()]
                break

        if real_col is not None:
            found_cols.append(real_col)
            rename_map[real_col] = t

    out = df[["idauniq"] + found_cols].copy()
    out = out.rename(columns=rename_map)
    return out


def normalize_id(s: pd.Series) -> pd.Series:
    # Convert 100035.0 / "100035.0" / "100035" -> "100035"
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").astype("Int64")
        return x.astype(str)
    tmp = s.astype(str).str.extract(r"(\d+)", expand=False)
    return tmp


def clean_binary_series(s: pd.Series) -> np.ndarray:
    """Coerce a pandas Series to numeric binary (0/1).

    Handles cases where values are strings like '0 valid answer' by extracting
    the first number found. Treats any positive numeric value as 1.
    Missing/unparseable values become 0.
    """
    if s is None:
        return np.array([], dtype=float)

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
    else:
        # Extract the first numeric token from strings like '0 valid answer'
        tmp = s.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        x = pd.to_numeric(tmp, errors="coerce")

    x = x.fillna(0)
    return (x.to_numpy(float) > 0).astype(float)

def normalize_id(s: pd.Series) -> pd.Series:
    # Convert 100035.0 / "100035.0" / "100035" -> "100035"
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").astype("Int64")
        return x.astype(str)
    tmp = s.astype(str).str.extract(r"(\d+)", expand=False)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--models", nargs="+", default=["LR", "DNN"])
    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--analysis-root", default="analysis")
    ap.add_argument("--out-root", default="analysis/metrics/hazard_ratio")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""

    transitions_root = Path(args.transitions_dir)
    analysis_root = Path(args.analysis_root)
    out_root = Path(args.out_root)

    manifest_path = transitions_root / run_tag / "manifest_transitions.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = build_transition_col(manifest)
    manifest = manifest[["transition", "file"]].copy()

    for model in args.models:
        model_u = model.strip().upper()
        run_dir = analysis_root / model_dir(model_u) / run_tag
        if not run_dir.exists():
            print(f"[SKIP] Missing probs dir: {run_dir}")
            continue

        prob_files = sorted(run_dir.glob("probs_S*_S*.csv"))
        if not prob_files:
            print(f"[SKIP] No probs files in: {run_dir}")
            continue

        out_dir = out_root / model_u / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        for pf in prob_files:
            tr = parse_transition_from_probs_filename(pf)
            dfp = pd.read_csv(pf)

            if "idauniq" not in dfp.columns:
                print(f"[SKIP] {pf.name}: missing idauniq")
                continue

            prob_cols = [c for c in dfp.columns if normalize_target_from_probcol(c) in CANON_SET]
            if not prob_cols:
                print(f"[SKIP] {pf.name}: no canonical prob columns")
                continue

            truth = load_truth(manifest, tr, targets=list(CANON_SET))

            dfp["idauniq"] = normalize_id(dfp["idauniq"])
            truth["idauniq"] = normalize_id(truth["idauniq"])
            df = dfp[["idauniq"] + prob_cols].merge(truth, on="idauniq", how="inner")
            
            if df.empty:
                print(f"[SKIP] {pf.name}: merge produced 0 rows")
                continue

            # Debug hint: if HR comes out empty, targets may be non-numeric strings in truth files.

            # compute HR per target
            for pc in prob_cols:
                tgt = normalize_target_from_probcol(pc)
                if tgt not in df.columns:
                    continue

                p = pd.to_numeric(df[pc], errors="coerce").to_numpy(float)
                y = clean_binary_series(df[tgt])

                hr = observed_hr(p, y)
                if not np.isfinite(hr):
                    continue

                rows.append({
                    "model": model_u,
                    "run_tag": run_tag,
                    "scenario": scenario,
                    "method": method,
                    "transition": tr,
                    "target": tgt,
                    "type": "Disease" if tgt in DIS_SET else ("ADL" if tgt in ADL_SET else "Other"),
                    "HR": float(hr),
                    "n": int(np.isfinite(p).sum()),
                    "base_rate": float(np.mean(y)),
                })

        hr_by_tr = pd.DataFrame(rows)
        if hr_by_tr.empty:
            print(f"[WARN] No HR computed for {model_u} {run_tag}")
            continue

        hr_by_tr.to_csv(out_dir / "HR_by_transition.csv", index=False)

        # average over transitions per target
        hr_by_tgt = (
            hr_by_tr.groupby(["run_tag", "target", "type"], as_index=False)
            .agg(
                HR=("HR", "mean"),
                HR_sd=("HR", "std"),
                n_transitions=("transition", "nunique"),
                n_total=("n", "sum"),
                base_rate_mean=("base_rate", "mean"),
            )
        )
        hr_by_tgt.to_csv(out_dir / "HR_by_target.csv", index=False)

        # ranked disease/adl tables for plotting
        disease_ranked = hr_by_tgt[hr_by_tgt["type"] == "Disease"].sort_values("HR").reset_index(drop=True)
        adl_ranked     = hr_by_tgt[hr_by_tgt["type"] == "ADL"].sort_values("HR").reset_index(drop=True)

        disease_ranked.to_csv(out_dir / "HR_ranked_Disease.csv", index=False)
        adl_ranked.to_csv(out_dir / "HR_ranked_ADL.csv", index=False)

        print(f"[OK] Wrote HR tables -> {out_dir}")

if __name__ == "__main__":
    main()
