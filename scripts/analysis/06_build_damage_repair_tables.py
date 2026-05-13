#!/usr/bin/env python3
"""
scripts/analysis/06_build_damage_repair_tables.py

Build tidy (long) tables for DAMAGE and REPAIR events using:
- Predicted probabilities from analysis/{lr|dnn}/{RUN_TAG}/probs_S*_S*.csv
- Observed transitions from Transition_data/.../{RUN_TAG}/S*_S*.csv

Definitions for target X:
- Damage:   Xt=0 and Xt+1=1  (event=1)
  Use p_damage = P(Xt+1=1 | features), restricted to Xt=0
- Repair:   Xt=1 and Xt+1=0  (event=1)
  Use p_repair = 1 - P(Xt+1=1 | features), restricted to Xt=1

Output:
analysis/derived/damage_repair/{MODEL}/{RUN_TAG}/
  events_{TRANSITION}.csv   (long format)
  events_all.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---- canonical target lists (paper) ----
ADL_TARGETS = [
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA","TOILTA","MEALSA","MAPA",
    "BEDA","DIMEA","SHOPA","BATHA","ARMSA","DRESSA","WALK100A","SITA","CLIM1A","HOUSEWKA",
    "PUSHA","LIFTA","CHAIRA","CLIMSA","STOOPA"
]
DISEASE_TARGETS = [
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE","LUNGE","CANCRE",
    "ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE","ASTHMAE","CATRACTE","HCHOLE","ARTHRE","HIBPE"
]
ALL_TARGETS = ADL_TARGETS + DISEASE_TARGETS


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def infer_transition_from_filename(fp: Path) -> tuple[int, int] | None:
    m = re.match(r"^S(\d+)_S(\d+)\.csv$", fp.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def normalize_id_series(s: pd.Series) -> pd.Series:
    """
    Convert idauniq values like 100035.0 / "100035.0" / "100035" -> "100035"
    so merges always work.
    """
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").astype("Int64")
        return x.astype(str)
    return s.astype(str).str.extract(r"(\d+)", expand=False)


def find_baseline_col(columns: list[str], t: int, target: str) -> str | None:
    """
    Baseline columns in transition files are usually like:
      S1HIBPE, S1EATA, ...
    but we also allow S1_HIBPE style just in case.
    """
    cand1 = f"S{t}{target}"
    cand2 = f"S{t}_{target}"
    if cand1 in columns:
        return cand1
    if cand2 in columns:
        return cand2
    # last resort: case-insensitive match
    up = {c.upper(): c for c in columns}
    if cand1.upper() in up:
        return up[cand1.upper()]
    if cand2.upper() in up:
        return up[cand2.upper()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, choices=["MAR", "MNAR"])
    ap.add_argument("--method", default=None, choices=["Cart", "Pmm"])
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")
    ap.add_argument("--model", required=True, choices=["LR", "DNN"])

    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--probs-root", default="analysis")  # expects analysis/lr/... or analysis/dnn/...
    ap.add_argument("--out-root", default="analysis/derived/damage_repair")

    ap.add_argument("--min-n", type=int, default=50, help="min rows required to keep a (transition,target,event_type)")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""

    trans_dir = Path(args.transitions_dir) / run_tag
    if not trans_dir.exists():
        raise FileNotFoundError(f"Transitions folder not found: {trans_dir}")

    model_dirname = "lr" if args.model == "LR" else "dnn"
    probs_dir = Path(args.probs_root) / model_dirname / run_tag
    if not probs_dir.exists():
        raise FileNotFoundError(f"Probs folder not found: {probs_dir}")

    out_dir = Path(args.out_root) / args.model / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in trans_dir.glob("S*_S*.csv") if infer_transition_from_filename(p) is not None])
    if not files:
        raise FileNotFoundError(f"No transition CSV files found in: {trans_dir}")

    all_rows = []

    print(f"[OK] transitions: {trans_dir}")
    print(f"[OK] probs:       {probs_dir}")
    print(f"[OK] out:         {out_dir}")
    print(f"[OK] run tag:     {run_tag}")
    print(f"[OK] found {len(files)} transition files")

    for fp in files:
        t_tp1 = infer_transition_from_filename(fp)
        if t_tp1 is None:
            continue
        t, tp1 = t_tp1
        trans_name = f"S{t}_S{tp1}"

        probs_fp = probs_dir / f"probs_{trans_name}.csv"
        if not probs_fp.exists():
            print(f"[SKIP] missing probs file: {probs_fp}")
            continue

        print(f"\n=== Build events: {trans_name} ===")
        dft = pd.read_csv(fp)
        dfp = pd.read_csv(probs_fp)

        if "idauniq" not in dft.columns or "idauniq" not in dfp.columns:
            print("[SKIP] idauniq missing in either transition or probs file")
            continue

        # normalize id so merge is reliable
        dft["idauniq"] = normalize_id_series(dft["idauniq"])
        dfp["idauniq"] = normalize_id_series(dfp["idauniq"])

        # determine available targets by checking p-columns and y-columns
        cols_t = list(dft.columns)
        cols_p = list(dfp.columns)

        # We’ll generate events for targets that exist in BOTH:
        # - probability column P_S{tp1}_{target}
        # - truth next-wave column Y_S{tp1}_{target}
        # - baseline state column S{t}{target} (or S{t}_{target})
        for target in ALL_TARGETS:
            pcol = f"P_S{tp1}_{target}"
            ycol = f"Y_S{tp1}_{target}"
            bcol = find_baseline_col(cols_t, t, target)

            if pcol not in cols_p:
                continue
            if ycol not in cols_t:
                continue
            if bcol is None:
                continue

            sub = (
                dfp[["idauniq", pcol]]
                .merge(dft[["idauniq", bcol, ycol]], on="idauniq", how="inner")
                .rename(columns={pcol: "p_next1", bcol: "x_t", ycol: "y_next"})
            )

            # force numeric
            sub["x_t"] = pd.to_numeric(sub["x_t"], errors="coerce")
            sub["y_next"] = pd.to_numeric(sub["y_next"], errors="coerce")
            sub["p_next1"] = pd.to_numeric(sub["p_next1"], errors="coerce")

            # keep only valid binary rows
            sub = sub.dropna(subset=["x_t", "y_next", "p_next1"])
            sub = sub[(sub["x_t"].isin([0, 1])) & (sub["y_next"].isin([0, 1]))]
            if sub.empty:
                continue

            # DAMAGE: restrict x_t==0, event = (y_next==1), p = p_next1
            dmg = sub[sub["x_t"] == 0].copy()
            if not dmg.empty:
                dmg["event_type"] = "damage"
                dmg["y_event"] = (dmg["y_next"] == 1).astype(int)
                dmg["p_event"] = dmg["p_next1"].astype(float)

                if len(dmg) >= args.min_n and dmg["y_event"].nunique() >= 1:
                    all_rows.append(pd.DataFrame({
                        "model": args.model,
                        "run_tag": run_tag,
                        "scenario": scenario,
                        "method": method,
                        "transition": trans_name,
                        "t": t,
                        "tp1": tp1,
                        "target": target,
                        "target_type": "adl" if target in ADL_TARGETS else "disease",
                        "event_type": "damage",
                        "idauniq": dmg["idauniq"].values,
                        "p": dmg["p_event"].values,
                        "y": dmg["y_event"].values,
                    }))

            # REPAIR: restrict x_t==1, event = (y_next==0), p = 1 - p_next1
            rep = sub[sub["x_t"] == 1].copy()
            if not rep.empty:
                rep["event_type"] = "repair"
                rep["y_event"] = (rep["y_next"] == 0).astype(int)
                rep["p_event"] = (1.0 - rep["p_next1"].astype(float))

                if len(rep) >= args.min_n and rep["y_event"].nunique() >= 1:
                    all_rows.append(pd.DataFrame({
                        "model": args.model,
                        "run_tag": run_tag,
                        "scenario": scenario,
                        "method": method,
                        "transition": trans_name,
                        "t": t,
                        "tp1": tp1,
                        "target": target,
                        "target_type": "adl" if target in ADL_TARGETS else "disease",
                        "event_type": "repair",
                        "idauniq": rep["idauniq"].values,
                        "p": rep["p_event"].values,
                        "y": rep["y_event"].values,
                    }))

        if not all_rows:
            print("[WARN] no events accumulated so far (maybe missing columns?)")
            continue

        # write per-transition slice
        df_all_tmp = pd.concat(all_rows, ignore_index=True)
        df_trans = df_all_tmp[df_all_tmp["transition"] == trans_name].copy()
        if df_trans.empty:
            print(f"[WARN] no rows for {trans_name}")
            continue

        out_fp = out_dir / f"events_{trans_name}.csv"
        df_trans.to_csv(out_fp, index=False)
        print(f"[OK] wrote {out_fp} | rows={len(df_trans):,}")

    if not all_rows:
        print(f"[WARN] No events computed for {args.model} {run_tag}")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    out_all = out_dir / "events_all.csv"
    df_all.to_csv(out_all, index=False)
    print(f"\n[OK] wrote {out_all} | rows={len(df_all):,}")


if __name__ == "__main__":
    main()
