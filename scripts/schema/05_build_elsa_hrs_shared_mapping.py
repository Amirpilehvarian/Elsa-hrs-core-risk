#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SAFE_DIRECT = {
    "WALKRA": ("walkra", "walkra_next", "difficulty walking across room", "Direct overlap"),
    "DRESSA": ("dressa", "dressa_next", "difficulty dressing", "Direct overlap"),
    "BATHA": ("batha", "batha_next", "difficulty bathing or showering", "Direct overlap"),
    "EATA": ("eata", "eata_next", "difficulty eating", "Direct overlap"),
    "BEDA": ("beda", "beda_next", "difficulty getting in or out of bed", "Direct overlap"),
    "TOILTA": ("toilta", "toilta_next", "difficulty using the toilet", "Direct overlap"),
    "SHOPA": ("shopa", "shopa_next", "difficulty shopping for groceries", "Direct overlap"),
    "PHONEA": ("phonea", "phonea_next", "difficulty using telephone", "Direct overlap"),
    "MONEYA": ("moneya", "moneya_next", "difficulty managing money", "Direct overlap"),
    "CHAIRA": ("chaira", "chaira_next", "difficulty getting up from chair", "Direct overlap"),
    "CLIMSA": ("climsa", "climsa_next", "difficulty climbing several flights of stairs", "Direct overlap"),
    "CLIM1A": ("clim1a", "clim1a_next", "difficulty climbing one flight of stairs", "Direct overlap"),
    "STOOPA": ("stoopa", "stoopa_next", "difficulty stooping, kneeling, or crouching", "Direct overlap"),
    "ARMSA": ("armsa", "armsa_next", "difficulty reaching or extending arms up", "Direct overlap"),
    "PUSHA": ("pusha", "pusha_next", "difficulty pushing or pulling large object", "Direct overlap"),
    "LIFTA": ("lifta", "lifta_next", "difficulty lifting or carrying 10 pounds", "Direct overlap"),
    "DIMEA": ("dimea", "dimea_next", "difficulty picking up a dime", "Direct overlap"),
    "HIBPE": ("hibpe", "hibpe_next", "ever had high blood pressure", "Direct overlap"),
    "DIABE": ("diabe", "diabe_next", "ever had diabetes", "Direct overlap"),
    "CANCRE": ("cancre", "cancre_next", "ever had cancer", "Direct overlap"),
    "LUNGE": ("lunge", "lunge_next", "ever had lung disease", "Direct overlap"),
    "HEARTE": ("hearte", "hearte_next", "ever had heart problems", "Direct overlap"),
    "STROKE": ("stroke", "stroke_next", "ever had stroke", "Direct overlap"),
    "ARTHRE": ("arthre", "arthre_next", "ever had arthritis", "Direct overlap"),
}

SAFE_SEMANTIC = {
    "INDAGER": ("age", "age_next", "age at measurement", "Safe semantic match for age"),
}

REVIEWED_EXCLUDED = {
    "SHLT": (
        "shlt",
        "shlt_next",
        "self-reported health deficit (binary in HRS)",
        "Excluded from shared model: HRS stores a binary deficit, while ELSA uses an ordinal self-rated-health scale.",
    ),
}

CANDIDATE_PROXY = {
    "WALK100A": ("walk1a", "walk1a_next", "difficulty walking one block", "Candidate proxy only; not a direct 100-yard measure"),
}


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "data").exists() and (cur / "scripts").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def role_of(var: str, predictors: set[str], targets: set[str]) -> str:
    if var in predictors and var in targets:
        return "both"
    if var in predictors:
        return "predictor"
    if var in targets:
        return "target"
    return "other"


def safe_float_pct(s: pd.Series) -> float | None:
    if s is None:
        return None
    return round(float(s.isna().mean() * 100.0), 4)


def build_rows(elsa_vars: list[str], predictors: set[str], targets: set[str], hrs: pd.DataFrame) -> list[dict]:
    hrs_cols = {c.lower(): c for c in hrs.columns}
    rows: list[dict] = []

    for var in elsa_vars:
        role = role_of(var, predictors, targets)
        status = "unavailable"
        include = False
        hrs_current = ""
        hrs_next = ""
        hrs_label = ""
        notes = ""

        if var in SAFE_DIRECT:
            hrs_current, hrs_next, hrs_label, notes = SAFE_DIRECT[var]
            status = "direct_exact"
            include = True
        elif var in SAFE_SEMANTIC:
            hrs_current, hrs_next, hrs_label, notes = SAFE_SEMANTIC[var]
            status = "direct_semantic"
            include = True
        elif var in REVIEWED_EXCLUDED:
            hrs_current, hrs_next, hrs_label, notes = REVIEWED_EXCLUDED[var]
            status = "semantic_mismatch_excluded"
            include = False
        elif var in CANDIDATE_PROXY:
            hrs_current, hrs_next, hrs_label, notes = CANDIDATE_PROXY[var]
            status = "candidate_proxy"
            include = False
        else:
            exact_current = hrs_cols.get(var.lower())
            exact_next = hrs_cols.get(f"{var.lower()}_next")
            if exact_current is not None or exact_next is not None:
                hrs_current = exact_current or ""
                hrs_next = exact_next or ""
                status = "exact_name_unreviewed"
                notes = "Name overlap found automatically; review before use"

        current_missing = None
        next_missing = None
        if hrs_current and hrs_current in hrs.columns:
            current_missing = safe_float_pct(hrs[hrs_current])
        if hrs_next and hrs_next in hrs.columns:
            next_missing = safe_float_pct(hrs[hrs_next])

        rows.append(
            {
                "elsa_name": var,
                "role": role,
                "hrs_current": hrs_current,
                "hrs_next": hrs_next,
                "mapping_status": status,
                "include_in_shared_model": int(include),
                "hrs_label": hrs_label,
                "hrs_current_missing_pct": current_missing,
                "hrs_next_missing_pct": next_missing,
                "notes": notes,
            }
        )

    return rows


def main() -> None:
    root = find_repo_root(Path(__file__).parent)

    ap = argparse.ArgumentParser(
        description="Build an explicit ELSA-to-HRS shared-variable mapping from the current ELSA transition schema and the HRS Markov-format dataset."
    )
    ap.add_argument(
        "--elsa-transition",
        default=str(root / "Transition_data/01_transitions/MAR_Cart/S2_S3.csv"),
        help="Reference ELSA transition CSV used to define the active predictor/target schema",
    )
    ap.add_argument(
        "--hrs-data",
        default=str(root / "data/restricted/HRS/hrs_rand_preproc.csv"),
        help="HRS Markov-format preprocessed CSV",
    )
    ap.add_argument(
        "--out-dir",
        default=str(root / "data/schema/harmonized"),
        help="Output directory for mapping tables",
    )
    args = ap.parse_args()

    elsa_transition = Path(args.elsa_transition)
    hrs_data = Path(args.hrs_data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not elsa_transition.exists():
        raise FileNotFoundError(f"ELSA transition file not found: {elsa_transition}")
    if not hrs_data.exists():
        raise FileNotFoundError(f"HRS data file not found: {hrs_data}")

    elsa = pd.read_csv(elsa_transition, nrows=1)
    hrs = pd.read_csv(hrs_data)

    predictors = [c[2:] for c in elsa.columns if c.startswith("S2")]
    targets = [c.replace("Y_S3_", "") for c in elsa.columns if c.startswith("Y_S3_")]

    predictor_set = set(predictors)
    target_set = set(targets)
    all_vars = ordered_unique(predictors + targets)

    rows = build_rows(all_vars, predictor_set, target_set, hrs)
    mapping = pd.DataFrame(rows)

    role_order = {"both": 0, "predictor": 1, "target": 2, "other": 3}
    status_order = {
        "direct_exact": 0,
        "direct_semantic": 1,
        "semantic_mismatch_excluded": 2,
        "candidate_proxy": 3,
        "exact_name_unreviewed": 4,
        "unavailable": 5,
    }
    mapping["_role_order"] = mapping["role"].map(role_order).fillna(9)
    mapping["_status_order"] = mapping["mapping_status"].map(status_order).fillna(9)
    mapping = mapping.sort_values(["_status_order", "_role_order", "elsa_name"]).drop(columns=["_role_order", "_status_order"])

    shared_predictors = mapping[
        (mapping["include_in_shared_model"] == 1) & (mapping["role"].isin(["predictor", "both"]))
    ].copy()
    shared_targets = mapping[
        (mapping["include_in_shared_model"] == 1) & (mapping["role"].isin(["target", "both"]))
    ].copy()
    summary = (
        mapping.groupby(["mapping_status", "role"], as_index=False)
        .agg(n=("elsa_name", "count"), n_included=("include_in_shared_model", "sum"))
        .sort_values(["mapping_status", "role"])
    )

    mapping_fp = out_dir / "elsa_hrs_shared_mapping.csv"
    predictors_fp = out_dir / "elsa_hrs_shared_predictors.csv"
    targets_fp = out_dir / "elsa_hrs_shared_targets.csv"
    summary_fp = out_dir / "elsa_hrs_shared_summary.csv"

    mapping.to_csv(mapping_fp, index=False)
    shared_predictors.to_csv(predictors_fp, index=False)
    shared_targets.to_csv(targets_fp, index=False)
    summary.to_csv(summary_fp, index=False)

    print(f"[OK] Wrote mapping: {mapping_fp}")
    print(f"[OK] Wrote shared predictors: {predictors_fp} | n={len(shared_predictors)}")
    print(f"[OK] Wrote shared targets: {targets_fp} | n={len(shared_targets)}")
    print(f"[OK] Wrote summary: {summary_fp}")


if __name__ == "__main__":
    main()
