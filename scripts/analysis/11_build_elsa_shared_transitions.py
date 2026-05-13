#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_EXCLUDED_ELSA = {"SHLT"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_shared_mapping(mapping_fp: Path, excluded_elsa: set[str]) -> tuple[list[str], list[str]]:
    mapping = pd.read_csv(mapping_fp)
    mapping["elsa_name"] = mapping["elsa_name"].astype(str).str.strip().str.upper()
    mapping["role"] = mapping["role"].astype(str).str.strip().str.lower()
    mapping["include_in_shared_model"] = (
        pd.to_numeric(mapping["include_in_shared_model"], errors="coerce").fillna(0).astype(int)
    )

    shared = mapping[
        (mapping["include_in_shared_model"] == 1)
        & (~mapping["elsa_name"].isin(excluded_elsa))
    ].copy()

    predictors = sorted(shared[shared["role"].isin(["predictor", "both"])]["elsa_name"].dropna().unique().tolist())
    targets = sorted(shared[shared["role"].isin(["target", "both"])]["elsa_name"].dropna().unique().tolist())
    return predictors, targets


def parse_transition_name(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"S(\d+)_S(\d+)\.csv", name)
    if not m:
        raise ValueError(f"Unexpected transition filename: {name}")
    return int(m.group(1)), int(m.group(2))


def build_subset_frame(df: pd.DataFrame, t: int, tp1: int, predictors: list[str], targets: list[str]) -> tuple[pd.DataFrame, dict]:
    predictor_cols = [f"S{t}{name}" for name in predictors if f"S{t}{name}" in df.columns]
    target_cols = [f"Y_S{tp1}_{name}" for name in targets if f"Y_S{tp1}_{name}" in df.columns]

    cols = ["idauniq", *predictor_cols, *target_cols]
    out = df.loc[:, cols].copy()

    keep = out[target_cols].notna().any(axis=1)
    out = out.loc[keep].reset_index(drop=True)

    summary = {
        "t": t,
        "tp1": tp1,
        "n_rows": int(len(out)),
        "n_predictors": int(len(predictor_cols)),
        "n_targets": int(len(target_cols)),
        "predictor_cols": "|".join(predictor_cols),
        "target_cols": "|".join(target_cols),
        "n_complete_predictors": int(out[predictor_cols].notna().all(axis=1).sum()) if len(out) and predictor_cols else 0,
        "n_complete_targets": int(out[target_cols].notna().all(axis=1).sum()) if len(out) and target_cols else 0,
        "n_complete_all": int((out[predictor_cols].notna().all(axis=1) & out[target_cols].notna().all(axis=1)).sum())
        if len(out) and predictor_cols and target_cols
        else 0,
    }
    return out, summary


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser(
        description="Build a strict ELSA shared-subset transition set that matches the HRS shared predictor/target schema."
    )
    ap.add_argument(
        "--mapping-csv",
        default=str(root / "data/schema/harmonized/elsa_hrs_shared_mapping.csv"),
        help="Shared ELSA-HRS mapping CSV",
    )
    ap.add_argument(
        "--source-dir",
        default=str(root / "Transition_data/01_transitions/MAR_Cart"),
        help="Existing ELSA transition directory to subset",
    )
    ap.add_argument(
        "--out-dir",
        default=str(root / "Transition_data/01_transitions"),
        help="Root output directory",
    )
    ap.add_argument("--run-tag", default="ELSA_SHARED", help="Subfolder name under --out-dir")
    ap.add_argument(
        "--exclude-elsa",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDED_ELSA),
        help="Canonical ELSA variables to exclude even if the mapping marks them shared",
    )
    args = ap.parse_args()

    mapping_fp = Path(args.mapping_csv)
    source_dir = Path(args.source_dir)
    out_root = Path(args.out_dir) / args.run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    if not mapping_fp.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_fp}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source transition directory not found: {source_dir}")

    predictors, targets = load_shared_mapping(mapping_fp, excluded_elsa={v.strip().upper() for v in args.exclude_elsa})

    manifest_rows: list[dict] = []
    pd.DataFrame({"elsa_name": predictors}).to_csv(out_root / "shared_predictors_used.csv", index=False)
    pd.DataFrame({"elsa_name": targets}).to_csv(out_root / "shared_targets_used.csv", index=False)

    for fp in sorted(source_dir.glob("S*_S*.csv")):
        t, tp1 = parse_transition_name(fp.name)
        df = pd.read_csv(fp)
        if "idauniq" not in df.columns:
            raise ValueError(f"Missing idauniq in {fp}")

        out, summary = build_subset_frame(df, t=t, tp1=tp1, predictors=predictors, targets=targets)
        if out.empty:
            print(f"[SKIP] {fp.name}: no rows with at least one observed shared next-wave target")
            continue

        out_fp = out_root / fp.name
        out.to_csv(out_fp, index=False)

        summary.update(
            {
                "run_tag": args.run_tag,
                "file": str(out_fp),
                "id_col": "idauniq",
            }
        )
        manifest_rows.append(summary)
        print(
            f"[OK] {out_fp.name} | rows={summary['n_rows']:,} | "
            f"X={summary['n_predictors']} | Y={summary['n_targets']} | "
            f"complete_all={summary['n_complete_all']:,}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_fp = out_root / "manifest_transitions.csv"
    manifest.to_csv(manifest_fp, index=False)
    print(f"[OK] Manifest: {manifest_fp}")
    print(f"[OK] Shared predictors used: {len(predictors)}")
    print(f"[OK] Shared targets used: {len(targets)}")


if __name__ == "__main__":
    main()
