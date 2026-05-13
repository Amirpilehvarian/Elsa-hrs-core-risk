#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ADL_TARGETS = {
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA","TOILTA","MEALSA","MAPA",
    "BEDA","DIMEA","SHOPA","BATHA","ARMSA","DRESSA","WALK100A","SITA","CLIM1A","HOUSEWKA",
    "PUSHA","LIFTA","CHAIRA","CLIMSA","STOOPA"
}

DISEASE_TARGETS = {
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE","LUNGE","CANCRE",
    "ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE","ASTHMAE","CATRACTE","HCHOLE","ARTHRE","HIBPE"
}

DEFAULT_EXCLUDED_ELSA = {"SHLT"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def target_type_for(name: str) -> str:
    name = str(name).strip().upper()
    if name in ADL_TARGETS:
        return "ADL"
    if name in DISEASE_TARGETS:
        return "Disease"
    return "Other"


def load_metrics_by_target(root: Path, model: str, run_tag: str) -> pd.DataFrame:
    fp = root / model.lower() / run_tag / "metrics_by_target.csv"
    df = pd.read_csv(fp)
    df["dataset_run"] = run_tag
    df["model"] = model.upper()
    df["target_type"] = df["target_type"].astype(str).str.upper().replace({"DISEASE": "Disease", "ADL": "ADL"})
    return df


def aggregate_metrics(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "target_type"], as_index=False)
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
    )
    grouped.insert(0, "dataset", dataset_label)
    return grouped


def load_manifest_summary(transitions_root: Path, run_tag: str, dataset_label: str, note: str) -> pd.DataFrame:
    manifest_fp = transitions_root / run_tag / "manifest_transitions.csv"
    manifest = pd.read_csv(manifest_fp)
    transition_files = sorted((transitions_root / run_tag).glob("S*_S*.csv"))

    ids = set()
    for fp in transition_files:
        df = pd.read_csv(fp, usecols=["idauniq"])
        ids.update(df["idauniq"].dropna().tolist())

    t_min = int(manifest["t"].min())
    tp1_max = int(manifest["tp1"].max())
    row = {
        "dataset": dataset_label,
        "run_tag": run_tag,
        "usable_transitions": f"S{t_min}->S{t_min + 1} to S{tp1_max - 1}->S{tp1_max}",
        "n_transitions": int(len(manifest)),
        "n_unique_people": int(len(ids)),
        "n_transition_rows": int(manifest["n_rows"].sum()),
        "predictors_per_transition": int(manifest["n_predictors"].max()),
        "targets_per_transition": int(manifest["n_targets"].max()),
        "note": note,
    }
    return pd.DataFrame([row])


def build_prevalence_table(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    out = (
        df.groupby(["target", "target_type"], as_index=False)
        .agg(
            n_transitions=("transition", "nunique"),
            mean_prevalence=("prevalence", "mean"),
            min_prevalence=("prevalence", "min"),
            max_prevalence=("prevalence", "max"),
        )
        .sort_values(["target_type", "target"])
        .reset_index(drop=True)
    )
    out.insert(0, "dataset", dataset_label)
    return out


def build_hr_summary(hr_root: Path, run_tag: str, dataset_label: str) -> pd.DataFrame:
    rows = []
    for model in ("LR", "DNN"):
        fp = hr_root / model / run_tag / "HR_by_target.csv"
        df = pd.read_csv(fp)
        df["target_type"] = df["type"].astype(str).str.strip().str.upper().replace({"DISEASE": "Disease", "ADL": "ADL"})
        grouped = (
            df.groupby("target_type", as_index=False)
            .agg(
                mean_hr=("HR", "mean"),
                sd_hr=("HR", "std"),
                n_targets=("target", "nunique"),
            )
        )
        grouped.insert(0, "model", model)
        grouped.insert(0, "dataset", dataset_label)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_damage_repair_summary(dr_root: Path, run_tag: str, dataset_label: str) -> pd.DataFrame:
    rows = []
    for model in ("LR", "DNN"):
        fp = dr_root / model.lower() / run_tag / "metrics_summary.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df["target_type"] = df["target_type"].astype(str).str.upper().replace({"DISEASE": "Disease", "ADL": "ADL"})
        df["event_type"] = df["event_type"].astype(str).str.lower()
        df.insert(0, "dataset", dataset_label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "dataset",
        "model",
        "run_tag",
        "event_type",
        "target_type",
        "n_target_transition_pairs",
        "n_unique_targets",
        "mean_auc",
        "mean_pr_auc",
        "mean_brier",
        "mean_ece",
        "mean_r2cal",
        "mean_prevalence",
    ]
    return out.loc[:, keep_cols]


def build_shared_schema(mapping_fp: Path, excluded_elsa: set[str]) -> pd.DataFrame:
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
    keep_cols = [
        "elsa_name",
        "role",
        "hrs_current",
        "hrs_next",
        "mapping_status",
        "hrs_label",
        "notes",
    ]
    shared = shared.loc[:, keep_cols].sort_values(["role", "elsa_name"]).reset_index(drop=True)
    shared["target_type"] = shared["elsa_name"].map(target_type_for)
    return shared


def write_figure_shortlist(out_fp: Path) -> None:
    text = """# Figure Shortlist

## Main text

1. ELSA pooled calibration:
   - `analysis/ELSA/figures/calibration/MNAR_Pmm/S6_S7/POOLED_all.png`
   - `analysis/ELSA/figures/calibration_loglog/MNAR_Pmm/S6_S7/POOLED_all.png`
2. HRS pooled calibration:
   - `analysis/HRS/figures/calibration/HRS_SHARED/S13_S14/POOLED_all.png`
   - `analysis/HRS/figures/calibration_loglog/HRS_SHARED/S13_S14/POOLED_all.png`
3. Hazard-ratio summaries:
   - `analysis/ELSA/figures/hazard_ratio/MNAR_Pmm/DNN/HR_Disease.png`
   - `analysis/ELSA/figures/hazard_ratio/MNAR_Pmm/DNN/HR_ADL.png`
   - `analysis/HRS/figures/hazard_ratio/HRS_SHARED/DNN/HR_Disease.png`
   - `analysis/HRS/figures/hazard_ratio/HRS_SHARED/DNN/HR_ADL.png`
4. Feature-selection summary:
   - `analysis/ELSA/feature_selection/global_filter_lr_vs_dnn/aggregate_curve_adl.png`
   - `analysis/ELSA/feature_selection/global_filter_lr_vs_dnn/aggregate_curve_disease.png`

## Appendix

1. ECE comparisons:
   - `analysis/ELSA/figures/ece_plots/MNAR_Pmm/01_ece_box_by_transition.png`
   - `analysis/HRS/figures/ece_plots/HRS_SHARED/01_ece_box_by_transition.png`
2. Per-transition calibration panels:
   - `analysis/ELSA/figures/calibration/MNAR_Pmm/`
   - `analysis/HRS/figures/calibration/HRS_SHARED/`
3. Optional log-log per-target examples:
   - `analysis/HRS/figures/calibration_loglog/HRS_SHARED/S13_S14/LUNGE.png`
   - `analysis/HRS/figures/calibration_loglog/HRS_SHARED/S13_S14/MONEYA.png`
"""
    out_fp.write_text(text, encoding="utf-8")


def write_results_highlights(out_fp: Path, elsa_full: pd.DataFrame, shared_comp: pd.DataFrame, hr_summary: pd.DataFrame) -> None:
    def pick(df: pd.DataFrame, dataset: str, model: str, target_type: str) -> pd.Series:
        row = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["target_type"] == target_type)]
        if row.empty:
            raise ValueError(f"Missing row for {dataset} {model} {target_type}")
        return row.iloc[0]

    lines: list[str] = ["# Results Highlights", ""]

    lines.append("## ELSA full model")
    for target_type in ("ADL", "Disease"):
        lr = pick(elsa_full, "ELSA full", "LR", target_type)
        dnn = pick(elsa_full, "ELSA full", "DNN", target_type)
        lines.append(
            f"- {target_type}: LR AUC {lr['mean_auc']:.3f}, Brier {lr['mean_brier']:.3f}, "
            f"ECE {lr['mean_ece']:.3f}, R2cal {lr['mean_r2cal']:.3f}; "
            f"DNN AUC {dnn['mean_auc']:.3f}, Brier {dnn['mean_brier']:.3f}, "
            f"ECE {dnn['mean_ece']:.3f}, R2cal {dnn['mean_r2cal']:.3f}."
        )
    lines.append("")

    lines.append("## Shared-subset external validation")
    for dataset in ("ELSA shared", "HRS shared"):
        for target_type in ("ADL", "Disease"):
            lr = pick(shared_comp, dataset, "LR", target_type)
            dnn = pick(shared_comp, dataset, "DNN", target_type)
            lines.append(
                f"- {dataset}, {target_type}: LR AUC {lr['mean_auc']:.3f}, Brier {lr['mean_brier']:.3f}, "
                f"ECE {lr['mean_ece']:.3f}, R2cal {lr['mean_r2cal']:.3f}; "
                f"DNN AUC {dnn['mean_auc']:.3f}, Brier {dnn['mean_brier']:.3f}, "
                f"ECE {dnn['mean_ece']:.3f}, R2cal {dnn['mean_r2cal']:.3f}."
            )
    lines.append("")

    lines.append("## Hazard-ratio summaries")
    for dataset in ("ELSA full", "ELSA shared", "HRS shared"):
        subset = hr_summary[hr_summary["dataset"] == dataset]
        if subset.empty:
            continue
        for model in ("LR", "DNN"):
            part = subset[subset["model"] == model]
            if part.empty:
                continue
            adl = part[part["target_type"] == "ADL"]
            dis = part[part["target_type"] == "Disease"]
            if not adl.empty and not dis.empty:
                lines.append(
                    f"- {dataset}, {model}: mean HR ADL {adl.iloc[0]['mean_hr']:.2f}, "
                    f"Disease {dis.iloc[0]['mean_hr']:.2f}."
                )

    out_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser(description="Build manuscript-ready tables and notes from the current canonical outputs.")
    ap.add_argument("--elsa-full-run", default="MNAR_Pmm")
    ap.add_argument("--elsa-shared-run", default="ELSA_SHARED")
    ap.add_argument("--hrs-shared-run", default="HRS_SHARED")
    ap.add_argument("--metrics-root", default=str(root / "analysis/ELSA/metrics"))
    ap.add_argument("--hrs-metrics-root", default=str(root / "analysis/HRS/metrics"))
    ap.add_argument("--transitions-root", default=str(root / "Transition_data/01_transitions"))
    ap.add_argument("--hrs-transitions-root", default=str(root / "Transition_data/HRS/01_transitions"))
    ap.add_argument("--hazard-root", default=str(root / "analysis/ELSA/metrics/hazard_ratio"))
    ap.add_argument("--hrs-hazard-root", default=str(root / "analysis/HRS/metrics/hazard_ratio"))
    ap.add_argument("--damage-repair-root", default=str(root / "analysis/ELSA/metrics/damage_repair"))
    ap.add_argument("--hrs-damage-repair-root", default=str(root / "analysis/HRS/metrics/damage_repair"))
    ap.add_argument("--mapping-csv", default=str(root / "data/schema/harmonized/elsa_hrs_shared_mapping.csv"))
    ap.add_argument("--out-dir", default=str(root / "manuscript/assets"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_root = Path(args.metrics_root)
    hrs_metrics_root = Path(args.hrs_metrics_root)
    transitions_root = Path(args.transitions_root)
    hrs_transitions_root = Path(args.hrs_transitions_root)
    hazard_root = Path(args.hazard_root)
    hrs_hazard_root = Path(args.hrs_hazard_root)
    damage_repair_root = Path(args.damage_repair_root)
    hrs_damage_repair_root = Path(args.hrs_damage_repair_root)

    elsa_full = pd.concat(
        [
            aggregate_metrics(load_metrics_by_target(metrics_root, "LR", args.elsa_full_run), "ELSA full"),
            aggregate_metrics(load_metrics_by_target(metrics_root, "DNN", args.elsa_full_run), "ELSA full"),
        ],
        ignore_index=True,
    )
    elsa_shared = pd.concat(
        [
            aggregate_metrics(load_metrics_by_target(metrics_root, "LR", args.elsa_shared_run), "ELSA shared"),
            aggregate_metrics(load_metrics_by_target(metrics_root, "DNN", args.elsa_shared_run), "ELSA shared"),
        ],
        ignore_index=True,
    )
    hrs_shared = pd.concat(
        [
            aggregate_metrics(load_metrics_by_target(hrs_metrics_root, "LR", args.hrs_shared_run), "HRS shared"),
            aggregate_metrics(load_metrics_by_target(hrs_metrics_root, "DNN", args.hrs_shared_run), "HRS shared"),
        ],
        ignore_index=True,
    )

    shared_comp = pd.concat([elsa_shared, hrs_shared], ignore_index=True)

    cohort_setup = pd.concat(
        [
            load_manifest_summary(
                transitions_root,
                args.elsa_full_run,
                "ELSA full",
                "Main all-wave core-only analysis with 87 predictors and 44 targets.",
            ),
            load_manifest_summary(
                transitions_root,
                args.elsa_shared_run,
                "ELSA shared",
                "Matched shared-subset analysis with the HRS predictor/target overlap.",
            ),
            load_manifest_summary(
                hrs_transitions_root,
                args.hrs_shared_run,
                "HRS shared",
                "External validation on the 25-predictor, 24-target shared subset.",
            ),
        ],
        ignore_index=True,
    )

    prevalence_full = build_prevalence_table(
        pd.concat(
            [
                load_metrics_by_target(metrics_root, "LR", args.elsa_full_run),
                load_metrics_by_target(metrics_root, "DNN", args.elsa_full_run),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["transition", "target"]),
        dataset_label="ELSA full",
    )
    prevalence_shared = pd.concat(
        [
            build_prevalence_table(
                load_metrics_by_target(metrics_root, "LR", args.elsa_shared_run).drop_duplicates(subset=["transition", "target"]),
                dataset_label="ELSA shared",
            ),
            build_prevalence_table(
                load_metrics_by_target(hrs_metrics_root, "LR", args.hrs_shared_run).drop_duplicates(subset=["transition", "target"]),
                dataset_label="HRS shared",
            ),
        ],
        ignore_index=True,
    )

    shared_schema = build_shared_schema(Path(args.mapping_csv), excluded_elsa=DEFAULT_EXCLUDED_ELSA)

    hr_summary = pd.concat(
        [
            build_hr_summary(hazard_root, args.elsa_full_run, "ELSA full"),
            build_hr_summary(hazard_root, args.elsa_shared_run, "ELSA shared"),
            build_hr_summary(hrs_hazard_root, args.hrs_shared_run, "HRS shared"),
        ],
        ignore_index=True,
    )
    dr_summary = pd.concat(
        [
            build_damage_repair_summary(damage_repair_root, args.elsa_full_run, "ELSA full"),
            build_damage_repair_summary(hrs_damage_repair_root, args.hrs_shared_run, "HRS shared"),
        ],
        ignore_index=True,
    )

    cohort_setup.to_csv(out_dir / "table_01_cohort_setup.csv", index=False)
    elsa_full.to_csv(out_dir / "table_02_elsa_full_model_summary.csv", index=False)
    shared_comp.to_csv(out_dir / "table_03_shared_external_validation_summary.csv", index=False)
    hr_summary.to_csv(out_dir / "table_04_hazard_ratio_summary.csv", index=False)
    if not dr_summary.empty:
        dr_summary.to_csv(out_dir / "table_05_damage_repair_summary.csv", index=False)
    prevalence_full.to_csv(out_dir / "appendix_table_01_target_prevalence_elsa_full.csv", index=False)
    prevalence_shared.to_csv(out_dir / "appendix_table_02_target_prevalence_shared_validation.csv", index=False)
    shared_schema.to_csv(out_dir / "appendix_table_03_shared_schema.csv", index=False)

    write_figure_shortlist(out_dir / "figure_shortlist.md")
    write_results_highlights(out_dir / "results_highlights.md", elsa_full=elsa_full, shared_comp=shared_comp, hr_summary=hr_summary)

    print(f"[OK] Wrote manuscript assets to {out_dir}")


if __name__ == "__main__":
    main()
