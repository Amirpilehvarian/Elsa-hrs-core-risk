#!/usr/bin/env python3
"""Run the HRS shared-subset validation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_step(label: str, cmd: list[str], root: Path, dry_run: bool) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg")
    subprocess.run(cmd, cwd=root, check=True, env=env)


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="HRS_SHARED")
    ap.add_argument("--hrs-data", default="data/restricted/HRS/hrs_rand_preproc.csv")
    ap.add_argument("--mapping-csv", default="data/schema/harmonized/elsa_hrs_shared_mapping.csv")

    ap.add_argument("--transitions-root", default="Transition_data/HRS/01_transitions")
    ap.add_argument("--analysis-root", default="analysis/HRS")
    ap.add_argument("--metrics-root", default="analysis/HRS/metrics")
    ap.add_argument("--figures-root", default="analysis/HRS/figures")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--lr-max-iter", type=int, default=2000)
    ap.add_argument("--lr-class-weight", default="none", choices=["none", "balanced"])
    ap.add_argument("--dnn-hidden", type=int, default=128)
    ap.add_argument("--dnn-hidden2", type=int, default=0)
    ap.add_argument("--dnn-hidden-sizes", default=None)
    ap.add_argument("--dnn-epochs", type=int, default=30)
    ap.add_argument("--dnn-batch", type=int, default=64)
    ap.add_argument("--dnn-lr", type=float, default=1e-3)
    ap.add_argument("--dnn-l2", type=float, default=0.0)
    ap.add_argument("--dnn-dropout", type=float, default=0.0)
    ap.add_argument("--dnn-patience", type=int, default=5)
    ap.add_argument("--dnn-activation", default="relu", choices=["relu", "gelu"])
    ap.add_argument("--dnn-batchnorm", action="store_true")
    ap.add_argument("--dnn-standardize-x", action="store_true")
    ap.add_argument("--dnn-reduce-lr-on-plateau", action="store_true")
    ap.add_argument("--dnn-min-lr", type=float, default=1e-5)
    ap.add_argument("--calibration-bins", type=int, default=10)

    ap.add_argument("--skip-mapping", action="store_true")
    ap.add_argument("--skip-transitions", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--skip-hazard", action="store_true")
    ap.add_argument("--skip-calibration-plots", action="store_true")
    ap.add_argument("--skip-hazard-plots", action="store_true")
    ap.add_argument("--skip-ece-plots", action="store_true")
    ap.add_argument("--skip-metric-summary-plots", action="store_true")
    ap.add_argument("--include-loglog-calibration", action="store_true")
    ap.add_argument("--include-damage-repair", action="store_true")
    ap.add_argument("--include-sex-stratified-analysis", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.skip_mapping:
        run_step(
            "Refresh ELSA-HRS Shared Mapping",
            [
                sys.executable,
                "scripts/schema/05_build_elsa_hrs_shared_mapping.py",
            ],
            root,
            args.dry_run,
        )

    if not args.skip_transitions:
        run_step(
            "Build HRS Shared Transitions",
            [
                sys.executable,
                "scripts/analysis/09_build_hrs_shared_transitions.py",
                "--hrs-data",
                args.hrs_data,
                "--mapping-csv",
                args.mapping_csv,
                "--out-dir",
                args.transitions_root,
                "--run-tag",
                args.run_tag,
            ],
            root,
            args.dry_run,
        )

    if not args.skip_models:
        run_step(
            "Fit HRS LR",
            [
                sys.executable,
                "scripts/analysis/02_LR_oof_probs_and_metrics.py",
                "--run-tag",
                args.run_tag,
                "--transitions-dir",
                args.transitions_root,
                "--out-dir",
                str(Path(args.analysis_root) / "lr"),
                "--cv-folds",
                str(args.cv_folds),
                "--max-iter",
                str(args.lr_max_iter),
                "--class-weight",
                args.lr_class_weight,
                "--export-probs",
                "--seed",
                str(args.seed),
            ],
            root,
            args.dry_run,
        )
        dnn_cmd = [
            sys.executable,
            "scripts/analysis/04_dnn_oof_probs_and_metrics.py",
            "--run-tag",
            args.run_tag,
            "--transitions-dir",
            args.transitions_root,
            "--out-dir",
            str(Path(args.analysis_root) / "dnn"),
            "--cv-folds",
            str(args.cv_folds),
            "--epochs",
            str(args.dnn_epochs),
            "--batch",
            str(args.dnn_batch),
            "--lr",
            str(args.dnn_lr),
            "--l2",
            str(args.dnn_l2),
            "--dropout",
            str(args.dnn_dropout),
            "--patience",
            str(args.dnn_patience),
            "--activation",
            args.dnn_activation,
            "--min-lr",
            str(args.dnn_min_lr),
            "--seed",
            str(args.seed),
        ]
        if args.dnn_hidden_sizes:
            dnn_cmd.extend(["--hidden-sizes", args.dnn_hidden_sizes])
        else:
            dnn_cmd.extend(["--hidden", str(args.dnn_hidden), "--hidden2", str(args.dnn_hidden2)])
        if args.dnn_batchnorm:
            dnn_cmd.append("--batchnorm")
        if args.dnn_standardize_x:
            dnn_cmd.append("--standardize-x")
        if args.dnn_reduce_lr_on_plateau:
            dnn_cmd.append("--reduce-lr-on-plateau")
        run_step("Fit HRS DNN", dnn_cmd, root, args.dry_run)

    if not args.skip_metrics:
        for model in ("LR", "DNN"):
            run_step(
                f"Evaluate HRS {model}",
                [
                    sys.executable,
                    "scripts/analysis/03_evaluate_performance.py",
                    "--run-tag",
                    args.run_tag,
                    "--model",
                    model,
                    "--transitions-dir",
                    args.transitions_root,
                    "--probs-dir",
                    args.analysis_root,
                    "--out-dir",
                    args.metrics_root,
                    "--calibration-bins",
                    str(args.calibration_bins),
                ],
                root,
                args.dry_run,
            )

    if not args.skip_hazard:
        run_step(
            "Build HRS Hazard Ratio Tables",
            [
                sys.executable,
                "scripts/analysis/05_build_hazard_ratio_tables.py",
                "--run-tag",
                args.run_tag,
                "--models",
                "LR",
                "DNN",
                "--transitions-dir",
                args.transitions_root,
                "--analysis-root",
                args.analysis_root,
                "--out-root",
                str(Path(args.metrics_root) / "hazard_ratio"),
            ],
            root,
            args.dry_run,
        )

    if not args.skip_calibration_plots:
        run_step(
            "Plot HRS Calibration",
            [
                sys.executable,
                "scripts/visualization/01_plot_calibration.py",
                "--run-tag",
                args.run_tag,
                "--which",
                "both",
                "--transitions-dir",
                args.transitions_root,
                "--analysis-root",
                args.analysis_root,
                "--out-root",
                str(Path(args.figures_root) / "calibration"),
                "--pooled",
                "three",
            ],
            root,
            args.dry_run,
        )
        if args.include_loglog_calibration:
            run_step(
                "Plot HRS Calibration (Log-Log)",
                [
                    sys.executable,
                    "scripts/visualization/01_plot_calibration.py",
                    "--run-tag",
                    args.run_tag,
                    "--which",
                    "both",
                    "--transitions-dir",
                    args.transitions_root,
                    "--analysis-root",
                    args.analysis_root,
                    "--out-root",
                    str(Path(args.figures_root) / "calibration_loglog"),
                    "--pooled",
                    "three",
                    "--loglog",
                ],
                root,
                args.dry_run,
            )

    if not args.skip_hazard_plots:
        run_step(
            "Plot HRS Hazard Ratios",
            [
                sys.executable,
                "scripts/visualization/03_hazard_ratio_plots.py",
                "--run-tag",
                args.run_tag,
                "--models",
                "LR",
                "DNN",
                "--hr-root",
                str(Path(args.metrics_root) / "hazard_ratio"),
                "--out-root",
                str(Path(args.figures_root) / "hazard_ratio"),
            ],
            root,
            args.dry_run,
        )

    if not args.skip_ece_plots:
        run_step(
            "Plot HRS ECE Comparisons",
            [
                sys.executable,
                "scripts/visualization/02_ece_plots.py",
                "--run-tag",
                args.run_tag,
                "--metrics-root",
                args.metrics_root,
                "--out-dir",
                str(Path(args.figures_root) / "ece_plots"),
            ],
            root,
            args.dry_run,
        )
    if args.include_damage_repair:
        for model in ("LR", "DNN"):
            run_step(
                f"Build HRS Damage Repair Events {model}",
                [
                    sys.executable,
                    "scripts/analysis/06_build_damage_repair_tables.py",
                    "--run-tag",
                    args.run_tag,
                    "--model",
                    model,
                    "--transitions-dir",
                    args.transitions_root,
                    "--probs-root",
                    args.analysis_root,
                    "--out-root",
                    str(Path(args.analysis_root) / "derived" / "damage_repair"),
                ],
                root,
                args.dry_run,
            )
            run_step(
                f"Evaluate HRS Damage Repair {model}",
                [
                    sys.executable,
                    "scripts/analysis/13_evaluate_damage_repair.py",
                    "--run-tag",
                    args.run_tag,
                    "--model",
                    model,
                    "--events-root",
                    str(Path(args.analysis_root) / "derived" / "damage_repair"),
                    "--out-root",
                    str(Path(args.metrics_root) / "damage_repair"),
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot HRS Damage Repair Calibration",
            [
                sys.executable,
                "scripts/visualization/05_damage_repair_calibration_plots.py",
                "--run-tag",
                args.run_tag,
                "--models",
                "LR",
                "DNN",
                "--events-root",
                str(Path(args.analysis_root) / "derived" / "damage_repair"),
                "--out-root",
                str(Path(args.figures_root) / "damage_repair_calibration"),
            ],
            root,
            args.dry_run,
        )
        if args.include_loglog_calibration:
            run_step(
                "Plot HRS Damage Repair Calibration (Log-Log)",
                [
                    sys.executable,
                    "scripts/visualization/05_damage_repair_calibration_plots.py",
                    "--run-tag",
                    args.run_tag,
                    "--models",
                    "LR",
                    "DNN",
                    "--events-root",
                    str(Path(args.analysis_root) / "derived" / "damage_repair"),
                    "--out-root",
                    str(Path(args.figures_root) / "damage_repair_calibration_loglog"),
                    "--loglog",
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot HRS Damage Risk vs Age",
            [
                sys.executable,
                "scripts/visualization/07_damage_risk_vs_age_plots.py",
                "--run-tag",
                args.run_tag,
                "--dataset-label",
                "HRS",
                "--events-root",
                str(Path(args.analysis_root) / "derived" / "damage_repair"),
                "--transitions-root",
                args.transitions_root,
                "--out-root",
                str(Path(args.figures_root) / "damage_risk_vs_age"),
            ],
            root,
            args.dry_run,
        )
        run_step(
            "Plot HRS Damage/Repair Transition Metrics",
            [
                sys.executable,
                "scripts/visualization/08_damage_repair_transition_metric_plots.py",
                "--run-tag",
                args.run_tag,
                "--dataset-label",
                "HRS",
                "--metrics-root",
                str(Path(args.metrics_root) / "damage_repair"),
                "--out-root",
                str(Path(args.figures_root) / "damage_repair_transition_metrics"),
                ],
                root,
                args.dry_run,
            )

    if args.include_sex_stratified_analysis:
        run_step(
            "Build Sex Lookups",
            [
                sys.executable,
                "scripts/analysis/17_build_sex_lookups.py",
            ],
            root,
            args.dry_run,
        )
        for model in ("LR", "DNN"):
            run_step(
                f"Evaluate HRS {model} By Sex",
                [
                    sys.executable,
                    "scripts/analysis/18_evaluate_performance_by_sex.py",
                    "--run-tag",
                    args.run_tag,
                    "--model",
                    model,
                    "--transitions-dir",
                    args.transitions_root,
                    "--probs-dir",
                    args.analysis_root,
                    "--sex-lookup",
                    "data/schema/harmonized/hrs_sex_lookup.csv",
                    "--out-dir",
                    str(Path(args.metrics_root) / "by_sex"),
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot HRS Standard Transition Metrics By Sex",
            [
                sys.executable,
                "scripts/visualization/09_transition_metric_plots_by_sex.py",
                "--run-tag",
                args.run_tag,
                "--dataset-label",
                "HRS",
                "--metrics-root",
                str(Path(args.metrics_root) / "by_sex"),
                "--out-root",
                str(Path(args.figures_root) / "by_sex" / "transition_metrics"),
                "--title-suffix",
                "Standard",
            ],
            root,
            args.dry_run,
        )
        run_step(
            "Plot HRS Standard Calibration By Sex",
            [
                sys.executable,
                "scripts/visualization/10_plot_calibration_by_sex.py",
                "--run-tag",
                args.run_tag,
                "--dataset-label",
                "HRS",
                "--transitions-dir",
                args.transitions_root,
                "--analysis-root",
                args.analysis_root,
                "--sex-lookup",
                "data/schema/harmonized/hrs_sex_lookup.csv",
                "--out-root",
                str(Path(args.figures_root) / "by_sex" / "calibration"),
            ],
            root,
            args.dry_run,
        )
        if args.include_damage_repair:
            for model in ("LR", "DNN"):
                run_step(
                    f"Evaluate HRS Damage Repair {model} By Sex",
                    [
                        sys.executable,
                        "scripts/analysis/19_evaluate_damage_repair_by_sex.py",
                        "--run-tag",
                        args.run_tag,
                        "--model",
                        model,
                        "--events-root",
                        str(Path(args.analysis_root) / "derived" / "damage_repair"),
                        "--sex-lookup",
                        "data/schema/harmonized/hrs_sex_lookup.csv",
                        "--out-root",
                        str(Path(args.metrics_root) / "by_sex" / "damage_repair"),
                    ],
                    root,
                    args.dry_run,
                )
            run_step(
                "Plot HRS Damage Transition Metrics By Sex",
                [
                    sys.executable,
                    "scripts/visualization/09_transition_metric_plots_by_sex.py",
                    "--run-tag",
                    args.run_tag,
                    "--dataset-label",
                    "HRS",
                    "--metrics-root",
                    str(Path(args.metrics_root) / "by_sex" / "damage_repair"),
                    "--out-root",
                    str(Path(args.figures_root) / "by_sex" / "damage_repair_transition_metrics"),
                    "--event-type",
                    "damage",
                    "--title-suffix",
                    "Damage",
                ],
                root,
                args.dry_run,
            )
            run_step(
                "Plot HRS Repair Transition Metrics By Sex",
                [
                    sys.executable,
                    "scripts/visualization/09_transition_metric_plots_by_sex.py",
                    "--run-tag",
                    args.run_tag,
                    "--dataset-label",
                    "HRS",
                    "--metrics-root",
                    str(Path(args.metrics_root) / "by_sex" / "damage_repair"),
                    "--out-root",
                    str(Path(args.figures_root) / "by_sex" / "damage_repair_transition_metrics"),
                    "--event-type",
                    "repair",
                    "--title-suffix",
                    "Repair",
                ],
                root,
                args.dry_run,
            )
            run_step(
                "Plot HRS Damage Repair Calibration By Sex",
                [
                    sys.executable,
                    "scripts/visualization/11_plot_damage_repair_calibration_by_sex.py",
                    "--run-tag",
                    args.run_tag,
                    "--dataset-label",
                    "HRS",
                    "--events-root",
                    str(Path(args.analysis_root) / "derived" / "damage_repair"),
                    "--sex-lookup",
                    "data/schema/harmonized/hrs_sex_lookup.csv",
                    "--out-root",
                    str(Path(args.figures_root) / "by_sex" / "damage_repair_calibration"),
                ],
                root,
                args.dry_run,
            )
            run_step(
                "Plot HRS Damage Risk Vs Age By Sex",
                [
                    sys.executable,
                    "scripts/visualization/12_damage_risk_vs_age_by_sex.py",
                    "--run-tag",
                    args.run_tag,
                    "--dataset-label",
                    "HRS",
                    "--events-root",
                    str(Path(args.analysis_root) / "derived" / "damage_repair"),
                    "--transitions-root",
                    args.transitions_root,
                    "--sex-lookup",
                    "data/schema/harmonized/hrs_sex_lookup.csv",
                    "--out-root",
                    str(Path(args.figures_root) / "by_sex" / "damage_risk_vs_age"),
                ],
                root,
                args.dry_run,
            )

    if not args.skip_metric_summary_plots:
        metric_summary_cmd = [
            sys.executable,
            "scripts/visualization/06_metrics_by_target_plots.py",
            "--run-tag",
            args.run_tag,
            "--dataset-label",
            "HRS",
            "--metrics-root",
            args.metrics_root,
            "--out-root",
            str(Path(args.figures_root) / "metrics_by_target"),
        ]
        if args.include_damage_repair:
            metric_summary_cmd.append("--include-damage-repair")
        run_step("Plot HRS Target-Level Metric Summaries", metric_summary_cmd, root, args.dry_run)


if __name__ == "__main__":
    main()
