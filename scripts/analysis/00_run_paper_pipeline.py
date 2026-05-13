#!/usr/bin/env python3
"""Run the canonical core-paper pipeline from imputed wave tables.

This entrypoint intentionally excludes the Kesten / power-law / risk-distribution
branches. It only runs the workflow that is relevant to the current paper:

1. build transition tables
2. fit LR and DNN models
3. evaluate metrics
4. build hazard-ratio tables
5. run feature-selection sweep
6. create paper-facing figures

Optional:
- damage/repair tables + calibration plots
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_step(label: str, cmd: list[str], root: Path, dry_run: bool) -> None:
    pretty = " ".join(cmd)
    print(f"\n=== {label} ===")
    print(pretty)
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
    ap.add_argument("--scenario", required=True, choices=["MAR", "MNAR"])
    ap.add_argument("--method", required=True, choices=["Cart", "Pmm"])

    ap.add_argument("--data-root", default="data/Imputed_data/Core/Imputed")
    ap.add_argument("--transitions-root", default="Transition_data/01_transitions")
    ap.add_argument("--analysis-root", default="analysis/ELSA")
    ap.add_argument("--metrics-root", default="analysis/ELSA/metrics")
    ap.add_argument("--figures-root", default="analysis/ELSA/figures")
    ap.add_argument("--derived-root", default="analysis/ELSA/derived")
    ap.add_argument("--feature-selection-root", default="analysis/ELSA/feature_selection/lr_filter")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--include-age", action="store_true")

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

    ap.add_argument("--skip-transitions", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--skip-hazard", action="store_true")
    ap.add_argument("--skip-feature-selection", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--skip-metric-summary-plots", action="store_true")
    ap.add_argument("--include-damage-repair", action="store_true")
    ap.add_argument("--include-sex-stratified-analysis", action="store_true")
    ap.add_argument("--include-loglog-calibration", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    analysis_root = Path(args.analysis_root)
    metrics_root = Path(args.metrics_root)
    figures_root = Path(args.figures_root)
    derived_root = Path(args.derived_root)

    common = ["--scenario", args.scenario, "--method", args.method]
    model_common = ["--seed", str(args.seed)]
    if args.include_age:
        model_common.append("--include-age")

    if not args.skip_transitions:
        run_step(
            "Build Transitions",
            [
                sys.executable,
                "scripts/analysis/01_build_transitions.py",
                *common,
                "--data-root",
                args.data_root,
                "--out-dir",
                args.transitions_root,
            ],
            root,
            args.dry_run,
        )

    if not args.skip_models:
        run_step(
            "Fit LR",
            [
                sys.executable,
                "scripts/analysis/02_LR_oof_probs_and_metrics.py",
                *common,
                "--transitions-dir",
                args.transitions_root,
                "--out-dir",
                str(analysis_root / "lr"),
                "--cv-folds",
                str(args.cv_folds),
                "--max-iter",
                str(args.lr_max_iter),
                "--class-weight",
                args.lr_class_weight,
                "--export-probs",
                *model_common,
            ],
            root,
            args.dry_run,
        )
        dnn_cmd = [
            sys.executable,
            "scripts/analysis/04_dnn_oof_probs_and_metrics.py",
            *common,
            "--transitions-dir",
            args.transitions_root,
            "--out-dir",
            str(analysis_root / "dnn"),
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
            *model_common,
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
        run_step("Fit DNN", dnn_cmd, root, args.dry_run)

    if not args.skip_metrics:
        for model in ("LR", "DNN"):
            run_step(
                f"Evaluate {model}",
                [
                    sys.executable,
                    "scripts/analysis/03_evaluate_performance.py",
                    *common,
                    "--model",
                    model,
                    "--transitions-dir",
                    args.transitions_root,
                    "--probs-dir",
                    str(analysis_root),
                    "--out-dir",
                    str(metrics_root),
                ],
                root,
                args.dry_run,
            )

    if not args.skip_hazard:
        run_step(
            "Build Hazard Ratio Tables",
            [
                sys.executable,
                "scripts/analysis/05_build_hazard_ratio_tables.py",
                *common,
                "--models",
                "LR",
                "DNN",
                "--transitions-dir",
                args.transitions_root,
                "--analysis-root",
                str(analysis_root),
                "--out-root",
                str(metrics_root / "hazard_ratio"),
            ],
            root,
            args.dry_run,
        )

    if not args.skip_feature_selection:
        run_step(
            "Feature Selection Sweep",
            [
                sys.executable,
                "scripts/analysis/07_feature_selection_sweep.py",
                *common,
                "--transitions-dir",
                args.transitions_root,
                "--out-dir",
                args.feature_selection_root,
                "--seed",
                str(args.seed),
            ],
            root,
            args.dry_run,
        )

    if args.include_damage_repair:
        for model in ("LR", "DNN"):
            run_step(
                f"Build Damage Repair Events {model}",
                [
                    sys.executable,
                    "scripts/analysis/06_build_damage_repair_tables.py",
                    *common,
                    "--model",
                    model,
                    "--transitions-dir",
                    args.transitions_root,
                    "--probs-root",
                    str(analysis_root),
                    "--out-root",
                    str(derived_root / "damage_repair"),
                ],
                root,
                args.dry_run,
            )
            run_step(
                f"Evaluate Damage Repair {model}",
                [
                    sys.executable,
                    "scripts/analysis/13_evaluate_damage_repair.py",
                    *common,
                    "--model",
                    model,
                    "--events-root",
                    str(derived_root / "damage_repair"),
                    "--out-root",
                    str(metrics_root / "damage_repair"),
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot Damage Repair Calibration",
            [
                sys.executable,
                "scripts/visualization/05_damage_repair_calibration_plots.py",
                *common,
                "--models",
                "LR",
                "DNN",
                "--events-root",
                str(derived_root / "damage_repair"),
                "--out-root",
                str(figures_root / "damage_repair_calibration"),
            ],
            root,
            args.dry_run,
        )
        if args.include_loglog_calibration:
            run_step(
                "Plot Damage Repair Calibration (Log-Log)",
                [
                    sys.executable,
                    "scripts/visualization/05_damage_repair_calibration_plots.py",
                    *common,
                    "--models",
                    "LR",
                    "DNN",
                    "--events-root",
                    str(derived_root / "damage_repair"),
                    "--out-root",
                    str(figures_root / "damage_repair_calibration_loglog"),
                    "--loglog",
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot Damage Risk vs Age",
            [
                sys.executable,
                "scripts/visualization/07_damage_risk_vs_age_plots.py",
                *common,
                "--dataset-label",
                "ELSA",
                "--events-root",
                str(derived_root / "damage_repair"),
                "--transitions-root",
                args.transitions_root,
                "--out-root",
                str(figures_root / "damage_risk_vs_age"),
            ],
            root,
            args.dry_run,
        )
        run_step(
            "Plot Damage/Repair Transition Metrics",
            [
                sys.executable,
                "scripts/visualization/08_damage_repair_transition_metric_plots.py",
                *common,
                "--dataset-label",
                "ELSA",
                "--metrics-root",
                str(metrics_root / "damage_repair"),
                "--out-root",
                str(figures_root / "damage_repair_transition_metrics"),
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
                f"Evaluate {model} By Sex",
                [
                    sys.executable,
                    "scripts/analysis/18_evaluate_performance_by_sex.py",
                    *common,
                    "--model",
                    model,
                    "--transitions-dir",
                    args.transitions_root,
                    "--probs-dir",
                    str(analysis_root),
                    "--sex-lookup",
                    "data/schema/harmonized/elsa_sex_lookup.csv",
                    "--out-dir",
                    str(metrics_root / "by_sex"),
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot Standard Transition Metrics By Sex",
            [
                sys.executable,
                "scripts/visualization/09_transition_metric_plots_by_sex.py",
                *common,
                "--dataset-label",
                "ELSA",
                "--metrics-root",
                str(metrics_root / "by_sex"),
                "--out-root",
                str(figures_root / "by_sex" / "transition_metrics"),
                "--title-suffix",
                "Standard",
            ],
            root,
            args.dry_run,
        )
        run_step(
            "Plot Standard Calibration By Sex",
            [
                sys.executable,
                "scripts/visualization/10_plot_calibration_by_sex.py",
                *common,
                "--dataset-label",
                "ELSA",
                "--transitions-dir",
                args.transitions_root,
                "--analysis-root",
                str(analysis_root),
                "--sex-lookup",
                "data/schema/harmonized/elsa_sex_lookup.csv",
                "--out-root",
                str(figures_root / "by_sex" / "calibration"),
            ],
            root,
            args.dry_run,
        )
        if args.include_damage_repair:
            for model in ("LR", "DNN"):
                run_step(
                    f"Evaluate Damage Repair {model} By Sex",
                    [
                        sys.executable,
                        "scripts/analysis/19_evaluate_damage_repair_by_sex.py",
                        *common,
                        "--model",
                        model,
                        "--events-root",
                        str(derived_root / "damage_repair"),
                        "--sex-lookup",
                        "data/schema/harmonized/elsa_sex_lookup.csv",
                        "--out-root",
                        str(metrics_root / "by_sex" / "damage_repair"),
                    ],
                    root,
                    args.dry_run,
                )
            run_step(
                "Plot Damage Repair Transition Metrics By Sex",
                [
                    sys.executable,
                    "scripts/visualization/09_transition_metric_plots_by_sex.py",
                    *common,
                    "--dataset-label",
                    "ELSA",
                    "--metrics-root",
                    str(metrics_root / "by_sex" / "damage_repair"),
                    "--out-root",
                    str(figures_root / "by_sex" / "damage_repair_transition_metrics"),
                    "--event-type",
                    "damage",
                    "--title-suffix",
                    "Damage",
                ],
                root,
                args.dry_run,
            )
            run_step(
                "Plot Repair Transition Metrics By Sex",
                [
                    sys.executable,
                    "scripts/visualization/09_transition_metric_plots_by_sex.py",
                    *common,
                    "--dataset-label",
                    "ELSA",
                    "--metrics-root",
                    str(metrics_root / "by_sex" / "damage_repair"),
                    "--out-root",
                    str(figures_root / "by_sex" / "damage_repair_transition_metrics"),
                    "--event-type",
                    "repair",
                    "--title-suffix",
                    "Repair",
                ],
                root,
                args.dry_run,
            )
            run_step(
                "Plot Damage Repair Calibration By Sex",
                [
                    sys.executable,
                    "scripts/visualization/11_plot_damage_repair_calibration_by_sex.py",
                    *common,
                    "--dataset-label",
                    "ELSA",
                    "--events-root",
                    str(derived_root / "damage_repair"),
                    "--sex-lookup",
                    "data/schema/harmonized/elsa_sex_lookup.csv",
                    "--out-root",
                    str(figures_root / "by_sex" / "damage_repair_calibration"),
                ],
                root,
                args.dry_run,
            )
            run_step(
                "Plot Damage Risk Vs Age By Sex",
                [
                    sys.executable,
                    "scripts/visualization/12_damage_risk_vs_age_by_sex.py",
                    *common,
                    "--dataset-label",
                    "ELSA",
                    "--events-root",
                    str(derived_root / "damage_repair"),
                    "--transitions-root",
                    args.transitions_root,
                    "--sex-lookup",
                    "data/schema/harmonized/elsa_sex_lookup.csv",
                    "--out-root",
                    str(figures_root / "by_sex" / "damage_risk_vs_age"),
                ],
                root,
                args.dry_run,
            )

    if not args.skip_figures:
        run_step(
            "Plot Calibration",
            [
                sys.executable,
                "scripts/visualization/01_plot_calibration.py",
                *common,
                "--which",
                "both",
                "--transitions-dir",
                args.transitions_root,
                "--analysis-root",
                str(analysis_root),
                "--out-root",
                str(figures_root / "calibration"),
                "--pooled",
                "three",
            ],
            root,
            args.dry_run,
        )
        if args.include_loglog_calibration:
            run_step(
                "Plot Calibration (Log-Log)",
                [
                    sys.executable,
                    "scripts/visualization/01_plot_calibration.py",
                    *common,
                    "--which",
                    "both",
                    "--transitions-dir",
                    args.transitions_root,
                    "--analysis-root",
                    str(analysis_root),
                    "--out-root",
                    str(figures_root / "calibration_loglog"),
                    "--pooled",
                    "three",
                    "--loglog",
                ],
                root,
                args.dry_run,
            )
        run_step(
            "Plot ECE",
            [
                sys.executable,
                "scripts/visualization/02_ece_plots.py",
                *common,
                "--metrics-root",
                str(metrics_root),
                "--out-dir",
                str(figures_root / "ece_plots"),
            ],
            root,
            args.dry_run,
        )
        run_step(
            "Plot Hazard Ratio",
            [
                sys.executable,
                "scripts/visualization/03_hazard_ratio_plots.py",
                *common,
                "--models",
                "LR",
                "DNN",
                "--hr-root",
                str(metrics_root / "hazard_ratio"),
                "--out-root",
                str(figures_root / "hazard_ratio"),
            ],
            root,
            args.dry_run,
        )
        if not args.skip_metric_summary_plots:
            metric_summary_cmd = [
                sys.executable,
                "scripts/visualization/06_metrics_by_target_plots.py",
                *common,
                "--dataset-label",
                "ELSA",
                "--metrics-root",
                str(metrics_root),
                "--out-root",
                str(figures_root / "metrics_by_target"),
            ]
            if args.include_damage_repair:
                metric_summary_cmd.append("--include-damage-repair")
            run_step("Plot Target-Level Metric Summaries", metric_summary_cmd, root, args.dry_run)

    print("\n[DONE] Canonical paper pipeline complete.")


if __name__ == "__main__":
    main()
