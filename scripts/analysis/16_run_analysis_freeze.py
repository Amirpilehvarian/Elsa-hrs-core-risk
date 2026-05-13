#!/usr/bin/env python3
"""Run the frozen final analysis stack for the paper.

This orchestrates:
1. ELSA full analysis
2. ELSA shared-subset validation
3. HRS shared-subset validation
4. Manuscript asset rebuild

The DNN settings are frozen here so the manuscript can cite one canonical run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


FROZEN_SPEC = {
    "lr": {
        "class_weight": "none",
        "max_iter": 2000,
        "cv_folds": 5,
        "seed": 42,
    },
    "dnn": {
        "hidden_sizes": "512",
        "epochs": 30,
        "batch": 64,
        "lr": 1e-3,
        "l2": 0.0,
        "dropout": 0.0,
        "patience": 5,
        "activation": "relu",
        "batchnorm": False,
        "standardize_x": True,
        "reduce_lr_on_plateau": True,
        "min_lr": 1e-5,
        "seed": 42,
        "selection_note": "Chosen from the focused ELSA S4_S5/S6_S7 sweep in analysis/dnn_tuning/sweep_summary_round2.csv.",
    },
}


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


def build_dnn_args() -> list[str]:
    spec = FROZEN_SPEC["dnn"]
    args = [
        "--dnn-hidden-sizes",
        str(spec["hidden_sizes"]),
        "--dnn-epochs",
        str(spec["epochs"]),
        "--dnn-batch",
        str(spec["batch"]),
        "--dnn-lr",
        str(spec["lr"]),
        "--dnn-l2",
        str(spec["l2"]),
        "--dnn-dropout",
        str(spec["dropout"]),
        "--dnn-patience",
        str(spec["patience"]),
        "--dnn-activation",
        str(spec["activation"]),
        "--dnn-min-lr",
        str(spec["min_lr"]),
    ]
    if spec["batchnorm"]:
        args.append("--dnn-batchnorm")
    if spec["standardize_x"]:
        args.append("--dnn-standardize-x")
    if spec["reduce_lr_on_plateau"]:
        args.append("--dnn-reduce-lr-on-plateau")
    return args


def write_frozen_spec(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_analysis_spec.json").write_text(
        json.dumps(FROZEN_SPEC, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser(description="Run the frozen final paper analysis and rebuild manuscript assets.")
    ap.add_argument("--scenario", default="MAR", choices=["MAR", "MNAR"])
    ap.add_argument("--method", default="Cart", choices=["Cart", "Pmm"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-elsa-full", action="store_true")
    ap.add_argument("--skip-elsa-shared", action="store_true")
    ap.add_argument("--skip-hrs-shared", action="store_true")
    ap.add_argument("--skip-manuscript-assets", action="store_true")
    args = ap.parse_args()

    lr = FROZEN_SPEC["lr"]
    dnn_args = build_dnn_args()
    write_frozen_spec(root / "manuscript" / "assets")
    elsa_run_tag = f"{args.scenario}_{args.method}"
    elsa_shared_source_dir = f"Transition_data/01_transitions/{elsa_run_tag}"

    if not args.skip_elsa_full:
        run_step(
            "ELSA Full Final Freeze",
            [
                sys.executable,
                "scripts/analysis/00_run_paper_pipeline.py",
                "--scenario",
                args.scenario,
                "--method",
                args.method,
                "--cv-folds",
                str(lr["cv_folds"]),
                "--seed",
                str(lr["seed"]),
                "--lr-max-iter",
                str(lr["max_iter"]),
                "--lr-class-weight",
                str(lr["class_weight"]),
                "--include-damage-repair",
                "--include-loglog-calibration",
                *dnn_args,
            ],
            root,
            args.dry_run,
        )

    if not args.skip_elsa_shared:
        run_step(
            "ELSA Shared Final Freeze",
            [
                sys.executable,
                "scripts/analysis/15_run_elsa_shared_pipeline.py",
                "--source-dir",
                elsa_shared_source_dir,
                "--cv-folds",
                str(lr["cv_folds"]),
                "--seed",
                str(lr["seed"]),
                "--lr-max-iter",
                str(lr["max_iter"]),
                "--lr-class-weight",
                str(lr["class_weight"]),
                "--include-damage-repair",
                "--include-loglog-calibration",
                *dnn_args,
            ],
            root,
            args.dry_run,
        )

    if not args.skip_hrs_shared:
        run_step(
            "HRS Shared Final Freeze",
            [
                sys.executable,
                "scripts/analysis/10_run_hrs_shared_pipeline.py",
                "--cv-folds",
                str(lr["cv_folds"]),
                "--seed",
                str(lr["seed"]),
                "--lr-max-iter",
                str(lr["max_iter"]),
                "--lr-class-weight",
                str(lr["class_weight"]),
                "--include-damage-repair",
                "--include-loglog-calibration",
                *dnn_args,
            ],
            root,
            args.dry_run,
        )

    if not args.skip_manuscript_assets:
        run_step(
            "Build Manuscript Assets",
            [
                sys.executable,
                "scripts/analysis/12_build_manuscript_assets.py",
                "--elsa-full-run",
                elsa_run_tag,
                ],
            root,
            args.dry_run,
        )

    print("\n[DONE] Frozen analysis stack complete.")


if __name__ == "__main__":
    main()
