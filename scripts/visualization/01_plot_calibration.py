#!/usr/bin/env python3
"""
scripts/visualization/01_plot_calibration.py

Make calibration plots using already-computed OOF probabilities.

Inputs:
- Transition truth files:
    Transition_data/01_transitions/<SCENARIO>_<METHOD>/S<t>_S<tp1>.csv
  which contain Y columns:  Y_S<tp1>_<TARGET>

- OOF probability files:
    analysis/lr/<SCENARIO>_<METHOD>/probs_S<t>_S<tp1>.csv
    analysis/dnn/<SCENARIO>_<METHOD>/probs_S<t>_S<tp1>.csv
  which contain P columns:  P_S<tp1>_<TARGET>

Outputs:
- PNG plots under:
    analysis/figures/calibration/<SCENARIO>_<METHOD>/S<t>_S<tp1>/<TARGET>.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from paper_style import apply_paper_style

apply_paper_style()

EPS = 1e-6
DEFAULT_CI_Z = 1.96

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


def infer_transition(fp: Path) -> tuple[int, int] | None:
    m = re.match(r"^S(\d+)_S(\d+)\.csv$", fp.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_targets(kind: str) -> list[str]:
    kind = kind.lower()
    if kind == "all":
        return ALL_TARGETS
    if kind == "adl":
        return ADL_TARGETS
    if kind == "disease":
        return DISEASE_TARGETS
    raise ValueError(f"Unknown targets-kind: {kind}")


def safe_binary(y: np.ndarray) -> bool:
    y = y[~np.isnan(y)]
    u = np.unique(y)
    return set(u.tolist()).issubset({0, 1})


def load_truth_and_probs(
    trans_fp: Path,
    probs_fp: Path,
    tp1: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns aligned (y, p) for one target.
    Join by idauniq if possible, else require same length.
    """
    df_t = pd.read_csv(trans_fp)
    df_p = pd.read_csv(probs_fp)

    y_col = f"Y_S{tp1}_{target}"
    p_col = f"P_S{tp1}_{target}"

    if y_col not in df_t.columns or p_col not in df_p.columns:
        return None

    # join on idauniq when possible
    if "idauniq" in df_t.columns and "idauniq" in df_p.columns:
        sub = df_t[["idauniq", y_col]].merge(df_p[["idauniq", p_col]], on="idauniq", how="inner")
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy()
        p = pd.to_numeric(sub[p_col], errors="coerce").to_numpy()
    else:
        if len(df_t) != len(df_p):
            return None
        y = pd.to_numeric(df_t[y_col], errors="coerce").to_numpy()
        p = pd.to_numeric(df_p[p_col], errors="coerce").to_numpy()

    mask = (~np.isnan(y)) & (~np.isnan(p))
    y = y[mask]
    p = p[mask]

    if y.size == 0:
        return None
    if not safe_binary(y):
        return None

    y = y.astype(int)
    if np.unique(y).size < 2:
        return None

    p = np.clip(p.astype(float), EPS, 1 - EPS)
    return y, p


def load_truth_and_probs_pooled(
    trans_fp: Path,
    probs_fp: Path,
    tp1: int,
    targets: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Micro-average pooled calibration:
    Concatenate all valid (y, p) pairs across `targets` into one long vector.

    This reads both CSVs once and joins by idauniq when possible.
    """
    df_t = pd.read_csv(trans_fp)
    df_p = pd.read_csv(probs_fp)

    # join on idauniq when possible
    if "idauniq" in df_t.columns and "idauniq" in df_p.columns:
        df = df_t.merge(df_p, on="idauniq", how="inner")
    else:
        if len(df_t) != len(df_p):
            return None
        df = pd.concat([df_t.reset_index(drop=True), df_p.reset_index(drop=True)], axis=1)

    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []

    for target in targets:
        y_col = f"Y_S{tp1}_{target}"
        p_col = f"P_S{tp1}_{target}"
        if y_col not in df.columns or p_col not in df.columns:
            continue

        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()

        mask = (~np.isnan(y)) & (~np.isnan(p))
        y = y[mask]
        p = p[mask]

        if y.size == 0:
            continue
        if not safe_binary(y):
            continue

        y = y.astype(int)
        if np.unique(y).size < 2:
            # degenerate target; skip it for pooling
            continue

        p = np.clip(p.astype(float), EPS, 1 - EPS)

        ys.append(y)
        ps.append(p)

    if not ys:
        return None

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(ps, axis=0)

    if y_all.size == 0 or np.unique(y_all).size < 2:
        return None

    return y_all.astype(int), p_all.astype(float)


def quantile_edges(p: np.ndarray, n_bins: int, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    edges[0] = max(edges[0], eps)
    edges[-1] = min(edges[-1], 1.0)
    return edges


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = DEFAULT_CI_Z) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    phat = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    z2 = z ** 2
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
        / denom
    )
    lower = np.clip(center - half, 0.0, 1.0)
    upper = np.clip(center + half, 0.0, 1.0)
    return lower, upper


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int, z: float = DEFAULT_CI_Z) -> dict[str, np.ndarray]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    edges = quantile_edges(p, n_bins=n_bins, eps=EPS)

    rows: list[tuple[float, float, float, float, int]] = []
    for k in range(n_bins):
        left, right = edges[k], edges[k + 1]
        if k < n_bins - 1:
            mask = (p >= left) & (p < right)
        else:
            mask = (p >= left) & (p <= right)
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = p[mask]
        yk = y[mask]
        mean_pred = float(pk.mean())
        obs_rate = float(yk.mean())
        low, high = wilson_interval(np.array([yk.sum()]), np.array([nk]), z=z)
        rows.append((mean_pred, obs_rate, float(low[0]), float(high[0]), nk))

    if not rows:
        empty = np.array([], dtype=float)
        return {"x": empty, "y": empty, "y_low": empty, "y_high": empty, "n": empty}

    arr = np.asarray(rows, dtype=float)
    return {
        "x": arr[:, 0],
        "y": arr[:, 1],
        "y_low": arr[:, 2],
        "y_high": arr[:, 3],
        "n": arr[:, 4],
    }


def maybe_log_axes(loglog: bool):
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1e-3, 1)
        plt.ylim(1e-3, 1)


def plot_target(
    out_fp: Path,
    title: str,
    curves: list[tuple[str, dict[str, np.ndarray]]],
    loglog: bool,
):
    """
    curves: list of (label, calibration_table_dict)
    """
    ensure_dir(out_fp.parent)
    plt.figure()

    # perfect calibration reference
    plt.plot([EPS, 1 - EPS], [EPS, 1 - EPS], linestyle="--", linewidth=1)

    for label, tab in curves:
        x = np.asarray(tab["x"], dtype=float)
        y = np.asarray(tab["y"], dtype=float)
        y_low = np.asarray(tab["y_low"], dtype=float)
        y_high = np.asarray(tab["y_high"], dtype=float)
        if loglog:
            x = np.clip(x, EPS, 1 - EPS)
            y = np.clip(y, EPS, 1 - EPS)
            y_low = np.clip(y_low, EPS, 1 - EPS)
            y_high = np.clip(y_high, EPS, 1 - EPS)
        yerr = np.vstack([
            np.maximum(y - y_low, 0.0),
            np.maximum(y_high - y, 0.0),
        ])
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=5,
            linewidth=1,
            elinewidth=1,
            capsize=2,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    maybe_log_axes(loglog)
    #plt.grid(alpha=0.3)
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_fp, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")

    # plot LR only, DNN only, or both
    ap.add_argument("--which", default="both", choices=["lr", "dnn", "both"])

    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--analysis-root", default="analysis")
    ap.add_argument("--out-root", default="analysis/figures/calibration")

    ap.add_argument("--transition", default="all", help="e.g. S1_S2 or 'all'")
    ap.add_argument("--targets-kind", default="all", choices=["all", "adl", "disease"])
    ap.add_argument("--target", default="all", help="e.g. DIABE or 'all'")

    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--loglog", action="store_true")
    ap.add_argument("--min-n", type=int, default=200)

    ap.add_argument(
        "--pooled",
        default="none",
        choices=["none", "disease", "adl", "all", "three"],
        help="Also write pooled calibration plots by concatenating (y,p) across targets: "
             "disease, adl, all, or three (all three).",
    )

    args = ap.parse_args()

    run_key = resolve_run_tag(args.run_tag, args.scenario, args.method)
    trans_dir = Path(args.transitions_dir) / run_key
    if not trans_dir.exists():
        raise FileNotFoundError(f"Transitions folder not found: {trans_dir}")

    # input probs dirs
    lr_dir = Path(args.analysis_root) / "lr" / run_key
    dnn_dir = Path(args.analysis_root) / "dnn" / run_key

    if args.which in ("lr", "both") and not lr_dir.exists():
        raise FileNotFoundError(f"LR probs folder not found: {lr_dir}")
    if args.which in ("dnn", "both") and not dnn_dir.exists():
        raise FileNotFoundError(f"DNN probs folder not found: {dnn_dir}")

    out_dir = Path(args.out_root) / run_key
    ensure_dir(out_dir)

    trans_files = sorted([p for p in trans_dir.glob("S*_S*.csv") if infer_transition(p)])
    if args.transition.lower() != "all":
        trans_files = [p for p in trans_files if p.stem == args.transition]
    if not trans_files:
        raise FileNotFoundError("No transition files matched your selection.")

    targets = pick_targets(args.targets_kind)
    if args.target.lower() != "all":
        targets = [t for t in targets if t.upper() == args.target.upper()]
        if not targets:
            raise ValueError(f"Target {args.target} not found in targets list.")

    print(f"[OK] run={run_key} | which={args.which} | loglog={args.loglog} | bins={args.n_bins}")
    print(f"[OK] transitions={len(trans_files)} | targets={len(targets)}")
    print(f"[OK] output: {out_dir}")
    print(f"[OK] pooled={args.pooled}")

    plotted = 0
    skipped = 0

    for trans_fp in trans_files:
        t, tp1 = infer_transition(trans_fp)  # type: ignore[misc]
        trans_name = trans_fp.stem
        print(f"\n=== {trans_name} ===")

        lr_probs_fp = lr_dir / f"probs_{trans_name}.csv"
        dnn_probs_fp = dnn_dir / f"probs_{trans_name}.csv"

        # --- pooled plots (micro-average across targets) ---
        pooled_specs: list[tuple[str, list[str]]] = []
        if args.pooled == "disease":
            pooled_specs = [("POOLED_disease", DISEASE_TARGETS)]
        elif args.pooled == "adl":
            pooled_specs = [("POOLED_adl", ADL_TARGETS)]
        elif args.pooled == "all":
            pooled_specs = [("POOLED_all", ALL_TARGETS)]
        elif args.pooled == "three":
            pooled_specs = [
                ("POOLED_disease", DISEASE_TARGETS),
                ("POOLED_adl", ADL_TARGETS),
                ("POOLED_all", ALL_TARGETS),
            ]

        for pooled_name, pooled_targets in pooled_specs:
            curves = []

            if args.which in ("lr", "both") and lr_probs_fp.exists():
                res = load_truth_and_probs_pooled(trans_fp, lr_probs_fp, tp1=tp1, targets=pooled_targets)
                if res is not None:
                    y_pool, p_pool = res
                    if len(y_pool) >= args.min_n:
                        curves.append(("LR", calibration_table(y_pool, p_pool, n_bins=args.n_bins)))

            if args.which in ("dnn", "both") and dnn_probs_fp.exists():
                res = load_truth_and_probs_pooled(trans_fp, dnn_probs_fp, tp1=tp1, targets=pooled_targets)
                if res is not None:
                    y_pool, p_pool = res
                    if len(y_pool) >= args.min_n:
                        curves.append(("DNN", calibration_table(y_pool, p_pool, n_bins=args.n_bins)))

            if curves:
                prev = float(y_pool.mean())
                n = int(len(y_pool))
                title = f"{run_key} | {trans_name} | {pooled_name} (n={n}, prev={prev:.3f})"
                out_fp = out_dir / trans_name / f"{pooled_name}.png"
                plot_target(out_fp, title, curves, loglog=args.loglog)
                plotted += 1

        for target in targets:
            curves = []

            if args.which in ("lr", "both") and lr_probs_fp.exists():
                res = load_truth_and_probs(trans_fp, lr_probs_fp, tp1=tp1, target=target)
                if res is not None:
                    y, p = res
                    if len(y) >= args.min_n:
                        curves.append(("LR", calibration_table(y, p, n_bins=args.n_bins)))

            if args.which in ("dnn", "both") and dnn_probs_fp.exists():
                res = load_truth_and_probs(trans_fp, dnn_probs_fp, tp1=tp1, target=target)
                if res is not None:
                    y, p = res
                    if len(y) >= args.min_n:
                        curves.append(("DNN", calibration_table(y, p, n_bins=args.n_bins)))

            if not curves:
                skipped += 1
                continue

            # use y from last successful load to show prevalence/n
            prev = float(y.mean())
            n = int(len(y))

            title = f"{run_key} | {trans_name} | {target} (n={n}, prev={prev:.3f})"
            out_fp = out_dir / trans_name / f"{target}.png"
            plot_target(out_fp, title, curves, loglog=args.loglog)

            plotted += 1

        print(f"[OK] {trans_name}: plotted so far={plotted} | skipped so far={skipped}")

    print(f"\n[DONE] Plotted={plotted} | Skipped={skipped}")
    print(f"[DONE] Output root: {out_dir}")


if __name__ == "__main__":
    main()
