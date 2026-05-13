#!/usr/bin/env python3
"""Build LR coefficient summaries and a supplemental beta heatmap.

The main LR pipeline exports out-of-fold probabilities, not coefficients. This
script refits the same transparent LR specification on each ELSA transition and
target, then averages standardized coefficients across transitions. The output is
descriptive feature-importance support for the LR reference model, not a
replacement for the OOF performance analysis.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ADL_TARGETS = [
    "DANGERA", "EATA", "MEDSA", "COMMUNA", "PHONEA", "MONEYA", "WALKRA", "TOILTA", "MEALSA", "MAPA",
    "BEDA", "DIMEA", "SHOPA", "BATHA", "ARMSA", "DRESSA", "WALK100A", "SITA", "CLIM1A", "HOUSEWKA",
    "PUSHA", "LIFTA", "CHAIRA", "CLIMSA", "STOOPA",
]
DISEASE_TARGETS = [
    "PARKINE", "CONHRTFE", "HIPE", "HEARTE", "HRTMRE", "HRTATTE", "STROKE", "LUNGE", "CANCRE",
    "ANGINE", "OSTEOE", "HRTRHME", "PSYCHE", "DIABE", "ASTHMAE", "CATRACTE", "HCHOLE", "ARTHRE", "HIBPE",
]
ALL_TARGETS = ADL_TARGETS + DISEASE_TARGETS


def infer_transition_from_filename(fp: Path) -> tuple[int, int] | None:
    match = re.match(r"^S(\d+)_S(\d+)\.csv$", fp.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def canonical_predictor(col: str, t: int) -> str:
    prefix = f"S{t}"
    return col[len(prefix):] if col.startswith(prefix) else col


def target_type(target: str) -> str:
    if target in ADL_TARGETS:
        return "ADL"
    if target in DISEASE_TARGETS:
        return "Disease"
    return "Other"


def build_lr(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=max_iter, solver="lbfgs", class_weight=None)),
        ]
    )


def fit_coefficients(transitions_dir: Path, max_iter: int) -> pd.DataFrame:
    rows: list[dict] = []
    files = sorted(p for p in transitions_dir.glob("S*_S*.csv") if infer_transition_from_filename(p) is not None)
    if not files:
        raise FileNotFoundError(f"No transition CSV files found in {transitions_dir}")

    for fp in files:
        t, tp1 = infer_transition_from_filename(fp)  # type: ignore[misc]
        transition = f"S{t}_S{tp1}"
        df = pd.read_csv(fp)
        x_cols = [c for c in df.columns if c.startswith(f"S{t}")]
        y_cols = [f"Y_S{tp1}_{target}" for target in ALL_TARGETS if f"Y_S{tp1}_{target}" in df.columns]
        x_num = df[x_cols].apply(pd.to_numeric, errors="coerce")

        for y_col in y_cols:
            target = y_col.replace(f"Y_S{tp1}_", "")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = x_num.notna().all(axis=1) & y.notna()
            if int(mask.sum()) < 50:
                continue
            y_arr = y.loc[mask].astype(int).to_numpy()
            if np.unique(y_arr).size < 2:
                continue

            model = build_lr(max_iter=max_iter)
            model.fit(x_num.loc[mask, x_cols].to_numpy(dtype=float), y_arr)
            coef = model.named_steps["clf"].coef_.ravel()

            for x_col, beta in zip(x_cols, coef):
                rows.append(
                    {
                        "transition": transition,
                        "t": t,
                        "tp1": tp1,
                        "target": target,
                        "target_type": target_type(target),
                        "predictor": canonical_predictor(x_col, t),
                        "beta_standardized": float(beta),
                        "n_obs": int(mask.sum()),
                        "prevalence": float(y.loc[mask].mean()),
                    }
                )

    if not rows:
        raise RuntimeError("No LR coefficients were estimated.")
    return pd.DataFrame(rows)


def aggregate_coefficients(coefs: pd.DataFrame) -> pd.DataFrame:
    return (
        coefs.groupby(["target", "target_type", "predictor"], as_index=False)
        .agg(
            mean_beta=("beta_standardized", "mean"),
            median_beta=("beta_standardized", "median"),
            mean_abs_beta=("beta_standardized", lambda x: float(np.mean(np.abs(x)))),
            n_transitions=("transition", "nunique"),
            mean_n_obs=("n_obs", "mean"),
            mean_prevalence=("prevalence", "mean"),
        )
        .sort_values(["target_type", "target", "mean_abs_beta"], ascending=[True, True, False])
    )


def plot_heatmap(agg: pd.DataFrame, out_fp: Path, top_n_predictors: int) -> None:
    top_predictors = (
        agg.groupby("predictor")["mean_abs_beta"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_predictors)
        .index.tolist()
    )
    target_order = ADL_TARGETS + DISEASE_TARGETS
    mat = (
        agg[agg["predictor"].isin(top_predictors)]
        .pivot_table(index="target", columns="predictor", values="mean_beta", aggfunc="mean")
        .reindex(target_order)
        .dropna(how="all")
        .loc[:, top_predictors]
    )
    mat = mat.fillna(0.0)

    vmax = float(np.nanpercentile(np.abs(mat.to_numpy()), 97))
    vmax = max(vmax, 0.25)

    fig_w = max(14, 0.42 * len(top_predictors) + 4)
    fig_h = max(10, 0.24 * len(mat.index) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_xlabel("Predictor", fontsize=13)
    ax.set_ylabel("Next-wave target", fontsize=13)
    ax.tick_params(length=0)

    adl_count = sum(target in ADL_TARGETS for target in mat.index)
    if 0 < adl_count < len(mat.index):
        ax.axhline(adl_count - 0.5, color="black", linewidth=1.0)
        ax.text(
            -0.045,
            1.0 - (adl_count / 2) / len(mat.index),
            "ADL",
            transform=ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            -0.045,
            1.0 - (adl_count + (len(mat.index) - adl_count) / 2) / len(mat.index),
            "Disease",
            transform=ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean standardized LR beta", fontsize=12)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="MNAR_Pmm")
    parser.add_argument("--transitions-root", default="Transition_data/01_transitions")
    parser.add_argument("--out-dir", default="analysis/ELSA/lr_coefficients")
    parser.add_argument("--figures-dir", default="analysis/ELSA/figures/lr_coefficients")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--top-n-predictors", type=int, default=35)
    args = parser.parse_args()

    transitions_dir = Path(args.transitions_root) / args.run_tag
    out_dir = Path(args.out_dir) / args.run_tag
    figures_dir = Path(args.figures_dir) / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    coefs = fit_coefficients(transitions_dir, max_iter=args.max_iter)
    agg = aggregate_coefficients(coefs)

    coefs_fp = out_dir / "lr_standardized_coefficients_by_transition.csv"
    agg_fp = out_dir / "lr_standardized_coefficients_mean.csv"
    fig_fp = figures_dir / "lr_beta_heatmap_top_predictors.png"
    coefs.to_csv(coefs_fp, index=False)
    agg.to_csv(agg_fp, index=False)
    plot_heatmap(agg, fig_fp, top_n_predictors=args.top_n_predictors)

    print(f"[OK] Wrote transition coefficients: {coefs_fp}")
    print(f"[OK] Wrote mean coefficients: {agg_fp}")
    print(f"[OK] Wrote coefficient heatmap: {fig_fp}")


if __name__ == "__main__":
    main()
