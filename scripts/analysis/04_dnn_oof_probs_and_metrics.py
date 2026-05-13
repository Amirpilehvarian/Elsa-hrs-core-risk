#!/usr/bin/env python3
"""scripts/analysis/04_dnn_oof_probs_and_metrics.py

Multi-output DNN (single model per transition) with OOF probabilities + metrics.

What this does
--------------
For each wave transition file produced by `scripts/analysis/01_build_transitions.py` (e.g. `S1_S2.csv`):

1) Builds ONE DNN that predicts ALL available targets simultaneously (multi-label).
2) Uses K-fold CV to create out-of-fold (OOF) probabilities for every row.
3) Saves a wide probability CSV per transition:
      analysis/dnn/<SCENARIO>_<METHOD>/probs_S1_S2.csv
   containing `idauniq` (or row_index fallback) and one probability column per target.
4) Saves metrics:
      metrics_by_target.csv  (one row per transition-target)
      metrics_summary.csv    (aggregated by transition + type)

Key design points
-----------------
- Multi-output DNN with a vector output layer (shape = n_targets).
- We use complete-case rows (drop any row with missing predictors OR missing targets for that transition).
- Unweighted binary cross-entropy (no class weighting) to match the writer/Jupyter baseline.
- OOF predictions are the right input for calibration/HR plots because each row's
  prediction comes from a model that did NOT train on that row.

Notes
-----
- This script expects targets in the transition files named like:
      Y_S<tp1>_<TARGET>
  and predictors as columns starting with S<t> (e.g. S1... for S1->S2).
- If you later want to match the "writer" notebook exactly (separate inputs for
  bin/cont/nurse), do that in a dedicated notebook; for the pipeline we keep one
  combined feature matrix per transition file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Canonical target lists (paper)
# -----------------------------
ADL_TARGETS = [
    "DANGERA","EATA","MEDSA","COMMUNA","PHONEA","MONEYA","WALKRA","TOILTA","MEALSA","MAPA",
    "BEDA","DIMEA","SHOPA","BATHA","ARMSA","DRESSA","WALK100A","SITA","CLIM1A","HOUSEWKA",
    "PUSHA","LIFTA","CHAIRA","CLIMSA","STOOPA",
]

DISEASE_TARGETS = [
    "PARKINE","CONHRTFE","HIPE","HEARTE","HRTMRE","HRTATTE","STROKE","LUNGE","CANCRE",
    "ANGINE","OSTEOE","HRTRHME","PSYCHE","DIABE","ASTHMAE","CATRACTE","HCHOLE","ARTHRE","HIBPE",
]

ALL_TARGETS = ADL_TARGETS + DISEASE_TARGETS


# Numerical stability when clipping probabilities
EPS = 1e-6


# -----------------------------
# Small helpers
# -----------------------------

def infer_transition_from_filename(fp: Path) -> tuple[int, int] | None:
    """Parse (t, tp1) from a filename like S1_S2.csv."""
    m = re.match(r"^S(\d+)_S(\d+)\.csv$", fp.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def parse_hidden_sizes(hidden: int, hidden2: int, hidden_sizes: str | None) -> list[int]:
    if hidden_sizes:
        vals = [int(v.strip()) for v in str(hidden_sizes).split(",") if v.strip()]
        vals = [v for v in vals if v > 0]
        if not vals:
            raise ValueError("--hidden-sizes must contain at least one positive integer")
        return vals
    sizes = [int(hidden)]
    if int(hidden2) > 0:
        sizes.append(int(hidden2))
    return sizes


def is_binary_array(y: np.ndarray) -> bool:
    """Return True if y only contains {0,1} (ignoring NaNs)."""
    y = y[~np.isnan(y)]
    if y.size == 0:
        return False
    u = np.unique(y)
    return set(u.tolist()).issubset({0, 1})


def make_stratify_label(Y: np.ndarray) -> np.ndarray | None:
    """Approximate stratification label for multi-output.

    We use: label = 1 if the row has any positive across targets (NaNs treated as 0).
    If label degenerates to a single class, we return None and fall back to KFold.
    """
    Yn = np.nan_to_num(Y, nan=0.0)
    lab = (Yn.sum(axis=1) > 0).astype(int)
    return None if np.unique(lab).size < 2 else lab


def safe_n_splits(label: np.ndarray | None, requested: int) -> int:
    """Pick a feasible number of splits for stratified CV."""
    if label is None:
        return max(2, int(requested))
    counts = np.bincount(label.astype(int), minlength=2)
    min_count = int(counts.min())
    n_splits = min(int(requested), min_count)
    return n_splits if n_splits >= 2 else 0


def build_sample_weight_matrix(Y: np.ndarray) -> np.ndarray:
    """Per-sample, per-target weight matrix.

    - Missing targets (NaN) -> weight 0 (ignored)
    - Observed targets -> balanced inverse-frequency weights per target

    Returns
    -------
    W : np.ndarray of shape (n_samples, n_targets)
    """
    n, m = Y.shape
    W = np.zeros((n, m), dtype=np.float32)

    for j in range(m):
        yj = Y[:, j]
        obs = ~np.isnan(yj)
        if obs.sum() == 0:
            continue

        yv = yj[obs].astype(int)

        # If only one class exists, do NOT balance; just keep weight=1 for observed.
        if np.unique(yv).size < 2:
            W[obs, j] = 1.0
            continue

        n0 = int((yv == 0).sum())
        n1 = int((yv == 1).sum())
        tot = n0 + n1
        w0 = tot / (2.0 * max(n0, 1))
        w1 = tot / (2.0 * max(n1, 1))

        W[obs, j] = np.where(yj[obs].astype(int) == 1, w1, w0).astype(np.float32)

    return W


# -----------------------------
# Model
# -----------------------------

def build_multioutput_dnn_vector(
    n_features: int,
    n_targets: int,
    hidden_sizes: list[int] | tuple[int, ...] = (128,),
    lr: float = 1e-3,
    l2: float = 0.0,
    dropout: float = 0.0,
    activation: str = "relu",
    batchnorm: bool = False,
) -> keras.Model:
    """Dense DNN with 1 or more hidden layers and vector output (n_targets)."""

    reg = keras.regularizers.l2(l2) if (l2 and l2 > 0) else None

    x_in = keras.Input(shape=(n_features,), name="X")
    h = x_in
    for units in hidden_sizes:
        h = layers.Dense(units, activation=activation, kernel_regularizer=reg)(h)
        if batchnorm:
            h = layers.BatchNormalization()(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout)(h)
    y_out = layers.Dense(n_targets, activation="sigmoid", name="Y")(h)

    model = keras.Model(inputs=x_in, outputs=y_out)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[],
    )

    return model


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--scenario", default=None, help="Imputation scenario for ELSA-style runs")
    ap.add_argument("--method", default=None, help="Imputation method for ELSA-style runs")
    ap.add_argument("--run-tag", default=None, help="Explicit run tag, e.g. HRS_SHARED")

    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--out-dir", default="analysis/dnn")

    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=0, help="Optional second hidden layer size; 0 disables it")
    ap.add_argument(
        "--hidden-sizes",
        default=None,
        help="Comma-separated hidden sizes, e.g. '256' or '256,64'. Overrides --hidden/--hidden2.",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--activation", default="relu", choices=["relu", "gelu"])
    ap.add_argument("--batchnorm", action="store_true")
    ap.add_argument("--standardize-x", action="store_true", help="Fit a StandardScaler within each training fold")
    ap.add_argument("--reduce-lr-on-plateau", action="store_true")
    ap.add_argument("--min-lr", type=float, default=1e-5)

    ap.add_argument("--include-age", action="store_true")
    ap.add_argument(
        "--transitions",
        nargs="*",
        default=None,
        help="Optional subset of transitions to run, e.g. S4_S5 S6_S7",
    )

    args = ap.parse_args()

    # Reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    scenario = args.scenario or ""
    method = args.method or ""
    hidden_sizes = parse_hidden_sizes(args.hidden, args.hidden2, args.hidden_sizes)

    trans_dir = Path(args.transitions_dir) / run_tag
    if not trans_dir.exists():
        raise FileNotFoundError(f"Transitions folder not found: {trans_dir}")

    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in trans_dir.glob("S*_S*.csv") if infer_transition_from_filename(p) is not None])
    if args.transitions:
        allowed = {str(x).strip() for x in args.transitions if str(x).strip()}
        files = [p for p in files if p.stem in allowed]
    if not files:
        raise FileNotFoundError(f"No transition CSV files found in: {trans_dir}")

    print(f"[OK] Run tag: {run_tag}")
    print(f"[OK] Using transitions: {trans_dir}")
    print(f"[OK] Found {len(files)} transition files")
    print(
        f"[OK] DNN(vector) hidden_sizes={hidden_sizes} | activation={args.activation} | "
        f"dropout={args.dropout} | l2={args.l2} | batchnorm={args.batchnorm} | "
        f"standardize_x={args.standardize_x} | folds={args.cv_folds} | seed={args.seed}"
    )

    metrics_rows: list[dict] = []

    for fp in files:
        tt = infer_transition_from_filename(fp)
        if tt is None:
            continue
        t, tp1 = tt
        trans_name = f"S{t}_S{tp1}"

        print(f"\n=== DNN OOF: {trans_name} ===")
        df = pd.read_csv(fp)

        # Predictors: columns that start with S{t}
        X_cols = [c for c in df.columns if c.startswith(f"S{t}")]
        if args.include_age and "Age" in df.columns and "Age" not in X_cols:
            X_cols.append("Age")

        X_df = df[X_cols].apply(pd.to_numeric, errors="coerce")

        # Targets: Y_S<tp1>_<TARGET>
        y_cols = [f"Y_S{tp1}_{tar}" for tar in ALL_TARGETS if f"Y_S{tp1}_{tar}" in df.columns]
        print(f"Rows: {len(df):,} | X: {len(X_cols)} | Targets available: {len(y_cols)}")
        if not y_cols:
            print(f"[SKIP] No targets found for {trans_name}")
            continue

        Y_df = df[y_cols].apply(pd.to_numeric, errors="coerce")
        Y_all = Y_df.to_numpy(dtype=np.float32)

        # Row mask:
        #  - all predictors observed
        #  - all targets for this transition observed (complete-case)
        X_mask = X_df.notna().all(axis=1).to_numpy()
        Y_mask = Y_df.notna().all(axis=1).to_numpy()
        mask = X_mask & Y_mask

        if int(mask.sum()) < 100:
            print(f"[WARN] Skipping {trans_name}: only {int(mask.sum())} complete-case rows")
            continue

        X = X_df.loc[mask, X_cols].to_numpy(dtype=np.float32)
        Y = Y_all[mask, :]
        n_rows, n_targets = X.shape[0], Y.shape[1]

        # OOF prediction buffer
        P_hat = np.full((n_rows, n_targets), np.nan, dtype=np.float32)

        # CV splitter
        strat_label = make_stratify_label(Y)
        n_splits = safe_n_splits(strat_label, args.cv_folds)
        if n_splits == 0:
            print(f"[WARN] Skipping {trans_name}: not enough samples to split")
            continue

        if strat_label is not None:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            split_iter = splitter.split(X, strat_label)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            split_iter = splitter.split(X)

        # Train one model per fold
        for fold, (tr_idx, te_idx) in enumerate(split_iter, start=1):
            Xtr, Ytr = X[tr_idx], Y[tr_idx]
            Xte = X[te_idx]

            if args.standardize_x:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            # Complete-case rows => Ytr has no NaNs; match Jupyter model_B: unweighted BCE
            model = build_multioutput_dnn_vector(
                n_features=X.shape[1],
                n_targets=n_targets,
                hidden_sizes=hidden_sizes,
                lr=args.lr,
                l2=args.l2,
                dropout=args.dropout,
                activation=args.activation,
                batchnorm=args.batchnorm,
            )

            callbacks: list[keras.callbacks.Callback] = [keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )]
            if args.reduce_lr_on_plateau:
                callbacks.append(
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=max(1, args.patience // 2),
                        min_lr=args.min_lr,
                        verbose=0,
                    )
                )

            model.fit(
                Xtr,
                Ytr,
                validation_split=0.2,
                epochs=args.epochs,
                batch_size=args.batch,
                verbose=0,
                callbacks=callbacks,
            )

            Pte = model.predict(Xte, verbose=0).astype(np.float32)
            if Pte.shape != (len(te_idx), n_targets):
                raise RuntimeError(f"Prediction shape mismatch: {Pte.shape} vs {(len(te_idx), n_targets)}")

            P_hat[te_idx, :] = Pte

        # Clip probabilities
        P_hat = np.clip(P_hat, EPS, 1.0 - EPS)

        # Build output probability table in the ORIGINAL row order
        if "idauniq" in df.columns:
            probs_out = pd.DataFrame({"idauniq": df["idauniq"].to_numpy()})
        else:
            probs_out = pd.DataFrame({"row_index": np.arange(len(df), dtype=int)})

        # Write probability columns
        for j, yc in enumerate(y_cols):
            target = yc.replace(f"Y_S{tp1}_", "")
            pcol = f"P_S{tp1}_{target}"
            probs_out[pcol] = np.nan
            probs_out.loc[np.where(mask)[0], pcol] = P_hat[:, j]

        probs_path = out_dir / f"probs_{trans_name}.csv"
        probs_out.to_csv(probs_path, index=False)
        print(f"[OK] Wrote probs: {probs_path.name} | rows={len(probs_out):,} | p-cols={len(y_cols)}")

        # Metrics per target (computed on all rows; Y is complete-case)
        for j, yc in enumerate(y_cols):
            target = yc.replace(f"Y_S{tp1}_", "")
            target_type = "adl" if target in ADL_TARGETS else "disease"

            obs = np.ones(len(Y), dtype=bool)
            y = Y[obs, j]
            p = P_hat[obs, j]

            # Basic metadata
            row = {
                "scenario": scenario,
                "method": method,
                "run_tag": run_tag,
                "transition": trans_name,
                "t": t,
                "tp1": tp1,
                "target": target,
                "target_type": target_type,
                "n_obs": int(obs.sum()),
                "prevalence": float(np.nanmean(y)) if obs.sum() else np.nan,
                "auc": np.nan,
                "pr_auc": np.nan,
                "brier": np.nan,
                "status": "ok",
            }

            # Skip if too small or not binary or only one class
            if obs.sum() < 50 or (not is_binary_array(y)) or (np.unique(y.astype(int)).size < 2):
                row["status"] = "skipped_too_rare_or_one_class"
                metrics_rows.append(row)
                continue

            y = y.astype(int)
            row["auc"] = float(roc_auc_score(y, p))
            row["pr_auc"] = float(average_precision_score(y, p))
            row["brier"] = float(brier_score_loss(y, p))
            metrics_rows.append(row)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    out_metrics = out_dir / "metrics_by_target.csv"
    metrics_df.to_csv(out_metrics, index=False)
    print(f"\n[OK] Saved metrics: {out_metrics}")

    # Summary
    if not metrics_df.empty:
        summary = (
            metrics_df
            .groupby(["run_tag", "scenario", "method", "transition", "target_type"], as_index=False)
            .agg(
                n_targets=("target", "count"),
                n_ok=("status", lambda s: int((s == "ok").sum())),
                mean_auc=("auc", "mean"),
                median_auc=("auc", "median"),
                mean_pr_auc=("pr_auc", "mean"),
                mean_brier=("brier", "mean"),
            )
            .sort_values(["run_tag", "scenario", "method", "transition", "target_type"])
        )
        out_sum = out_dir / "metrics_summary.csv"
        summary.to_csv(out_sum, index=False)
        print(f"[OK] Saved summary: {out_sum}")


if __name__ == "__main__":
    main()
