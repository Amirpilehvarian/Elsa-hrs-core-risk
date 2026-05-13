#!/usr/bin/env python3
"""ELSA-only stable feature selection with cross-transition validation.

Method summary
--------------
1. Use only ELSA transition tables from one canonical imputation branch.
2. For each held-out transition and target family (`disease`, `adl`):
   - pool the remaining transitions after stripping wave prefixes from predictors
   - rank features with a greedy mRMR-style criterion:
       relevance = mean mutual information with the family targets
       redundancy = mean absolute Spearman correlation with already-selected features
   - evaluate top-K subsets on the held-out transition with LR OOF predictions
3. Choose the final K with the one-standard-error rule on mean Brier score.
4. Refit the ranking on all ELSA transitions and export the final stable feature sets.
5. Compare reduced vs full feature sets with both LR and DNN on all ELSA transitions.

This script is intentionally ELSA-only. HRS is kept untouched so the selector
does not leak information from the external validation dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


ADL_TARGETS = [
    "DANGERA", "EATA", "MEDSA", "COMMUNA", "PHONEA", "MONEYA", "WALKRA", "TOILTA", "MEALSA", "MAPA",
    "BEDA", "DIMEA", "SHOPA", "BATHA", "ARMSA", "DRESSA", "WALK100A", "SITA", "CLIM1A", "HOUSEWKA",
    "PUSHA", "LIFTA", "CHAIRA", "CLIMSA", "STOOPA",
]
DISEASE_TARGETS = [
    "PARKINE", "CONHRTFE", "HIPE", "HEARTE", "HRTMRE", "HRTATTE", "STROKE", "LUNGE", "CANCRE",
    "ANGINE", "OSTEOE", "HRTRHME", "PSYCHE", "DIABE", "ASTHMAE", "CATRACTE", "HCHOLE", "ARTHRE", "HIBPE",
]
TARGET_FAMILIES = {
    "disease": DISEASE_TARGETS,
    "adl": ADL_TARGETS,
}

EPS = 1e-6
DEFAULT_K_GRID = [4, 8, 12, 16, 24, 32, 40, 48, 64, 87]
FROZEN_DNN = {
    "hidden_sizes": [512],
    "epochs": 30,
    "batch": 64,
    "lr": 1e-3,
    "patience": 5,
    "activation": "relu",
    "batchnorm": False,
    "dropout": 0.0,
    "l2": 0.0,
    "standardize_x": True,
    "reduce_lr_on_plateau": True,
    "min_lr": 1e-5,
}


@dataclass
class TransitionFrame:
    name: str
    t: int
    tp1: int
    df: pd.DataFrame
    x_cols: list[str]
    x_canonical: list[str]


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def infer_transition_from_filename(path: Path) -> tuple[int, int] | None:
    match = re.match(r"^S(\d+)_S(\d+)\.csv$", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def strip_wave_prefix(column: str) -> str:
    return re.sub(r"^S\d+", "", column)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def quantile_edges(p: np.ndarray, n_bins: int = 10, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    edges[0] = max(edges[0], eps)
    edges[-1] = min(edges[-1], 1.0)
    return edges


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    edges = quantile_edges(p, n_bins=n_bins, eps=EPS)
    rows: list[tuple[float, float, int]] = []
    for k in range(n_bins):
        left, right = edges[k], edges[k + 1]
        if k < n_bins - 1:
            mask = (p >= left) & (p < right)
        else:
            mask = (p >= left) & (p <= right)
        nk = int(mask.sum())
        if nk == 0:
            continue
        rows.append((float(p[mask].mean()), float(y[mask].mean()), nk))
    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)


def r2_calibration_from_bins(tab: np.ndarray) -> float:
    if tab.shape[0] < 3:
        return np.nan
    reg = LinearRegression().fit(tab[:, 0].reshape(-1, 1), tab[:, 1])
    return float(reg.score(tab[:, 0].reshape(-1, 1), tab[:, 1]))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for k in range(n_bins):
        mask = idx == k
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def choose_splitter(y: np.ndarray, requested_folds: int, seed: int):
    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts.min())
    n_splits = min(int(requested_folds), min_class)
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits
    return None, 0


def make_multioutput_splitter(Y: np.ndarray, requested_folds: int, seed: int):
    labels = (Y.sum(axis=1) > 0).astype(int)
    if np.unique(labels).size >= 2:
        counts = np.bincount(labels, minlength=2)
        min_class = int(counts.min())
        n_splits = min(int(requested_folds), min_class)
        if n_splits >= 2:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), labels, n_splits
    n_splits = min(int(requested_folds), len(Y))
    if n_splits >= 2:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None, n_splits
    return None, None, 0


def build_transition_frames(transitions_dir: Path) -> list[TransitionFrame]:
    frames: list[TransitionFrame] = []
    for path in sorted(transitions_dir.glob("S*_S*.csv")):
        tt = infer_transition_from_filename(path)
        if tt is None:
            continue
        t, tp1 = tt
        df = pd.read_csv(path)
        x_cols = [c for c in df.columns if c.startswith(f"S{t}")]
        x_canonical = [strip_wave_prefix(c) for c in x_cols]
        frames.append(TransitionFrame(name=f"S{t}_S{tp1}", t=t, tp1=tp1, df=df, x_cols=x_cols, x_canonical=x_canonical))
    if not frames:
        raise FileNotFoundError(f"No transition CSV files found in {transitions_dir}")
    return frames


def build_pooled_training_frame(frames: list[TransitionFrame], family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_names = TARGET_FAMILIES[family]
    pooled_x: list[pd.DataFrame] = []
    pooled_y: list[pd.DataFrame] = []
    for frame in frames:
        xdf = frame.df[frame.x_cols].copy()
        xdf.columns = frame.x_canonical
        xdf = xdf.apply(pd.to_numeric, errors="coerce")
        ycols = [f"Y_S{frame.tp1}_{target}" for target in target_names if f"Y_S{frame.tp1}_{target}" in frame.df.columns]
        ydf = frame.df[ycols].copy()
        ydf.columns = [c.replace(f"Y_S{frame.tp1}_", "") for c in ycols]
        ydf = ydf.apply(pd.to_numeric, errors="coerce")
        pooled_x.append(xdf)
        pooled_y.append(ydf)
    x_all = pd.concat(pooled_x, ignore_index=True)
    y_all = pd.concat(pooled_y, ignore_index=True)
    shared_targets = [t for t in target_names if t in y_all.columns]
    y_all = y_all[shared_targets]
    mask = x_all.notna().all(axis=1) & y_all.notna().all(axis=1)
    return x_all.loc[mask].reset_index(drop=True), y_all.loc[mask].reset_index(drop=True)


def feature_is_discrete(series: pd.Series) -> bool:
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return True
    unique = np.unique(vals)
    if len(unique) <= 12 and np.allclose(unique, np.round(unique)):
        return True
    return False


def compute_relevance_scores(x_all: pd.DataFrame, y_all: pd.DataFrame, seed: int) -> pd.Series:
    scores: list[dict[str, float]] = []
    for feature in x_all.columns:
        x = x_all[feature]
        discrete = feature_is_discrete(x)
        feature_scores: list[float] = []
        for target in y_all.columns:
            y = y_all[target]
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 20:
                continue
            yv = y.loc[mask].astype(int).to_numpy()
            if np.unique(yv).size < 2:
                continue
            xv = x.loc[mask].to_numpy().reshape(-1, 1)
            mi = mutual_info_classif(
                xv,
                yv,
                discrete_features=[discrete],
                random_state=seed,
            )[0]
            feature_scores.append(float(mi))
        scores.append(
            {
                "feature": feature,
                "mean_mutual_information": float(np.mean(feature_scores)) if feature_scores else 0.0,
                "n_target_scores": int(len(feature_scores)),
            }
        )
    rel = pd.DataFrame(scores).set_index("feature")
    return rel["mean_mutual_information"]


def compute_redundancy_matrix(x_all: pd.DataFrame) -> pd.DataFrame:
    corr = x_all.corr(method="spearman").abs().fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    return corr


def greedy_mrmr_ranking(relevance: pd.Series, redundancy: pd.DataFrame) -> list[str]:
    features = list(relevance.index)
    if not features:
        return []
    rel = relevance.copy()
    rel_max = float(rel.max()) if float(rel.max()) > 0 else 1.0
    rel = rel / rel_max
    selected: list[str] = []
    remaining = set(features)
    while remaining:
        if not selected:
            next_feature = max(remaining, key=lambda f: (float(rel[f]), f))
        else:
            def score(feature: str) -> tuple[float, float, str]:
                red = float(redundancy.loc[feature, selected].mean()) if selected else 0.0
                return (float(rel[feature]) - red, float(rel[feature]), feature)
            next_feature = max(remaining, key=score)
        selected.append(next_feature)
        remaining.remove(next_feature)
    return selected


def evaluate_probabilities(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int) -> dict[str, float]:
    p = np.clip(np.asarray(p_pred, float), EPS, 1.0 - EPS)
    y = np.asarray(y_true, int)
    tab = calibration_table(y, p, n_bins=n_bins)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p, n_bins=n_bins)),
        "r2cal": float(r2_calibration_from_bins(tab)),
    }


def evaluate_lr_family_transition(
    frame: TransitionFrame,
    family: str,
    feature_set: list[str],
    cv_folds: int,
    seed: int,
    n_bins: int,
    label: str,
) -> pd.DataFrame:
    target_names = TARGET_FAMILIES[family]
    orig_features = [f"S{frame.t}{feature}" for feature in feature_set if f"S{frame.t}{feature}" in frame.df.columns]
    xdf = frame.df[orig_features].apply(pd.to_numeric, errors="coerce")
    rows: list[dict] = []
    for target in target_names:
        ycol = f"Y_S{frame.tp1}_{target}"
        if ycol not in frame.df.columns:
            continue
        y = pd.to_numeric(frame.df[ycol], errors="coerce")
        mask = xdf.notna().all(axis=1) & y.notna()
        base = {
            "transition": frame.name,
            "family": family,
            "model": "LR",
            "feature_set": label,
            "target": target,
            "target_type": family,
            "n_features": int(len(orig_features)),
            "n_obs": int(mask.sum()),
            "prevalence": float(y.loc[mask].mean()) if mask.any() else np.nan,
            "status": "ok",
        }
        if int(mask.sum()) < 20:
            base["status"] = "too_few_rows"
            rows.append(base)
            continue
        x = xdf.loc[mask].to_numpy(dtype=float)
        yv = y.loc[mask].to_numpy(dtype=int)
        if np.unique(yv).size < 2:
            base["status"] = "one_class"
            rows.append(base)
            continue
        splitter, n_splits = choose_splitter(yv, cv_folds, seed)
        if splitter is None:
            base["status"] = "cannot_split"
            rows.append(base)
            continue
        oof = np.full(len(yv), np.nan, dtype=float)
        for train_idx, test_idx in splitter.split(x, yv):
            model = Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=False)),
                    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
                ]
            )
            model.fit(x[train_idx], yv[train_idx])
            oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        ok = np.isfinite(oof)
        if ok.sum() < 10 or np.unique(yv[ok]).size < 2:
            base["status"] = "invalid_oof"
            rows.append(base)
            continue
        base.update(evaluate_probabilities(yv[ok], oof[ok], n_bins))
        base["cv_folds"] = int(n_splits)
        rows.append(base)
    return pd.DataFrame(rows)


def build_multioutput_dnn_vector(
    n_features: int,
    n_targets: int,
    hidden_sizes: list[int],
    lr: float,
    l2: float,
    dropout: float,
    activation: str,
    batchnorm: bool,
) -> keras.Model:
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
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


def evaluate_dnn_family_transition(
    frame: TransitionFrame,
    family: str,
    feature_set: list[str],
    cv_folds: int,
    seed: int,
    n_bins: int,
    label: str,
) -> pd.DataFrame:
    target_names = TARGET_FAMILIES[family]
    orig_features = [f"S{frame.t}{feature}" for feature in feature_set if f"S{frame.t}{feature}" in frame.df.columns]
    ycols = [f"Y_S{frame.tp1}_{target}" for target in target_names if f"Y_S{frame.tp1}_{target}" in frame.df.columns]
    xdf = frame.df[orig_features].apply(pd.to_numeric, errors="coerce")
    ydf = frame.df[ycols].apply(pd.to_numeric, errors="coerce")
    mask = xdf.notna().all(axis=1) & ydf.notna().all(axis=1)
    rows: list[dict] = []
    if int(mask.sum()) < 50 or not ycols:
        for target in target_names:
            rows.append(
                {
                    "transition": frame.name,
                    "family": family,
                    "model": "DNN",
                    "feature_set": label,
                    "target": target,
                    "target_type": family,
                    "n_features": int(len(orig_features)),
                    "n_obs": int(mask.sum()),
                    "prevalence": np.nan,
                    "status": "too_few_rows",
                }
            )
        return pd.DataFrame(rows)

    x = xdf.loc[mask].to_numpy(dtype=np.float32)
    y = ydf.loc[mask].to_numpy(dtype=int)
    splitter, strat_labels, n_splits = make_multioutput_splitter(y, cv_folds, seed)
    if splitter is None:
        for target in target_names:
            rows.append(
                {
                    "transition": frame.name,
                    "family": family,
                    "model": "DNN",
                    "feature_set": label,
                    "target": target,
                    "target_type": family,
                    "n_features": int(len(orig_features)),
                    "n_obs": int(mask.sum()),
                    "prevalence": np.nan,
                    "status": "cannot_split",
                }
            )
        return pd.DataFrame(rows)

    oof = np.full_like(y, np.nan, dtype=np.float32)
    split_iter = splitter.split(x, strat_labels) if strat_labels is not None else splitter.split(x)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        set_all_seeds(seed + fold_idx)
        tf.keras.backend.clear_session()
        x_tr = x[train_idx].copy()
        x_te = x[test_idx].copy()
        if FROZEN_DNN["standardize_x"]:
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr).astype(np.float32)
            x_te = scaler.transform(x_te).astype(np.float32)
        model = build_multioutput_dnn_vector(
            n_features=x.shape[1],
            n_targets=y.shape[1],
            hidden_sizes=list(FROZEN_DNN["hidden_sizes"]),
            lr=float(FROZEN_DNN["lr"]),
            l2=float(FROZEN_DNN["l2"]),
            dropout=float(FROZEN_DNN["dropout"]),
            activation=str(FROZEN_DNN["activation"]),
            batchnorm=bool(FROZEN_DNN["batchnorm"]),
        )
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(FROZEN_DNN["patience"]),
                restore_best_weights=True,
            )
        ]
        if FROZEN_DNN["reduce_lr_on_plateau"]:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=max(1, int(FROZEN_DNN["patience"]) // 2),
                    min_lr=float(FROZEN_DNN["min_lr"]),
                    verbose=0,
                )
            )
        model.fit(
            x_tr,
            y[train_idx],
            validation_data=(x_te, y[test_idx]),
            epochs=int(FROZEN_DNN["epochs"]),
            batch_size=int(FROZEN_DNN["batch"]),
            verbose=0,
            callbacks=callbacks,
        )
        pred = model.predict(x_te, verbose=0)
        oof[test_idx, :] = pred

    for j, target in enumerate([c.replace(f"Y_S{frame.tp1}_", "") for c in ycols]):
        yv = y[:, j].astype(int)
        pv = np.asarray(oof[:, j], dtype=float)
        base = {
            "transition": frame.name,
            "family": family,
            "model": "DNN",
            "feature_set": label,
            "target": target,
            "target_type": family,
            "n_features": int(len(orig_features)),
            "n_obs": int(len(yv)),
            "prevalence": float(yv.mean()),
            "status": "ok",
            "cv_folds": int(n_splits),
        }
        ok = np.isfinite(pv)
        if ok.sum() < 10 or np.unique(yv[ok]).size < 2:
            base["status"] = "invalid_oof"
            rows.append(base)
            continue
        base.update(evaluate_probabilities(yv[ok], pv[ok], n_bins))
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["auc", "brier", "ece", "r2cal"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work["status"].astype(str).eq("ok")].copy()
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(["family", "model", "feature_set"], as_index=False)
        .agg(
            n_targets=("target", "count"),
            mean_auc=("auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
            mean_r2cal=("r2cal", "mean"),
        )
        .sort_values(["family", "model", "feature_set"])
        .reset_index(drop=True)
    )


def select_k_one_se(curve_summary: pd.DataFrame, family: str) -> dict[str, float]:
    sub = curve_summary[curve_summary["family"] == family].sort_values("k").copy()
    best_idx = sub["mean_brier"].idxmin()
    best_row = sub.loc[best_idx]
    threshold = float(best_row["mean_brier"] + best_row["se_brier"])
    eligible = sub[sub["mean_brier"] <= threshold].sort_values("k")
    chosen = eligible.iloc[0] if not eligible.empty else best_row
    return {
        "family": family,
        "k_best_brier": int(best_row["k"]),
        "k_one_se": int(chosen["k"]),
        "best_mean_brier": float(best_row["mean_brier"]),
        "best_se_brier": float(best_row["se_brier"]),
        "one_se_threshold": threshold,
        "chosen_mean_brier": float(chosen["mean_brier"]),
        "chosen_mean_auc": float(chosen["mean_auc"]),
        "chosen_mean_ece": float(chosen["mean_ece"]),
        "chosen_mean_r2cal": float(chosen["mean_r2cal"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--transitions-dir", default="Transition_data/01_transitions")
    ap.add_argument("--out-dir", default="analysis/ELSA/feature_selection/stable_mrmr")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--k-grid",
        default=",".join(str(v) for v in DEFAULT_K_GRID),
        help="Comma-separated feature counts to evaluate.",
    )
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    transitions_dir = Path(args.transitions_dir) / run_tag
    out_dir = Path(args.out_dir) / run_tag
    rankings_dir = out_dir / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    k_grid = sorted({int(v.strip()) for v in str(args.k_grid).split(",") if v.strip()})
    frames = build_transition_frames(transitions_dir)
    candidate_features = frames[0].x_canonical

    curve_rows: list[dict] = []
    ranking_rows: list[dict] = []
    heldout_topk: list[dict] = []

    for family in TARGET_FAMILIES:
        for held_out in frames:
            train_frames = [frame for frame in frames if frame.name != held_out.name]
            x_train, y_train = build_pooled_training_frame(train_frames, family)
            relevance = compute_relevance_scores(x_train[candidate_features], y_train, seed=args.seed)
            redundancy = compute_redundancy_matrix(x_train[candidate_features])
            ranking = greedy_mrmr_ranking(relevance, redundancy)

            rank_df = pd.DataFrame(
                {
                    "family": family,
                    "held_out_transition": held_out.name,
                    "feature": ranking,
                    "rank": np.arange(1, len(ranking) + 1, dtype=int),
                    "mean_mutual_information": [float(relevance[f]) for f in ranking],
                }
            )
            rank_df.to_csv(rankings_dir / f"ranking_{family}_{held_out.name}.csv", index=False)
            ranking_rows.extend(rank_df.to_dict("records"))

            for k in k_grid:
                selected = ranking[: min(k, len(ranking))]
                heldout_topk.extend(
                    {
                        "family": family,
                        "held_out_transition": held_out.name,
                        "k": int(k),
                        "feature": feature,
                    }
                    for feature in selected
                )
                lr_metrics = evaluate_lr_family_transition(
                    frame=held_out,
                    family=family,
                    feature_set=selected,
                    cv_folds=args.cv_folds,
                    seed=args.seed,
                    n_bins=args.calibration_bins,
                    label=f"top_{k}",
                )
                ok = lr_metrics[lr_metrics["status"].eq("ok")].copy()
                row = {
                    "family": family,
                    "held_out_transition": held_out.name,
                    "k": int(k),
                    "n_features": int(len(selected)),
                    "n_ok_targets": int(ok.shape[0]),
                    "mean_auc": float(ok["auc"].mean()) if not ok.empty else np.nan,
                    "mean_brier": float(ok["brier"].mean()) if not ok.empty else np.nan,
                    "mean_ece": float(ok["ece"].mean()) if not ok.empty else np.nan,
                    "mean_r2cal": float(ok["r2cal"].mean()) if not ok.empty else np.nan,
                }
                curve_rows.append(row)

    curve_raw = pd.DataFrame(curve_rows)
    curve_raw.to_csv(out_dir / "selection_curve_raw.csv", index=False)

    curve_summary = (
        curve_raw.groupby(["family", "k"], as_index=False)
        .agg(
            mean_auc=("mean_auc", "mean"),
            mean_brier=("mean_brier", "mean"),
            mean_ece=("mean_ece", "mean"),
            mean_r2cal=("mean_r2cal", "mean"),
            transition_count=("held_out_transition", "count"),
            sd_auc=("mean_auc", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            sd_brier=("mean_brier", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            sd_ece=("mean_ece", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            sd_r2cal=("mean_r2cal", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        )
        .sort_values(["family", "k"])
        .reset_index(drop=True)
    )
    for metric in ["auc", "brier", "ece", "r2cal"]:
        curve_summary[f"se_{metric}"] = curve_summary[f"sd_{metric}"] / np.sqrt(curve_summary["transition_count"].clip(lower=1))
    curve_summary.to_csv(out_dir / "selection_curve_summary.csv", index=False)

    selection_summary = pd.DataFrame([select_k_one_se(curve_summary, family) for family in TARGET_FAMILIES])

    full_rank_rows: list[dict] = []
    selected_feature_rows: list[dict] = []
    full_rankings: dict[str, list[str]] = {}
    selected_sets: dict[str, list[str]] = {}
    for family in TARGET_FAMILIES:
        x_all, y_all = build_pooled_training_frame(frames, family)
        relevance = compute_relevance_scores(x_all[candidate_features], y_all, seed=args.seed)
        redundancy = compute_redundancy_matrix(x_all[candidate_features])
        ranking = greedy_mrmr_ranking(relevance, redundancy)
        full_rankings[family] = ranking
        k_final = int(selection_summary.loc[selection_summary["family"] == family, "k_one_se"].iloc[0])
        selected_sets[family] = ranking[:k_final]

        topk_rows = pd.DataFrame(heldout_topk)
        freq = (
            topk_rows[(topk_rows["family"] == family) & (topk_rows["k"] == k_final)]
            .groupby("feature")
            .size()
            .div(len(frames))
            .rename("selection_frequency")
        )
        rank_df = pd.DataFrame(
            {
                "family": family,
                "feature": ranking,
                "rank": np.arange(1, len(ranking) + 1, dtype=int),
                "mean_mutual_information": [float(relevance[f]) for f in ranking],
                "selection_frequency_top_k": [float(freq.get(f, 0.0)) for f in ranking],
                "selected_final": [int(f in selected_sets[family]) for f in ranking],
            }
        )
        rank_df.to_csv(out_dir / f"ranking_full_{family}.csv", index=False)
        full_rank_rows.extend(rank_df.to_dict("records"))

        sel_df = rank_df[rank_df["selected_final"].eq(1)].copy()
        sel_df.to_csv(out_dir / f"selected_features_{family}.csv", index=False)
        selected_feature_rows.extend(sel_df.to_dict("records"))

    union_features = sorted(set(selected_sets["disease"]) | set(selected_sets["adl"]))
    union_df = pd.DataFrame({"feature": union_features, "selected_in_disease": [int(f in selected_sets["disease"]) for f in union_features], "selected_in_adl": [int(f in selected_sets["adl"]) for f in union_features]})
    union_df.to_csv(out_dir / "selected_features_union.csv", index=False)

    selection_summary["n_selected_features"] = selection_summary["family"].map(lambda fam: len(selected_sets[fam]))
    selection_summary["union_size"] = int(len(union_features))
    selection_summary.to_csv(out_dir / "selection_summary.csv", index=False)

    model_rows: list[pd.DataFrame] = []
    for frame in frames:
        for family in TARGET_FAMILIES:
            full_set = full_rankings[family]
            reduced_set = selected_sets[family]
            model_rows.append(
                evaluate_lr_family_transition(frame, family, full_set, args.cv_folds, args.seed, args.calibration_bins, "full")
            )
            model_rows.append(
                evaluate_lr_family_transition(frame, family, reduced_set, args.cv_folds, args.seed, args.calibration_bins, "reduced")
            )
            model_rows.append(
                evaluate_dnn_family_transition(frame, family, full_set, args.cv_folds, args.seed, args.calibration_bins, "full")
            )
            model_rows.append(
                evaluate_dnn_family_transition(frame, family, reduced_set, args.cv_folds, args.seed, args.calibration_bins, "reduced")
            )

    model_metrics = pd.concat(model_rows, ignore_index=True)
    model_metrics.to_csv(out_dir / "reduced_vs_full_metrics_by_target.csv", index=False)
    model_summary = summarize_metrics(model_metrics)
    model_summary.to_csv(out_dir / "reduced_vs_full_summary.csv", index=False)

    with open(out_dir / "feature_selection_spec.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "cross_transition_stable_mrmr",
                "run_tag": run_tag,
                "candidate_features": len(candidate_features),
                "k_grid": k_grid,
                "cv_folds": args.cv_folds,
                "calibration_bins": args.calibration_bins,
                "seed": args.seed,
                "family_final_k": {family: int(selection_summary.loc[selection_summary["family"] == family, "k_one_se"].iloc[0]) for family in TARGET_FAMILIES},
                "union_feature_count": len(union_features),
                "frozen_dnn": FROZEN_DNN,
            },
            handle,
            indent=2,
        )

    print(f"[OK] Wrote feature-selection outputs to {out_dir}")
    print(selection_summary.to_string(index=False))


if __name__ == "__main__":
    main()
