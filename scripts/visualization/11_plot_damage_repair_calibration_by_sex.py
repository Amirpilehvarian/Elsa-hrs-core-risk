#!/usr/bin/env python3
"""Make pooled damage/repair calibration plots stratified by sex."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-6
DEFAULT_CI_Z = 1.96
SEX_ORDER = ("female", "male")
MODEL_ORDER = ("LR", "DNN")
MODEL_STYLE = {
    "LR": {"color": "#1f77b4", "marker": "o"},
    "DNN": {"color": "#ff7f0e", "marker": "s"},
}
TARGET_GROUPS = ("all", "disease", "adl")


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def normalize_id_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    return s.astype(str).str.extract(r"(\d+)", expand=False)


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
    half = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    lower = np.clip(center - half, 0.0, 1.0)
    upper = np.clip(center + half, 0.0, 1.0)
    return lower, upper


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int, z: float = DEFAULT_CI_Z) -> dict[str, np.ndarray]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    edges = quantile_edges(p, n_bins=n_bins, eps=EPS)
    rows = []
    for k in range(n_bins):
        left, right = edges[k], edges[k + 1]
        mask = (p >= left) & (p < right) if k < n_bins - 1 else (p >= left) & (p <= right)
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = p[mask]
        yk = y[mask]
        low, high = wilson_interval(np.array([yk.sum()]), np.array([nk]), z=z)
        rows.append((float(pk.mean()), float(yk.mean()), float(low[0]), float(high[0]), nk))
    if not rows:
        empty = np.array([], dtype=float)
        return {"x": empty, "y": empty, "y_low": empty, "y_high": empty, "n": empty}
    arr = np.asarray(rows, dtype=float)
    return {"x": arr[:, 0], "y": arr[:, 1], "y_low": arr[:, 2], "y_high": arr[:, 3], "n": arr[:, 4]}


def pooled_event_data(events_root: Path, run_tag: str, model: str, sex_lookup: pd.DataFrame, sex: str, event_type: str, target_group: str) -> tuple[np.ndarray, np.ndarray] | None:
    events_fp = events_root / model / run_tag / "events_all.csv"
    df = pd.read_csv(events_fp)
    df["idauniq"] = normalize_id_series(df["idauniq"])
    df["sex"] = df["idauniq"].map(sex_lookup.set_index("idauniq")["sex"])
    df = df[df["sex"] == sex].copy()
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
    df = df[df["event_type"] == event_type].copy()
    if target_group != "all":
        df = df[df["target_type"] == target_group].copy()
    if df.empty:
        return None
    y = pd.to_numeric(df["y"], errors="coerce")
    p = pd.to_numeric(df["p"], errors="coerce")
    mask = y.notna() & p.notna()
    if mask.sum() < 10:
        return None
    yv = y[mask].astype(int).to_numpy()
    if np.unique(yv).size < 2:
        return None
    pv = np.clip(p[mask].to_numpy(dtype=float), EPS, 1 - EPS)
    return yv, pv


def plot_calibration(tab_by_model: dict[str, dict[str, np.ndarray]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#1f77b4", linewidth=1.5, alpha=0.85)
    any_data = False
    for model in MODEL_ORDER:
        tab = tab_by_model.get(model)
        if not tab or len(tab["x"]) == 0:
            continue
        any_data = True
        style = MODEL_STYLE[model]
        yerr = np.vstack([
            np.maximum(0.0, tab["y"] - tab["y_low"]),
            np.maximum(0.0, tab["y_high"] - tab["y"]),
        ])
        ax.errorbar(tab["x"], tab["y"], yerr=yerr, fmt=style["marker"], color=style["color"], ecolor=style["color"], capsize=3, markersize=7, linewidth=0, label=model)
    if not any_data:
        ax.text(0.5, 0.5, "No evaluable data", ha="center", va="center", transform=ax.transAxes, fontsize=13)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if any_data:
        ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dataset-label", default=None)
    ap.add_argument("--events-root", required=True)
    ap.add_argument("--sex-lookup", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    dataset_label = args.dataset_label or run_tag
    sex_lookup = pd.read_csv(args.sex_lookup)[["idauniq", "sex"]].copy()
    sex_lookup["idauniq"] = normalize_id_series(sex_lookup["idauniq"])
    sex_lookup["sex"] = sex_lookup["sex"].astype(str).str.strip().str.lower()
    sex_lookup = sex_lookup[sex_lookup["sex"].isin(SEX_ORDER)].drop_duplicates()

    for sex in SEX_ORDER:
        for event_type in ("damage", "repair"):
            for target_group in TARGET_GROUPS:
                tabs = {}
                for model in MODEL_ORDER:
                    res = pooled_event_data(Path(args.events_root), run_tag, model, sex_lookup, sex, event_type, target_group)
                    if res is None:
                        continue
                    y, p = res
                    tabs[model] = calibration_table(y, p, n_bins=args.n_bins)
                title = f"{dataset_label} | {run_tag} | {sex.title()} | {event_type.title()} | pooled {target_group}"
                out_fp = Path(args.out_root) / run_tag / sex / event_type / f"pooled_{target_group}.png"
                plot_calibration(tabs, out_fp, title=title)


if __name__ == "__main__":
    main()
