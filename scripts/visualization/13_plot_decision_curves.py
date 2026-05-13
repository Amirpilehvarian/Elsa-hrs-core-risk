#!/usr/bin/env python3
"""Plot pooled decision curves and compact summary tables.

Inputs:
- <metrics-root>/<RUN_TAG>/<analysis-kind>/pooled_curves.csv
- <metrics-root>/<RUN_TAG>/<analysis-kind>/summary.csv

Outputs:
- <out-root>/<RUN_TAG>/<analysis-kind>/pooled_decision_curve.png
- <out-root>/<RUN_TAG>/<analysis-kind>/decision_curve_summary_table.csv
- <out-root>/<RUN_TAG>/<analysis-kind>/decision_curve_summary_table.png
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-codex"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_style import apply_paper_style

apply_paper_style()


PANEL_ORDER = ("Disease", "ADL")
STRATEGY_ORDER = ("LR", "DNN", "Treat all", "Treat none")
STYLE = {
    "LR": {"color": "#1f77b4", "lw": 2.4, "ls": "-", "label": "LR"},
    "DNN": {"color": "#ff7f0e", "lw": 2.4, "ls": "-", "label": "DNN"},
    "Treat all": {"color": "#6f6f6f", "lw": 2.0, "ls": "--", "label": "Treat all"},
    "Treat none": {"color": "#222222", "lw": 1.8, "ls": ":", "label": "Treat none"},
}


def resolve_run_tag(run_tag: str | None, scenario: str | None, method: str | None) -> str:
    if run_tag:
        return str(run_tag).strip()
    if scenario and method:
        return f"{scenario}_{method}"
    raise ValueError("Provide either --run-tag or both --scenario and --method.")


def load_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pooled decision-curve file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")
    df["target_type"] = df["target_type"].astype(str).str.strip()
    df["strategy"] = df["strategy"].astype(str).str.strip()
    return df


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["target_type"] = df["target_type"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    return df


def panel_limits(curves: pd.DataFrame) -> tuple[float, float]:
    vals: list[float] = [0.0]
    for strategy in STRATEGY_ORDER:
        sub = curves[curves["strategy"] == strategy]
        if sub.empty:
            continue
        y = pd.to_numeric(sub["mean_net_benefit"], errors="coerce").to_numpy(dtype=float)
        vals.extend(y[np.isfinite(y)].tolist())
        if strategy in {"LR", "DNN"} and "se_net_benefit" in sub.columns:
            se = pd.to_numeric(sub["se_net_benefit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            vals.extend((y - se)[np.isfinite(y - se)].tolist())
            vals.extend((y + se)[np.isfinite(y + se)].tolist())
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = hi - lo
    pad = 0.015 if span <= 0 else 0.10 * span
    y0 = min(-0.01, lo - pad)
    y1 = hi + pad
    if y1 <= y0:
        y1 = y0 + 0.05
    return (y0, y1)


def pooled_plot(
    pooled: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    dataset_label: str,
    run_tag: str,
    analysis_kind: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=True)
    fig.suptitle(f"{dataset_label} | {run_tag} | {analysis_kind.upper()} | Decision-curve analysis", fontsize=18, y=0.98)

    all_handles = []
    all_labels = []
    x_min = float(pooled["threshold"].min()) if not pooled.empty else 0.0
    x_max = float(pooled["threshold"].max()) if not pooled.empty else 1.0

    for ax, panel in zip(axes, PANEL_ORDER):
        sub = pooled[pooled["target_type"] == panel].copy().sort_values("threshold")
        ax.set_title(panel.upper(), fontsize=15)
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_xlabel("Threshold probability")
        ax.set_xlim(x_min, x_max)

        if sub.empty:
            ax.text(0.5, 0.5, "No evaluable targets", ha="center", va="center", transform=ax.transAxes, fontsize=13)
            ax.set_ylim(-0.02, 0.05)
            continue

        y0, y1 = panel_limits(sub)
        ax.set_ylim(y0, y1)

        for strategy in STRATEGY_ORDER:
            cur = sub[sub["strategy"] == strategy].sort_values("threshold")
            if cur.empty:
                continue
            style = STYLE[strategy]
            x = cur["threshold"].to_numpy(dtype=float)
            y = cur["mean_net_benefit"].to_numpy(dtype=float)
            handle = ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=style["lw"],
                linestyle=style["ls"],
                label=style["label"],
            )[0]
            if strategy in {"LR", "DNN"}:
                se = cur["se_net_benefit"].to_numpy(dtype=float)
                ax.fill_between(x, y - se, y + se, color=style["color"], alpha=0.12)
            if style["label"] not in all_labels:
                all_handles.append(handle)
                all_labels.append(style["label"])

        prevalence = float(sub["mean_prevalence"].mean())
        n_pairs = int(round(sub.loc[sub["strategy"].isin(["LR", "DNN"]), "n_pairs"].max()))
        best_rows = summary[summary["target_type"] == panel].copy()
        if not best_rows.empty:
            best_row = best_rows.sort_values("integrated_net_benefit", ascending=False).iloc[0]
            note = (
                f"pairs={n_pairs} | mean prev={prevalence:.3f}\n"
                f"best={best_row['model']} | useful range={best_row['beneficial_threshold_range'] or 'none'}"
            )
        else:
            note = f"pairs={n_pairs} | mean prev={prevalence:.3f}"
        ax.text(0.03, 0.95, note, transform=ax.transAxes, va="top", fontsize=10.5)

    axes[0].set_ylabel("Net benefit")
    fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "Outcome family",
                "LR mean NB",
                "DNN mean NB",
                "DNN - LR",
                "LR > default",
                "DNN > default",
                "Best model",
                "Useful thresholds",
            ]
        )

    wide = summary.pivot(index="target_type", columns="model")
    rows: list[dict] = []
    for target_type in PANEL_ORDER:
        if target_type not in wide.index:
            rows.append(
                {
                    "Outcome family": target_type,
                    "LR mean NB": np.nan,
                    "DNN mean NB": np.nan,
                    "DNN - LR": np.nan,
                    "LR > default": np.nan,
                    "DNN > default": np.nan,
                    "Best model": "",
                    "Useful thresholds": "",
                }
            )
            continue

        row = wide.loc[target_type]
        lr_nb = row.get(("integrated_net_benefit", "LR"), np.nan)
        dnn_nb = row.get(("integrated_net_benefit", "DNN"), np.nan)
        lr_share = row.get(("share_thresholds_better_than_default", "LR"), np.nan)
        dnn_share = row.get(("share_thresholds_better_than_default", "DNN"), np.nan)
        lr_range = row.get(("beneficial_threshold_range", "LR"), "")
        dnn_range = row.get(("beneficial_threshold_range", "DNN"), "")

        if pd.notna(lr_nb) and pd.notna(dnn_nb):
            if dnn_nb > lr_nb:
                best_model = "DNN"
                useful_range = dnn_range
            elif lr_nb > dnn_nb:
                best_model = "LR"
                useful_range = lr_range
            else:
                best_model = "Tie"
                useful_range = lr_range or dnn_range
        elif pd.notna(lr_nb):
            best_model = "LR"
            useful_range = lr_range
        elif pd.notna(dnn_nb):
            best_model = "DNN"
            useful_range = dnn_range
        else:
            best_model = ""
            useful_range = ""

        rows.append(
            {
                "Outcome family": target_type,
                "LR mean NB": lr_nb,
                "DNN mean NB": dnn_nb,
                "DNN - LR": dnn_nb - lr_nb if pd.notna(lr_nb) and pd.notna(dnn_nb) else np.nan,
                "LR > default": lr_share,
                "DNN > default": dnn_share,
                "Best model": best_model,
                "Useful thresholds": useful_range,
            }
        )
    return pd.DataFrame(rows)


def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.to_csv(path, index=False)


def table_figure(df: pd.DataFrame, out_path: Path, dataset_label: str, run_tag: str, analysis_kind: str) -> None:
    display = df.copy()
    for col in ("LR mean NB", "DNN mean NB", "DNN - LR"):
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    for col in ("LR > default", "DNN > default"):
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{100.0 * x:.0f}%")
    display["Outcome family"] = display["Outcome family"].map(lambda s: str(s).upper())

    fig_h = 1.65 + 0.55 * max(1, len(display))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")
    ax.set_title(f"{dataset_label} | {run_tag} | {analysis_kind.upper()} | Decision-curve summary", fontsize=17, pad=16)

    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=display.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cfcfcf")
        if r == 0:
            cell.set_facecolor("#eaf2fb")
            cell.set_text_props(weight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pooled decision curves and compact summary tables.")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--dataset-label", required=True)
    ap.add_argument("--analysis-kind", required=True, choices=["standard", "damage", "repair"])
    ap.add_argument("--metrics-root", default="analysis/ELSA/metrics/decision_curve")
    ap.add_argument("--out-root", default="analysis/ELSA/figures/decision_curve")
    args = ap.parse_args()

    run_tag = resolve_run_tag(args.run_tag, args.scenario, args.method)
    in_dir = Path(args.metrics_root) / run_tag / args.analysis_kind
    out_dir = Path(args.out_root) / run_tag / args.analysis_kind

    pooled = load_curves(in_dir / "pooled_curves.csv")
    summary = load_summary(in_dir / "summary.csv")
    table_df = build_summary_table(summary)

    pooled_plot(
        pooled=pooled,
        summary=summary,
        out_path=out_dir / "pooled_decision_curve.png",
        dataset_label=args.dataset_label,
        run_tag=run_tag,
        analysis_kind=args.analysis_kind,
    )
    save_table_csv(table_df, out_dir / "decision_curve_summary_table.csv")
    table_figure(
        df=table_df,
        out_path=out_dir / "decision_curve_summary_table.png",
        dataset_label=args.dataset_label,
        run_tag=run_tag,
        analysis_kind=args.analysis_kind,
    )

    print(f"[OK] Wrote {out_dir / 'pooled_decision_curve.png'}")
    print(f"[OK] Wrote {out_dir / 'decision_curve_summary_table.csv'}")
    print(f"[OK] Wrote {out_dir / 'decision_curve_summary_table.png'}")


if __name__ == "__main__":
    main()
