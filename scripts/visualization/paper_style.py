"""Shared publication plotting style for manuscript figures."""

from __future__ import annotations

from functools import wraps

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

MIN_AXIS_LABEL_SIZE = 17
MIN_TICK_LABEL_SIZE = 13
MIN_LEGEND_SIZE = 13


def apply_paper_style(remove_titles: bool = True) -> None:
    """Use readable axis fonts and optionally suppress plot titles.

    The manuscript supplies figure captions and panel letters in LaTeX, so PNG
    titles are intentionally omitted to avoid duplicate headings in the paper.
    """
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": MIN_AXIS_LABEL_SIZE,
            "axes.titlesize": 1,
            "xtick.labelsize": MIN_TICK_LABEL_SIZE,
            "ytick.labelsize": MIN_TICK_LABEL_SIZE,
            "legend.fontsize": MIN_LEGEND_SIZE,
            "figure.titlesize": 1,
        }
    )

    if not remove_titles or getattr(plt, "_paper_style_titles_removed", False):
        return

    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel
    original_tick_params = matplotlib.axes.Axes.tick_params
    original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels
    original_set_yticklabels = matplotlib.axes.Axes.set_yticklabels

    def no_title(self, label="", *args, **kwargs):
        return self.title

    def no_suptitle(self, *args, **kwargs):
        return None

    def no_pyplot_title(*args, **kwargs):
        return None

    @wraps(original_set_xlabel)
    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        kwargs["fontsize"] = max(int(kwargs.get("fontsize", 0) or 0), MIN_AXIS_LABEL_SIZE)
        return original_set_xlabel(self, xlabel, fontdict=fontdict, labelpad=labelpad, loc=loc, **kwargs)

    @wraps(original_set_ylabel)
    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        kwargs["fontsize"] = max(int(kwargs.get("fontsize", 0) or 0), MIN_AXIS_LABEL_SIZE)
        return original_set_ylabel(self, ylabel, fontdict=fontdict, labelpad=labelpad, loc=loc, **kwargs)

    @wraps(original_tick_params)
    def tick_params(self, axis="both", **kwargs):
        if "labelsize" in kwargs:
            kwargs["labelsize"] = max(int(kwargs["labelsize"]), MIN_TICK_LABEL_SIZE)
        return original_tick_params(self, axis=axis, **kwargs)

    @wraps(original_set_xticklabels)
    def set_xticklabels(self, labels, *args, **kwargs):
        if "fontsize" in kwargs:
            kwargs["fontsize"] = max(int(kwargs["fontsize"]), MIN_TICK_LABEL_SIZE)
        return original_set_xticklabels(self, labels, *args, **kwargs)

    @wraps(original_set_yticklabels)
    def set_yticklabels(self, labels, *args, **kwargs):
        if "fontsize" in kwargs:
            kwargs["fontsize"] = max(int(kwargs["fontsize"]), MIN_TICK_LABEL_SIZE)
        return original_set_yticklabels(self, labels, *args, **kwargs)

    matplotlib.axes.Axes.set_title = no_title
    matplotlib.figure.Figure.suptitle = no_suptitle
    matplotlib.axes.Axes.set_xlabel = set_xlabel
    matplotlib.axes.Axes.set_ylabel = set_ylabel
    matplotlib.axes.Axes.tick_params = tick_params
    matplotlib.axes.Axes.set_xticklabels = set_xticklabels
    matplotlib.axes.Axes.set_yticklabels = set_yticklabels
    plt.title = no_pyplot_title
    plt._paper_style_titles_removed = True
