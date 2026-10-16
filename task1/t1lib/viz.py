"""[1.3] raincloud plots, [1.4] scatter matrix, [1.6]-[1.8] scatter panels.

Everything visual is driven by :class:`PlotStyle`, so the same call can be
re-run with different axis colours, tick density, fonts, transparency or
toggled-off legends without touching the plotting code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

Estimator = Union[str, float, Callable[[np.ndarray], float]]

DEFAULT_PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")
DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "P")


@dataclass
class PlotStyle:
    """User-regulated visual parameters ([1.4] asks for all of these)."""

    # --- figure geometry -------------------------------------------------
    cell: float = 2.7                 # size of one panel, inches
    square_cells: bool = True         # force a square box per panel
    dpi: int = 110
    facecolor: str = "white"
    wspace: float = 0.28
    hspace: float = 0.28

    # --- axes / spines ---------------------------------------------------
    axis_color: str = "#3A3A3A"
    axis_linewidth: float = 1.1
    visible_spines: Sequence[str] = ("left", "bottom")

    # --- ticks -----------------------------------------------------------
    n_major: int = 5                  # used when *_step is None
    major_step: Optional[float] = None
    n_minor: int = 4                  # minor subdivisions inside one major
    minor_step: Optional[float] = None
    tick_labelsize: float = 8
    tick_color: Optional[str] = None  # None -> axis_color
    major_tick_len: float = 4.0
    minor_tick_len: float = 2.0
    tick_width: float = 1.0

    # --- grid ------------------------------------------------------------
    grid_major: bool = True
    grid_minor: bool = False
    grid_color: str = "#C9C9C9"
    grid_linestyle: str = "-"
    grid_minor_linestyle: str = ":"
    grid_linewidth: float = 0.6
    grid_alpha: float = 0.7

    # --- fonts / toggles -------------------------------------------------
    show_suptitle: bool = True
    show_title: bool = True
    show_axis_labels: bool = True
    show_legend: bool = True
    suptitle_fontsize: float = 14
    title_fontsize: float = 10
    axis_label_fontsize: float = 9.5
    legend_fontsize: float = 9
    annot_fontsize: float = 8
    label_color: str = "#1A1A1A"

    # --- marks -----------------------------------------------------------
    point_size: float = 18
    point_alpha: float = 0.55
    point_linewidth: float = 0.0
    point_edgecolor: str = "none"
    palette: Sequence[str] = DEFAULT_PALETTE
    markers: Sequence[str] = DEFAULT_MARKERS

    # --- densities / rainclouds -----------------------------------------
    density_alpha: float = 0.35
    density_fill: bool = True
    density_linewidth: float = 1.4
    density_gridsize: int = 200
    violin_width: float = 0.38
    jitter_width: float = 0.16
    box_width: float = 0.10

    def colors_for(self, levels: Sequence) -> dict:
        return {lv: self.palette[i % len(self.palette)] for i, lv in enumerate(levels)}

    def markers_for(self, levels: Sequence) -> dict:
        return {lv: self.markers[i % len(self.markers)] for i, lv in enumerate(levels)}

    def derive(self, **changes) -> "PlotStyle":
        """A copy with some knobs turned -- handy for one-off variants."""
        return replace(self, **changes)


def apply_axes_style(ax, style: PlotStyle, square: Optional[bool] = None) -> None:
    """Spines, tick locators, grid and colours for one axes object."""
    tick_color = style.tick_color or style.axis_color
    for name, spine in ax.spines.items():
        spine.set_visible(name in style.visible_spines)
        spine.set_color(style.axis_color)
        spine.set_linewidth(style.axis_linewidth)

    for axis in (ax.xaxis, ax.yaxis):
        if style.major_step is not None:
            axis.set_major_locator(MultipleLocator(style.major_step))
        else:
            axis.set_major_locator(MaxNLocator(nbins=style.n_major, prune=None))
        if style.minor_step is not None:
            axis.set_minor_locator(MultipleLocator(style.minor_step))
        elif style.n_minor and style.n_minor > 1:
            axis.set_minor_locator(AutoMinorLocator(style.n_minor))

    ax.tick_params(
        which="major", labelsize=style.tick_labelsize, colors=tick_color,
        length=style.major_tick_len, width=style.tick_width,
    )
    ax.tick_params(which="minor", length=style.minor_tick_len, width=style.tick_width * 0.8, colors=tick_color)

    ax.set_axisbelow(True)
    # line properties must only be passed when switching a grid ON: matplotlib
    # reads "grid(False, color=...)" as a request to enable it.
    if style.grid_major:
        ax.grid(True, which="major", color=style.grid_color, linestyle=style.grid_linestyle,
                linewidth=style.grid_linewidth, alpha=style.grid_alpha)
    else:
        ax.grid(False, which="major")
    if style.grid_minor:
        ax.grid(True, which="minor", color=style.grid_color, linestyle=style.grid_minor_linestyle,
                linewidth=style.grid_linewidth * 0.75, alpha=style.grid_alpha * 0.7)
    else:
        ax.grid(False, which="minor")
    if square if square is not None else style.square_cells:
        ax.set_box_aspect(1)


# ---------------------------------------------------------------------------
# [1.3] raincloud
# ---------------------------------------------------------------------------

@dataclass
class BoxSpec:
    """Which sample estimates the box summarises -- all three are swappable.

    ``center``  -- "median", "mean", a quantile in [0, 1], or a callable.
    ``box``     -- "iqr", "std", "sem", or an explicit ``(lo_pct, hi_pct)`` pair.
    ``whisker`` -- "range", "1.5iqr", or an explicit ``(lo_pct, hi_pct)`` pair.
    """

    center: Estimator = "median"
    box: Union[str, tuple[float, float]] = "iqr"
    whisker: Union[str, tuple[float, float]] = "range"

    def compute(self, values: np.ndarray) -> dict[str, float]:
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return {k: np.nan for k in ("center", "lo", "hi", "wlo", "whi")}

        if callable(self.center):
            center = float(self.center(v))
        elif self.center == "median":
            center = float(np.median(v))
        elif self.center == "mean":
            center = float(np.mean(v))
        else:
            center = float(np.quantile(v, float(self.center)))

        if self.box == "iqr":
            lo, hi = np.percentile(v, [25, 75])
        elif self.box == "std":
            lo, hi = center - v.std(ddof=1), center + v.std(ddof=1)
        elif self.box == "sem":
            sem = v.std(ddof=1) / np.sqrt(v.size)
            lo, hi = center - sem, center + sem
        else:
            lo, hi = np.percentile(v, list(self.box))

        if self.whisker == "range":
            wlo, whi = v.min(), v.max()
        elif self.whisker == "1.5iqr":
            q1, q3 = np.percentile(v, [25, 75])
            span = 1.5 * (q3 - q1)
            inside = v[(v >= q1 - span) & (v <= q3 + span)]
            wlo, whi = (inside.min(), inside.max()) if inside.size else (v.min(), v.max())
        else:
            wlo, whi = np.percentile(v, list(self.whisker))

        return {"center": center, "lo": float(lo), "hi": float(hi),
                "wlo": float(wlo), "whi": float(whi)}


def _kde(values: np.ndarray, gridsize: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import gaussian_kde

    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2 or np.allclose(v, v[0]):
        return np.array([]), np.array([])
    grid = np.linspace(v.min(), v.max(), gridsize)
    return grid, gaussian_kde(v)(grid)


def _half_violin(ax, values, position, width, color, style: PlotStyle, side: int = +1) -> None:
    grid, dens = _kde(values, style.density_gridsize)
    if grid.size == 0:
        return
    dens = dens / dens.max() * width
    x = position + side * dens
    if style.density_fill:
        ax.fill_betweenx(grid, position, x, color=color, alpha=style.density_alpha, linewidth=0)
    ax.plot(x, grid, color=color, linewidth=style.density_linewidth, alpha=0.95)


def _box(ax, stats: dict, position: float, width: float, color: str, style: PlotStyle) -> None:
    ax.vlines(position, stats["wlo"], stats["whi"], color=color, linewidth=style.axis_linewidth, zorder=4)
    for y in (stats["wlo"], stats["whi"]):  # whisker caps
        ax.hlines(y, position - width * 0.45, position + width * 0.45, color=color,
                  linewidth=style.axis_linewidth, zorder=4)
    ax.add_patch(plt.Rectangle(
        (position - width / 2, stats["lo"]), width, stats["hi"] - stats["lo"],
        facecolor="white", edgecolor=color, linewidth=style.axis_linewidth * 1.2, zorder=5,
    ))
    ax.hlines(stats["center"], position - width / 2, position + width / 2,
              color=color, linewidth=style.axis_linewidth * 2.0, zorder=6)


def raincloud_grid(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    class_col: Optional[str] = None,
    box: BoxSpec = BoxSpec(),
    style: PlotStyle = PlotStyle(),
    ncols: int = 2,
    suptitle: Optional[str] = "[1.3] Raincloud plot по каждому признаку",
    random_state: int = 0,
    annotate: bool = True,
) -> Figure:
    """[1.3] Cloud (half-violin) + rain (jittered points) + box, one panel per feature.

    The box next to each cloud reports ``box.center``, the ``box.box`` spread and
    the ``box.whisker`` extent; swap the :class:`BoxSpec` to summarise the same
    data with different estimates.
    """
    features = list(features if features is not None else df.select_dtypes("number").columns)
    if class_col is not None:
        levels = list(df[class_col].dropna().unique())
        levels = [lv for lv in df[class_col].cat.categories if lv in levels] if hasattr(
            df[class_col], "cat") else sorted(levels)
    else:
        levels = ["все"]
    colors = style.colors_for(levels)
    rng = np.random.default_rng(random_state)

    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(style.cell * 1.9 * ncols, style.cell * 1.35 * nrows),
        dpi=style.dpi, squeeze=False,
    )
    fig.patch.set_facecolor(style.facecolor)

    for ax, feature in zip(axes.ravel(), features):
        for i, level in enumerate(levels):
            values = (df[feature] if class_col is None else df.loc[df[class_col] == level, feature])
            values = values.to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            color = colors[level]
            _half_violin(ax, values, i, style.violin_width, color, style, side=+1)
            jitter = rng.uniform(-style.jitter_width, style.jitter_width, size=values.size)
            ax.scatter(
                i - style.violin_width * 0.55 + jitter, values,
                s=style.point_size * 0.55, color=color, alpha=style.point_alpha,
                linewidths=style.point_linewidth, edgecolors=style.point_edgecolor, zorder=2,
            )
            stats = box.compute(values)
            _box(ax, stats, i + style.violin_width * 0.62, style.box_width, color, style)
            if annotate:
                # above the upper whisker: to the right it would sit on the
                # whisker cap and run into the next class's cloud
                ax.annotate(
                    f"{stats['center']:.2f}", (i + style.violin_width * 0.62, stats["whi"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                    fontsize=style.annot_fontsize, color=color,
                )

        apply_axes_style(ax, style, square=False)
        ax.set_xticks(range(len(levels)), [str(lv) for lv in levels], fontsize=style.tick_labelsize)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(False, axis="x", which="both")
        if style.show_title:
            ax.set_title(feature, fontsize=style.title_fontsize, color=style.label_color)
        if style.show_axis_labels:
            ax.set_ylabel("значение", fontsize=style.axis_label_fontsize, color=style.label_color)
            ax.set_xlabel(class_col or "", fontsize=style.axis_label_fontsize, color=style.label_color)

    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)

    if suptitle and style.show_suptitle:
        centre = box.center if isinstance(box.center, str) else "estimator"
        fig.suptitle(f"{suptitle}  (центр: {centre}, ящик: {box.box}, усы: {box.whisker})",
                     fontsize=style.suptitle_fontsize, color=style.label_color)
    fig.subplots_adjust(wspace=style.wspace + 0.05, hspace=style.hspace + 0.2)
    return fig


# ---------------------------------------------------------------------------
# [1.4] scatter matrix
# ---------------------------------------------------------------------------

def scatter_matrix(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    class_col: Optional[str] = "target",
    style: PlotStyle = PlotStyle(),
    suptitle: Optional[str] = "[1.4] 2D визуализация по всем парам признаков",
    lower_only: bool = False,
    max_points: Optional[int] = None,
    random_state: int = 0,
) -> Figure:
    """[1.4] Off-diagonal: classes as differently coloured/shaped points.
    Diagonal (same variable against itself): per-class density curves.
    """
    features = list(features if features is not None else df.columns.drop(class_col))
    data = df if max_points is None or len(df) <= max_points else df.sample(max_points, random_state=random_state)

    if class_col is not None:
        levels = (list(data[class_col].cat.categories) if hasattr(data[class_col], "cat")
                  else sorted(data[class_col].dropna().unique()))
    else:
        levels = ["все"]
    colors, markers = style.colors_for(levels), style.markers_for(levels)

    k = len(features)
    fig, axes = plt.subplots(k, k, figsize=(style.cell * k, style.cell * k), dpi=style.dpi, squeeze=False)
    fig.patch.set_facecolor(style.facecolor)

    for i, yf in enumerate(features):
        for j, xf in enumerate(features):
            ax = axes[i, j]
            if lower_only and j > i:
                ax.set_visible(False)
                continue

            if i == j:
                for level in levels:
                    part = data if class_col is None else data[data[class_col] == level]
                    grid, dens = _kde(part[xf].to_numpy(dtype=float), style.density_gridsize)
                    if grid.size == 0:
                        continue
                    if style.density_fill:
                        ax.fill_between(grid, 0, dens, color=colors[level],
                                        alpha=style.density_alpha, linewidth=0)
                    ax.plot(grid, dens, color=colors[level], linewidth=style.density_linewidth)
                ax.set_yticks([])
            else:
                for level in levels:
                    part = data if class_col is None else data[data[class_col] == level]
                    ax.scatter(
                        part[xf], part[yf], s=style.point_size, color=colors[level],
                        marker=markers[level], alpha=style.point_alpha,
                        linewidths=style.point_linewidth, edgecolors=style.point_edgecolor,
                        label=str(level),
                    )

            apply_axes_style(ax, style)
            if i < k - 1:
                ax.set_xticklabels([])
            if j > 0 and i != j:
                ax.set_yticklabels([])
            if style.show_axis_labels:
                if i == k - 1:
                    ax.set_xlabel(xf, fontsize=style.axis_label_fontsize, color=style.label_color)
                if j == 0:
                    ax.set_ylabel(yf if i != j else f"{yf} (плотность)",
                                  fontsize=style.axis_label_fontsize, color=style.label_color)

    if style.show_legend and class_col is not None:
        handles = [
            Line2D([], [], linestyle="none", marker=markers[lv], color=colors[lv],
                   markersize=7, label=str(lv))
            for lv in levels
        ]
        fig.legend(handles=handles, title=class_col, fontsize=style.legend_fontsize,
                   title_fontsize=style.legend_fontsize, loc="upper right",
                   frameon=False, bbox_to_anchor=(0.995, 0.995))
    if suptitle and style.show_suptitle:
        fig.suptitle(suptitle, fontsize=style.suptitle_fontsize, color=style.label_color)
    fig.subplots_adjust(wspace=style.wspace, hspace=style.hspace)
    return fig


# ---------------------------------------------------------------------------
# [1.6] [1.7] [1.8] m x n panels of one feature pair across many datasets
# ---------------------------------------------------------------------------

def scatter_panel(
    datasets: Mapping[str, pd.DataFrame],
    pair: tuple[str, str] = ("A", "B"),
    class_col: Optional[str] = "target",
    style: PlotStyle = PlotStyle(),
    ncols: Optional[int] = None,
    suptitle: Optional[str] = None,
    share_limits: bool = True,
    reference: Optional[pd.DataFrame] = None,
    annotate_corr: bool = True,
    ghost_reference: bool = True,
) -> Figure:
    """m x n panel of ``pair`` scatterplots, one panel per dataset.

    With ``reference`` given, its points are drawn underneath in grey and the
    panel title carries the surviving-pair count and Pearson r, which is what
    makes the 0% -> 80% progression readable at a glance.
    """
    items = list(datasets.items())
    xf, yf = pair
    ncols = ncols or min(len(items), 4)
    nrows = int(np.ceil(len(items) / ncols))

    if class_col is not None:
        first = items[0][1]
        levels = (list(first[class_col].cat.categories) if hasattr(first[class_col], "cat")
                  else sorted(first[class_col].dropna().unique()))
    else:
        levels = ["все"]
    colors, markers = style.colors_for(levels), style.markers_for(levels)

    if share_limits:
        base = reference if reference is not None else pd.concat([d for _, d in items])
        xlim = (np.nanmin(base[xf]), np.nanmax(base[xf]))
        ylim = (np.nanmin(base[yf]), np.nanmax(base[yf]))
        pad = 0.05
        xlim = (xlim[0] - pad * np.ptp(xlim), xlim[1] + pad * np.ptp(xlim))
        ylim = (ylim[0] - pad * np.ptp(ylim), ylim[1] + pad * np.ptp(ylim))

    fig, axes = plt.subplots(nrows, ncols, figsize=(style.cell * ncols, style.cell * nrows),
                             dpi=style.dpi, squeeze=False)
    fig.patch.set_facecolor(style.facecolor)

    for ax, (name, data) in zip(axes.ravel(), items):
        if reference is not None and ghost_reference:
            ax.scatter(reference[xf], reference[yf], s=style.point_size * 0.7,
                       color="#D6D6D6", alpha=0.5, linewidths=0, zorder=1)
        for level in levels:
            part = data if class_col is None else data[data[class_col] == level]
            ax.scatter(part[xf], part[yf], s=style.point_size, color=colors[level],
                       marker=markers[level], alpha=style.point_alpha,
                       linewidths=style.point_linewidth, edgecolors=style.point_edgecolor,
                       label=str(level), zorder=2)

        apply_axes_style(ax, style)
        if share_limits:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        if style.show_title:
            title = name
            if annotate_corr:
                both = data[[xf, yf]].dropna()
                r = both[xf].corr(both[yf]) if len(both) > 1 else np.nan
                title = f"{name}\nn={len(both)}, r={r:.3f}" if len(both) > 1 else f"{name}\nn={len(both)}"
            ax.set_title(title, fontsize=style.title_fontsize, color=style.label_color)
        if style.show_axis_labels:
            ax.set_xlabel(xf, fontsize=style.axis_label_fontsize, color=style.label_color)
            ax.set_ylabel(yf, fontsize=style.axis_label_fontsize, color=style.label_color)

    for ax in axes.ravel()[len(items):]:
        ax.set_visible(False)

    if style.show_legend and class_col is not None:
        handles = [Line2D([], [], linestyle="none", marker=markers[lv], color=colors[lv],
                          markersize=7, label=str(lv)) for lv in levels]
        fig.legend(handles=handles, title=class_col, fontsize=style.legend_fontsize,
                   title_fontsize=style.legend_fontsize, loc="upper right", frameon=False,
                   bbox_to_anchor=(0.999, 0.999))
    if suptitle and style.show_suptitle:
        fig.suptitle(suptitle, fontsize=style.suptitle_fontsize, color=style.label_color)
    fig.subplots_adjust(wspace=style.wspace + 0.12, hspace=style.hspace + 0.35)
    return fig
