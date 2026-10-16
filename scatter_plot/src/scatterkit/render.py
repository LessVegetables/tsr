"""Draws the scatter matrix. Every visual decision is read from ``style``;
the defaults and their meaning are in ``defaults.py``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator, NullLocator, StrMethodFormatter
from scipy.stats import gaussian_kde

from .config import Style
from .defaults import FONT_FAMILY
from .stats import Fit, fit_polynomial, pair_stats

_RC = {"font.family": [FONT_FAMILY]}
_SIDES = {"x": ("bottom", "top"), "y": ("left", "right")}


@dataclass
class _Group:
    label: Optional[str]        # None: all points, no hue
    mask: np.ndarray
    color: Any
    marker: str


@dataclass
class _Prepared:
    features: list[str]
    labels: list[str]
    values: list[np.ndarray]
    limits: list[tuple[float, float]]
    groups: list[_Group]        # what the points are split into
    pooled: _Group              # everything, for a single regression
    hue_name: Optional[str]


@dataclass
class ScatterResult:
    figure: Figure
    axes: dict[tuple[str, str], Axes]   # (row feature, column feature) -> panel
    stats: pd.DataFrame                 # one row per scatter panel
    style: Any

    def savefig(self, path: Union[str, Path, None] = None) -> Path:
        s = self.style
        path = Path(path or s.output_path)
        with mpl.rc_context(_RC):
            self.figure.savefig(
                path, dpi=s.output_dpi, transparent=s.output_transparent,
                bbox_inches="tight" if s.output_tight_crop else None,
                pad_inches=s.output_tight_pad,
            )
        return path

    def show(self) -> None:
        with mpl.rc_context(_RC):
            plt.show()


def plot_scatter_matrix(data: pd.DataFrame, style: Optional[Style] = None) -> ScatterResult:
    """Scatter plots of every pair of ``style.features`` (all numeric columns
    when empty), as an N x N matrix or one triangle of it."""
    style = style if style is not None else Style()
    _check_font()
    with mpl.rc_context(_RC):
        result = _build(data, style)
        # Tick labels only exist after a draw; pin the family on every text so
        # the figure keeps it outside this rc_context too.
        result.figure.canvas.draw()
        for text in result.figure.findobj(Text):
            text.set_fontfamily(FONT_FAMILY)
    return result


def _check_font() -> None:
    try:
        font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    except ValueError:
        warnings.warn(f"font '{FONT_FAMILY}' is not installed; matplotlib will substitute another", stacklevel=3)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _prepare(data: pd.DataFrame, style) -> _Prepared:
    hue_name = style.hue_column or None
    if hue_name is not None and hue_name not in data.columns:
        raise ValueError(f"hue_column: no column '{hue_name}' in the data")

    features = list(style.features)
    if not features:
        features = [c for c in data.select_dtypes("number").columns if c != hue_name]
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(f"features: no such column(s): {', '.join(map(str, missing))}")
    if not features:
        raise ValueError("no numeric columns to plot")

    labels = list(style.feature_labels) or [str(f) for f in features]
    if len(labels) != len(features):
        raise ValueError(f"feature_labels: {len(labels)} names for {len(features)} features")

    if style.max_points and len(data) > style.max_points:
        data = data.sample(style.max_points, random_state=style.random_seed)

    values = [pd.to_numeric(data[f], errors="coerce").to_numpy(dtype=float) for f in features]
    limits = [_limits(v, style.axis_limits_padding) for v in values]

    everything = np.ones(len(data), dtype=bool)
    pooled = _Group(None, everything, style.points_color, style.points_marker)
    if hue_name is None:
        return _Prepared(features, labels, values, limits, [pooled], pooled, None)

    hue = data[hue_name].to_numpy()
    present = pd.unique(data[hue_name].dropna())
    by_text = {str(level): level for level in sorted(present, key=lambda v: (str(type(v)), v))}
    order = list(style.hue_order) or list(by_text)
    unknown = [name for name in order if name not in by_text]
    if unknown:
        raise ValueError(f"hue_order: no such level(s) in '{hue_name}': {', '.join(unknown)}")
    palette, markers = style.points_hue_palette, style.points_hue_markers
    if not palette or not markers:
        raise ValueError("points_hue_palette and points_hue_markers must not be empty")
    groups = [
        _Group(name, hue == by_text[name], palette[k % len(palette)], markers[k % len(markers)])
        for k, name in enumerate(order)
    ]
    return _Prepared(features, labels, values, limits, groups, pooled, hue_name)


def _limits(values: np.ndarray, padding: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    if span == 0:
        return lo - 0.5, hi + 0.5
    return lo - padding * span, hi + padding * span


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def _build(data: pd.DataFrame, style) -> ScatterResult:
    prep = _prepare(data, style)
    n = len(prep.features)
    part = style.matrix_part

    def visible(r: int, c: int) -> bool:
        if r == c:
            return style.diagonal_shown
        return part in ("full", "lower") if c < r else part in ("full", "upper")

    rows = [r for r in range(n) if any(visible(r, c) for c in range(n))]
    cols = [c for c in range(n) if any(visible(r, c) for r in range(n))]
    if not rows:
        raise ValueError("nothing to draw: one feature and diagonal_shown=false")

    # Geometry in inches; a given figure size squeezes the panels instead.
    left, right = style.figure_margin_left, style.figure_margin_right
    top, bottom = style.figure_margin_top, style.figure_margin_bottom
    gap_w, gap_h = style.panel_gap_horizontal, style.panel_gap_vertical
    panel_w, panel_h = style.panel_width, style.panel_height
    width, height = style.figure_width, style.figure_height
    if width is None:
        width = left + right + len(cols) * panel_w + (len(cols) - 1) * gap_w
    else:
        panel_w = (width - left - right - (len(cols) - 1) * gap_w) / len(cols)
    if height is None:
        height = top + bottom + len(rows) * panel_h + (len(rows) - 1) * gap_h
    else:
        panel_h = (height - top - bottom - (len(rows) - 1) * gap_h) / len(rows)
    if panel_w <= 0 or panel_h <= 0:
        raise ValueError("figure_width / figure_height leave no room for the panels")

    fig = plt.figure(figsize=(width, height), dpi=style.output_dpi,
                     facecolor=style.figure_background_color)
    axes: dict[tuple[str, str], Axes] = {}
    records = []

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if not visible(r, c):
                continue
            ax = fig.add_axes((
                (left + j * (panel_w + gap_w)) / width,
                (height - top - (i + 1) * panel_h - i * gap_h) / height,
                panel_w / width, panel_h / height,
            ))
            axes[prep.features[r], prep.features[c]] = ax
            ax.set_facecolor(style.panel_background_color)
            ax.set_xlim(prep.limits[c])
            ax.set_ylim(prep.limits[r])

            if r == c:
                _draw_diagonal(ax, style, prep, r)
            else:
                found = _draw_scatter(ax, style, prep, r, c)
                records.append({"y": prep.features[r], "x": prep.features[c], **found})

            x_outer = not any(visible(below, c) for below in rows[i + 1:])
            y_outer = not any(visible(r, before) for before in cols[:j])
            _style_axes(ax, style, x_outer, y_outer)
            _label_axes(ax, style, prep.labels[c], prep.labels[r], x_outer, y_outer)

    if style.axis_labels_aligned:
        fig.align_labels()
    _draw_title(fig, style, left / width, 1 - right / width, 1 - top / height, height)
    _draw_legend(fig, style, prep)
    return ScatterResult(fig, axes, pd.DataFrame(records), style)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def _draw_scatter(ax: Axes, style, prep: _Prepared, r: int, c: int) -> dict:
    x, y = prep.values[c], prep.values[r]

    if style.points_shown:
        for group in prep.groups:
            ax.scatter(
                x[group.mask], y[group.mask],
                s=style.points_size, color=group.color, marker=group.marker,
                alpha=style.points_alpha, edgecolors=style.points_edge_color,
                linewidths=style.points_edge_width, zorder=style.points_zorder, gid="points",
            )

    pooled_fit = fit_polynomial(x, y, style.regression_degree)
    if style.regression_shown:
        if style.regression_per_hue:
            for group in prep.groups:
                fit = fit_polynomial(x[group.mask], y[group.mask], style.regression_degree)
                _draw_fit(ax, style, fit, x[group.mask & np.isfinite(y)], group.color)
        else:
            _draw_fit(ax, style, pooled_fit, x[np.isfinite(y)], prep.pooled.color)

    found = pair_stats(x, y, pooled_fit)
    if style.annotation_shown:
        _draw_annotation(ax, style, found)
    return found


def _draw_fit(ax: Axes, style, fit: Optional[Fit], x: np.ndarray, group_color) -> None:
    if fit is None:
        return
    if style.regression_line_span == "axes":
        lo, hi = ax.get_xlim()
    else:
        lo, hi = np.nanmin(x), np.nanmax(x)
    grid = np.linspace(lo, hi, style.regression_line_resolution)
    center = fit.predict(grid)
    color = style.regression_line_color if style.regression_line_color is not None else group_color

    edges = sorted(style.regression_line_predict_bands)
    if style.regression_line_predict_bands_shown and edges:
        error = fit.standard_error(grid, style.regression_line_predict_bands_kind)
        zorder = style.regression_line_predict_bands_zorder
        if style.regression_line_predict_bands_fill_shown:
            upper = center + edges[-1] * error if len(edges) > 1 else center
            fill_color = style.regression_line_predict_bands_fill_color
            ax.fill_between(
                grid, center + edges[0] * error, upper,
                color=fill_color if fill_color is not None else color,
                alpha=style.regression_line_predict_bands_fill_alpha,
                linewidth=0, zorder=zorder, gid="regression_band_fill",
            )
        if style.regression_line_predict_bands_line_shown:
            line_color = style.regression_line_predict_bands_line_color
            for edge in edges:
                ax.plot(
                    grid, center + edge * error,
                    color=line_color if line_color is not None else color,
                    linewidth=style.regression_line_predict_bands_line_width,
                    linestyle=style.regression_line_predict_bands_line_style,
                    alpha=style.regression_line_predict_bands_line_alpha,
                    zorder=zorder, gid="regression_band_line",
                )

    if style.regression_line_shown:
        ax.plot(
            grid, center, color=color, linewidth=style.regression_line_width,
            linestyle=style.regression_line_style, alpha=style.regression_line_alpha,
            zorder=style.regression_line_zorder, gid="regression_line",
        )


def _draw_annotation(ax: Axes, style, found: dict) -> None:
    try:
        text = style.annotation_format.format(**found)
    except (KeyError, IndexError) as error:
        raise ValueError(
            f"annotation_format: unknown field {error}; available: {', '.join(found)}"
        ) from None
    box = None
    if style.annotation_box_shown:
        box = dict(
            boxstyle=f"square,pad={style.annotation_box_padding / style.annotation_font_size}",
            facecolor=style.annotation_box_color, alpha=style.annotation_box_alpha,
            edgecolor=style.annotation_box_edge_color, linewidth=style.annotation_box_edge_width,
        )
    ax.text(
        style.annotation_x, style.annotation_y, text, transform=ax.transAxes,
        ha=style.annotation_align_horizontal, va=style.annotation_align_vertical,
        fontsize=style.annotation_font_size, fontweight=style.annotation_font_weight,
        fontstyle=style.annotation_font_style, color=style.annotation_color,
        bbox=box, zorder=style.annotation_zorder, gid="annotation",
    )


def _draw_diagonal(ax: Axes, style, prep: _Prepared, r: int) -> None:
    kind = style.diagonal_kind
    if kind == "name":
        ax.text(
            0.5, 0.5, prep.labels[r], transform=ax.transAxes, ha="center", va="center",
            fontsize=style.diagonal_name_font_size, fontweight=style.diagonal_name_font_weight,
            fontstyle=style.diagonal_name_font_style, color=style.diagonal_name_color,
        )
        return

    values = prep.values[r]
    finite = np.isfinite(values)
    total = int(finite.sum())
    if total == 0:
        return
    lo, hi = prep.limits[r]

    # Densities share one scale (each group weighted by its share of the
    # points), so bars and curves are comparable within the panel.
    bars, curves = [], []
    if "hist" in kind:
        edges = np.histogram_bin_edges(values[finite], bins=style.diagonal_hist_bins or "auto")
        for group in prep.groups:
            counts, _ = np.histogram(values[finite & group.mask], bins=edges)
            bars.append((group, counts / (total * np.diff(edges))))
    if "kde" in kind:
        grid = np.linspace(lo, hi, style.diagonal_kde_resolution)
        for group in prep.groups:
            part = values[finite & group.mask]
            if part.size < 2 or np.ptp(part) == 0:
                continue
            scale = style.diagonal_kde_bandwidth_scale
            kde = gaussian_kde(part, bw_method=lambda k: k.scotts_factor() * scale)
            curves.append((group, kde(grid) * part.size / total))

    peak = max([d.max() for _, d in bars] + [d.max() for _, d in curves], default=0.0)
    if peak <= 0:
        return
    # The panel's y axis keeps the feature's own scale (its numbers label the
    # whole row), so the density is drawn rescaled into that range.
    base = lo
    to_axis = style.diagonal_height * (hi - lo) / peak

    for group, density in bars:
        color = style.diagonal_hist_color
        ax.bar(
            (edges[:-1] + edges[1:]) / 2, density * to_axis,
            width=np.diff(edges) * style.diagonal_hist_bar_width, bottom=base,
            color=color if color is not None else group.color,
            alpha=style.diagonal_hist_alpha, edgecolor=style.diagonal_hist_edge_color,
            linewidth=style.diagonal_hist_edge_width, zorder=style.diagonal_hist_zorder,
            gid="diagonal_hist",
        )
    for group, density in curves:
        if style.diagonal_kde_fill_shown:
            color = style.diagonal_kde_fill_color
            ax.fill_between(
                grid, base, base + density * to_axis,
                color=color if color is not None else group.color,
                alpha=style.diagonal_kde_fill_alpha, linewidth=0,
                zorder=style.diagonal_kde_zorder, gid="diagonal_kde_fill",
            )
        if style.diagonal_kde_line_shown:
            color = style.diagonal_kde_line_color
            ax.plot(
                grid, base + density * to_axis,
                color=color if color is not None else group.color,
                linewidth=style.diagonal_kde_line_width, linestyle=style.diagonal_kde_line_style,
                alpha=style.diagonal_kde_line_alpha, zorder=style.diagonal_kde_zorder,
                gid="diagonal_kde_line",
            )


# ---------------------------------------------------------------------------
# axes furniture
# ---------------------------------------------------------------------------

def _style_axes(ax: Axes, style, x_outer: bool, y_outer: bool) -> None:
    for side in ("left", "bottom", "right", "top"):
        spine = ax.spines[side]
        spine.set_visible(getattr(style, f"spine_{side}_shown"))
        side_color = getattr(style, f"spine_{side}_color")
        side_width = getattr(style, f"spine_{side}_width")
        spine.set_edgecolor(side_color if side_color is not None else style.spine_color)
        spine.set_linewidth(side_width if side_width is not None else style.spine_width)
        spine.set_linestyle(style.spine_style)
        if style.spine_offset:
            spine.set_position(("outward", style.spine_offset))

    for name, axis, outer in (("x", ax.xaxis, x_outer), ("y", ax.yaxis, y_outer)):
        first, second = _SIDES[name]

        count = getattr(style, f"ticks_{name}_major_count")
        if count is not None:
            exact = style.ticks_major_count_mode == "exact"
            axis.set_major_locator(LinearLocator(count) if exact else MaxNLocator(nbins=max(count - 1, 1)))
        between = getattr(style, f"ticks_{name}_minor_count")
        axis.set_minor_locator(AutoMinorLocator(between + 1) if between > 0 else NullLocator())

        for which in ("major", "minor"):
            shown = getattr(style, f"ticks_{name}_{which}_shown")
            ax.tick_params(
                axis=name, which=which,
                length=getattr(style, f"ticks_{name}_{which}_length"),
                width=getattr(style, f"ticks_{name}_{which}_width"),
                color=getattr(style, f"ticks_{name}_{which}_color"),
                direction=getattr(style, f"ticks_{name}_{which}_direction"),
                **{side: shown and getattr(style, f"ticks_{name}_on_{side}") for side in (first, second)},
            )

        labelled = getattr(style, f"tick_labels_{name}_shown") and (outer or not style.tick_labels_only_outer)
        ax.tick_params(
            axis=name, which="major",
            labelsize=getattr(style, f"tick_labels_{name}_font_size"),
            labelcolor=getattr(style, f"tick_labels_{name}_color"),
            labelrotation=getattr(style, f"tick_labels_{name}_rotation"),
            pad=getattr(style, f"tick_labels_{name}_pad"),
            **{f"label{first}": labelled, f"label{second}": False},
        )
        number_format = getattr(style, f"tick_labels_{name}_format")
        if number_format is not None:
            axis.set_major_formatter(StrMethodFormatter(number_format))

        # gridlines are drawn by the axis, so its zorder is the grid's
        axis.set_zorder(style.grid_zorder)

    for which in ("major", "minor"):
        if getattr(style, f"grid_{which}_shown"):
            ax.grid(
                True, which=which, axis=style.grid_axis,
                color=getattr(style, f"grid_{which}_color"),
                linewidth=getattr(style, f"grid_{which}_width"),
                linestyle=getattr(style, f"grid_{which}_style"),
                alpha=getattr(style, f"grid_{which}_alpha"),
            )


def _label_axes(ax: Axes, style, x_label: str, y_label: str, x_outer: bool, y_outer: bool) -> None:
    if style.axis_labels_x_shown and (x_outer or not style.axis_labels_only_outer):
        ax.set_xlabel(
            x_label, fontsize=style.axis_labels_x_font_size,
            fontweight=style.axis_labels_x_font_weight, fontstyle=style.axis_labels_x_font_style,
            color=style.axis_labels_x_color, rotation=style.axis_labels_x_rotation,
            labelpad=style.axis_labels_x_pad,
        )
    if style.axis_labels_y_shown and (y_outer or not style.axis_labels_only_outer):
        rotation = style.axis_labels_y_rotation
        upright = abs(rotation) < 45       # a horizontal name has to hang off its right end
        ax.set_ylabel(
            y_label, fontsize=style.axis_labels_y_font_size,
            fontweight=style.axis_labels_y_font_weight, fontstyle=style.axis_labels_y_font_style,
            color=style.axis_labels_y_color, rotation=rotation,
            labelpad=style.axis_labels_y_pad,
            ha="right" if upright else "center", va="center" if upright else "bottom",
        )


def _draw_title(fig: Figure, style, grid_left: float, grid_right: float, grid_top: float,
                height: float) -> None:
    if not (style.title_shown and style.title_text):
        return
    x = {"left": grid_left, "center": (grid_left + grid_right) / 2, "right": grid_right}[style.title_align]
    fig.text(
        x, grid_top + style.title_pad / height, style.title_text,
        ha=style.title_align, va="bottom", fontsize=style.title_font_size,
        fontweight=style.title_font_weight, fontstyle=style.title_font_style,
        color=style.title_color, gid="title",
    )


def _draw_legend(fig: Figure, style, prep: _Prepared) -> None:
    if not style.legend_shown:
        return
    handles = []
    if prep.hue_name is not None:
        handles += [
            Line2D([], [], linestyle="none", marker=group.marker, markersize=style.legend_marker_size,
                   markerfacecolor=group.color, markeredgecolor=style.points_edge_color,
                   markeredgewidth=style.points_edge_width, label=group.label)
            for group in prep.groups
        ]
    if style.legend_regression_shown and style.regression_shown:
        # with one fit per hue level there is no single colour; fall back on the base one
        line_color = style.regression_line_color
        color = line_color if line_color is not None else style.points_color
        if style.regression_line_shown:
            handles.append(Line2D(
                [], [], color=color, linewidth=style.regression_line_width,
                linestyle=style.regression_line_style, alpha=style.regression_line_alpha,
                label=style.legend_regression_label,
            ))
        if style.regression_line_predict_bands_shown and style.regression_line_predict_bands:
            band_line = style.regression_line_predict_bands_line_color
            band_fill = style.regression_line_predict_bands_fill_color
            if style.regression_line_predict_bands_line_shown:
                handles.append(Line2D(
                    [], [], color=band_line if band_line is not None else color,
                    linewidth=style.regression_line_predict_bands_line_width,
                    linestyle=style.regression_line_predict_bands_line_style,
                    alpha=style.regression_line_predict_bands_line_alpha,
                    label=style.legend_bands_label,
                ))
            elif style.regression_line_predict_bands_fill_shown:
                handles.append(Patch(
                    facecolor=band_fill if band_fill is not None else color,
                    alpha=style.regression_line_predict_bands_fill_alpha, linewidth=0,
                    label=style.legend_bands_label,
                ))
    if not handles:
        return

    title = style.legend_title if style.legend_title is not None else prep.hue_name
    legend = fig.legend(
        handles=handles, title=title or None,
        loc=style.legend_location, bbox_to_anchor=(style.legend_anchor_x, style.legend_anchor_y),
        ncols=style.legend_columns, fontsize=style.legend_font_size,
        title_fontsize=style.legend_title_font_size, labelcolor=style.legend_text_color,
        labelspacing=style.legend_label_spacing, handlelength=style.legend_handle_length,
        handletextpad=style.legend_handle_text_pad, columnspacing=style.legend_column_spacing,
        borderpad=style.legend_border_pad, frameon=style.legend_frame_shown,
        facecolor=style.legend_frame_color, framealpha=style.legend_frame_alpha,
        edgecolor=style.legend_frame_edge_color,
    )
    legend.get_frame().set_linewidth(style.legend_frame_edge_width)
    legend.get_title().set_fontweight(style.legend_title_font_weight)
    legend.get_title().set_color(style.legend_text_color)
