from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import upsetplot
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Patch

from .conditions import Conditions, build_indicators
from .style import SubsetRule, UpsetStyle

_FLIP_ALIGN = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


@dataclass
class UpsetResult:
    figure: Figure
    axes: dict[str, Axes]  # "intersections", "matrix", "totals", "shading"
    indicators: pd.DataFrame
    intersections: pd.Series  # size per plotted intersection, in plotted order
    totals: pd.Series  # size per condition, in plotted order


def plot_upset(
    data: pd.DataFrame,
    conditions: Conditions,
    style: Union[UpsetStyle, dict, None] = None,
    fig: Optional[Figure] = None,
) -> UpsetResult:
    """Draw an UpSet plot of ``conditions`` evaluated on ``data``.

    Pass ``fig`` to draw into an existing figure (e.g. one embedded in a GUI);
    it is cleared first and its size is kept.
    """
    if style is None:
        style = UpsetStyle()
    elif isinstance(style, dict):
        style = UpsetStyle.from_dict(style)
    style.validate()

    indicators = build_indicators(data, conditions)
    _check_names(style, list(indicators.columns))

    horizontal = style.intersections_side in ("top", "bottom")
    if horizontal:
        mirror_rows = style.intersections_side == "bottom"
        mirror_cols = style.conditions_side == "right"
    else:
        mirror_rows = style.conditions_side == "bottom"
        mirror_cols = style.intersections_side == "left"

    keep_size = fig is not None or style.figsize is not None
    upset = _MirroredUpSet(
        upsetplot.from_indicators(indicators),
        mirror_rows=mirror_rows,
        mirror_cols=mirror_cols,
        orientation="horizontal" if horizontal else "vertical",
        subset_size="count",
        sort_by=style.sort_by,
        sort_categories_by=style.sort_conditions_by,
        min_subset_size=style.min_subset_size,
        max_subset_size=style.max_subset_size,
        min_degree=style.min_degree,
        max_degree=style.max_degree,
        show_counts=style.show_counts,
        show_percentages=style.show_percentages,
        with_lines=style.with_lines,
        # upsetplot resizes the figure to fit element_size; a fixed size must win
        element_size=None if keep_size else style.element_size,
        include_empty_subsets=style.include_empty_subsets,
    )
    if len(upset.intersections) == 0:
        raise ValueError("No intersections remain after applying the size/degree filters.")

    if fig is None:
        fig = plt.figure(figsize=style.figsize) if style.figsize else None
    else:
        fig.clear()

    with warnings.catch_warnings():
        # upsetplot 0.9 uses chained inplace fillna, which pandas 2.x warns about
        warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
        axes = {k: v for k, v in upset.plot(fig=fig).items() if v is not None}
    fig = axes["matrix"].figure

    _fix_text_positions(axes)
    _mirror_value_axes(axes, horizontal, mirror_rows, mirror_cols)
    _place_condition_labels(axes, horizontal, mirror_rows, mirror_cols)
    _apply_style(fig, axes, upset, style, horizontal)
    _fit_totals_counts(axes["totals"], horizontal)

    return UpsetResult(
        figure=fig,
        axes=axes,
        indicators=indicators,
        intersections=upset.intersections,
        totals=upset.totals,
    )


class _MirroredUpSet(upsetplot.UpSet):
    """UpSet whose grid can be flipped, placing panels on the opposite sides."""

    def __init__(self, *args, mirror_rows: bool, mirror_cols: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._mirror_rows = mirror_rows
        self._mirror_cols = mirror_cols
        self._base_totals_elements = self._totals_plot_elements

    def make_grid(self, fig=None):
        if fig is None:
            fig = plt.gcf()
        self._totals_plot_elements = self._totals_elements_fitting_counts(fig)
        specs = super().make_grid(fig)
        # upsetplot puts label text between totals and matrix; move totals next to the matrix
        gs = specs["gs"]
        nrows, ncols = gs.get_geometry()
        n_inters, n_totals = len(self.intersections), self._totals_plot_elements
        totals = specs["totals"]
        if self._horizontal:
            specs["totals"] = gs[totals.rowspan.start : totals.rowspan.stop, ncols - n_inters - n_totals : ncols - n_inters]
        else:
            specs["totals"] = gs[nrows - n_inters - n_totals : nrows - n_inters, totals.colspan.start : totals.colspan.stop]
        return {k: self._mirror(v) if isinstance(v, SubplotSpec) else v for k, v in specs.items()}

    def _totals_elements_fitting_counts(self, fig: Figure) -> int:
        """Widen the totals panel so its bars keep their base length next to the count labels."""
        base = self._base_totals_elements
        if not base or not (self._show_counts or self._show_percentages):
            return base

        def measure(text: str) -> float:
            artist = fig.text(0, 0, text)
            extent = artist.get_window_extent()
            artist.remove()
            return extent.width if self._horizontal else extent.height

        labels = []
        for value in self.totals:
            parts = [f"{value:.0f}"] if self._show_counts else []
            if self._show_percentages:
                pct = f"{value / self.total:.1%}"
                parts.append(f"({pct})" if parts else pct)
            labels.append((" " if self._horizontal else "\n").join(parts))
        needed = max(measure(label) for label in labels) * 1.1 + 6

        length = fig.bbox.width if self._horizontal else fig.bbox.height
        n_inters = len(self.intersections)
        if self._element_size is not None:
            colw = self._element_size / 72 * fig.dpi
            return base + int(np.ceil(needed / colw))
        # without element_size upsetplot divides the free length evenly across columns
        names_width = fig.text(0, 0, "\n".join(f"{n}x" for n in self.totals.index)).get_window_extent()
        fig.texts[-1].remove()
        for extra in range(0, 60):
            colw = (length - names_width.width) / (n_inters + base + extra)
            if extra * colw >= needed:
                return base + extra
        return base

    def _mirror(self, spec: SubplotSpec) -> SubplotSpec:
        gs = spec.get_gridspec()
        nrows, ncols = gs.get_geometry()
        rows, cols = spec.rowspan, spec.colspan
        if self._mirror_rows:
            rows = range(nrows - rows.stop, nrows - rows.start)
        if self._mirror_cols:
            cols = range(ncols - cols.stop, ncols - cols.start)
        return gs[rows.start : rows.stop, cols.start : cols.stop]


def _check_names(style: UpsetStyle, names: list[str]) -> None:
    known = set(names)
    referenced = set(style.condition_styles)
    for rule in style.subset_rules:
        referenced.update(rule.present, rule.absent)
    missing = referenced - known
    if missing:
        raise ValueError(f"Style references unknown conditions: {sorted(missing)}. Known: {names}")


def _fix_text_positions(axes: dict[str, Axes]) -> None:
    # upsetplot passes 1-element arrays as text coordinates; matplotlib >= 3.11 rejects them
    for ax in axes.values():
        for text in ax.texts:
            x, y = text.get_position()
            text.set_position((float(np.ravel(x)[0]), float(np.ravel(y)[0])))


def _place_condition_labels(axes, horizontal: bool, mirror_rows: bool, mirror_cols: bool) -> None:
    """Show condition names on the outer side of the totals panel instead of on the matrix."""
    totals = axes["totals"]
    if horizontal:
        axes["matrix"].tick_params(axis="y", labelleft=False, labelright=False)
        axis, side = totals.yaxis, ("right" if mirror_cols else "left")
    else:
        axes["matrix"].tick_params(axis="x", labeltop=False, labelbottom=False)
        axis, side = totals.xaxis, ("bottom" if mirror_rows else "top")
    axis.set_visible(True)
    axis.set_ticks_position(side)
    totals.tick_params(axis="y" if horizontal else "x", length=0)


def _fit_totals_counts(totals: Axes, horizontal: bool) -> None:
    """Extend the totals value axis so count labels stay inside it, clear of the condition names."""
    if not totals.texts:
        return
    fig = totals.figure
    fig.draw_without_rendering()
    box = totals.get_window_extent()
    length = box.width if horizontal else box.height
    get_lim, set_lim = (totals.get_xlim, totals.set_xlim) if horizontal else (totals.get_ylim, totals.set_ylim)

    needed = 0.0
    for bar, text in zip(totals.patches, totals.texts):
        extent = text.get_window_extent()
        text_len = extent.width if horizontal else extent.height
        free = length - text_len - 0.03 * length
        if free <= 0:
            return
        value = bar.get_width() if horizontal else bar.get_height()
        needed = max(needed, value * length / free)

    lo, hi = get_lim()
    if needed > max(lo, hi):
        set_lim((needed, 0) if lo > hi else (0, needed))
        margin = 0.01 * needed
        for bar, text in zip(totals.patches, totals.texts):
            x, y = text.get_position()
            if horizontal:
                text.set_position((bar.get_width() + margin, y))
            else:
                text.set_position((x, bar.get_height() + margin))


def _mirror_value_axes(axes, horizontal: bool, mirror_rows: bool, mirror_cols: bool) -> None:
    x_valued = axes.get("totals" if horizontal else "intersections")
    y_valued = axes.get("intersections" if horizontal else "totals")

    if mirror_cols:
        for ax in axes.values():
            _swap_sides(ax, ax.yaxis, "left", "right")
        if x_valued is not None:
            x_valued.invert_xaxis()
            for text in x_valued.texts:
                text.set_horizontalalignment(_FLIP_ALIGN[text.get_horizontalalignment()])

    if mirror_rows:
        for ax in axes.values():
            _swap_sides(ax, ax.xaxis, "bottom", "top")
        if y_valued is not None:
            y_valued.invert_yaxis()
            for text in y_valued.texts:
                text.set_verticalalignment(_FLIP_ALIGN[text.get_verticalalignment()])


def _swap_sides(ax: Axes, axis, a: str, b: str) -> None:
    vis_a, vis_b = ax.spines[a].get_visible(), ax.spines[b].get_visible()
    ax.spines[a].set_visible(vis_b)
    ax.spines[b].set_visible(vis_a)
    position = axis.get_ticks_position()
    if position in (a, b):
        other = b if position == a else a
        axis.set_ticks_position(other)
        axis.set_label_position(other)


def _matches(rule: SubsetRule, membership: dict[str, bool], size: float) -> bool:
    degree = sum(membership.values())
    return (
        all(membership[name] for name in rule.present)
        and not any(membership[name] for name in rule.absent)
        and (rule.min_degree is None or degree >= rule.min_degree)
        and (rule.max_degree is None or degree <= rule.max_degree)
        and (rule.min_size is None or size >= rule.min_size)
        and (rule.max_size is None or size <= rule.max_size)
    )


def _subset_overrides(upset: upsetplot.UpSet, rules: list[SubsetRule]) -> list[dict]:
    names = list(upset.intersections.index.names)
    overrides = []
    for combo, size in upset.intersections.items():
        membership = dict(zip(names, map(bool, combo)))
        resolved: dict = {}
        for rule in rules:
            if _matches(rule, membership, size):
                for attr in ("bar_color", "active_dot_color", "inactive_dot_color", "line_color", "count_color"):
                    value = getattr(rule, attr)
                    if value is not None:
                        resolved[attr] = value
        overrides.append(resolved)
    return overrides


def _pick(*candidates):
    return next(c for c in candidates if c is not None)


def _apply_style(fig: Figure, axes: dict[str, Axes], upset, style: UpsetStyle, horizontal: bool) -> None:
    subset_over = _subset_overrides(upset, style.subset_rules)
    cond_names = list(upset.intersections.index.names)
    cond_style = {name: style.condition_styles.get(name) for name in cond_names}

    def cond_attr(name, attr):
        s = cond_style.get(name)
        return getattr(s, attr) if s is not None else None

    fig.set_facecolor(style.background_color)
    for ax in axes.values():
        ax.set_facecolor("none")
        # grid lines otherwise draw over the totals bars
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(style.axis_color)
        ax.tick_params(colors=style.axis_color, grid_color=style.grid_color)
        ax.xaxis.label.set_color(style.axis_color)
        ax.yaxis.label.set_color(style.axis_color)

    matrix = axes["matrix"]
    inclusion = upset.intersections.index.to_frame().to_numpy(dtype=bool)
    hide_under_lines = style.with_lines and style.hide_inactive_dots_on_lines

    def dot_color(i, j, name):
        if inclusion[i, j]:
            return _pick(subset_over[i].get("active_dot_color"), cond_attr(name, "active_dot_color"), style.active_dot_color)
        if hide_under_lines:
            active = np.flatnonzero(inclusion[i])
            if len(active) >= 2 and active[0] < j < active[-1]:
                return "none"
        return _pick(subset_over[i].get("inactive_dot_color"), cond_attr(name, "inactive_dot_color"), style.inactive_dot_color)

    dot_colors = [
        dot_color(i, j, name)
        for i in range(len(inclusion))
        for j, name in enumerate(cond_names)
    ]
    dots = next(c for c in matrix.collections if isinstance(c, PathCollection))
    lines = next((c for c in matrix.collections if isinstance(c, LineCollection)), None)
    if lines is not None:
        with_line = np.flatnonzero(inclusion.any(axis=1))
        lines.set_colors([_pick(subset_over[i].get("line_color"), style.line_color) for i in with_line])

    active = inclusion.ravel()
    active_colors = [c if a else "none" for c, a in zip(dot_colors, active)]
    dots.set_facecolors(active_colors)
    dots.set_edgecolors(active_colors)
    if not active.all():
        # a separate collection so inactive dots sit below the connecting lines
        offsets = dots.get_offsets()
        matrix.scatter(
            offsets[~active, 0],
            offsets[~active, 1],
            s=dots.get_sizes()[0],
            c=[c for c, a in zip(dot_colors, active) if not a],
            linewidths=0,
            zorder=(lines.get_zorder() if lines is not None else dots.get_zorder()) - 1,
        )

    label_axis = axes["totals"].yaxis if horizontal else axes["totals"].xaxis
    for label in label_axis.get_ticklabels():
        name = label.get_text()
        label.set_color(_pick(cond_attr(name, "label_color"), style.label_color))
        if not horizontal:
            label.set_rotation(90)

    inters = axes["intersections"]
    for i, bar in enumerate(inters.patches):
        color = _pick(subset_over[i].get("bar_color"), style.intersection_bar_color)
        bar.set_facecolor(color)
        bar.set_edgecolor(color)
    for i, text in enumerate(inters.texts):
        text.set_color(_pick(subset_over[i].get("count_color"), style.count_color))

    totals_order = list(upset.totals.index)
    if "totals" in axes:
        for name, bar in zip(totals_order, axes["totals"].patches):
            color = _pick(cond_attr(name, "totals_bar_color"), style.totals_bar_color)
            bar.set_facecolor(color)
            bar.set_edgecolor(color)
        for name, text in zip(totals_order, axes["totals"].texts):
            text.set_color(_pick(cond_attr(name, "count_color"), style.count_color))
    for k, (name, patch) in enumerate(zip(totals_order, axes["shading"].patches)):
        default = style.shading_color if k % 2 == 0 else style.alternate_shading_color
        patch.set_facecolor(_pick(cond_attr(name, "shading_color"), default))

    value_label = inters.set_ylabel if horizontal else inters.set_xlabel
    value_label(style.intersections_label, color=style.axis_color)
    if "totals" in axes and style.totals_label:
        totals_label = axes["totals"].set_xlabel if horizontal else axes["totals"].set_ylabel
        totals_label(style.totals_label, color=style.axis_color)

    legend_entries = {rule.label: rule.bar_color for rule in style.subset_rules if rule.label}
    if legend_entries:
        inters.legend(
            [Patch(facecolor=c, edgecolor=c) for c in legend_entries.values()],
            list(legend_entries),
            facecolor=style.background_color,
            edgecolor=style.axis_color,
            labelcolor=style.legend_text_color,
        )

    if style.title:
        fig.suptitle(style.title, color=style.title_color)
