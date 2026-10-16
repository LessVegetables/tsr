"""Draws the three layers -- cloud (half violin), rain (jittered points) and
the box -- onto one axes.

Everything is written once, in *value* and *position* coordinates, and mapped
to x/y at the last moment by the small helpers at the top. That is the only
place the orientation is allowed to matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, NullLocator

from .stats import STAT_KEYS, BoxSpec, RaincloudError, clean, jitter, kde
from .style import GroupStyle, RainStyle

VALUE, GROUP, HUE = "_value", "_group", "_hue"


@dataclass
class RaincloudResult:
    """What the call drew, and the numbers behind it."""

    figure: Figure
    ax: Axes
    groups: list
    stats: pd.DataFrame          # one row per drawn group, columns from BoxSpec.compute()
    colorbar: Optional[Colorbar] = None

    def savefig(self, path, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("facecolor", self.figure.get_facecolor())
        self.figure.savefig(path, **kwargs)
        return path


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def _horizontal(style: RainStyle) -> bool:
    return style.orientation == "horizontal"


def _flip(style: RainStyle) -> float:
    """+1, or -1 when the group axis runs downwards.

    The offsets in :class:`RainStyle` are written as if positive meant "up".
    Inverting the group axis so the first group reads at the top would mirror
    that and hang the cloud below the rain, so the offsets get negated instead.
    """
    return -1.0 if (_horizontal(style) and style.invert_group_axis) else 1.0


def _oriented(style: RainStyle) -> RainStyle:
    """A copy whose offsets already point the way the axis is drawn."""
    sign = _flip(style)
    if sign > 0:
        return style
    return style.derive(cloud_base=-style.cloud_base, cloud_width=-style.cloud_width,
                        box_offset=-style.box_offset, rain_offset=-style.rain_offset)


def _points(values, positions, style: RainStyle) -> tuple[Any, Any]:
    """(value, position) -> (x, y)."""
    return (values, positions) if _horizontal(style) else (positions, values)


def _span(ax: Axes, position: float, v_from: float, v_to: float, style: RainStyle, **kw) -> None:
    """A segment at a fixed position, running along the value axis."""
    if _horizontal(style):
        ax.hlines(position, v_from, v_to, **kw)
    else:
        ax.vlines(position, v_from, v_to, **kw)


def _cross(ax: Axes, value: float, p_from: float, p_to: float, style: RainStyle, **kw) -> None:
    """A segment at a fixed value, running across the group axis."""
    if _horizontal(style):
        ax.vlines(value, p_from, p_to, **kw)
    else:
        ax.hlines(value, p_from, p_to, **kw)


def _fill(ax: Axes, grid, base, edge, style: RainStyle, **kw):
    if _horizontal(style):
        return ax.fill_between(grid, base, edge, **kw)
    return ax.fill_betweenx(grid, base, edge, **kw)


def _rect(lo: float, hi: float, position: float, width: float, style: RainStyle, **kw) -> Rectangle:
    if _horizontal(style):
        return Rectangle((lo, position - width / 2), hi - lo, width, **kw)
    return Rectangle((position - width / 2, lo), width, hi - lo, **kw)


def _value_axis(ax: Axes, style: RainStyle):
    return ax.xaxis if _horizontal(style) else ax.yaxis


def _group_axis(ax: Axes, style: RainStyle):
    return ax.yaxis if _horizontal(style) else ax.xaxis


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def _tidy(data, value, group, point_hue) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    """Normalise every accepted input shape into a 3-column frame.

    Accepts a DataFrame plus column names, a ``{label: values}`` mapping, or a
    bare sequence of numbers.
    """
    names = {"value": None, "group": None, "hue": None}

    if isinstance(data, pd.DataFrame):
        if value is None:
            numeric = data.select_dtypes("number").columns
            if len(numeric) != 1:
                raise RaincloudError(
                    "value=... is required for a DataFrame with more than one numeric column "
                    f"(found {list(numeric)})"
                )
            value = numeric[0]
        for label, col in (("value", value), ("group", group), ("hue", point_hue)):
            if col is not None and col not in data.columns:
                raise RaincloudError(f"{label} column {col!r} is not in the data ({list(data.columns)})")
        names = {"value": str(value),
                 "group": None if group is None else str(group),
                 "hue": None if point_hue is None else str(point_hue)}
        out = pd.DataFrame({VALUE: pd.to_numeric(data[value], errors="coerce")})
        out[GROUP] = data[group].to_numpy() if group is not None else ""
        out[HUE] = data[point_hue].to_numpy() if point_hue is not None else np.nan
        # a categorical group column carries the order the caller declared
        if group is not None and isinstance(data[group].dtype, pd.CategoricalDtype):
            out[GROUP] = pd.Categorical(out[GROUP], categories=list(data[group].cat.categories))
        return out.reset_index(drop=True), names

    if isinstance(data, Mapping):
        if group is not None or point_hue is not None or value is not None:
            raise RaincloudError("value/group/point_hue only apply to a DataFrame input")
        frames = [pd.DataFrame({VALUE: clean(v), GROUP: label, HUE: np.nan}) for label, v in data.items()]
        stacked = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[VALUE, GROUP, HUE])
        # dict order is the caller's order, so pin it down before anything sorts
        stacked[GROUP] = pd.Categorical(stacked[GROUP], categories=list(data.keys()))
        return stacked, names

    if group is not None or point_hue is not None:
        raise RaincloudError("group/point_hue need a DataFrame input")
    series = pd.Series(np.asarray(data, dtype=float).ravel())
    names["value"] = getattr(data, "name", None)
    return pd.DataFrame({VALUE: series, GROUP: "", HUE: np.nan}), names


def _levels(tidy: pd.DataFrame, style: RainStyle) -> list:
    col = tidy[GROUP]
    if style.group_order is not None:
        missing = [lv for lv in style.group_order if lv not in set(col)]
        if missing:
            raise RaincloudError(f"group_order refers to levels that are not in the data: {missing}")
        return list(style.group_order)
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(col.dropna())
        return [lv for lv in col.cat.categories if lv in present]
    seen = col.dropna().unique()
    try:
        return sorted(seen)
    except TypeError:  # mixed types -- keep first-seen order rather than fail
        return list(seen)


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

def _darken(color, amount: float = 0.45):
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def _resolve(style: RainStyle, override: GroupStyle, field: str, base):
    """Per-group override, then the global setting, then ``base``.

    ``base`` is already the group's colour, so an edge colour can pass ``None``
    and stay unset -- the caller then derives it from the face colour instead of
    accidentally inheriting the flat palette colour.
    """
    value = getattr(override, field)
    if value is None:
        value = getattr(style, field, None)
    return base if value is None else value


class _HueMap:
    """Turns the ``point_hue`` column into per-point colours, and remembers how,
    so the legend or the colorbar can say the same thing."""

    def __init__(self, tidy: pd.DataFrame, style: RainStyle, name: Optional[str]):
        self.style = style
        self.name = name
        self.active = name is not None
        self.numeric = False
        self.mappable = None
        self.levels: list = []

        if not self.active:
            return
        raw = tidy[HUE]
        numeric = pd.to_numeric(raw, errors="coerce")
        # "numeric" only if the column really is numbers -- a coerced column of
        # NaNs would silently paint every point the same colour
        self.numeric = numeric.notna().any() and not isinstance(raw.dtype, pd.CategoricalDtype)

        if self.numeric:
            finite = numeric[np.isfinite(numeric)]
            lo, hi = style.hue_norm if style.hue_norm else (finite.min(), finite.max())
            self.norm = Normalize(vmin=float(lo), vmax=float(hi))
            self.cmap = mpl.colormaps[style.cmap]
            self.mappable = mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
            self.mappable.set_array([])
        else:
            present = raw.dropna()
            self.levels = (list(raw.cat.categories) if isinstance(raw.dtype, pd.CategoricalDtype)
                           else sorted(present.unique(), key=str))
            self.lookup = {lv: style.color_for(i) for i, lv in enumerate(self.levels)}

    def colors(self, chunk: pd.Series, fallback):
        if not self.active:
            return fallback
        if self.numeric:
            return self.cmap(self.norm(pd.to_numeric(chunk, errors="coerce").to_numpy(dtype=float)))
        return [self.lookup.get(v, fallback) for v in chunk]


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

def _draw_cloud(ax: Axes, values, position: float, style: RainStyle, face, edge) -> None:
    grid, dens = kde(values, style.cloud_gridsize, style.cloud_bw_adjust, style.cloud_cut, style.cloud_clip)
    if grid.size == 0:
        return
    base = position + style.cloud_base
    outer = base + dens * style.cloud_width

    _fill(ax, grid, base, outer, style, facecolor=face, alpha=style.cloud_alpha,
          linewidth=0, zorder=style.cloud_zorder)
    if style.cloud_linewidth > 0:
        line_v, line_p = grid, outer
        if style.cloud_close_outline:
            # close along the flat side, otherwise a lone cloud looks cut open
            line_v = np.concatenate([[grid[0]], grid, [grid[-1]]])
            line_p = np.concatenate([[base], outer, [base]])
        ax.plot(*_points(line_v, line_p, style), color=edge,
                linewidth=style.cloud_linewidth, zorder=style.cloud_zorder + 0.1,
                solid_joinstyle="round")


def _draw_box(ax: Axes, stats: dict, position: float, style: RainStyle, face, edge) -> None:
    if not np.isfinite(stats["center"]):
        return
    p = position + style.box_offset
    z = style.box_zorder
    whisker_lw = style.whisker_linewidth if style.whisker_linewidth is not None else style.box_linewidth
    center_lw = style.center_linewidth if style.center_linewidth is not None else style.box_linewidth * 1.6
    center_color = style.center_color if style.center_color is not None else edge
    cap = style.box_cap_width / 2

    # two segments rather than one through-line: a translucent box would show it
    for v_from, v_to in ((stats["wlo"], stats["lo"]), (stats["hi"], stats["whi"])):
        if np.isfinite(v_from) and np.isfinite(v_to):
            _span(ax, p, v_from, v_to, style, color=edge, linewidth=whisker_lw, zorder=z)
    if cap > 0:
        for v in (stats["wlo"], stats["whi"]):
            if np.isfinite(v):
                _cross(ax, v, p - cap, p + cap, style, color=edge, linewidth=whisker_lw, zorder=z)

    ax.add_patch(_rect(stats["lo"], stats["hi"], p, style.box_width, style,
                       facecolor=face, edgecolor=edge, alpha=style.box_alpha,
                       linewidth=style.box_linewidth, zorder=z + 0.1))
    _cross(ax, stats["center"], p - style.box_width / 2, p + style.box_width / 2, style,
           color=center_color, linewidth=center_lw, zorder=z + 0.2)


def _draw_rain(ax: Axes, values, chunk_hue, position: float, style: RainStyle,
               hue: _HueMap, color, size: float, alpha: float, rng) -> None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return
    offsets = position + style.rain_offset + jitter(values.size, style.rain_width, rng, style.jitter)
    x, y = _points(values, offsets, style)
    ax.scatter(x, y, s=size, c=hue.colors(chunk_hue, color),
               marker=style.point_marker, alpha=alpha,
               linewidths=style.point_linewidth, edgecolors=style.point_edgecolor,
               zorder=style.point_zorder)


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------

def _style_axes(ax: Axes, style: RainStyle, layout: _Layout, names: dict) -> None:
    n = len(layout.ticks)
    spines = style.visible_spines
    if spines is None:
        spines = ("bottom",) if _horizontal(style) else ("left",)
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_color(style.axis_color)
        spine.set_linewidth(style.axis_linewidth)
        if spine.get_visible() and style.spine_offset:
            spine.set_position(("outward", style.spine_offset))

    v_axis, g_axis = _value_axis(ax, style), _group_axis(ax, style)
    if style.value_ticks is not None:
        v_axis.set_ticks(list(style.value_ticks))
    else:
        v_axis.set_major_locator(MaxNLocator(nbins=style.n_value_ticks))
    v_axis.set_minor_locator(NullLocator())

    g_axis.set_ticks(layout.ticks)
    rotated = bool(style.group_tick_rotation % 180)
    g_axis.set_ticklabels(
        list(layout.tick_labels) if style.show_group_ticks else [""] * n,
        rotation=style.group_tick_rotation,
        va="center" if _horizontal(style) or rotated else "top",
        ha="center" if rotated or not _horizontal(style) else "right",
    )
    g_axis.set_minor_locator(NullLocator())

    ax.tick_params(axis="both", which="both", labelsize=style.tick_labelsize,
                   colors=style.axis_color, length=style.tick_length, width=style.tick_width)
    # the group axis carries names, not measurements -- ticks there are noise
    ax.tick_params(axis="y" if _horizontal(style) else "x", length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(style.label_color)

    group_lim = layout.lim
    if _horizontal(style):
        ax.set_ylim(*group_lim)
        if style.value_limits:
            ax.set_xlim(*style.value_limits)
        if style.invert_group_axis:
            ax.invert_yaxis()
    else:
        ax.set_xlim(*group_lim)
        if style.value_limits:
            ax.set_ylim(*style.value_limits)

    ax.set_axisbelow(True)
    if style.grid:
        ax.grid(True, axis="x" if _horizontal(style) else "y", color=style.grid_color,
                linestyle=style.grid_linestyle, linewidth=style.grid_linewidth, alpha=style.grid_alpha)
    else:
        ax.grid(False)

    if style.show_axis_labels:
        value_label = style.value_label if style.value_label is not None else (names["value"] or "")
        group_label = style.group_label if style.group_label is not None else (names["group"] or "")
        setter = (ax.set_xlabel, ax.set_ylabel) if _horizontal(style) else (ax.set_ylabel, ax.set_xlabel)
        setter[0](value_label, fontsize=style.axis_label_fontsize, color=style.label_color)
        setter[1](group_label, fontsize=style.axis_label_fontsize, color=style.label_color)
    if style.title and style.show_title:
        ax.set_title(style.title, fontsize=style.title_fontsize, color=style.label_color)


@dataclass
class _Layout:
    """Where each group's three layers go, in data coordinates on the group axis.

    The draw functions still add the style's offsets, which are zeroed for the
    layered layout, so these are exact positions there.
    """

    cloud: list[float]
    rain: list[float]
    box: list[float]
    ticks: list[float]
    tick_labels: list[str]
    lim: tuple[float, float]     # group-axis limits
    length: float                # group-axis extent in slot units, for the default figsize


def _layout(style: RainStyle, labels: Sequence[str], sign: float) -> _Layout:
    n = len(labels)
    pad = style.group_padding / 2
    if style.layout == "grouped":
        pos = [float(i) for i in range(n)]
        lo, hi = _slot_extent(style)
        return _Layout(pos, pos, pos, pos, list(labels), (lo - pad, n - 1 + hi + pad), float(n))

    # Layered: work in "up" units (+ is the side the cloud grows towards), then
    # map to data with ``sign``. Rows inside a band follow the group axis'
    # reading order -- top to bottom on an inverted axis, otherwise increasing.
    h, gap = style.layer_row_height, style.layer_gap
    slot = [i if sign < 0 else n - 1 - i for i in range(n)]
    nan = [float("nan")] * n
    rain_u, box_u, ticks_u, tick_labels = nan, nan, [], []
    cursor = 0.0
    for shown, which in ((style.show_rain, "rain"), (style.show_box, "box")):
        if not shown:
            continue
        cursor -= gap
        band = [cursor - (j + 0.5) * h for j in slot]
        cursor -= n * h
        ticks_u += band
        tick_labels += list(labels)
        if which == "rain":
            rain_u = band
        else:
            box_u = band
    top = abs(style.cloud_width) if style.show_cloud else 0.0
    ends = sorted((sign * (top + pad), sign * (cursor - pad)))
    return _Layout(
        cloud=[0.0] * n,
        rain=[sign * u for u in rain_u],
        box=[sign * u for u in box_u],
        ticks=[sign * u for u in ticks_u],
        tick_labels=tick_labels,
        lim=(ends[0], ends[1]),
        length=top - cursor + 2 * pad,
    )


def _slot_extent(style: RainStyle) -> tuple[float, float]:
    """How far a group's drawing reaches on either side of its centre."""
    edges: list[float] = []
    if style.show_cloud:
        edges += [style.cloud_base, style.cloud_base + style.cloud_width]
    if style.show_box:
        half = max(style.box_width, style.box_cap_width) / 2
        edges += [style.box_offset - half, style.box_offset + half]
    if style.show_rain:
        edges += [style.rain_offset - style.rain_width, style.rain_offset + style.rain_width]
    # cloud_width may be negative on an inverted axis, so sort rather than pair up
    return (min(edges or [0.0]), max(edges or [0.0]))


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def plot_raincloud(
    data,
    value: Optional[str] = None,
    group: Optional[str] = None,
    style: RainStyle = RainStyle(),
    box: BoxSpec = BoxSpec(),
    *,
    point_hue: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> RaincloudResult:
    """Draw a raincloud: a half-violin cloud, a summary box and jittered rain.

    ``data`` may be a DataFrame (with ``value``/``group``/``point_hue`` column
    names), a ``{label: values}`` mapping, or a bare sequence of numbers.
    ``point_hue`` colours the rain by a fourth variable -- a numeric column gets
    a colormap and a colorbar, anything else gets the palette and a legend.
    """
    tidy, names = _tidy(data, value, group, point_hue)
    if style.dropna:
        tidy = tidy[np.isfinite(tidy[VALUE])]

    levels = _levels(tidy, style)
    chunks = [(lv, tidy[tidy[GROUP] == lv]) for lv in levels]
    chunks = [(lv, ch) for lv, ch in chunks if len(ch) >= style.min_group_size]
    if style.show_total:
        total = (style.total_label, tidy)
        if names["group"] is None:
            chunks = []  # the lone unnamed group *is* the total; do not draw it twice
        chunks = [total] + chunks if style.total_position == "first" else chunks + [total]
    if not chunks:
        raise RaincloudError("nothing to draw: no group survived dropna/min_group_size")

    hue = _HueMap(tidy, style, names["hue"])
    rng = np.random.default_rng(style.random_state)
    if style.layout == "layered":
        style = style.derive(cloud_base=0.0, rain_offset=0.0, box_offset=0.0)
    sign, style = _flip(style), _oriented(style)

    overrides = [style.group_styles.get(level, GroupStyle()) for level, _ in chunks]
    labels = [o.label if o.label is not None else str(level) for o, (level, _) in zip(overrides, chunks)]
    layout = _layout(style, labels, sign)

    with mpl.rc_context({"font.family": style.font_family} if style.font_family else {}):
        if ax is None:
            span = style.slot_size * layout.length
            figsize = style.figsize or (
                (style.value_size, span) if _horizontal(style) else (span, style.value_size)
            )
            fig, ax = plt.subplots(figsize=figsize, dpi=style.dpi)
        else:
            fig = ax.figure
        fig.patch.set_facecolor(style.background_color)
        ax.set_facecolor(style.background_color)

        rows = []
        for i, ((level, chunk), override) in enumerate(zip(chunks, overrides)):
            base = override.color if override.color is not None else style.color_for(i)
            cloud_face = _resolve(style, override, "cloud_facecolor", base)
            cloud_edge = _resolve(style, override, "cloud_edgecolor", None) or _darken(cloud_face)
            box_face = _resolve(style, override, "box_facecolor", base)
            box_edge = _resolve(style, override, "box_edgecolor", None) or _darken(box_face)
            point_color = _resolve(style, override, "point_color", base)

            values = chunk[VALUE].to_numpy(dtype=float)
            if style.show_cloud:
                _draw_cloud(ax, values, layout.cloud[i], style, cloud_face, cloud_edge)
            if style.show_rain:
                _draw_rain(ax, values, chunk[HUE], layout.rain[i], style, hue, point_color,
                           _resolve(style, override, "point_size", None),
                           _resolve(style, override, "point_alpha", None), rng)
            stats = box.compute(values)
            if style.show_box:
                _draw_box(ax, stats, layout.box[i], style, box_face, box_edge)
            # a hidden box band in the layered layout leaves no position to label
            if style.annotate and np.isfinite(layout.box[i]) and np.isfinite(stats["center"]):
                _annotate(ax, stats, layout.box[i], style, style.annotate_color or box_edge, sign)

            rows.append({"group": level, **{k: stats[k] for k in STAT_KEYS}})

        _style_axes(ax, style, layout, names)
        bar = _decorate(fig, ax, style, hue)

    return RaincloudResult(figure=fig, ax=ax, groups=[lv for lv, _ in chunks],
                           stats=pd.DataFrame(rows), colorbar=bar)


def _annotate(ax: Axes, stats: dict, position: float, style: RainStyle, color, sign: float) -> None:
    # past the end of the whisker, on the box's own line: anywhere inside the
    # slot would sit on top of the cloud or in the rain
    anchor = stats["whi"] if np.isfinite(stats["whi"]) else stats["hi"]
    ax.annotate(
        style.annotate_format.format(**stats),
        _points(anchor, position + style.box_offset, style),
        textcoords="offset points",
        xytext=(6, 0) if _horizontal(style) else (0, 6),
        ha="left" if _horizontal(style) else "center",
        va="center" if _horizontal(style) else "bottom",
        fontsize=style.annot_fontsize, color=color, annotation_clip=False,
    )


def _decorate(fig: Figure, ax: Axes, style: RainStyle, hue: _HueMap) -> Optional[Colorbar]:
    """Colorbar for a numeric point hue, legend for a categorical one."""
    if not hue.active:
        return None
    label = style.colorbar_label if style.colorbar_label is not None else (hue.name or "")
    if hue.numeric:
        if not style.colorbar:
            return None
        bar = fig.colorbar(hue.mappable, ax=ax, shrink=style.colorbar_shrink,
                           aspect=style.colorbar_aspect, pad=style.colorbar_pad,
                           extend=style.colorbar_extend)
        if style.colorbar_label_position == "top":
            bar.ax.set_title(label, fontsize=style.axis_label_fontsize, color=style.label_color, pad=10)
        else:
            bar.set_label(label, fontsize=style.axis_label_fontsize, color=style.label_color)
        bar.ax.tick_params(labelsize=style.tick_labelsize, colors=style.label_color, length=style.tick_length)
        bar.outline.set_linewidth(0)
        return bar
    if not style.legend:
        return None
    handles = [Line2D([], [], marker=style.point_marker, linestyle="none", color=hue.lookup[lv],
                      markersize=np.sqrt(style.point_size), label=str(lv)) for lv in hue.levels]
    ax.legend(handles=handles, title=style.legend_title if style.legend_title is not None else label,
              loc=style.legend_loc, fontsize=style.tick_labelsize,
              title_fontsize=style.axis_label_fontsize, frameon=False)
    return None
