"""[1.3] raincloud plots, [1.4] scatter matrix, [1.6]-[1.10] scatter panels.

Nothing here draws on its own: the rainclouds come from ``raincloudkit``
(``../raincloud_plot``) and every scatter plot from ``scatterkit``
(``../scatter_plot``). This module only lays their figures out the way the task
asks -- one raincloud per feature, one scatter panel per dataset -- and adds the
two things the libraries do not have: a tick *step* and a shared panel grid.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from raincloudkit import BoxSpec, RainStyle, plot_raincloud
from scatterkit import FONT_FAMILY, ScatterResult, plot_scatter_matrix
from scatterkit import Style as ScatterStyle

# House style for this task. Every field is a RainStyle / scatterkit parameter,
# so the notebook overrides any of them with .derive(...).
RAIN_STYLE = RainStyle(
    point_size=9,
    point_alpha=0.45,
    cloud_alpha=0.55,
    box_width=0.16,
    annotate=True,
    annotate_format="{center:.2f}",
    value_size=6.0,
    slot_size=1.6,
    title_fontsize=13,
    axis_label_fontsize=11,
    tick_labelsize=10,
    random_state=0,
)

SCATTER_STYLE = ScatterStyle(
    points_size=8.0,
    points_alpha=0.45,
    points_hue_markers=["o", "s", "^", "D", "v", "P"],
    diagonal_kind="kde",                 # [1.4]: density of each class on the diagonal
    diagonal_kde_fill_shown=True,
    diagonal_kde_fill_alpha=0.3,
    annotation_format="r = {r:.2f}",
    output_dpi=110,
)


# ---------------------------------------------------------------------------
# [1.3] one raincloud per feature
# ---------------------------------------------------------------------------

def raincloud_grid(
    df: pd.DataFrame,
    features: Sequence[str],
    class_col: Optional[str] = None,
    box: BoxSpec = BoxSpec(),
    style: RainStyle = RAIN_STYLE,
    ncols: int = 1,
    suptitle: Optional[str] = None,
) -> Figure:
    """A panel per feature, each a ``plot_raincloud`` with one row per class."""
    features = list(features)
    n_groups = df[class_col].nunique() if class_col is not None else 1
    nrows = int(np.ceil(len(features) / ncols))
    row_h = style.slot_size * (n_groups + 2 * style.group_padding) + 0.9

    fig, axes = plt.subplots(nrows, ncols, figsize=(style.value_size * ncols, row_h * nrows),
                             dpi=style.dpi, squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        plot_raincloud(df, feature, class_col, style=style.derive(title=feature, value_label=feature),
                       box=box, ax=ax)
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)

    if suptitle is None:
        suptitle = f"[1.3] Raincloud: центр — {box.center}, ящик — {box.box}, усы — {box.whisker}"
    if suptitle and style.show_title:
        fig.suptitle(suptitle, fontsize=style.title_fontsize + 1, color=style.label_color)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# [1.4] scatter matrix
# ---------------------------------------------------------------------------

def set_tick_step(
    result: ScatterResult,
    major: Optional[float] = None,
    minor: Optional[float] = None,
) -> ScatterResult:
    """Ticks every ``major`` / ``minor`` data units -- scatterkit sets a count, [1.4] also asks for a step.

    Applied to every panel of ``result``; ``None`` leaves that level as scatterkit drew it.
    """
    for ax in result.axes.values():
        for axis in (ax.xaxis, ax.yaxis):
            if major is not None:
                axis.set_major_locator(MultipleLocator(major))
            if minor is not None:
                axis.set_minor_locator(MultipleLocator(minor))
            elif major is not None:
                axis.set_minor_locator(AutoMinorLocator(result.style.ticks_x_minor_count + 1))
    _pin_font(result.figure)
    return result


def scatter_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
    class_col: Optional[str] = None,
    style: ScatterStyle = SCATTER_STYLE,
    title: Optional[str] = "[1.4] 2D визуализация по всем парам признаков",
) -> ScatterResult:
    """``plot_scatter_matrix`` over ``features``, classes as colour + marker."""
    changes = dict(features=list(features), hue_column=class_col or "")
    if title is not None:
        changes["title_text"] = title
    return plot_scatter_matrix(df, style.derive(**changes))


# ---------------------------------------------------------------------------
# [1.6] [1.7] [1.8] [1.10] m x n panels of one feature pair across datasets
# ---------------------------------------------------------------------------

def _pin_font(fig: Figure) -> None:
    # scatterkit pins its font on every text after a draw; new tick labels need the same
    with mpl.rc_context({"font.family": [FONT_FAMILY]}):
        fig.canvas.draw()
        for text in fig.findobj(Text):
            text.set_fontfamily(FONT_FAMILY)


def _limits(values: np.ndarray, padding: float) -> tuple[float, float]:
    lo, hi = np.nanmin(values), np.nanmax(values)
    return lo - padding * (hi - lo), hi + padding * (hi - lo)


def scatter_panel(
    datasets: Mapping[str, pd.DataFrame],
    pair: tuple[str, str] = ("A", "B"),
    class_col: Optional[str] = None,
    style: ScatterStyle = SCATTER_STYLE,
    ncols: int = 4,
    suptitle: Optional[str] = None,
    reference: Optional[pd.DataFrame] = None,
    share_limits: bool = True,
) -> Figure:
    """m x n panel of ``pair`` scatterplots, one scatterkit plot per dataset.

    Each panel is a single-pair ``plot_scatter_matrix`` (lower triangle, no
    diagonal), titled with the dataset name and annotated with the surviving
    pair count and Pearson r. ``reference`` is drawn underneath in grey, and with
    ``share_limits`` every panel uses its range, so the panels compare directly.
    """
    xf, yf = pair
    items = list(datasets.items())
    nrows = int(np.ceil(len(items) / ncols))
    panel_style = style.derive(
        features=[xf, yf], matrix_part="lower", diagonal_shown=False,
        hue_column=class_col or "", legend_shown=False,
        annotation_format="n = {n},  r = {r:.3f}",
        title_font_size=style.title_font_size - 2,
    )
    if share_limits:
        base = reference if reference is not None else pd.concat([d for _, d in items])
        xlim = _limits(base[xf].to_numpy(dtype=float), style.axis_limits_padding)
        ylim = _limits(base[yf].to_numpy(dtype=float), style.axis_limits_padding)

    tiles = []
    for name, data in items:
        result = plot_scatter_matrix(data, panel_style.derive(title_text=name))
        ax = result.axes[(yf, xf)]
        if reference is not None:
            ax.scatter(reference[xf], reference[yf], s=style.points_size * 0.7, color="#CFCFCF",
                       alpha=0.5, linewidths=0, zorder=style.points_zorder - 1)
        if share_limits:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        _pin_font(result.figure)
        tiles.append(np.asarray(result.figure.canvas.buffer_rgba()))
        size = result.figure.get_size_inches()
        plt.close(result.figure)

    # Lay the rendered tiles out on one figure, pixel for pixel.
    tile_w, tile_h = size
    head = 0.5 if suptitle and style.title_shown else 0.0
    fig = plt.figure(figsize=(tile_w * ncols, tile_h * nrows + head), dpi=style.output_dpi,
                     facecolor=style.figure_background_color)
    total_h = tile_h * nrows + head
    for k, tile in enumerate(tiles):
        r, c = divmod(k, ncols)
        ax = fig.add_axes((c / ncols, (total_h - head - (r + 1) * tile_h) / total_h,
                           1 / ncols, tile_h / total_h))
        ax.imshow(tile, interpolation="none")
        ax.set_axis_off()

    if head:
        fig.text(0.5, 1 - 0.15 / total_h, suptitle, ha="center", va="top",
                 fontsize=style.title_font_size, fontweight=style.title_font_weight,
                 color=style.title_color, fontfamily=FONT_FAMILY)
    if class_col is not None and style.legend_shown:
        levels = list(pd.unique(pd.concat([d[class_col] for _, d in items]).dropna()))
        if len(levels) > 1:
            pal, marks = style.points_hue_palette, style.points_hue_markers
            handles = [Line2D([], [], linestyle="none", marker=marks[k % len(marks)],
                              color=pal[k % len(pal)], label=str(lv)) for k, lv in enumerate(levels)]
            fig.legend(handles=handles, title=class_col, loc="upper right", frameon=False)
    return fig
