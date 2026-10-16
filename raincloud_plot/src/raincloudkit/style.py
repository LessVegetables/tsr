"""Every visual knob of a raincloud, in one dataclass.

Geometry inside a group is expressed in *slot units*: each group owns one unit
of the categorical axis, and the offsets below are measured from its centre.
Positive offsets point towards the cloud, so the defaults lay the three layers
out the way the name suggests -- cloud on top, the rain falling under it, and the
box summarising them at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Sequence

Color = Any  # anything matplotlib accepts: "red", "#ff0000", (1, 0, 0), "none"
Orientation = Literal["horizontal", "vertical"]
Layout = Literal["grouped", "layered"]
Jitter = Literal["uniform", "normal", "none"]
TotalPosition = Literal["first", "last"]

DEFAULT_PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")


@dataclass
class GroupStyle:
    """Overrides for one group's row. ``None`` keeps the global value."""

    color: Optional[Color] = None            # shorthand: cloud + box + point colour
    cloud_facecolor: Optional[Color] = None
    cloud_edgecolor: Optional[Color] = None
    box_facecolor: Optional[Color] = None
    box_edgecolor: Optional[Color] = None
    point_color: Optional[Color] = None
    point_size: Optional[float] = None       # rows of very different n rarely suit one size/alpha
    point_alpha: Optional[float] = None
    label: Optional[str] = None              # tick label, when it should differ from the level


@dataclass
class RainStyle:
    # -- figure ----------------------------------------------------------
    orientation: Orientation = "horizontal"  # "horizontal" -> values on x, groups on y
    # "grouped": each group gets its own cloud/rain/box slot.
    # "layered": every cloud overlaid on one baseline, then a band with one rain
    # row per group, then a band with one box row per group.
    layout: Layout = "grouped"
    figsize: Optional[tuple[float, float]] = None
    slot_size: float = 1.35                  # inches per group, when figsize is None
    value_size: float = 7.0                  # inches along the value axis, when figsize is None
    dpi: int = 120
    background_color: Color = "white"

    # -- what is drawn ---------------------------------------------------
    show_cloud: bool = True
    show_box: bool = True
    show_rain: bool = True
    show_total: bool = False                 # one extra group pooling every observation
    total_label: str = "Total"
    total_position: TotalPosition = "last"
    group_order: Optional[Sequence[Any]] = None
    dropna: bool = True
    min_group_size: int = 1                  # groups smaller than this are skipped

    # -- geometry (slot units, measured from the group's centre) ---------
    # Stacked cloud -> rain -> box, from the top of the slot downwards.
    cloud_base: float = 0.10                 # where the flat side of the cloud sits
    cloud_width: float = 0.45                # peak height of the density
    rain_offset: float = -0.08               # centre of the rain strip
    rain_width: float = 0.12                 # half-spread of the jitter
    box_offset: float = -0.34
    box_width: float = 0.18
    box_cap_width: float = 0.18              # whisker cap, as a fraction of a slot

    # -- layered layout only (slot units) --------------------------------
    # cloud_base / rain_offset / box_offset are ignored: the bands set positions
    layer_row_height: float = 0.35           # one rain row or one box row
    layer_gap: float = 0.25                  # cloud baseline -> rain band, rain band -> box band

    # -- cloud -----------------------------------------------------------
    cloud_facecolor: Optional[Color] = None  # None -> the group's palette colour
    cloud_edgecolor: Optional[Color] = None  # None -> a darker take on the face
    cloud_alpha: float = 0.6
    cloud_linewidth: float = 1.2
    cloud_close_outline: bool = True         # False: curve only -- clearer when clouds overlap
    cloud_gridsize: int = 256
    cloud_bw_adjust: float = 1.0
    cloud_cut: float = 0.0                   # bandwidths of tail past the data
    cloud_clip: Optional[tuple[Optional[float], Optional[float]]] = None
    cloud_zorder: float = 2

    # -- box -------------------------------------------------------------
    box_facecolor: Optional[Color] = None
    box_edgecolor: Optional[Color] = "black"
    box_alpha: float = 1.0
    box_linewidth: float = 1.4
    whisker_linewidth: Optional[float] = None  # None -> box_linewidth
    center_linewidth: Optional[float] = None   # None -> box_linewidth * 1.6
    center_color: Optional[Color] = None       # None -> box_edgecolor
    box_zorder: float = 4

    # -- rain ------------------------------------------------------------
    point_color: Optional[Color] = None      # None -> the group's palette colour
    point_size: float = 26
    point_alpha: float = 1
    point_linewidth: float = 0.5
    point_edgecolor: Color = "none"
    point_marker: str = "o"
    point_zorder: float = 3
    jitter: Jitter = "uniform"
    random_state: Optional[int] = 0

    # -- rain coloured by a third variable -------------------------------
    cmap: str = "Blues"
    hue_norm: Optional[tuple[float, float]] = None   # None -> the hue column's range
    colorbar: bool = True
    colorbar_label: Optional[str] = None             # None -> the hue column's name
    colorbar_shrink: float = 0.7
    colorbar_aspect: float = 18
    colorbar_pad: float = 0.02
    colorbar_extend: Literal["neither", "both", "min", "max"] = "neither"
    colorbar_label_position: Literal["side", "top"] = "side"
    legend: bool = True                              # for a categorical hue
    legend_title: Optional[str] = None
    legend_loc: str = "best"

    # -- palette ---------------------------------------------------------
    palette: Sequence[Color] = DEFAULT_PALETTE
    group_styles: dict = field(default_factory=dict)

    # -- axes, ticks, labels ---------------------------------------------
    title: Optional[str] = None
    value_label: Optional[str] = None        # None -> the value column's name
    group_label: Optional[str] = None        # None -> the group column's name
    show_title: bool = True
    show_axis_labels: bool = True
    show_group_ticks: bool = True
    group_tick_rotation: float = 0
    title_fontsize: float = 14
    axis_label_fontsize: float = 13
    tick_labelsize: float = 12
    annot_fontsize: float = 10
    font_family: Optional[str] = None
    label_color: Color = "#111111"
    axis_color: Color = "#111111"
    axis_linewidth: float = 1.6
    visible_spines: Optional[Sequence[str]] = None  # None -> the value axis only
    spine_offset: float = 12                     # points outward, 0 to re-attach
    value_limits: Optional[tuple[Optional[float], Optional[float]]] = None
    value_ticks: Optional[Sequence[float]] = None
    n_value_ticks: int = 6
    tick_length: float = 6.0
    tick_width: float = 1.4
    invert_group_axis: bool = True           # first group at the top, reading order
    group_padding: float = 0.6               # slack in slot units at both ends

    # -- grid ------------------------------------------------------------
    grid: bool = False
    grid_color: Color = "#C9C9C9"
    grid_linestyle: str = "-"
    grid_linewidth: float = 0.6
    grid_alpha: float = 0.7

    # -- annotation ------------------------------------------------------
    annotate: bool = False                   # print the centre next to each box
    annotate_format: str = "{center:.1f}"    # gets every key of BoxSpec.compute()
    annotate_color: Optional[Color] = None

    def __post_init__(self) -> None:
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {self.orientation!r}")
        if self.layout not in ("grouped", "layered"):
            raise ValueError(f"layout must be 'grouped' or 'layered', got {self.layout!r}")
        if self.total_position not in ("first", "last"):
            raise ValueError(f"total_position must be 'first' or 'last', got {self.total_position!r}")
        if not self.palette:
            raise ValueError("palette must not be empty")

    def color_for(self, index: int) -> Color:
        return self.palette[index % len(self.palette)]

    def derive(self, **changes) -> "RainStyle":
        """A copy with some knobs turned -- handy for one-off variants."""
        return replace(self, **changes)
