from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional

Side = Literal["top", "bottom", "left", "right"]
Color = Any  # anything matplotlib accepts: "red", "#ff0000", (1, 0, 0), "none"

SORT_SUBSETS = ("degree", "-degree", "cardinality", "-cardinality", "input", "-input")
SORT_CONDITIONS = ("cardinality", "-cardinality", "input", "-input")


@dataclass
class ConditionStyle:
    """Overrides for everything drawn in one condition's row (or column)."""

    totals_bar_color: Optional[Color] = None
    shading_color: Optional[Color] = None
    label_color: Optional[Color] = None
    active_dot_color: Optional[Color] = None
    inactive_dot_color: Optional[Color] = None
    count_color: Optional[Color] = None


@dataclass
class SubsetRule:
    """Styles every intersection matching all given filters.

    Rules are applied in order, so later rules win. A rule with ``label``
    adds a legend entry and must set ``bar_color``.
    """

    present: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    min_degree: Optional[int] = None
    max_degree: Optional[int] = None
    min_size: Optional[float] = None
    max_size: Optional[float] = None

    bar_color: Optional[Color] = None
    active_dot_color: Optional[Color] = None
    inactive_dot_color: Optional[Color] = None
    line_color: Optional[Color] = None
    count_color: Optional[Color] = None
    label: Optional[str] = None


@dataclass
class UpsetStyle:
    # Layout: intersections and conditions must sit on perpendicular sides.
    intersections_side: Side = "top"
    conditions_side: Side = "left"
    figsize: Optional[tuple[float, float]] = None
    element_size: Optional[float] = 32
    title: Optional[str] = None
    intersections_label: str = "Intersection size"
    totals_label: Optional[str] = None

    # What is shown
    sort_by: str = "degree"
    sort_conditions_by: str = "cardinality"
    show_counts: bool = True
    show_percentages: bool = False
    with_lines: bool = True
    hide_inactive_dots_on_lines: bool = False
    min_subset_size: Optional[float] = None
    max_subset_size: Optional[float] = None
    min_degree: Optional[int] = None
    max_degree: Optional[int] = None
    include_empty_subsets: bool = False

    # Global colors
    background_color: Color = "white"
    intersection_bar_color: Color = "black"
    totals_bar_color: Color = "black"
    active_dot_color: Color = "black"
    inactive_dot_color: Color = "lightgray"
    line_color: Color = "black"
    shading_color: Color = "#eeeeee"
    alternate_shading_color: Color = "none"
    label_color: Color = "black"
    count_color: Color = "black"
    axis_color: Color = "black"
    grid_color: Color = "#cccccc"
    title_color: Color = "black"
    legend_text_color: Color = "black"

    # Per-element overrides
    condition_styles: dict[str, ConditionStyle] = field(default_factory=dict)
    subset_rules: list[SubsetRule] = field(default_factory=list)

    def validate(self) -> None:
        sides = {"top", "bottom", "left", "right"}
        if self.intersections_side not in sides or self.conditions_side not in sides:
            raise ValueError("Sides must be one of 'top', 'bottom', 'left', 'right'.")
        vertical_pair = {"top", "bottom"}
        if (self.intersections_side in vertical_pair) == (self.conditions_side in vertical_pair):
            raise ValueError(
                f"intersections_side={self.intersections_side!r} and "
                f"conditions_side={self.conditions_side!r} must be perpendicular "
                "(one of top/bottom, the other left/right)."
            )
        if self.sort_by not in SORT_SUBSETS:
            raise ValueError(f"sort_by must be one of {SORT_SUBSETS}.")
        if self.sort_conditions_by not in SORT_CONDITIONS:
            raise ValueError(f"sort_conditions_by must be one of {SORT_CONDITIONS}.")
        for rule in self.subset_rules:
            if rule.label and rule.bar_color is None:
                raise ValueError(f"Subset rule labelled {rule.label!r} needs a bar_color for its legend entry.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UpsetStyle":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown style keys: {sorted(unknown)}")
        data = dict(data)
        data["condition_styles"] = {
            name: s if isinstance(s, ConditionStyle) else ConditionStyle(**s)
            for name, s in data.get("condition_styles", {}).items()
        }
        data["subset_rules"] = [
            r if isinstance(r, SubsetRule) else SubsetRule(**r)
            for r in data.get("subset_rules", [])
        ]
        if data.get("figsize") is not None:
            data["figsize"] = tuple(data["figsize"])
        return cls(**data)
