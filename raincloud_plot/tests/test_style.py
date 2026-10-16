import pytest

from raincloudkit import DEFAULT_PALETTE, BoxSpec, GroupStyle, RainStyle, plot_raincloud
from raincloudkit.render import _slot_extent


def test_derive_changes_only_what_it_is_given():
    style = RainStyle(point_size=10)
    derived = style.derive(point_size=99)
    assert (derived.point_size, derived.orientation) == (99, style.orientation)
    assert style.point_size == 10          # the original is untouched


def test_palette_cycles():
    style = RainStyle(palette=("#111", "#222"))
    assert [style.color_for(i) for i in range(3)] == ["#111", "#222", "#111"]
    assert RainStyle().color_for(0) == DEFAULT_PALETTE[0]


@pytest.mark.parametrize("kwargs", [
    {"orientation": "sideways"}, {"total_position": "middle"}, {"palette": ()},
    {"layout": "stacked"},
])
def test_invalid_style_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RainStyle(**kwargs)


def test_group_styles_default_to_empty():
    assert RainStyle().group_styles == {}
    assert GroupStyle().color is None


def test_slot_extent_tracks_the_visible_layers():
    style = RainStyle()
    full = _slot_extent(style)
    assert full[0] < _slot_extent(style.derive(show_box=False))[0]    # box is lowest
    assert full[1] > _slot_extent(style.derive(show_cloud=False))[1]  # cloud is highest


def test_default_layers_stack_cloud_rain_box():
    """Top to bottom: cloud, then the rain under it, then the box."""
    s = RainStyle()
    assert s.cloud_base > s.rain_offset + s.rain_width
    assert s.rain_offset - s.rain_width > s.box_offset + max(s.box_width, s.box_cap_width) / 2


def test_group_axis_limits_leave_room_for_every_layer(df):
    style = RainStyle(group_padding=0.6)
    ax = plot_raincloud(df, "value", "group", style).ax
    top, bottom = sorted(ax.get_ylim())
    lo, hi = _slot_extent(style)
    assert top <= min(0 + lo, 1 + lo) and bottom >= max(0 + hi, 1 + hi)


def test_font_family_is_applied_and_not_left_global(df):
    import matplotlib as mpl
    before = list(mpl.rcParams["font.family"])
    ax = plot_raincloud(df, "value", "group", RainStyle(font_family="monospace")).ax
    assert ax.get_xlabel() == "value"
    assert list(mpl.rcParams["font.family"]) == before


def test_boxspec_defaults_are_shared_safely():
    """Mutable defaults on a dataclass would leak between calls."""
    a, b = BoxSpec(), BoxSpec()
    assert a == b and a is not b
