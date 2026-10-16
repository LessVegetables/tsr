import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection

from raincloudkit import BoxSpec, GroupStyle, RainStyle, RaincloudError, plot_raincloud


def points(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_returns_figure_and_per_group_stats(df):
    result = plot_raincloud(df, "value", "group")
    assert result.groups == ["a", "b"]
    assert list(result.stats["group"]) == ["a", "b"]
    assert set(result.stats.columns) == {"group", "n", "center", "lo", "hi", "wlo", "whi"}
    assert result.stats["n"].tolist() == [60, 60]
    assert result.colorbar is None


def test_stats_match_the_boxspec(df):
    spec = BoxSpec(center="mean", box="std")
    result = plot_raincloud(df, "value", "group", box=spec)
    expected = spec.compute(df.loc[df["group"] == "a", "value"])
    assert result.stats.iloc[0]["center"] == pytest.approx(expected["center"])
    assert result.stats.iloc[0]["hi"] == pytest.approx(expected["hi"])


def test_one_scatter_per_group(df):
    assert len(points(plot_raincloud(df, "value", "group").ax)) == 2


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_both_orientations_put_values_on_the_right_axis(df, orientation):
    result = plot_raincloud(df, "value", "group", RainStyle(orientation=orientation))
    offsets = points(result.ax)[0].get_offsets()
    values = offsets[:, 0] if orientation == "horizontal" else offsets[:, 1]
    group_axis = offsets[:, 1] if orientation == "horizontal" else offsets[:, 0]
    assert values.min() > 1 and values.max() < 30          # the data's own scale
    assert np.all(np.abs(group_axis) < 1)                  # slot 0, jitter only


@pytest.mark.parametrize("invert", [True, False])
def test_cloud_stays_above_the_rain_on_screen(df, invert):
    """Inverting the group axis must not turn the raincloud upside down."""
    ax = plot_raincloud(df, "value", "group", RainStyle(invert_group_axis=invert)).ax
    rain = points(ax)[0].get_offsets()[:, 1].mean()
    cloud = fills(ax)[0].get_paths()[0].vertices[:, 1].mean()
    bottom, top = ax.get_ylim()                             # in display order
    assert bool(bottom > top) is invert
    # "above" is whichever direction the axis grows towards on screen
    assert bool(cloud < rain) is invert


def test_layers_can_be_switched_off(df):
    ax = plot_raincloud(df, "value", "group", RainStyle(show_cloud=False, show_box=False)).ax
    assert len(points(ax)) == 2
    assert len(ax.patches) == 0
    ax2 = plot_raincloud(df, "value", "group", RainStyle(show_rain=False)).ax
    assert points(ax2) == []


def test_total_row_pools_every_observation(df):
    result = plot_raincloud(df, "value", "group", RainStyle(show_total=True))
    assert result.groups == ["a", "b", "Total"]
    assert result.stats.iloc[-1]["n"] == len(df)


def test_total_position_first(df):
    style = RainStyle(show_total=True, total_position="first", total_label="All")
    assert plot_raincloud(df, "value", "group", style).groups[0] == "All"


def test_total_without_a_group_column_is_drawn_once(df):
    result = plot_raincloud(df, "value", style=RainStyle(show_total=True))
    assert result.groups == ["Total"]
    assert result.stats.iloc[0]["n"] == len(df)


def test_group_order_is_honoured(df):
    assert plot_raincloud(df, "value", "group", RainStyle(group_order=["b", "a"])).groups == ["b", "a"]


def test_categorical_group_keeps_its_declared_order(df):
    df = df.assign(group=pd.Categorical(df["group"], categories=["b", "a"]))
    assert plot_raincloud(df, "value", "group").groups == ["b", "a"]


def test_min_group_size_skips_small_groups(df):
    df = pd.concat([df, pd.DataFrame({"value": [1.0], "group": ["tiny"], "score": [0], "sex": ["f"]})])
    assert "tiny" in set(plot_raincloud(df, "value", "group").groups)
    assert "tiny" not in set(plot_raincloud(df, "value", "group", RainStyle(min_group_size=5)).groups)


def test_nan_values_are_dropped(df):
    df.loc[:9, "value"] = np.nan
    assert plot_raincloud(df, "value", "group").stats["n"].sum() == len(df) - 10


def test_numeric_hue_builds_a_colorbar(df):
    result = plot_raincloud(df, "value", "group", point_hue="score")
    assert result.colorbar is not None
    assert result.colorbar.mappable.norm.vmin == pytest.approx(df["score"].min())
    colors = points(result.ax)[0].get_facecolors()
    assert len(np.unique(colors, axis=0)) > 1        # points really are coloured apart


def test_hue_norm_overrides_the_data_range(df):
    result = plot_raincloud(df, "value", "group", RainStyle(hue_norm=(0, 1000)), point_hue="score")
    assert result.colorbar.mappable.norm.vmax == 1000


def test_colorbar_can_be_suppressed(df):
    assert plot_raincloud(df, "value", "group", RainStyle(colorbar=False), point_hue="score").colorbar is None


def test_categorical_hue_builds_a_legend(df):
    result = plot_raincloud(df, "value", "group", point_hue="sex")
    assert result.colorbar is None
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == ["f", "m"]


def test_group_styles_override_one_row(df):
    style = RainStyle(group_styles={"b": GroupStyle(color="#123456", label="B row")})
    result = plot_raincloud(df, "value", "group", style)
    labels = [t.get_text() for t in result.ax.get_yticklabels()]
    assert labels == ["a", "B row"]


def test_group_styles_override_point_size_and_alpha(df):
    style = RainStyle(point_size=30, point_alpha=0.5,
                      group_styles={"b": GroupStyle(point_size=4, point_alpha=0.1)})
    a, b = points(plot_raincloud(df, "value", "group", style).ax)
    assert (a.get_sizes()[0], a.get_alpha()) == (30, 0.5)     # global value kept
    assert (b.get_sizes()[0], b.get_alpha()) == (4, 0.1)      # override applied


def test_mapping_input_keeps_key_order():
    result = plot_raincloud({"second": [1.0, 2, 3, 4], "first": [5.0, 6, 7, 8]})
    assert result.groups == ["second", "first"]


def test_bare_sequence_input():
    result = plot_raincloud(np.arange(50, dtype=float))
    assert len(result.groups) == 1 and result.stats.iloc[0]["n"] == 50


def test_series_input_names_the_axis():
    result = plot_raincloud(pd.Series(np.arange(50, dtype=float), name="depth"))
    assert result.ax.get_xlabel() == "depth"


def test_random_state_makes_the_rain_reproducible(df):
    a = points(plot_raincloud(df, "value", "group").ax)[0].get_offsets()
    b = points(plot_raincloud(df, "value", "group").ax)[0].get_offsets()
    assert np.array_equal(a, b)
    c = points(plot_raincloud(df, "value", "group", RainStyle(random_state=7)).ax)[0].get_offsets()
    assert not np.array_equal(a, c)


def test_drawing_into_an_existing_axes(df):
    fig, ax = plt.subplots()
    result = plot_raincloud(df, "value", "group", ax=ax)
    assert result.ax is ax and result.figure is fig


def test_labels_and_title(df):
    style = RainStyle(title="T", value_label="V", group_label="G")
    ax = plot_raincloud(df, "value", "group", style).ax
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("T", "V", "G")


def test_column_names_become_default_labels(df):
    ax = plot_raincloud(df, "value", "group").ax
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("value", "group")


def test_value_ticks_and_limits(df):
    style = RainStyle(value_limits=(0, 30), value_ticks=[0, 15, 30])
    ax = plot_raincloud(df, "value", "group", style).ax
    assert ax.get_xlim() == (0, 30)
    assert ax.get_xticks().tolist() == [0, 15, 30]


def test_savefig_writes_a_file(df, tmp_path):
    out = plot_raincloud(df, "value", "group").savefig(tmp_path / "r.png")
    assert out.exists() and out.stat().st_size > 0


def test_missing_column_is_reported(df):
    with pytest.raises(RaincloudError, match="nope"):
        plot_raincloud(df, "nope", "group")
    with pytest.raises(RaincloudError, match="group"):
        plot_raincloud(df, "value", "nope")


def test_ambiguous_value_column_is_reported(df):
    with pytest.raises(RaincloudError, match="value="):
        plot_raincloud(df)


def test_group_order_with_an_unknown_level_is_reported(df):
    with pytest.raises(RaincloudError, match="group_order"):
        plot_raincloud(df, "value", "group", RainStyle(group_order=["a", "zzz"]))


def test_group_on_a_non_frame_is_reported():
    with pytest.raises(RaincloudError, match="DataFrame"):
        plot_raincloud([1.0, 2.0, 3.0], group="g")


def test_everything_filtered_out_is_reported(df):
    with pytest.raises(RaincloudError, match="nothing to draw"):
        plot_raincloud(df, "value", "group", RainStyle(min_group_size=10_000))


def test_degenerate_group_still_draws(df):
    """A constant group has no KDE; it must lose its cloud, not the whole plot."""
    df.loc[df["group"] == "a", "value"] = 5.0
    result = plot_raincloud(df, "value", "group")
    assert result.stats.iloc[0]["center"] == 5.0
    assert len(points(result.ax)) == 2


# -- layered layout ----------------------------------------------------------

def fills(ax):
    # by type: whiskers are LineCollections, so "not a scatter" is not "a cloud"
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def box_centres(ax, axis=1):
    """Group-axis centre of every box rectangle, in drawing order."""
    return [(p.get_y() + p.get_height() / 2) if axis == 1 else (p.get_x() + p.get_width() / 2)
            for p in ax.patches]


def test_layered_stacks_clouds_then_rain_then_boxes(df):
    ax = plot_raincloud(df, "value", "group", RainStyle(layout="layered")).ax
    bottom, top = ax.get_ylim()
    assert bottom > top                                   # inverted: smaller y is higher up
    clouds = [c.get_paths()[0].vertices[:, 1] for c in fills(ax)]
    rain = [s.get_offsets()[:, 1] for s in points(ax)]
    boxes = box_centres(ax)
    # every cloud on one shared baseline, growing upwards from it
    assert all(c.max() == pytest.approx(0) and c.min() < 0 for c in clouds)
    assert max(c.max() for c in clouds) < min(r.min() for r in rain)
    assert max(r.max() for r in rain) < min(boxes)
    # inside each band, the group order reads top to bottom
    assert rain[0].mean() < rain[1].mean()
    assert boxes[0] < boxes[1]


def test_layered_labels_every_rain_and_box_row(df):
    ax = plot_raincloud(df, "value", "group", RainStyle(layout="layered")).ax
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "a", "b"]


def test_layered_drops_the_band_of_a_hidden_layer(df):
    ax = plot_raincloud(df, "value", "group", RainStyle(layout="layered", show_rain=False)).ax
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
    assert points(ax) == []


def test_layered_vertical_puts_clouds_at_the_far_end(df):
    ax = plot_raincloud(df, "value", "group", RainStyle(layout="layered", orientation="vertical")).ax
    clouds = [c.get_paths()[0].vertices[:, 0] for c in fills(ax)]
    rain = [s.get_offsets()[:, 0] for s in points(ax)]
    boxes = box_centres(ax, axis=0)
    # boxes, then rain, then clouds growing rightwards -- groups left to right
    assert max(boxes) < min(r.min() for r in rain)
    assert max(r.max() for r in rain) < min(c.min() for c in clouds)
    assert boxes[0] < boxes[1] and rain[0].mean() < rain[1].mean()


def test_layered_annotation_follows_the_box_row(df):
    style = RainStyle(layout="layered", annotate=True, annotate_format="{n}")
    ax = plot_raincloud(df, "value", "group", style).ax
    texts = [t for t in ax.texts if t.get_text() == "60"]
    assert [t.xy[1] for t in texts] == pytest.approx(box_centres(ax))


def test_cloud_outline_can_be_left_open(df):
    def outline(style):
        return plot_raincloud(df, "value", "group", style).ax.lines[0].get_xdata().size
    assert outline(RainStyle()) == 256 + 2               # closed along the baseline
    assert outline(RainStyle(cloud_close_outline=False)) == 256


@pytest.mark.parametrize("orientation,spine", [("horizontal", "bottom"), ("vertical", "left")])
def test_default_spine_is_the_value_axis(df, orientation, spine):
    ax = plot_raincloud(df, "value", "group", RainStyle(orientation=orientation)).ax
    assert [n for n, s in ax.spines.items() if s.get_visible()] == [spine]
