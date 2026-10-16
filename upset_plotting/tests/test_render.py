import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from upsetkit import ConditionStyle, SubsetRule, UpsetStyle, plot_upset

LAYOUTS = [
    ("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"),
    ("left", "top"), ("right", "top"), ("left", "bottom"), ("right", "bottom"),
]


def _visible_dots(matrix):
    """Map (intersection index, condition index) -> RGBA for every visible dot (horizontal layout)."""
    visible = {}
    for coll in matrix.collections:
        if isinstance(coll, PathCollection):
            colors = coll.get_facecolors()
            for k, (x, y) in enumerate(coll.get_offsets()):
                color = tuple(colors[k if len(colors) > 1 else 0])
                if color[3] > 0:
                    visible[(round(x), round(y))] = color
    return visible


def _center(ax):
    b = ax.get_position()
    return (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2


@pytest.mark.parametrize("inter_side,cond_side", LAYOUTS)
def test_panels_on_requested_sides(df, conditions, inter_side, cond_side):
    result = plot_upset(df, conditions, {"intersections_side": inter_side, "conditions_side": cond_side})
    result.figure.canvas.draw()
    mx, my = _center(result.axes["matrix"])
    ix, iy = _center(result.axes["intersections"])
    tx, ty = _center(result.axes["totals"])
    side_of = {"top": iy > my, "bottom": iy < my, "left": ix < mx, "right": ix > mx}
    assert side_of[inter_side]
    side_of = {"top": ty > my, "bottom": ty < my, "left": tx < mx, "right": tx > mx}
    assert side_of[cond_side]


def test_colors_follow_precedence(df, conditions):
    style = UpsetStyle(
        intersection_bar_color="gray",
        totals_bar_color="green",
        active_dot_color="black",
        inactive_dot_color="white",
        hide_inactive_dots_on_lines=False,
        line_color="black",
        condition_styles={"Male": ConditionStyle(totals_bar_color="blue", active_dot_color="navy")},
        subset_rules=[SubsetRule(present=["Survived"], bar_color="red", active_dot_color="pink", line_color="red", label="s")],
    )
    result = plot_upset(df, conditions, style)
    inters = result.intersections
    names = list(inters.index.names)

    for bar, combo in zip(result.axes["intersections"].patches, inters.index):
        survived = combo[names.index("Survived")]
        assert bar.get_facecolor() == to_rgba("red" if survived else "gray")

    totals = dict(zip(result.totals.index, result.axes["totals"].patches))
    assert totals["Male"].get_facecolor() == to_rgba("blue")
    assert totals["Child"].get_facecolor() == to_rgba("green")

    matrix = result.axes["matrix"]
    dots = _visible_dots(matrix)
    for i, combo in enumerate(inters.index):
        membership = dict(zip(names, combo))
        for j, name in enumerate(names):
            if not membership[name]:
                expected = "white"
            elif membership["Survived"]:
                expected = "pink"  # subset rule beats condition style
            elif name == "Male":
                expected = "navy"
            else:
                expected = "black"
            assert dots[(i, j)] == to_rgba(expected), (combo, name)

    lines = next(c for c in matrix.collections if isinstance(c, LineCollection))
    non_empty = [c for c in inters.index if any(c)]
    for color, combo in zip(lines.get_colors(), non_empty):
        assert tuple(color) == to_rgba("red" if combo[names.index("Survived")] else "black")

    legend = result.axes["intersections"].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["s"]


def test_condition_label_color(df, conditions):
    result = plot_upset(
        df, conditions,
        UpsetStyle(intersections_side="right", conditions_side="bottom", label_color="black",
                   condition_styles={"Child": ConditionStyle(label_color="orange")}),
    )
    labels = {t.get_text(): t.get_color() for t in result.axes["totals"].xaxis.get_ticklabels()}
    assert labels["Child"] == "orange"
    assert labels["Male"] == "black"


def test_draws_into_existing_figure_without_resizing(df, conditions):
    fig = Figure(figsize=(7, 3))
    for _ in range(2):  # redraw, as a GUI would
        result = plot_upset(df, conditions, fig=fig)
    assert result.figure is fig
    assert tuple(fig.get_size_inches()) == (7, 3)
    assert len(fig.axes) == 4


def test_filters_and_counts(df, conditions):
    result = plot_upset(df, conditions, UpsetStyle(min_degree=2, show_counts=False))
    assert all(sum(c) >= 2 for c in result.intersections.index)
    assert len(result.axes["intersections"].texts) == 0


def test_unknown_condition_in_style(df, conditions):
    with pytest.raises(ValueError, match="unknown conditions"):
        plot_upset(df, conditions, UpsetStyle(subset_rules=[SubsetRule(present=["Nope"], bar_color="red")]))


def test_everything_filtered_out(df, conditions):
    with pytest.raises(ValueError, match="No intersections"):
        plot_upset(df, conditions, UpsetStyle(min_subset_size=1000))


@pytest.mark.parametrize("with_lines", [True, False])
def test_hide_inactive_dots_on_lines(df, conditions, with_lines):
    style = UpsetStyle(inactive_dot_color="gray", hide_inactive_dots_on_lines=True, with_lines=with_lines)
    result = plot_upset(df, conditions, style)
    dots = _visible_dots(result.axes["matrix"])

    hidden_somewhere = False
    for i, combo in enumerate(result.intersections.index):
        active = [j for j, member in enumerate(combo) if member]
        for j, member in enumerate(combo):
            under_line = with_lines and not member and len(active) >= 2 and active[0] < j < active[-1]
            hidden_somewhere |= under_line
            assert ((i, j) not in dots) == under_line, (combo, j)
    assert hidden_somewhere == with_lines


@pytest.mark.parametrize("inter_side,cond_side", LAYOUTS)
def test_grid_below_bars_and_vertical_labels_read_upwards(df, conditions, inter_side, cond_side):
    result = plot_upset(df, conditions, {"intersections_side": inter_side, "conditions_side": cond_side})
    result.figure.canvas.draw()
    assert all(ax.get_axisbelow() is True for ax in result.axes.values())
    matrix = result.axes["matrix"]
    totals = result.axes["totals"]
    labels = (totals.yaxis if inter_side in ("top", "bottom") else totals.xaxis).get_ticklabels()
    expected = 0 if inter_side in ("top", "bottom") else 90
    assert {label.get_rotation() for label in labels} == {expected}


@pytest.mark.parametrize("hide", [True, False])
def test_inactive_dots_below_lines_active_above(df, conditions, hide):
    result = plot_upset(df, conditions, UpsetStyle(hide_inactive_dots_on_lines=hide))
    matrix = result.axes["matrix"]
    dots = [c for c in matrix.collections if isinstance(c, PathCollection)]
    lines = next(c for c in matrix.collections if isinstance(c, LineCollection))
    active, inactive = dots
    assert len(inactive.get_offsets()) > 0
    assert inactive.get_zorder() < lines.get_zorder() < active.get_zorder()


@pytest.mark.parametrize("show_percentages", [False, True])
@pytest.mark.parametrize("inter_side,cond_side", LAYOUTS)
def test_count_labels_do_not_overlap_condition_names(df, conditions, inter_side, cond_side, show_percentages):
    result = plot_upset(
        df, conditions,
        {"intersections_side": inter_side, "conditions_side": cond_side, "show_percentages": show_percentages},
    )
    fig = result.figure
    fig.canvas.draw()
    totals = result.axes["totals"]
    horizontal = inter_side in ("top", "bottom")
    names = (totals.yaxis if horizontal else totals.xaxis).get_ticklabels()
    counts = totals.texts
    assert len(names) == len(counts) > 0
    for count in counts:
        cb = count.get_window_extent()
        for name in names:
            assert not cb.overlaps(name.get_window_extent()), (count.get_text(), name.get_text())
