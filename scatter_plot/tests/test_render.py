import warnings

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.text import Text

from scatterkit import FONT_FAMILY, Style, plot_scatter_matrix
from scatterkit.__main__ import main
from scatterkit.config import SPEC


def tagged(result, gid):
    return [a for ax in result.figure.axes for a in ax.get_children() if a.get_gid() == gid]


def test_defaults_draw_the_full_matrix(df):
    result = plot_scatter_matrix(df)
    assert len(result.axes) == 9
    assert len(result.stats) == 6
    assert len(tagged(result, "points")) == 6
    assert len(tagged(result, "regression_line")) == 6
    assert len(tagged(result, "regression_band_line")) == 12
    assert len(tagged(result, "regression_band_fill")) == 6


def test_regression_switches(df):
    off = plot_scatter_matrix(df, Style(regression_shown=False))
    assert not tagged(off, "regression_line") and not tagged(off, "regression_band_line")
    assert not tagged(off, "regression_band_fill")

    no_bands = plot_scatter_matrix(df, Style(regression_line_predict_bands_shown=False))
    assert len(tagged(no_bands, "regression_line")) == 6
    assert not tagged(no_bands, "regression_band_line") and not tagged(no_bands, "regression_band_fill")

    three = plot_scatter_matrix(df, Style(regression_line_predict_bands=[-1, 1, 2]))
    assert len(tagged(three, "regression_band_line")) == 18


def test_band_edges_sit_where_the_config_says(df):
    style = Style(features=["feature_1", "feature_2"], matrix_part="lower", diagonal_shown=False,
                  regression_line_predict_bands=[-1, 2], regression_line_predict_bands_kind="residual")
    result = plot_scatter_matrix(df, style)
    (center,) = tagged(result, "regression_line")
    lower, upper = tagged(result, "regression_band_line")
    below = center.get_ydata() - lower.get_ydata()
    above = upper.get_ydata() - center.get_ydata()
    assert np.allclose(above, 2 * below) and np.ptp(below) == pytest.approx(0, abs=1e-9)


def test_two_features_without_diagonal_is_one_panel(df):
    style = Style(features=["feature_1", "feature_2"], matrix_part="lower", diagonal_shown=False)
    result = plot_scatter_matrix(df, style)
    assert list(result.axes) == [("feature_2", "feature_1")]


@pytest.mark.parametrize("part, panels", [("full", 9), ("lower", 6), ("upper", 6)])
def test_matrix_part(df, part, panels):
    assert len(plot_scatter_matrix(df, Style(matrix_part=part)).axes) == panels


def test_style_reaches_the_artists(df):
    style = Style(points_alpha=0.3, points_size=40, points_color="#ff0000", spine_width=3,
                  spine_top_shown=False, ticks_x_major_length=9, ticks_y_minor_width=2.5,
                  regression_line_color="#00ff00", regression_line_width=4)
    result = plot_scatter_matrix(df, style)
    ax = result.axes["feature_2", "feature_1"]
    (points,) = [a for a in ax.collections if a.get_gid() == "points"]
    assert points.get_alpha() == 0.3
    assert points.get_sizes()[0] == 40
    assert to_hex(points.get_facecolor()[0]) == "#ff0000"
    assert ax.spines["left"].get_linewidth() == 3
    assert not ax.spines["top"].get_visible()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 9
    assert ax.yaxis.get_minor_ticks()[0].tick1line.get_markeredgewidth() == 2.5
    (line,) = [a for a in ax.lines if a.get_gid() == "regression_line"]
    assert to_hex(line.get_color()) == "#00ff00" and line.get_linewidth() == 4


def test_tick_counts(df):
    style = Style(ticks_major_count_mode="exact", ticks_x_major_count=6, ticks_y_major_count=3,
                  ticks_x_minor_count=4)
    ax = plot_scatter_matrix(df, style).axes["feature_2", "feature_1"]
    assert len(ax.get_xticks()) == 6
    assert len(ax.get_yticks()) == 3
    assert len(ax.xaxis.get_minorticklocs()) == 5 * 4


def test_hue_splits_points_and_adds_a_legend(df):
    result = plot_scatter_matrix(df, Style(hue_column="group", regression_per_hue=True))
    assert len(tagged(result, "points")) == 6 * 3
    assert len(tagged(result, "regression_line")) == 6 * 3
    (legend,) = result.figure.legends
    assert [t.get_text() for t in legend.get_texts()] == ["A", "B", "C"]


def test_every_text_is_in_the_fixed_font(df):
    result = plot_scatter_matrix(df, Style(hue_column="group", legend_regression_shown=True))
    families = {t.get_fontfamily()[0] for t in result.figure.findobj(Text) if t.get_text()}
    assert families == {FONT_FAMILY}


def test_every_parameter_is_read_by_the_renderer(df):
    """A config key nothing reads is a knob that silently does nothing."""

    class Recorder:
        def __init__(self, style):
            self._style, self.read = style, set()

        def __getattr__(self, name):
            self.read.add(name)
            return getattr(self._style, name)

    base = dict(hue_column="group", legend_regression_shown=True, grid_major_shown=True,
                grid_minor_shown=True)
    read = set()
    for variant in (dict(diagonal_kind="hist_kde", diagonal_kde_fill_shown=True),
                    dict(diagonal_kind="name", max_points=30)):
        recorder = Recorder(Style(**base, **variant))
        plot_scatter_matrix(df, recorder)
        read |= recorder.read
    # read by the loader / the command line / savefig rather than while drawing
    elsewhere = {"data_path", "data_separator", "demo_features", "demo_points", "output_path",
                 "output_transparent", "output_tight_crop", "output_tight_pad", "output_show_window"}
    assert set(SPEC) - read - elsewhere == set()


def test_command_line_writes_the_picture(tmp_path, capsys):
    config = tmp_path / "config.txt"
    out = tmp_path / "out.png"
    config.write_text(f"output_path={out}\nregression_is_shown=false\ndemo_features=3\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert main([f"--config={config}", "--set", "output_dpi=60"]) == 0
    assert out.stat().st_size > 0
    assert "saved" in capsys.readouterr().out


def test_command_line_reports_config_errors(tmp_path, capsys):
    assert main(["--set", "no_such_param=1"]) == 2
    assert "unknown parameter" in capsys.readouterr().err
