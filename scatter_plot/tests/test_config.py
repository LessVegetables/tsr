import pytest

from scatterkit import ConfigError, Style, dump_config, load_config, parse_config
from scatterkit.config import SPEC


def test_values_are_typed():
    got = parse_config(
        "regression_shown=false\n"
        "regression_line_predict_bands=-1,2\n"
        "points_alpha=0.3\n"
        "ticks_x_major_count=7\n"
        "figure_width=auto\n"
        "features=Age, Fare\n"
        "regression_line_style=4,2\n"
    )
    assert got == {
        "regression_shown": False,
        "regression_line_predict_bands": [-1.0, 2.0],
        "points_alpha": 0.3,
        "ticks_x_major_count": 7,
        "figure_width": None,
        "features": ["Age", "Fare"],
        "regression_line_style": (0, (4.0, 2.0)),
    }


def test_is_shown_is_an_alias_for_shown():
    assert parse_config("regression_is_shown=false") == {"regression_shown": False}
    assert Style(regression_is_shown=False).regression_shown is False


def test_comments_and_hex_colours():
    got = parse_config(
        "# a comment\n"
        "\n"
        "points_color = #ff0000\n"
        "spine_color=#00ff00   # inline comment\n"
        "title_text=A = B\n"
    )
    assert got == {"points_color": "#ff0000", "spine_color": "#00ff00", "title_text": "A = B"}


def test_unknown_parameter_suggests_the_close_one():
    with pytest.raises(ConfigError, match=r"cfg:1: unknown parameter 'regresion_shown'.*regression_shown"):
        parse_config("regresion_shown=false", "cfg")


@pytest.mark.parametrize("line", ["points_alpha=abc", "regression_shown=maybe", "points_color=notacolour",
                                  "matrix_part=diagonal", "just some words"])
def test_bad_lines_name_their_line(line):
    with pytest.raises(ConfigError, match="cfg:2"):
        parse_config("\n" + line, "cfg")


def test_every_default_survives_dump_and_load(tmp_path):
    path = tmp_path / "config.txt"
    path.write_text(dump_config(), encoding="utf-8")
    assert load_config(path) == Style()
    assert set(parse_config(path.read_text(encoding="utf-8"))) == set(SPEC)


def test_dump_carries_overrides(tmp_path):
    style = Style(points_alpha=0.25, regression_line_predict_bands=[-1.0, 2.0], title_text="a\nb")
    path = tmp_path / "config.txt"
    path.write_text(dump_config(style), encoding="utf-8")
    assert load_config(path) == style
