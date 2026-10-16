import json

import pytest

from upsetkit import ConditionStyle, SubsetRule, UpsetStyle


def test_json_round_trip():
    style = UpsetStyle(
        intersections_side="right",
        conditions_side="bottom",
        figsize=(8, 5),
        condition_styles={"Male": ConditionStyle(totals_bar_color="blue")},
        subset_rules=[SubsetRule(present=["Male"], bar_color="red", label="men")],
    )
    restored = UpsetStyle.from_dict(json.loads(json.dumps(style.to_dict())))
    assert restored == style


@pytest.mark.parametrize("inter,cond", [("top", "bottom"), ("left", "right"), ("top", "top")])
def test_sides_must_be_perpendicular(inter, cond):
    with pytest.raises(ValueError, match="perpendicular"):
        UpsetStyle(intersections_side=inter, conditions_side=cond).validate()


def test_labelled_rule_needs_bar_color():
    with pytest.raises(ValueError, match="bar_color"):
        UpsetStyle(subset_rules=[SubsetRule(count_color="red", label="x")]).validate()


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="Unknown style keys"):
        UpsetStyle.from_dict({"bar_colour": "red"})
