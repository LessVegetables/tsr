import numpy as np
import pandas as pd
import pytest

from upsetkit import ConditionError, build_indicators


def test_all_spec_kinds(df):
    ind = build_indicators(
        df,
        {
            "col": "survived",
            "expr": "age < 18",
            "method": "cabin.isna()",
            "callable": lambda d: d["sex"] == "f",
            "array": np.array([True] * 4 + [False] * 4),
        },
    )
    assert list(ind.columns) == ["col", "expr", "method", "callable", "array"]
    assert (ind.dtypes == bool).all()
    assert ind["col"].tolist() == [True, False, True, True, False, False, True, False]
    assert ind["expr"].sum() == 3  # NaN age is not a child


def test_list_of_column_names(df):
    df = df.assign(flag=[True, False] * 4)
    ind = build_indicators(df, ["survived", "flag"])
    assert list(ind.columns) == ["survived", "flag"]


def test_nan_in_binary_column_is_false(df):
    df = df.assign(flag=[1.0, np.nan, 0.0, 1.0, np.nan, 0.0, 1.0, 1.0])
    assert build_indicators(df, ["flag"])["flag"].tolist() == [True, False, False, True, False, False, True, True]


def test_non_binary_column_rejected(df):
    with pytest.raises(ConditionError, match="not binary"):
        build_indicators(df, {"Age": "age"})


def test_wrong_length_rejected(df):
    with pytest.raises(ConditionError, match="one value per row"):
        build_indicators(df, {"x": [True, False]})


def test_misaligned_series_rejected(df):
    with pytest.raises(ConditionError, match="different index"):
        build_indicators(df, {"x": lambda d: pd.Series([True] * len(d))[::-1].reset_index(drop=True).set_axis(range(10, 18))})


def test_bad_expression_names_condition(df):
    with pytest.raises(ConditionError, match="'Broken'"):
        build_indicators(df, {"Broken": "no_such_column > 1"})


def test_empty_conditions_rejected(df):
    with pytest.raises(ConditionError):
        build_indicators(df, {})
