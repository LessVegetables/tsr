from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Union

import numpy as np
import pandas as pd

ConditionSpec = Union[str, Callable[[pd.DataFrame], pd.Series], pd.Series, np.ndarray]
Conditions = Union[Mapping[str, ConditionSpec], Iterable[str]]


class ConditionError(ValueError):
    pass


def build_indicators(df: pd.DataFrame, conditions: Conditions) -> pd.DataFrame:
    """Evaluate each condition against ``df`` into a boolean indicator frame.

    A spec may be a column name, a ``DataFrame.eval`` expression (``"Age < 18"``),
    a callable ``df -> Series``, or a precomputed boolean Series/array.
    NaN counts as "not a member".
    """
    if not isinstance(conditions, Mapping):
        conditions = {name: name for name in conditions}
    if not conditions:
        raise ConditionError("At least one condition is required.")

    columns = {}
    for name, spec in conditions.items():
        try:
            values = _evaluate(df, spec)
        except ConditionError:
            raise
        except Exception as exc:
            raise ConditionError(f"Condition {name!r} failed to evaluate: {exc}") from exc
        columns[name] = _to_bool(name, values, df.index)
    return pd.DataFrame(columns, index=df.index)


def _evaluate(df: pd.DataFrame, spec: ConditionSpec):
    if callable(spec):
        return spec(df)
    if isinstance(spec, str):
        if spec in df.columns:
            return df[spec]
        # python engine allows method calls such as "Cabin.isna()"
        return df.eval(spec, engine="python")
    return spec


def _to_bool(name: str, values, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise ConditionError(f"Condition {name!r} returned a Series with a different index.")
    else:
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != len(index):
            raise ConditionError(
                f"Condition {name!r} must produce one value per row "
                f"(got shape {values.shape}, expected ({len(index)},))."
            )
        values = pd.Series(values, index=index)

    non_null = values.dropna()
    if not non_null.isin([True, False]).all():
        bad = non_null[~non_null.isin([True, False])].unique()[:5]
        raise ConditionError(
            f"Condition {name!r} is not binary; found values like {list(bad)}. "
            "Use an expression such as 'Age < 18' to binarize it."
        )
    return non_null.astype(bool).reindex(index, fill_value=False)
