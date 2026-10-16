"""[1.9] UpSet views, drawn with the project's own ``upsetkit`` module.

Three readings of "groupings of features" are offered:

``upset_missing``           -- (a) objects whose value is *absent*, grouped by feature.
``upset_duplicated_values`` -- (b) objects whose value in a feature *coincides with
                               another object's*, grouped by feature. On continuous
                               data this is ~empty before imputation and lights up
                               exactly where an imputer wrote a constant, so it reads
                               as a fingerprint of the filling method.
``upset_pairwise_equal``    -- (b, alternative) one condition per feature *pair*
                               ``A == B``, with a tolerance, so the intersections
                               enumerate which pairs coincide together.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from upsetkit import UpsetStyle, UpsetResult, build_indicators, plot_upset

MISSING_STYLE = dict(
    intersection_bar_color="#6C5CE7",
    totals_bar_color="#A29BFE",
    active_dot_color="#4834A4",
    line_color="#4834A4",
    inactive_dot_color="#E4E1F5",
    sort_by="cardinality",
)

DUPLICATE_STYLE = dict(
    intersection_bar_color="#0F8C79",
    totals_bar_color="#6FC6B6",
    active_dot_color="#0B6B5C",
    line_color="#0B6B5C",
    inactive_dot_color="#DCEFEB",
    sort_by="cardinality",
)


def _style(defaults: dict, title: Optional[str], overrides: Optional[dict]) -> UpsetStyle:
    merged = {**defaults, "title": title, **(overrides or {})}
    return UpsetStyle(**merged)


def _safe_plot(df, conditions, style, verbose: bool = True) -> Optional[UpsetResult]:
    """Draw the plot, but first deal with conditions that nobody satisfies.

    ``upsetplot.from_indicators`` quietly drops every all-False column. That is
    a trap here: a feature with no missing values would simply disappear from
    the matrix with no hint that it was ever considered. So we evaluate the
    indicators ourselves, name the empty conditions out loud, and bail out
    when nothing is left rather than letting pandas raise
    "Must pass non-zero number of levels/codes" from three frames down.
    """
    indicators = build_indicators(df, conditions)
    empty = indicators.columns[~indicators.any(axis=0)].tolist()
    if empty and verbose:
        print(f"  [внимание] {style.title}: условия без единого объекта "
              f"({', '.join(map(str, empty))}) — upsetplot их не рисует.")
    if len(empty) == len(indicators.columns):
        if verbose:
            print(f"  [пропущено] {style.title}: ни одно условие не выполняется.")
        return None

    try:
        return plot_upset(df, conditions, style)
    except ValueError as exc:
        if "No intersections remain" in str(exc):
            if verbose:
                print(f"  [пропущено] {style.title}: нет непустых пересечений.")
            return None
        raise


def upset_missing(
    df: pd.DataFrame,
    features: Sequence[str],
    title: Optional[str] = "Пропущенные значения",
    style: Optional[dict] = None,
) -> Optional[UpsetResult]:
    """(a) How many objects miss each combination of features."""
    conditions = {f: (lambda d, f=f: d[f].isna()) for f in features}
    return _safe_plot(df, conditions, _style(MISSING_STYLE, title, style))


def upset_duplicated_values(
    df: pd.DataFrame,
    features: Sequence[str],
    title: Optional[str] = "Совпадающие значения",
    style: Optional[dict] = None,
    round_to: Optional[int] = None,
) -> Optional[UpsetResult]:
    """(b) How many objects share a value with some other object, per feature combination.

    ``round_to`` coarsens the comparison first: on raw continuous draws exact
    repeats essentially never happen, so rounding turns the question into
    "coincides to within 10^-round_to".
    """
    def condition(d: pd.DataFrame, f: str) -> pd.Series:
        col = d[f]
        if round_to is not None:
            col = col.round(round_to)
        return col.duplicated(keep=False) & col.notna()

    conditions = {f: (lambda d, f=f: condition(d, f)) for f in features}
    return _safe_plot(df, conditions, _style(DUPLICATE_STYLE, title, style))


def upset_pairwise_equal(
    df: pd.DataFrame,
    features: Sequence[str],
    title: Optional[str] = "Совпадение значений в парах признаков",
    style: Optional[dict] = None,
    atol: float = 1e-8,
) -> Optional[UpsetResult]:
    """(b, alternative) One condition per feature pair: ``|x_i - x_j| <= atol``."""
    pairs = [(a, b) for i, a in enumerate(features) for b in list(features)[i + 1:]]

    def condition(d: pd.DataFrame, a: str, b: str) -> pd.Series:
        diff = (d[a] - d[b]).abs()
        return (diff <= atol).fillna(False)

    conditions = {f"{a}={b}": (lambda d, a=a, b=b: condition(d, a, b)) for a, b in pairs}
    return _safe_plot(df, conditions, _style(DUPLICATE_STYLE, title, style))


def upset_series(
    datasets: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    kind: str = "missing",
    outdir: Optional[str] = None,
    prefix: str = "upset",
    style: Optional[dict] = None,
    **kwargs,
) -> dict[str, UpsetResult]:
    """Draw the same UpSet view for every dataset; optionally save each figure.

    Returns only the datasets that actually produced a plot, so a fully-complete
    frame (nothing missing, nothing duplicated) simply drops out of the result.
    """
    draw = {
        "missing": upset_missing,
        "duplicated": upset_duplicated_values,
        "pairwise": upset_pairwise_equal,
    }[kind]

    out = {}
    for name, data in datasets.items():
        res = draw(data, features, title=f"{name} — {_CAPTION[kind]}", style=style, **kwargs)
        if res is None:
            continue
        out[name] = res
        if outdir is not None:
            res.figure.savefig(f"{outdir}/{prefix}_{kind}_{name}.png",
                               bbox_inches="tight", dpi=120)
    return out


_CAPTION = {
    "missing": "пропуски по признакам",
    "duplicated": "совпадающие значения по признакам",
    "pairwise": "совпадения в парах признаков",
}


def intersections_table(results: Mapping[str, UpsetResult], top: int = 8) -> pd.DataFrame:
    """Side-by-side sizes of the largest intersections across datasets."""
    rows = {}
    for name, res in results.items():
        labels = [
            "+".join(n for n, flag in zip(res.intersections.index.names, combo) if flag) or "—"
            for combo in res.intersections.index
        ]
        rows[name] = pd.Series(res.intersections.to_numpy(), index=labels)
    table = pd.DataFrame(rows).fillna(0).astype(int)
    return table.loc[table.max(axis=1).sort_values(ascending=False).index[:top]]
