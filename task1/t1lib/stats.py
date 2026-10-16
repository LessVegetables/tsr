"""[1.2] Tabular summary of a dataset and [1.5] correlation heat-maps."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

METHODS = ("pearson", "spearman")


def describe_dataset(
    df: pd.DataFrame,
    class_col: str = "target",
    features: Optional[Sequence[str]] = None,
    zero_atol: float = 1e-12,
) -> dict[str, pd.DataFrame]:
    """[1.2] Every table [1.2] asks for, as ``{caption: DataFrame}``.

    ``zero_atol`` widens "is a zero" into "is within ``zero_atol`` of zero":
    continuous Gaussian draws hit exact 0.0 with probability 0, so the exact
    count is reported next to a tolerance-based one to keep the table useful.
    """
    features = list(features if features is not None else df.columns.drop(class_col))
    values = df[features]
    classes = df[class_col]

    shape = pd.DataFrame(
        {"value": [len(df), df.shape[1], len(features), classes.nunique()]},
        index=["objects (rows)", "columns", "features", "classes"],
    )

    dtypes = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "role": ["class" if c == class_col else "feature" for c in df.columns],
            "unique": df.nunique(dropna=True),
            "non-null": df.notna().sum(),
        }
    )

    counts = classes.value_counts().sort_index()
    class_table = pd.DataFrame(
        {"objects": counts, "share, %": (counts / len(df) * 100).round(2)}
    )
    class_table.index.name = class_col

    summary = values.describe().T
    summary["IQR"] = summary["75%"] - summary["25%"]
    summary["range"] = summary["max"] - summary["min"]
    summary["skew"] = values.skew()
    summary["kurtosis"] = values.kurtosis()
    summary["missing"] = values.isna().sum()
    summary["missing, %"] = (values.isna().mean() * 100).round(2)

    na_by_class = _by_class(values.isna(), classes, features)
    na_by_class["rows with >=1 NaN"] = (
        values.isna().any(axis=1).groupby(classes, observed=False).sum()
    )
    na_by_class["rows fully complete"] = (
        values.notna().all(axis=1).groupby(classes, observed=False).sum()
    )

    zeros_exact = _by_class(values == 0, classes, features)
    zeros_near = _by_class(values.abs() <= zero_atol, classes, features)
    zero_table = pd.concat(
        {"exact 0": zeros_exact, f"|x| <= {zero_atol:g}": zeros_near}, axis=1
    )

    return {
        "Размерность": shape,
        "Типы признаков": dtypes,
        "Классы": class_table,
        "Оценки выборки": summary.round(4),
        "Объекты с NaN по классам": na_by_class,
        "Объекты с нулями по классам": zero_table,
    }


def _by_class(mask: pd.DataFrame, classes: pd.Series, features: Sequence[str]) -> pd.DataFrame:
    out = mask.groupby(classes, observed=False).sum()[list(features)]
    out.index.name = classes.name
    return out


def correlation_tables(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    class_col: Optional[str] = "target",
    methods: Sequence[str] = METHODS,
    per_class: bool = True,
    min_periods: int = 2,
) -> dict[tuple[str, str], pd.DataFrame]:
    """[1.5] ``{(scope, method): matrix}`` for the whole frame and each class.

    Pairs are compared on rows where both values survive, so a matrix can be
    computed even at 80% deletion -- read it together with the pair-wise support
    from :func:`pair_support`.
    """
    features = list(features if features is not None else df.columns.drop(class_col))
    out = {}
    for method in methods:
        out[("весь датасет", method)] = df[features].corr(method=method, min_periods=min_periods)
    if per_class and class_col is not None:
        for name, part in df.groupby(class_col, observed=True):
            for method in methods:
                out[(f"класс {name}", method)] = part[features].corr(
                    method=method, min_periods=min_periods
                )
    return out


def pair_support(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """How many rows actually back each correlation cell (both values present)."""
    present = df[list(features)].notna().astype(int)
    return pd.DataFrame(
        present.T.to_numpy() @ present.to_numpy(), index=list(features), columns=list(features)
    )


def plot_correlation_heatmaps(
    tables: dict[tuple[str, str], pd.DataFrame],
    ncols: Optional[int] = None,
    cell: float = 2.9,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
    annot_fontsize: float = 9,
    label_fontsize: float = 9,
    title_fontsize: float = 10,
    suptitle: Optional[str] = None,
    suptitle_fontsize: float = 13,
    show_colorbar: bool = True,
    mask_upper: bool = False,
) -> Figure:
    """[1.5] Lay the correlation matrices out as an m x n panel of heat-maps.

    A diverging map anchored at 0 with a fixed [-1, 1] range: every panel in the
    figure -- and across figures -- is then comparable by colour alone.
    """
    items = list(tables.items())
    if not items:
        raise ValueError("Nothing to plot: tables is empty.")
    ncols = ncols or min(len(items), 4)
    nrows = int(np.ceil(len(items) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell * ncols + 1.2 * show_colorbar, cell * nrows), squeeze=False
    )
    mesh = None
    for ax, ((scope, method), matrix) in zip(axes.ravel(), items):
        data = matrix.to_numpy(dtype=float).copy()
        if mask_upper:
            data[np.triu_indices_from(data, k=1)] = np.nan
        mesh = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(matrix)), matrix.columns, fontsize=label_fontsize)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=label_fontsize)
        ax.set_title(f"{method.capitalize()} | {scope}", fontsize=title_fontsize)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if annot:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if np.isnan(v):
                        continue
                    # white on the saturated ends, dark in the pale middle
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=annot_fontsize,
                        color="white" if abs(v) > 0.55 else "#111111",
                    )
    for ax in axes.ravel()[len(items):]:
        ax.set_visible(False)

    if show_colorbar and mesh is not None:
        fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02, label="коэффициент корреляции")
    if suptitle:
        # anchored above the figure, so it clears the panel titles even in one short row
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=1.0, va="bottom")
    return fig
