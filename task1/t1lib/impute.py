"""[1.7] [1.8] Filling the gaps: median, kNN, regression and MICE."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


def _median(random_state: int):
    return SimpleImputer(strategy="median")


def _mean(random_state: int):
    return SimpleImputer(strategy="mean")


def _knn(random_state: int, n_neighbors: int = 5):
    # distances are computed on the observed coordinates only (nan_euclidean)
    return KNNImputer(n_neighbors=n_neighbors, weights="distance")


def _regression(random_state: int):
    """Single deterministic pass: each feature regressed on the others once."""
    return IterativeImputer(max_iter=1, sample_posterior=False, random_state=random_state)


def _mice(random_state: int, max_iter: int = 15):
    """Chained equations proper: many passes, each draw taken from the posterior."""
    return IterativeImputer(
        max_iter=max_iter, sample_posterior=True, random_state=random_state
    )


IMPUTERS: dict[str, Callable] = {
    "med": _median,
    "mean": _mean,
    "knn": _knn,
    "reg": _regression,
    "mice": _mice,
}

RU_NAMES = {
    "med": "медиана",
    "mean": "среднее",
    "knn": "kNN",
    "reg": "регрессия",
    "mice": "MICE",
}


def impute(
    df: pd.DataFrame,
    method: str = "med",
    features: Optional[Sequence[str]] = None,
    class_col: Optional[str] = "target",
    random_state: int = 0,
    use_class_as_predictor: bool = False,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``features`` filled by ``method``.

    The class column is carried through untouched. It is only fed to the model
    as a one-hot predictor when ``use_class_as_predictor`` is set -- off by
    default, since in a real task the label is usually not available at
    imputation time for every row.
    """
    if method not in IMPUTERS:
        raise ValueError(f"Unknown method {method!r}. Known: {sorted(IMPUTERS)}")
    features = list(features if features is not None else
                    df.columns.drop(class_col) if class_col else df.columns)

    X = df[features].astype(float)
    extra = []
    if use_class_as_predictor and class_col is not None:
        extra = pd.get_dummies(df[class_col], prefix=class_col, dtype=float)
        X = pd.concat([X, extra], axis=1)

    if X[features].isna().all().any():
        empty = X[features].columns[X[features].isna().all()].tolist()
        raise ValueError(f"Columns {empty} are entirely missing; nothing to learn from.")

    filled = IMPUTERS[method](random_state).fit_transform(X.to_numpy())
    out = df.copy()
    out[features] = filled[:, : len(features)]
    return out


def impute_family(
    df: pd.DataFrame,
    methods: Sequence[str] = ("med", "knn", "reg", "mice"),
    features: Optional[Sequence[str]] = None,
    class_col: Optional[str] = "target",
    random_state: int = 0,
    prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """[1.8] ``{f"{prefix}{name}": filled_frame}`` for each requested method."""
    return {
        f"{prefix}{name}": impute(df, name, features, class_col, random_state)
        for name in methods
    }


def median_family(
    datasets: Mapping[str, pd.DataFrame],
    features: Optional[Sequence[str]] = None,
    class_col: Optional[str] = "target",
) -> dict[str, pd.DataFrame]:
    """[1.7] Median-fill every ``dfSdXX`` into the matching ``dfSdXXmed``."""
    return {f"{name}med": impute(d, "med", features, class_col) for name, d in datasets.items()}


def imputation_error(
    truth: pd.DataFrame,
    filled: pd.DataFrame,
    damaged: pd.DataFrame,
    features: Sequence[str],
) -> pd.Series:
    """RMSE over the cells that were actually blanked -- how far each method lands."""
    mask = damaged[list(features)].isna()
    if not mask.to_numpy().any():
        return pd.Series({"RMSE": 0.0, "MAE": 0.0, "n_filled": 0})
    diff = (filled[list(features)] - truth[list(features)]).where(mask)
    flat = diff.to_numpy()[mask.to_numpy()]
    return pd.Series({
        "RMSE": float(np.sqrt(np.nanmean(flat ** 2))),
        "MAE": float(np.nanmean(np.abs(flat))),
        "n_filled": int(mask.to_numpy().sum()),
    })
