"""[1.1] [1.6] [1.10] Generating the base dataset and its damaged clones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_SHARE = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80)


@dataclass
class DatasetSpec:
    """Everything the user can turn when generating ``dfS00``.

    [1.1] asks for a single class of ``n`` objects, so that is the default. More
    classes are still supported: their mean shifts live in the ``C``/``D`` plane
    and are built so that the *between-class* covariance vanishes (see
    :meth:`shift_matrix`), which keeps the pooled correlation structure equal to
    the within-class one. With one class every shift is zero.
    """

    features: Sequence[str] = ("A", "B", "C", "D")
    n: int = 1000
    class_col: str = "target"
    class_names: Sequence[str] = ("K0",)
    class_weights: Optional[Sequence[float]] = None  # None -> balanced
    corr_pair: tuple[str, str] = ("A", "B")
    corr: float = 0.9
    stds: Sequence[float] = (1.0, 1.5, 2.0, 0.8)
    means: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    shift_features: tuple[str, str] = ("C", "D")
    shift_scale: tuple[float, float] = (3.0, 1.6)
    random_state: int = 42

    def __post_init__(self) -> None:
        if len(self.stds) != len(self.features) or len(self.means) != len(self.features):
            raise ValueError("stds and means must have one entry per feature.")
        if not -1 < self.corr < 1:
            raise ValueError("corr must lie strictly between -1 and 1.")
        for name in (*self.corr_pair, *self.shift_features):
            if name not in self.features:
                raise ValueError(f"{name!r} is not one of the features {tuple(self.features)}.")
        if self.class_weights is not None and len(self.class_weights) != len(self.class_names):
            raise ValueError("class_weights must have one entry per class.")

    @property
    def weights(self) -> np.ndarray:
        if self.class_weights is None:
            return np.full(len(self.class_names), 1 / len(self.class_names))
        w = np.asarray(self.class_weights, dtype=float)
        return w / w.sum()

    def correlation_matrix(self) -> pd.DataFrame:
        """Identity plus the single requested off-diagonal correlation."""
        R = np.eye(len(self.features))
        i = list(self.features).index(self.corr_pair[0])
        j = list(self.features).index(self.corr_pair[1])
        R[i, j] = R[j, i] = self.corr
        return pd.DataFrame(R, index=list(self.features), columns=list(self.features))

    def covariance_matrix(self) -> pd.DataFrame:
        s = np.asarray(self.stds, dtype=float)
        cov = self.correlation_matrix().to_numpy() * np.outer(s, s)
        return pd.DataFrame(cov, index=list(self.features), columns=list(self.features))

    def shift_matrix(self) -> pd.DataFrame:
        """Per-class mean shifts, orthogonalised against the pooled covariance.

        We need three things of the shift vectors ``s_k`` with class weights
        ``w_k``: ``sum w_k s_k = 0`` (the pooled mean stays put) and
        ``sum w_k s_k[C] s_k[D] = 0`` (the classes add no spurious C-D
        covariance). A star layout in the shift plane satisfies both: classes
        alternate between the two axes, each axis balanced around zero.
        """
        k = len(self.class_names)
        u, v = self.shift_features
        du, dv = self.shift_scale
        raw = np.zeros((k, len(self.features)))
        iu = list(self.features).index(u)
        iv = list(self.features).index(v)
        for c in range(k):
            axis, sign = divmod(c, 2)
            step = 1 + c // 4  # further classes sit further out on the same rays
            value = (du if axis % 2 == 0 else dv) * step * (1 if sign == 0 else -1)
            raw[c, iu if axis % 2 == 0 else iv] = value

        w = self.weights
        for col in (iu, iv):
            raw[:, col] -= w @ raw[:, col]  # enforce sum w_k s_k = 0
        cross = w @ (raw[:, iu] * raw[:, iv])
        if not np.isclose(cross, 0.0):  # last resort: kill the cross term
            raw[:, iv] -= cross / max(np.abs(raw[:, iu]).max(), 1e-12)
        return pd.DataFrame(raw, index=list(self.class_names), columns=list(self.features))

    def class_sizes(self) -> np.ndarray:
        """Largest-remainder apportionment of ``n`` over the classes."""
        exact = self.weights * self.n
        sizes = np.floor(exact).astype(int)
        for idx in np.argsort(-(exact - sizes))[: self.n - sizes.sum()]:
            sizes[idx] += 1
        return sizes


def make_base_dataset(spec: DatasetSpec = DatasetSpec()) -> pd.DataFrame:
    """[1.1] Draw ``dfS00``: 4 continuous features + a class label, ``spec.n`` rows.

    One ``multivariate_normal`` call per class (a single call by default).
    """
    rng = np.random.default_rng(spec.random_state)
    cov = spec.covariance_matrix().to_numpy()
    shifts = spec.shift_matrix()
    base = np.asarray(spec.means, dtype=float)

    blocks, labels = [], []
    for name, size in zip(spec.class_names, spec.class_sizes()):
        mean = base + shifts.loc[name].to_numpy()
        blocks.append(rng.multivariate_normal(mean, cov, size=size))
        labels.extend([name] * size)

    df = pd.DataFrame(np.vstack(blocks), columns=list(spec.features))
    df[spec.class_col] = pd.Categorical(labels, categories=list(spec.class_names))
    return df.sample(frac=1.0, random_state=spec.random_state).reset_index(drop=True)


def drop_cells_mcar(
    df: pd.DataFrame,
    share: float,
    features: Optional[Sequence[str]] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """[1.6] Blank ``share`` of the feature cells completely at random (MCAR).

    The class column is never touched, and the count is exact rather than
    per-cell Bernoulli, so a requested 5% really removes 5% of the cells.
    """
    if not 0 <= share <= 1:
        raise ValueError("share must lie in [0, 1].")
    features = list(features if features is not None else df.select_dtypes("number").columns)
    out = df.copy()
    total = len(df) * len(features)
    k = int(round(share * total))
    if k == 0:
        return out

    rng = np.random.default_rng(random_state)
    flat = rng.choice(total, size=k, replace=False)
    rows, cols = np.divmod(flat, len(features))
    for j, name in enumerate(features):
        out.loc[out.index[rows[cols == j]], name] = np.nan
    return out


def make_mcar_family(
    df: pd.DataFrame,
    shares: Sequence[float] = DEFAULT_SHARE,
    features: Optional[Sequence[str]] = None,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    """[1.6] ``{"dfSd00": ..., "dfSd01": ...}`` keyed by percentage of deleted cells."""
    out = {}
    for i, share in enumerate(shares):
        key = f"dfSd{round(share * 100):02d}"
        out[key] = drop_cells_mcar(df, share, features, random_state=random_state + i)
    return out


def make_mnar_below_median(
    df: pd.DataFrame,
    features: Sequence[str] = ("C", "D"),
    quantile: float = 0.5,
) -> pd.DataFrame:
    """[1.10] ``dfMNAR``: drop every value of ``features`` below its own median.

    Missingness now depends on the value that went missing, so it is MNAR --
    the case where median/kNN/regression filling is expected to bias the
    correlations rather than merely blur them.
    """
    out = df.copy()
    for name in features:
        out.loc[out[name] < out[name].quantile(quantile), name] = np.nan
    return out
