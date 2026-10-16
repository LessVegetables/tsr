"""Sample estimates behind the box, and the KDE behind the cloud.

Kept apart from the drawing code so a raincloud can be re-summarised (mean
instead of median, std instead of IQR) without touching anything visual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

Estimator = Union[str, float, Callable[[np.ndarray], float]]
Spread = Union[str, Sequence[float]]

CENTERS = ("median", "mean")
BOXES = ("iqr", "std", "sem", "ci95")
WHISKERS = ("range", "1.5iqr", "std", "none")

STAT_KEYS = ("n", "center", "lo", "hi", "wlo", "whi")


class RaincloudError(ValueError):
    """Raised for input the caller can fix (bad column, bad estimator name)."""


def clean(values) -> np.ndarray:
    """1-D float array with NaN/inf dropped -- every estimate starts here."""
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


@dataclass
class BoxSpec:
    """Which sample estimates the box summarises -- all three are swappable.

    ``center``  -- ``"median"``, ``"mean"``, a quantile in [0, 1], or a callable.
    ``box``     -- ``"iqr"``, ``"std"``, ``"sem"``, ``"ci95"``, or an explicit
                   ``(lo_pct, hi_pct)`` percentile pair.
    ``whisker`` -- ``"range"``, ``"1.5iqr"``, ``"std"``, ``"none"``, or an
                   explicit ``(lo_pct, hi_pct)`` percentile pair.

    ``"ci95"`` is the normal-approximation interval around the centre, not a
    bootstrap, so it stays cheap and deterministic.
    """

    center: Estimator = "median"
    box: Spread = "iqr"
    whisker: Spread = "range"
    whisker_factor: float = 1.5

    def __post_init__(self) -> None:
        if isinstance(self.center, str) and self.center not in CENTERS:
            raise RaincloudError(f"center must be one of {CENTERS}, a quantile or a callable, got {self.center!r}")
        _check_spread("box", self.box, BOXES)
        _check_spread("whisker", self.whisker, WHISKERS)

    def compute(self, values) -> dict[str, float]:
        """All five box numbers for one sample, plus ``n``."""
        v = clean(values)
        if v.size == 0:
            return {"n": 0, **{k: float("nan") for k in STAT_KEYS[1:]}}

        center = self._center(v)
        lo, hi = self._box(v, center)
        wlo, whi = self._whisker(v, center, lo, hi)
        return {"n": int(v.size), "center": center,
                "lo": float(lo), "hi": float(hi), "wlo": float(wlo), "whi": float(whi)}

    # -- pieces ----------------------------------------------------------
    def _center(self, v: np.ndarray) -> float:
        if callable(self.center):
            return float(self.center(v))
        if self.center == "median":
            return float(np.median(v))
        if self.center == "mean":
            return float(np.mean(v))
        return float(np.quantile(v, float(self.center)))

    def _box(self, v: np.ndarray, center: float) -> tuple[float, float]:
        if self.box == "iqr":
            return tuple(np.percentile(v, [25, 75]))
        # a single observation has no spread; ddof=1 would hand back NaN
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        if self.box == "std":
            return center - sd, center + sd
        if self.box == "sem":
            sem = sd / np.sqrt(v.size)
            return center - sem, center + sem
        if self.box == "ci95":
            sem = sd / np.sqrt(v.size)
            return center - 1.96 * sem, center + 1.96 * sem
        return tuple(np.percentile(v, list(self.box)))

    def _whisker(self, v: np.ndarray, center: float, lo: float, hi: float) -> tuple[float, float]:
        if self.whisker == "none":
            return lo, hi
        if self.whisker == "range":
            return float(v.min()), float(v.max())
        if self.whisker == "1.5iqr":
            q1, q3 = np.percentile(v, [25, 75])
            span = self.whisker_factor * (q3 - q1)
            inside = v[(v >= q1 - span) & (v <= q3 + span)]
            # an all-outlier sample can only happen for IQR == 0; fall back to range
            return (float(inside.min()), float(inside.max())) if inside.size else (float(v.min()), float(v.max()))
        if self.whisker == "std":
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            return center - self.whisker_factor * sd, center + self.whisker_factor * sd
        return tuple(np.percentile(v, list(self.whisker)))


def _check_spread(name: str, value: Spread, allowed: Sequence[str]) -> None:
    if isinstance(value, str):
        if value not in allowed:
            raise RaincloudError(f"{name} must be one of {allowed} or a (lo, hi) percentile pair, got {value!r}")
        return
    pair = list(value)
    if len(pair) != 2 or not 0 <= pair[0] <= pair[1] <= 100:
        raise RaincloudError(f"{name} percentile pair must be (lo, hi) within 0..100, got {value!r}")


def kde(
    values,
    gridsize: int = 256,
    bw_adjust: float = 1.0,
    cut: float = 0.0,
    clip: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE on a regular grid, normalised to a peak of 1.

    ``cut`` extends the grid that many bandwidths past the extreme samples
    (0 stops the cloud exactly at the data, which is what most rainclouds
    want). Returns two empty arrays when a density is not defined -- fewer
    than two points, or every point identical.
    """
    from scipy.stats import gaussian_kde

    v = clean(values)
    if v.size < 2 or np.allclose(v, v[0]):
        return np.array([]), np.array([])

    estimator = gaussian_kde(v)
    estimator.set_bandwidth(estimator.factor * bw_adjust)
    bw = float(np.sqrt(estimator.covariance[0, 0]))

    lo, hi = v.min() - cut * bw, v.max() + cut * bw
    if clip is not None:
        lo = lo if clip[0] is None else max(lo, clip[0])
        hi = hi if clip[1] is None else min(hi, clip[1])
    if not hi > lo:
        return np.array([]), np.array([])

    grid = np.linspace(lo, hi, gridsize)
    dens = estimator(grid)
    peak = dens.max()
    return grid, dens / peak if peak > 0 else dens


def jitter(n: int, width: float, rng: np.random.Generator, kind: str = "uniform") -> np.ndarray:
    """Offsets for the rain. ``"uniform"`` fills the strip evenly, ``"normal"``
    concentrates points on the centre line, ``"none"`` stacks them in a row."""
    if kind == "none" or width == 0:
        return np.zeros(n)
    if kind == "normal":
        # clip so the tail cannot wander into the box or the next group
        return np.clip(rng.normal(0, width / 2, n), -width, width)
    if kind == "uniform":
        return rng.uniform(-width, width, n)
    raise RaincloudError(f"jitter must be 'uniform', 'normal' or 'none', got {kind!r}")
