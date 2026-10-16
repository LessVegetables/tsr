"""Least-squares fit with standard-error bands, and the per-pair statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sps


@dataclass
class Fit:
    """A polynomial least-squares fit of y on x.

    x is centred and scaled before the powers are taken, so high degrees stay
    well conditioned; ``predict`` and ``standard_error`` take raw x.
    """

    degree: int
    coef: np.ndarray       # in the centred/scaled x, lowest power first
    x_mean: float
    x_scale: float
    sigma: float           # residual standard deviation, sqrt(SSE / dof)
    xtx_inv: np.ndarray
    r2: float
    n: int

    def _design(self, x) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.x_mean) / self.x_scale
        return np.vander(z, self.degree + 1, increasing=True)

    def predict(self, x) -> np.ndarray:
        return self._design(x) @ self.coef

    def standard_error(self, x, kind: str = "prediction") -> np.ndarray:
        """``prediction``: of a new observation; ``confidence``: of the fitted
        mean; ``residual``: sigma itself, the same at every x."""
        design = self._design(x)
        leverage = np.einsum("ij,jk,ik->i", design, self.xtx_inv, design)
        if kind == "prediction":
            return self.sigma * np.sqrt(1.0 + leverage)
        if kind == "confidence":
            return self.sigma * np.sqrt(leverage)
        if kind == "residual":
            return np.full(len(design), self.sigma)
        raise ValueError(f"unknown band kind '{kind}'")


def _finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def fit_polynomial(x, y, degree: int = 1) -> Optional[Fit]:
    """None when the fit is not determined: too few points, or x is constant."""
    x, y = _finite_pair(x, y)
    dof = len(x) - (degree + 1)
    scale = float(np.std(x)) if len(x) else 0.0
    if degree < 1 or dof < 1 or scale == 0.0:
        return None
    mean = float(np.mean(x))
    design = np.vander((x - mean) / scale, degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    sse = float(np.sum((y - design @ coef) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return Fit(
        degree=degree, coef=coef, x_mean=mean, x_scale=scale,
        sigma=float(np.sqrt(sse / dof)),
        xtx_inv=np.linalg.pinv(design.T @ design),
        r2=1.0 - sse / sst if sst > 0 else float("nan"),
        n=len(x),
    )


def pair_stats(x, y, fit: Optional[Fit] = None) -> dict[str, float]:
    """The fields available to ``annotation_format``.

    ``slope`` and ``intercept`` always describe the straight line; ``r2`` is
    that of ``fit`` when one is given (so of the curve, for degree > 1).
    """
    x, y = _finite_pair(x, y)
    out = dict.fromkeys(("r", "p", "rho", "r2", "slope", "intercept"), float("nan"))
    out["n"] = len(x)
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        line = sps.linregress(x, y)
        out.update(r=line.rvalue, p=line.pvalue, slope=line.slope, intercept=line.intercept,
                   rho=sps.spearmanr(x, y).statistic, r2=line.rvalue ** 2)
    if fit is not None:
        out["r2"] = fit.r2
    return out
