import numpy as np
import pytest

from scatterkit import fit_polynomial, pair_stats


def test_exact_line_is_recovered():
    x = np.arange(10.0)
    fit = fit_polynomial(x, 3 * x + 2)
    assert fit.predict([0, 1, 20]) == pytest.approx([2, 5, 62])
    assert fit.sigma == pytest.approx(0, abs=1e-9)
    found = pair_stats(x, 3 * x + 2, fit)
    assert found["r"] == pytest.approx(1)
    assert found["slope"] == pytest.approx(3)
    assert found["intercept"] == pytest.approx(2)
    assert found["n"] == 10


def test_standard_errors_match_the_textbook_formulas():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = 2 * x + rng.normal(size=40)
    fit = fit_polynomial(x, y)

    slope, intercept = np.polyfit(x, y, 1)
    sigma = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (len(x) - 2))
    x0 = np.array([-1.0, 0.0, 2.5])
    leverage = 1 / len(x) + (x0 - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)

    assert fit.standard_error(x0, "residual") == pytest.approx(sigma)
    assert fit.standard_error(x0, "confidence") == pytest.approx(sigma * np.sqrt(leverage))
    assert fit.standard_error(x0, "prediction") == pytest.approx(sigma * np.sqrt(1 + leverage))


def test_quadratic_fit():
    x = np.linspace(-3, 3, 30)
    fit = fit_polynomial(x, x ** 2 - x + 1, degree=2)
    assert fit.predict([4.0]) == pytest.approx([13.0])
    assert fit.r2 == pytest.approx(1)


def test_nan_rows_are_dropped_pairwise():
    x = np.array([0.0, 1, 2, 3, np.nan, 5])
    y = np.array([0.0, 2, 4, np.nan, 8, 10])
    assert fit_polynomial(x, y).n == 4
    assert pair_stats(x, y)["n"] == 4


@pytest.mark.parametrize("x, y", [([1.0, 2.0], [1.0, 2.0]), ([3.0] * 5, [1.0, 2, 3, 4, 5])])
def test_undetermined_fit_is_none(x, y):
    assert fit_polynomial(x, y) is None
    assert np.isnan(pair_stats(x, y)["r"])
