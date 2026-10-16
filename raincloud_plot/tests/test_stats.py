import numpy as np
import pytest

from raincloudkit import BoxSpec, RaincloudError, kde
from raincloudkit.stats import clean, jitter


def test_default_box_is_median_iqr_range(sample):
    stats = BoxSpec().compute(sample)
    assert stats["n"] == 11
    assert stats["center"] == pytest.approx(50)
    assert (stats["lo"], stats["hi"]) == pytest.approx((25, 75))
    assert (stats["wlo"], stats["whi"]) == pytest.approx((0, 100))


def test_mean_std_whisker_std(sample):
    stats = BoxSpec(center="mean", box="std", whisker="std", whisker_factor=2).compute(sample)
    sd = sample.std(ddof=1)
    assert stats["center"] == pytest.approx(50)
    assert stats["hi"] - stats["center"] == pytest.approx(sd)
    assert stats["whi"] - stats["center"] == pytest.approx(2 * sd)


def test_quantile_centre_and_percentile_pairs(sample):
    stats = BoxSpec(center=0.9, box=(10, 90), whisker=(0, 100)).compute(sample)
    assert stats["center"] == pytest.approx(np.quantile(sample, 0.9))
    assert (stats["lo"], stats["hi"]) == pytest.approx((10, 90))


def test_callable_centre(sample):
    assert BoxSpec(center=lambda v: v.max()).compute(sample)["center"] == pytest.approx(100)


def test_sem_is_narrower_than_ci95(sample):
    sem = BoxSpec(box="sem").compute(sample)
    ci = BoxSpec(box="ci95").compute(sample)
    assert ci["hi"] - ci["lo"] > sem["hi"] - sem["lo"]


def test_15iqr_clips_an_outlier():
    values = np.append(np.arange(10, dtype=float), 1000.0)
    assert BoxSpec(whisker="1.5iqr").compute(values)["whi"] < 1000
    assert BoxSpec(whisker="range").compute(values)["whi"] == 1000


def test_empty_and_single_value_samples():
    empty = BoxSpec().compute([])
    assert empty["n"] == 0 and np.isnan(empty["center"])
    # ddof=1 on one point would be NaN; the spread must simply collapse
    one = BoxSpec(box="std").compute([5.0])
    assert one["n"] == 1 and one["center"] == 5 and one["lo"] == one["hi"] == 5


def test_clean_drops_nan_and_inf():
    assert clean([1, np.nan, 2, np.inf, -np.inf]).tolist() == [1, 2]


@pytest.mark.parametrize("kwargs", [
    {"center": "mode"}, {"box": "variance"}, {"whisker": "minmax"},
    {"box": (90, 10)}, {"whisker": (0, 200)},
])
def test_bad_estimators_are_rejected(kwargs):
    with pytest.raises(RaincloudError):
        BoxSpec(**kwargs)


def test_kde_is_peak_normalised(sample):
    grid, dens = kde(sample)
    assert grid.size == dens.size == 256
    assert dens.max() == pytest.approx(1.0)
    assert (grid.min(), grid.max()) == pytest.approx((0, 100))


def test_kde_cut_extends_past_the_data(sample):
    grid, _ = kde(sample, cut=2.0)
    assert grid.min() < 0 and grid.max() > 100


def test_kde_clip_bounds_the_grid(sample):
    grid, _ = kde(sample, cut=2.0, clip=(0, None))
    assert grid.min() == pytest.approx(0)


def test_kde_is_empty_when_undefined():
    assert kde([1.0])[0].size == 0
    assert kde([2.0, 2.0, 2.0])[0].size == 0


@pytest.mark.parametrize("kind,check", [
    ("uniform", lambda o: np.all(np.abs(o) <= 0.2)),
    ("normal", lambda o: np.all(np.abs(o) <= 0.2)),
    ("none", lambda o: np.all(o == 0)),
])
def test_jitter_stays_inside_its_strip(kind, check):
    offsets = jitter(500, 0.2, np.random.default_rng(0), kind)
    assert offsets.size == 500 and check(offsets)


def test_unknown_jitter_is_rejected():
    with pytest.raises(RaincloudError):
        jitter(3, 0.1, np.random.default_rng(0), "swarm")
