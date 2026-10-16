import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, n), rng.normal(16, 3, n)]),
            "group": ["a"] * n + ["b"] * n,
            "score": rng.uniform(0, 100, 2 * n),
            "sex": rng.choice(["f", "m"], 2 * n),
        }
    )


@pytest.fixture
def sample():
    # 0..100 in tens: median 50, IQR 25..75, range 0..100 -- all checkable by hand
    return np.arange(0, 101, 10, dtype=float)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
