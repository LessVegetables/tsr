import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from scatterkit import demo_data


@pytest.fixture
def df():
    return demo_data(n_features=3, n_points=60, seed=0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
