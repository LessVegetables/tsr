import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "survived": [1, 0, 1, 1, 0, 0, 1, 0],
            "sex": ["f", "m", "f", "m", "m", "f", "f", "m"],
            "age": [30, 5, np.nan, 12, 40, 60, 8, 22],
            "cabin": ["A1", None, "B2", None, None, "C3", None, None],
        }
    )


@pytest.fixture
def conditions():
    return {
        "Survived": "survived",
        "Male": "sex == 'm'",
        "Child": "age < 18",
        "No cabin": "cabin.isna()",
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
