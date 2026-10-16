"""Where the numbers come from: a file named in the config, or built-in demo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_EXCEL_MAGIC = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")   # .xlsx (zip), .xls (OLE)


def demo_data(n_features: int = 4, n_points: int = 150, seed: int = 0) -> pd.DataFrame:
    """Correlated features ``feature_1..N`` and a three-level ``group`` column."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, n_points)
    base = rng.normal(size=n_points) + 0.8 * group
    columns = {}
    for k in range(n_features):
        loading = np.cos(k * 1.1)             # some pairs correlate up, some down, some barely
        noise = rng.normal(scale=0.6, size=n_points)
        columns[f"feature_{k + 1}"] = (10 + 3 * k) + (2 + k) * (loading * base + noise)
    frame = pd.DataFrame(columns)
    frame["group"] = np.array(["A", "B", "C"])[group]
    return frame


def load_data(style) -> pd.DataFrame:
    if not style.data_path:
        return demo_data(style.demo_features, style.demo_points, style.random_seed)
    path = Path(style.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"data_path: no such file: {path}")
    # by content, not by extension: a ".xls" that is really CSV is common
    with path.open("rb") as handle:
        is_excel = handle.read(4) in _EXCEL_MAGIC
    if is_excel:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=style.data_separator, engine="python")
