"""scatterkit -- a scatter plot matrix whose every visual choice is a config line."""

from .config import ConfigError, Style, dump_config, load_config, parse_config
from .data import demo_data, load_data
from .defaults import FONT_FAMILY, PARAMS
from .render import ScatterResult, plot_scatter_matrix
from .stats import Fit, fit_polynomial, pair_stats

__all__ = [
    "ConfigError", "Style", "dump_config", "load_config", "parse_config",
    "demo_data", "load_data", "FONT_FAMILY", "PARAMS",
    "ScatterResult", "plot_scatter_matrix", "Fit", "fit_polynomial", "pair_stats",
]
