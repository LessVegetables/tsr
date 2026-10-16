"""python -m scatterkit [--config=PATH] [--set PARAM=VALUE ...] [--dump-config[=PATH]]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import ConfigError, Style, dump_config, load_config, parse_config


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scatterkit",
        description="Scatter plot matrix for N features. With no arguments: the defaults, on demo data.",
    )
    parser.add_argument("--config", metavar="PATH",
                        help="text file with one param=value per line; anything not listed keeps its default")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PARAM=VALUE",
                        help="override one parameter on top of the config; repeatable")
    parser.add_argument("--dump-config", nargs="?", const="-", metavar="PATH",
                        help="write every parameter with its current value to PATH (stdout if omitted) and exit")
    args = parser.parse_args(argv)

    try:
        style = load_config(args.config) if args.config else Style()
        style = style.derive(**parse_config("\n".join(args.overrides), "--set"))

        if args.dump_config is not None:
            text = dump_config(style)
            if args.dump_config == "-":
                sys.stdout.write(text)
            else:
                Path(args.dump_config).write_text(text, encoding="utf-8")
            return 0

        import matplotlib
        if not style.output_show_window:
            matplotlib.use("Agg")
        from .data import load_data
        from .render import plot_scatter_matrix

        result = plot_scatter_matrix(load_data(style), style)
        path = result.savefig()
    except (ConfigError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"saved {path}")
    if style.output_show_window:
        result.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
