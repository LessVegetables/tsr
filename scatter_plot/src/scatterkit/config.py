"""``Style`` (one attribute per parameter) and the ``param=value`` config format."""

from __future__ import annotations

import copy
import difflib
import re
from pathlib import Path
from typing import Any, Optional, Union

from .defaults import PARAMS
from .kinds import P, Section

SPEC: dict[str, P] = {p.name: p for p in PARAMS if isinstance(p, P)}

_INLINE_COMMENT = re.compile(r"\s#(\s.*)?$")   # " # ..." -- but not the # of "#ff0000"


class ConfigError(ValueError):
    pass


def _canonical(name: str) -> str:
    # regression_is_shown and regression_shown are the same parameter
    return name[: -len("_is_shown")] + "_shown" if name.endswith("_is_shown") else name


def _unknown(name: str) -> str:
    close = difflib.get_close_matches(name, SPEC, n=3, cutoff=0.6)
    hint = f" -- did you mean {', '.join(close)}?" if close else ""
    return f"unknown parameter '{name}'{hint}"


class Style:
    """Every parameter of ``defaults.py`` as an attribute, defaults filled in."""

    def __init__(self, **overrides: Any) -> None:
        overrides = {_canonical(name): value for name, value in overrides.items()}
        for name in overrides:
            if name not in SPEC:
                raise ConfigError(_unknown(name))
        for name, param in SPEC.items():
            setattr(self, name, overrides[name] if name in overrides else copy.copy(param.default))

    def derive(self, **changes: Any) -> "Style":
        return Style(**{**vars(self), **changes})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Style) and vars(self) == vars(other)

    def __repr__(self) -> str:
        changed = {k: v for k, v in vars(self).items() if v != SPEC[k].default}
        return f"Style({', '.join(f'{k}={v!r}' for k, v in changed.items())})"


def parse_config(text: str, source: str = "<config>") -> dict[str, Any]:
    """``param=value`` lines -> ``{param: typed value}``. ``#`` starts a comment."""
    values: dict[str, Any] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{source}:{number}"
        if "=" not in line:
            raise ConfigError(f"{where}: expected param=value, got '{line}'")
        name, value = line.split("=", 1)
        name = _canonical(name.strip())
        value = _INLINE_COMMENT.sub("", value).strip()
        if name not in SPEC:
            raise ConfigError(f"{where}: {_unknown(name)}")
        try:
            values[name] = SPEC[name].kind.parse(value)
        except ValueError as error:
            raise ConfigError(f"{where}: bad value '{value}' for {name}: {error}") from None
    return values


def load_config(path: Union[str, Path], base: Optional[Style] = None) -> Style:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error.strerror}") from None
    return (base or Style()).derive(**parse_config(text, str(path)))


def dump_config(style: Optional[Style] = None) -> str:
    """A complete config file -- every parameter, its value, kind and description."""
    style = style or Style()
    lines = [
        "# scatterkit config -- one param=value per line, '#' starts a comment.",
        "# Every parameter is listed with its current value; delete the lines you",
        "# do not want to override. The defaults themselves live in defaults.py.",
    ]
    for item in PARAMS:
        if isinstance(item, Section):
            lines += ["", "", f"# ===== {item.title} ====="]
            continue
        note = f"{item.help}  [{item.kind.name}]" if item.help else f"[{item.kind.name}]"
        lines += ["", f"# {note}", f"{item.name}={item.kind.format(getattr(style, item.name))}"]
    return "\n".join(lines) + "\n"
