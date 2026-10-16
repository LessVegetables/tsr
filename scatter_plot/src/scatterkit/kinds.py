"""Value kinds: how a ``param=value`` string becomes a Python value and back.

Only the machinery lives here. The parameters themselves -- names, defaults,
descriptions -- are all in ``defaults.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from matplotlib.colors import is_color_like

AUTO = "auto"

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}
_NAMED_LINESTYLES = {"-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted", "none"}


@dataclass(frozen=True)
class Kind:
    name: str                       # shown in the dumped config, e.g. "float" or "in | out | inout"
    parse: Callable[[str], Any]
    format: Callable[[Any], str]


@dataclass(frozen=True)
class P:
    """One configurable parameter."""

    name: str
    default: Any
    kind: Kind
    help: str = ""


@dataclass(frozen=True)
class Section:
    """A heading between groups of parameters; only affects the dumped config."""

    title: str


def _parse_bool(text: str) -> bool:
    low = text.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError("expected true or false")


def _parse_str(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.replace("\\n", "\n")


def _format_str(value: str) -> str:
    text = value.replace("\n", "\\n")
    return f'"{text}"' if text != text.strip() or " #" in text else text


def _parse_color(text: str) -> str:
    if not is_color_like(text):
        raise ValueError("not a colour matplotlib understands (try a name, #rrggbb, or none)")
    return text


def _split(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_linestyle(text: str):
    if text in _NAMED_LINESTYLES:
        return text
    try:
        dashes = tuple(float(item) for item in _split(text))
    except ValueError:
        dashes = ()
    if len(dashes) < 2 or len(dashes) % 2:
        raise ValueError("expected -, --, -., :, or a dash pattern such as 4,2 (on,off in points)")
    return (0, dashes)


def _format_linestyle(value) -> str:
    return value if isinstance(value, str) else ",".join(f"{v:g}" for v in value[1])


def _list_of(item: Kind, name: str) -> Kind:
    return Kind(
        name,
        lambda text: [item.parse(part) for part in _split(text)],
        lambda values: ",".join(item.format(v) for v in values),
    )


BOOL = Kind("true | false", _parse_bool, lambda v: "true" if v else "false")
INT = Kind("int", int, str)
FLOAT = Kind("float", float, lambda v: f"{v:g}")
STR = Kind("text", _parse_str, _format_str)
COLOR = Kind("colour", _parse_color, str)
LINESTYLE = Kind("- | -- | -. | : | on,off,...", _parse_linestyle, _format_linestyle)
FLOATS = _list_of(FLOAT, "comma-separated floats")
STRS = _list_of(STR, "comma-separated text")
COLORS = _list_of(COLOR, "comma-separated colours")


def choice(*options: str) -> Kind:
    def parse(text: str) -> str:
        if text not in options:
            raise ValueError(f"expected one of: {', '.join(options)}")
        return text

    return Kind(" | ".join(options), parse, str)


def auto(kind: Kind) -> Kind:
    """``kind``, or the word ``auto`` -- stored as None and resolved while drawing."""
    return Kind(
        f"{kind.name} | {AUTO}",
        lambda text: None if text.lower() == AUTO else kind.parse(text),
        lambda value: AUTO if value is None else kind.format(value),
    )
