"""Output rendering. Data and errors both go to stdout in the same shape;
stderr is reserved for progress and diagnostics."""
import json as _json
from typing import Iterable, Sequence

from .errors import ComfyrackError


def table(rows: Iterable[Sequence[str]], headers: Sequence[str] | None = None) -> str:
    rows = [tuple(str(c) for c in r) for r in rows]
    if not rows:
        return ""
    ncols = len(rows[0])
    widths = [0] * ncols
    all_rows = ([tuple(headers)] if headers else []) + rows
    for r in all_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    def fmt(r):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip()
    return "\n".join(fmt(r) for r in all_rows)


def one_line(text: str, width: int = 90) -> str:
    """First sentence of a description, cut to `width`, for table cells."""
    text = " ".join(str(text).split())
    first = text.split(". ")[0].rstrip(".")
    return first if len(first) <= width else first[: width - 3].rstrip() + "..."


def empty(noun: str, context: str) -> str:
    """Definitive empty state. States the zero with context so an agent does not
    re-run with different flags to check whether the command actually worked."""
    return f"{noun}: 0 found {context}"


def error(exc: ComfyrackError) -> str:
    if exc.help_text:
        return f"error: {exc.message}\nhelp: {exc.help_text}"
    return f"error: {exc.message}"


class Result:
    """What a subcommand returns: the table text AND the JSON payload behind it.

    Every command used to reimplement `if args.json: return json.dumps(...)` inline
    and eight of fifteen simply forgot, so `comfyrack run --json` returned
    newline-joined paths and `comfyrack preflight --json` returned an ASCII table.
    Accepting a flag and ignoring it is worse than rejecting it -- the
    loud-unknown-flag rule exists so an agent can detect exactly this. Carrying both
    shapes out of the command and letting render() pick makes --json structural: a
    new subcommand cannot forget it, it can only choose the payload."""

    __slots__ = ("text", "data")

    def __init__(self, text: str, data=None):
        self.text = text
        self.data = data

    def __str__(self) -> str:
        return self.text


def render(data, as_json: bool = False) -> str:
    """The single rendering site. `data` is a Result, or a plain value for callers
    that have no separate JSON shape."""
    if isinstance(data, Result):
        return (_json.dumps(data.data, indent=2, default=str) if as_json
                else data.text)
    if as_json:
        return _json.dumps(data, indent=2, default=str)
    return str(data)
