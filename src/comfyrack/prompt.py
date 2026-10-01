"""Prompt assembly from lexicon terms.

The agent supplies terms and an action; comfyrack renders each term in the target
family's dialect and joins them. The agent does not write positive_text, because an
agent writing free prompt text produces fluent, confident, wrong vocabulary.

Wildcards expand HERE, client-side. Expanding them in a node would leave the final
prompt unknown at submit time, defeating provenance and the guard."""
import random
import re
from pathlib import Path

from .errors import UsageError

TERM_RE = re.compile(r"\{([a-zA-Z0-9_\-]+)\}")          # {char_a} -- no pipe
CHOICE_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")          # {red|blue} -- has a pipe
FILE_RE = re.compile(r"__([a-zA-Z0-9_\-/]+)__")          # __weather__

JOIN = {"booru": ", ", "natural": ". "}


def expand_wildcards(text: str, rng=None, wildcards_dir=None) -> str:
    """Expand {a|b|c} choices and __file__ references. A brace group without a pipe
    is a lexicon term and is left alone for substitute() to handle."""
    rng = rng or random.Random()

    def pick_choice(m):
        return rng.choice(m.group(1).split("|")).strip()

    text = CHOICE_RE.sub(pick_choice, text)

    def pick_file(m):
        name = m.group(1)
        if wildcards_dir is None:
            raise UsageError(
                f"wildcard file __{name}__ used but no wildcards directory is configured",
                help_text='add [paths] wildcards = "..." to .comfyrack/config.toml',
            )
        path = Path(wildcards_dir) / f"{name}.txt"
        if not path.is_file():
            raise UsageError(
                f"wildcard file __{name}__ not found",
                help_text=f"expected {path} under {wildcards_dir}",
            )
        options = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
        if not options:
            raise UsageError(f"wildcard file __{name}__ is empty",
                             help_text=f"add one option per line to {path}")
        return rng.choice(options)

    return FILE_RE.sub(pick_file, text)


RESIDUAL_BRACE_RE = re.compile(r"\{[^{}]*\}?|\}")


def substitute(text: str, lexicon, lineage: str) -> str:
    """Replace every {term} with its render for this lineage. An unknown term raises
    rather than passing through -- an unresolved brace reaches the model as literal
    punctuation and silently contributes nothing.

    Callers must run expand_wildcards() on `text` BEFORE calling substitute(): a
    {a|b} choice or {unclosed brace that reaches this function is, by this point,
    genuinely an error rather than an unexpanded wildcard, and is rejected loudly
    below rather than passed through as literal punctuation. (This check is brace-
    based, so it does not catch an unexpanded __file__ wildcard -- that syntax has
    no braces and is still the caller's responsibility; see assemble()'s docstring.)"""
    out = TERM_RE.sub(lambda m: lexicon.render(m.group(1), lineage), text)
    leftover = RESIDUAL_BRACE_RE.search(out)
    if leftover:
        raise UsageError(
            f"unresolved brace group {leftover.group(0)!r} in prompt text",
            help_text="lexicon terms are {letters/digits/_/- only}; a piped choice "
                       "or a __file__ wildcard must go through expand_wildcards() "
                       "before substitute() sees the text",
        )
    return out


def assemble(lexicon, lineage: str, dialect: str, subject=None, location=None,
             action: str = "", style: str = "", extra: str = "") -> str:
    """Assemble a prompt from lexicon-rendered parts.

    Callers are responsible for running expand_wildcards() on any free-text field
    (chiefly `action`) BEFORE calling assemble() -- this function does not expand
    wildcards itself. An unexpanded {a|b} choice left in `action` is not silent:
    substitute()'s residual-brace check (see above) raises UsageError as soon as it
    reaches the action slot. An unexpanded __file__ wildcard is NOT caught by that
    check (no braces involved) and will currently pass through as literal text --
    callers must expand wildcards before calling assemble() to avoid that gap."""
    if dialect not in JOIN:
        raise UsageError(
            f"unknown dialect {dialect!r}",
            help_text=f"valid dialects: {', '.join(sorted(JOIN))}",
        )
    joiner = JOIN[dialect]
    parts = []
    if subject:
        parts.append(lexicon.render(subject, lineage))
    if location:
        parts.append(lexicon.render(location, lineage))
    if action:
        parts.append(substitute(action, lexicon, lineage))
    if style:
        parts.append(style)
    if extra:
        parts.append(extra)
    return joiner.join(p for p in (s.strip() for s in parts) if p)
