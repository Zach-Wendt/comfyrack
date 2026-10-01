"""The confabulation guard.

An agent is a co-user and is the principal source of the failure this prevents: it
writes project vocabulary the model was never trained on, fluently and with no felt
uncertainty, and the render comes back plausible and wrong.

Strictness tracks oracle confidence. Where the source is authoritative -- the
lexicon, a LoRA's trained tags -- an unknown or contradicting term is a hard failure.
Where no authoritative source exists, it is a warning. A uniformly strict guard over
a noisy oracle gets --allow-unknown passed reflexively, after which it protects
nothing."""
import re
from dataclasses import dataclass

from .errors import UsageError
from .prompt import TERM_RE

# Mutually exclusive attribute groups. A LoRA trained on one member of a group and
# prompted with another produces a silently degraded render rather than an error.
CONTRADICTION_GROUPS = [
    {"blonde", "brunette", "black hair", "red hair", "dark brown hair", "dark hair",
     "white hair", "blue hair", "pink hair"},
    {"fair skin", "pale skin", "brown skin", "dark skin", "tan skin"},
    {"blue eyes", "green eyes", "brown eyes", "blue-green eyes", "grey eyes",
     "amber eyes"},
    {"long hair", "short hair", "bob cut"},
]

# Where a finding's authority comes from. This is what --allow-unknown keys off:
# the flag exists to relax the LEXICON check ("this term is new, let me through"),
# and it must not be able to switch off the LoRA check, which is the strongest
# oracle in the package -- a contradiction there is a render that comes back
# plausible and wrong with no error anywhere.
SOURCE_LEXICON = "lexicon"        # authoritative: the project's own vocabulary
SOURCE_LORA = "lora"              # authoritative: the LoRA's trained tags
SOURCE_UNGROUNDED = "ungrounded"  # no authoritative source exists -> warning tier

DOWNGRADABLE_SOURCES = frozenset({SOURCE_LEXICON})

# A capitalised word is a proper-noun candidate unless it is KNOWN to be sentence
# case. Mid-text, "known" means it follows sentence-ending punctuation. At position
# 0 the string itself has to be prose for that to hold -- see _head_is_tag_shaped.
PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)\b([A-Z][a-z]{2,})\b")

# A snake_case token in free prompt text. Sometimes real booru vocabulary
# (looking_at_viewer), sometimes pure agent invention (flowing_water_magic), and
# nothing in the project can tell the two apart on its own -- which is exactly why
# these are the warning tier rather than a hard failure.
FREE_TAG_RE = re.compile(r"\b[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+\b")

SENTENCE_PUNCT_RE = re.compile(r"[.!?]")

# A comma-delimited segment this short is a tag, not a clause. Real prompt text in
# this pipeline runs 8-20 words per comma segment ("standing in a market square as
# a stone fountain erupts violently behind her"); real tags run one to three
# ("moonlight", "cel shading", "looking at viewer").
MAX_TAG_WORDS = 3


@dataclass
class Finding:
    severity: str    # error | warning
    term: str
    detail: str
    source: str = SOURCE_LEXICON   # lexicon | lora | ungrounded


def _phrases(text: str) -> str:
    return text.lower()


def _head_is_tag_shaped(text: str) -> bool:
    """Does the text OPEN with a tag rather than a sentence?

    A capitalised word at string position 0 means two completely different things
    depending on the shape of the string, and the old `(?<!^)` lookbehind blanket-
    skipped both. In prose, position-0 capitalisation is sentence case and says
    nothing about the word. In a comma-separated tag list -- the dominant prompt
    shape in this pipeline -- it says nothing grammatically at all, so an unknown
    capitalised word there is exactly as suspicious as one in the middle, and
    skipping it left the guard blind to `Location X, night, moonlight`.

    The discriminator is the first comma segment, and ONLY that segment: a tag is
    short and carries no sentence punctuation; a clause is longer, or ends in a
    stop. Both votes are scoped to the head because the question is what position 0
    is, and nothing further down the string can answer it -- while a period further
    down is routine. `(straight blonde hair:1.2)` is this pipeline's canonical
    weighted-tag syntax and contains one, so a whole-string punctuation scan would
    read every weighted tag list as prose and quietly reopen the position-0 blind
    spot for the project's own standard notation.

    Erring toward "tag" is deliberate -- a false positive here is a loud, escapable
    error, while a false negative is the silent wrong render this module exists to
    prevent."""
    head = text.split(",", 1)[0].strip()
    if not head or SENTENCE_PUNCT_RE.search(head):
        return False
    return len(head.split()) <= MAX_TAG_WORDS


def check_terms(text: str, lexicon, lora_meta=None) -> list:
    """Ground every piece of vocabulary in `text` against the best oracle available.

    `lora_meta`, when given, is only consulted to CLEAR free-floating tags: a tag
    the LoRA was demonstrably trained on is grounded, so it is not reported at all.
    Contradictions against that same metadata are check_contradiction's job."""
    findings = []
    for m in TERM_RE.finditer(text):
        term = m.group(1)
        if not lexicon.has(term):
            findings.append(Finding("error", term, "not in lexicon",
                                    source=SOURCE_LEXICON))

    known = {t.lower() for t in lexicon.terms()}
    stripped = TERM_RE.sub("", text)
    head_is_tag = _head_is_tag_shaped(stripped)
    for m in PROPER_NOUN_RE.finditer(stripped):
        if m.start() == 0 and not head_is_tag:
            continue    # sentence case at the head of a prose string
        word = m.group(1)
        if word.lower() not in known:
            findings.append(Finding(
                "error", word,
                "capitalised term not in lexicon; a model renders it as nothing "
                "or pattern-matches it to something unrelated",
                source=SOURCE_LEXICON))

    findings.extend(_check_free_tags(stripped, known, lora_meta))
    return findings


def _check_free_tags(stripped: str, known: set, lora_meta=None) -> list:
    """The third strictness tier: warn on free-floating tags.

    A bare snake_case token in the prompt is vocabulary the agent asserted without
    routing it through the lexicon. If the target LoRA was trained on it, it is
    grounded and there is nothing to say. If the lexicon has it, likewise. Otherwise
    NO authoritative source exists either way -- the token may be a real booru tag
    or an invention -- and §10a's answer to that is a warning, not a hard failure:
    hard-failing on an oracle this weak is what trains a user to pass
    --allow-unknown reflexively, after which the guard protects nothing."""
    trained = set()
    if lora_meta is not None and getattr(lora_meta, "tags", None):
        trained = {str(t).lower() for t in lora_meta.tags}
    findings = []
    for m in FREE_TAG_RE.finditer(stripped):
        tag = m.group(0)
        lowered = tag.lower()
        if lowered in known or lowered in trained:
            continue
        where = (f"{lora_meta.filename} was not trained on it"
                 if trained else "no LoRA vocabulary was available to check it against")
        findings.append(Finding(
            "warning", tag,
            f"free-floating tag: not in the lexicon and {where}, so nothing "
            f"authoritative says whether the model knows it",
            source=SOURCE_UNGROUNDED))
    return findings


def check_contradiction(text: str, lora_meta) -> list:
    """Flag a description that contradicts the LoRA's trained vocabulary."""
    if not lora_meta or not lora_meta.tags:
        return []
    body = _phrases(text)
    trained = _phrases(" ".join(lora_meta.tags))
    findings = []
    for group in CONTRADICTION_GROUPS:
        trained_members = sorted(a for a in group if a in trained)
        if not trained_members:
            continue
        for attr in sorted(group):
            if attr in trained_members:
                continue
            if attr in body:
                findings.append(Finding(
                    "error", attr,
                    f"{lora_meta.filename} was trained on "
                    f"{' / '.join(trained_members)}, but the prompt says {attr!r}",
                    source=SOURCE_LORA))
    return findings


def check(text: str, lexicon, lora_meta=None) -> list:
    return (check_terms(text, lexicon, lora_meta=lora_meta)
            + check_contradiction(text, lora_meta))


def _help_for(hard: list) -> str:
    """Name the fix that actually applies. Offering --allow-unknown for a LoRA
    contradiction would advertise an escape hatch that deliberately does not open,
    which reads as a bug the first time someone tries it."""
    parts = []
    if any(f.source == SOURCE_LEXICON for f in hard):
        parts.append("comfyrack lex add <term> --kind <character|location|prop|subject|scene>, "
                     "or pass --allow-unknown")
    if any(f.source == SOURCE_LORA for f in hard):
        parts.append("describe the subject with the vocabulary its LoRA was "
                     "trained on, or drop the contradicting attribute -- a trained "
                     "attribute cannot be overridden from the prompt")
    return "; ".join(parts)


def enforce(findings: list, allow_unknown: bool = False) -> list:
    """Raise on errors; return warning lines for the caller to print to stderr.

    --allow-unknown downgrades UNKNOWN-TERM errors only. A finding sourced from a
    LoRA's trained tags stays a hard failure: the flag says "this vocabulary is new
    to the lexicon", which is no answer at all to "this prompt contradicts what the
    LoRA was trained on"."""
    hard, soft = [], []
    for f in findings:
        if f.severity != "error":
            soft.append(f)
        elif allow_unknown and f.source in DOWNGRADABLE_SOURCES:
            soft.append(f)
        else:
            hard.append(f)
    if hard:
        lines = "; ".join(f"{f.term} ({f.detail})" for f in hard)
        raise UsageError(
            f"prompt vocabulary not grounded: {lines}",
            help_text=_help_for(hard),
        )
    return [f"warning: {f.term} -- {f.detail}" for f in soft]
