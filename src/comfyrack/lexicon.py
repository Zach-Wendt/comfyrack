"""Project vocabulary -> model vocabulary.

A model knows only its training data. "Subject" is not a word Anima knows; trig_a is,
because a LoRA installed it. The lexicon holds that mapping, keyed by base model
lineage because a booru render and a natural-language render are different text."""
import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import lora_meta
from .errors import ComfyrackError, NotFoundError, UsageError

# ss_base_model_version records the checkpoint a LoRA was trained against, which is
# often a specific model line rather than its architecture lineage. anima and
# illustrious are distinct families sharing the illustrious lineage.
DEFAULT_LINEAGE_MAP = {
    "anima": "illustrious",
    "illustrious": "illustrious",
    "sdxl": "sdxl",
    "flux": "flux",
    "krea": "flux",
    "qwen": "qwen",
    "ltx": "ltx",
    "wan": "wan",
}

# How many trained tags to carry into a render. The long tail of a caption set is
# scene-specific noise, not identity.
MAX_TAGS = 12


@dataclass
class Entry:
    term: str
    kind: str = "term"        # character | location | prop | term | subject | scene
    renders: dict = field(default_factory=dict)   # lineage -> {trigger, tags} | {text}


class Lexicon:
    def __init__(self, config, lexicon_dir=None):
        self.config = config
        if lexicon_dir is not None:
            self.dir = Path(lexicon_dir)
        else:
            self.dir = config.path("lexicon")
        self._cache = None

    def all(self) -> dict:
        """Every known lexicon entry, keyed by lowercased term.

        Returns a deep copy of the cache, never the cache itself. A shallow
        copy would still share each Entry's `renders` dict -- the exact
        mutable state a caller is most likely to edit -- so a caller
        mutating what this returns (or two `Lexicon` instances built over
        the same directory) cannot corrupt each other's view.
        """
        if self._cache is None:
            out = {}
            if self.dir and self.dir.is_dir():
                for path in sorted(self.dir.glob("*.yaml")):
                    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                    if raw is None:
                        raw = {}
                    # A lexicon file that parses to a non-mapping (a list, a
                    # bare string/number) would make raw.get(...) below raise
                    # AttributeError -- a raw traceback, not a ComfyrackError
                    # -- and since callers glob many lexicon files at once,
                    # that traceback would not even say which file is broken.
                    if not isinstance(raw, dict):
                        raise ComfyrackError(
                            f"{path.name} is not a readable lexicon entry: "
                            f"top-level YAML is a {type(raw).__name__}, not a mapping",
                            help_text="a lexicon file must be a mapping with "
                                      "term/kind/renders keys",
                        )
                    term = str(raw.get("term") or path.stem).lower()
                    out[term] = Entry(term=term, kind=raw.get("kind", "term"),
                                      renders=raw.get("renders") or {})
            self._cache = out
        return copy.deepcopy(self._cache)

    def terms(self) -> set:
        return set(self.all())

    def has(self, term: str) -> bool:
        return str(term).lower() in self.all()

    def get(self, term: str) -> Entry:
        key = str(term).lower()
        entry = self.all().get(key)
        if entry is None:
            raise NotFoundError(
                f"unknown term {term!r} — not in lexicon",
                help_text=f"comfyrack lex add {key} --kind <character|location|prop|subject|scene>, "
                          f"or pass --allow-unknown",
            )
        return entry

    def render(self, term: str, lineage: str) -> str:
        """The text this term becomes for a given base model lineage.

        A trigger token, when present, is always prepended -- it activates a
        LoRA and is orthogonal to whether the rest of the block is
        natural-language `text` or booru `tags`. Dropping it just because
        `text` is also present would produce a fluent prompt with no
        activation: the plausible-and-wrong render this module exists to
        prevent.
        """
        entry = self.get(term)
        block = entry.renders.get(lineage)
        if block is None:
            have = ", ".join(sorted(entry.renders)) or "(none)"
            raise NotFoundError(
                f"term {term!r} has no render for lineage {lineage!r}",
                help_text=f"lineages defined for {term!r}: {have}",
            )
        parts = []
        if block.get("trigger"):
            parts.append(str(block["trigger"]))
        if "text" in block:
            parts.append(str(block["text"]).strip())
        elif block.get("tags"):
            parts.append(str(block["tags"]))
        rendered = ", ".join(p for p in parts if p).strip()
        if not rendered:
            have_keys = ", ".join(sorted(block)) or "(none)"
            # An empty or mis-keyed render block (e.g. "tag:" where "tags:"
            # was meant) must not fall through to "" -- prompt.assemble's
            # filter drops empty strings, so the subject would silently
            # vanish from the prompt instead of erroring.
            raise UsageError(
                f"term {term!r} lineage {lineage!r} has a render block with "
                f"no usable text (keys present: {have_keys})",
                help_text="check for a typo in the render block (e.g. 'tag:' "
                          "instead of 'tags:') or an empty trigger/tags/text value",
            )
        return rendered

    def save(self, entry: Entry) -> Path:
        if not self.dir:
            raise UsageError(
                "no lexicon path configured",
                help_text='add [paths] lexicon = "..." to .comfyrack/config.toml',
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        # Filename must agree with the lowercased key all() derives from
        # "term", or Entry(term="Subject") and Entry(term="subject") write two
        # files that collapse to one key on load -- last glob-wins, silently.
        path = self.dir / f"{entry.term.lower()}.yaml"
        path.write_text(
            yaml.safe_dump({"term": entry.term, "kind": entry.kind,
                            "renders": entry.renders}, sort_keys=False),
            encoding="utf-8")
        self._cache = None
        return path

    def sync(self, loras_dir, lineage_map=None, overwrite: bool = False) -> list:
        """Derive lexicon entries from LoRA metadata.

        Entries are DERIVED, not authored: ss_tag_frequency is exactly the vocabulary
        the LoRA responds to. Existing entries are preserved unless overwrite=True,
        so a hand-tuned render is never silently clobbered by a re-sync."""
        lineage_map = {**DEFAULT_LINEAGE_MAP, **(lineage_map or {})}
        all_entries = self.all()
        existing = set(all_entries)
        results = []

        # `ref` is the path relative to loras_dir (`Anima/AnimaEditV1.safetensors`
        # for the 13 of 206 real LoRAs that live in a subfolder). The TERM comes
        # from the bare stem -- a lexicon term is a word a prompt can contain, so
        # it must never carry a directory. The `lora:`/`derived_from:` values keep
        # the full reference, because those are what routing and a `lora_name`
        # widget compare against, and a bare name there is a LoRA dispatch cannot
        # find.
        for ref, meta in lora_meta.scan(loras_dir).items():
            term = ref.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if not meta.trigger:
                results.append((term, "skipped"))
                continue
            if term in existing and not overwrite:
                results.append((term, "skipped"))
                continue

            lineage = lineage_map.get(meta.base_model_family, meta.base_model_family)
            tags = [t for t in meta.tags[:MAX_TAGS] if t != meta.trigger]
            # Merge into the existing entry's renders rather than replacing
            # them wholesale: an entry can carry hand-authored or
            # previously-synced blocks for OTHER lineages (e.g. flux
            # alongside illustrious), and overwrite=True must only mean
            # "replace THIS lineage's block", never "replace every lineage
            # on this term".
            current = all_entries.get(term)
            renders = dict(current.renders) if current is not None else {}
            renders[lineage] = {
                "trigger": meta.trigger,
                "tags": ", ".join(tags),
                "lora": ref,
                "derived_from": ref,
            }
            entry = Entry(term=term, kind="character", renders=renders)
            self.save(entry)
            results.append((term, "updated" if term in existing else "created"))

        self._cache = None
        return results
