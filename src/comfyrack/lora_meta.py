"""Read trained vocabulary out of LoRA safetensors headers.

A safetensors file begins with an 8-byte little-endian header length followed by
that many bytes of JSON. Kohya writes training provenance into __metadata__,
including ss_tag_frequency -- the caption tokens the LoRA was actually trained on.

That is the authoritative vocabulary oracle: it is exactly the set of tokens this
LoRA responds to, which no general word list can tell you."""
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .discover import normalize_ref
from .errors import ComfyrackError

MAX_HEADER_BYTES = 100 * 1024 * 1024

# Frequent descriptors that are never a trigger. A trigger is a token invented for
# the training run; these are ordinary vocabulary that happens to appear in every
# caption.
COMMON_TAGS = {
    "1girl", "1boy", "2girls", "solo", "looking at viewer", "smiling",
    "fair skin", "brown skin", "dark skin", "long hair", "short hair",
}


@dataclass
class LoraMeta:
    filename: str
    output_name: str = ""
    base_model_family: str = ""
    trigger: str | None = None
    tags: list = field(default_factory=list)
    train_images: int = 0


def read_header(path) -> dict:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                raise ValueError("file too short")
            (n,) = struct.unpack("<Q", raw_len)
            if n <= 0 or n > MAX_HEADER_BYTES:
                raise ValueError(f"implausible header length {n}")
            header = json.loads(f.read(n))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ComfyrackError(
            f"{path.name} is not a readable safetensors file: {exc}",
            help_text="check the file is a LoRA and is not truncated",
        )
    # These two guards are what keep scan() honest. A truncated file's surviving
    # bytes can still parse as valid JSON of the wrong SHAPE -- an array, a string,
    # null -- and .get() on that raises AttributeError, which is not a
    # ComfyrackError, so it sails straight through scan()'s except clause and aborts
    # the whole directory walk. That is the one thing scan() exists to prevent.
    if not isinstance(header, dict):
        raise ComfyrackError(
            f"{path.name} is not a readable safetensors file: header is a "
            f"{type(header).__name__}, not an object",
            help_text="check the file is a LoRA and is not truncated",
        )
    md = header.get("__metadata__", {})
    if not isinstance(md, dict):
        raise ComfyrackError(
            f"{path.name} has a malformed __metadata__ block: "
            f"{type(md).__name__}, not an object",
            help_text="the file may be truncated, or written by a trainer that "
                      "does not follow the kohya metadata convention",
        )
    return md


def _pick_trigger(ordered_tags: list, freq: dict) -> str | None:
    """The trigger is a token coined for the training run: it appears in every
    caption, so it sits at maximum frequency, and it is not ordinary vocabulary."""
    if not ordered_tags:
        return None
    top_freq = freq[ordered_tags[0]]
    top = [t for t in ordered_tags if freq[t] == top_freq]
    for tag in top:
        if tag in COMMON_TAGS or tag.startswith("("):
            continue
        if any(ch.isdigit() for ch in tag) or " " not in tag:
            return tag
    for tag in top:
        if tag not in COMMON_TAGS and not tag.startswith("("):
            return tag
    return None


def read_meta(path) -> LoraMeta:
    path = Path(path)
    md = read_header(path)

    freq = {}
    raw_freq = md.get("ss_tag_frequency")
    if raw_freq:
        try:
            for _dataset, tags in json.loads(raw_freq).items():
                for tag, count in tags.items():
                    freq[tag] = freq.get(tag, 0) + int(count)
        except (ValueError, TypeError, AttributeError) as exc:
            # PRESENT-but-unparseable is not the same as ABSENT, and the difference
            # matters more here than anywhere else in the package. Resetting to {}
            # would make a LoRA whose caption data is corrupt look byte-identical,
            # to every caller, to a style LoRA that never had captions: no trigger,
            # no tags, no complaint. `lex sync` would then report it "skipped" and
            # move on, and the vocabulary oracle this whole ring is built on would
            # be quietly missing an entry. Raise; scan() turns it back into a skip,
            # so one bad file still cannot stop a sync.
            raise ComfyrackError(
                f"{path.name} has a malformed ss_tag_frequency block: {exc}",
                help_text="re-download the LoRA, or write its lexicon entry by hand "
                          "with `comfyrack lex add`",
            ) from exc

    ordered = sorted(freq, key=lambda t: (-freq[t], t))
    try:
        train_images = int(md.get("ss_num_train_images", 0))
    except (TypeError, ValueError):
        train_images = 0

    return LoraMeta(
        filename=path.name,
        output_name=md.get("ss_output_name", ""),
        base_model_family=md.get("ss_base_model_version", ""),
        trigger=_pick_trigger(ordered, freq),
        tags=ordered,
        train_images=train_images,
    )


def scan(loras_dir) -> dict:
    """reference -> LoraMeta for every readable LoRA under loras_dir.

    The key is the path RELATIVE to `loras_dir`, separator-normalised --
    `Anima/AnimaEditV1.safetensors`, not `AnimaEditV1.safetensors`. That is the
    same shape ComfyUI's own `/object_info` combo lists and a workflow's
    `lora_name` widget use, which is what makes this mapping directly comparable
    to them. Keying by the bare `path.name` while rglob-ing subdirectories made
    the two disagree for the 13 of 206 real LoRAs that live in a subfolder, and
    disagree ASYMMETRICALLY: `dispatch._missing()` called such a LoRA
    permanently unroutable (loud, wrong) while `runner.lora_lineages()` just
    dropped it out of the lineage check (silent, wrong).

    Unreadable files are skipped, not fatal: one corrupt LoRA in a directory of 205
    must not stop a lexicon sync."""
    root = Path(loras_dir)
    out = {}
    for path in sorted(root.rglob("*.safetensors")):
        try:
            meta = read_meta(path)
        except ComfyrackError:
            continue
        try:
            ref = path.relative_to(root)
        except ValueError:                                  # pragma: no cover
            ref = path.name
        out[normalize_ref(ref)] = meta
    return out
