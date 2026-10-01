"""Civitai trainedWords fallback.

Some downloaded LoRAs have had their safetensors metadata stripped, so
ss_tag_frequency is absent and there is no local trigger to read. Civitai indexes
model versions by file hash and records trainedWords, which recovers the trigger.

Strictly a fallback: local file metadata always wins, since it is what the file
actually was trained with rather than what someone typed on a model page."""
import hashlib

import requests

from .errors import ComfyrackError

API = "https://civitai.com/api/v1/model-versions/by-hash"


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _lookup(hash_: str, session=None) -> dict:
    session = session or requests.Session()
    resp = session.get(f"{API}/{hash_}", timeout=30)
    if getattr(resp, "status_code", 200) >= 400:
        return {}
    return resp.json() or {}


def trained_words(hash_: str, session=None) -> list:
    """Public entry point: never lets a transport failure or malformed response
    body escape as a bare exception -- every user-triggerable error surfaces as a
    ComfyrackError with help_text."""
    try:
        data = _lookup(hash_, session=session)
    except Exception as exc:
        raise ComfyrackError(
            f"Civitai lookup for hash {hash_} failed: {exc}",
            help_text="check network connectivity and try again",
        ) from exc
    return list(data.get("trainedWords") or [])


def enrich(meta, path, session=None):
    """Fill gaps in a LoraMeta from Civitai. Never overwrites local values, and a
    network failure returns the input unchanged rather than breaking a sync."""
    if meta.trigger and meta.base_model_family:
        return meta
    try:
        data = _lookup(file_hash(path), session=session)
    except OSError:
        return meta
    if not data:
        return meta

    words = list(data.get("trainedWords") or [])
    if not meta.trigger and words:
        meta.trigger = words[0]
    if not meta.tags and len(words) > 1:
        meta.tags = words[1:]
    if not meta.base_model_family and data.get("baseModel"):
        meta.base_model_family = str(data["baseModel"]).lower()
    return meta
