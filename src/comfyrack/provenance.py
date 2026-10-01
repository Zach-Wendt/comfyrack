"""Run provenance sidecars.

A record is built on every run. Without the resolved parameters, seed, machine, and
graph hash, an ablation cannot be reproduced and a judging pass cannot state what
it judged.

A SIDECAR FILE is written next to every artifact a run produces on disk -- which
means every image/video/audio run, and no text run. An `output_type: "text"`
workflow writes no artifact at all (the verdict goes to stdout), so there is
nothing for a sidecar to sit beside; its record is still built and returned on
RunResult.record, so a caller that wants to persist it can. Writing a stray
.prov.json into the working directory for a run that produced no file would be
worse than not writing one."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def content_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_record(manifest, values: dict, machine_url: str, graph,
                 recipe_name: str | None = None) -> dict:
    return {
        "workflow": manifest.name,
        "family": manifest.family,
        "layer": manifest.layer,
        "recipe": recipe_name,
        "machine": machine_url,
        "graph_hash": graph.content_hash(),
        "values": dict(values),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_sidecar(out_path, record: dict) -> Path:
    out_path = Path(out_path)
    record = dict(record)
    if out_path.is_file():
        record["output_sha256"] = content_hash(out_path)
    side = out_path.with_suffix(".json")
    if side == out_path:
        # The output is itself a .json (a --out the caller chose, or a text render).
        # with_suffix() collapses onto it, and the sidecar would overwrite the very
        # artifact it exists to describe.
        side = out_path.with_name(out_path.name + ".prov.json")
    side.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return side
