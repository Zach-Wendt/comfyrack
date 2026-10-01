"""Multi-machine routing.

Machines are declared in config, not expressed as workflow nodes. ComfyUI has no
native job federation, and the in-graph options are custom-node extensions that must
be installed and version-matched on every machine. Fan-out across N jobs and M machines
is a scheduler, which is a client concern -- and routing needs knowledge the graph
does not have: which machine holds which LoRA.

Model weights do not sync between machines (only code does), so a job must go only to
a machine that actually holds what it needs. That turns a documented footgun into a
routing constraint."""
from dataclasses import dataclass, field

from . import discover
from .errors import ComfyrackError


@dataclass
class Capability:
    machine: str
    url: str
    models: dict = field(default_factory=dict)     # category -> [filenames]
    node_classes: set = field(default_factory=set)


@dataclass
class Job:
    shot_name: str
    requirements: dict = field(default_factory=dict)   # category -> [filenames]
    fn: object = None
    required_classes: set = field(default_factory=set)


def index(machines: dict, client_factory) -> tuple[dict, dict]:
    """Probe each machine for its models and node classes.

    NOTE: this calls client.object_info() directly and discover.models() calls it
    again internally -- two call sites, but ComfyClient memoizes /object_info per
    instance, so a real machine still takes exactly one HTTP round trip here, not
    two.

    Returns (capabilities, errors). One unreachable machine must not abort indexing
    of the rest -- the same "one failure never stops the queue" rule run_queue
    follows three functions below. A machine whose probe raises is left out of
    `capabilities` and its ComfyrackError is recorded in `errors`, keyed by machine
    name, so an offline machine is reported rather than silently vanishing from the
    capability index."""
    caps = {}
    errors = {}
    for name, url in machines.items():
        client = client_factory(url)
        try:
            oi = client.object_info()
            caps[name] = Capability(machine=name, url=url,
                                    models=discover.models(client),
                                    node_classes=set(oi.keys()))
        except ComfyrackError as exc:
            errors[name] = exc
    return caps, errors


def _missing(capability: Capability, requirements: dict, required_classes) -> list:
    """Which of a job's requirements this machine does not hold.

    Model names are compared through `discover.ref_index`/`ref_lookup`, not with
    `in`. A requirement and a machine's enumerated list can spell the same file
    differently -- separator (`Anima\\x.safetensors` on a Windows machine vs
    `Anima/x.safetensors` on a Linux one) and subfolder (a character recipe
    writing the bare filename where the machine reports the subpath). A raw `in`
    called such a LoRA missing, and because "missing everywhere" is reported as
    `unroutable`, the job was declared permanently unroutable over a spelling
    while the file sat on the machine."""
    missing = []
    for category, names in (requirements or {}).items():
        have = discover.ref_index(capability.models.get(category) or [])
        missing.extend(n for n in names if discover.ref_lookup(have, n) is None)
    missing.extend(c for c in (required_classes or set())
                   if c not in capability.node_classes)
    return missing


def capable(capability: Capability, requirements: dict, required_classes) -> bool:
    return not _missing(capability, requirements, required_classes)


def route(requirements: dict, capabilities: dict, required_classes=None) -> list:
    return [name for name, cap in sorted(capabilities.items())
            if capable(cap, requirements, required_classes or set())]


def _unroutable_detail(job, capabilities: dict) -> str:
    """The documented remedy for `unroutable` is "copy the missing file to a
    candidate machine" -- so this must name only files that are missing
    EVERYWHERE, not the union of what any one machine lacks. A job needing
    char_b.safetensors + nobody.safetensors where `local` already holds char_b must
    not send the user chasing char_b."""
    if not capabilities:
        return "no machines are configured"
    per_machine = [set(_missing(cap, job.requirements, job.required_classes))
                   for cap in capabilities.values()]
    universally_missing = sorted(set.intersection(*per_machine))
    if universally_missing:
        return f"no machine holds: {', '.join(universally_missing)}"
    # Every individual file/node class exists on SOME machine, but no single
    # machine holds the whole combination the job needs at once.
    return ("no single machine holds every required model/node class together "
            "-- requirements are split across machines")


def run_queue(jobs, capabilities: dict, runner_fn, max_retries: int = 1) -> list:
    """Run every job on a capable machine. One failure never stops the queue: a batch
    of 9 shots should report 8 successes and 1 failure, not abort at shot 3.

    Retries rotate round-robin across every capable machine, in sorted name order.
    When only one machine is capable, `targets[attempt % len(targets)]` retries
    that SAME machine -- intentional: a transient fault (e.g. CUDA OOM under
    concurrent load) can clear on a second attempt even with nowhere else to send
    the job."""
    results = []
    for job in jobs:
        targets = route(job.requirements, capabilities, job.required_classes)
        if not targets:
            results.append({
                "shot": job.shot_name, "status": "unroutable", "machine": None,
                "detail": _unroutable_detail(job, capabilities),
            })
            continue

        last_error = None
        last_machine = None
        attempts = []
        for attempt in range(max_retries + 1):
            machine = targets[attempt % len(targets)]
            try:
                payload = runner_fn(job, machine)
                results.append({"shot": job.shot_name, "status": "ok",
                                "machine": machine, "detail": "", "result": payload})
                break
            except Exception as exc:                      # noqa: BLE001
                # Deliberately broad, not narrowed to "transient-looking" types: a
                # render can fail in ways that don't fit a tidy taxonomy, and
                # guessing wrong would silently drop a job that a retry (possibly
                # on a different machine) could still have rescued. The cost of
                # one extra bounded retry is smaller than the cost of a false
                # "this can't possibly work" classification.
                last_error = exc
                last_machine = machine
                attempts.append({"machine": machine, "error": str(exc)})
        else:
            # last_machine/last_error come from the SAME attempt, so the report
            # blames the machine that actually produced the error instead of
            # pairing the first-tried machine with the last-tried machine's
            # message.
            results.append({"shot": job.shot_name, "status": "failed",
                            "machine": last_machine, "detail": str(last_error),
                            "attempts": attempts})
    return results
