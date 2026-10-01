"""One function per subcommand. Each returns an out.Result carrying the table text
and the JSON payload behind it; main() calls out.render() once and --json picks.
Output rules: compact tables, total counts, definitive empty states,
and a `help:` next-step line where the next action is not obvious."""
import re
import sys

from .. import out
from .. import discover as _discover, runner
from ..registry import FORM_KEYS
from ..errors import ComfyrackError, NotFoundError, UsageError


def _stderr_progress(prefix: str = ""):
    """A progress callback that writes to STDERR.

    STDERR is not a stylistic choice here: stdout carries the command's result
    and has to stay parseable (`comfyrack run --json` is piped, `comfyrack run`
    prints bare paths a caller reads), so a five-minute render's step counter
    cannot go there. `Rack.run()` defaults to no callback, which is what keeps
    library use and every existing test byte-identical -- only a CLI opts in."""
    def report(line: str) -> None:
        print(f"{prefix}{line}", file=sys.stderr, flush=True)
    return report

# --family and --name both become path segments under the registry root, so an
# unvalidated `--family ../../..` writes outside it entirely.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_segment(kind: str, value: str) -> str:
    if not _SAFE_SEGMENT.match(value or ""):
        raise UsageError(
            f"invalid --{kind} {value!r}: it becomes a directory or file name",
            help_text=f"--{kind} may contain only letters, digits, '.', '_' and '-'",
        )
    return value


def cmd_home(rack, args) -> out.Result:
    """Bare `comfyrack` shows live data, never help text."""
    counts = rack.registry.family_counts()
    shadowed = rack.registry.shadowed()
    data = {"families": [{"family": fam, "count": n, "note": note}
                         for fam, n, note in counts],
            "shadowed": [{"name": name, "winner": win, "shadowed": lost}
                         for name, win, lost in shadowed]}
    rows = [(f"{fam}/", str(n), note) for fam, n, note in counts]
    total = sum(int(r[1]) for r in rows)

    # Registry.shadowed() computed this and nothing consumed it, so a project
    # manifest winning over a builtin of the same name -- the exact scenario the
    # three-layer design creates -- produced no signal at all, and was
    # indistinguishable from the builtin having changed underneath you.
    shadow_line = ""
    if shadowed:
        listed = ", ".join(f"{name} ({win} over {lost})" for name, win, lost in shadowed)
        shadow_line = f"\nshadowed: {len(shadowed)} name(s) resolved by precedence: {listed}"

    if not rows:
        return out.Result(
            out.empty("workflows", "in any registry layer")
            + "\nhelp: comfyrack onboard <path.json> --family <f> --name <n>",
            data)
    body = out.table(rows)
    return out.Result(
        f"registry: {total} workflows across {len(rows)} families\n{body}"
        f"{shadow_line}\n"
        "help:\n"
        "  comfyrack list --family <f>     expand one family\n"
        "  comfyrack find <query>          search every family\n"
        "  comfyrack describe <name>       flags for one workflow",
        data)


def cmd_list(rack, args) -> out.Result:
    layer = getattr(args, "layer", None)
    mode = getattr(args, "mode", None)
    if not (args.family or args.all or layer or mode):
        return cmd_home(rack, args)

    if args.family:
        manifests = rack.list(family=args.family)
        scope_desc = f"in family {args.family!r}"
        headers = ("name", "layer", "description")
        row_of = lambda m: (m.name, m.layer, out.one_line(m.description))
    else:
        manifests = rack.list()
        scope_desc = "in any registry layer"
        headers = ("workflow", "layer", "description")
        row_of = lambda m: (f"{m.family}/{m.name}", m.layer, out.one_line(m.description))

    if layer:
        manifests = [m for m in manifests if m.layer == layer]
        scope_desc += f" at layer {layer!r}"

    if mode:
        manifests = [m for m in manifests if mode in m.mode]
        scope_desc += f" with mode {mode!r}"

    data = [_m(m) for m in manifests]
    if not manifests:
        return out.Result(out.empty("workflows", scope_desc), data)
    return out.Result(out.table([row_of(m) for m in manifests], headers=headers)
                      + f"\ncount: {len(manifests)}", data)


def cmd_find(rack, args) -> out.Result:
    matches = rack.find(args.query)
    mode = getattr(args, "mode", None)
    scope_desc = f"matching {args.query!r}"
    if mode:
        matches = [m for m in matches if mode in m.mode]
        scope_desc += f" with mode {mode!r}"
    data = [_m(m) for m in matches]
    if not matches:
        return out.Result(out.empty("workflows", scope_desc), data)
    return out.Result(
        out.table([(f"{m.family}/{m.name}", m.layer, out.one_line(m.description))
                   for m in matches],
                  headers=("workflow", "layer", "description"))
        + f"\ncount: {len(matches)}"
        + "\nhelp: comfyrack describe <name>",
        data)


def cmd_describe(rack, args) -> out.Result:
    m = rack.describe(args.name)
    data = _m(m, full=True)
    # Form metadata comes from the target ComfyUI's /object_info (machine chosen like
    # `run`: --machine, else COMFY_URL, else config default). Nothing answering is
    # not an error: the manifest-only output stands and live is false.
    live = None
    if m.flags:
        try:
            client = rack.client(args.machine)
            live = _discover.form_meta(m, runner.load_graph(m, client), client)
        except (ComfyrackError, OSError, ValueError) as exc:
            if isinstance(exc, UsageError):
                raise
    data["live"] = live is not None
    for k, extra in (live or {}).items():
        data["flags"][k].update(extra)
    fam = rack.registry.family(m.family)
    head = [f"{m.family}/{m.name} [{m.layer}] -- {m.description}",
            f"output: {m.output_type} from node titled {m.output_node_title!r}"]
    if fam and fam.prompt_dialect:
        head.append(f"prompt dialect: {fam.prompt_dialect} "
                    f"(base model family: {fam.base_model_family})")
    if not m.flags:
        return out.Result("\n".join(head) + "\n" + out.empty("flags", "on this workflow"),
                          data)
    rows = [(k, f.type, "required" if f.required else f"default={f.default!r}", f.help)
            for k, f in sorted(m.flags.items())]
    return out.Result(
        "\n".join(head) + "\n"
        + out.table(rows, headers=("key", "type", "requirement", "help"))
        + f"\nhelp: comfyrack run {m.name} --set <key>=<value>",
        data)


def cmd_run(rack, args) -> out.Result:
    overrides_pairs = args.set or []
    manifest = rack.describe(args.name)
    from ..recipes import parse_set
    overrides = parse_set(overrides_pairs, manifest)
    res = rack.run(args.name, recipe=args.recipe, overrides=overrides,
                   machine=args.machine, out=args.out,
                   skip_preflight=args.skip_preflight,
                   on_progress=_stderr_progress())
    data = {"workflow": args.name,
            "paths": [str(p) for p in res.paths],
            "text": res.text,
            **({"outputs": res.outputs} if res.outputs else {}),
            "provenance": res.record}
    if res.text is not None:
        return out.Result(res.text, data)
    return out.Result("\n".join(str(p) for p in res.paths), data)


def cmd_preflight(rack, args) -> out.Result:
    problems = rack.preflight(args.name, machine=args.machine)
    url = rack.client(args.machine).url
    data = {"workflow": args.name, "machine": url, "count": len(problems),
            "problems": [{"kind": p.kind, "node": p.node_id, "detail": p.detail}
                         for p in problems]}
    if not problems:
        return out.Result(f"preflight: 0 problems for {args.name!r} on {url}", data)
    return out.Result(
        out.table([(p.kind, p.node_id, p.detail) for p in problems],
                  headers=("kind", "node", "detail")) + f"\ncount: {len(problems)}",
        data)


def cmd_nodes(rack, args) -> out.Result:
    names = rack.nodes(args.pattern, machine=args.machine)
    if not names:
        return out.Result(out.empty("node types", f"matching {args.pattern!r}"), names)
    return out.Result("\n".join(names) + f"\ncount: {len(names)}", names)


def cmd_node(rack, args) -> out.Result:
    info = rack.node(args.class_type, machine=args.machine)
    rows = [(k, str(v)) for k, v in info["inputs"].items()]
    return out.Result(
        f"{info['class_type']} -> {', '.join(info['outputs'])}\n"
        + out.table(rows, headers=("input", "type"))
        + f"\nwidgets: {', '.join(info['widgets'])}",
        info)


def cmd_models(rack, args) -> out.Result:
    grouped = rack.models(args.type, machine=args.machine)
    # discover.models returns {type_: []} for an unknown --type, so `--type bogus`
    # rendered a `bogus 0` row: a positive-looking result for a category that does
    # not exist. Drop empty categories and fall through to the definitive zero.
    rows = [(cat, str(len(names)), ", ".join(names[:4]))
            for cat, names in sorted(grouped.items()) if names]
    if not rows:
        where = (f"of type {args.type!r} on this machine" if args.type
                 else "on this machine")
        return out.Result(out.empty("models", where), grouped)
    return out.Result(out.table(rows, headers=("type", "count", "sample")), grouped)


def cmd_queue(rack, args) -> out.Result:
    q = rack.queue(machine=args.machine)
    return out.Result(f"queue: {q['running']} running, {q['pending']} pending", q)


def cmd_stats(rack, args) -> out.Result:
    s = rack.stats(machine=args.machine)
    return out.Result(f"device: {s['device']}\n"
                      f"vram: {s['vram_free_gb']} GB free of {s['vram_total_gb']} GB", s)


def cmd_machines(rack, args) -> out.Result:
    """Configured machines in config order, plus the default URL. A pure config read, no network:
    Superflow's backend calls `machines --json` to learn the fleet even when it is down."""
    machines = [{"name": name, "url": url} for name, url in rack.config.machines.items()]
    default = rack.config.machine_url(None)
    lines = [f"{name}  {url}" for name, url in rack.config.machines.items()]
    if not lines:
        lines.append(out.empty("machines", "in any config layer"))
    lines.append(f"default: {default}")
    return out.Result("\n".join(lines), {"machines": machines, "default": default})


def cmd_interrupt(rack, args) -> out.Result:
    rack.client(args.machine).interrupt()
    return out.Result("interrupt: sent", {"interrupt": "sent"})


def cmd_upload(rack, args) -> out.Result:
    """Upload a local image and print the server-side name.

    The existence check is here, not inside ComfyClient.upload_image: an
    open() on a missing path raises FileNotFoundError, which would escape as an
    `unexpected FileNotFoundError` bug report instead of the structured error
    every other missing-file path in this CLI produces."""
    from pathlib import Path

    path = Path(args.path)
    if not path.is_file():
        raise NotFoundError(
            f"no image at {path}",
            help_text="pass a path to a local image file",
        )
    name = rack.client(args.machine).upload_image(path)
    return out.Result(f"uploaded: {name}", {"path": str(path), "name": name})


def cmd_outputs(rack, args) -> out.Result:
    """List every output file (and text) a prompt produced, optionally downloading.

    Reads the /history entry rather than the /jobs record so a prompt from an
    earlier session -- one this process never waited on -- can be collected.
    Files across all nodes are listed in node order; text outputs (text/string
    keys, e.g. PreviewAny) are carried separately. With --out each file is
    fetched via ComfyClient.view into that directory (created if absent) and
    the local path replaces the subfolder/filename line."""
    from pathlib import Path

    client = rack.client(args.machine)
    entry = client.history(args.prompt_id)
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    texts = []
    for node_id in sorted(entry.get("outputs", {})):
        node_out = entry["outputs"][node_id]
        for key in ("images", "gifs", "audio", "video"):
            for item in node_out.get(key) or []:
                rec = {"node": node_id,
                       "filename": item.get("filename", ""),
                       "subfolder": item.get("subfolder", ""),
                       "type": item.get("type", "output"),
                       "path": None}
                if out_dir is not None:
                    # Basename only: the name comes from the server and must not
                    # escape --out.
                    dest = out_dir / Path(rec["filename"]).name
                    dest.write_bytes(client.view(rec["filename"],
                                                 subfolder=rec["subfolder"],
                                                 type_=rec["type"]))
                    rec["path"] = str(dest)
                files.append(rec)
        for key in ("text", "string"):
            val = node_out.get(key)
            if val is None:
                continue
            for text in (val if isinstance(val, list) else [val]):
                texts.append({"node": node_id, "text": text})

    lines = []
    for rec in files:
        if rec["path"]:
            lines.append(rec["path"])
        else:
            lines.append(f"{rec['subfolder']}/{rec['filename']}"
                         if rec["subfolder"] else rec["filename"])
    return out.Result("\n".join(lines) + f"\ncount: {len(files)}",
                      {"prompt_id": args.prompt_id, "files": files, "text": texts})


def cmd_onboard(rack, args) -> out.Result:
    from pathlib import Path
    from ..onboard import onboard

    _check_segment("family", args.family)
    _check_segment("name", args.name)
    layer_root = args.layer_root
    if layer_root is None:
        if rack.config.project_root is None:
            raise UsageError(
                "no project found and --layer-root was not given",
                help_text="run `comfyrack init` first, or pass --layer-root <dir>",
            )
        layer_root = rack.config.project_root / ".comfyrack" / "registry"
    dest = Path(layer_root) / args.family
    object_info = rack.client(args.machine).object_info()
    json_path, manifest_path = onboard(args.json_path, dest, args.name, object_info,
                                        force=args.force, ui_path=args.ui,
                                        comfy_dir=args.comfy_dir)
    import yaml
    written = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    unknown = _unknowns(written.get("requires") or {})
    overwrite_line = "overwrite: --force applied\n" if args.force else ""
    unknown_line = f"unknown source: {', '.join(unknown)}\n" if unknown else ""
    return out.Result(
        f"workflow: {json_path}\n"
        f"manifest: {manifest_path}\n"
        f"{overwrite_line}{unknown_line}"
        "help:\n"
        f"  edit the manifest's `description` and trim inferred flags\n"
        f"  comfyrack describe {args.name}",
        {"workflow": str(json_path), "manifest": str(manifest_path),
         "force": bool(args.force), "unknown": unknown})


def _unknowns(requires: dict) -> list:
    """Classes and models whose source no lookup found, read from a written record."""
    from ..deps import UNKNOWN
    return ([f"node {c}" for c, v in (requires.get("nodes") or {}).items() if v == UNKNOWN]
            + [f"model {m.get('name')}" for m in requires.get("models") or []
               if m.get("source") == UNKNOWN])


def cmd_deps_backfill(rack, args) -> out.Result:
    """Fill `requires` for registered workflows. Prints a diff per manifest; writes only
    with --write."""
    import difflib
    from pathlib import Path
    from .. import deps

    client = rack.client(args.machine)
    object_info = client.object_info()
    lookups = deps.Lookups()
    ui_dir = Path(args.ui_dir) if args.ui_dir else None
    manifests = rack.list(family=args.family)
    if not manifests:
        where = f"in family {args.family!r}" if args.family else "in any registry layer"
        return out.Result(out.empty("workflows", where), {"workflows": []})
    chunks, rows = [], []
    for m in manifests:
        ui_path = ui_dir / f"{m.name}.json" if ui_dir else None
        ui = None
        if ui_path is not None and ui_path.is_file():
            import json
            ui = json.loads(ui_path.read_text(encoding="utf-8"))
        graph = runner.load_graph(m, client)
        requires, unknown = deps.record(graph, object_info, ui=ui, comfy_dir=args.comfy_dir,
                                        existing=m.requires, lookups=lookups)
        path = m.workflow_path.parent / f"{m.name}.manifest.yaml"
        old = path.read_text(encoding="utf-8")
        new = deps.replace_requires(old, requires)
        diff = "".join(difflib.unified_diff(old.splitlines(keepends=True),
                                            new.splitlines(keepends=True),
                                            f"a/{m.family}/{path.name}",
                                            f"b/{m.family}/{path.name}"))
        if diff and args.write:
            path.write_bytes(new.encode("utf-8"))
        if diff:
            chunks.append(diff)
        rows.append({"workflow": m.name, "manifest": str(path), "changed": bool(diff),
                     "written": bool(diff and args.write), "ui": str(ui_path) if ui else None,
                     "unknown": unknown, "requires": requires})
    changed = sum(r["changed"] for r in rows)
    unknown_lines = [f"  {r['workflow']}: {', '.join(r['unknown'])}" for r in rows if r["unknown"]]
    tail = (f"{changed} of {len(rows)} manifest(s) "
            + ("written" if args.write else "would change") + "\n")
    if unknown_lines:
        tail += "unknown source:\n" + "\n".join(unknown_lines) + "\n"
    if changed and not args.write:
        tail += "help: rerun with --write to save these records"
    return out.Result("".join(chunks) + tail.rstrip("\n"), {"workflows": rows})


_SHARE_HINT = "\nhelp: comfyrack share   # use this ComfyUI from other machines"


def cmd_setup(rack, args) -> out.Result:
    """What a machine lacks to run a workflow, from the manifest's `requires` record.
    --apply hands the missing packs and models to ComfyUI-Manager."""
    from .. import deps

    manifest = rack.describe(args.name)
    client = rack.client(args.machine)
    graph = runner.load_graph(manifest, client)
    plan = deps.plan(manifest, graph, client)
    lines = []
    if plan["packs"]:
        lines.append("packs:")
        for p in plan["packs"]:
            source = (f"registry {p['registry_id']}" if p.get("registry_id")
                      else p.get("git") or "source unknown")
            git = f"  {p['git']}" if p.get("registry_id") and p.get("git") else ""
            lines.append(f"  {p['name']}  {source}{git}  ({', '.join(p['classes'])})")
    if plan["models"]:
        lines.append("models:")
        for m in plan["models"]:
            lines.append(f"  {m['directory']}/{m['name']}  {deps.human_size(m.get('size'))}  "
                         f"{m.get('url') or 'source unknown'}")
    if plan["loras"]:
        lines.append("your LoRAs (not installed by setup; put them in models/loras):")
        lines.extend(f"  {n}" for n in plan["loras"])
    if plan["unplaced_classes"]:
        lines.append("missing classes with no record: " + ", ".join(plan["unplaced_classes"]))
    head = (f"setup {args.name} on {plan['machine']}: {len(plan['packs'])} pack(s), "
            f"{len(plan['models'])} model(s), {len(plan['loras'])} LoRA(s) missing")
    nothing = not (plan["packs"] or plan["models"] or plan["loras"]
                   or plan["unplaced_classes"])
    if nothing:
        return out.Result(f"setup: nothing missing for {args.name!r} on {plan['machine']}" + _SHARE_HINT,
                          {**plan, "applied": None})
    if plan["unplaced_classes"]:
        lines.append(f"help: comfyrack deps backfill --family {manifest.family}  "
                     "# records where those classes come from")
    if not args.apply:
        if plan["packs"] or any(m.get("url") for m in plan["models"]):
            lines.append(f"help: comfyrack setup {args.name} --apply   "
                         "# installs these through ComfyUI-Manager")
        return out.Result(head + "\n" + "\n".join(lines) + _SHARE_HINT,
                          {**plan, "applied": None})

    result = deps.apply(client, plan, on_progress=_stderr_progress())
    lines.append(f"manager: {result['api']} API, {len(result['queued'])} queued")
    for q in result["queued"]:
        status = result["results"].get(q["ui_id"], "done")
        lines.append(f"  {q['kind']} {q['name']}: {status}")
    for r in result["refused"]:
        where = r.get("git") or r.get("url") or ""
        lines.append(f"  refused {r['kind']} {r['name']}: {r.get('reason', '')}"
                     + (f"  install by hand: {where}" if where else ""))
    if result["restart"]:
        lines.append("help: restart ComfyUI to load the new packs, then "
                     f"`comfyrack setup {args.name}` to confirm")
    return out.Result(head + "\n" + "\n".join(lines) + _SHARE_HINT,
                      {**plan, "applied": result})


def cmd_init(rack, args) -> out.Result:
    from pathlib import Path
    from ..onboard import init_project

    root = Path(args.dir or Path.cwd())
    existed = (root / ".comfyrack" / "config.toml").is_file()
    path = init_project(root, *([args.url] if args.url else []))
    note = ("note: config already exists, left unchanged; edit [machines] by hand\n"
            if existed and args.url else "")
    return out.Result(f"config: {path}\n{note}"
                      "help:\n"
                      "  edit [machines] to point at your ComfyUI instances\n"
                      "  comfyrack list\n"
                      "  comfyrack share   # use this ComfyUI from other machines",
                      {"config": str(path)})


def cmd_share(rack, args) -> out.Result:
    from urllib.parse import urlparse
    from .. import share

    port = args.port or urlparse(rack.config.machine_url(args.machine)).port or 8188
    if args.off and args.status:
        raise UsageError("--off and --status cannot be combined",
                         help_text="use one: comfyrack share, share --off, share --status")
    if args.off:
        data = share.disable(port)
        return out.Result(f"sharing off for port {port}", data)
    if args.status:
        data = share.status(port)
        lines = [f"shared: {'yes' if data['shared'] else 'no'}",
                 f"comfyui: {'up' if data['comfyui_up'] else 'not answering'} on 127.0.0.1:{port}",
                 f"url: {data['url']}"]
        if not data["shared"]:
            lines.append("help: comfyrack share")
        return out.Result("\n".join(lines), data)
    data = share.enable(port)
    return out.Result(f"shared: {data['url']}\n"
                      f"help: on another machine, `comfyrack init --url {data['url']}`", data)


def cmd_lex_list(rack, args) -> out.Result:
    entries = rack.lexicon.all()
    data = {t: {"kind": e.kind, "renders": e.renders} for t, e in entries.items()}
    if not entries:
        return out.Result(
            out.empty("lexicon terms", "in this project")
            + "\nhelp: comfyrack lex sync   # derive entries from your LoRAs", data)
    rows = [(t, e.kind, ", ".join(sorted(e.renders))) for t, e in sorted(entries.items())]
    return out.Result(
        out.table(rows, headers=("term", "kind", "lineages"))
        + f"\ncount: {len(rows)}"
        + "\nhelp: comfyrack lex show <term>", data)


def cmd_lex_show(rack, args) -> out.Result:
    entry = rack.lexicon.get(args.term)
    lines = [f"{entry.term} [{entry.kind}]"]
    rendered = {}
    for lineage in sorted(entry.renders):
        rendered[lineage] = rack.lexicon.render(entry.term, lineage)
        lines.append(f"  {lineage}: {rendered[lineage]}")
    return out.Result("\n".join(lines),
                      {"term": entry.term, "kind": entry.kind,
                       "renders": entry.renders, "rendered": rendered})


def cmd_lex_sync(rack, args) -> out.Result:
    loras_dir = args.loras or rack.config.path("loras")
    if loras_dir is None:
        raise UsageError(
            "no LoRA directory configured",
            help_text='add [paths] loras = "..." to .comfyrack/config.toml, '
                      "or pass --loras <dir>",
        )
    results = rack.lexicon.sync(loras_dir, overwrite=args.overwrite)
    data = [{"term": t, "action": a} for t, a in results]
    if not results:
        return out.Result(out.empty("LoRAs", f"under {loras_dir}"), data)
    counts = {}
    for _term, action in results:
        counts[action] = counts.get(action, 0) + 1
    body = out.table(sorted(results), headers=("term", "action"))
    summary = ", ".join(f"{n} {a}" for a, n in sorted(counts.items()))
    return out.Result(
        f"{body}\ncount: {len(results)} ({summary})\nhelp: comfyrack lex show <term>",
        data)


def cmd_lex_add(rack, args) -> out.Result:
    from ..lexicon import Entry

    block = {}
    if args.trigger:
        block["trigger"] = args.trigger
    if args.tags:
        block["tags"] = args.tags
    if args.text:
        block["text"] = args.text
    if not block:
        raise UsageError(
            "nothing to store for this term",
            help_text="pass --tags (booru lineages) or --text (natural lineages)",
        )
    existing = rack.lexicon.all().get(args.term.lower())
    renders = dict(existing.renders) if existing else {}
    renders[args.lineage] = block
    # --kind is argparse.SUPPRESS by default (see main.py), so an explicit
    # `--kind` is distinguishable from the caller not passing it at all. Only
    # fall back to "term" for a brand-new entry; an existing entry keeps its
    # own kind unless this invocation explicitly overrides it.
    explicit_kind = getattr(args, "kind", None)
    if explicit_kind is not None:
        kind = explicit_kind
    elif existing:
        kind = existing.kind
    else:
        kind = "term"
    path = rack.lexicon.save(Entry(term=args.term.lower(), kind=kind,
                                   renders=renders))
    return out.Result(
        f"lexicon: {path}\nhelp: comfyrack lex show {args.term.lower()}",
        {"term": args.term.lower(), "kind": kind, "lineage": args.lineage,
         "path": str(path)})


def cmd_prompt_preview(rack, args) -> out.Result:
    import sys

    text, warnings = rack.assemble_prompt(
        args.lineage, args.dialect, subject=args.subject, location=args.location,
        scene=getattr(args, "scene", None),
        action=args.action or "", style=args.style or "",
        allow_unknown=args.allow_unknown, seed=args.seed)
    # Warnings go to stderr: stdout carries the prompt and nothing else, so a caller
    # can pipe it straight into a --set value.
    for w in warnings:
        print(w, file=sys.stderr)
    return out.Result(text, {"prompt": text, "warnings": warnings})


def cmd_char_list(rack, args) -> out.Result:
    chars = rack.profile.characters()
    data = {n: c.raw for n, c in chars.items()}
    if not chars:
        return out.Result(
            out.empty("characters", "in this project")
            + '\nhelp: add [paths] characters = "..." to .comfyrack/config.toml', data)
    rows = [(n, c.trigger, c.id_lora, f"{c.id_strength}") for n, c in sorted(chars.items())]
    return out.Result(
        out.table(rows, headers=("character", "trigger", "id lora", "strength"))
        + f"\ncount: {len(rows)}\nhelp: comfyrack char show <name>", data)


def cmd_char_show(rack, args) -> out.Result:
    c = rack.profile.character(args.name)
    rows = [
        ("trigger", c.trigger),
        ("base model family", c.base_model_family),
        ("id lora", f"{c.id_lora} @ {c.id_strength}"),
        ("style lora", f"{c.style_lora} @ {c.style_strength}"),
        ("sampler", ", ".join(f"{k}={v}" for k, v in sorted(c.sampler.items()))),
    ]
    if c.age_bands:
        rows.append(("age bands", ", ".join(f"{k}={v}" for k, v in sorted(
            c.age_bands.items()) if not isinstance(v, dict))))
    return out.Result(f"{c.name}\n" + out.table(rows), c.raw)


def cmd_subject_list(rack, args) -> out.Result:
    subjs = rack.profile.subjects()
    data = {n: s.raw for n, s in subjs.items()}
    if not subjs:
        return out.Result(
            out.empty("subjects", "in this project")
            + '\nhelp: add [paths] subjects = "..." to .comfyrack/config.toml', data)
    rows = [(n, s.trigger, s.id_lora, f"{s.id_strength}") for n, s in sorted(subjs.items())]
    return out.Result(
        out.table(rows, headers=("subject", "trigger", "id lora", "strength"))
        + f"\ncount: {len(rows)}\nhelp: comfyrack subject show <name>", data)


def cmd_subject_show(rack, args) -> out.Result:
    s = rack.profile.subject(args.name)
    rows = [
        ("trigger", s.trigger),
        ("base model family", s.base_model_family),
        ("id lora", f"{s.id_lora} @ {s.id_strength}"),
        ("style lora", f"{s.style_lora} @ {s.style_strength}"),
        ("sampler", ", ".join(f"{k}={v}" for k, v in sorted(s.sampler.items()))),
    ]
    if s.age_bands:
        rows.append(("age bands", ", ".join(f"{k}={v}" for k, v in sorted(
            s.age_bands.items()) if not isinstance(v, dict))))
    return out.Result(f"{s.name}\n" + out.table(rows), s.raw)


def cmd_loc_list(rack, args) -> out.Result:
    locs = rack.profile.locations()
    data = {k: {"plate": str(v.plate_path) if v.plate_path else None,
                "description": v.description} for k, v in locs.items()}
    if not locs:
        return out.Result(
            out.empty("locations", "in this project")
            + '\nhelp: add [paths] locations = "..." to .comfyrack/config.toml', data)
    rows = [(k, "plate" if v.plate_path else "no plate", v.description)
            for k, v in sorted(locs.items())]
    return out.Result(
        out.table(rows, headers=("location", "status", "description"))
        + f"\ncount: {len(rows)}\nhelp: comfyrack loc show <key>", data)


def cmd_loc_show(rack, args) -> out.Result:
    loc = rack.profile.location(args.key)
    data = {"key": loc.key,
            "plate": str(loc.plate_path) if loc.plate_path else None,
            "description": loc.description}
    return out.Result(
        f"{loc.key}\n"
        + out.table([("plate", str(loc.plate_path) if loc.plate_path else "(none)"),
                     ("description", loc.description or "(none)")]), data)


def cmd_scene_list(rack, args) -> out.Result:
    scenes = rack.profile.scenes()
    data = {k: {"plate": str(v.plate_path) if v.plate_path else None,
                "description": v.description} for k, v in scenes.items()}
    if not scenes:
        return out.Result(
            out.empty("scenes", "in this project")
            + '\nhelp: add [paths] scenes = "..." to .comfyrack/config.toml', data)
    rows = [(k, "plate" if v.plate_path else "no plate", v.description)
            for k, v in sorted(scenes.items())]
    return out.Result(
        out.table(rows, headers=("scene", "status", "description"))
        + f"\ncount: {len(rows)}\nhelp: comfyrack scene show <key>", data)


def cmd_scene_show(rack, args) -> out.Result:
    sc = rack.profile.scene(args.key)
    data = {"key": sc.key,
            "plate": str(sc.plate_path) if sc.plate_path else None,
            "description": sc.description}
    return out.Result(
        f"{sc.key}\n"
        + out.table([("plate", str(sc.plate_path) if sc.plate_path else "(none)"),
                     ("description", sc.description or "(none)")]), data)


def cmd_recipe_list(rack, args) -> out.Result:
    names = rack.profile.recipes()
    if not names:
        return out.Result(
            out.empty("recipes", "in this project")
            + '\nhelp: add [paths] recipes = "..." to .comfyrack/config.toml', names)
    return out.Result(
        "\n".join(names) + f"\ncount: {len(names)}"
        + "\nhelp: comfyrack recipe show <name>", names)


def cmd_recipe_show(rack, args) -> out.Result:
    values = rack.profile.recipe(args.name)
    if not values:
        return out.Result(out.empty("values", f"in recipe {args.name!r}"), values)
    return out.Result(
        out.table([(k, str(v)) for k, v in sorted(values.items())],
                  headers=("key", "value")), values)


def cmd_batch(rack, args) -> out.Result:
    from pathlib import Path

    from ..batch import build_jobs, compare_sheet, run_batch
    from ..shots import load_shots, validate

    path = Path(args.shots)
    if not path.is_file():
        raise NotFoundError(f"no shot list at {path}",
                            help_text="pass a path to a shots YAML file")
    shot_list = load_shots(path)

    problems = validate(shot_list)
    if args.validate:
        if problems:
            raise UsageError(
                f"{len(problems)} problem(s) in {path.name}:\n  "
                + "\n  ".join(problems),
                help_text="fix the shot list, then re-run --validate")
        return out.Result(
            f"{path.name}: {len(shot_list.shots)} shots, 0 problems",
            {"shots": len(shot_list.shots), "problems": [],
             "names": [s.name for s in shot_list.shots]})
    if problems:
        raise UsageError(
            f"{len(problems)} problem(s) in {path.name}:\n  " + "\n  ".join(problems),
            help_text=f"comfyrack batch {args.shots} --validate")

    jobs = build_jobs(shot_list, rack.profile, only=args.shot, lexicon=rack.lexicon)
    if not jobs:
        return out.Result(
            out.empty("shots", f"matching {args.shot!r} in {path.name}"), [])

    if args.dry_run:
        rows = [(j.shot_name, ", ".join(j.requirements.get("loras", []) or ["(none)"]))
                for j in jobs]
        return out.Result(
            out.table(rows, headers=("shot", "required loras"))
            + f"\ncount: {len(rows)}"
            + f"\nhelp: comfyrack batch {args.shots} --workflow <name>",
            [{"shot": j.shot_name, "requirements": j.requirements} for j in jobs])

    if not args.workflow:
        raise UsageError("--workflow is required to render",
                         help_text="comfyrack list --family anima")

    # `--machine` is honored, not ignored. batch is the highest-GPU-time command
    # in the package, so dropping a user's explicit machine choice for a real
    # render batch -- silently, with no signal -- is the worst place to do it.
    # One named machine is just the degenerate case of `--machines`: it becomes
    # the only entry in the routing map, so the batch runs there or reports
    # `unroutable` per shot. Passing both is a contradiction (which one
    # constrains routing?) and is refused rather than resolved by precedence.
    machine = getattr(args, "machine", None)
    if machine and args.machines:
        raise UsageError(
            "--machine and --machines both given; they set the same thing",
            help_text="use --machine <name> to pin one machine, or --machines "
                      "<a,b> to fan out across several",
        )
    if args.machines:
        machines = args.machines.split(",")
    elif machine:
        machines = [machine]
    else:
        machines = None
    outcome = run_batch(rack, shot_list, args.workflow, machines=machines,
                        only=args.shot, recipe=args.recipe,
                        on_progress=_stderr_progress())
    # run_batch() returns {"results": [...], "index_errors": {...}}, not a bare
    # list -- index_errors is data the caller (here) decides how to surface,
    # not a warnings.warn() side channel. See batch.run_batch's docstring.
    results = outcome["results"]
    index_errors = outcome["index_errors"]

    rows = [(r["shot"], r["status"], r.get("machine") or "-", r.get("detail", ""))
            for r in results]
    body = out.table(rows, headers=("shot", "status", "machine", "detail"))
    ok = sum(1 for r in results if r["status"] == "ok")

    index_errors_line = ""
    if index_errors:
        names = ", ".join(f"{name} ({msg})" for name, msg in sorted(index_errors.items()))
        index_errors_line = (f"\nindex_errors: {len(index_errors)} machine(s) could "
                             f"not be indexed and were excluded from routing: {names}")

    # A batch where NOTHING rendered is an error, not a zero-exit no-op. Only the
    # `count: 0/N` line used to say so, which a scripted caller checking the exit
    # code alone -- the normal way to chain a render into an upscale or a judge
    # pass -- could not see. Partial success keeps exit 0: 8 of 9 shots rendering
    # is a real result the caller should go collect, and re-running the one that
    # failed is a different action from re-running the batch.
    if results and ok == 0:
        raise ComfyrackError(
            f"batch failed: 0 of {len(results)} shot(s) rendered\n"
            f"{body}{index_errors_line}",
            help_text=f"comfyrack batch {args.shots} --dry-run  # what each shot "
                      f"needs, and which machine would have to hold it",
        )

    sheet_line, sheet_path = "", None
    paths = [p for r in results for p in (r.get("result") or {}).get("paths", [])]
    if paths and args.sheet:
        sheet = compare_sheet(paths, args.sheet)
        sheet_path = str(sheet) if sheet else None
        sheet_line = f"\nsheet: {sheet}" if sheet else \
            "\nsheet: skipped (Pillow not installed)"

    return out.Result(
        f"{body}\ncount: {ok}/{len(results)} ok{index_errors_line}{sheet_line}"
        + "\nhelp: comfyrack judge <sheet> --against " + str(args.shots),
        {"results": results, "ok": ok, "total": len(results), "sheet": sheet_path,
         "index_errors": index_errors})


def cmd_judge(rack, args) -> out.Result:
    from pathlib import Path

    from .. import judge as _judge
    from ..shots import load_shots

    shot_list = load_shots(args.against)
    if args.rubric_only:
        # Deliberately before the sheet check: --rubric-only never reads the sheet,
        # so requiring one would make the cheap dry path depend on a render.
        rubric = _judge.build_rubric(shot_list, only=args.shot)
        return out.Result(rubric, {"rubric": rubric})
    sheet = Path(args.sheet)
    if not sheet.is_file():
        raise NotFoundError(f"no sheet at {sheet}",
                            help_text="pass the compare sheet produced by `comfyrack batch`")
    verdict = _judge.run(rack, sheet, shot_list, workflow=args.workflow,
                         machine=args.machine, only=args.shot)
    return out.Result(verdict, {"sheet": str(sheet), "verdict": verdict})


def _m(m, full: bool = False) -> dict:
    d = {"name": m.name, "family": m.family, "layer": m.layer,
         "description": m.description, "output": m.output_type, "mode": m.mode}
    if full:
        d["flags"] = {k: {"param": f.param, "node": f.node, "type": f.type,
                          "required": f.required, "default": f.default, "help": f.help,
                          **{key: getattr(f, key) for key in FORM_KEYS
                             if getattr(f, key) is not None}}
                      for k, f in m.flags.items()}
        if m.outputs:
            d["outputs"] = m.outputs
        if m.requires:
            d["requires"] = m.requires
    return d
