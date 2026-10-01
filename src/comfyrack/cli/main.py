"""Argument parsing and dispatch.

argparse's default behaviour on an unknown flag is exit 2 with usage on stderr.
That violates the contract: errors go to STDOUT, name the flag, and list the valid
flags inline so the agent self-corrects in one turn rather than making a follow-up
--help call. _Parser overrides error() to raise UsageError instead."""
import argparse
import os
import sys
import traceback

from .. import Rack, out
from ..errors import ComfyrackError, UsageError
from . import commands


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("comfyrack")
    except PackageNotFoundError:
        return "unknown"


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        opts = [s for a in self._actions for s in a.option_strings]
        help_text = f"valid flags for `{self.prog}`: {', '.join(sorted(set(opts)))}"
        raise UsageError(message, help_text=help_text)


def build_parser() -> _Parser:
    p = _Parser(prog="comfyrack", add_help=True,
                description="Find, check and run ComfyUI workflows by name. "
                            "Run `comfyrack <command> --help` for one command.")
    p.add_argument("--version", action="version", version=f"comfyrack {_version()}")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--machine", default=None, help="target machine name from config")
    sub = p.add_subparsers(dest="cmd")
    p.subparsers_by_name = {}

    def add(name, fn, **kw):
        sp = sub.add_parser(name, description=kw.get("help"), **kw)
        # SUPPRESS, not a default: argparse copies EVERY attribute of a subparser's
        # namespace onto the parent's, so a plain default here silently overwrites
        # `comfyrack --json list` back to False. SUPPRESS leaves the attribute unset
        # when the flag is absent, so the top-level value survives.
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="print JSON instead of a table")
        sp.add_argument("--machine", default=argparse.SUPPRESS,
                        help="machine name from config (default: the default machine)")
        sp.set_defaults(func=fn)
        p.subparsers_by_name[name] = sp
        return sp

    sp = add("list", commands.cmd_list,
             help="list registered workflows, grouped by family")
    sp.add_argument("--family", default=None, help="only this family (for example anima)")
    sp.add_argument("--layer", default=None, help="only this layer: builtin, user or project")
    sp.add_argument("--mode", default=None, help="only this mode (for example t2i or i2v)")
    sp.add_argument("--all", action="store_true", help="list every workflow, not a summary per family")

    sp = add("find", commands.cmd_find,
             help="search workflows by name and description")
    sp.add_argument("query", help="words to look for in names and descriptions")
    sp.add_argument("--mode", default=None, help="only this mode (for example t2i or i2v)")

    sp = add("describe", commands.cmd_describe,
             help="show a workflow's flags, models and prompt dialect")
    sp.add_argument("name", help="registered workflow name")

    sp = add("run", commands.cmd_run,
             help="run a workflow and print the path of each output file")
    sp.add_argument("name", help="registered workflow name")
    sp.add_argument("--recipe", default=None, help="recipe whose locked settings to apply")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="set one flag; repeat for more (see `describe`)")
    sp.add_argument("--out", default=None, help="folder to download the outputs into")
    sp.add_argument("--skip-preflight", action="store_true", dest="skip_preflight",
                    help="submit without checking the machine first")

    sp = add("preflight", commands.cmd_preflight,
             help="check a workflow against a machine without running it")
    sp.add_argument("name", help="registered workflow name")

    sp = add("nodes", commands.cmd_nodes,
             help="list the node types a machine has")
    sp.add_argument("pattern", nargs="?", default=None,
                    help="only node types whose name contains this text")

    sp = add("node", commands.cmd_node,
             help="show the inputs and outputs of one node type")
    sp.add_argument("class_type", help="node type name (see `nodes`)")

    sp = add("models", commands.cmd_models,
             help="list the model files a machine has, by type")
    sp.add_argument("--type", default=None,
                    help="only this model folder (for example checkpoints or loras)")

    sp = add("onboard", commands.cmd_onboard,
             help="register an API-format workflow JSON in the registry")
    sp.add_argument("json_path", help="workflow saved in API format")
    sp.add_argument("--family", required=True, help="family folder to register it under")
    sp.add_argument("--name", required=True, help="name for the new workflow")
    sp.add_argument("--layer-root", default=None, dest="layer_root",
                    help="registry folder to write into (default: the project registry)")
    sp.add_argument("--force", action="store_true", help="overwrite a workflow with the same name")
    sp.add_argument("--ui", default=None, metavar="UI_JSON",
                    help="the same workflow's UI-format export; adds model URLs and pack IDs "
                         "to `requires`")
    sp.add_argument("--comfy-dir", default=None, dest="comfy_dir", metavar="PATH",
                    help="ComfyUI folder on this machine; reads each pack's repo from "
                         "custom_nodes")

    sp = add("init", commands.cmd_init,
             help="write .comfyrack/config.toml for this project")
    sp.add_argument("--dir", default=None, help="project folder (default: the current folder)")
    sp.add_argument("--url", default=None, metavar="URL",
                    help="ComfyUI address to write as the default machine "
                         "(default http://127.0.0.1:8188)")

    sp = add("setup", commands.cmd_setup,
             help="list the packs and models a machine lacks for a workflow")
    sp.add_argument("name", help="registered workflow name")
    sp.add_argument("--apply", action="store_true",
                    help="install the listed packs and models through ComfyUI-Manager")

    sp = add("share", commands.cmd_share,
             help="share this machine's ComfyUI on your Tailscale tailnet")
    sp.add_argument("--off", action="store_true", help="stop sharing")
    sp.add_argument("--status", action="store_true", help="show whether it is shared and the URL")
    sp.add_argument("--port", type=int, default=None,
                    help="ComfyUI port (default: from the machine's config, else 8188)")

    # A nested sub-subparser must NOT redeclare --json/--machine with a plain
    # default: argparse copies every attribute of a subparser's namespace onto the
    # parent's, so `comfyrack --json lex list` would be silently reset to False.
    # This helper redeclares them with SUPPRESS, matching add().
    def add_nested(group, name, fn, help=None):
        np = group.add_parser(name, help=help, description=help)
        np.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="print JSON instead of a table")
        np.add_argument("--machine", default=argparse.SUPPRESS,
                        help="machine name from config (default: the default machine)")
        np.set_defaults(func=fn)
        return np

    def needs_sub(group_name, verbs):
        """Default for a group invoked bare. Its children carry required flags, so
        falling through to a child's handler raises AttributeError on a missing
        arg and surfaces as an `unexpected AttributeError` bug report. Fail as a
        usage error naming the verbs instead."""
        def _fn(rack, args):
            raise UsageError(
                f"`comfyrack {group_name}` needs a subcommand",
                help_text=f"try: {', '.join(f'comfyrack {group_name} {v}' for v in verbs)}",
            )
        return _fn

    sp = add("lex", needs_sub("lex", ["list", "show", "sync", "add"]),
             help="the lexicon: project terms and the words each model family knows for them")
    lex_sub = sp.add_subparsers(dest="lex_cmd")

    add_nested(lex_sub, "list", commands.cmd_lex_list, help="list every term")

    lp = add_nested(lex_sub, "show", commands.cmd_lex_show, help="show one term")
    lp.add_argument("term", help="the term to show")

    lp = add_nested(lex_sub, "sync", commands.cmd_lex_sync,
                    help="add a term for each LoRA, from its trained words")
    lp.add_argument("--loras", default=None,
                    help="LoRA folder (default: [paths] loras in config)")
    lp.add_argument("--overwrite", action="store_true", help="replace terms that already exist")

    lp = add_nested(lex_sub, "add", commands.cmd_lex_add,
                    help="add a term, or add a model family's wording to one")
    lp.add_argument("term", help="the term")
    # SUPPRESS, not default="term": a plain default makes "user passed --kind"
    # and "user didn't" indistinguishable, so cmd_lex_add could not tell an
    # explicit override from the argparse default and would silently downgrade
    # an existing entry's kind on every lineage-only `lex add`. Leaving the
    # attribute unset when absent lets cmd_lex_add fall back to the existing
    # entry's kind, and only default to "term" for a brand-new term.
    lp.add_argument("--kind", default=argparse.SUPPRESS,
                    help="character, location, prop or term (default: term for a new entry)")
    lp.add_argument("--lineage", required=True, help="model lineage this wording is for")
    lp.add_argument("--trigger", default=None, help="trigger word the LoRA was trained on")
    lp.add_argument("--tags", default=None, help="tag text for booru lineages")
    lp.add_argument("--text", default=None, help="sentence text for natural-language lineages")

    sp = add("deps", needs_sub("deps", ["backfill"]),
             help="manage the models and packs each workflow requires")
    deps_sub = sp.add_subparsers(dest="deps_cmd")
    dp = add_nested(deps_sub, "backfill", commands.cmd_deps_backfill,
                    help="fill in each workflow's `requires` record from its graph")
    dp.add_argument("--family", default=None, help="only this family's workflows")
    dp.add_argument("--ui-dir", default=None, dest="ui_dir", metavar="DIR",
                    help="folder of UI-format exports, matched to workflows by file stem")
    dp.add_argument("--comfy-dir", default=None, dest="comfy_dir", metavar="PATH",
                    help="ComfyUI folder on the target machine; reads each pack's repo "
                         "from custom_nodes")
    dp.add_argument("--write", action="store_true",
                    help="save the records; without it, only print the diff")

    sp = add("prompt", needs_sub("prompt", ["preview"]),
             help="build a prompt from lexicon terms")
    prompt_sub = sp.add_subparsers(dest="prompt_cmd")
    pp = add_nested(prompt_sub, "preview", commands.cmd_prompt_preview,
                    help="print the prompt for a subject, location and action")
    pp.add_argument("--lineage", required=True, help="model lineage to write for")
    pp.add_argument("--dialect", default="booru", choices=["booru", "natural"],
                    help="booru tags or natural sentences (default booru)")
    pp.add_argument("--subject", default=None, help="subject name from the profile")
    pp.add_argument("--location", default=None, help="location key from the profile")
    pp.add_argument("--scene", default=None, help="scene key from the profile")
    pp.add_argument("--action", default=None, help="what the subject is doing")
    pp.add_argument("--style", default=None, help="style text to append")
    pp.add_argument("--seed", type=int, default=None, help="seed for wildcard choices")
    pp.add_argument("--allow-unknown", action="store_true", dest="allow_unknown",
                    help="allow terms that are not in the lexicon")

    # Reuses Task 7's add_nested, so --json/--machine stay SUPPRESS here too. A plain
    # `--machine", default=None` on these children would silently override
    # `comfyrack --machine NAME char list` back to None.
    def add_group(name, noun, pairs):
        sp = add(name, needs_sub(name, [p[0] for p in pairs]),
                 help=f"list or show {noun} in the project profile")
        group = sp.add_subparsers(dest=f"{name}_cmd")
        for sub_name, fn, positional in pairs:
            gp = add_nested(group, sub_name, fn, help=f"{sub_name} {noun}")
            if positional:
                gp.add_argument(positional, help="name or key to show")
        return sp

    add_group("char", "characters", [("list", commands.cmd_char_list, None),
                       ("show", commands.cmd_char_show, "name")])
    add_group("subject", "subjects", [("list", commands.cmd_subject_list, None),
                          ("show", commands.cmd_subject_show, "name")])
    add_group("loc", "locations", [("list", commands.cmd_loc_list, None),
                      ("show", commands.cmd_loc_show, "key")])
    add_group("scene", "scenes", [("list", commands.cmd_scene_list, None),
                        ("show", commands.cmd_scene_show, "key")])
    add_group("recipe", "recipes", [("list", commands.cmd_recipe_list, None),
                         ("show", commands.cmd_recipe_show, "name")])

    sp = add("batch", commands.cmd_batch,
             help="run every shot in a shot list and build a compare sheet")
    sp.add_argument("shots", help="shot list YAML file")
    sp.add_argument("--workflow", default=None, help="workflow to run each shot with")
    sp.add_argument("--recipe", default=None, help="recipe whose locked settings to apply")
    sp.add_argument("--shot", default=None, help="run only this shot")
    sp.add_argument("--machines", default=None,
                    help="comma-separated machine names to fan out across")
    sp.add_argument("--sheet", default=None, help="where to write the compare sheet")
    sp.add_argument("--validate", action="store_true",
                    help="check the shot list and stop; renders nothing")
    sp.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="show what would run and where; renders nothing")

    sp = add("judge", commands.cmd_judge,
             help="have a vision model check a compare sheet against its shot list")
    sp.add_argument("sheet", help="compare sheet image from `batch`")
    sp.add_argument("--against", required=True, help="the shot list the sheet was made from")
    sp.add_argument("--workflow", default="vl-judge", help="vision workflow to judge with")
    sp.add_argument("--shot", default=None, help="judge only this shot")
    sp.add_argument("--rubric-only", action="store_true", dest="rubric_only",
                    help="print the rubric and stop; does not read the sheet")

    add("queue", commands.cmd_queue, help="show the jobs running and waiting on a machine")
    add("stats", commands.cmd_stats, help="show a machine's GPU and memory use")
    add("interrupt", commands.cmd_interrupt, help="stop the job a machine is running")
    add("machines", commands.cmd_machines,
        help="list the machines in config (no network call)")

    sp = add("upload", commands.cmd_upload,
             help="upload a local image to a machine and print its server-side name")
    sp.add_argument("path", help="local image file")

    sp = add("outputs", commands.cmd_outputs,
             help="list the files a finished job produced, and optionally download them")
    sp.add_argument("prompt_id", help="job ID printed by `run`")
    sp.add_argument("--out", default=None, help="folder to download the files into")
    return p


def main(argv=None) -> int:
    # Manifests and node titles carry non-ASCII text (emoji in some node titles). A Windows pipe or console
    # defaults to cp1252 and would raise UnicodeEncodeError on print, so always write UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args, extra = parser.parse_known_args(argv)
        if extra:
            # A leftover unrecognized flag bubbles up to the TOP-level parser's
            # parse_args() in stock argparse, which would report it against the
            # wrong parser's flags (see task-14 report for the mechanism). Route
            # it to the subparser that actually owns this invocation instead, so
            # the help text lists that subcommand's own flags.
            target = parser.subparsers_by_name.get(getattr(args, "cmd", None), parser)
            target.error(f"unrecognized arguments: {' '.join(extra)}")
        as_json = getattr(args, "json", False)
        rack = Rack(machine=args.machine)
        fn = args.func if getattr(args, "cmd", None) else commands.cmd_home
        # ONE rendering site. Commands return an out.Result carrying both shapes, so
        # --json is structural: a subcommand cannot accept the flag and ignore it,
        # which is exactly what eight of fifteen used to do.
        print(out.render(fn(rack, args), as_json))
        return 0
    except ComfyrackError as exc:
        print(out.error(exc))
        return exc.exit_code
    except BrokenPipeError:
        return 0
    except Exception as exc:  # noqa: BLE001 -- last resort, ordered after the specific handlers
        # Nothing may reach stderr as a traceback: stderr is progress and
        # diagnostics only, and an agent parsing stdout would see nothing at all.
        # Anything landing here is a bug -- so say so, and still obey the output
        # contract (stdout, structured, a `help:` line, exit 1).
        print(out.error(ComfyrackError(
            f"unexpected {type(exc).__name__}: {exc}",
            help_text="this is a comfyrack bug; re-run with COMFYRACK_TRACEBACK=1 "
                      "for the traceback and report it",
        )))
        if os.environ.get("COMFYRACK_TRACEBACK"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
