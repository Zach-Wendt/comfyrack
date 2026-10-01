---
name: comfyrack
description: Look up or run a registered ComfyUI workflow via the comfyrack CLI instead of reading a workflow JSON or writing a Python graph-builder. Use whenever a render, edit, upscale, video, or VL-judge task needs an already-registered workflow, or for any direct ComfyUI server task (node types, models, queue, upload, outputs, interrupt) in place of a ComfyUI MCP server.
---

# comfyrack

Use `comfyrack` to find and run every registered ComfyUI workflow. Do not read a
workflow's raw JSON to use it. Read the JSON only to onboard a new workflow.

## Lookup path

1. `comfyrack` — bare, shows every family and its workflow count. Cheap.
2. `comfyrack list --family <f>` — expand one family.
3. `comfyrack find <query>` — search every family, including out-of-scope ones.
4. `comfyrack describe <name>` — the flag table for one workflow, plus its prompt
   dialect. Do this before writing any prompt. With `--json`, each flag also carries
   `choices`, `min`/`max`, `min_length`/`max_length` when known, read live from the
   target ComfyUI's `/object_info` (`--machine`, else `COMFY_URL`, else the default).
   `"live": false` means nothing answered and only manifest data is shown. A manifest
   `requires:` block is included too: `nodes` maps each class to `core` or a pack folder,
   `custom_node_packs` gives each pack's git URL and registry ID, `models` gives each
   file's folder and download URL.

## Running

```
comfyrack run <name> --recipe <r> --set key=value [--machine m] [--out path]
```

A recipe holds locked settings. `--set` overrides single values. Each flag takes
the first value found in this order: `--set`, recipe, manifest default, the value
saved in the workflow.

A manifest with an `outputs:` list (`- {title: <node title>, kind: image|text}`)
makes `run --json` add `"outputs": {<title>: {"paths": [...]} | {"text": "..."}}`.
`paths` and `text` still hold the `output_node_title` result.

`comfyrack preflight <name> --machine <m>` validates node classes, model
availability, and LoRA lineage against a machine before spending GPU time. `run`
does this automatically.

When preflight reports a missing node class or model, run `comfyrack setup <name>
--machine <m>`. It lists the missing packs (registry ID, git URL) and models (URL, size).
`--apply` installs them through ComfyUI-Manager and changes that machine's ComfyUI.
Ask the machine's owner before `--apply`. After packs install, ComfyUI needs a
restart; then run `setup` again to confirm.

## Server tasks

Every command takes `--json` and `--machine <m>`. These replace a ComfyUI MCP server.

| Task | Command |
|---|---|
| node types, one node's inputs | `comfyrack nodes [pattern]`, `comfyrack node <ClassType>` |
| models, embeddings | `comfyrack models [--type loras\|embeddings]` |
| queue, GPU and VRAM | `comfyrack queue`, `comfyrack stats` |
| upload an image | `comfyrack upload <path>` (prints the server-side name) |
| files and text a prompt produced | `comfyrack outputs <prompt_id> [--out DIR]` |
| stop the running job | `comfyrack interrupt` |

To run a workflow JSON that is not registered, onboard it first (`comfyrack onboard
<api.json> --family <f> --name <n> [--ui <ui.json>]`), then `describe` and `run --set`.
That is the path for load-a-graph, set-a-param, run-it work. `--ui` takes the same
workflow's UI export, which carries model URLs. Onboard prints any class or model whose
source it could not find; add those to `requires` by hand.

`comfyrack deps backfill --family <f> [--ui-dir DIR] [--comfy-dir PATH]` fills
`requires` for registered workflows. It prints a diff; `--write` saves it.

## Prompt dialect

`describe` prints the family's dialect. Illustrious-lineage families want
comma-separated Danbooru tags; Flux/Krea/LTX families want natural language. These
are not interchangeable — a prompt written in the wrong dialect renders badly
without erroring.

## When not to use

A workflow needing real orchestration on top of it — sequential multi-pass
compositing, conditional structural node surgery — still needs Python. Import the
library instead:

```python
from comfyrack import Rack, runner
rack = Rack()
manifest = rack.describe("multiedit-composite")
graph = runner.load_graph(manifest, rack.client())
# ... structural surgery ...
runner.submit(rack.client(), graph, manifest)
```

comfyrack builds the graph for one run. It does not sequence several runs. Use Python for that.
