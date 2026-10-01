# comfyrack

comfyrack is a catalog and CLI for ComfyUI workflows. Agents and apps find, check, and run a
workflow by name and read `--json` output, instead of editing graph JSON.

Status: alpha. The CLI and manifest format can still change.

## Demo

Three workflows in a chain: make an image, edit it, then turn it into a video. Each step uses the
output of the step before.

![comfyrack finds a workflow, reads its flags, and runs three workflows](docs/demo/terminal.gif)

**32 GB card (RTX 5090)**

| 1. `t2i` | 2. `edit` | 3. `i2v-t2v-basic` (fp8) |
|---|---|---|
| ![a lighthouse at dusk](docs/demo/1-lighthouse.jpg) | ![the same lighthouse in snow](docs/demo/2-snow.jpg) | ![the lighthouse in snow, its beam turning](docs/demo/3-video.webp) |
| "a lighthouse on a rocky coast at dusk" | "make it a snowy winter evening" | the beam turns and the snow falls ([mp4 with sound](docs/demo/3-video.mp4)) |

**12 GB card (RTX 3060)**

| 1. `t2i` | 2. `edit` | 3. `i2v-t2v-basic-gguf` |
|---|---|---|
| ![a cup of coffee by a rainy window](docs/demo/low-1-coffee.jpg) | ![the same cup at night by candlelight](docs/demo/low-2-candle.jpg) | ![steam rising from the cup, the candle flickering](docs/demo/low-3-video.webp) |
| "a steaming cup of coffee beside a rainy window" | "make it night, lit by a single candle" | steam curls, the flame flickers, rain runs down the glass ([mp4 with sound](docs/demo/low-3-video.mp4)) |

### What runs on which card

Measured with the settings above (images 1024x1024, video 768x768 and 4 seconds).

| Step | 12 GB (RTX 3060, 64 GB RAM) | 16-24 GB | 32 GB (RTX 5090) |
|---|---|---|---|
| `t2i` | 60 s warm (182 s with the first model load) | untested | 18 s |
| `edit` | 53 s | untested | 27 s |
| video | `i2v-t2v-basic-gguf`: 283 s. It used all 12 GB of VRAM and almost all 64 GB of RAM. | untested | `i2v-t2v-basic` (fp8): 148 s |

On a 12 GB card, use the `-gguf` video workflow. ComfyUI moves what does not fit in VRAM to system
RAM, so for video the RAM matters as much as the card. The fp8 video workflow was not tried on 12 GB.

## Quickstart (one machine)

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and a running ComfyUI. If you
have none, download the portable build from the
[ComfyUI releases page](https://github.com/comfyanonymous/ComfyUI/releases/latest) (Windows:
`ComfyUI_windows_portable_nvidia.7z`, or the AMD or CPU build). It is a `.7z` archive, so you need
[7-Zip](https://www.7-zip.org/) to unpack it. Start it with `run_nvidia_gpu.bat`. It listens on
`127.0.0.1:8188`; if yours uses another port, pass `--url` to `comfyrack init` below.

To let comfyrack install models for you (`setup --apply`), ComfyUI needs the Manager. On the portable
build, run `python_embeded\python.exe -m pip install -r ComfyUI\manager_requirements.txt` from the
unpacked folder, then start ComfyUI with `--enable-manager`. Without the package the flag does
nothing. [ComfyUI's README](https://github.com/comfyanonymous/ComfyUI#comfyui-manager) says the same.

```sh
uv tool install git+https://github.com/Zach-Wendt/comfyrack

cd your-project
comfyrack init                      # writes .comfyrack/config.toml (--url http://127.0.0.1:8190 for another port)
comfyrack find "text to image"      # search the catalog by name and description
comfyrack describe t2i --json
comfyrack setup t2i --apply         # installs the models t2i needs (about 17 GB); needs the Manager
comfyrack preflight t2i
comfyrack run t2i --set prompt="a lighthouse on a cliff at dusk" --set seed=1
```

`t2i` is a text-to-image workflow. It needs three model files, and `setup` has their download links.
`preflight` lists any that are still missing.

`run` prints the path of each output file on stdout. Progress goes to stderr. Add `--json` to get the
outputs plus a `provenance` block: workflow, layer, recipe, machine, graph hash, and the values used.

To use Claude Code, install the plugin. It installs the CLI and the skills together:

```
/plugin marketplace add Zach-Wendt/comfyrack
/plugin install comfyrack@comfyrack
```

For Antigravity, run `agy plugin install /path/to/comfyrack`.

## Config

`comfyrack init` writes `.comfyrack/config.toml` in the current directory. comfyrack finds the nearest
one by walking up from where you run it. It merges that file over the user file
`~/.comfyrack/config.toml`.

```toml
[machines]
default = "http://127.0.0.1:8188"

[paths]
results = ".comfyrack/results"
recipes = ".comfyrack/recipes"
```

- `[machines]` maps a name to a ComfyUI URL. `--machine NAME` picks one.
- `[paths]` points comfyrack at directories your project already has. Relative paths start at the
  project root.
- `[list] scope` limits which families `comfyrack list` shows. It never blocks a run.

The ComfyUI URL comes from the first of these that is set:

1. `--machine NAME`
2. the `COMFY_URL` environment variable
3. `default` under `[machines]`
4. `http://127.0.0.1:8188`

Set `COMFYRACK_TRACEBACK=1` to see a full traceback on errors.

## How it works

**Registry.** A workflow is an API-format graph JSON plus a `.manifest.yaml`. The registry has three
layers. Higher layers win when names collide.

| Layer | Location |
|---|---|
| project | `<project>/.comfyrack/registry` |
| user | `~/.comfyrack/registry` |
| builtin | ships with comfyrack, one folder per family under `registry/` |

`comfyrack list --layer NAME` shows one layer.

**Manifests and flags.** The manifest names the graph nodes you may change. Each is a flag with a
type, a default, and whether it is required. Flags can also carry `choices`, `min`, `max`,
`min_length`, and `max_length`. `describe --json` refreshes these limits from the live ComfyUI
`/object_info` and reports `"live": true` or `false`. `requires:` lists custom node packs (with git
URLs) and model files.

**Modes.** `mode:` tells you what a workflow does: `t2i`, `t2v`, `i2i`, `i2v`, `v2v`, `r2v`, or `r2i`.
Filter with `comfyrack find QUERY --mode t2i` or `comfyrack list --mode t2i`.

**Recipes.** Workflows are shared. A recipe is a named set of flag values that a project owns. Pass
one with `--recipe NAME`. See `comfyrack recipe list` and `comfyrack recipe show`.

**Preflight.** `preflight` checks the graph against the target machine before it queues anything. It
checks node classes and every combo value (model files, samplers, input files), in both the old and
the V3 `COMBO` input shapes. It names the node pack when a class is missing.
`run` does this first. `--skip-preflight` turns it off.

**Setup.** `comfyrack setup NAME` compares a workflow's `requires:` record with the target machine. It
prints the node packs (registry ID, git URL) and model files (download URL, size) that the machine
lacks. `--apply` sends them to ComfyUI-Manager, waits for its queue, and tells you to restart ComfyUI
when it installed packs. It works with the Manager that ComfyUI can load (install
`manager_requirements.txt` into ComfyUI's Python, then start ComfyUI with `--enable-manager`) and
with the older ComfyUI-Manager custom node. Manager can refuse an install,
for example a pack that is not in the comfy registry or, in the older Manager, a model that is not in
its model list. `setup` then prints the reason and the URL so you can install it by hand.

LoRAs are yours to bring. `setup` lists a missing one under "your LoRAs" and never downloads it; put the
file in `models/loras` and pass its name with `--set` (for example `--set id_lora_name=<file>`).

`requires:` records each node class as `core` or a pack folder, each pack's git URL and registry ID,
and each model's folder and URL. `onboard` writes it. `comfyrack deps backfill --family F` fills it
for workflows already registered: it prints a diff, and `--write` saves it.

**Adding a workflow.** In ComfyUI, save it with Workflow > Export (API), then
`comfyrack onboard FILE.json --family F --name N`. Onboard writes the graph and a manifest whose flags
come from the machine's `/object_info`; trim the flags by hand. The graph file must be API format:
onboard rejects UI-format files. That format changes with each ComfyUI release, so a copy converted outside the browser can break. Add
`--ui UI.json` (the same workflow saved in UI format) to read model download URLs from it.

**Results.** comfyrack reads each result from the ComfyUI job record, not from the websocket. The
websocket carries progress only. A manifest can list extra `outputs:` (`{title, kind: image|text}`). `run --json` returns them under `"outputs"`,
keyed by title, so one workflow can return several images or text.

## Advanced: several machines

Add one entry per machine, then pick one per command:

```toml
[machines]
default = "http://127.0.0.1:8188"
gpu-box = "http://192.0.2.10:8188"
```

```sh
comfyrack preflight concept-gen --machine gpu-box
comfyrack run concept-gen --machine gpu-box --set prompt="..." --set negative="..." --set seed=1
```

`comfyrack machines` lists every configured machine and the default; `--json` returns that as data.

`comfyrack batch` renders a shot list from a YAML file. A shot can set its own `comfy_url`. Use
`--machine NAME` to pin the batch to one machine, or `--machines a,b` to spread shots across several.
Use one option or the other, not both. comfyrack sends each shot only to a machine that holds the
models it needs. Model files do not sync between machines. `--validate` and `--dry-run` check a shot
list without rendering. See `skills/comfyrack-batch/SKILL.md` for the shot-list format and result
statuses.

## Use this ComfyUI from other machines

On the machine that runs ComfyUI, install [Tailscale](https://tailscale.com/download) and sign in. Then run:

```sh
comfyrack share
```

comfyrack checks that Tailscale is signed in and that ComfyUI answers, turns on `tailscale serve` for the
ComfyUI port, and prints the address, for example `http://my-pc.example.ts.net:8188`. The setting
survives reboots, so you run it once. On your other machines, put that address under `[machines]`
(or run `comfyrack init --url <address>`). Only devices on your tailnet can reach it.

`comfyrack share --status` shows whether it is on. `comfyrack share --off` turns it off.
`--port N` picks another port (the default is the port in your machine config, else 8188). On Linux,
if Tailscale refuses, run `sudo tailscale set --operator=$USER` once.

## Limitations

- Alpha. Commands, flags and the manifest format can change between versions.
- Tested on Windows and Linux with NVIDIA cards (an RTX 3060 12 GB and an RTX 5090 32 GB). macOS, AMD
  and 16-24 GB cards are untested.
- Most workflows have not been measured on a 12 GB card yet. The demo table above shows the ones that were.
- Prompt guidance is general for most model families. Each video and image model wants its own prompt
  style, and only some have a written guide (see `skills/comfyrack-prompting`).
- LoRAs are yours to supply. `setup` lists a missing LoRA but never downloads one.
- Some models have no known download source. `setup` marks them "source unknown".
- `setup --apply` is tested with the ComfyUI-Manager bundled with ComfyUI. The older custom-node Manager is
  untested.
- More than one machine needs Tailscale.

## Roadmap

1. **Low and high VRAM for every workflow.** Run each workflow on a 12 GB and a 32 GB card, record the time
   and peak memory in its manifest, and name the variant to use on a smaller card. `describe` shows it.
2. **A prompt guide per model.** An LTX video prompt and a MiniMax H3 prompt are written differently. Each
   model family gets a short guide, delivered as a skill that `describe` points to, so an agent loads only
   the guide for the model it is about to run.
3. **Few tokens per answer.** Agents read every line comfyrack prints. Keep the default output short, keep
   `--json` complete, and cut text an agent does not need.
4. **New models.** Add workflows for new models as they come out, and retire ones that newer models replace.
5. **Releases.** A changelog in plain language, the same checks before every release, and a documented way
   to roll a bad release back.

## Contributing

Pull requests are open to collaborators only. To report a bug, use the bug form (see "Report a problem").
To work on your own copy, clone the repository and install it with the test tools:

```sh
git clone https://github.com/Zach-Wendt/comfyrack
cd comfyrack
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

The tests need no ComfyUI and no GPU. The tests in `tests/integration` talk to a live ComfyUI and
are off by default; run them with `uv run pytest -m integration`. Before a pull request, run
`sh scripts/release-grep`: it fails if a file holds personal paths or names.

To add a workflow to the registry, follow "Adding a workflow" above, then run `comfyrack describe`
on it and check that the flags and notes read well.

## Report a problem

Open an issue with the bug form at <https://github.com/Zach-Wendt/comfyrack/issues/new/choose>. Every
field is required: the command you ran, its full output, what you expected, `comfyrack --version` and your
ComfyUI version, and your operating system. Set
`COMFYRACK_TRACEBACK=1` and run the command again to get a traceback for an "unexpected error"
message.

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) runs every workflow. comfyrack only drives its
  HTTP API.
- Each workflow manifest names its source and author in its `description`.

comfyrack is released under the [MIT License](LICENSE). The license covers the comfyrack code. See
each manifest for the terms of the workflow it describes.
