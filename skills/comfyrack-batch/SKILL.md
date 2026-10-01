---
name: comfyrack-batch
description: Render a shot list end to end and judge the result, including fanning shots across multiple machines. Use whenever rendering a whole sequence or scene batch, re-rendering selected shots, or checking a finished batch against what was asked for.
---

# Batch rendering

## Always validate first

```
comfyrack batch shots/seq01_shots.yaml --validate
```

Catches missing seeds (renders that cannot be reproduced), duplicate shot names, and
shots with no action text. Cheap, and it runs before any GPU time is spent.

## Then dry-run

```
comfyrack batch shots/seq01_shots.yaml --dry-run
```

Lists each shot with the model files it needs. This is what routing uses.

## Render

```
comfyrack batch shots/seq01_shots.yaml --workflow character-scene --recipe anima_v1 \
  --sheet results/seq01_sheet.png
```

One shot only: `--shot s05_char_a_arrival_reunion`.

## Machines

Pin one machine:

```
comfyrack batch shots/seq01_shots.yaml --workflow character-scene --machine remote
```

Or fan out across several:

```
comfyrack batch shots/seq01_shots.yaml --workflow character-scene --machines local,remote
```

Pass one or the other, not both. comfyrack sends each shot only to a machine that
has its LoRAs. Model files do not sync between machines. A shot whose LoRA is on
one machine runs there. If a shot fails, comfyrack retries it on another machine
that can run it.

One failure never stops the queue. A 9-shot batch reports 8 ok and 1 failed rather
than aborting partway.

## The four per-shot statuses

Every shot comes back as exactly one of these:

| status | meaning | what to do |
| --- | --- | --- |
| `ok` | rendered; `result.paths` holds the images | nothing |
| `failed` | routed to a capable machine, but the render itself raised — OOM, an offline machine mid-job, a preflight refusal. Already retried up to `max_retries`; `attempts` lists each try and its error | read `detail`, fix, re-run that shot with `--shot` |
| `unroutable` | **no machine holds the required model.** `detail` names only files missing *everywhere*, or says the requirements are split across machines with no single machine holding the combination | copy the file across; re-running changes nothing |
| `unsupported` | the shot is a `via: shot_builder` composite. `batch` has no compositing pipeline wired in, and rendering just its `base` would produce a plausible single-character image silently missing its second lead — so it refuses instead. | composite those shots with the multi-character edit workflow, not with `batch` |

`unroutable` and `unsupported` shots never reach a GPU, so they cost nothing but
also produce nothing.

## Exit codes

- `0` — at least one shot rendered. **A partial success still exits 0**: 8 of 9 is a
  real result worth collecting, and the one that failed is re-run on its own.
- `1` — *no* shot rendered. Every shot failed, was unroutable, or was unsupported.
  A scripted caller chaining `batch` into an upscale or a judge pass can detect a
  totally-dead batch from the exit code alone.
- `2` — usage error: a bad shot list, both `--machine` and `--machines`, or a
  workflow that has no identity-LoRA strength flag for shots that declare
  `id_strength`.

Progress goes to **stderr**, labelled per shot (`s01: step 12/20 (60%)`). stdout
carries only the result table or `--json` payload, so it stays parseable.

## Judge the result

```
comfyrack judge results/seq01_sheet.png --against shots/seq01_shots.yaml
```

The rubric is generated from the shot list, so what is checked is exactly what was
asked for: content match, headcount, and defects per shot. `--rubric-only` prints the
rubric without running the judge workflow.

## Re-rendering

Change the seed for a shot that came back wrong, then re-run that shot alone with
`--shot`. Every render writes a provenance sidecar recording the resolved parameters,
seed, machine, and graph hash — read it rather than guessing what produced an image.
