---
name: comfyrack-characters
description: Look up a character's locked character profile (id-LoRA, trigger token, strengths, sampler, age bands) or a location's reference plate before rendering them. Use whenever a render involves a named character or a recurring location, or when a character's likeness drifts across a batch.
---

# Characters and locations

Never recall a character's appearance or settings from memory. Read them.

```
comfyrack char list                  # every character, trigger, id-LoRA, strength
comfyrack char show char_a           # the locked profile
comfyrack loc list                   # locations and whether a plate exists
comfyrack loc show location_x        # the plate path
```

## What a character profile holds

- **trigger** — the token the id-LoRA was trained on. It is meaningful only while
  that specific LoRA is loaded — it does not carry over to a different model or a
  graph that dropped the LoRA. Without it in the prompt, the LoRA nudges weakly and
  inconsistently; identity does not lock.
- **id lora + strength** — the locked file and its production strength.
- **style lora + strength** — the style stack applied alongside it.
- **sampler** — the sampler, scheduler, and step count this character was locked at.
- **age bands** — per-sequence and close-shot id-strength overrides. A close shot
  usually wants a lower strength than a full-body one.

## Identity drift in a batch

When likeness slips across renders, check in this order before re-rendering anything:

1. `comfyrack char show <name>` — is the strength being used the locked one?
2. `comfyrack lex show <name>` — does the prompt vocabulary match what the LoRA was
   trained on? A contradicting descriptor degrades identity silently.
3. `comfyrack preflight <workflow> --machine <m>` — is the id-LoRA actually present
   on the machine that ran it? Model files do not sync between machines.

## Two real leads in one frame

One id-LoRA slot means two identity-locked characters cannot both hold in a single
text-conditioned pass — stacking them bleeds identity. That case needs a compositing
workflow, with the second character added as a separate edit pass.
