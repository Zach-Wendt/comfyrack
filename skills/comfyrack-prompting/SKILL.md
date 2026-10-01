---
name: comfyrack-prompting
description: Write a prompt for any ComfyUI generation through comfyrack's lexicon instead of composing prompt text directly. Use whenever a render needs a prompt, whenever a character or location name appears in one, or when output looks stylistically wrong for the model family being used.
---

# Prompting through comfyrack

Do not write `positive_text` yourself. Build the prompt from lexicon terms.

A model knows only the words in its training data. It does not fail on a word it
does not know. It ignores the word, or reads it as something else, and the render
looks fine but is wrong.

- A trigger word such as `trig_a` works only while the LoRA trained on it is
  loaded. Another model, or the same model without that LoRA, ignores it.
- A made-up place name such as "Location X" means nothing to the model.
- A name that is also a common word, such as "Ruby", may render as a gemstone.

## Before writing anything

```
comfyrack describe <workflow>     # prints the family's prompt dialect
comfyrack lex list                # the terms that actually resolve
```

## Dialects are not interchangeable

- **booru** (Illustrious lineage — `anima`, `illustrious`): comma-separated Danbooru
  tags. `1girl, (straight blonde hair:1.2), fair skin, standing on a dock`
- **natural** (Flux/Krea/LTX/Wan/Qwen): prose sentences. `a young girl with straight
  blonde hair standing on a wooden dock`

Weighted tokens are exact. `(straight blonde hair:1.2)` is not the same token as
`straight blonde hair` — never paraphrase one into the other.

## Video (LTX-2.3)

From the LTX-2 README (github.com/Lightricks/LTX-2): one flowing paragraph, present tense,
chronological, at most 200 words. Start with the main action. Then the specific movements,
appearance, environment, camera (angle and movement), lighting and colors, and any change
over the clip. LTX-2 makes audio too: describe the sound.

Name every motion you want with an explicit verb and its direction. The model does not
infer motion from a noun. "the beam sweeps across the sea" left a lighthouse lamp still;
"the lantern rotates steadily and its beam swings in a full circle, left to right" names
the motion. In image-to-video, describe what moves, not what the image already shows.

## Assembling

```
comfyrack prompt preview --lineage illustrious --dialect booru \
  --subject char_a --location location_x \
  --action "mid-dance, both arms raised, {char_b} watching closely"
```

`--subject` and `--location` resolve through the lexicon. `{term}` inside `--action`
resolves too. Free text belongs only in `--action`, and only for things no model
needs taught: poses, camera framing, lighting.

Always preview before spending GPU time.

## When a term is rejected

```
error: prompt vocabulary not grounded: Location (capitalised term not in lexicon)
help: comfyrack lex add <term> --kind <character|location|prop>, or pass --allow-unknown
```

Add the term rather than reaching for `--allow-unknown`:

```
comfyrack lex add location_x --kind location --lineage illustrious \
  --tags "ocean, wooden dock, overcast sky, coastal huts"
```

`--allow-unknown` is for a genuinely novel term you are deliberately testing. Using
it to silence the guard defeats it.

## Contradictions

The guard also refuses a description contradicting a LoRA's trained vocabulary —
writing "dark hair" against a LoRA trained on `(straight blonde hair:1.2)`. Check
what a character actually looks like rather than recalling it:

```
comfyrack lex show char_a
comfyrack char show char_a
```

## Keeping the lexicon current

```
comfyrack lex sync            # derive entries from LoRA metadata
```

Entries come from each LoRA's own `ss_tag_frequency` — the exact vocabulary it
responds to. Existing entries are preserved; pass `--overwrite` to regenerate.
