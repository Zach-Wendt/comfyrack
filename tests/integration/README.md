# Live integration tests

These talk to a real ComfyUI instance. They are excluded from the default run.

```sh
python -m pytest tests/integration -m integration -v
COMFYRACK_TEST_URL=http://127.0.0.1:8188 python -m pytest tests/integration -m integration -v
```

Optional: `COMFYRACK_TEST_LORAS` (a LoRA directory), `COMFYRACK_TEST_ID_LORA` and
`COMFYRACK_TEST_ID_TRIGGER` (one LoRA file and its trigger word), and `COMFYRACK_TEST_PROJECT`
(a project path). The tests that need them skip when unset.

Each test skips (rather than fails) when no instance answers at the URL. The
one exception is `test_preflight_passes_for_a_registered_workflow_on_this_machine`,
which does not talk to `COMFYRACK_TEST_URL` directly -- `Rack()` resolves its
own machine URL from config -- but still takes the `client` fixture solely as
a skip gate, so it does not fail outright when nothing answers.

`test_upload_image_round_trips` writes a tiny 1x1 probe PNG into the target
ComfyUI's real input directory. That is intentional and harmless, but this
suite is not purely read-only -- keep that in mind before pointing
`COMFYRACK_TEST_URL` at a machine you don't want touched.
