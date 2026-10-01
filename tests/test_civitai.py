import pytest
from comfyrack import civitai
from comfyrack.errors import ComfyrackError
from comfyrack.lora_meta import LoraMeta


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self.response


def test_file_hash_is_uppercase_sha256(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(b"content")
    h = civitai.file_hash(p)
    # Pinned to the actual sha256 of b"content" (verified independently via
    # `printf 'content' | sha256sum`), not just shape-checked -- a hash-of-the-
    # wrong-input mutation (e.g. hashing str(path)) still produces 64 uppercase
    # hex chars, so length/case alone can't catch it.
    assert h == "ED7002B439E9AC845F22357D822BAC1444730FBDB6016D3EC9432297B9EC9F73"


def test_trained_words_are_returned_from_the_version_lookup():
    s = FakeSession(FakeResponse({"trainedWords": ["trig_a", "cel style"],
                                  "baseModel": "Illustrious"}))
    assert civitai.trained_words("ABC123", session=s) == ["trig_a", "cel style"]


def test_lookup_uses_the_by_hash_endpoint():
    s = FakeSession(FakeResponse({"trainedWords": []}))
    civitai.trained_words("ABC123", session=s)
    assert "by-hash/ABC123" in s.urls[0]


def test_a_404_returns_no_words_rather_than_raising():
    # The fake body still contains trainedWords, so only the status_code guard
    # (civitai.py:29) can be responsible for the empty result -- if that guard
    # were deleted, this would return ["ghost"] instead of [].
    s = FakeSession(FakeResponse({"trainedWords": ["ghost"]}, status=404))
    assert civitai.trained_words("NOPE", session=s) == []


def test_trained_words_wraps_a_transport_failure_in_a_comfyrack_error():
    class Boom:
        def get(self, url, timeout=None):
            raise OSError("no network")

    with pytest.raises(ComfyrackError) as exc_info:
        civitai.trained_words("ABC123", session=Boom())
    assert exc_info.value.help_text


def test_trained_words_wraps_a_malformed_json_body_in_a_comfyrack_error():
    class Malformed:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    s = FakeSession(Malformed())
    with pytest.raises(ComfyrackError):
        civitai.trained_words("ABC123", session=s)


def test_enrich_fills_in_a_missing_trigger(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(b"x")
    meta = LoraMeta(filename="a.safetensors")
    s = FakeSession(FakeResponse({"trainedWords": ["trig_a"], "baseModel": "Illustrious"}))
    out = civitai.enrich(meta, p, session=s)
    assert out.trigger == "trig_a"


def test_enrich_fills_tags_from_the_remaining_trained_words(tmp_path):
    # words[0] becomes the trigger; tags must be the *rest*, not the whole list --
    # otherwise a trigger+tags render would emit the trigger twice.
    p = tmp_path / "a.safetensors"
    p.write_bytes(b"x")
    meta = LoraMeta(filename="a.safetensors")
    s = FakeSession(FakeResponse({"trainedWords": ["trig_a", "cel style", "anime"],
                                  "baseModel": "Illustrious"}))
    out = civitai.enrich(meta, p, session=s)
    assert out.trigger == "trig_a"
    assert out.tags == ["cel style", "anime"]
    assert "trig_a" not in out.tags


def test_enrich_fills_in_a_missing_lineage_lowercased(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(b"x")
    s = FakeSession(FakeResponse({"trainedWords": ["x"], "baseModel": "Illustrious"}))
    assert civitai.enrich(LoraMeta(filename="a"), p, session=s).base_model_family == \
        "illustrious"


def test_enrich_never_overwrites_metadata_already_read_from_the_file(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(b"x")
    meta = LoraMeta(filename="a.safetensors", trigger="local_trigger",
                    base_model_family="anima")
    s = FakeSession(FakeResponse({"trainedWords": ["remote"], "baseModel": "Flux"}))
    out = civitai.enrich(meta, p, session=s)
    assert out.trigger == "local_trigger"
    assert out.base_model_family == "anima"


def test_enrich_fills_only_the_one_field_that_is_actually_missing(tmp_path):
    # With only trigger set, the early-return guard at civitai.py:39-40 does NOT
    # fire (base_model_family is falsy), so this exercises the per-field
    # `not meta.X and` guards themselves rather than short-circuiting past them.
    p = tmp_path / "a.safetensors"
    p.write_bytes(b"x")
    meta = LoraMeta(filename="a.safetensors", trigger="local_trigger",
                    base_model_family="")
    s = FakeSession(FakeResponse({"trainedWords": ["remote"], "baseModel": "Flux"}))
    out = civitai.enrich(meta, p, session=s)
    assert out.trigger == "local_trigger"
    assert out.base_model_family == "flux"


def test_enrich_tolerates_a_network_failure_and_returns_the_input(tmp_path):
    class Boom:
        def get(self, url, timeout=None):
            raise OSError("no network")

    p = tmp_path / "a.safetensors"
    p.write_bytes(b"x")
    meta = LoraMeta(filename="a.safetensors")
    assert civitai.enrich(meta, p, session=Boom()) is meta
