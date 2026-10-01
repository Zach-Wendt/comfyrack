import io
import sys

from comfyrack.cli.main import main


def test_describe_with_emoji_node_title_on_a_cp1252_stream(monkeypatch):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert main(["describe", "ltx2.3/i2v-t2v-basic"]) == 0
    sys.stdout.flush()
    assert "\U0001f3a5" in raw.getvalue().decode("utf-8")
