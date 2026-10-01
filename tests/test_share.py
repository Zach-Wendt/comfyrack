"""Hermetic tests for `comfyrack share`, driven by a fake `tailscale`.

The share module reaches out to the outside world through exactly four seams:

* `_run`        -- subprocess.run of `tailscale`
* `_answers`    -- urllib fetch of the ComfyUI / MagicDNS URL
* `shutil.which`-- PATH lookup for the `tailscale` binary
* `sys.platform` / `os.name` -- OS branching in `_tailscale` and `enable`

`FakeTailscale` replaces those seams. `os` and `sys` are swapped for shims on the
*share module* (not the process-wide module), so tests that build a `Rack`/`Config`
keep the real `pathlib` behaviour on Windows while `share._tailscale` still sees the
fake OS. No real `tailscale` binary runs and no network is touched."""
import json
import os
import subprocess
import sys

import pytest

from comfyrack import share
from comfyrack.cli.main import main
from comfyrack.errors import ComfyrackError


def _cp(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["tailscale", *args],
                                       returncode=returncode, stdout=stdout, stderr=stderr)


class _OSShim:
    """Stand-in for the `os` module visible only to `comfyrack.share`.

    `name` reads live from the fake; everything else (`path`, ...) delegates to
    the real `os` so `os.path.isfile` in `_tailscale` still works."""

    def __init__(self, fake):
        self._fake = fake

    @property
    def name(self):
        return self._fake.os_name

    def __getattr__(self, attr):
        return getattr(os, attr)


class _SysShim:
    """Stand-in for the `sys` module visible only to `comfyrack.share`.

    `platform` reads live from the fake; everything else delegates to the real
    `sys`. `share` only consults `sys.platform` (in `enable`), so this is safe."""

    def __init__(self, fake):
        self._fake = fake

    @property
    def platform(self):
        return self._fake.platform

    def __getattr__(self, attr):
        return getattr(sys, attr)


class FakeTailscale:
    """Stateful double for the share module's four external seams.

    Configure attributes in any order -- `platform` and `os_name` are read live
    through shims -- then call `.patch(monkeypatch)` once before exercising share
    code. `_run` records every tailscale arg list in `self.calls` so tests can
    assert on what would have been invoked."""

    def __init__(self):
        self.calls = []
        self.which_result = "/fake/bin/tailscale"   # "tailscale is installed"
        self.answers_result = True                   # ComfyUI / URL answers
        self.platform = "linux"
        self.os_name = "posix"                        # not the Windows default-path branch
        # `tailscale status --json` body -> _dns_name
        self.status_json = {"BackendState": "Running",
                            "Self": {"DNSName": "m.x.ts.net."}}
        self.status_stdout = None                    # raw override (non-JSON stdout)
        # `tailscale serve status --json` body -> _is_served
        self.serve_status_json = {"Web": {}}
        # `tailscale serve --bg ...`:
        self.serve_returncode = 0
        self.serve_stderr = ""

    def _run(self, args):
        self.calls.append(list(args))
        a = list(args)
        if a == ["status", "--json"]:
            out = self.status_stdout
            if out is None:
                out = json.dumps(self.status_json)
            return _cp(a, stdout=out)
        if a == ["serve", "status", "--json"]:
            return _cp(a, stdout=json.dumps(self.serve_status_json))
        if a[0] == "serve":
            return _cp(a, returncode=self.serve_returncode, stderr=self.serve_stderr)
        return _cp(a)

    def _answers(self, url):
        r = self.answers_result
        return r(url) if callable(r) else r

    def which(self, name):
        return self.which_result if name == "tailscale" else None

    def patch(self, monkeypatch):
        # Replace the module-level `os`/`sys` names in `share` (not the globals),
        # so pathlib/Rack/ComfyClient keep the real OS view while _tailscale/enable
        # see the fake one.
        monkeypatch.setattr(share, "os", _OSShim(self))
        monkeypatch.setattr(share, "sys", _SysShim(self))
        monkeypatch.setattr(share, "_run", self._run)
        monkeypatch.setattr(share, "_answers", self._answers)
        monkeypatch.setattr(share.shutil, "which", self.which)
        return self


# A fully-run serve config: host m.x.ts.net:8188 proxying 127.0.0.1:8188.
SERVED = {"Web": {"m.x.ts.net:8188": {
    "Handlers": {"/": {"Proxy": "http://127.0.0.1:8188"}}}}}


# -- hermetic project setup (mirrors test_cli's helpers, without importing it) ---

def _hermetic_project(tmp_path, monkeypatch, default_url="http://a:8188"):
    """A project .comfyrack/config.toml under tmp_path, empty builtin registry,
    no COMFY_URL, cwd pinned to the project."""
    from comfyrack import registry
    cfg = tmp_path / ".comfyrack"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "config.toml").write_text(
        f'[machines]\ndefault = "{default_url}"\n', encoding="utf-8")
    monkeypatch.setattr(registry, "BUILTIN_DIR", tmp_path / "builtin_empty")
    (tmp_path / "builtin_empty").mkdir()
    monkeypatch.delenv("COMFY_URL", raising=False)
    monkeypatch.chdir(tmp_path)


def _share_main(tmp_path, monkeypatch, capsys, argv, *, default_url="http://a:8188",
                **fake_attrs):
    """Run `comfyrack` from a hermetic project with a fake tailscale installed."""
    _hermetic_project(tmp_path, monkeypatch, default_url)
    fake = FakeTailscale()
    for key, value in fake_attrs.items():
        setattr(fake, key, value)
    fake.patch(monkeypatch)
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, fake


# -- _tailscale ----------------------------------------------------------------

def test_tailscale_not_installed_raises_with_download_help(tmp_path, monkeypatch):
    # `shutil.which` finds nothing and we are not on the Windows default path:
    # the module must surface the download URL as its one-line fix.
    fake = FakeTailscale()
    fake.which_result = None
    fake.os_name = "posix"
    fake.patch(monkeypatch)
    with pytest.raises(ComfyrackError) as exc:
        share._tailscale()
    err = exc.value
    assert "not installed" in err.message.lower()
    assert "tailscale.com/download" in err.help_text


def test_tailscale_installed_returns_the_binary_path(tmp_path, monkeypatch):
    FakeTailscale().patch(monkeypatch)
    assert share._tailscale() == "/fake/bin/tailscale"


# -- _dns_name / status() ------------------------------------------------------

def test_status_signed_out_via_needs_login(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.status_json = {"BackendState": "NeedsLogin", "Self": {}}
    with pytest.raises(ComfyrackError) as exc:
        share.status(8188)
    err = exc.value
    assert "not signed in" in err.message
    assert "tailscale up" in err.help_text


def test_dns_name_strips_trailing_dot_and_status_builds_url(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.status_json = {"BackendState": "Running", "Self": {"DNSName": "m.x.ts.net."}}
    fake.serve_status_json = {"Web": {}}
    assert share._dns_name() == "m.x.ts.net"          # trailing dot stripped
    data = share.status(8188)
    assert data == {"port": 8188, "url": "http://m.x.ts.net:8188",
                    "shared": False, "comfyui_up": True}


def test_status_empty_magic_dns_name_raises(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.status_json = {"BackendState": "Running", "Self": {"DNSName": ""}}
    with pytest.raises(ComfyrackError) as exc:
        share.status(8188)
    err = exc.value
    assert "MagicDNS" in err.message
    assert "MagicDNS" in err.help_text


def test_status_non_json_stdout_is_treated_as_signed_out(tmp_path, monkeypatch):
    # `tailscale status --json` printing anything that is not JSON must not
    # crash; it falls through to the "not signed in" error.
    fake = FakeTailscale().patch(monkeypatch)
    fake.status_stdout = "tailscale status: not running"   # not parseable as JSON
    with pytest.raises(ComfyrackError) as exc:
        share.status(8188)
    err = exc.value
    assert "not signed in" in err.message
    assert "tailscale up" in err.help_text


# -- enable --------------------------------------------------------------------

def test_enable_comfyui_not_answering_errors_with_port_hint(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.answers_result = False                       # ComfyUI is down locally
    with pytest.raises(ComfyrackError) as exc:
        share.enable(8188)
    err = exc.value
    assert "does not answer on 127.0.0.1:8188" in err.message
    assert "--port" in err.help_text


def test_enable_not_yet_served_runs_serve_once_then_shares(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.serve_status_json = {"Web": {}}              # nothing served yet
    data = share.enable(8188)
    assert data == {"port": 8188, "url": "http://m.x.ts.net:8188",
                    "shared": True, "comfyui_up": True}
    # Exactly the one `serve --bg` invocation the helper is meant to hide.
    assert ["serve", "--bg", "--http=8188", "http://127.0.0.1:8188"] in fake.calls


def test_enable_already_served_does_not_serve_again(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.serve_status_json = SERVED                   # already served -> idempotent
    data = share.enable(8188)
    assert data["shared"] is True
    assert not any(c[:2] == ["serve", "--bg"] for c in fake.calls)


def test_enable_linux_serve_denied_mentions_operator(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.platform = "linux"
    fake.serve_status_json = {"Web": {}}
    fake.serve_returncode = 1
    fake.serve_stderr = "Access denied"
    with pytest.raises(ComfyrackError) as exc:
        share.enable(8188)
    err = exc.value
    assert "Tailscale refused to serve for this user" in err.message
    assert "sudo tailscale set --operator=$USER" in err.help_text


def test_enable_win32_serve_denied_is_generic_error(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.platform = "win32"
    fake.serve_status_json = {"Web": {}}
    fake.serve_returncode = 1
    fake.serve_stderr = "Access denied"
    with pytest.raises(ComfyrackError) as exc:
        share.enable(8188)
    err = exc.value
    assert err.message.startswith("tailscale serve failed")
    assert "sudo" not in err.help_text
    assert "tailscale serve status" in err.help_text


def test_enable_magic_dns_url_not_answering_errors(tmp_path, monkeypatch):
    # ComfyUI on 127.0.0.1 answers, the serve succeeds, but the MagicDNS URL
    # does not -- so sharing is not reachable from the tailnet.
    fake = FakeTailscale().patch(monkeypatch)
    fake.serve_status_json = {"Web": {}}
    fake.answers_result = lambda url: "127.0.0.1" in url
    with pytest.raises(ComfyrackError) as exc:
        share.enable(8188)
    err = exc.value
    assert "does not answer from this machine" in err.message
    assert "MagicDNS" in err.help_text


# -- disable -------------------------------------------------------------------

def test_disable_served_calls_serve_off(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.serve_status_json = SERVED
    data = share.disable(8188)
    assert data["shared"] is False
    assert ["serve", "--http=8188", "off"] in fake.calls


def test_disable_not_served_does_not_touch_serve(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.serve_status_json = {"Web": {}}
    fake.answers_result = False
    data = share.disable(8188)
    assert data["shared"] is False
    assert ["serve", "--http=8188", "off"] not in fake.calls


# -- status() reflects the fakes ----------------------------------------------

def test_status_served_and_up_reflect_fakes(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    fake.serve_status_json = SERVED
    fake.answers_result = True
    served_up = share.status(8188)
    assert served_up["shared"] is True
    assert served_up["comfyui_up"] is True
    assert served_up["url"] == "http://m.x.ts.net:8188"

    fake.serve_status_json = {"Web": {}}
    fake.answers_result = False
    down = share.status(8188)
    assert down["shared"] is False
    assert down["comfyui_up"] is False


def test_status_a_proxy_to_a_different_port_is_not_served(tmp_path, monkeypatch):
    fake = FakeTailscale().patch(monkeypatch)
    # Tailscale is proxying a *different* port; the one we ask about is not served.
    fake.serve_status_json = {"Web": {"m.x.ts.net:9000": {
        "Handlers": {"/": {"Proxy": "http://127.0.0.1:9000"}}}}}
    assert share.status(8188)["shared"] is False


# -- CLI via comfyrack.cli.main.main ------------------------------------------

def test_share_json_prints_url_and_shared(tmp_path, capsys, monkeypatch):
    code, out, _err, _fake = _share_main(
        tmp_path, monkeypatch, capsys, ["share", "--json"],
        serve_status_json={"Web": {}})
    assert code == 0
    data = json.loads(out)
    assert data["url"] == "http://m.x.ts.net:8188"
    assert data["shared"] is True


def test_share_status_text_shows_yes_and_url(tmp_path, capsys, monkeypatch):
    code, out, _err, _fake = _share_main(
        tmp_path, monkeypatch, capsys, ["share", "--status"],
        serve_status_json=SERVED)
    assert code == 0
    assert "shared: yes" in out
    assert "url: http://m.x.ts.net:8188" in out


def test_share_off_and_status_together_exits_two_with_usage_error(
        tmp_path, capsys, monkeypatch):
    code, out, _err, _fake = _share_main(
        tmp_path, monkeypatch, capsys, ["share", "--off", "--status"],
        serve_status_json=SERVED)
    assert code == 2
    assert out.startswith("error:")
    assert "help:" in out


def test_share_no_flags_with_failing_fake_prints_error_and_help(
        tmp_path, capsys, monkeypatch):
    code, out, _err, _fake = _share_main(
        tmp_path, monkeypatch, capsys, ["share"],
        answers_result=False)                        # ComfyUI not answering
    assert code == 1
    assert out.startswith("error:")
    assert "help:" in out
    assert "--port" in out


def test_share_port_flag_reaches_enable(tmp_path, capsys, monkeypatch):
    code, out, _err, fake = _share_main(
        tmp_path, monkeypatch, capsys, ["share", "--port", "9000", "--json"],
        serve_status_json={"Web": {}})
    assert code == 0
    assert json.loads(out)["url"] == "http://m.x.ts.net:9000"
    assert ["serve", "--bg", "--http=9000", "http://127.0.0.1:9000"] in fake.calls


def test_share_port_is_read_from_config_machine_url_when_no_flag(
        tmp_path, capsys, monkeypatch):
    code, out, _err, fake = _share_main(
        tmp_path, monkeypatch, capsys, ["share", "--json"],
        default_url="http://127.0.0.1:9100",
        serve_status_json={"Web": {}})
    assert code == 0
    assert json.loads(out)["url"] == "http://m.x.ts.net:9100"
    assert ["serve", "--bg", "--http=9100", "http://127.0.0.1:9100"
            ] in fake.calls
