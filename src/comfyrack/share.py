"""Share this machine's ComfyUI on the user's Tailscale tailnet.

Three steps, so nobody types the serve command: check Tailscale is installed and
signed in, check ComfyUI answers locally, run `tailscale serve --bg`. `tailscale serve`
keeps its settings across reboots, so this runs once per machine.

Every failure is a ComfyrackError whose help line is the one-line fix."""
import json
import os
import shutil
import subprocess
import sys
import urllib.request

from .errors import ComfyrackError

_WIN_DEFAULT = r"C:\Program Files\Tailscale\tailscale.exe"


def _tailscale() -> str:
    exe = shutil.which("tailscale")
    if exe is None and os.name == "nt" and os.path.isfile(_WIN_DEFAULT):
        exe = _WIN_DEFAULT
    if exe is None:
        raise ComfyrackError("Tailscale is not installed",
                             help_text="install it from https://tailscale.com/download, "
                                       "sign in, then run `comfyrack share` again")
    return exe


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([_tailscale(), *args], capture_output=True, text=True, timeout=30)


def _answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def _dns_name() -> str:
    """The machine's MagicDNS name; raises when Tailscale is not signed in."""
    proc = _run(["status", "--json"])
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        data = {}
    if data.get("BackendState") != "Running":
        raise ComfyrackError("Tailscale is not signed in",
                             help_text="run `tailscale up`, sign in, then run `comfyrack share` again")
    name = (data.get("Self") or {}).get("DNSName", "").rstrip(".")
    if not name:
        raise ComfyrackError("Tailscale has no MagicDNS name for this machine",
                             help_text="turn on MagicDNS in the Tailscale admin console (DNS tab)")
    return name


def _is_served(port: int) -> bool:
    proc = _run(["serve", "status", "--json"])
    try:
        web = json.loads(proc.stdout).get("Web") or {}
    except ValueError:
        return False
    target = f"http://127.0.0.1:{port}"
    return any(host.endswith(f":{port}") and h.get("Proxy") == target
               for host, cfg in web.items() for h in (cfg.get("Handlers") or {}).values())


def status(port: int) -> dict:
    """Read-only. Raises only when Tailscale is missing or signed out."""
    dns = _dns_name()
    return {"port": port, "url": f"http://{dns}:{port}", "shared": _is_served(port),
            "comfyui_up": _answers(f"http://127.0.0.1:{port}")}


def enable(port: int) -> dict:
    dns = _dns_name()
    if not _answers(f"http://127.0.0.1:{port}"):
        raise ComfyrackError(f"ComfyUI does not answer on 127.0.0.1:{port}",
                             help_text="start ComfyUI first, or pass --port with the port it uses")
    url = f"http://{dns}:{port}"
    if not _is_served(port):
        proc = _run(["serve", "--bg", f"--http={port}", f"http://127.0.0.1:{port}"])
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            if sys.platform.startswith("linux") and ("denied" in err.lower() or "operator" in err.lower()):
                raise ComfyrackError("Tailscale refused to serve for this user",
                                     help_text="run once: sudo tailscale set --operator=$USER")
            raise ComfyrackError(f"tailscale serve failed: {err.splitlines()[-1] if err else proc.returncode}",
                                 help_text="run `tailscale serve status` to see what is set")
    if not _answers(url):
        raise ComfyrackError(f"{url} does not answer from this machine",
                             help_text="check Tailscale is connected and MagicDNS is on, then retry")
    return {"port": port, "url": url, "shared": True, "comfyui_up": True}


def disable(port: int) -> dict:
    dns = _dns_name()
    if _is_served(port):
        proc = _run(["serve", f"--http={port}", "off"])
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            raise ComfyrackError(f"tailscale serve off failed: {err.splitlines()[-1] if err else proc.returncode}",
                                 help_text="run `tailscale serve status` to see what is set")
    return {"port": port, "url": f"http://{dns}:{port}", "shared": False,
            "comfyui_up": _answers(f"http://127.0.0.1:{port}")}
