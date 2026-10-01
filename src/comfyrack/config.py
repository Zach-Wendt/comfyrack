"""Layered configuration.

Resolution order: explicit argument -> COMFY_URL env -> project
.comfyrack/config.toml -> user ~/.comfyrack/config.toml -> builtin default.

Paths in [paths] are resolved relative to the project root, so a project can
point comfyrack at directories it already has rather than moving files."""
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ComfyrackError, UsageError

DEFAULT_URL = "http://127.0.0.1:8188"
_HOME = Path.home()
USER_CONFIG = _HOME / ".comfyrack" / "config.toml"


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ComfyrackError(
            f"cannot read config file {path}",
            help_text=f"fix the file format or remove it: {e}"
        ) from e


def _find_project_root(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        cfg = d / ".comfyrack" / "config.toml"
        # The home folder's config is the user layer, not a project that holds it.
        if cfg.is_file() and d != _HOME:
            return d
    return None


@dataclass
class Config:
    project_root: Path | None = None
    machines: dict = field(default_factory=dict)
    _paths: dict = field(default_factory=dict)
    _list: dict = field(default_factory=dict)
    _env_url: str | None = None

    @classmethod
    def load(cls, start_dir=None, user_config=None) -> "Config":
        start = Path(start_dir or Path.cwd()).resolve()
        root = _find_project_root(start)
        user_raw = _read_toml(Path(user_config) if user_config else USER_CONFIG)
        proj_raw = _read_toml(root / ".comfyrack" / "config.toml") if root else {}

        machines = dict(user_raw.get("machines", {}))
        machines.update(proj_raw.get("machines", {}))

        paths = dict(user_raw.get("paths", {}))
        paths.update(proj_raw.get("paths", {}))

        lst = dict(user_raw.get("list", {}))
        lst.update(proj_raw.get("list", {}))

        return cls(project_root=root, machines=machines, _paths=paths,
                   _list=lst, _env_url=os.environ.get("COMFY_URL"))

    def machine_url(self, name: str | None) -> str:
        if name:
            if name not in self.machines:
                known = ", ".join(sorted(self.machines)) or "(none configured)"
                raise UsageError(
                    f"unknown machine {name!r}",
                    help_text=f"configured machines: {known}",
                )
            return self.machines[name]
        if self._env_url:
            return self._env_url
        return self.machines.get("default", DEFAULT_URL)

    def path(self, key: str) -> Path | None:
        """Resolve a [paths] entry against the project root.

        An absolute value passes through unchanged -- pathlib guarantees this:
        `Path('/proj') / Path('/abs')` is `/abs`, because joining with an absolute
        right operand discards the left. There is deliberately NO `is_absolute()`
        branch here; one was removed after a reviewer proved it could not affect
        any outcome, and a guard that cannot change behaviour misleads its next
        reader into thinking it is load-bearing."""
        raw = self._paths.get(key)
        if raw is None:
            return None
        p = Path(raw)
        if self.project_root is None:
            return p
        return self.project_root / p

    def list_scope(self) -> list[str]:
        return list(self._list.get("scope", []))
