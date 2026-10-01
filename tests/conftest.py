"""Tests never reach the network through comfyrack.deps, and never see the real ~/.comfyrack.

Network: the registry, Manager's lists,
Hugging Face and HTTP HEAD all answer "nothing found" unless a test passes its own fake."""
import pytest


@pytest.fixture(autouse=True)
def _no_deps_network(monkeypatch):
    from comfyrack import deps
    monkeypatch.setattr(deps, "_get_json", lambda url, params=None: None)

    def _no_head(url):
        raise deps.requests.ConnectionError("network disabled in tests")
    monkeypatch.setattr(deps, "_head", _no_head)


@pytest.fixture(autouse=True)
def _no_real_user_home(tmp_path_factory, monkeypatch):
    """No test sees the real ~/.comfyrack: the user config and user registry point at an
    empty temp home, and HOME/USERPROFILE too. A test that wants a user config sets its own."""
    from comfyrack import config, registry
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(config, "USER_CONFIG", home / ".comfyrack" / "config.toml")
    monkeypatch.setattr(registry, "USER_DIR", home / ".comfyrack" / "registry")
