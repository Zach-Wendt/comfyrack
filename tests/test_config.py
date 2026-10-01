import pytest
from pathlib import Path
from comfyrack.config import Config
from comfyrack import errors


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_builtin_default_when_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "nonexistent.toml")
    assert cfg.machine_url(None) == "http://127.0.0.1:8188"


def test_project_config_beats_user_config(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    write(tmp_path / ".comfyrack" / "config.toml",
          '[machines]\ndefault = "http://project:8188"\n')
    user = tmp_path / "user.toml"
    write(user, '[machines]\ndefault = "http://user:8188"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=user)
    assert cfg.machine_url(None) == "http://project:8188"


def test_env_beats_project_config(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_URL", "http://env:8188")
    write(tmp_path / ".comfyrack" / "config.toml",
          '[machines]\ndefault = "http://project:8188"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert cfg.machine_url(None) == "http://env:8188"


def test_named_machine_resolves_and_ignores_env(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_URL", "http://env:8188")
    write(tmp_path / ".comfyrack" / "config.toml",
          '[machines]\ndefault = "http://a:8188"\nremote = "http://b:8188"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert cfg.machine_url("remote") == "http://b:8188"


def test_unknown_machine_name_is_a_usage_error_listing_known_machines(tmp_path):
    write(tmp_path / ".comfyrack" / "config.toml",
          '[machines]\ndefault = "http://a:8188"\nremote = "http://b:8188"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    with pytest.raises(errors.UsageError) as ei:
        cfg.machine_url("nope")
    assert "remote" in ei.value.help_text


def test_config_found_by_walking_up_from_a_subdirectory(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    write(tmp_path / ".comfyrack" / "config.toml",
          '[machines]\ndefault = "http://root:8188"\n')
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    cfg = Config.load(start_dir=deep, user_config=tmp_path / "none.toml")
    assert cfg.machine_url(None) == "http://root:8188"
    assert cfg.project_root == tmp_path


def test_paths_and_list_scope_are_read_from_config(tmp_path):
    write(tmp_path / ".comfyrack" / "config.toml",
          '[list]\nscope = ["anima", "qwen"]\n\n[paths]\nresults = "out/renders"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert cfg.list_scope() == ["anima", "qwen"]
    assert cfg.path("results") == tmp_path / "out" / "renders"


def test_malformed_toml_raises_comfyrack_error_with_help(tmp_path):
    write(tmp_path / ".comfyrack" / "config.toml",
          "[machines]\ndefault = unclosed_string\n")
    with pytest.raises(errors.ComfyrackError) as ei:
        Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert ei.value.help_text is not None
    assert "fix the file format" in ei.value.help_text


def test_unreadable_config_file_raises_comfyrack_error_with_help(tmp_path, monkeypatch):
    # Monkeypatch open() to raise OSError when trying to read project config
    write(tmp_path / ".comfyrack" / "config.toml", '[machines]\ndefault = "http://a:8188"\n')

    original_open = open
    def failing_open(path, *args, **kwargs):
        path_str = str(path).replace("\\", "/")  # Normalize to forward slashes
        if ".comfyrack/config.toml" in path_str:
            raise OSError("Permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    with pytest.raises(errors.ComfyrackError) as ei:
        Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert ei.value.help_text is not None
    assert "fix the file format" in ei.value.help_text


def test_user_config_section_not_in_project_survives_merge(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    write(tmp_path / ".comfyrack" / "config.toml",
          '[machines]\ndefault = "http://project:8188"\n')
    user = tmp_path / "user.toml"
    write(user, '[machines]\nuser_local = "http://user-only:8188"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=user)
    assert cfg.machine_url("user_local") == "http://user-only:8188"


def test_absolute_path_passthrough_unmodified(tmp_path):
    # This test documents the pathlib guarantee: `Path(root) / Path(absolute)` is
    # `absolute`, because joining with an absolute right operand discards the left.
    # It is not testing a guard in the code; the `is_absolute()` check was removed
    # after a reviewer proved it could not affect any outcome.
    absolute = (Path(tmp_path.anchor) / "absolute" / "path").as_posix()
    write(tmp_path / ".comfyrack" / "config.toml",
          f'[paths]\nabsolute = "{absolute}"\n')
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    result = cfg.path("absolute")
    # Verify it's returned as-is, not joined with project_root.
    assert not str(result).startswith(str(cfg.project_root))


def test_path_when_project_root_is_none(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    # Load from a directory with NO .comfyrack/config.toml anywhere in the tree
    deep = tmp_path / "deep" / "nested" / "dir"
    deep.mkdir(parents=True)
    user = tmp_path / "user.toml"
    write(user, '[paths]\nrelative = "output/data"\n')
    cfg = Config.load(start_dir=deep, user_config=user)
    assert cfg.project_root is None
    # When project_root is None, path() returns the raw path unchanged
    assert cfg.path("relative") == Path("output/data")
