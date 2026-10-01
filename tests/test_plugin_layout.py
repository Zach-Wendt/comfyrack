import json
import os
import pathlib
import shutil
import subprocess
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_claude_code_manifest_exists_with_a_name_and_description():
    m = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert m["name"] == "comfyrack"
    assert m["description"]


def test_claude_code_manifest_omits_version_so_every_commit_is_an_update():
    m = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert "version" not in m


def test_marketplace_manifest_lists_this_repo_as_a_plugin():
    m = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert any(p["name"] == "comfyrack" for p in m["plugins"])


def test_antigravity_manifest_exists_at_the_repo_root():
    m = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    assert m["name"] == "comfyrack"


def test_skills_live_at_the_repo_root_where_both_harnesses_read_them():
    assert (ROOT / "skills" / "comfyrack" / "SKILL.md").is_file()


def test_skill_frontmatter_has_a_trigger_shaped_description():
    text = (ROOT / "skills" / "comfyrack" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    head = text.split("---")[1]
    assert "name:" in head and "description:" in head


def test_both_bin_shims_exist():
    assert (ROOT / "bin" / "comfyrack").is_file()
    assert (ROOT / "bin" / "comfyrack.cmd").is_file()


def test_posix_shim_runs_the_bundled_copy_not_a_path_lookup():
    text = (ROOT / "bin" / "comfyrack").read_text(encoding="utf-8")
    assert "comfyrack.cli.main" in text
    # A bare `comfyrack "$@"` would recurse or hit an unrelated install.
    assert "\ncomfyrack " not in text


def test_windows_shim_runs_the_bundled_copy():
    text = (ROOT / "bin" / "comfyrack.cmd").read_text(encoding="utf-8")
    assert "comfyrack.cli.main" in text


import sys

@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX sh on PATH")
def test_posix_shim_actually_executes():
    # --help exits inside argparse before any Rack or network construction,
    # so this stays hermetic with no registry/config pinning required.
    env = dict(os.environ)
    env.setdefault("COMFYRACK_PYTHON", sys.executable)
    result = subprocess.run(
        ["sh", str(ROOT / "bin" / "comfyrack"), "--help"],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert result.returncode == 0
    assert "usage: comfyrack" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows-only shim")
def test_windows_shim_actually_executes():
    result = subprocess.run(
        ["cmd", "/c", str(ROOT / "bin" / "comfyrack.cmd"), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0
    assert "usage: comfyrack" in result.stdout


# -- wheel packaging of the builtin corpus ----------------------------------
#
# Static assertions over pyproject.toml rather than a real wheel build: building
# inside a test is slow and drags in the network/build backend. The failure being
# guarded is that `packages = ["src/comfyrack"]` alone ships a wheel containing no
# registry data at all, and Registry._iter_paths reports that as a clean, definitive,
# and totally wrong "0 workflows found in any registry layer".

def _wheel_target():
    raw = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return raw["tool"]["hatch"]["build"]["targets"]["wheel"]


def test_wheel_force_includes_the_builtin_registry_into_the_package():
    force_include = _wheel_target().get("force-include")
    assert force_include is not None, (
        "pyproject.toml declares no [tool.hatch.build.targets.wheel.force-include]; "
        "the built wheel would ship zero workflows")
    assert force_include.get("registry") == "comfyrack/registry", (
        f"registry/ must map to comfyrack/registry so the installed layout matches "
        f"registry._PKG_REGISTRY; got {force_include.get('registry')!r}")


def test_the_force_included_source_directory_actually_exists_and_holds_manifests():
    """A force-include naming a path that is not there builds a wheel with nothing
    in it and no warning, so pin the source side too."""
    force_include = _wheel_target().get("force-include") or {}
    assert force_include, "no force-include entries to check"
    for rel_src in force_include:
        assert (ROOT / rel_src).is_dir(), f"force-include source {rel_src!r} does not exist"
    assert list((ROOT / "registry").glob("*/*.manifest.yaml"))


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_posix_shim_has_the_executable_bit_set_in_git():
    result = subprocess.run(
        ["git", "ls-files", "-s", "bin/comfyrack"],
        capture_output=True, text=True, cwd=ROOT,
    )
    mode = result.stdout.split()[0]
    assert mode == "100755"


def test_every_manifest_carries_the_same_package_description():
    import tomllib
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    want = pyproject["project"]["description"]
    assert json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))["description"] == want
    assert json.loads((ROOT / ".claude-plugin" / "plugin.json")
                      .read_text(encoding="utf-8"))["description"] == want
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert market["plugins"][0]["description"] == want
