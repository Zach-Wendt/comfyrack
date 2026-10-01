import pytest
from comfyrack import errors, out


def test_usage_error_carries_exit_code_two():
    exc = errors.UsageError("unknown flag --stat", help_text="valid flags: --family, --layer")
    assert exc.exit_code == 2


def test_error_rendering_puts_help_on_its_own_line():
    exc = errors.UsageError("unknown flag --stat", help_text="valid flags: --family, --layer")
    assert out.error(exc) == "error: unknown flag --stat\nhelp: valid flags: --family, --layer"


def test_error_without_help_omits_the_help_line():
    exc = errors.ComfyrackError("connection refused")
    assert out.error(exc) == "error: connection refused"
