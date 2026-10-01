"""Exception hierarchy. Every error carries the exit code the CLI should return
and, where possible, a `help:` line naming the command that fixes it."""


class ComfyrackError(Exception):
    exit_code = 1

    def __init__(self, message: str, help_text: str | None = None):
        super().__init__(message)
        self.message = message
        self.help_text = help_text


class UsageError(ComfyrackError):
    """Bad invocation: unknown flag, missing required value, malformed --set."""
    exit_code = 2


class NotFoundError(ComfyrackError):
    """A named thing (workflow, recipe, machine) does not resolve."""
    exit_code = 1
