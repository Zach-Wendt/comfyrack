"""Every command and every flag carries help text."""
import argparse

from comfyrack.cli.main import build_parser


def _walk(parser, path):
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            helps = {a.dest: a.help for a in action._choices_actions}
            for name, sub in action.choices.items():
                yield f"{path} {name} (command)", helps.get(name)
                yield from _walk(sub, f"{path} {name}")
            continue
        label = action.option_strings[0] if action.option_strings else action.dest
        yield f"{path} {label}", action.help


def test_root_has_a_description():
    assert build_parser().description


def test_every_command_and_flag_has_help():
    missing = [label for label, text in _walk(build_parser(), "comfyrack") if not text]
    assert not missing, missing
