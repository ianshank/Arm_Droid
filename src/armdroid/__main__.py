"""Allow ``python -m armdroid <subcommand>`` to dispatch to the CLI."""

from __future__ import annotations

from armdroid.main import cli_entry

if __name__ == "__main__":
    cli_entry()
