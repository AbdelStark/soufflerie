"""Installed command-line adapter.

The command surface is implemented in issue #5. Keeping this module import-only
allows packaging and dependency-boundary checks to land without advertising
commands that do not exist yet.
"""
