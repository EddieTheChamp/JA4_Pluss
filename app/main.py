"""Compatibility entrypoint for ``python -m app.main``."""

from current.app.cli import run


if __name__ == "__main__":
    raise SystemExit(run())
