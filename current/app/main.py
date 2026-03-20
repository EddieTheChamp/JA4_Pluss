"""Module entrypoint for ``python -m app.main``."""

from __future__ import annotations

from current.app.cli import run


if __name__ == "__main__":
    raise SystemExit(run())
