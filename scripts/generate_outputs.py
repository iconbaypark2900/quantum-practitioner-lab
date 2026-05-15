"""Backward-compatible entrypoint; prefer ``run_first_three_tutorial_outputs.py``."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_first_three_tutorial_outputs.py")), run_name="__main__")
