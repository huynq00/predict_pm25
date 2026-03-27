from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_path: Path, *, override_all: bool = False) -> None:
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue

        if override_all:
            os.environ[key] = value
            continue

        if key.startswith("LLM_") or key.startswith("GEMINI_"):
            os.environ[key] = value
        elif key not in os.environ:
            os.environ[key] = value

