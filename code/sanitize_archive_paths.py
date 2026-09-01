#!/usr/bin/env python3
"""Replace machine-specific mesh paths in archived JSON with portable paths."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and "/meshes/" in value:
        return f"meshes/{value.rsplit('/meshes/', 1)[1]}"
    return value


def main() -> None:
    changed = 0
    for path in sorted(RESULTS.rglob("*.json")):
        original = json.loads(path.read_text(encoding="utf-8"))
        sanitized = portable(original)
        if sanitized != original:
            path.write_text(
                json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            changed += 1
    print(f"Sanitized {changed} JSON files.")


if __name__ == "__main__":
    main()
