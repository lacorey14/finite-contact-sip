#!/usr/bin/env python3
"""Write SHA-256 checksums for the frozen public archive."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SHA256SUMS"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and p != OUT
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc", ".tif", ".tiff"}
        and not ("results" in p.parts and p.name.startswith("Figure_") and p.suffix in {".png", ".pdf", ".svg"})
    ]
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in sorted(files)]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUT.name}")


if __name__ == "__main__":
    main()
