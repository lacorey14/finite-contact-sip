"""Package-local paths for the self-contained FEM_IP tree."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
PROJECT = PKG.parent
LIB = PKG / "lib"
MESHES = PROJECT / "meshes"
RESULTS = PROJECT / "results"
CAMPAIGN = RESULTS / "campaign"

# Prefer bundled lib; keep project postprocessing as fallback for optional OF tools.
for p in (PKG, LIB, PROJECT / "postprocessing"):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)
