#!/usr/bin/env python3
"""One-frequency, same-geometry test of independent versus connected particles."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from _paths import MESHES, RESULTS
import run_scenario0 as rs


def main() -> None:
    length = 47e-3
    gap = 5e-3
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = 0.5 * length
    rs.V_RIGHT = rs.E0 * length
    x1 = 0.5 * length - 0.5 * gap - rs.R
    x2 = 0.5 * length + 0.5 * gap + rs.R
    centres = [np.array([x1, rs.YC, rs.ZC]), np.array([x2, rs.YC, rs.ZC])]
    mesh_path = MESHES / "connectivity_pilot_L47_d5.msh"
    if not mesh_path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp) / "connectivity_pilot.geo"
            rs.write_geo_two_grain(geo, length, gap, 0.006, 0.0012)
            if not rs.run_gmsh(geo, mesh_path):
                raise RuntimeError("Gmsh failed")

    mesh, facets = rs.load_msh_with_tags(mesh_path, centres)
    rows = []
    for case_id, labels in (
        ("independent", [0, 1]),
        ("ideal_connected", [0, 0]),
    ):
        basis, ur, ui, potentials = rs.solve(
            mesh,
            facets,
            v_right=rs.V_RIGHT,
            centres=centres,
            vim_electrode="dirichlet",
            component_ids=labels,
        )
        sigma_left = rs.extract_sigma_electrode(
            mesh, ur, ui, e0=rs.E0, L=length, side="left"
        )
        sigma_right = rs.extract_sigma_electrode(
            mesh, ur, ui, e0=rs.E0, L=length, side="right"
        )
        sigma = 0.5 * (sigma_left + sigma_right)
        rows.append(
            {
                "case_id": case_id,
                "component_ids": labels,
                "frequency_hz": rs.F_HZ,
                "sigma_re_S_m": sigma.real,
                "sigma_im_S_m": sigma.imag,
                "phase_deg": float(np.degrees(np.angle(sigma))),
                "floating_potentials": [[v.real, v.imag] for v in potentials],
                "left_right_relative_mismatch": abs(sigma_left - sigma_right)
                / max(abs(sigma), 1e-30),
            }
        )

    out = RESULTS / "topology_research" / "connectivity_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    print(json.dumps({"output": str(out), "cases": rows}, indent=2))


if __name__ == "__main__":
    main()

