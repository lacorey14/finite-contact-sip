#!/usr/bin/env python3
"""Continuous field-angle test of contact-mode visibility for a 3-grain chain."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from _paths import MESHES, RESULTS
from electronic_components import contact_laplacian
import run_scenario0 as rs


ANGLES_DEG = (22.5, 45, 67.5)
CASES = {
    "disconnected": None,
    "transition_chain": [(0, 1, 0.02), (1, 2, 0.02)],
}


def prepare_mesh(angle_deg: float):
    length, spacing = 70e-3, 15e-3
    centre = 0.5 * length
    theta = math.radians(angle_deg)
    direction = np.array([math.cos(theta), math.sin(theta), 0.0])
    centres = [np.array([centre, centre, centre]) + d * direction for d in (-spacing, 0, spacing)]
    token = str(angle_deg).replace(".", "p")
    path = MESHES / f"three_particle_angle_{token}_L70.msh"
    if not path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp) / "angle.geo"
            rs.write_geo_n_grain(geo, length, length, length, centres, 0.008, 0.0013)
            if not rs.run_gmsh(geo, path):
                raise RuntimeError("Gmsh failed")
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = centre
    rs.V_RIGHT = rs.E0 * length
    mesh, facets = rs.load_msh_with_tags(path, centres)
    return length, centres, mesh, facets


def graph_visibility(centres, edges):
    if not edges:
        return 0.0
    lap = contact_laplacian(3, edges)
    b = rs.E0 * np.array([c[0] for c in centres])
    b -= b.mean()
    return float(b @ lap @ b), float(b @ b)


def main():
    frequencies = rs.FP * np.logspace(-1.0, 1.0, 5)
    rows = []
    for angle in ANGLES_DEG:
        length, centres, mesh, facets = prepare_mesh(angle)
        for case_id, contacts in CASES.items():
            visibility = graph_visibility(centres, contacts)
            for frequency in frequencies:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, ur, ui, _ = rs.solve(
                        mesh, facets, v_right=rs.V_RIGHT, centres=centres,
                        vim_electrode="dirichlet", component_ids=[0, 1, 2],
                        omega=2 * math.pi * float(frequency), contact_edges=contacts,
                    )
                left = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="left")
                right = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="right")
                sigma = 0.5 * (left + right)
                rows.append({
                    "angle_deg": angle, "case_id": case_id,
                    "frequency_hz": float(frequency),
                    "frequency_over_f0": float(frequency / rs.FP),
                    "sigma_re_S_m": sigma.real, "sigma_im_S_m": sigma.imag,
                    "graph_visibility_bLb": visibility[0] if contacts else 0.0,
                    "background_projection_norm_b2": visibility[1] if contacts else 0.0,
                    "electrode_mismatch": abs(left-right) / max(abs(sigma), 1e-30),
                })
            print(f"angle={angle:g} case={case_id} done")
    outdir = RESULTS / "topology_research" / "angle_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw_results.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"output": str(outdir), "n_solves": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
