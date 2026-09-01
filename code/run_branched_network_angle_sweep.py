#!/usr/bin/env python3
"""Direction-dependent SIP of a non-collinear four-particle T network."""

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


ANGLES_DEG = (0.0, 22.5, 45.0, 67.5, 90.0)
CONTACTS = [(0, 1, 0.02), (1, 2, 0.02), (1, 3, 0.02)]


def rotated_centres(angle_deg: float, length=0.075, spacing=0.015):
    centre = np.array([0.5 * length] * 3)
    # left--junction--right plus one upper branch; junction is particle 1.
    local = np.array([[-spacing, 0, 0], [0, 0, 0], [spacing, 0, 0], [0, spacing, 0]])
    theta = math.radians(angle_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    return [centre + rotation @ point for point in local]


def prepare_mesh(angle_deg: float):
    length = 0.075
    centres = rotated_centres(angle_deg, length)
    token = str(angle_deg).replace(".", "p")
    path = MESHES / f"four_particle_T_angle_{token}_L75.msh"
    if not path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp) / "branched.geo"
            rs.write_geo_n_grain(geo, length, length, length, centres, 0.0085, 0.0014)
            if not rs.run_gmsh(geo, path):
                raise RuntimeError("Gmsh failed")
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = 0.5 * length
    rs.V_RIGHT = rs.E0 * length
    mesh, facets = rs.load_msh_with_tags(path, centres)
    return length, centres, mesh, facets


def modal_geometry(centres):
    lap = contact_laplacian(4, CONTACTS)
    b = rs.E0 * np.array([c[0] for c in centres])
    b -= b.mean()
    values, vectors = np.linalg.eigh(lap)
    weights = np.abs(vectors.T @ b) ** 2
    return {
        "bLb_W": float(b @ lap @ b),
        "b2_V2": float(b @ b),
        "eigenvalues_S": values.tolist(),
        "modal_weights_V2": weights.tolist(),
    }


def main():
    frequencies = rs.FP * np.logspace(-1, 1, 5)
    rows, geometries = [], {}
    for angle in ANGLES_DEG:
        length, centres, mesh, facets = prepare_mesh(angle)
        geometries[str(angle)] = modal_geometry(centres)
        for case_id, contacts in (("disconnected", None), ("T_contact", CONTACTS)):
            for frequency in frequencies:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, ur, ui, _ = rs.solve(
                        mesh, facets, v_right=rs.V_RIGHT, centres=centres,
                        vim_electrode="dirichlet", component_ids=[0, 1, 2, 3],
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
                    "electrode_mismatch": abs(left-right) / max(abs(sigma), 1e-30),
                })
            print(f"angle={angle:g} case={case_id} done", flush=True)
    outdir = RESULTS / "topology_research" / "branched_network"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw_results.json").write_text(
        json.dumps({"network":"four-particle T", "contacts":CONTACTS,
                    "geometry_modal_data":geometries, "rows":rows}, indent=2) + "\n")
    print(json.dumps({"output":str(outdir), "n_solves":len(rows)}, indent=2))


if __name__ == "__main__":
    main()
