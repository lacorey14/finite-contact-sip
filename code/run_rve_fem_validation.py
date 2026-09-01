#!/usr/bin/env python3
"""Full 3-D FEM endpoints for RVE self-averaging/fabric conclusions."""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from _paths import MESHES, RESULTS
import run_scenario0 as rs


OUT = RESULTS / "topology_research" / "rve_directional_scaling"
LENGTH = 0.080
OFFSET = 0.008
G_ISO = 0.020
RATIOS = (0.1, 1.0, 10.0)
AXES = ("x", "y", "z")


def base_centres():
    c = LENGTH / 2
    out = []
    # Node bits are ordered x, y, z.
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                out.append(np.array([c + (2 * ix - 1) * OFFSET,
                                     c + (2 * iy - 1) * OFFSET,
                                     c + (2 * iz - 1) * OFFSET]))
    return out


def node(ix, iy, iz):
    return 4 * ix + 2 * iy + iz


def cube_edges():
    edges = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                if ix == 0: edges.append((node(ix, iy, iz), node(1, iy, iz), "x"))
                if iy == 0: edges.append((node(ix, iy, iz), node(ix, 1, iz), "y"))
                if iz == 0: edges.append((node(ix, iy, iz), node(ix, iy, 1), "z"))
    return edges


def contacts(kind):
    out = []
    for u, v, axis in cube_edges():
        if kind == "isotropic":
            g = G_ISO
        elif kind == "axial_fabric":
            # Same graph and same total conductance as isotropic: z contacts
            # carry 5/6 of total G, x/y contacts share the remaining 1/6.
            g = 0.050 if axis == "z" else 0.005
        else:
            raise ValueError(kind)
        out.append((u, v, g))
    return out


def orient(centres, measured_axis):
    p = np.asarray(centres)
    if measured_axis == "x":
        q = p
    elif measured_axis == "y":
        q = p[:, [1, 0, 2]]
    elif measured_axis == "z":
        q = p[:, [2, 1, 0]]
    else:
        raise ValueError(measured_axis)
    return [x.copy() for x in q]


def prepare(centres):
    path = MESHES / "rve_cube_8particle_L80.msh"
    if not path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp) / "rve.geo"
            rs.write_geo_n_grain(geo, LENGTH, LENGTH, LENGTH, centres, 0.010, 0.0018)
            if not rs.run_gmsh(geo, path):
                raise RuntimeError("Gmsh failed")
    rs.LX = rs.LY = rs.LZ = LENGTH
    rs.XC = rs.YC = rs.ZC = LENGTH / 2
    rs.V_RIGHT = rs.E0 * LENGTH
    mesh, facets = rs.load_msh_with_tags(path, centres)
    return mesh, facets, path


def graph_prediction(kind, measured_axis, ratio):
    # Same reduced RC equation used in the large ensemble, evaluated on the
    # deterministic cube endpoint with physical conductance ratios.
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import spsolve
    pts = np.asarray(base_centres())
    coord = {"x": 0, "y": 1, "z": 2}[measured_axis]
    b = pts[:, coord] - pts[:, coord].mean()
    es = contacts(kind)
    u = np.array([x[0] for x in es]); v = np.array([x[1] for x in es]); g = np.array([x[2] for x in es])
    rows = np.r_[u, v, u, v]; cols = np.r_[u, v, v, u]; vals = np.r_[g, g, -g, -g]
    lg = coo_matrix((vals, (rows, cols)), shape=(8, 8)).tocsc()
    # Particle interface scale at f0: omega0*C0*A = 0.02513 S.
    omega_c = ratio * (2 * math.pi * rs.FP) * rs.C0 * (4 * math.pi * rs.R**2)
    vm = spsolve(lg + 1j * omega_c * eye(8, format="csc"), 1j * omega_c * b)
    q = b - vm
    return complex(np.vdot(b, q) / np.vdot(b, b))


def main():
    rows = []
    base = base_centres()
    for axis in AXES:
        centres = orient(base, axis)
        mesh, facets, path = prepare(centres)
        for kind in ("isotropic", "axial_fabric"):
            for state, es in (("complete", contacts(kind)), ("disconnected", [])):
                for ratio in RATIOS:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, ur, ui, _ = rs.solve(
                            mesh, facets, v_right=rs.V_RIGHT, centres=centres,
                            vim_electrode="dirichlet", component_ids=list(range(8)),
                            omega=2 * math.pi * rs.FP * ratio, contact_edges=es)
                    left = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=LENGTH, side="left")
                    right = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=LENGTH, side="right")
                    sigma = 0.5 * (left + right)
                    gp = graph_prediction(kind, axis, ratio)
                    rows.append({
                        "network": kind, "measured_axis": axis, "state": state,
                        "frequency_over_f0": ratio,
                        "sigma_re_S_m": sigma.real, "sigma_im_S_m": sigma.imag,
                        "electrode_mismatch": abs(left - right) / max(abs(sigma), 1e-30),
                        "graph_prediction_re": gp.real, "graph_prediction_im": gp.imag,
                        "mesh": str(path),
                    })
            print(f"axis={axis} network={kind} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "eight-particle cube; identical geometry and edge graph; isotropic versus conductance fabric",
        "isotropic_G_S": {"x": 0.02, "y": 0.02, "z": 0.02},
        "fabric_G_S": {"x": 0.005, "y": 0.005, "z": 0.05},
        "total_G_both_S": 0.24,
        "rows": rows,
    }
    (OUT / "fem_raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / 'fem_raw.json'), "solves": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
