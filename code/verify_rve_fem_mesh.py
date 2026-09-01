#!/usr/bin/env python3
"""Fine-mesh verification of the axial-fabric FEM contrast at f0."""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
from pathlib import Path

from _paths import MESHES, RESULTS
import run_scenario0 as rs
import run_rve_fem_validation as rv


OUT = RESULTS / "topology_research" / "rve_directional_scaling"


def main():
    rows = []
    for axis in ("x", "z"):
        centres = rv.orient(rv.base_centres(), axis)
        path = MESHES / f"rve_cube_8particle_L80_{axis}_fine.msh"
        if not path.exists():
            with tempfile.TemporaryDirectory() as tmp:
                geo = Path(tmp) / "rve_fine.geo"
                rs.write_geo_n_grain(geo, rv.LENGTH, rv.LENGTH, rv.LENGTH, centres, 0.008, 0.0012)
                if not rs.run_gmsh(geo, path):
                    raise RuntimeError("Gmsh failed")
        rs.LX = rs.LY = rs.LZ = rv.LENGTH
        rs.XC = rs.YC = rs.ZC = rv.LENGTH / 2
        rs.V_RIGHT = rs.E0 * rv.LENGTH
        mesh, facets = rs.load_msh_with_tags(path, centres)
        for state, edges in (("complete", rv.contacts("axial_fabric")), ("disconnected", [])):
            with contextlib.redirect_stdout(io.StringIO()):
                _, ur, ui, _ = rs.solve(
                    mesh, facets, v_right=rs.V_RIGHT, centres=centres,
                    vim_electrode="dirichlet", component_ids=list(range(8)),
                    omega=2 * math.pi * rs.FP, contact_edges=edges)
            left = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=rv.LENGTH, side="left")
            right = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=rv.LENGTH, side="right")
            sigma = 0.5 * (left + right)
            rows.append({
                "axis": axis, "state": state,
                "sigma_re_S_m": sigma.real, "sigma_im_S_m": sigma.imag,
                "electrode_mismatch": abs(left-right)/abs(sigma),
                "grain_facets": int(len(facets["grain"])), "mesh": str(path),
            })
        print(f"fine axis={axis} done", flush=True)
    idx = {(r["axis"], r["state"]): r for r in rows}
    delta = {}
    for axis in ("x", "z"):
        c, d = idx[(axis, "complete")], idx[(axis, "disconnected")]
        delta[axis] = abs(complex(c["sigma_re_S_m"]-d["sigma_re_S_m"], c["sigma_im_S_m"]-d["sigma_im_S_m"]))
    base = json.loads((OUT / "fem_analysis.json").read_text())
    b = next(r for r in base["metrics"] if r["network"] == "axial_fabric" and r["frequency_over_f0"] == 1.0)
    result = {
        "rows": rows,
        "fine_z_over_x": delta["z"]/delta["x"],
        "base_z_over_x": b["fem_z_over_xy"],
        "relative_ratio_change": abs(delta["z"]/delta["x"]-b["fem_z_over_xy"])/b["fem_z_over_xy"],
        "max_electrode_mismatch": max(r["electrode_mismatch"] for r in rows),
    }
    (OUT / "fem_mesh_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
