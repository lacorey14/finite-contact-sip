#!/usr/bin/env python3
"""Validate G = sigma A/L with explicit 3-D conductive-neck meshes."""

from __future__ import annotations

import json
import tempfile

from _paths import MESHES, RESULTS
from lib.explicit_neck import (
    analytic_cylinder_conductance,
    load_tet_mesh,
    mesh_cylinder,
    solve_neck_conductance,
    write_cylinder_geo,
)


def main():
    geometries = (
        (0.25e-3, 0.05e-3),
        (0.25e-3, 0.10e-3),
        (0.50e-3, 0.10e-3),
        (0.50e-3, 0.25e-3),
        (1.00e-3, 0.10e-3),
        (1.00e-3, 0.25e-3),
        (1.00e-3, 0.50e-3),
    )
    conductivities = (1.0, 100.0, 1e4)
    rows = []
    outmesh = MESHES / "explicit_necks"
    outmesh.mkdir(parents=True, exist_ok=True)
    for length, radius in geometries:
        stem = f"neck_L{length*1e3:g}mm_R{radius*1e3:g}mm"
        msh = outmesh / f"{stem}.msh"
        if not msh.exists():
            with tempfile.TemporaryDirectory() as tmp:
                from pathlib import Path
                geo = Path(tmp) / f"{stem}.geo"
                write_cylinder_geo(geo, length, radius, min(radius / 2.5, length / 8))
                mesh_cylinder(geo, msh)
        mesh = load_tet_mesh(msh)
        for conductivity in conductivities:
            numerical = solve_neck_conductance(mesh, conductivity, length)
            analytic = analytic_cylinder_conductance(conductivity, length, radius)
            rows.append({
                "length_m": length, "radius_m": radius,
                "conductivity_S_m": conductivity,
                "analytic_conductance_S": analytic,
                **numerical,
                "relative_error": abs(numerical["conductance_energy_S"] - analytic) / analytic,
            })
    result = {
        "model": "explicit 3-D cylindrical neck, equipotential ends, insulated side",
        "mapping": "G = sigma*pi*r^2/L",
        "max_relative_error": max(row["relative_error"] for row in rows),
        "max_terminal_mismatch": max(row["terminal_mismatch"] for row in rows),
        "rows": rows,
    }
    out = RESULTS / "topology_research" / "explicit_neck_validation.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()

