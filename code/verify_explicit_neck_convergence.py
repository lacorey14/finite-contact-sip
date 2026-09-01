#!/usr/bin/env python3
"""Mesh convergence for one explicit cylindrical contact neck."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _paths import MESHES, RESULTS
from lib.explicit_neck import (
    analytic_cylinder_conductance, load_tet_mesh, mesh_cylinder,
    solve_neck_conductance, write_cylinder_geo,
)


def main():
    length, radius, conductivity = 1e-3, 0.25e-3, 100.0
    analytic = analytic_cylinder_conductance(conductivity, length, radius)
    rows=[]
    outmesh=MESHES/'explicit_necks'
    for divisor in (2.5,4.0,6.0,8.0):
        stem=f"neck_convergence_d{divisor:g}"
        msh=outmesh/f"{stem}.msh"
        if not msh.exists():
            with tempfile.TemporaryDirectory() as tmp:
                geo=Path(tmp)/f"{stem}.geo"
                write_cylinder_geo(geo,length,radius,radius/divisor)
                mesh_cylinder(geo,msh)
        mesh=load_tet_mesh(msh)
        result=solve_neck_conductance(mesh,conductivity,length)
        rows.append({"radius_over_h":divisor,"analytic_S":analytic,**result,
                     "relative_error":abs(result['conductance_energy_S']-analytic)/analytic})
    output={"rows":rows,"finest_relative_error":rows[-1]['relative_error'],
            "finest_terminal_mismatch":rows[-1]['terminal_mismatch']}
    out=RESULTS/'topology_research'/'explicit_neck_convergence.json'
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

