#!/usr/bin/env python3
"""Three-particle chain topology and field-orientation spectra."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from _paths import MESHES, RESULTS
from spectral_metrics import log_peak_and_width
from electronic_components import contact_laplacian
import run_scenario0 as rs


def prepare_mesh(orientation: str):
    length, spacing = 70e-3, 15e-3
    centre = 0.5 * length
    offsets = (-spacing, 0.0, spacing)
    if orientation == "parallel":
        centres = [np.array([centre+d, centre, centre]) for d in offsets]
    else:
        centres = [np.array([centre, centre+d, centre]) for d in offsets]
    path = MESHES / f"three_particle_{orientation}_L70.msh"
    if not path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp) / f"three_particle_{orientation}.geo"
            rs.write_geo_n_grain(geo, length, length, length, centres, 0.008, 0.0013)
            if not rs.run_gmsh(geo, path):
                raise RuntimeError("Gmsh failed")
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = centre
    rs.V_RIGHT = rs.E0 * length
    mesh, facets = rs.load_msh_with_tags(path, centres)
    return length, centres, mesh, facets


def main():
    frequencies = rs.FP * np.logspace(-1, 1, 9)
    cases = (
        ("disconnected", [0,1,2], None),
        ("single_edge_0p02", [0,1,2], [(0,1,0.02)]),
        ("weak_chain", [0,1,2], [(0,1,0.002),(1,2,0.002)]),
        ("transition_chain", [0,1,2], [(0,1,0.02),(1,2,0.02)]),
        ("strong_chain", [0,1,2], [(0,1,0.2),(1,2,0.2)]),
        ("bottleneck_chain", [0,1,2], [(0,1,0.2),(1,2,0.002)]),
        ("ideal_pair_plus_isolated", [0,0,1], None),
        ("ideal_cluster", [0,0,0], None),
    )
    rows, summaries = [], []
    for orientation in ("parallel", "transverse"):
        length, centres, mesh, facets = prepare_mesh(orientation)
        for case_id, labels, contacts in cases:
            group=[]
            ncomp=max(labels)+1
            eig=np.linalg.eigvalsh(contact_laplacian(ncomp, contacts)).tolist()
            for frequency in frequencies:
                with contextlib.redirect_stdout(io.StringIO()):
                    _,ur,ui,potentials=rs.solve(
                        mesh,facets,v_right=rs.V_RIGHT,centres=centres,
                        vim_electrode="dirichlet",component_ids=labels,
                        omega=2*math.pi*float(frequency),contact_edges=contacts,
                    )
                left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side="left")
                right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side="right")
                sigma=0.5*(left+right)
                row={
                    "orientation":orientation,"case_id":case_id,
                    "frequency_hz":float(frequency),"frequency_over_f0":float(frequency/rs.FP),
                    "sigma_re_S_m":sigma.real,"sigma_im_S_m":sigma.imag,
                    "phase_deg":float(np.degrees(np.angle(sigma))),
                    "n_components":len(potentials),
                    "max_component_voltage_span_V":max((abs(a-b) for a in potentials for b in potentials),default=0.0),
                    "electrode_mismatch":abs(left-right)/max(abs(sigma),1e-30),
                }
                rows.append(row); group.append(row)
            peak=log_peak_and_width(
                [r["frequency_hz"] for r in group],[r["sigma_im_S_m"] for r in group]
            )
            summaries.append({
                "orientation":orientation,"case_id":case_id,
                "contact_laplacian_eigenvalues_S":eig,**peak,
                "max_electrode_mismatch":max(r["electrode_mismatch"] for r in group),
            })
            print(summaries[-1])
    outdir=RESULTS/'topology_research'/'three_particle_topology'
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'spectra.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={"summaries":summaries,"rows":rows}
    (outdir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"output":str(outdir),"n_solves":len(rows)},indent=2))


if __name__=='__main__': main()
