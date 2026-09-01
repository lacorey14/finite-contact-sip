#!/usr/bin/env python3
"""Mesh and outer-domain checks for representative contact states."""

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


def make_case(length, gap, h_far, h_grain, stem):
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = 0.5 * length
    rs.V_RIGHT = rs.E0 * length
    x1 = 0.5 * length - 0.5 * gap - rs.R
    x2 = 0.5 * length + 0.5 * gap + rs.R
    centres = [np.array([x1, rs.YC, rs.ZC]), np.array([x2, rs.YC, rs.ZC])]
    mesh_path = MESHES / f"{stem}.msh"
    if not mesh_path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo = Path(tmp) / f"{stem}.geo"
            rs.write_geo_two_grain(geo, length, gap, h_far, h_grain)
            if not rs.run_gmsh(geo, mesh_path):
                raise RuntimeError("Gmsh failed")
    mesh, facets = rs.load_msh_with_tags(mesh_path, centres)
    return centres, mesh, facets


def solve_case(length, centres, mesh, facets, frequency, labels, contacts):
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = 0.5 * length
    rs.V_RIGHT = rs.E0 * length
    with contextlib.redirect_stdout(io.StringIO()):
        _, ur, ui, _ = rs.solve(
            mesh, facets, v_right=rs.V_RIGHT, centres=centres,
            vim_electrode="dirichlet", component_ids=labels,
            omega=2 * math.pi * frequency, contact_edges=contacts,
        )
    left = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="left")
    right = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="right")
    return 0.5 * (left + right), abs(left-right)/abs(0.5*(left+right))


def main():
    gap = 5e-3
    meshes = {
        "base": (47e-3, *make_case(47e-3, gap, 0.006, 0.0012, "connectivity_pilot_L47_d5")),
        "fine": (47e-3, *make_case(47e-3, gap, 0.004, 0.0008, "connectivity_fine_L47_d5")),
        "large": (65e-3, *make_case(65e-3, gap, 0.007, 0.0012, "connectivity_large_L65_d5")),
    }
    cases = (
        ("disconnected", 39.674363833226906, [0,1], None),
        ("contact_0p02S", 16.848208520070816, [0,1], [(0,1,0.02)]),
        ("ideal_connected", 25.31607298536216, [0,0], None),
    )
    rows=[]
    for case_id, frequency, labels, contacts in cases:
        for mesh_id, (length, centres, mesh, facets) in meshes.items():
            sigma, mismatch=solve_case(length, centres, mesh, facets, frequency, labels, contacts)
            volume=length**3
            rows.append({
                "case_id":case_id,"mesh_id":mesh_id,"length_m":length,
                "nodes":int(mesh.nvertices),"elements":int(mesh.nelements),
                "frequency_hz":frequency,"sigma_re_S_m":sigma.real,
                "sigma_im_S_m":sigma.imag,
                "volume_scaled_sigma_im":sigma.imag*volume,
                "electrode_mismatch":mismatch,
            })
    checks=[]
    for case_id, *_ in cases:
        group={r['mesh_id']:r for r in rows if r['case_id']==case_id}
        base,fine,large=group['base'],group['fine'],group['large']
        checks.append({
            "case_id":case_id,
            "fine_vs_base_sigma_im_rel":abs(fine['sigma_im_S_m']-base['sigma_im_S_m'])/abs(fine['sigma_im_S_m']),
            "large_vs_base_volume_scaled_im_rel":abs(large['volume_scaled_sigma_im']-base['volume_scaled_sigma_im'])/abs(large['volume_scaled_sigma_im']),
            "max_electrode_mismatch":max(r['electrode_mismatch'] for r in group.values()),
        })
    result={"rows":rows,"checks":checks}
    out=RESULTS/'topology_research'/'contact_mesh_domain_verification.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"output":str(out),"checks":checks},indent=2))


if __name__=='__main__': main()

