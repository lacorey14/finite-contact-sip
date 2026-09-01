#!/usr/bin/env python3
"""Same geometry, one-edge topology contrast measured in two directions."""

from __future__ import annotations

import contextlib
import io
import json
import math

import numpy as np

from _paths import RESULTS
import run_scenario0 as rs
from run_branched_network_angle_sweep import prepare_mesh


HORIZONTAL_ONLY = [(0, 1, 0.02), (1, 2, 0.02)]


def main():
    frequencies = rs.FP * np.logspace(-1, 1, 5)
    rows=[]
    for angle in (0.0, 90.0):
        length, centres, mesh, facets=prepare_mesh(angle)
        for frequency in frequencies:
            with contextlib.redirect_stdout(io.StringIO()):
                _,ur,ui,_=rs.solve(
                    mesh,facets,v_right=rs.V_RIGHT,centres=centres,
                    vim_electrode="dirichlet",component_ids=[0,1,2,3],
                    omega=2*math.pi*float(frequency),contact_edges=HORIZONTAL_ONLY,
                )
            left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side="left")
            right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side="right")
            sigma=0.5*(left+right)
            rows.append({"angle_deg":angle,"frequency_hz":float(frequency),
                         "frequency_over_f0":float(frequency/rs.FP),
                         "sigma_re_S_m":sigma.real,"sigma_im_S_m":sigma.imag,
                         "electrode_mismatch":abs(left-right)/max(abs(sigma),1e-30)})
        print(f"angle={angle:g} horizontal-only done",flush=True)
    outdir=RESULTS/"topology_research"/"branched_network"
    (outdir/"topology_contrast_raw.json").write_text(
        json.dumps({"contacts":HORIZONTAL_ONLY,"rows":rows},indent=2)+"\n")
    print(json.dumps({"output":str(outdir),"n_solves":len(rows)},indent=2))


if __name__=="__main__":main()
