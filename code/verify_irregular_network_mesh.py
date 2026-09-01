#!/usr/bin/env python3
"""Fine-mesh verification of irregular-network directional suppression."""

from __future__ import annotations
import contextlib,io,json,math,tempfile
from pathlib import Path
from _paths import MESHES,RESULTS
import run_scenario0 as rs
from run_irregular_network_visibility import FULL,DELETED,centres_for_branch_angle


def main():
    length=.080;rows=[]
    for angle in (0.,90.):
        centres=centres_for_branch_angle(angle,length)
        token=str(angle).replace('.','p');path=MESHES/f'five_particle_irregular_branch_{token}_fine_L80.msh'
        if not path.exists():
            with tempfile.TemporaryDirectory() as tmp:
                geo=Path(tmp)/'fine.geo';rs.write_geo_n_grain(geo,length,length,length,centres,.007,.00105)
                if not rs.run_gmsh(geo,path):raise RuntimeError('Gmsh failed')
        rs.LX=rs.LY=rs.LZ=length;rs.XC=rs.YC=rs.ZC=length/2;rs.V_RIGHT=rs.E0*length
        mesh,facets=rs.load_msh_with_tags(path,centres)
        for case,contacts in (('complete',FULL),('branch_deleted',DELETED)):
            with contextlib.redirect_stdout(io.StringIO()):
                _,ur,ui,_=rs.solve(mesh,facets,v_right=rs.V_RIGHT,centres=centres,
                    vim_electrode='dirichlet',component_ids=list(range(5)),omega=2*math.pi*rs.FP,
                    contact_edges=contacts)
            left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side='left')
            right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side='right');sigma=.5*(left+right)
            rows.append({'branch_angle_deg':angle,'case_id':case,'sigma_re_S_m':sigma.real,
                'sigma_im_S_m':sigma.imag,'electrode_mismatch':abs(left-right)/max(abs(sigma),1e-30)})
            print(f'fine angle={angle:g} case={case} done',flush=True)
    out=RESULTS/'topology_research'/'random_networks'/'irregular_fine_mesh.json'
    out.write_text(json.dumps({'frequency_over_f0':1.0,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
