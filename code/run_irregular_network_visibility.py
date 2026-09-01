#!/usr/bin/env python3
"""Full 3-D SIP test of a deleted dangling contact in an irregular network."""

from __future__ import annotations
import contextlib, io, json, math, tempfile
from pathlib import Path
import numpy as np
from _paths import MESHES, RESULTS
import run_scenario0 as rs

ANGLES=(0.0,30.0,60.0,90.0)  # angle between deleted branch and measurement field
FULL=[(0,1,.02),(1,2,.02),(1,3,.02),(1,4,.02),(2,4,.02)]
DELETED=[e for e in FULL if (e[0],e[1])!=(1,3)]


def centres_for_branch_angle(angle,length=.080):
    local=np.array([[-.015,-.006,0],[0,0,0],[.014,.007,0],[-.004,.015,0],[.012,-.013,0]])
    branch=local[3,:2]-local[1,:2]
    original=math.atan2(branch[1],branch[0])
    rotation=math.radians(angle)-original
    q=np.array([[math.cos(rotation),-math.sin(rotation),0],
                [math.sin(rotation), math.cos(rotation),0],[0,0,1]])
    centre=np.array([length/2]*3)
    return [centre+q@p for p in local]


def prepare(angle):
    length=.080; centres=centres_for_branch_angle(angle,length)
    token=str(angle).replace('.','p'); path=MESHES/f"five_particle_irregular_branch_{token}_L80.msh"
    if not path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            geo=Path(tmp)/'irregular.geo'
            rs.write_geo_n_grain(geo,length,length,length,centres,.009,.0015)
            if not rs.run_gmsh(geo,path):raise RuntimeError('Gmsh failed')
    rs.LX=rs.LY=rs.LZ=length;rs.XC=rs.YC=rs.ZC=length/2;rs.V_RIGHT=rs.E0*length
    mesh,facets=rs.load_msh_with_tags(path,centres)
    return length,centres,mesh,facets


def main():
    rows=[]
    for angle in ANGLES:
        length,centres,mesh,facets=prepare(angle)
        for case,contacts in (('disconnected',None),('complete',FULL),('branch_deleted',DELETED)):
            for ratio in (.1,1.,10.):
                frequency=rs.FP*ratio
                with contextlib.redirect_stdout(io.StringIO()):
                    _,ur,ui,_=rs.solve(mesh,facets,v_right=rs.V_RIGHT,centres=centres,
                        vim_electrode='dirichlet',component_ids=list(range(5)),
                        omega=2*math.pi*frequency,contact_edges=contacts)
                left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side='left')
                right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side='right')
                sigma=.5*(left+right)
                rows.append({'branch_angle_deg':angle,'case_id':case,'frequency_over_f0':ratio,
                    'sigma_re_S_m':sigma.real,'sigma_im_S_m':sigma.imag,
                    'electrode_mismatch':abs(left-right)/max(abs(sigma),1e-30)})
            print(f'angle={angle:g} case={case} done',flush=True)
    out=RESULTS/'topology_research'/'random_networks';out.mkdir(parents=True,exist_ok=True)
    (out/'irregular_fem_raw.json').write_text(json.dumps({'full_contacts':FULL,
        'deleted_contacts':DELETED,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
