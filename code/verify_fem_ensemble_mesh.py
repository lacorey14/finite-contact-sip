#!/usr/bin/env python3
"""Fine-mesh anchors for weakly and strongly suppressed ensemble networks."""
from __future__ import annotations
import contextlib,io,json,math,tempfile
from pathlib import Path
from _paths import MESHES,RESULTS
import run_scenario0 as rs
from run_irregular_fem_ensemble import rotated_centres,contacts,LENGTH

def main():
 rows=[]
 for n in (0,5):
  for angle in (0.,90.):
   centres=rotated_centres(n,angle);path=MESHES/f'irregular_ensemble_n{n:02d}_a{int(angle):02d}_fine.msh'
   if not path.exists():
    with tempfile.TemporaryDirectory() as tmp:
     geo=Path(tmp)/'fine.geo';rs.write_geo_n_grain(geo,LENGTH,LENGTH,LENGTH,centres,.007,.00105)
     if not rs.run_gmsh(geo,path):raise RuntimeError('Gmsh failed')
   rs.R=.005;rs.LX=rs.LY=rs.LZ=LENGTH;rs.XC=rs.YC=rs.ZC=LENGTH/2;rs.V_RIGHT=rs.E0*LENGTH
   mesh,facets=rs.load_msh_with_tags(path,centres)
   for state,edges in (('complete',contacts(n)),('target_deleted',contacts(n,True))):
    with contextlib.redirect_stdout(io.StringIO()):
     _,ur,ui,_=rs.solve(mesh,facets,v_right=rs.V_RIGHT,centres=centres,
      vim_electrode='dirichlet',component_ids=list(range(5)),omega=2*math.pi*rs.FP,
      contact_edges=edges)
    left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=LENGTH,side='left')
    right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=LENGTH,side='right');sigma=.5*(left+right)
    rows.append({'network':n,'angle_deg':angle,'state':state,'frequency_over_f0':1.,
     'sigma_re_S_m':sigma.real,'sigma_im_S_m':sigma.imag,
     'electrode_mismatch':abs(left-right)/max(abs(sigma),1e-30),'mesh':str(path)})
   print(f'fine network={n} angle={angle:g} done',flush=True)
 out=RESULTS/'topology_research'/'fem_ensemble'/'fine_mesh_raw.json'
 out.write_text(json.dumps({'networks':[0,5],'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
