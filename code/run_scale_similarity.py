#!/usr/bin/env python3
"""Dimensionless scale-similarity test for the directional contact response."""
from __future__ import annotations
import contextlib,io,json,math,tempfile
from pathlib import Path
import numpy as np
from _paths import MESHES,RESULTS
import run_scenario0 as rs
from run_irregular_fem_ensemble import local_geometry,CORE_GRAPHS

NETWORK=2;SCALES=(.75,1.25);RATIOS=(.1,1.,10.)

def prepare(scale,angle):
 length=.080*scale;local=local_geometry(NETWORK)*scale;v=local[4]-local[0]
 rot=math.radians(angle)-math.atan2(v[1],v[0]);q=np.array([[math.cos(rot),-math.sin(rot)],[math.sin(rot),math.cos(rot)]])
 xy=local@q.T+length/2;centres=[np.array([x,y,length/2]) for x,y in xy]
 rs.R=.005*scale;rs.LX=rs.LY=rs.LZ=length;rs.XC=rs.YC=rs.ZC=length/2;rs.V_RIGHT=rs.E0*length
 token=str(scale).replace('.','p');path=MESHES/f'scale_similarity_s{token}_a{int(angle):02d}.msh'
 if not path.exists():
  with tempfile.TemporaryDirectory() as tmp:
   geo=Path(tmp)/'scale.geo';rs.write_geo_n_grain(geo,length,length,length,centres,.009*scale,.0015*scale)
   if not rs.run_gmsh(geo,path):raise RuntimeError('Gmsh failed')
 mesh,facets=rs.load_msh_with_tags(path,centres);return length,centres,mesh,facets,path

def edges(scale,deleted):
 e=list(CORE_GRAPHS[NETWORK]);
 if not deleted:e.append((0,4))
 return [(i,j,.02*scale) for i,j in e]

def main():
 rows=[]
 for scale in SCALES:
  for angle in (0.,90.):
   length,centres,mesh,facets,path=prepare(scale,angle)
   f0=rs.SIGMA0/(math.pi*rs.R*rs.C0)
   for state,deleted in (('complete',False),('target_deleted',True)):
    for ratio in RATIOS:
     with contextlib.redirect_stdout(io.StringIO()):
      _,ur,ui,_=rs.solve(mesh,facets,v_right=rs.V_RIGHT,centres=centres,
       vim_electrode='dirichlet',component_ids=list(range(5)),omega=2*math.pi*f0*ratio,
       contact_edges=edges(scale,deleted))
     left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side='left')
     right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=length,side='right');sigma=.5*(left+right)
     rows.append({'scale':scale,'particle_radius_m':rs.R,'domain_length_m':length,'angle_deg':angle,
      'state':state,'frequency_over_f0':ratio,'frequency_hz':f0*ratio,'contact_conductance_S':.02*scale,
      'sigma_re_S_m':sigma.real,'sigma_im_S_m':sigma.imag,
      'electrode_mismatch':abs(left-right)/max(abs(sigma),1e-30),'mesh':str(path)})
   print(f'scale={scale:g} angle={angle:g} done',flush=True)
 rs.R=.005
 out=RESULTS/'topology_research'/'scale_similarity';out.mkdir(parents=True,exist_ok=True)
 (out/'raw_results.json').write_text(json.dumps({'network':NETWORK,
  'scaling_rule':'lengths x s; frequency / s; contact conductance x s','rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
