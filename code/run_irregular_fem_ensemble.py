#!/usr/bin/env python3
"""Full 3-D FEM ensemble for directional visibility in irregular networks."""
from __future__ import annotations
import contextlib,io,json,math,tempfile
from pathlib import Path
import numpy as np
from _paths import MESHES,RESULTS
import run_scenario0 as rs

N_NETWORKS=6;ANGLES=(0.,90.);RATIOS=(.1,1.,10.);G=.02;LENGTH=.080

CORE_GRAPHS=(
 ((0,1),(1,2),(2,3),(3,0),(0,2)),
 ((0,1),(1,2),(2,3),(0,2)),
 ((0,1),(0,2),(0,3),(1,2),(2,3)),
 ((0,1),(1,2),(2,3),(0,3),(1,3)),
 ((0,1),(1,2),(2,3),(0,2),(1,3)),
 ((0,1),(0,2),(2,3),(1,3)),
)

def local_geometry(seed):
 rng=np.random.default_rng(4100+seed)
 # junction node 0, three irregular core nodes, target leaf node 4.
 while True:
  core=rng.uniform(-.017,.017,(3,2));pts=np.vstack(([0.,0.],core))
  d=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)+np.eye(4)
  if d.min()>.0108 and np.max(np.linalg.norm(pts,axis=1))<.021:break
 angle=rng.uniform(0,2*math.pi);length=rng.uniform(.012,.018)
 target=length*np.array([math.cos(angle),math.sin(angle)])
 # require leaf not overlapping the core.
 if np.min(np.linalg.norm(pts-target,axis=1)[1:])<.0108:return local_geometry(seed+100)
 return np.vstack((pts,target))

def rotated_centres(network,angle):
 local=local_geometry(network);v=local[4]-local[0]
 rotation=math.radians(angle)-math.atan2(v[1],v[0]);q=np.array([[math.cos(rotation),-math.sin(rotation)],
  [math.sin(rotation),math.cos(rotation)]])
 xy=local@q.T+LENGTH/2
 return [np.array([x,y,LENGTH/2]) for x,y in xy]

def contacts(network,deleted=False):
 e=list(CORE_GRAPHS[network]);
 if not deleted:e.append((0,4))
 return [(i,j,G) for i,j in e]

def prepare(network,angle):
 centres=rotated_centres(network,angle);token=int(angle)
 path=MESHES/f'irregular_ensemble_n{network:02d}_a{token:02d}.msh'
 if not path.exists():
  with tempfile.TemporaryDirectory() as tmp:
   geo=Path(tmp)/'network.geo';rs.write_geo_n_grain(geo,LENGTH,LENGTH,LENGTH,centres,.009,.0015)
   if not rs.run_gmsh(geo,path):raise RuntimeError('Gmsh failed')
 rs.LX=rs.LY=rs.LZ=LENGTH;rs.XC=rs.YC=rs.ZC=LENGTH/2;rs.V_RIGHT=rs.E0*LENGTH
 mesh,facets=rs.load_msh_with_tags(path,centres);return centres,mesh,facets,path

def main():
 rows=[];geometries=[]
 for n in range(N_NETWORKS):
  geometries.append({'network':n,'local_points_m':local_geometry(n).tolist(),
   'core_edges':[list(e) for e in CORE_GRAPHS[n]],'target_edge':[0,4]})
  for angle in ANGLES:
   centres,mesh,facets,path=prepare(n,angle)
   for state,edges in (('complete',contacts(n)),('target_deleted',contacts(n,True))):
    for ratio in RATIOS:
     with contextlib.redirect_stdout(io.StringIO()):
      _,ur,ui,_=rs.solve(mesh,facets,v_right=rs.V_RIGHT,centres=centres,
       vim_electrode='dirichlet',component_ids=list(range(5)),omega=2*math.pi*rs.FP*ratio,
       contact_edges=edges)
     left=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=LENGTH,side='left')
     right=rs.extract_sigma_electrode(mesh,ur,ui,e0=rs.E0,L=LENGTH,side='right');sigma=.5*(left+right)
     rows.append({'network':n,'target_angle_deg':angle,'state':state,'frequency_over_f0':ratio,
      'sigma_re_S_m':sigma.real,'sigma_im_S_m':sigma.imag,
      'electrode_mismatch':abs(left-right)/max(abs(sigma),1e-30),'mesh':str(path)})
   print(f'network={n} angle={angle:g} done',flush=True)
 out=RESULTS/'topology_research'/'fem_ensemble';out.mkdir(parents=True,exist_ok=True)
 (out/'raw_results.json').write_text(json.dumps({'geometries':geometries,'rows':rows},indent=2)+'\n')
 print(json.dumps({'output':str(out),'n_solves':len(rows)},indent=2))
if __name__=='__main__':main()
