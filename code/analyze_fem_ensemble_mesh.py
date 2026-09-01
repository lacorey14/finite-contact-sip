#!/usr/bin/env python3
"""Base/fine comparison for ensemble endpoint networks."""
from __future__ import annotations
import json
from _paths import RESULTS
def z(r):return complex(r['sigma_re_S_m'],r['sigma_im_S_m'])
def main():
 root=RESULTS/'topology_research'/'fem_ensemble'
 base=json.loads((root/'raw_results.json').read_text())['rows'];fine=json.loads((root/'fine_mesh_raw.json').read_text())['rows']
 b={(r['network'],r['target_angle_deg'],r['state']):r for r in base if r['frequency_over_f0']==1 and r['network'] in (0,5)}
 f={(r['network'],r['angle_deg'],r['state']):r for r in fine};rows=[]
 for n in (0,5):
  values={}
  for name,d in (('base',b),('fine',f)):
   c={a:abs(z(d[(n,a,'complete')])-z(d[(n,a,'target_deleted')])) for a in (0.,90.)}
   values[name]={'parallel':c[0.],'perpendicular':c[90.],'ratio':c[90.]/c[0.]}
  rows.append({'network':n,'base_ratio':values['base']['ratio'],'fine_ratio':values['fine']['ratio'],
   'parallel_relative_change':abs(values['fine']['parallel']-values['base']['parallel'])/values['fine']['parallel'],
   'perpendicular_relative_change':abs(values['fine']['perpendicular']-values['base']['perpendicular'])/values['fine']['perpendicular'],
   'fine_suppression_factor':1/values['fine']['ratio'],
   'max_fine_electrode_mismatch':max(r['electrode_mismatch'] for r in fine if r['network']==n)})
 out={'rows':rows,'minimum_fine_suppression_factor':min(r['fine_suppression_factor'] for r in rows),
  'max_parallel_relative_change':max(r['parallel_relative_change'] for r in rows),
  'max_fine_electrode_mismatch':max(r['max_fine_electrode_mismatch'] for r in rows)}
 (root/'mesh_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
