#!/usr/bin/env python3
"""Compare dimensionlessly scaled contact-response spectra."""
from __future__ import annotations
import csv,json
import numpy as np
from _paths import RESULTS
def z(r):return complex(r['sigma_re_S_m'],r['sigma_im_S_m'])
def main():
 root=RESULTS/'topology_research'/'scale_similarity';new=json.loads((root/'raw_results.json').read_text())['rows']
 base=json.loads((RESULTS/'topology_research'/'fem_ensemble'/'raw_results.json').read_text())['rows']
 g={(r['scale'],r['angle_deg'],r['frequency_over_f0'],r['state']):r for r in new}
 for r in base:
  if r['network']==2:g[(1.,r['target_angle_deg'],r['frequency_over_f0'],r['state'])]=r
 rows=[]
 for s in (.75,1.,1.25):
  for a in (0.,90.):
   for f in (.1,1.,10.):
    delta=z(g[(s,a,f,'complete')])-z(g[(s,a,f,'target_deleted')])
    ref=z(g[(1.,a,f,'complete')])-z(g[(1.,a,f,'target_deleted')])
    rows.append({'scale':s,'angle_deg':a,'frequency_over_f0':f,'delta_re':delta.real,'delta_im':delta.imag,
     'delta_abs':abs(delta),'relative_complex_difference_from_scale1':abs(delta-ref)/abs(ref),
     'electrode_mismatch_max':max(g[(s,a,f,x)]['electrode_mismatch'] for x in ('complete','target_deleted'))})
 with (root/'similarity_metrics.csv').open('w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 parallel=[r for r in rows if r['angle_deg']==0 and r['scale']!=1]
 allx=[r for r in rows if r['scale']!=1]
 out={'scaling_rule':'all lengths x s; frequency / s; contact conductance x s',
  'parallel_response_max_relative_difference':max(r['relative_complex_difference_from_scale1'] for r in parallel),
  'all_response_max_relative_difference':max(r['relative_complex_difference_from_scale1'] for r in allx),
  'all_response_median_relative_difference':float(np.median([r['relative_complex_difference_from_scale1'] for r in allx])),
  'max_electrode_mismatch':max(r['electrode_mismatch_max'] for r in rows),'rows':rows}
 (root/'similarity_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
