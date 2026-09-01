#!/usr/bin/env python3
"""Statistical closure of the full-FEM irregular-network ensemble."""
from __future__ import annotations
import csv,json
import numpy as np
from _paths import RESULTS

def z(r):return complex(r['sigma_re_S_m'],r['sigma_im_S_m'])
def main():
 root=RESULTS/'topology_research'/'fem_ensemble';data=json.loads((root/'raw_results.json').read_text())
 g={(r['network'],r['target_angle_deg'],r['frequency_over_f0'],r['state']):r for r in data['rows']}
 rows=[]
 for n in range(6):
  for f in (.1,1.,10.):
   c={};share={}
   for a in (0.,90.):
    full=z(g[(n,a,f,'complete')]);deleted=z(g[(n,a,f,'target_deleted')]);c[a]=abs(full-deleted);share[a]=c[a]/abs(full)
   rows.append({'network':n,'frequency_over_f0':f,'parallel_delta_sigma_abs':c[0.],
    'perpendicular_delta_sigma_abs':c[90.],'perpendicular_over_parallel':c[90.]/c[0.],
    'parallel_over_bulk_percent':100*share[0.],'perpendicular_over_bulk_percent':100*share[90.],
    'suppression_factor_parallel_over_perpendicular':c[0.]/c[90.],
    'electrode_mismatch_max':max(g[(n,a,f,s)]['electrode_mismatch'] for a in (0.,90.) for s in ('complete','target_deleted'))})
 with (root/'ensemble_metrics.csv').open('w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 ratio=np.array([r['perpendicular_over_parallel'] for r in rows]);supp=1/ratio
 byf={}
 for f in (.1,1.,10.):
  q=np.array([r['perpendicular_over_parallel'] for r in rows if r['frequency_over_f0']==f])
  byf[str(f)]={'median_perpendicular_over_parallel':float(np.median(q)),
   'range_perpendicular_over_parallel':[float(q.min()),float(q.max())],
   'median_suppression_factor':float(np.median(1/q))}
 output={'n_networks':6,'n_full_fem_solves':72,'n_network_frequency_pairs':18,
  'overall_median_perpendicular_over_parallel':float(np.median(ratio)),
  'overall_range_perpendicular_over_parallel':[float(ratio.min()),float(ratio.max())],
  'overall_median_suppression_factor':float(np.median(supp)),
  'fraction_below_1_percent':float(np.mean(ratio<.01)),
  'fraction_below_5_percent':float(np.mean(ratio<.05)),
  'max_electrode_mismatch':float(max(r['electrode_mismatch_max'] for r in rows)),
  'by_frequency':byf,'rows':rows}
 (root/'ensemble_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
