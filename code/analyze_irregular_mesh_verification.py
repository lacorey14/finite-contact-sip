#!/usr/bin/env python3
"""Base/fine comparison for irregular-network directional suppression."""
from __future__ import annotations
import json
from _paths import RESULTS

def z(r):return complex(r['sigma_re_S_m'],r['sigma_im_S_m'])
def main():
 root=RESULTS/'topology_research'/'random_networks'
 raw=json.loads((root/'irregular_fem_raw.json').read_text())['rows']
 fine=json.loads((root/'irregular_fine_mesh.json').read_text())['rows']
 b={(r['branch_angle_deg'],r['case_id']):r for r in raw if r['frequency_over_f0']==1.0 and r['branch_angle_deg'] in (0.,90.)}
 f={(r['branch_angle_deg'],r['case_id']):r for r in fine}
 def contrast(d,a):return z(d[(a,'complete')])-z(d[(a,'branch_deleted')])
 bc={a:contrast(b,a) for a in (0.,90.)};fc={a:contrast(f,a) for a in (0.,90.)}
 out={'base_perpendicular_over_parallel':abs(bc[90.])/abs(bc[0.]),
      'fine_perpendicular_over_parallel':abs(fc[90.])/abs(fc[0.]),
      'parallel_contrast_relative_change':abs(fc[0.]-bc[0.])/abs(fc[0.]),
      'perpendicular_contrast_relative_change':abs(fc[90.]-bc[90.])/abs(fc[90.]),
      'max_fine_electrode_mismatch':max(r['electrode_mismatch'] for r in fine)}
 (root/'irregular_mesh_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
