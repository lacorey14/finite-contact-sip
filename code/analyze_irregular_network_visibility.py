#!/usr/bin/env python3
"""Quantify directional suppression and tensor behavior in irregular-network FEM."""

from __future__ import annotations
import json, math
import numpy as np
from _paths import RESULTS


def z(r):return complex(r['sigma_re_S_m'],r['sigma_im_S_m'])


def main():
    root=RESULTS/'topology_research'/'random_networks'
    data=json.loads((root/'irregular_fem_raw.json').read_text())
    g={(float(r['branch_angle_deg']),float(r['frequency_over_f0']),r['case_id']):r for r in data['rows']}
    rows=[];freq=[]
    for ratio in (.1,1.,10.):
        contrast={a:z(g[a,ratio,'complete'])-z(g[a,ratio,'branch_deleted']) for a in (0.,30.,60.,90.)}
        total={a:z(g[a,ratio,'complete'])-z(g[a,ratio,'disconnected']) for a in contrast}
        # General symmetric 2-D response tensor: calibrate xx, yy and xy at 0, 90, 30; hold out 60.
        xx,yy=contrast[0.],contrast[90.]
        t=math.radians(30); xy=(contrast[30.]-xx*math.cos(t)**2-yy*math.sin(t)**2)/(2*math.sin(t)*math.cos(t))
        t=math.radians(60); predicted=xx*math.cos(t)**2+yy*math.sin(t)**2+2*xy*math.sin(t)*math.cos(t)
        tensor_error=abs(predicted-contrast[60.])/max(abs(contrast[60.]),1e-30)
        freq.append({'frequency_over_f0':ratio,
            'perpendicular_over_parallel_contrast':abs(contrast[90.])/abs(contrast[0.]),
            'parallel_contrast_over_total_contact_effect':abs(contrast[0.])/abs(total[0.]),
            'perpendicular_contrast_over_total_contact_effect':abs(contrast[90.])/abs(total[90.]),
            'held_out_60deg_tensor_relative_error':tensor_error})
        for a in contrast:
            rows.append({'frequency_over_f0':ratio,'branch_angle_deg':a,
                'contrast_abs':abs(contrast[a]),'contrast_over_parallel':abs(contrast[a])/abs(contrast[0.]),
                'cos2_angle':math.cos(math.radians(a))**2,
                'contrast_over_total_contact_effect':abs(contrast[a])/abs(total[a])})
    output={'interpretation':'contrast is the SIP change caused only by deleting edge (1,3)',
        'frequency_metrics':freq,
        'perpendicular_suppression_range':[min(x['perpendicular_over_parallel_contrast'] for x in freq),max(x['perpendicular_over_parallel_contrast'] for x in freq)],
        'tensor_holdout_error_range':[min(x['held_out_60deg_tensor_relative_error'] for x in freq),max(x['held_out_60deg_tensor_relative_error'] for x in freq)],
        'max_electrode_mismatch':max(float(r['electrode_mismatch']) for r in data['rows']),
        'rows':rows}
    (root/'irregular_fem_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
