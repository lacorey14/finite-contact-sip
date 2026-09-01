#!/usr/bin/env python3
"""Monte Carlo test of directional contact visibility in embedded networks."""

from __future__ import annotations

import json
import math

import numpy as np

from _paths import RESULTS


def connected(n, edges):
    seen={0}; changed=True
    while changed:
        changed=False
        for i,j,_ in edges:
            if i in seen and j not in seen: seen.add(j); changed=True
            if j in seen and i not in seen: seen.add(i); changed=True
    return len(seen)==n


def random_network(rng, n=12):
    points=rng.uniform(-1,1,(n,2))
    # k-nearest candidates, then retain a random connected graph.
    pairs=[]
    for i in range(n):
        for j in range(i+1,n):
            d=float(np.linalg.norm(points[i]-points[j])); pairs.append((d,i,j))
    pairs.sort()
    edges=[]; parent=list(range(n))
    def root(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    # Randomized Euclidean minimum spanning backbone.
    for d,i,j in sorted(pairs,key=lambda x:x[0]*rng.lognormal(0,0.35)):
        a,b=root(i),root(j)
        if a!=b:
            parent[a]=b; edges.append((i,j,float(rng.lognormal(0,1))))
        if len(edges)==n-1: break
    # Add local loops.
    existing={tuple(sorted((i,j))) for i,j,_ in edges}
    for d,i,j in pairs[:3*n]:
        if (i,j) not in existing and rng.random()<0.22:
            edges.append((i,j,float(rng.lognormal(0,1))))
            existing.add((i,j))
    assert connected(n,edges)
    return points,edges


def edge_fractions(points, edges, directions):
    values=[]
    for i,j,g in edges:
        delta=points[j]-points[i]
        values.append([g*float(np.dot(delta,e))**2 for e in directions])
    values=np.asarray(values)
    totals=np.maximum(values.sum(axis=0),1e-30)
    return values/totals


def main():
    rng=np.random.default_rng(20260813)
    angles=np.radians(np.arange(0,180,15))
    directions=np.column_stack((np.cos(angles),np.sin(angles)))
    all_single=[]; all_best_two=[]; all_xy=[]; all_geometric=[]; network_rows=[]
    for realization in range(2000):
        points,edges=random_network(rng)
        frac=edge_fractions(points,edges,directions)
        single=frac[:,0]
        geometric=[]
        for i,j,g in edges:
            d=points[j]-points[i]
            geometric.append(float(np.dot(d,directions[0])**2/np.dot(d,d)))
        # Equal-budget two orthogonal measurements: sum raw visibility first.
        xy_raw=[]
        for i,j,g in edges:
            d=points[j]-points[i]; xy_raw.append(g*float(d@d))
        xy=np.asarray(xy_raw); xy/=xy.sum()
        # Best direction among the 12 tested for each edge (diagnostic upper bound).
        best=frac.max(axis=1)
        all_single.extend(single.tolist()); all_xy.extend(xy.tolist()); all_best_two.extend(best.tolist())
        all_geometric.extend(geometric)
        network_rows.append({
            "realization":realization,"n_edges":len(edges),
            "single_edges_below_0p1pct":int(np.sum(single<1e-3)),
            "xy_edges_below_0p1pct":int(np.sum(xy<1e-3)),
            "minimum_single_fraction":float(single.min()),
            "minimum_xy_fraction":float(xy.min()),
        })
    def stats(values):
        a=np.asarray(values)
        return {"n_edges":int(a.size),"median_fraction":float(np.median(a)),
                "p05_fraction":float(np.quantile(a,.05)),
                "fraction_below_0p1pct":float(np.mean(a<1e-3)),
                "fraction_below_1pct":float(np.mean(a<1e-2))}
    output={
        "seed":20260813,"n_networks":2000,"nodes_per_network":12,
        "metric":"edge contribution G[e dot (rj-ri)]^2 divided by network total",
        "direction_only_single_measurement":{
            "metric":"cos^2 of edge angle to measurement field; independent of edge length and conductance",
            "fraction_below_1pct_of_own_max":float(np.mean(np.asarray(all_geometric)<.01)),
            "fraction_below_10pct_of_own_max":float(np.mean(np.asarray(all_geometric)<.1)),
            "two_orthogonal_sum_relative_to_own_max":1.0,
        },
        "single_direction":stats(all_single),
        "two_orthogonal_directions":stats(all_xy),
        "best_of_12_directions_diagnostic":stats(all_best_two),
        "network_fraction_with_at_least_one_edge_below_0p1pct_single":float(np.mean([r["single_edges_below_0p1pct"]>0 for r in network_rows])),
        "network_fraction_with_at_least_one_edge_below_0p1pct_xy":float(np.mean([r["xy_edges_below_0p1pct"]>0 for r in network_rows])),
        "network_rows":network_rows,
    }
    outdir=RESULTS/"topology_research"/"random_networks";outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"visibility_ensemble.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k!="network_rows"},indent=2))


if __name__=="__main__":main()
