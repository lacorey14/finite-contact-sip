#!/usr/bin/env python3
"""RVE-scale screening of when contact direction survives statistical averaging.

The reduced model is the Schur-complement graph analogue of the full FEM:

    (L_G + i omega C) v_m = i omega C phi_e,

where phi_e is the background electrolyte potential at each particle centre.
The induced interfacial charge is q = C(phi_e-v_m).  Solving the three Cartesian
right-hand sides gives a complex polarization tensor for each realization.

This is a topology/scale screening model, not a replacement for full electrolyte
FEM.  Selected ensemble endpoints are validated separately with the full model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

from _paths import RESULTS


OUT = RESULTS / "topology_research" / "rve_directional_scaling"
FREQUENCIES = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class Design:
    family: str
    L: int
    p: float
    fabric: float
    replicate: int


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.sz = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = int(self.p[x])
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def candidate_edges(L: int):
    """Nearest-neighbour bonds of an open L x L x L cubic lattice."""
    ids = np.arange(L**3, dtype=np.int32).reshape(L, L, L)
    edges, axes = [], []
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(0, L - 1)
        b[axis] = slice(1, L)
        u = ids[tuple(a)].ravel()
        v = ids[tuple(b)].ravel()
        edges.append(np.column_stack((u, v)))
        axes.append(np.full(len(u), axis, dtype=np.int8))
    return np.vstack(edges), np.concatenate(axes)


def occupation_probabilities(p: float, fabric: float):
    """Preserve mean bond occupancy while biasing bonds toward z."""
    probs = np.array([p * (1.0 - fabric), p * (1.0 - fabric), p * (1.0 + 2.0 * fabric)])
    return np.clip(probs, 0.0, 1.0)


def build_network(design: Design, seed: int):
    rng = np.random.default_rng(seed)
    cand, axes = candidate_edges(design.L)
    probs = occupation_probabilities(design.p, design.fabric)
    keep = rng.random(len(cand)) < probs[axes]
    edges = cand[keep]
    edge_axes = axes[keep]
    # Unit median and moderate natural variability; independent of orientation.
    g = np.exp(rng.normal(0.0, 0.65, len(edges)))
    return edges, edge_axes, g, probs


def laplacian(n: int, edges: np.ndarray, g: np.ndarray):
    if len(edges) == 0:
        return coo_matrix((n, n), dtype=float).tocsc()
    u, v = edges[:, 0], edges[:, 1]
    rows = np.concatenate((u, v, u, v))
    cols = np.concatenate((u, v, v, u))
    vals = np.concatenate((g, g, -g, -g))
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def coordinates(L: int):
    q = np.indices((L, L, L), dtype=float).reshape(3, -1).T
    q -= 0.5 * (L - 1)
    # Normalize each axis to unit root-mean-square coordinate so response axes
    # are directly comparable across system size.
    q /= np.sqrt(np.mean(q[:, 0] ** 2))
    return q


def components_and_spanning(L: int, edges: np.ndarray):
    n = L**3
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(int(u), int(v))
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    _, counts = np.unique(roots, return_counts=True)
    largest = float(counts.max() / n)
    ids = np.arange(n, dtype=np.int32).reshape(L, L, L)
    spans = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = 0
        hi[axis] = L - 1
        rlo = set(roots[ids[tuple(lo)].ravel()].tolist())
        rhi = set(roots[ids[tuple(hi)].ravel()].tolist())
        spans.append(bool(rlo.intersection(rhi)))
    return largest, spans


def tensor_response(L: int, edges: np.ndarray, g: np.ndarray, omega: float):
    n = L**3
    x = coordinates(L)
    lg = laplacian(n, edges, g)
    a = (lg + 1j * omega * eye(n, format="csc")).tocsc()
    v = splu(a).solve((1j * omega * x).astype(complex))
    q = x - v
    denom = np.diag(x.T @ x)
    tensor = (x.T @ q) / denom[:, None]
    diag = np.abs(np.diag(tensor))

    # Contact-current concentration. A small effective number indicates that a
    # few contacts dominate and statistical self-averaging is weak.
    neff = []
    top01 = []
    if len(edges):
        u, w = edges[:, 0], edges[:, 1]
        for k in range(3):
            e = g * np.abs(v[u, k] - v[w, k]) ** 2
            s = float(e.sum())
            if s <= 1e-30:
                neff.append(0.0)
                top01.append(0.0)
                continue
            neff.append(float(s * s / np.sum(e * e)))
            m = max(1, int(np.ceil(0.01 * len(e))))
            top01.append(float(np.partition(e, -m)[-m:].sum() / s))
    else:
        neff = [0.0] * 3
        top01 = [0.0] * 3
    return tensor, diag, neff, top01


def realized_fabric(edge_axes: np.ndarray, g: np.ndarray):
    if len(edge_axes) == 0:
        return np.nan
    wz = float(g[edge_axes == 2].sum() / g.sum())
    return 0.5 * (3.0 * wz - 1.0)


def run_design(d: Design, master_seed: int):
    seed = int(master_seed + 1_000_003 * d.L + 10_007 * d.replicate + round(1000 * d.p) * 101 + round(1000 * d.fabric))
    edges, edge_axes, g, probs = build_network(d, seed)
    largest, spans = components_and_spanning(d.L, edges)
    base = {
        "family": d.family,
        "L": d.L,
        "n_nodes": d.L**3,
        "p_mean": d.p,
        "fabric_input": d.fabric,
        "replicate": d.replicate,
        "seed": seed,
        "p_x": float(probs[0]),
        "p_y": float(probs[1]),
        "p_z": float(probs[2]),
        "n_edges": int(len(edges)),
        "fabric_realized": float(realized_fabric(edge_axes, g)),
        "largest_component_fraction": largest,
        "span_x": spans[0],
        "span_y": spans[1],
        "span_z": spans[2],
    }
    rows = []
    for omega in FREQUENCIES:
        tensor, diag, neff, top01 = tensor_response(d.L, edges, g, omega)
        mean = float(diag.mean())
        row = dict(base)
        row.update({
            "omega_over_GC": omega,
            "response_x": float(diag[0]),
            "response_y": float(diag[1]),
            "response_z": float(diag[2]),
            "response_mean": mean,
            "anisotropy_cv": float(np.std(diag) / max(mean, 1e-30)),
            "min_over_max": float(diag.min() / max(diag.max(), 1e-30)),
            "max_over_min": float(diag.max() / max(diag.min(), 1e-30)),
            "neff_x": neff[0], "neff_y": neff[1], "neff_z": neff[2],
            "neff_fraction_mean": float(np.mean(neff) / max(len(edges), 1)),
            "top1pct_energy_mean": float(np.mean(top01)),
            "tensor_re": np.real(tensor).tolist(),
            "tensor_im": np.imag(tensor).tolist(),
        })
        rows.append(row)
    return rows


def designs(replicates: int):
    out = []
    # Focused size series for statistical self-averaging in isotropic networks.
    for L in (4, 6, 8, 10, 12, 16):
        for p in (0.18, 0.25, 0.32, 0.45, 0.65):
            for r in range(replicates):
                out.append(Design("size", L, p, 0.0, r))
    # Fabric/percolation phase map at a representative RVE size.
    for p in (0.18, 0.25, 0.32, 0.45, 0.65):
        for fabric in (0.0, 0.2, 0.4, 0.6):
            if fabric == 0.0:
                continue  # already present in size family at L=12
            for r in range(replicates):
                out.append(Design("fabric", 12, p, fabric, r))
    return out


def validate_model():
    # No contacts -> no contact-network polarization.
    _, d0, _, _ = tensor_response(4, np.empty((0, 2), dtype=int), np.empty(0), 1.0)
    if np.max(d0) > 1e-12:
        raise AssertionError(f"disconnected limit failed: {d0}")
    # Fully occupied, equal-conductance cube is exactly permutation symmetric.
    e, _, = candidate_edges(4)
    _, d1, _, _ = tensor_response(4, e, np.ones(len(e)), 1.0)
    if np.ptp(d1) / np.mean(d1) > 1e-10:
        raise AssertionError(f"isotropic full-lattice limit failed: {d1}")
    return {"disconnected_max": float(np.max(d0)), "full_lattice_axis_spread": float(np.ptp(d1) / np.mean(d1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=24)
    ap.add_argument("--seed", type=int, default=73129)
    ap.add_argument("--limit", type=int, default=0, help="debug: run first N designs")
    args = ap.parse_args()
    validation = validate_model()
    ds = designs(args.replicates)
    if args.limit:
        ds = ds[: args.limit]
    rows = []
    for i, d in enumerate(ds, 1):
        rows.extend(run_design(d, args.seed))
        if i % 50 == 0 or i == len(ds):
            print(f"{i}/{len(ds)} networks; {len(rows)} frequency rows", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "3-D capacitive-particle finite-contact graph RVE",
        "equation": "(L_G + i*omega*C)v = i*omega*C*phi_e",
        "frequencies": FREQUENCIES,
        "replicates": args.replicates,
        "validation": validation,
        "rows": rows,
    }
    (OUT / "raw_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "networks": len(ds), "rows": len(rows), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
