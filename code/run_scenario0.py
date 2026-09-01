#!/usr/bin/env python3
"""
方案 D — Scenario 0 native FEM (scikit-fem), Mac, no Docker.

Improvements vs first MVP:
  - Gmsh geo tags left/right/grain/insul + fine MeshSize on grain surface
  - Facet groups from physical tags when available
  - Bulk σ*_eff from volume-mean J* = -σ₀ ∇V*
  - Optional --two-grain for H1-like cross-check scaffold

Usage:
  export PATH="/opt/anaconda3/bin:$PATH"
  .venv/bin/python FEM_IP/run_scenario0.py --h-far 0.01 --h-grain 0.0008
  .venv/bin/python FEM_IP/run_scenario0.py --two-grain --d-mm 5 --L-mm 47
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve

import _paths  # noqa: E402
from _paths import PKG, PROJECT, MESHES, RESULTS
from electronic_components import combine_facet_groups, contact_laplacian

ROOT = PROJECT  # optional OF / paper assets in parent repo

R = 5e-3
SIGMA0 = 0.2
C0 = 0.4
E0 = 50.0
OMEGA = 154.48591479
F_HZ = OMEGA / (2 * math.pi)
FP = SIGMA0 / (math.pi * R * C0)

# Scenario 0 defaults (Izumoto baseCase blockMesh: 0.1 × 0.2 × 0.2 m)
XC, YC, ZC = 0.05, 0.1, 0.1
LX, LY, LZ = 0.1, 0.2, 0.2
V_LEFT, V_RIGHT = 0.0, 5.0

TAG = {"left": 1, "right": 2, "insul": 3, "grain": 4, "volume": 10}


def feng_along_x(x: np.ndarray, x_center: float = XC):
    ei = E0 * (FP - 2j * F_HZ) / (2 * (FP + 1j * F_HZ))
    re = np.full_like(x, np.nan, dtype=float)
    im = np.full_like(x, np.nan, dtype=float)
    for i, xi in enumerate(x):
        rr = abs(xi - x_center)
        if rr < R:
            continue
        ct = 1.0 if xi >= x_center else -1.0
        phi_raw = -(E0 * rr + ei * R**3 / rr**2) * ct
        v = -phi_raw + E0 * x_center
        re[i], im[i] = v.real, v.imag
    return re, im


def feng_sigma_eff_single(phi: float) -> complex:
    """Feng MG at volume fraction phi (single-inclusion limit used for Scenario 0 check)."""
    from effective_medium import feng_maxwell_garnett

    em = feng_maxwell_garnett(np.array([F_HZ]), phi, SIGMA0, R, C0)
    return complex(em.sigma_eff[0])


def write_geo_scenario0(path: Path, h_far: float, h_grain: float) -> None:
    # Gmsh 4.11-compatible: Ball size field + boolean; classify facets in Python.
    path.write_text(
        f"""
SetFactory("OpenCASCADE");
Box(1) = {{0, 0, 0, {LX}, {LY}, {LZ}}};
Sphere(2) = {{{XC}, {YC}, {ZC}, {R}}};
BooleanDifference(3) = {{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }};
Physical Volume({TAG["volume"]}) = {{3}};

Field[1] = Ball;
Field[1].XCenter = {XC};
Field[1].YCenter = {YC};
Field[1].ZCenter = {ZC};
Field[1].Radius = {3.5 * R};
Field[1].VIn = {h_grain};
Field[1].VOut = {h_far};
Background Field = 1;
Mesh.CharacteristicLengthMax = {h_far};
Mesh.CharacteristicLengthMin = {h_grain};
Mesh 3;
"""
    )


def write_geo_n_grain(
    path: Path,
    Lx: float,
    Ly: float,
    Lz: float,
    centres: list,
    h_far: float,
    h_grain: float,
    *,
    refine_radius: float | None = None,
) -> None:
    """Rectangular box with N spherical cavities (centres as array-like)."""
    import numpy as np

    cents = [np.asarray(c, dtype=float).ravel() for c in centres]
    if not cents:
        raise ValueError("centres empty")
    # Span of chain along x for clearance-aware refine radius
    xs = [float(c[0]) for c in cents]
    clear_x = min(min(xs) - R, Lx - (max(xs) + R))
    if refine_radius is None:
        refine_radius = min(3.5 * R, max(1.25 * R, R + max(clear_x, 0.0)))
    sphere_block = "\n".join(
        f"Sphere({i + 2}) = {{{c[0]}, {c[1]}, {c[2]}, {R}}};" for i, c in enumerate(cents)
    )
    vol_ids = ", ".join(str(i + 2) for i in range(len(cents)))
    field_blocks = []
    for i, c in enumerate(cents):
        fid = i + 1
        field_blocks.append(
            f"""
Field[{fid}] = Ball;
Field[{fid}].XCenter = {c[0]};
Field[{fid}].YCenter = {c[1]};
Field[{fid}].ZCenter = {c[2]};
Field[{fid}].Radius = {refine_radius};
Field[{fid}].VIn = {h_grain};
Field[{fid}].VOut = {h_far};"""
        )
    n_f = len(cents)
    min_id = n_f + 1
    fields_list = ", ".join(str(i + 1) for i in range(n_f))
    path.write_text(
        f"""
SetFactory("OpenCASCADE");
Box(1) = {{0, 0, 0, {Lx}, {Ly}, {Lz}}};
{sphere_block}
BooleanDifference({n_f + 2}) = {{ Volume{{1}}; Delete; }}{{ Volume{{{vol_ids}}}; Delete; }};
Physical Volume({TAG["volume"]}) = {{{n_f + 2}}};
{"".join(field_blocks)}
Field[{min_id}] = Min;
Field[{min_id}].FieldsList = {{{fields_list}}};
Background Field = {min_id};
Mesh.CharacteristicLengthMax = {h_far};
Mesh.CharacteristicLengthMin = {h_grain};
Mesh 3;
"""
    )


def write_geo_two_grain(
    path: Path,
    L: float,
    d: float,
    h_far: float,
    h_grain: float,
    *,
    refine_radius: float | None = None,
) -> None:
    x1 = 0.5 * L - 0.5 * d - R
    x2 = 0.5 * L + 0.5 * d + R
    yc = zc = 0.5 * L
    write_geo_n_grain(
        path,
        L,
        L,
        L,
        [[x1, yc, zc], [x2, yc, zc]],
        h_far,
        h_grain,
        refine_radius=refine_radius,
    )


def run_gmsh(geo_path: Path, msh_path: Path) -> bool:
    gmsh_bin = shutil.which("gmsh") or "/opt/anaconda3/bin/gmsh"
    if not Path(gmsh_bin).exists():
        print("gmsh CLI not found")
        return False
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"}
    cmd = [gmsh_bin, str(geo_path), "-3", "-format", "msh2", "-o", str(msh_path)]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return False
    print(f"Gmsh mesh → {msh_path}")
    return msh_path.exists()


def load_msh_with_tags(mesh_path: Path, grain_centres: list[np.ndarray] | None = None):
    """Load tet mesh; classify boundary facets geometrically."""
    import meshio
    from skfem.io.meshio import from_meshio

    raw = meshio.read(str(mesh_path))
    tet_i = next(i for i, c in enumerate(raw.cells) if c.type.startswith("tetra"))
    mesh = from_meshio(
        meshio.Mesh(points=raw.points[:, :3], cells=[("tetra", raw.cells[tet_i].data)])
    )

    centres = grain_centres or [np.array([XC, YC, ZC])]
    Lx = float(mesh.p[0].max() - mesh.p[0].min())
    Ly = float(mesh.p[1].max() - mesh.p[1].min())
    Lz = float(mesh.p[2].max() - mesh.p[2].min())
    x0, y0, z0 = mesh.p[0].min(), mesh.p[1].min(), mesh.p[2].min()
    wall = 1e-7 + 1e-4 * max(Lx, Ly, Lz)

    facets = mesh.boundary_facets()
    mids = mesh.p[:, mesh.facets[:, facets]].mean(axis=1).T
    edge = np.linalg.norm(
        mesh.p[:, mesh.facets[0, facets[: min(50, len(facets))]]]
        - mesh.p[:, mesh.facets[1, facets[: min(50, len(facets))]]],
        axis=0,
    )
    grain_tol = max(0.35 * R, 1.2 * float(np.mean(edge))) if len(edge) else 0.5 * R

    groups = {"left": [], "right": [], "insul": [], "grain": []}
    for fi, mid in zip(facets, mids):
        x, y, z = mid
        if x <= x0 + wall:
            groups["left"].append(fi)
        elif x >= x0 + Lx - wall:
            groups["right"].append(fi)
        elif y <= y0 + wall or y >= y0 + Ly - wall or z <= z0 + wall or z >= z0 + Lz - wall:
            groups["insul"].append(fi)
        else:
            # Triangulated sphere face centroids lie slightly inside r<R
            if min(np.linalg.norm(mid - c) for c in centres) < R + grain_tol:
                groups["grain"].append(fi)
            else:
                groups["insul"].append(fi)

    out = {k: np.asarray(v, dtype=int) for k, v in groups.items() if len(v)}
    for k, v in out.items():
        print(f"  facets {k}: {len(v)}")
    return mesh, out


def split_grain_facets(mesh, grain_facets: np.ndarray, centres: list[np.ndarray]) -> list[np.ndarray]:
    """Assign each grain facet to nearest sphere centre (Izumoto: per-grain mean)."""
    if len(centres) == 1:
        return [grain_facets]
    mids = mesh.p[:, mesh.facets[:, grain_facets]].mean(axis=1).T
    groups = [[] for _ in centres]
    for fi, mid in zip(grain_facets, mids):
        j = int(np.argmin([np.linalg.norm(mid - c) for c in centres]))
        groups[j].append(fi)
    out = [np.asarray(g, dtype=int) for g in groups if len(g)]
    for i, g in enumerate(out):
        print(f"  grain[{i}] facets: {len(g)}")
    return out


def solve(
    mesh,
    facets,
    v_right: float,
    centres: list[np.ndarray],
    p2: bool = False,
    *,
    vim_electrode: str = "neumann",
    component_ids: list[int] | None = None,
    omega: float | None = None,
    contact_edges: list[tuple[int, int, float]] | None = None,
):
    """Complex Laplace + per-grain floating Robin BC (Izumoto continuum).

    vim_electrode:
      'neumann'   — V_im zeroGradient on electrodes (Izumoto; S0 near-field)
      'dirichlet' — V_im=0 on electrodes (pins gauge; better for bulk σ* / impedance)
    """
    from skfem import (
        Basis,
        BilinearForm,
        ElementTetP1,
        ElementTetP2,
        FacetBasis,
        LinearForm,
        asm,
    )
    from skfem.helpers import dot, grad

    if vim_electrode not in ("neumann", "dirichlet"):
        raise ValueError(vim_electrode)

    el = ElementTetP2() if p2 else ElementTetP1()
    basis = Basis(mesh, el)
    particle_groups = split_grain_facets(mesh, facets["grain"], centres)
    grain_groups, particle_to_component = combine_facet_groups(
        particle_groups, component_ids
    )
    print(
        "  electronic components: "
        f"particles={len(particle_groups)} components={len(grain_groups)} "
        f"mapping={particle_to_component.tolist()}"
    )

    @BilinearForm
    def laplace(u, v, w):
        return SIGMA0 * dot(grad(u), grad(v))

    @BilinearForm
    def mass_s(u, v, w):
        return u * v

    @LinearForm
    def ones_s(v, w):
        return v

    A = asm(laplace, basis)
    N = A.shape[0]
    omega = OMEGA if omega is None else float(omega)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")
    kappa = omega * C0
    area_ref = 4 * math.pi * R * R

    Ms, ss, areas = [], [], []
    for gfac in grain_groups:
        fb = FacetBasis(mesh, el, facets=gfac)
        Mg = asm(mass_s, fb)
        sg = np.asarray(asm(ones_s, fb)).ravel()
        ag = float(sg.sum())
        print(f"  grain area={ag:.6e} ({ag/area_ref:.2f}×4πR²)")
        Ms.append(Mg)
        ss.append(sg)
        areas.append(ag)

    Ng = len(grain_groups)
    Msum = sum(Ms)
    K = bmat([[A, -kappa * Msum], [kappa * Msum, A]], format="csr")

    cols_r = []
    for g in range(Ng):
        cols_r.append(csr_matrix((N, 1)))
    for g in range(Ng):
        cols_r.append(kappa * csr_matrix(ss[g].reshape(-1, 1)))
    C_r_top = bmat([cols_r], format="csr")

    cols_i = []
    for g in range(Ng):
        cols_i.append(-kappa * csr_matrix(ss[g].reshape(-1, 1)))
    for g in range(Ng):
        cols_i.append(csr_matrix((N, 1)))
    C_r_bot = bmat([cols_i], format="csr")
    C_right = bmat([[C_r_top], [C_r_bot]], format="csr")

    rows = []
    for g in range(Ng):
        rows.append([csr_matrix(ss[g].reshape(1, -1)), csr_matrix((1, N))])
    for g in range(Ng):
        rows.append([csr_matrix((1, N)), csr_matrix(ss[g].reshape(1, -1))])
    C_bottom = bmat(rows, format="csr")

    corner = np.zeros((2 * Ng, 2 * Ng))
    for g in range(Ng):
        corner[g, g] = -areas[g]
        corner[Ng + g, Ng + g] = -areas[g]
    C_corner = csr_matrix(corner)

    # Finite electronic contacts between otherwise distinct floating components.
    # Current conservation is iω C∫(V-Vm)dS - L_G Vm = 0.  The original
    # capacitance-weighted constraint is divided by iω; this adds +i L_G/ω.
    contact_lap = contact_laplacian(Ng, contact_edges)
    if np.any(contact_lap):
        corner[:Ng, Ng:] -= contact_lap / omega
        corner[Ng:, :Ng] += contact_lap / omega
        C_corner = csr_matrix(corner)
        print(
            f"  finite contacts: edges={len(contact_edges or [])} "
            f"G_total={0.5 * np.trace(contact_lap):.6g} S"
        )

    Big = bmat([[K, C_right], [C_bottom, C_corner]], format="csr").tolil()
    rhs = np.zeros(2 * N + 2 * Ng)

    left_dofs = np.unique(basis.get_dofs(facets=facets["left"]).flatten())
    right_dofs = np.unique(basis.get_dofs(facets=facets["right"]).flatten())
    print(
        f"Element={'P2' if p2 else 'P1'}  N={N}  grains={Ng}  "
        f"Dir L/R={len(left_dofs)}/{len(right_dofs)}  Vim_el={vim_electrode}"
    )
    for dof in left_dofs:
        Big[dof, :] = 0.0
        Big[dof, dof] = 1.0
        rhs[dof] = 0.0
        if vim_electrode == "dirichlet":
            Big[N + dof, :] = 0.0
            Big[N + dof, N + dof] = 1.0
            rhs[N + dof] = 0.0
    for dof in right_dofs:
        Big[dof, :] = 0.0
        Big[dof, dof] = 1.0
        rhs[dof] = v_right
        if vim_electrode == "dirichlet":
            Big[N + dof, :] = 0.0
            Big[N + dof, N + dof] = 1.0
            rhs[N + dof] = 0.0

    print(f"Solving {Big.shape} ...")
    sol = spsolve(Big.tocsr(), rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("Non-finite solution")
    mrs = sol[2 * N : 2 * N + Ng]
    mis = sol[2 * N + Ng :]
    gauge = 0.0
    if vim_electrode == "neumann":
        # V_im + {m_i} null space under Robin+Neumann electrodes
        gauge = float(mis[0])
        sol[N : 2 * N] -= gauge
        mis = mis - gauge
    vms = [complex(float(mrs[g]), float(mis[g])) for g in range(Ng)]
    for g, vm in enumerate(vms):
        print(f"  Vm[{g}] = {vm.real:.6f} + i {vm.imag:.6f}  (V_im gauge shifted by {-gauge:.6e})")
    npz = RESULTS / "last_solution.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz,
        ur=sol[:N],
        ui=sol[N : 2 * N],
        vm_re=mrs,
        vm_im=mis,
        N=N,
        Ng=Ng,
        vim_gauge=gauge,
        vim_electrode=vim_electrode,
    )
    return basis, sol[:N], sol[N : 2 * N], vms


def _eval_at(interp, x: float, y: float, z: float) -> float:
    """skfem interpolator returns shape (n,), not a Python scalar."""
    return float(np.asarray(interp(np.array([[x], [y], [z]]))).ravel()[0])


def sample_x_line(
    basis,
    ur,
    ui,
    x_center: float,
    Lx: float,
    yc: float,
    zc: float,
    n=241,
    grain_centres: list[np.ndarray] | None = None,
):
    xs = np.linspace(0, Lx, n)
    vre = np.full(n, np.nan)
    vim = np.full(n, np.nan)
    P = basis.mesh.p
    centres = grain_centres or [np.array([x_center, yc, zc])]
    irr = iri = None
    try:
        irr = basis.interpolator(ur)
        iri = basis.interpolator(ui)
    except Exception:
        pass
    for i, xv in enumerate(xs):
        # skip interiors of any grain
        if any(abs(xv - float(c[0])) < R * 0.98 for c in centres):
            continue
        ok = False
        if irr is not None:
            try:
                vre[i] = _eval_at(irr, xv, yc, zc)
                vim[i] = _eval_at(iri, xv, yc, zc)
                ok = True
            except Exception:
                ok = False
        if not ok:
            dyz = (P[1] - yc) ** 2 + (P[2] - zc) ** 2
            j = int(np.argmin((P[0] - xv) ** 2 + 25 * dyz))
            if dyz[j] < (4 * R) ** 2:
                vre[i], vim[i] = ur[j], ui[j]
    return xs, vre, vim


def extract_sigma_electrode(
    mesh,
    ur,
    ui,
    e0: float,
    L: float,
    *,
    side: str = "right",
    tol: float = 1e-5,
) -> complex:
    """Bulk impedance σ* from electrode normal current: I = ∫ σ₀ ∂ₙV dA, σ*=I/(A E₀).

    Under V_im Dirichlet electrodes this matches Feng MG imag part closely.
    Under V_im Neumann, imag electrode current ≈ 0 by construction — do not use.
    """
    from skfem import ElementTetP1, FacetBasis, Functional, asm
    from skfem.helpers import dot

    facets_all = mesh.boundary_facets()
    mids = mesh.p[:, mesh.facets[:, facets_all]].mean(axis=1).T
    if side == "right":
        sel = facets_all[mids[:, 0] > L - tol]
        # outward n points +x on right; current into domain from electrode uses +∂ₙ
        sign = 1.0
    elif side == "left":
        sel = facets_all[mids[:, 0] <= tol]
        # outward n points −x on left; flip so current into domain is positive
        sign = -1.0
    else:
        raise ValueError(side)
    if len(sel) < 1:
        raise RuntimeError(f"no {side} electrode facets (L={L})")
    fb = FacetBasis(mesh, ElementTetP1(), facets=sel)

    @Functional
    def area(w):
        return 1.0 + 0.0 * w.x[0]

    @Functional
    def dn(w):
        return dot(w["u"].grad, w.n)

    A = float(asm(area, fb))
    I_re = sign * SIGMA0 * float(asm(dn, fb, u=ur))
    I_im = sign * SIGMA0 * float(asm(dn, fb, u=ui))
    return complex(I_re / (A * e0), I_im / (A * e0))


def extract_sigma_eff(
    basis,
    ur,
    ui,
    e0: float,
    *,
    method: str = "midplane",
    slab_half: float = 0.005,
    x_cut: float | None = None,
) -> complex:
    """σ*_eff = σ₀ <∂V/∂x> / E0  (equiv. -<J_x>/E0, J=-σ₀∇V*).

    Prefer mid-plane slab: electrode-normal V_im flux is ~0 under zeroGradient,
    and full-volume mean of dilute V_im is noisy.
    """
    from skfem import Functional, asm

    if method == "volume":
        @Functional
        def dv_dx(w):
            return w["u"].grad[0]

        @Functional
        def vol(w):
            return 1.0 + 0.0 * w["x"][0]

        V = float(asm(vol, basis))
        return complex(
            SIGMA0 * float(asm(dv_dx, basis, u=ur)) / V / e0,
            SIGMA0 * float(asm(dv_dx, basis, u=ui)) / V / e0,
        )

    mesh = basis.mesh
    xc = float(np.mean(mesh.p[0])) if x_cut is None else float(x_cut)
    cents = mesh.p[:, mesh.t].mean(axis=1).T
    elems = np.nonzero(np.abs(cents[:, 0] - xc) <= slab_half)[0]
    if elems.size < 10:
        raise RuntimeError(f"midplane slab empty (xc={xc}, half={slab_half})")
    return _sigma_on_elements(basis, ur, ui, e0, elems)


def _sigma_on_elements(basis, ur, ui, e0: float, elems: np.ndarray) -> complex:
    """Volume-weighted mean of P1 ∂V/∂x on selected tets."""
    p, t = basis.mesh.p, basis.mesh.t
    dvr = np.zeros(len(elems))
    dvi = np.zeros(len(elems))
    vols = np.zeros(len(elems))
    for i, e in enumerate(elems):
        vs = t[:, e]
        A = np.column_stack((np.ones(4), p[:, vs].T))
        try:
            cr = np.linalg.solve(A, ur[vs])
            ci = np.linalg.solve(A, ui[vs])
        except np.linalg.LinAlgError:
            continue
        dvr[i], dvi[i] = cr[1], ci[1]
        m = np.column_stack(
            (p[:, vs[1]] - p[:, vs[0]], p[:, vs[2]] - p[:, vs[0]], p[:, vs[3]] - p[:, vs[0]])
        )
        vols[i] = abs(np.linalg.det(m)) / 6.0
    wsum = float(vols.sum())
    if wsum <= 0:
        raise RuntimeError("midplane volume weight vanished")
    return complex(
        SIGMA0 * float(np.sum(dvr * vols) / wsum) / e0,
        SIGMA0 * float(np.sum(dvi * vols) / wsum) / e0,
    )


def validate_s0(xs, vre, vim, sigma_eff: complex, out_dir: Path, tag: str, vm: complex):
    """Manuscript-style near-grain validation only (finite box ≠ infinite Feng far field)."""
    from effective_medium import feng_maxwell_garnett

    ana_re, ana_im = feng_along_x(xs, XC)
    # near band: R < |x-xc| < 3R ; imag threshold like sipFoam validate
    near = (
        np.isfinite(vre)
        & np.isfinite(ana_re)
        & (np.abs(xs - XC) > R + 1e-6)
        & (np.abs(xs - XC) < 3 * R)
    )
    re_rel = np.abs(vre[near] - ana_re[near]) / (np.abs(ana_re[near]) + 1e-12) * 100
    im_sig = near & (np.abs(ana_im) > 0.01)
    im_rel = np.abs(vim[im_sig] - ana_im[im_sig]) / (np.abs(ana_im[im_sig]) + 1e-12) * 100

    phi = (4 / 3 * math.pi * R**3) / (LX * LY * LZ)
    em = feng_maxwell_garnett(np.array([F_HZ]), phi, SIGMA0, R, C0)
    sem = complex(em.sigma_eff[0])

    metrics = {
        "engine": tag,
        "frequency_hz": F_HZ,
        "Vm_re": vm.real,
        "Vm_im": vm.imag,
        "phi_true": phi,
        "sigma_eff_re": sigma_eff.real,
        "sigma_eff_im": sigma_eff.imag,
        "sigma_em_re": sem.real,
        "sigma_em_im": sem.imag,
        "near_band": "R < |x-xc| < 3R",
        "V_re_mean_rel_near_pct": float(np.mean(re_rel)) if near.any() else None,
        "V_re_max_rel_near_pct": float(np.max(re_rel)) if near.any() else None,
        "V_im_mean_rel_near_pct": float(np.mean(im_rel)) if im_sig.any() else None,
        "V_im_max_rel_near_pct": float(np.max(im_rel)) if im_sig.any() else None,
        "V_re_rmse_near": float(np.sqrt(np.mean((vre[near] - ana_re[near]) ** 2))) if near.any() else None,
        "V_im_rmse_near": float(np.sqrt(np.mean((vim[near] - ana_im[near]) ** 2))) if near.any() else None,
        "corr_V_re_near": float(np.corrcoef(vre[near], ana_re[near])[0, 1]) if near.sum() > 5 else None,
        "corr_V_im_near": float(np.corrcoef(vim[im_sig], ana_im[im_sig])[0, 1]) if im_sig.sum() > 5 else None,
        "pass_Vre_mean_lt_2pct": bool(np.mean(re_rel) < 2.0) if near.any() else False,
        "pass_Vim_mean_lt_5pct": bool(np.mean(im_rel) < 5.0) if im_sig.any() else False,
    }
    print("=== FEM Scenario 0 vs Feng (NEAR FIELD ONLY) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs * 1e3, vre, "o", ms=3, label="FEM")
    axes[0].plot(xs * 1e3, ana_re, "-", label="Feng")
    axes[0].axvspan((XC - R) * 1e3, (XC + R) * 1e3, alpha=0.15, color="gray")
    axes[0].set_xlim((XC - 4 * R) * 1e3, (XC + 4 * R) * 1e3)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("V_re [V]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"V_re near grain f={F_HZ:.1f} Hz")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs * 1e3, vim, "o", ms=3, label="FEM")
    axes[1].plot(xs * 1e3, ana_im, "-", label="Feng")
    axes[1].axvspan((XC - R) * 1e3, (XC + R) * 1e3, alpha=0.15, color="gray")
    axes[1].set_xlim((XC - 4 * R) * 1e3, (XC + 4 * R) * 1e3)
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("V_im [V]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("V_im near grain")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Plan D Scenario 0 correctness check")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"scenario0_fem_vs_feng.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    np.savetxt(
        out_dir / "scenario0_fem_xline.csv",
        np.c_[xs, vre, vim, ana_re, ana_im],
        delimiter=",",
        header="x,V_re_fem,V_im_fem,V_re_feng,V_im_feng",
        comments="",
    )
    (out_dir / "scenario0_fem_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    global XC, YC, ZC, LX, LY, LZ, V_RIGHT

    ap = argparse.ArgumentParser()
    ap.add_argument("--h-far", type=float, default=0.01)
    ap.add_argument("--h-grain", type=float, default=0.0008)
    ap.add_argument("--p2", action="store_true", help="Use quadratic tet elements")
    ap.add_argument("--two-grain", action="store_true")
    ap.add_argument("--d-mm", type=float, default=5.0)
    ap.add_argument("--L-mm", type=float, default=47.0)
    ap.add_argument("--reuse-mesh", action="store_true", help="Skip Gmsh if .msh already exists")
    args = ap.parse_args()

    out = RESULTS
    meshes = MESHES
    meshes.mkdir(parents=True, exist_ok=True)

    if args.two_grain:
        L = args.L_mm / 1000.0
        d = args.d_mm / 1000.0
        LX = LY = LZ = L
        XC = 0.5 * L
        YC = ZC = 0.5 * L
        V_RIGHT = E0 * L
        msh = meshes / f"two_grain_L{args.L_mm:g}_d{args.d_mm:g}.msh"
        tag = "gmsh-two-grain"
        x1 = 0.5 * L - 0.5 * d - R
        x2 = 0.5 * L + 0.5 * d + R
        grain_centres = [np.array([x1, YC, ZC]), np.array([x2, YC, ZC])]
        write_geo = lambda geo: write_geo_two_grain(geo, L, d, args.h_far, args.h_grain)
    else:
        msh = meshes / "scenario0.msh"
        tag = "gmsh-s0-refined" + ("-P2" if args.p2 else "-P1")
        grain_centres = [np.array([XC, YC, ZC])]
        write_geo = lambda geo: write_geo_scenario0(geo, args.h_far, args.h_grain)

    if args.reuse_mesh and msh.is_file():
        print(f"Reusing mesh {msh}")
    else:
        with tempfile.TemporaryDirectory() as td:
            geo = Path(td) / "case.geo"
            write_geo(geo)
            if not run_gmsh(geo, msh):
                raise SystemExit("Gmsh failed")

    mesh, facets = load_msh_with_tags(msh, grain_centres=grain_centres)
    print(mesh)
    if not all(k in facets for k in ("left", "right", "grain")):
        raise SystemExit(f"bad facets: {list(facets)}")

    e0 = V_RIGHT / LX
    basis, ur, ui, vms = solve(
        mesh, facets, v_right=V_RIGHT, centres=grain_centres, p2=args.p2
    )
    vm0 = vms[0]
    # Midplane: domain centre for two-grain (gap); upstream of grain for S0
    x_cut = 0.5 * LX if args.two_grain else min(0.25 * LX, XC - 2.5 * R)
    try:
        sigma_eff = extract_sigma_eff(
            basis, ur, ui, e0=e0, method="midplane", slab_half=0.003, x_cut=x_cut
        )
        sigma_vol = extract_sigma_eff(basis, ur, ui, e0=e0, method="volume")
        print(
            f"σ*_eff (midplane x={x_cut:.4f}) ≈ {sigma_eff.real:.6f} + i {sigma_eff.imag:.6f}"
        )
        print(f"σ*_eff (volume)            ≈ {sigma_vol.real:.6f} + i {sigma_vol.imag:.6f}")
    except Exception as exc:
        print(f"σ*_eff extraction failed: {exc}")
        sigma_eff = complex(float("nan"), float("nan"))
        sigma_vol = sigma_eff

    xs, vre, vim = sample_x_line(
        basis, ur, ui, XC, LX, YC, ZC, grain_centres=grain_centres
    )

    if args.two_grain:
        out_tg = out / "two_grain"
        out_tg.mkdir(parents=True, exist_ok=True)
        phi = 2 * (4 / 3 * math.pi * R**3) / (LX**3)
        row = {
            "engine": tag,
            "L_mm": args.L_mm,
            "d_mm": args.d_mm,
            "phi_true": phi,
            "Vm": [{"re": v.real, "im": v.imag} for v in vms],
            "sigma_eff_re": sigma_eff.real,
            "sigma_eff_im": sigma_eff.imag,
            "sigma_vol_re": sigma_vol.real,
            "sigma_vol_im": sigma_vol.imag,
            "sigma_x_cut": x_cut,
            "frequency_hz": F_HZ,
        }
        if np.isfinite(sigma_eff.real):
            from interpretation_bias import bias_at_frequency, usable_band

            b = bias_at_frequency(sigma_eff, F_HZ, phi)
            row.update(
                {
                    "phi_inv": b.phi_inv,
                    "phi_bias_rel": b.phi_bias_rel,
                    "phase_error_deg": b.phase_error_true_deg,
                    "band": usable_band(
                        b.phi_bias_rel, b.phase_error_true_deg, reliable=b.inversion_reliable
                    ),
                }
            )
            print(
                f"Two-grain: |Δφ|={b.phase_error_true_deg:.3f}°, "
                f"φ_inv/φ={b.phi_inv / phi:.2f}, band={row['band']}"
            )
        (out_tg / "metrics.json").write_text(json.dumps(row, indent=2))
        np.savetxt(
            out_tg / "xline.csv",
            np.c_[xs, vre, vim],
            delimiter=",",
            header="x,V_re,V_im",
            comments="",
        )
        print(f"Two-grain results → {out_tg}")
    else:
        metrics = validate_s0(xs, vre, vim, sigma_eff, out, tag=tag, vm=vm0)
        ok = metrics.get("pass_Vre_mean_lt_2pct") and metrics.get("pass_Vim_mean_lt_5pct")
        print("CORRECTNESS:", "PASS" if ok else "FAIL (need finer mesh / BC check)")
        print(f"Results → {out}")


if __name__ == "__main__":
    main()
