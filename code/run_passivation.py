#!/usr/bin/env python3
"""
Phase 1 — partial passivation of a single floating metal/sulfide grain.

Implements the locked cases in IP_Sulfide/partial_passivation_project.md:
  C0(x) = C_active on Γ_active, 0 on Γ_passive.

Equal-active-area pairs target f_active = 0.5 for cases 2–5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve

import _paths  # noqa: E402
from _paths import MESHES, PKG, RESULTS

import run_scenario0 as rs  # noqa: E402

OUT = RESULTS / "passivation_phase1"
C_ACTIVE = rs.C0
R = rs.R
SIGMA0 = rs.SIGMA0
E0 = rs.E0
XC, YC, ZC = rs.XC, rs.YC, rs.ZC
LX, LY, LZ = rs.LX, rs.LY, rs.LZ
V_LEFT, V_RIGHT = rs.V_LEFT, rs.V_RIGHT


@dataclass(frozen=True)
class GeometrySpec:
    case_id: str
    label: str
    f_active_target: float
    pair_group: str  # equal-area group label
    kind: str
    # for polar caps / six-caps: polar half-angle of ACTIVE (or PASSIVE) patches
    theta_deg: float = 0.0
    axis: str = "x"  # 'x' || E0, 'y' or 'z' ⊥ E0
    # for hemisphere: which half is PASSIVE ('+x' = passivate x>xc)
    passive_side: str = "+x"


def _cap_fraction(theta_rad: float) -> float:
    """Single spherical cap area / 4πR² for opening angle theta from pole."""
    return 0.5 * (1.0 - math.cos(theta_rad))


def theta_for_cap_fraction(frac: float) -> float:
    """Opening angle (rad) of one cap with given area fraction."""
    frac = float(np.clip(frac, 1e-9, 0.5 - 1e-9))
    return math.acos(1.0 - 2.0 * frac)


# Equal-active-area f=0.5 pairs:
#  - hemisphere: one half active
#  - two polar caps: each 0.25 → θ = 60°
#  - six passivated caps: total passive 0.5 → each passive 1/12 → θ_p ≈ 33.557°
_THETA_CAP_025 = math.degrees(theta_for_cap_fraction(0.25))  # 60°
_THETA_PASS_SIX = math.degrees(theta_for_cap_fraction(1.0 / 12.0))

GEOMETRIES: dict[str, GeometrySpec] = {
    "full_active": GeometrySpec(
        "full_active", "Fully active sphere", 1.0, "anchor", "full_active"
    ),
    "full_passive": GeometrySpec(
        "full_passive", "Fully passivated sphere", 0.0, "anchor", "full_passive"
    ),
    "hemisphere_par": GeometrySpec(
        "hemisphere_par",
        "One passivated hemisphere, axis || E0",
        0.5,
        "f05",
        "hemisphere",
        axis="x",
        passive_side="+x",
    ),
    "polar_par": GeometrySpec(
        "polar_par",
        "Two active polar caps, axis || E0",
        0.5,
        "f05",
        "two_active_caps",
        theta_deg=_THETA_CAP_025,
        axis="x",
    ),
    "polar_perp": GeometrySpec(
        "polar_perp",
        "Two active polar caps, axis ⊥ E0",
        0.5,
        "f05",
        "two_active_caps",
        theta_deg=_THETA_CAP_025,
        axis="y",
    ),
    "six_caps": GeometrySpec(
        "six_caps",
        "Six distributed passivated caps (Gurin-2 analogue)",
        0.5,
        "f05",
        "six_passive_caps",
        theta_deg=_THETA_PASS_SIX,
    ),
}


def facet_mids_unit(mesh, facets: np.ndarray, centre: np.ndarray) -> np.ndarray:
    mids = mesh.p[:, mesh.facets[:, facets]].mean(axis=1).T
    u = mids - centre
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-30)
    return u / nrm


def active_mask(spec: GeometrySpec, u: np.ndarray) -> np.ndarray:
    """Boolean mask length = n facets; True = active (C0=C_active)."""
    n = len(u)
    if spec.kind == "full_active":
        return np.ones(n, dtype=bool)
    if spec.kind == "full_passive":
        return np.zeros(n, dtype=bool)
    if spec.kind == "hemisphere":
        # passivate +x half of sphere → active where x_local <= 0
        if spec.passive_side == "+x":
            return u[:, 0] <= 0.0
        return u[:, 0] >= 0.0
    if spec.kind == "two_active_caps":
        th = math.radians(spec.theta_deg)
        ax = {"x": 0, "y": 1, "z": 2}[spec.axis]
        mu = np.abs(u[:, ax])
        return mu >= math.cos(th)  # near both poles
    if spec.kind == "six_passive_caps":
        th = math.radians(spec.theta_deg)
        cth = math.cos(th)
        # passivated near ±x,±y,±z poles
        passive = (
            (np.abs(u[:, 0]) >= cth)
            | (np.abs(u[:, 1]) >= cth)
            | (np.abs(u[:, 2]) >= cth)
        )
        return ~passive
    raise ValueError(spec.kind)


def activity_score(spec: GeometrySpec, u: np.ndarray) -> np.ndarray:
    """Rank facets from most to least active for area-controlled discretization."""
    if spec.kind == "hemisphere":
        return -u[:, 0] if spec.passive_side == "+x" else u[:, 0]
    if spec.kind == "two_active_caps":
        ax = {"x": 0, "y": 1, "z": 2}[spec.axis]
        return np.abs(u[:, ax])
    if spec.kind == "six_passive_caps":
        # Active surface is furthest from the six coordinate-axis poles.
        return -np.max(np.abs(u), axis=1)
    raise ValueError(f"no activity score for {spec.kind}")


def classify_grain(
    mesh, grain_facets: np.ndarray, centre: np.ndarray, spec: GeometrySpec
) -> tuple[np.ndarray, np.ndarray, float]:
    u = facet_mids_unit(mesh, grain_facets, centre)
    mask = active_mask(spec, u)

    # Analytical angular thresholds do not produce exactly equal areas on an
    # unstructured triangulation.  For comparison cases, rank facets by the
    # same geometric criterion and choose the prefix closest to the prescribed
    # true facet area.  The residual mismatch is then at most one facet.
    p = mesh.p
    fidx = mesh.facets[:, grain_facets]
    facet_area = 0.5 * np.linalg.norm(
        np.cross(
            p[:, fidx[1]] - p[:, fidx[0]],
            p[:, fidx[2]] - p[:, fidx[0]],
            axis=0,
        ).T,
        axis=1,
    )
    if spec.kind not in ("full_active", "full_passive"):
        order = np.argsort(-activity_score(spec, u), kind="stable")
        cumulative = np.cumsum(facet_area[order])
        target = spec.f_active_target * float(facet_area.sum())
        k = int(np.argmin(np.abs(cumulative - target))) + 1
        mask = np.zeros(len(grain_facets), dtype=bool)
        mask[order[:k]] = True
    active = grain_facets[mask]
    passive = grain_facets[~mask]
    # area fraction via facet areas
    from skfem import ElementTetP1, FacetBasis, LinearForm, asm

    el = ElementTetP1()

    @LinearForm
    def ones(v, w):
        return v

    if len(active):
        a_act = float(np.asarray(asm(ones, FacetBasis(mesh, el, facets=active))).sum())
    else:
        a_act = 0.0
    if len(grain_facets):
        a_tot = float(np.asarray(asm(ones, FacetBasis(mesh, el, facets=grain_facets))).sum())
    else:
        a_tot = 0.0
    f_geom = a_act / a_tot if a_tot > 0 else 0.0
    return active, passive, f_geom


def solve_partial(
    mesh,
    facets: dict,
    *,
    active_facets: np.ndarray,
    passive_facets: np.ndarray | None = None,
    omega: float,
    c_active: float = C_ACTIVE,
    c_passive: float = 0.0,
    v_right: float = V_RIGHT,
    vim_electrode: str = "dirichlet",
):
    """Laplace + floating conductor with C0 only on active facets."""
    from skfem import (
        Basis,
        BilinearForm,
        ElementTetP1,
        FacetBasis,
        LinearForm,
        asm,
    )
    from skfem.helpers import dot, grad

    el = ElementTetP1()
    basis = Basis(mesh, el)

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
    passive_facets = (
        np.asarray(passive_facets, dtype=int)
        if passive_facets is not None
        else np.asarray([], dtype=int)
    )
    if (len(active_facets) == 0 or c_active == 0.0) and (
        len(passive_facets) == 0 or c_passive == 0.0
    ):
        # Fully insulated cavity: no floating DOFs
        Big = bmat([[A, None], [None, A]], format="csr").tolil()
        rhs = np.zeros(2 * N)
        left_dofs = np.unique(basis.get_dofs(facets=facets["left"]).flatten())
        right_dofs = np.unique(basis.get_dofs(facets=facets["right"]).flatten())
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
        sol = spsolve(Big.tocsr(), rhs)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("Non-finite solution (passive)")
        return basis, sol[:N], sol[N:], complex(0.0, 0.0), 0.0

    M_weighted = csr_matrix(A.shape)
    s_weighted = np.zeros(N)
    area_a = 0.0
    if len(active_facets) and c_active != 0.0:
        fb_a = FacetBasis(mesh, el, facets=active_facets)
        M_a = asm(mass_s, fb_a)
        s_a = np.asarray(asm(ones_s, fb_a)).ravel()
        area_a = float(s_a.sum())
        M_weighted = M_weighted + c_active * M_a
        s_weighted = s_weighted + c_active * s_a
    if len(passive_facets) and c_passive != 0.0:
        fb_p = FacetBasis(mesh, el, facets=passive_facets)
        M_p = asm(mass_s, fb_p)
        s_p = np.asarray(asm(ones_s, fb_p)).ravel()
        M_weighted = M_weighted + c_passive * M_p
        s_weighted = s_weighted + c_passive * s_p

    B = omega * M_weighted
    t = omega * s_weighted
    weighted_area = float(s_weighted.sum())
    K = bmat([[A, -B], [B, A]], format="csr")
    # floating Vm unknowns: m_re, m_im  (mean over ACTIVE surface)
    C_right = bmat(
        [
            [csr_matrix((N, 1)), csr_matrix(t.reshape(-1, 1))],
            [-csr_matrix(t.reshape(-1, 1)), csr_matrix((N, 1))],
        ],
        format="csr",
    )
    C_bottom = bmat(
        [
            [csr_matrix(s_weighted.reshape(1, -1)), csr_matrix((1, N))],
            [csr_matrix((1, N)), csr_matrix(s_weighted.reshape(1, -1))],
        ],
        format="csr",
    )
    C_corner = csr_matrix(np.diag([-weighted_area, -weighted_area]))
    if C_right.shape != (2 * N, 2) or C_bottom.shape != (2, 2 * N):
        raise RuntimeError(
            f"invalid floating-potential coupling shapes: "
            f"C_right={C_right.shape}, C_bottom={C_bottom.shape}, N={N}"
        )
    Big = bmat([[K, C_right], [C_bottom, C_corner]], format="csr").tolil()
    rhs = np.zeros(2 * N + 2)

    left_dofs = np.unique(basis.get_dofs(facets=facets["left"]).flatten())
    right_dofs = np.unique(basis.get_dofs(facets=facets["right"]).flatten())
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

    sol = spsolve(Big.tocsr(), rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("Non-finite solution")
    vm = complex(float(sol[2 * N]), float(sol[2 * N + 1]))
    return basis, sol[:N], sol[N : 2 * N], vm, area_a


def surface_integrals(
    mesh,
    ur,
    ui,
    grain_facets: np.ndarray,
    active_facets: np.ndarray,
    centre: np.ndarray,
    vm: complex,
    omega: float,
    c_active: float,
):
    """Jn, qs, dipole, net current, field-weighted Seff diagnostics."""
    from skfem import ElementTetP1, FacetBasis, Functional, asm
    from skfem.helpers import dot

    el = ElementTetP1()
    if len(grain_facets) == 0:
        raise RuntimeError("no grain facets")

    fb_all = FacetBasis(mesh, el, facets=grain_facets)

    @Functional
    def area(w):
        return 1.0 + 0.0 * w.x[0]

    @Functional
    def dn(w):
        return dot(w["u"].grad, w.n)

    @Functional
    def u_s(w):
        return w["u"]

    A_tot = float(asm(area, fb_all))
    # outward normal from electrolyte domain points into grain; Jn_into_grain = -σ0 ∂n V
    # (n = outward from mesh volume = into grain for cavity grain)
    I_re = -SIGMA0 * float(asm(dn, fb_all, u=ur))
    I_im = -SIGMA0 * float(asm(dn, fb_all, u=ui))
    net_I = complex(I_re, I_im)

    # Per-facet quantities via midpoint sampling of P1 field
    mids = mesh.p[:, mesh.facets[:, grain_facets]].mean(axis=1).T
    # facet areas
    p = mesh.p
    fidx = mesh.facets[:, grain_facets]
    a = 0.5 * np.linalg.norm(
        np.cross(p[:, fidx[1]] - p[:, fidx[0]], p[:, fidx[2]] - p[:, fidx[0]], axis=0).T,
        axis=1,
    )

    # interpolate V at facet mids using nearest node average of facet nodes
    vre_f = ur[fidx].mean(axis=0)
    vim_f = ui[fidx].mean(axis=0)
    # approximate ∂n V at facet: use cell gradient of adjacent tet if available — fallback 0
    # Use FacetBasis projection of dn for each facet group is expensive; use global asm density:
    # store facet-wise Jn from P1: reconstruct from nodal grads on adjacent elements.
    Jn_re = np.zeros(len(grain_facets))
    Jn_im = np.zeros(len(grain_facets))
    # Build map facet -> one adjacent tet (boundary facets have one tet)
    # skfem mesh.f2t
    f2t = mesh.f2t
    for i, fi in enumerate(grain_facets):
        t0 = int(f2t[0, fi])
        vs = mesh.t[:, t0]
        Amat = np.column_stack((np.ones(4), p[:, vs].T))
        try:
            cr = np.linalg.solve(Amat, ur[vs])
            ci = np.linalg.solve(Amat, ui[vs])
        except np.linalg.LinAlgError:
            continue
        # grad = (cr[1], cr[2], cr[3]); outward n from volume at facet
        # approximate n from cross product of facet edges, oriented out of tet
        v0, v1, v2 = fidx[:, i]
        nvec = np.cross(p[:, v1] - p[:, v0], p[:, v2] - p[:, v0])
        # flip so n points away from tet centroid
        cent = p[:, vs].mean(axis=1)
        mid = mids[i]
        if np.dot(nvec, mid - cent) < 0:
            nvec = -nvec
        nrm = np.linalg.norm(nvec)
        if nrm < 1e-30:
            continue
        nh = nvec / nrm
        grad_r = np.array([cr[1], cr[2], cr[3]])
        grad_i = np.array([ci[1], ci[2], ci[3]])
        # electrolyte outward n (= into grain); Jn into grain = -σ0 ∂n V
        Jn_re[i] = -SIGMA0 * float(np.dot(grad_r, nh))
        Jn_im[i] = -SIGMA0 * float(np.dot(grad_i, nh))

    active_set = set(int(x) for x in active_facets)
    is_act = np.array([int(fi) in active_set for fi in grain_facets], dtype=bool)
    # capacitive surface charge qs = C0 (V - Vm) on active (complex)
    qs_re = np.zeros(len(grain_facets))
    qs_im = np.zeros(len(grain_facets))
    qs_re[is_act] = c_active * (vre_f[is_act] - vm.real)
    qs_im[is_act] = c_active * (vim_f[is_act] - vm.imag)

    # dipole moment p = ∫ (x-xc) qs dS
    r = mids - centre
    p_re = np.sum(r * qs_re[:, None] * a[:, None], axis=0)
    p_im = np.sum(r * qs_im[:, None] * a[:, None], axis=0)

    J0 = SIGMA0 * E0
    q0 = c_active * E0 * R  # scale
    phi0 = E0 * R

    wJ = np.abs(Jn_re + 1j * Jn_im) / max(J0, 1e-30)
    wq = np.abs(qs_re + 1j * qs_im) / max(q0, 1e-30)
    wphi = np.abs(vim_f) / max(phi0, 1e-30)
    abs_interface_current = float(
        np.sum(np.abs(Jn_re + 1j * Jn_im) * a)
    )
    background_electrode_current = SIGMA0 * E0 * LY * LZ
    net_I_rel = abs(net_I) / max(
        abs_interface_current, background_electrode_current * 1e-12
    )

    # Seff only integrated on geometrically active patches (definition in plan)
    Seff_J = float(np.sum(wJ[is_act] * a[is_act]))
    Seff_q = float(np.sum(wq[is_act] * a[is_act]))
    Seff_phi = float(np.sum(wphi[is_act] * a[is_act]))
    S_active = float(np.sum(a[is_act]))
    S_total = float(np.sum(a))

    return {
        "area_total": A_tot,
        "S_active_geom": S_active,
        "S_total_geom": S_total,
        "f_active_geom": S_active / S_total if S_total > 0 else 0.0,
        "net_I_re": net_I.real,
        "net_I_im": net_I.imag,
        "net_I_abs": abs(net_I),
        "abs_interface_current": abs_interface_current,
        "net_I_rel_l1": net_I_rel,
        "dipole_re": p_re.tolist(),
        "dipole_im": p_im.tolist(),
        "dipole_x_abs": abs(complex(p_re[0], p_im[0])),
        "Seff_J": Seff_J,
        "Seff_q": Seff_q,
        "Seff_phi": Seff_phi,
        "mean_abs_Jn_active": float(np.mean(np.abs(Jn_re + 1j * Jn_im)[is_act])) if is_act.any() else 0.0,
        "mean_abs_Vim_active": float(np.mean(np.abs(vim_f[is_act]))) if is_act.any() else 0.0,
        "facet_table": {
            "a": a,
            "Jn_re": Jn_re,
            "Jn_im": Jn_im,
            "qs_re": qs_re,
            "qs_im": qs_im,
            "vre": vre_f,
            "vim": vim_f,
            "is_active": is_act,
            "mids": mids,
        },
    }


def frequencies_hz(n: int = 12) -> np.ndarray:
    # Izumoto/Feng characteristic frequency for the reference sphere is rs.FP.
    # Span three decades around it so both plateaus and the relaxation are sampled.
    return np.logspace(np.log10(0.03 * rs.FP), np.log10(30.0 * rs.FP), n)


def run_case(
    spec: GeometrySpec,
    mesh,
    facets,
    freqs: np.ndarray,
    *,
    h_tag: str,
    reuse_weights_from: dict | None = None,
    reference_sink: dict | None = None,
):
    centre = np.array([XC, YC, ZC])
    active, passive, f_geom = classify_grain(mesh, facets["grain"], centre, spec)
    case_dir = OUT / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        **asdict(spec),
        "f_active_geom_mesh": f_geom,
        "n_active_facets": int(len(active)),
        "n_passive_facets": int(len(passive)),
        "mesh_tag": h_tag,
        "C_active": C_ACTIVE,
        "sigma0": SIGMA0,
        "R": R,
        "domain_mm": [LX * 1e3, LY * 1e3, LZ * 1e3],
    }
    (case_dir / "geometry.json").write_text(json.dumps(meta, indent=2))
    print(
        f"\n=== {spec.case_id}: f_geom={f_geom:.4f} "
        f"(target {spec.f_active_target:.2f}), "
        f"facets act/pas={len(active)}/{len(passive)} ==="
    )

    rows = []
    # Reference weights from full_active at each frequency (for Seff predictor)
    for f in freqs:
        omega = 2 * math.pi * float(f)
        basis, ur, ui, vm, area_a = solve_partial(
            mesh,
            facets,
            active_facets=active,
            omega=omega,
            c_active=C_ACTIVE,
            vim_electrode="dirichlet",
        )
        sigma = rs.extract_sigma_electrode(mesh, ur, ui, e0=E0, L=LX, side="right")
        sigma_L = rs.extract_sigma_electrode(mesh, ur, ui, e0=E0, L=LX, side="left")
        sigma_avg = 0.5 * (sigma + sigma_L)
        surf = surface_integrals(
            mesh, ur, ui, facets["grain"], active, centre, vm, omega, C_ACTIVE
        )
        phase_deg = float(np.degrees(np.angle(sigma_avg)))
        row = {
            "case_id": spec.case_id,
            "frequency_hz": float(f),
            "omega": omega,
            "f_active_geom": surf["f_active_geom"],
            "S_active": surf["S_active_geom"],
            "S_total": surf["S_total_geom"],
            "sigma_re": sigma_avg.real,
            "sigma_im": sigma_avg.imag,
            "phase_deg": phase_deg,
            "Vm_re": vm.real,
            "Vm_im": vm.imag,
            "net_I_abs": surf["net_I_abs"],
            "net_I_rel_l1": surf["net_I_rel_l1"],
            "dipole_x_abs": surf["dipole_x_abs"],
            "Seff_J": surf["Seff_J"],
            "Seff_q": surf["Seff_q"],
            "Seff_phi": surf["Seff_phi"],
            "area_active_constraint": area_a,
        }
        ft = surf["facet_table"]
        fkey = f"{float(f):.12g}"
        if reference_sink is not None:
            # A predictor must be defined independently of the partially
            # passivated response.  Store weights only from the fully active
            # reference sphere, with the common facet ordering of this mesh.
            reference_sink[fkey] = {
                "a": ft["a"].copy(),
                "wJ": np.abs(ft["Jn_re"] + 1j * ft["Jn_im"]) / max(SIGMA0 * E0, 1e-30),
                "wq": np.abs(ft["qs_re"] + 1j * ft["qs_im"]) / max(C_ACTIVE * E0 * R, 1e-30),
                "wphi": np.abs(ft["vim"]) / max(E0 * R, 1e-30),
            }
        prediction_weights = reuse_weights_from or reference_sink
        if prediction_weights is not None and fkey in prediction_weights:
            wref = prediction_weights[fkey]
            is_act = ft["is_active"]
            if len(is_act) != len(wref["a"]):
                raise RuntimeError("reference/current facet ordering mismatch")
            for name in ("wJ", "wq", "wphi"):
                weighted_all = float(np.sum(wref[name] * wref["a"]))
                weighted_active = float(np.sum(wref[name][is_act] * wref["a"][is_act]))
                row[f"Seff_ref_{name[1:]}"] = weighted_active
                row[f"Seff_ref_{name[1:]}_frac"] = (
                    weighted_active / weighted_all if weighted_all > 0 else 0.0
                )
        rows.append(row)
        # save one mid-band solution for viz
        if abs(f - freqs[len(freqs) // 2]) < 1e-9:
            np.savez_compressed(
                case_dir / "solution_midband.npz",
                ur=ur,
                ui=ui,
                vm_re=vm.real,
                vm_im=vm.imag,
                frequency_hz=float(f),
            )
            np.savez_compressed(
                case_dir / "surface_midband.npz",
                a=ft["a"],
                Jn_re=ft["Jn_re"],
                Jn_im=ft["Jn_im"],
                qs_re=ft["qs_re"],
                qs_im=ft["qs_im"],
                vre=ft["vre"],
                vim=ft["vim"],
                is_active=ft["is_active"],
                mids=ft["mids"],
            )
        print(
            f"  f={f:7.3f} Hz  σ*={sigma_avg.real:.5f}{sigma_avg.imag:+.5f}j  "
            f"φ={phase_deg:.3f}°  |I_net|={surf['net_I_abs']:.3e}  "
            f"|p_x|={surf['dipole_x_abs']:.3e}"
        )

    # spectrum metrics
    freqs_a = np.array([r["frequency_hz"] for r in rows])
    sig_im = np.array([r["sigma_im"] for r in rows])
    phase = np.array([r["phase_deg"] for r in rows])
    # peak of |σ″| and of |phase|
    i_peak = int(np.argmax(np.abs(sig_im)))
    i_ph = int(np.argmax(np.abs(phase)))
    # chargeability proxy: max |σ″|/σ′ or Cole-Cole-like Δσ/σ∞
    sigma_re = np.array([r["sigma_re"] for r in rows])
    m_proxy = float(np.max(np.abs(sig_im)) / max(np.mean(sigma_re), 1e-30))
    summary = {
        **meta,
        "f_p_sigma_im_hz": float(freqs_a[i_peak]),
        "tau_sigma_im_s": 1.0 / (2 * math.pi * float(freqs_a[i_peak])),
        "f_p_phase_hz": float(freqs_a[i_ph]),
        "max_abs_sigma_im": float(np.max(np.abs(sig_im))),
        "max_abs_phase_deg": float(np.max(np.abs(phase))),
        "chargeability_proxy": m_proxy,
        "max_net_I_abs": float(max(r["net_I_abs"] for r in rows)),
        "max_net_I_rel_l1": float(max(r["net_I_rel_l1"] for r in rows)),
        "spectrum": rows,
    }
    (case_dir / "spectrum.json").write_text(json.dumps(summary, indent=2))
    with (case_dir / "spectrum.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary


def homogeneous_reference(mesh, facets, freqs: np.ndarray) -> list[dict]:
    """No grain polarization: treat grain as insulated cavity (same as full_passive)."""
    return []  # full_passive case serves as this reference


def main():
    ap = argparse.ArgumentParser(description="Phase-1 partial passivation campaign")
    ap.add_argument(
        "--cases",
        nargs="*",
        default=list(GEOMETRIES.keys()),
        help="Subset of case ids",
    )
    ap.add_argument("--n-freq", type=int, default=10)
    ap.add_argument("--reuse-mesh", action="store_true", default=True)
    ap.add_argument("--h-far", type=float, default=0.01)
    ap.add_argument("--h-grain", type=float, default=0.0005)
    ap.add_argument("--quick", action="store_true", help="coarser grain mesh + fewer freqs")
    ap.add_argument(
        "--screening-spectrum",
        action="store_true",
        help="reuse the coarse mesh but retain the requested full frequency grid",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # snapshot plan
    plan_src = PKG.parent.parent / "IP_Sulfide" / "partial_passivation_project.md"
    if not plan_src.is_file():
        plan_src = Path("/Users/lihai/Library/CloudStorage/OneDrive-个人/桌面/Codex/IP_Sulfide/partial_passivation_project.md")
    if plan_src.is_file():
        (OUT / "partial_passivation_project.md").write_text(plan_src.read_text())

    if args.quick:
        args.n_freq = 3
        args.h_far = max(args.h_far, 0.018)
        args.h_grain = max(args.h_grain, 0.0012)

    use_coarse_mesh = args.quick or args.screening_spectrum
    if args.screening_spectrum:
        args.h_far = max(args.h_far, 0.018)
        args.h_grain = max(args.h_grain, 0.0012)
    mesh_name = "scenario0_passivation_quick.msh" if use_coarse_mesh else "scenario0_passivation.msh"
    msh = MESHES / mesh_name
    if args.reuse_mesh and (MESHES / "scenario0.msh").is_file() and not msh.is_file():
        # reuse validated S0 mesh
        msh = MESHES / "scenario0.msh"
        h_tag = "reuse-scenario0"
    elif msh.is_file() and args.reuse_mesh:
        h_tag = "scenario0_passivation"
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            geo = Path(td) / "s0.geo"
            rs.write_geo_scenario0(geo, args.h_far, args.h_grain)
            ok = rs.run_gmsh(geo, msh)
            if not ok:
                raise SystemExit("gmsh failed")
        h_tag = f"hfar{args.h_far}_hgrain{args.h_grain}"

    mesh, facets = rs.load_msh_with_tags(msh, grain_centres=[np.array([XC, YC, ZC])])
    if "grain" not in facets:
        raise SystemExit("no grain facets")

    freqs = (
        np.asarray([0.1 * rs.FP, rs.FP, 10.0 * rs.FP])
        if args.quick
        else frequencies_hz(args.n_freq)
    )
    (OUT / "frequencies.json").write_text(json.dumps(freqs.tolist(), indent=2))

    # geometry fraction check table
    centre = np.array([XC, YC, ZC])
    geo_rows = []
    for cid in args.cases:
        spec = GEOMETRIES[cid]
        active, passive, f_geom = classify_grain(mesh, facets["grain"], centre, spec)
        geo_rows.append(
            {
                "case_id": cid,
                "f_target": spec.f_active_target,
                "f_geom": f_geom,
                "n_active": len(active),
                "n_passive": len(passive),
                "pair_group": spec.pair_group,
            }
        )
        print(f"geometry {cid}: f={f_geom:.4f} (target {spec.f_active_target})")
    with (OUT / "geometry_fractions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(geo_rows[0].keys()))
        w.writeheader()
        w.writerows(geo_rows)

    summaries = {}
    reference_weights: dict = {}
    ordered_cases = list(args.cases)
    if "full_active" in ordered_cases:
        ordered_cases.remove("full_active")
        ordered_cases.insert(0, "full_active")
    for cid in ordered_cases:
        is_reference = cid == "full_active"
        summaries[cid] = run_case(
            GEOMETRIES[cid],
            mesh,
            facets,
            freqs,
            h_tag=h_tag,
            reuse_weights_from=(reference_weights if reference_weights else None),
            reference_sink=(reference_weights if is_reference else None),
        )

    # acceptance + decision gate
    gate = analyze_gate(summaries)
    (OUT / "PHASE1_GATE.json").write_text(json.dumps(gate, indent=2))
    (OUT / "PHASE1_GATE.md").write_text(gate_markdown(gate, summaries))
    print("\n" + gate_markdown(gate, summaries))
    print(f"\nResults → {OUT}")


def analyze_gate(summaries: dict) -> dict:
    """Decision gate after Phase 1 (project §Decision gate)."""
    fa = summaries.get("full_active", {})
    fp = summaries.get("full_passive", {})
    # fully passive: |σ″| and |phase| should be tiny vs full_active
    pass_passive = True
    if fa and fp:
        ratio = fp.get("max_abs_sigma_im", 1) / max(fa.get("max_abs_sigma_im", 1e-30), 1e-30)
        pass_passive = ratio < 0.05
    # Floating-grain conservation normalized by the L1 interfacial current.
    # The fully passive cavity has no physical interfacial current and is
    # therefore excluded from this particular relative diagnostic.
    net_ok = all(
        s.get("max_net_I_rel_l1", 1) < 0.02
        for s in summaries.values()
        if s.get("case_id") != "full_passive"
    )

    f05 = [s for s in summaries.values() if s.get("pair_group") == "f05"]
    phases = [s.get("max_abs_phase_deg", 0) for s in f05]
    fps = [s.get("f_p_sigma_im_hz", 0) for s in f05]
    ms = [s.get("chargeability_proxy", 0) for s in f05]
    spread_phase = float(max(phases) - min(phases)) if phases else 0.0
    spread_fp = float(max(fps) / max(min(fps), 1e-30)) if fps else 1.0
    spread_m = float(max(ms) / max(min(ms), 1e-30)) if ms else 1.0

    # material difference thresholds (operational)
    geom_differs = spread_phase > 0.2 or spread_fp > 1.15 or spread_m > 1.15
    orientation_shift = False
    if "polar_par" in summaries and "polar_perp" in summaries:
        orientation_shift = abs(
            summaries["polar_par"]["f_p_sigma_im_hz"] - summaries["polar_perp"]["f_p_sigma_im_hz"]
        ) / max(summaries["polar_par"]["f_p_sigma_im_hz"], 1e-30) > 0.1 or abs(
            summaries["polar_par"]["max_abs_phase_deg"] - summaries["polar_perp"]["max_abs_phase_deg"]
        ) > 0.15

    # Compare geometric area with a non-circular field-weighted predictor built
    # exclusively from the fully-active reference solution.
    def mid_seff(s, key):
        rows = s.get("spectrum", [])
        if not rows:
            return float("nan")
        return float(rows[len(rows) // 2].get(key, float("nan")))

    xs_f, ys_m, xs_sj = [], [], []
    for s in summaries.values():
        if s.get("case_id") == "full_passive":
            continue
        xs_f.append(s.get("f_active_geom_mesh", s.get("spectrum", [{}])[0].get("f_active_geom", 0)))
        ys_m.append(s.get("chargeability_proxy", 0))
        xs_sj.append(mid_seff(s, "Seff_ref_J_frac"))

    def corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    c_geom = corr(xs_f, ys_m)
    c_seff = corr(xs_sj, ys_m)
    seff_better = (not math.isnan(c_seff)) and (math.isnan(c_geom) or abs(c_seff) > abs(c_geom) + 0.05)

    proceed = bool(geom_differs or orientation_shift or seff_better)
    return {
        "acceptance": {
            "full_passive_suppressed": pass_passive,
            "net_current_ok": net_ok,
            "passive_over_active_sigma_im_ratio": (
                fp.get("max_abs_sigma_im", None) / max(fa.get("max_abs_sigma_im", 1e-30), 1e-30)
                if fa and fp
                else None
            ),
        },
        "f05_spread": {
            "phase_deg": spread_phase,
            "fp_ratio": spread_fp,
            "m_ratio": spread_m,
            "cases": [s["case_id"] for s in f05],
        },
        "gate": {
            "equal_area_spectra_differ": geom_differs,
            "orientation_shift": orientation_shift,
            "seff_explains_better_than_f_geom": seff_better,
            "corr_m_vs_f_geom": c_geom,
            "corr_m_vs_Seff_ref_J": c_seff,
            "proceed_to_core_shell": proceed,
        },
    }


def gate_markdown(gate: dict, summaries: dict) -> str:
    lines = [
        "# Phase 1 passivation — decision gate",
        "",
        "## Acceptance",
        f"- Full passive suppressed: **{gate['acceptance']['full_passive_suppressed']}**",
        f"- Net current OK: **{gate['acceptance']['net_current_ok']}**",
        f"- σ″_passive / σ″_active: {gate['acceptance']['passive_over_active_sigma_im_ratio']}",
        "",
        "## Equal-area (f≈0.5) spread",
        f"- Δphase_max: {gate['f05_spread']['phase_deg']:.3f}°",
        f"- f_p ratio: {gate['f05_spread']['fp_ratio']:.3f}",
        f"- m ratio: {gate['f05_spread']['m_ratio']:.3f}",
        "",
        "## Case summaries",
    ]
    for cid, s in summaries.items():
        lines.append(
            f"- **{cid}**: f_p={s.get('f_p_sigma_im_hz', float('nan')):.3f} Hz, "
            f"|φ|_max={s.get('max_abs_phase_deg', float('nan')):.3f}°, "
            f"m̃={s.get('chargeability_proxy', float('nan')):.4f}, "
            f"ε_I,max={s.get('max_net_I_rel_l1', float('nan')):.2e}"
        )
    g = gate["gate"]
    lines += [
        "",
        "## Decision gate",
        f"1. Equal-area spectra differ: **{g['equal_area_spectra_differ']}**",
        f"2. Orientation shift: **{g['orientation_shift']}**",
        f"3. Seff better than f_geom: **{g['seff_explains_better_than_f_geom']}** "
        f"(corr f_geom={g['corr_m_vs_f_geom']}, "
        f"Seff_ref,J={g['corr_m_vs_Seff_ref_J']})",
        "",
        f"### Proceed to core–shell Phase 2? **{g['proceed_to_core_shell']}**",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
