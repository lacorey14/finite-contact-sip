#!/usr/bin/env python3
"""
Interpretation-bias utilities for JGR plan v2 (RQ1).

Invert conductor volume fraction phi under Feng Maxwell–Garnett so that the
forward model matches a numerical (or measured) complex conductivity at one
or more frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from config import EM_ERROR_THRESHOLD_PCT, IzumotoBaseParams
from effective_medium import feng_maxwell_garnett, peak_frequency_hz


@dataclass
class BiasResult:
    frequency_hz: float
    phi_true: float
    phi_inv: float
    phi_bias_rel: float  # (phi_inv - phi_true) / phi_true
    phase_num_deg: float
    phase_em_true_deg: float
    phase_em_inv_deg: float
    phase_error_true_deg: float
    fails_em_10deg: bool
    sigma_num: complex
    f_p_hz: float
    inversion_misfit: float = 0.0
    inversion_reliable: bool = True


def phase_deg(z: complex) -> float:
    return float(np.degrees(np.angle(z)))


def invert_phi_feng(
    sigma_obs: complex,
    frequency_hz: float,
    *,
    sigma_m: float | None = None,
    radius_m: float | None = None,
    c0: float | None = None,
    phi_max: float = 0.25,
) -> tuple[float, float]:
    """
    Invert phi by matching imaginary conductivity (polarization), with a weak
    real-part penalty. Returns (phi_inv, rmse_rel) where rmse_rel is a
    dimensionless misfit; large misfit ⇒ inversion unreliable at this frequency.
    """
    params = IzumotoBaseParams()
    sigma_m = params.sigma_0 if sigma_m is None else sigma_m
    radius_m = params.radius_m if radius_m is None else radius_m
    c0 = params.c0 if c0 is None else c0
    f = float(frequency_hz)
    imag_obs = float(np.imag(sigma_obs))
    real_obs = float(np.real(sigma_obs))

    # Wrong-sign polarization vs Feng (which has sigma_imag <= 0 in our EM form
    # near peak depending on convention) — still fit magnitude of imag.
    def objective(phi: float) -> float:
        em = feng_maxwell_garnett(np.array([f]), float(phi), sigma_m, radius_m, c0)
        dimag = abs(float(em.sigma_imag[0]) - imag_obs)
        dreal = abs(float(em.sigma_real[0]) - real_obs)
        return dimag + 0.02 * dreal

    res = minimize_scalar(objective, bounds=(1e-6, phi_max), method="bounded")
    phi_inv = float(res.x)
    em = feng_maxwell_garnett(np.array([f]), phi_inv, sigma_m, radius_m, c0)
    scale = max(abs(imag_obs), abs(float(em.sigma_imag[0])), 1e-6)
    misfit = abs(float(em.sigma_imag[0]) - imag_obs) / scale
    return phi_inv, float(misfit)


def bias_at_frequency(
    sigma_num: complex,
    frequency_hz: float,
    phi_true: float,
) -> BiasResult:
    params = IzumotoBaseParams()
    em = feng_maxwell_garnett(
        np.array([frequency_hz]),
        phi_true,
        params.sigma_0,
        params.radius_m,
        params.c0,
    )
    sigma_em = complex(em.sigma_real[0], em.sigma_imag[0])
    phase_num = phase_deg(sigma_num)
    phase_em = phase_deg(sigma_em)
    err = abs(phase_num - phase_em)
    phi_inv, misfit = invert_phi_feng(sigma_num, frequency_hz)
    em_inv = feng_maxwell_garnett(
        np.array([frequency_hz]),
        phi_inv,
        params.sigma_0,
        params.radius_m,
        params.c0,
    )
    reliable = misfit < 0.5 and abs(sigma_num.imag) > 1e-4
    return BiasResult(
        frequency_hz=float(frequency_hz),
        phi_true=float(phi_true),
        phi_inv=float(phi_inv),
        phi_bias_rel=(phi_inv - phi_true) / phi_true if phi_true > 0 else float("nan"),
        phase_num_deg=phase_num,
        phase_em_true_deg=phase_em,
        phase_em_inv_deg=phase_deg(complex(em_inv.sigma_real[0], em_inv.sigma_imag[0])),
        phase_error_true_deg=err,
        fails_em_10deg=err > EM_ERROR_THRESHOLD_PCT,
        sigma_num=sigma_num,
        f_p_hz=peak_frequency_hz(params.sigma_0, params.radius_m, params.c0),
        inversion_misfit=misfit,
        inversion_reliable=reliable,
    )



def usable_band(
    phi_bias_rel: float,
    phase_error_deg: float,
    *,
    reliable: bool = True,
) -> str:
    """Operational labels for Fig. 3 shading."""
    if not reliable:
        return "unreliable"
    if phase_error_deg > EM_ERROR_THRESHOLD_PCT or abs(phi_bias_rel) > 0.20:
        return "fail"
    if abs(phi_bias_rel) > 0.10 or phase_error_deg > 5.0:
        return "caution"
    return "usable"
