"""
Effective-medium models for metal-grain IP (Feng et al. 2020 / Maxwell-Garnett).

Reference:
  Feng, L., et al. (2020). Scientific Reports, 10, 3456.
  Izumoto, S. (2023). JGR: Solid Earth, 128, e2023JB026757.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from config import IzumotoBaseParams


@dataclass
class EffectiveMediumResult:
    frequencies_hz: np.ndarray
    sigma_eff: np.ndarray          # complex [S/m]
    sigma_real: np.ndarray
    sigma_imag: np.ndarray
    phase_rad: np.ndarray
    peak_frequency_hz: float
    volume_fraction: float


def peak_frequency_hz(sigma_m: float, radius_m: float, c0: float) -> float:
    """f_c = sigma_m / (pi * a * C0)  — Feng/Izumoto Eq. (8)."""
    return sigma_m / (np.pi * radius_m * c0)


def feng_maxwell_garnett(
    frequencies_hz: np.ndarray,
    volume_fraction: float,
    sigma_m: float,
    radius_m: float,
    c0: float,
) -> EffectiveMediumResult:
    """
    Feng et al. (2020, Sci. Rep.) Eq. (1) — dilute conductive spheres with surface capacitance.

    sigma_eff = sigma_m * [1 + 3*V_cond * (1 - (3/2) / (1 + i*f/f_c))]
    with f_c = sigma_m / (pi * a * C0).

    Chargeability m = 9*V_cond / (2*(1 + 3*V_cond)); peak phase φ_c = (9/4)*V/(1+3V) [rad].
    """
    phi = volume_fraction
    f_c = peak_frequency_hz(sigma_m, radius_m, c0)
    f = np.asarray(frequencies_hz, dtype=float)

    polar = 1.0 - 1.5 / (1.0 + 1j * f / f_c)
    sigma_eff = sigma_m * (1.0 + 3.0 * phi * polar)

    phase = np.angle(sigma_eff)

    return EffectiveMediumResult(
        frequencies_hz=f,
        sigma_eff=sigma_eff,
        sigma_real=np.real(sigma_eff),
        sigma_imag=np.imag(sigma_eff),
        phase_rad=phase,
        peak_frequency_hz=f_c,
        volume_fraction=phi,
    )


def chargeability_feng(phi: float) -> float:
    """Maximum chargeability m = 9*phi / (2*(1 + 3*phi)) — Feng Eq. for Cole-Cole c=1."""
    return 9.0 * phi / (2.0 * (1.0 + 3.0 * phi))


def max_phase_rad(phi: float) -> float:
    """Peak phase φ_c = (9/4)*phi/(1+3*phi) [rad] — Feng et al. (2020) Eq. after (1)."""
    return 0.25 * 9.0 * phi / (1.0 + 3.0 * phi)


def non_interacting_superposition(
    single_grain_delta_sigma: np.ndarray,
    n_grains: int,
    sigma_background: float,
) -> np.ndarray:
    """
    Naive superposition baseline: sigma_super = sigma_0 + N * Delta_sigma_single.

    single_grain_delta_sigma: complex array from one-grain simulation (same frequencies).
    """
    return sigma_background + n_grains * single_grain_delta_sigma


def _demo_plot():
    import matplotlib.pyplot as plt

    params = IzumotoBaseParams()
    freqs = params.frequencies_hz()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for phi in [0.01, 0.03, 0.05, 0.08]:
        res = feng_maxwell_garnett(
            freqs, phi, params.sigma_0, params.radius_m, params.c0
        )
        axes[0].semilogx(
            res.frequencies_hz,
            np.degrees(res.phase_rad),
            label=f"φ={phi:.0%}",
        )
        axes[1].semilogx(
            res.frequencies_hz,
            res.sigma_imag,
            label=f"φ={phi:.0%}",
        )

    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Phase [°]")
    axes[0].legend()
    axes[0].set_title("Feng effective-medium model")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Im σ* [S/m]")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = "postprocessing/demo_effective_medium.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"Peak frequency f_c = {params.peak_frequency_hz():.2f} Hz")


def main():
    parser = argparse.ArgumentParser(description="Feng (2020) effective-medium IP model")
    parser.add_argument("--demo", action="store_true", help="Run demo and save plot")
    parser.add_argument("--phi", type=float, default=0.05, help="Volume fraction")
    args = parser.parse_args()

    params = IzumotoBaseParams()
    freqs = params.frequencies_hz()
    res = feng_maxwell_garnett(
        freqs, args.phi, params.sigma_0, params.radius_m, params.c0
    )

    idx_peak = np.argmax(np.abs(res.sigma_imag))
    print(f"Volume fraction phi = {args.phi:.3f}")
    print(f"Peak frequency f_c   = {res.peak_frequency_hz:.3f} Hz")
    print(f"Max |Im(sigma*)| at f = {res.frequencies_hz[idx_peak]:.3f} Hz")
    print(f"Chargeability m       = {chargeability_feng(args.phi):.4f}")
    print(f"Max phase             = {np.degrees(max_phase_rad(args.phi)):.2f} deg")

    if args.demo:
        _demo_plot()


if __name__ == "__main__":
    main()
