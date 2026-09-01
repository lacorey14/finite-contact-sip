"""Physical and numerical parameters shared across postprocessing scripts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IzumotoBaseParams:
    """Scenario 0 reference values (Izumoto 2023, Section 2.2)."""

    radius_m: float = 5e-3          # grain radius [m]
    sigma_0: float = 0.2              # background conductivity [S/m]
    c0: float = 0.4                   # surface capacitance [F/m^2]
    e0: float = 50.0                  # applied electric field [V/m]

    # Log-spaced frequencies [Hz], 20 points, 1.59 – 159 Hz
    f_min: float = 1.59
    f_max: float = 159.0
    n_freq: int = 20

    def frequencies_hz(self):
        import numpy as np
        return np.logspace(
            np.log10(self.f_min),
            np.log10(self.f_max),
            self.n_freq,
        )

    def peak_frequency_hz(self) -> float:
        """f_p = sigma_0 / (pi * R * C0)  — Izumoto Eq. (8)."""
        import numpy as np
        return self.sigma_0 / (np.pi * self.radius_m * self.c0)


# Hupfer et al. (2016) pyrite fraction range for trend comparison [vol.% in solid]
HUPFER_PYRITE_FRACTIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Error threshold for effective-medium failure criterion
EM_ERROR_THRESHOLD_PCT = 10.0
