"""Shared spectral peak and width descriptors used by the contact analyses."""

from __future__ import annotations

import numpy as np


def log_peak_and_width(freq, response):
    freq = np.asarray(freq, float)
    y = np.abs(np.asarray(response, float))
    x = np.log10(freq)
    i = int(np.argmax(y))
    fp, yp = float(freq[i]), float(y[i])
    if 0 < i < len(y) - 1:
        coef = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
        if coef[0] < 0:
            xp = float(-coef[1] / (2.0 * coef[0]))
            if x[i - 1] <= xp <= x[i + 1]:
                fp = 10.0**xp
                yp = float(np.polyval(coef, xp))

    half = 0.5 * yp
    left = right = None
    for j in range(i - 1, -1, -1):
        if y[j] <= half <= y[j + 1]:
            left = float(np.interp(half, [y[j], y[j + 1]], [x[j], x[j + 1]]))
            break
    for j in range(i, len(y) - 1):
        if y[j] >= half >= y[j + 1]:
            right = float(np.interp(half, [y[j + 1], y[j]], [x[j + 1], x[j]]))
            break
    return {
        "fp_interp_hz": fp,
        "peak_interp": yp,
        "fwhm_decades": right - left if left is not None and right is not None else None,
        "f_half_low_hz": 10.0**left if left is not None else None,
        "f_half_high_hz": 10.0**right if right is not None else None,
    }
