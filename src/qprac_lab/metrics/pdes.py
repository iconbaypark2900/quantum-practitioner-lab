from __future__ import annotations

import numpy as np


def l2_error(reference, candidate) -> float:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    return float(np.linalg.norm(reference - candidate))


def residual_loss(residuals) -> float:
    residuals = np.asarray(residuals)
    return float(np.mean(residuals ** 2))
