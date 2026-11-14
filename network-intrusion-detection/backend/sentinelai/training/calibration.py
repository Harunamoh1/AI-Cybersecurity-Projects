"""Temperature scaling (Guo et al., 2017) fitted on the validation split.

A single scalar T divides the softmax logits. It changes confidence but never
the predicted class, so classification metrics are unaffected and SHAP values
(computed in logit space) stay exactly consistent: calibrated contributions
are the raw ones divided by T.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def loss(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(y)), y].mean())

    res = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))
