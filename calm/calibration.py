"""
calibration.py  —  M3 : turn a fused belief into a real probability
==================================================================

The fused belief bel(A) from M2 is monotone in "how anomalous" but it is not a
probability: bel(A) = 0.7 does not mean "70% of windows scored 0.7 are real
incidents". M3 fits a post-hoc, monotone map on a held-out CALIBRATION split so
that it does.

Methods (pick in config, default isotonic):
  * isotonic     — non-parametric, monotone, no shape assumption. Best when the
                   calibration split is not tiny. (sklearn.isotonic)
  * platt        — logistic fit  p = sigmoid(a*x + b). Robust on small splits.
  * temperature  — single scalar T on the logit of x. Fewest parameters.

We also provide the evaluation side that peers do NOT report for edge VAD:
  * expected_calibration_error  — binned ECE and adaptive (equal-mass) ECE
  * reliability_curve           — points for a reliability diagram
  * brier_score

Fit on the calibration split, freeze, then report ECE on the TEST split.
"""

from __future__ import annotations

import math
import numpy as np


# --------------------------------------------------------------------------- #
#  metrics                                                                   #
# --------------------------------------------------------------------------- #
def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15,
                               adaptive: bool = False) -> float:
    """
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    adaptive=False : equal-width bins on [0, 1]
    adaptive=True  : equal-mass bins (each bin holds ~n/n_bins points) — more
                     honest when scores pile up in one region.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    n = len(p)
    if n == 0:
        return float("nan")

    if adaptive:
        order = np.argsort(p)
        edges_idx = np.linspace(0, n, n_bins + 1).astype(int)
        bins = [order[edges_idx[i]:edges_idx[i + 1]] for i in range(n_bins)]
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = [np.where((p >= edges[i]) & (p < edges[i + 1]))[0] for i in range(n_bins)]
        bins[-1] = np.where((p >= edges[-2]) & (p <= edges[-1]))[0]

    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        conf = p[idx].mean()
        acc = y[idx].mean()
        ece += (len(idx) / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Returns (bin_conf, bin_acc, bin_count) for plotting a reliability diagram."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((p >= lo) & (p < hi))[0] if i < n_bins - 1 else \
              np.where((p >= lo) & (p <= hi))[0]
        if len(idx) == 0:
            conf.append((lo + hi) / 2); acc.append(np.nan); cnt.append(0)
        else:
            conf.append(float(p[idx].mean()))
            acc.append(float(y[idx].mean()))
            cnt.append(int(len(idx)))
    return np.array(conf), np.array(acc), np.array(cnt)


# --------------------------------------------------------------------------- #
#  the calibrator                                                            #
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._fitted = False
        self._iso = None
        self._a = 1.0
        self._b = 0.0
        self._T = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Calibrator":
        """x = raw fused belief in [0, 1]; y = {0, 1} incident label on the same window."""
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        if len(x) < 10 or y.min() == y.max():
            # not enough to calibrate -- identity map, warn
            print(f"[calibration] weak calibration set (n={len(x)}, "
                  f"pos={int(y.sum())}); using identity map")
            self.method = "identity"
            self._fitted = True
            return self

        if self.method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
                self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                               out_of_bounds="clip")
                self._iso.fit(x, y)
            except ImportError:
                print("[calibration] sklearn missing -- falling back to platt")
                self.method = "platt"

        if self.method == "platt":
            a, b = 1.0, 0.0
            z = _logit(x)
            for _ in range(200):                     # newton-ish gradient steps
                p = _sigmoid(a * z + b)
                ga = np.mean((p - y) * z)
                gb = np.mean(p - y)
                a -= 0.5 * ga
                b -= 0.5 * gb
            self._a, self._b = float(a), float(b)

        if self.method == "temperature":
            T = 1.0
            z = _logit(x)
            for _ in range(200):
                p = _sigmoid(z / T)
                g = np.mean((p - y) * (-z / (T * T)))
                T -= 0.5 * g
                T = max(T, 1e-3)
            self._T = float(T)

        self._fitted = True
        return self

    def predict(self, x) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator.fit() first")
        x = np.asarray(x, float).ravel()
        if self.method == "identity":
            return np.clip(x, 0, 1)
        if self.method == "isotonic":
            return np.clip(self._iso.predict(x), 0, 1)
        if self.method == "platt":
            return _sigmoid(self._a * _logit(x) + self._b)
        if self.method == "temperature":
            return _sigmoid(_logit(x) / self._T)
        return np.clip(x, 0, 1)

    def __call__(self, x):
        scalar = np.isscalar(x)
        out = self.predict(np.atleast_1d(x))
        return float(out[0]) if scalar else out

    def report(self, x, y, n_bins: int = 15) -> dict:
        """ECE / adaptive-ECE / Brier before vs after calibration."""
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        p = self.predict(x)
        return {
            "method": self.method,
            "n": int(len(x)),
            "ece_raw": expected_calibration_error(x, y, n_bins),
            "ece_cal": expected_calibration_error(p, y, n_bins),
            "aece_raw": expected_calibration_error(x, y, n_bins, adaptive=True),
            "aece_cal": expected_calibration_error(p, y, n_bins, adaptive=True),
            "brier_raw": brier_score(x, y),
            "brier_cal": brier_score(p, y),
        }
