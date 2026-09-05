"""
risk_control.py  —  M4 : an alarm-rate budget with a guarantee
=============================================================

The operator names a budget:  "no more than  beta  false alarms per hour"
plus an error level  delta  (e.g. 0.05).

M4 returns the *most sensitive* probability threshold  tau  such that, with
probability at least  1 - delta,  the true false-alarm rate at  tau  is  <= beta.

This is Learn-then-Test (Angelopoulos et al., 2021) specialised to a 1-D
threshold family and a rate-type risk:

  * Risk to control:  FAPH(tau) = (false alarm events per hour on a NORMAL stream)
  * On a calibration stream of  T_hours  of confirmed-normal video, count the
    false-alarm EVENTS  N(tau)  the system would raise at threshold  tau
    (event = a maximal run of frames with p >= tau, after debouncing).
  * Model  N(tau) ~ Poisson(FAPH(tau) * T).  The exact upper 1-delta confidence
    bound on the rate is
        FAPH_hi(tau) = 0.5 * chi2_ppf(1 - delta, 2 * (N + 1)) / T
    (Garwood / "exact" Poisson interval; no distributional approximation beyond
     Poisson-ness of rare independent alarm events).
  * Multiple thresholds are tested with FIXED-SEQUENCE testing: order tau from
    most conservative (high) to least (low); walk down; stop at the first tau
    whose FAPH_hi exceeds beta. Everything admitted before the stop is valid at
    family-wise level delta with NO Bonferroni penalty (a standard LTT trick,
    valid because the nulls are nested: if FAPH(tau) <= beta fails, it fails for
    every smaller tau too).

If scipy is unavailable we use a Wilson-Hilferty approximation to the chi-square
quantile — accurate to <1% in the range that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


# --------------------------------------------------------------------------- #
#  chi-square quantile (scipy if present, else a good approximation)         #
# --------------------------------------------------------------------------- #
def _chi2_ppf(q: float, df: float) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(q, df))
    except Exception:
        # Wilson-Hilferty: chi2_q ≈ df * (1 - 2/(9df) + z_q * sqrt(2/(9df)))^3
        z = _norm_ppf(q)
        t = 2.0 / (9.0 * df)
        return float(df * (1.0 - t + z * math.sqrt(t)) ** 3)


def _norm_ppf(q: float) -> float:
    """Acklam's rational approximation to the standard-normal quantile."""
    if q <= 0:
        return -math.inf
    if q >= 1:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > 1 - plow:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    t = r * r
    return (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5]) * r / \
           (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1)


def faph_upper_bound(n_false_alarms: int, hours: float, delta: float = 0.05) -> float:
    """Exact Poisson upper (1 - delta) confidence bound on false alarms / hour."""
    if hours <= 0:
        return math.inf
    n = max(int(n_false_alarms), 0)
    return 0.5 * _chi2_ppf(1.0 - delta, 2 * (n + 1)) / hours


# --------------------------------------------------------------------------- #
#  event debouncing on a probability stream                                  #
# --------------------------------------------------------------------------- #
def count_alarm_events(p: np.ndarray, tau: float, fps: float,
                       min_gap_s: float = 3.0, min_len_s: float = 0.3) -> int:
    """
    An alarm EVENT = a maximal run of frames with p >= tau, lasting at least
    min_len_s, where runs separated by a gap shorter than min_gap_s are merged
    (matches how alerts.py cooldown behaves — operators see one alarm, not one
    per frame).
    """
    p = np.asarray(p, float)
    above = p >= tau
    if not above.any():
        return 0
    gap_frames = max(int(round(min_gap_s * fps)), 1)
    min_len_frames = max(int(round(min_len_s * fps)), 1)

    events = 0
    i = 0
    n = len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        # look ahead: merge if the next 'above' run starts within gap_frames
        k = j
        while k < n and (k - j) < gap_frames:
            if above[k]:
                j = k + 1
                while j < n and above[j]:
                    j += 1
                k = j
            else:
                k += 1
        if (j - i) >= min_len_frames:
            events += 1
        i = j
    return events


# --------------------------------------------------------------------------- #
#  Learn-then-Test                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class RiskControlResult:
    tau: float                     # chosen threshold (most sensitive admissible)
    beta: float                    # requested budget (false alarms / hour)
    delta: float                   # requested error level
    faph_hi_at_tau: float          # upper bound on FAPH at the chosen tau
    admissible_taus: list          # every tau that passed, high -> low
    grid: list                     # [(tau, n_events, faph_hat, faph_hi, admitted)]
    calib_hours: float
    guarantee: str


class LearnThenTest:
    """
    Usage:
        ltt = LearnThenTest(beta=5.0, delta=0.05, fps=20)
        res = ltt.select(p_normal_stream)      # p on a confirmed-NORMAL stream
        # deploy res.tau
    """

    def __init__(self, beta: float, delta: float = 0.05, fps: float = 20.0,
                 grid: np.ndarray | None = None, min_gap_s: float = 3.0):
        self.beta = float(beta)
        self.delta = float(delta)
        self.fps = float(fps)
        self.min_gap_s = float(min_gap_s)
        self.grid = np.linspace(0.99, 0.05, 48) if grid is None else np.asarray(grid, float)
        # ensure descending (fixed-sequence walks high -> low)
        self.grid = np.sort(self.grid)[::-1]

    def select(self, p_normal: np.ndarray, hours: float | None = None) -> RiskControlResult:
        p_normal = np.asarray(p_normal, float)
        T = float(hours) if hours is not None else len(p_normal) / self.fps / 3600.0
        if T <= 0:
            raise ValueError("calibration stream has zero duration")

        rows = []
        admissible = []
        stopped = False
        for tau in self.grid:
            n_ev = count_alarm_events(p_normal, tau, self.fps, self.min_gap_s)
            faph_hat = n_ev / T
            faph_hi = faph_upper_bound(n_ev, T, self.delta)
            admit = (not stopped) and (faph_hi <= self.beta)
            if admit:
                admissible.append(float(tau))
            elif not stopped:
                stopped = True          # fixed-sequence: first failure ends the walk
            rows.append((float(tau), int(n_ev), float(faph_hat),
                         float(faph_hi), bool(admit)))

        if admissible:
            tau_star = min(admissible)          # most sensitive that still passed
            faph_hi_star = next(r[3] for r in rows if r[0] == tau_star)
            guarantee = (f"P(FAPH(tau={tau_star:.3f}) <= {self.beta:g}) "
                         f">= {1 - self.delta:.2f}  over {T:.2f} normal-stream hours")
        else:
            tau_star = float(self.grid[0])       # nothing admissible — most conservative
            faph_hi_star = rows[0][3]
            guarantee = (f"NO threshold met the budget beta={self.beta:g} at "
                         f"delta={self.delta:g}; returning the most conservative "
                         f"tau={tau_star:.3f}. Collect more normal-stream hours or "
                         f"raise beta.")

        return RiskControlResult(
            tau=tau_star, beta=self.beta, delta=self.delta,
            faph_hi_at_tau=faph_hi_star, admissible_taus=admissible,
            grid=rows, calib_hours=T, guarantee=guarantee,
        )
