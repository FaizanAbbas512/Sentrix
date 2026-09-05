"""
selftest.py  —  end-to-end smoke test, no dataset required
=========================================================

    python -m calm.selftest

Exercises every part of the M1..M4 chain on tiny hand-built inputs and prints
PASS/FAIL for a set of invariants. This is NOT an experiment — it only proves
the code is wired correctly and the maths behaves.
"""

from __future__ import annotations

import numpy as np

from .reliability import ReliabilityEstimator, ReliabilityFeatures
from .evidence import EvidenceFusion, MassFunction, discount, combine
from .calibration import Calibrator, expected_calibration_error
from .risk_control import LearnThenTest, faph_upper_bound, count_alarm_events
from .detector import CalmVAD

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f"  - {detail}" if detail and not cond else ""))


def main():
    print("CALM-VAD selftest\n" + "-" * 40)

    # ---- M1 ----------------------------------------------------------
    rel = ReliabilityEstimator()
    good_kp = np.hstack([np.random.default_rng(0).uniform(100, 300, (17, 2)),
                         np.full((17, 1), 0.95)])
    bad_kp = np.hstack([np.random.default_rng(1).uniform(100, 300, (17, 2)),
                        np.full((17, 1), 0.15)])
    r_good, _ = rel.reliability(1, good_kp, (100, 100, 200, 400), (480, 640))
    r_bad, _ = rel.reliability(2, bad_kp, (100, 100, 140, 180), (480, 640))
    check("M1 high-conf skeleton -> high r", r_good > 0.6, f"r={r_good:.2f}")
    check("M1 low-conf tiny skeleton -> low r", r_bad < r_good, f"{r_bad:.2f} < {r_good:.2f}")
    check("M1 r stays in [0,1]", 0 <= r_bad <= 1 and 0 <= r_good <= 1)

    # ---- M2 ----------------------------------------------------------
    m = MassFunction(0.6, 0.1, 0.3)
    check("M2 mass normalised", abs(m.mA + m.mN + m.mTheta - 1) < 1e-9)
    d = discount(m, 0.0)
    check("M2 discount r=0 -> vacuous", d.mTheta > 0.999)
    d1 = discount(m, 1.0)
    check("M2 discount r=1 -> unchanged", abs(d1.mA - m.mA) < 1e-9)
    fused, K = combine(MassFunction(0.7, 0.1, 0.2), MassFunction(0.1, 0.7, 0.2))
    check("M2 opposing evidence -> conflict K>0", K > 0.3, f"K={K:.2f}")

    fus = EvidenceFusion()
    hi = fus.fuse({"FALL": 0.9, "FIGHT": 0.0, "ABANDONED": 0.0,
                   "LOITERING": 0.0, "CROWD": 0.0}, r_pose=1.0)
    lo = fus.fuse({"FALL": 0.9, "FIGHT": 0.0, "ABANDONED": 0.0,
                   "LOITERING": 0.0, "CROWD": 0.0}, r_pose=0.1)
    check("M2 strong fall -> high belief", hi.bel_A > 0.6, f"bel={hi.bel_A:.2f}")
    check("M2 same cue, low pose reliability -> lower belief",
          lo.bel_A < hi.bel_A, f"{lo.bel_A:.2f} < {hi.bel_A:.2f}")
    check("M2 no cues -> near-zero belief",
          fus.fuse({k: 0.0 for k in ("FALL", "FIGHT", "ABANDONED", "LOITERING", "CROWD")}).bel_A < 0.15)

    # ---- M3 ----------------------------------------------------------
    rng = np.random.default_rng(42)
    n = 4000
    x = rng.uniform(0, 1, n)
    # a miscalibrated "belief": true P(incident) = x**2  -> needs a concave map
    y = (rng.uniform(0, 1, n) < x ** 2).astype(int)
    ece_before = expected_calibration_error(x, y)
    cal = Calibrator("isotonic").fit(x[: n // 2], y[: n // 2])
    p = cal.predict(x[n // 2:])
    ece_after = expected_calibration_error(p, y[n // 2:])
    check("M3 isotonic reduces ECE", ece_after < ece_before,
          f"{ece_before:.3f} -> {ece_after:.3f}")
    check("M3 monotonic map", np.all(np.diff(cal.predict(np.sort(x))) >= -1e-9))

    # ---- M4 ----------------------------------------------------------
    check("M4 Poisson bound grows with fewer hours",
          faph_upper_bound(3, 1.0) > faph_upper_bound(3, 10.0))
    check("M4 zero observed alarms still gives positive bound",
          faph_upper_bound(0, 5.0, 0.05) > 0)
    # a normal stream: mostly low p with rare bumps
    fps = 20
    normal = rng.uniform(0, 0.3, fps * 3600 * 2)          # 2 hours
    normal[rng.integers(0, len(normal), 40)] = rng.uniform(0.7, 0.95, 40)
    res = LearnThenTest(beta=5.0, delta=0.05, fps=fps).select(normal)
    ev_at_tau = count_alarm_events(normal, res.tau, fps)
    emp_faph = ev_at_tau / (len(normal) / fps / 3600.0)
    check("M4 chosen tau respects the budget empirically", emp_faph <= 5.0,
          f"{emp_faph:.2f}/h vs beta=5")
    check("M4 upper bound >= empirical rate", res.faph_hi_at_tau >= emp_faph - 1e-6)
    res_tight = LearnThenTest(beta=0.5, delta=0.05, fps=fps).select(normal)
    check("M4 tighter budget -> higher (more conservative) tau",
          res_tight.tau >= res.tau, f"{res_tight.tau:.2f} >= {res.tau:.2f}")

    # ---- orchestrator ---------------------------------------------
    calm = CalmVAD({"calm": {"fallback_tau": 0.5}})
    dec0 = calm.assess({"FALL": 0.0, "FIGHT": 0.0, "ABANDONED": 0.0,
                        "LOITERING": 0.0, "CROWD": 0.0})
    dec1 = calm.assess({"FALL": 0.95, "FIGHT": 0.0, "ABANDONED": 0.0,
                        "LOITERING": 0.0, "CROWD": 0.0}, pose_reliability=0.95)
    check("orchestrator quiet scene -> no alarm", not dec0.alarm)
    check("orchestrator strong fall -> alarm", dec1.alarm, f"p={dec1.p:.2f}")
    rec = dec1.record
    check("evidence record has the audit fields",
          {"p_incident", "belief_A", "conflict_K", "pose_reliability_r",
           "contributing_cues", "threshold_tau"} <= set(rec))
    check("evidence record names the firing cue",
          any(c["label"] == "FALL" for c in rec["contributing_cues"]))

    # fit calibration + budget, re-check
    bel = np.concatenate([np.full(500, 0.05), np.linspace(0.1, 0.95, 500)])
    lab = np.concatenate([np.zeros(500), (np.linspace(0.1, 0.95, 500) > 0.5).astype(int)])
    calm.fit_calibration(bel, lab)
    calm.set_threshold_from_normal_stream(np.random.default_rng(0).uniform(0, 0.3, 20 * 3600))
    check("orchestrator reports a guarantee after M4",
          "FAPH" in calm.summary()["guarantee"] or "NO threshold" in calm.summary()["guarantee"])

    print("-" * 40)
    print(f"  {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
