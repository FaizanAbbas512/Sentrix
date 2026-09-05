r"""
detector.py  —  the CALM-VAD orchestrator
=========================================

Puts M1..M4 together and produces the final output: not a score, an
EVIDENCE RECORD.

    strengths (from cues.py)           M1  ReliabilityEstimator  -> r_pose
              \                        /
               M2  EvidenceFusion  ---+---> bel(A), pl(A), conflict K, per-cue mass
                                      |
               M3  Calibrator  -------+---> p  = P(incident)   (calibrated)
                                      |
               M4  LearnThenTest tau -+---> alarm = p >= tau   (budgeted guarantee)

Typical lifecycle:

    calm = CalmVAD(cfg)
    # ---- offline, once: fit M3 and pick M4's tau on calibration data ----
    calm.fit_calibration(belief_scores, incident_labels)      # M3
    calm.set_threshold_from_normal_stream(p_on_normal_stream) # M4
    # ---- online: per frame ----
    decision = calm.assess(strengths, pose_reliability=r)
    if decision.alarm: ...
    log(decision.record)          # the auditable JSON

`assess()` also works before M3/M4 are fitted — it just returns the raw belief
as `p` and uses a config fallback threshold, so you can run the live pipeline
immediately and calibrate later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time
import json

from .evidence import EvidenceFusion, FusionResult
from .calibration import Calibrator
from .risk_control import LearnThenTest, RiskControlResult


@dataclass
class CalmDecision:
    p: float                       # calibrated P(incident) for this window
    alarm: bool                    # p >= tau
    level: str                     # NONE / LOW / MEDIUM / HIGH (banded p, for UI parity)
    belief_A: float                # raw fused belief before calibration
    plausibility_A: float
    conflict: float                # max pairwise DS conflict (cues disagreeing)
    r_pose: float                  # M1 reliability applied this frame
    tau: float                     # threshold in force
    contributions: list = field(default_factory=list)
    calibrated: bool = False
    budgeted: bool = False
    ts: float = 0.0

    @property
    def record(self) -> dict:
        """The auditable evidence record — dump this to the alert log."""
        return {
            "ts": self.ts,
            "p_incident": round(self.p, 4),
            "alarm": self.alarm,
            "level": self.level,
            "threshold_tau": round(self.tau, 4),
            "belief_A": round(self.belief_A, 4),
            "plausibility_A": round(self.plausibility_A, 4),
            "conflict_K": round(self.conflict, 4),
            "pose_reliability_r": round(self.r_pose, 4),
            "contributing_cues": self.contributions,
            "calibrated": self.calibrated,
            "budget_guaranteed": self.budgeted,
        }

    def to_json(self) -> str:
        return json.dumps(self.record, ensure_ascii=False)


class CalmVAD:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        c = cfg.get("calm", {})

        self.fusion = EvidenceFusion(cfg)
        self.calibrator = Calibrator(method=c.get("calibration_method", "isotonic"))
        self._calibrated = False

        # M4 config
        self.beta = float(c.get("budget_faph", 5.0))          # false alarms / hour
        self.delta = float(c.get("delta", 0.05))
        self.fps = float(c.get("fps", cfg.get("fps", 20.0)))
        self._ltt: LearnThenTest | None = None
        self._rc_result: RiskControlResult | None = None
        self.tau = float(c.get("fallback_tau", 0.6))          # used until M4 runs
        self._budgeted = False

        # banding for UI parity with src/fusion.py
        self.band_med = float(c.get("band_medium", 0.4))
        self.band_high = float(c.get("band_high", 0.7))

    # ------------------------------------------------------------------ M3 fit
    def fit_calibration(self, belief_scores, incident_labels) -> dict:
        """belief_scores = raw bel(A) per window; incident_labels in {0,1}."""
        self.calibrator.fit(belief_scores, incident_labels)
        self._calibrated = True
        return self.calibrator.report(belief_scores, incident_labels)

    # ------------------------------------------------------------------ M4 select
    def set_threshold_from_normal_stream(self, p_normal, hours: float | None = None,
                                         beta: float | None = None,
                                         delta: float | None = None) -> RiskControlResult:
        """
        p_normal = calibrated probabilities on a CONFIRMED-NORMAL calibration
        stream. Picks tau to guarantee <= beta false alarms/hour at level delta.
        """
        self.beta = float(beta) if beta is not None else self.beta
        self.delta = float(delta) if delta is not None else self.delta
        self._ltt = LearnThenTest(beta=self.beta, delta=self.delta, fps=self.fps)
        self._rc_result = self._ltt.select(p_normal, hours=hours)
        self.tau = self._rc_result.tau
        self._budgeted = True
        return self._rc_result

    # ------------------------------------------------------------------ online
    def assess(self, strengths: dict, pose_reliability: float = 1.0,
               now: float | None = None) -> CalmDecision:
        """
        strengths        : {cue_name: s in [0,1]}  (cues.py)
        pose_reliability : r_pose from M1 (mean over the people who drove the cues;
                           1.0 if you have no pose signal, e.g. crowd-only frame)
        """
        now = now or time.time()
        fr: FusionResult = self.fusion.fuse(strengths, r_pose=pose_reliability)

        if self._calibrated:
            p = float(self.calibrator(fr.bel_A))
        else:
            p = float(fr.bel_A)             # uncalibrated fallback

        level = ("HIGH" if p >= self.band_high else
                 "MEDIUM" if p >= self.band_med else
                 "LOW" if p > 0.05 else "NONE")

        return CalmDecision(
            p=p,
            alarm=bool(p >= self.tau),
            level=level,
            belief_A=fr.bel_A,
            plausibility_A=fr.pl_A,
            conflict=fr.conflict,
            r_pose=fr.r_pose,
            tau=self.tau,
            contributions=fr.contributions,
            calibrated=self._calibrated,
            budgeted=self._budgeted,
            ts=now,
        )

    # ------------------------------------------------------------------ introspection
    def summary(self) -> dict:
        return {
            "calibrated": self._calibrated,
            "calibration_method": self.calibrator.method,
            "budgeted": self._budgeted,
            "tau": round(self.tau, 4),
            "beta_faph": self.beta,
            "delta": self.delta,
            "guarantee": self._rc_result.guarantee if self._rc_result else "none (fallback tau)",
        }
