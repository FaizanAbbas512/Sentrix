"""
calm/  —  CALM-VAD decision layer for SENTRIX
============================================

SENTRIX (src/) gives us perception + 5 hand-crafted behavioural cues, then
ends in two weak links:

    weighted sum  ->  hand-tuned threshold  ->  alert

CALM-VAD replaces those two links with a 4-part chain:

    M1  reliability.py   : how much do we trust this frame's pose / tracking?
    M2  evidence.py      : fuse the cues as BELIEF MASS, each discounted by M1
    M3  calibration.py   : map fused belief -> a real probability (ECE-verified)
    M4  risk_control.py   : pick the threshold that satisfies an operator's
                            "<= beta false alarms / hour" budget, with a
                            distribution-free guarantee (Learn-then-Test)

The output is not a score but an *evidence record* (see detector.CalmDecision).

Nothing here touches the detector or the analyzers — CALM-VAD consumes their
per-track state and produces the final decision. Run `python -m calm.selftest`
for an end-to-end synthetic demo that needs no dataset.
"""

from .reliability import ReliabilityEstimator, ReliabilityFeatures
from .evidence import MassFunction, discount, combine, EvidenceFusion
from .calibration import Calibrator, expected_calibration_error, reliability_curve
from .risk_control import LearnThenTest, faph_upper_bound
from .detector import CalmVAD, CalmDecision

__all__ = [
    "ReliabilityEstimator", "ReliabilityFeatures",
    "MassFunction", "discount", "combine", "EvidenceFusion",
    "Calibrator", "expected_calibration_error", "reliability_curve",
    "LearnThenTest", "faph_upper_bound",
    "CalmVAD", "CalmDecision",
]
