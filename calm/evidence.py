"""
evidence.py  —  M2 : reliability-discounted evidence fusion
==========================================================

Frame of discernment  Theta = {A, N}   (A = anomaly / incident, N = normal).
A mass function assigns belief to the subsets:  {A}, {N}, {A,N}=Theta.
Mass on Theta is "don't know" — it is NOT split between A and N.

Each SENTRIX cue is turned into one mass function from its own confirmation
state (how far past its threshold it is). Then:

  1. DISCOUNT each cue's mass by its source reliability. Two reliabilities
     multiply in:
        * r_cue   — how trustworthy the cue *type* is in general
                    (a fall held for 2 s is stronger evidence than a fleeting
                     crowd count); set in config.
        * r_pose  — M1's per-frame pose/tracking reliability. A cue built on a
                    broken skeleton gets discounted hard.
     Shafer discounting with rate  (1 - r):
        m'(X)     = r * m(X)                for every focal X != Theta
        m'(Theta) = r * m(Theta) + (1 - r)

  2. COMBINE the discounted cues with Dempster's rule of combination, plus a
     standing "normal prior" source. We keep the conflict mass K as an output:
     high K = the cues disagree = worth surfacing to an operator even if the
     fused belief is middling.

The result exposes  bel(A), pl(A), K  and the per-cue contributions, so every
alert decomposes into "which cue, how much, discounted by how much". That
decomposition is the auditable evidence record (see detector.py).

This module is pure numpy + stdlib. DS combination for a 2-element frame is a
handful of lines; we keep it explicit rather than pulling a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# ---- the three focal elements of a 2-hypothesis frame -----------------------
A = frozenset({"A"})
N = frozenset({"N"})
THETA = frozenset({"A", "N"})


@dataclass
class MassFunction:
    """m(A) + m(N) + m(Theta) == 1 (within a small tolerance)."""
    mA: float = 0.0
    mN: float = 0.0
    mTheta: float = 1.0
    label: str = ""          # which cue produced this (for the evidence record)

    def __post_init__(self):
        s = self.mA + self.mN + self.mTheta
        if s <= 0:
            self.mA, self.mN, self.mTheta = 0.0, 0.0, 1.0
            return
        # renormalise defensively
        self.mA, self.mN, self.mTheta = self.mA / s, self.mN / s, self.mTheta / s

    # Dempster-Shafer readouts for hypothesis A
    @property
    def bel_A(self) -> float:          # support that *must* go to A
        return self.mA

    @property
    def pl_A(self) -> float:           # support that *could* go to A
        return self.mA + self.mTheta

    @property
    def pignistic_A(self) -> float:    # betting probability (Smets)
        return self.mA + 0.5 * self.mTheta

    def as_dict(self) -> dict:
        return {"label": self.label, "mA": round(self.mA, 4),
                "mN": round(self.mN, 4), "mTheta": round(self.mTheta, 4)}


# --------------------------------------------------------------------------- #
#  cue  ->  mass                                                             #
# --------------------------------------------------------------------------- #
# For each cue we need two numbers:
#   strength s in [0, 1]  — how far past threshold the cue is (0 = not firing,
#                           1 = firmly confirmed). The analyzers already track
#                           the raw quantities; cues.py maps them to s.
#   rho in [0, 1]         — how much "not firing" counts as positive evidence
#                           of NORMAL. Loitering not firing barely means normal
#                           (rho low). A confirmed fall not firing means the
#                           person is upright (rho higher). Set per cue.
def cue_to_mass(strength: float, rho: float, r_cue: float, label: str = "") -> MassFunction:
    s = _clip01(strength)
    rho = _clip01(rho)
    r_cue = _clip01(r_cue)
    mA = s
    mN = (1.0 - s) * rho
    mTheta = max(1.0 - mA - mN, 0.0)
    m = MassFunction(mA, mN, mTheta, label=label)
    return discount(m, r_cue)          # apply the cue-type reliability now


def discount(m: MassFunction, r: float) -> MassFunction:
    """Shafer discounting at rate (1 - r). r=1 -> unchanged, r=0 -> vacuous."""
    r = _clip01(r)
    return MassFunction(
        mA=r * m.mA,
        mN=r * m.mN,
        mTheta=r * m.mTheta + (1.0 - r),
        label=m.label,
    )


def combine(m1: MassFunction, m2: MassFunction) -> tuple[MassFunction, float]:
    """
    Dempster's rule for the 2-hypothesis frame. Returns (combined, K) where
    K is the conflict mass that was normalised away.
    """
    # unnormalised intersections
    a = m1.mA * m2.mA + m1.mA * m2.mTheta + m1.mTheta * m2.mA
    n = m1.mN * m2.mN + m1.mN * m2.mTheta + m1.mTheta * m2.mN
    t = m1.mTheta * m2.mTheta
    K = m1.mA * m2.mN + m1.mN * m2.mA          # {A} ∩ {N} = empty -> conflict
    denom = 1.0 - K
    if denom <= 1e-12:
        # total conflict — fall back to a vacuous mass, report K = 1
        return MassFunction(0.0, 0.0, 1.0, label="conflict"), 1.0
    return MassFunction(a / denom, n / denom, t / denom, label="fused"), float(K)


# --------------------------------------------------------------------------- #
#  the fusion layer                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class FusionResult:
    mass: MassFunction                       # fused mass over {A, N, Theta}
    bel_A: float                             # belief in anomaly
    pl_A: float                              # plausibility of anomaly
    pignistic_A: float                       # betting probability
    conflict: float                          # max pairwise conflict seen (0..1)
    total_conflict: float                    # accumulated conflict across combos
    contributions: list = field(default_factory=list)   # per-cue dicts, most -> least
    r_pose: float = 1.0


class EvidenceFusion:
    """
    Combine a set of already-strength-scored cues into one FusionResult,
    discounting every cue by the per-frame pose reliability r_pose (M1).

    cfg["evidence"]:
        prior_normal   : standing mass on {N} before any cue (default 0.15)
        rho            : {cue_name: rho}          how "silence" implies normal
        r_cue          : {cue_name: r_cue}        cue-type reliability
    """

    DEFAULT_RHO = {
        "FALL": 0.35, "FIGHT": 0.25, "ABANDONED": 0.30,
        "LOITERING": 0.15, "CROWD": 0.10,
    }
    DEFAULT_RCUE = {
        "FALL": 0.90, "FIGHT": 0.75, "ABANDONED": 0.80,
        "LOITERING": 0.85, "CROWD": 0.60,
    }

    def __init__(self, cfg: dict | None = None):
        e = (cfg or {}).get("evidence", {}) if cfg else {}
        self.prior_normal = float(e.get("prior_normal", 0.15))
        self.rho = {**self.DEFAULT_RHO, **e.get("rho", {})}
        self.r_cue = {**self.DEFAULT_RCUE, **e.get("r_cue", {})}

    def fuse(self, strengths: dict[str, float], r_pose: float = 1.0) -> FusionResult:
        """
        strengths : {cue_name: s in [0, 1]}  from cues.py
        r_pose    : M1 per-frame pose/tracking reliability
        """
        r_pose = _clip01(r_pose)

        # standing "the world is usually normal" source
        prior = MassFunction(0.0, self.prior_normal, 1.0 - self.prior_normal,
                             label="prior_normal")
        fused = prior
        max_conflict = 0.0
        total_conflict = 0.0
        contributions = []

        # combine cues in descending strength so the record reads sensibly
        for name, s in sorted(strengths.items(), key=lambda kv: -kv[1]):
            if s <= 0:
                continue
            m_cue = cue_to_mass(
                strength=s,
                rho=self.rho.get(name, 0.2),
                r_cue=self.r_cue.get(name, 0.7),
                label=name,
            )
            # then discount again by the per-frame pose reliability
            m_cue = discount(m_cue, r_pose)
            fused, K = combine(fused, m_cue)
            max_conflict = max(max_conflict, K)
            total_conflict += K
            contributions.append({
                **m_cue.as_dict(),
                "strength": round(float(s), 4),
                "running_bel_A": round(fused.bel_A, 4),
            })

        return FusionResult(
            mass=fused,
            bel_A=fused.bel_A,
            pl_A=fused.pl_A,
            pignistic_A=fused.pignistic_A,
            conflict=float(max_conflict),
            total_conflict=float(total_conflict),
            contributions=contributions,
            r_pose=r_pose,
        )


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)
