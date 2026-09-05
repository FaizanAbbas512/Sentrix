"""
cues.py  —  adapters : SENTRIX analyzer state  ->  cue strength s in [0, 1]
=========================================================================

M2 (evidence.py) wants each cue as a single "confirmation strength":
    0   = cue not firing at all
    1   = cue firmly confirmed (well past its threshold, held long enough)

The existing analyzers in src/analyzers/ already track the raw quantities
(dwell time, unattended time, agitation, torso angle, person count). This
module turns those raw quantities into a bounded, comparable strength without
changing the analyzers themselves.

There are two ways to use it:

  A) LIVE  — call `strength_from_pipeline(pipe, ...)` inside a running
     SENTRIX Pipeline; it reads the analyzers' public state.

  B) OFFLINE / benchmark — you have per-frame skeletons from a dataset and no
     running Pipeline. Use `CueBank`, a light re-implementation of the five
     cue timers that consumes (track_id, keypoints, bbox) per frame. This is
     what harness.py uses so evaluation does not depend on OpenCV / a camera.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

CUE_NAMES = ("FALL", "FIGHT", "ABANDONED", "LOITERING", "CROWD")


# --------------------------------------------------------------------------- #
#  small helpers                                                             #
# --------------------------------------------------------------------------- #
def _ramp(value: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear between. hi>lo required."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def _torso_len(kp: np.ndarray) -> float:
    sh = kp[[5, 6], :2].mean(axis=0)
    hp = kp[[11, 12], :2].mean(axis=0)
    d = float(np.linalg.norm(sh - hp))
    return d if d > 1 else 100.0


def _centroid(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


# --------------------------------------------------------------------------- #
#  A) live adapter                                                           #
# --------------------------------------------------------------------------- #
def strength_from_pipeline(loiter_ids, abandon_ids, fall_ids, fight_ids,
                           people, crowd_threshold,
                           loiter_obj=None, abandon_obj=None, now=None) -> dict:
    """
    Map the sets/counts a SENTRIX Pipeline already computes into strengths.
    Where we can read a duration we ramp it; otherwise a fired set -> 1.0.
    """
    s = {k: 0.0 for k in CUE_NAMES}

    if fall_ids:
        s["FALL"] = 1.0                       # fall.py only flags after confirm_seconds
    if fight_ids:
        s["FIGHT"] = 1.0                      # fight.py only flags after confirm_seconds

    if loiter_ids and loiter_obj is not None and now is not None:
        d = max(loiter_obj.dwell_time(tid, now) for tid in loiter_ids)
        s["LOITERING"] = _ramp(d, loiter_obj.dwell, loiter_obj.dwell * 2.5)
    elif loiter_ids:
        s["LOITERING"] = 0.8

    if abandon_ids and abandon_obj is not None and now is not None:
        u = max(abandon_obj.unattended_time(tid, now) for tid in abandon_ids)
        s["ABANDONED"] = _ramp(u, abandon_obj.unattended, abandon_obj.unattended * 2.5)
    elif abandon_ids:
        s["ABANDONED"] = 0.8

    if people >= crowd_threshold:
        s["CROWD"] = _ramp(people, crowd_threshold, crowd_threshold * 2.0)

    return s


# --------------------------------------------------------------------------- #
#  B) offline cue bank (no OpenCV, no camera)                                #
# --------------------------------------------------------------------------- #
class CueBank:
    """
    Minimal re-implementation of the five SENTRIX cue timers for benchmark use.
    Feed it one frame at a time as a list of detections; get back strengths.

    detection = dict(track_id=int, keypoints=(17,3) array or None,
                     bbox=(x1,y1,x2,y2), is_person=bool, label=str)
    """

    def __init__(self, cfg: dict, fps: float = 20.0):
        self.fps = float(fps)
        self.t = 0.0
        self.dt = 1.0 / self.fps

        lo = cfg.get("loitering", {})
        ab = cfg.get("abandoned", {})
        cr = cfg.get("crowd", {})
        be = cfg.get("behaviour", {})
        fi = cfg.get("fight", {})

        self.dwell = float(lo.get("dwell_seconds", 15))
        self.loiter_tol = float(lo.get("movement_tolerance", 60))
        self.unattended = float(ab.get("unattended_seconds", 12))
        self.owner_radius = float(ab.get("owner_radius", 150))
        self.obj_tol = float(ab.get("movement_tolerance", 40))
        self.crowd_th = float(cr.get("threshold", 8))
        self.fall_confirm = float(be.get("fall_confirm_seconds", 1.2))
        self.agit_th = float(be.get("agitation_threshold", 0.5))
        self.hist_len = int(be.get("history_frames", 5))
        self.proximity = float(fi.get("proximity", 130))
        self.fight_confirm = float(fi.get("confirm_seconds", 1.0))
        self.fight_agit_th = float(fi.get("agitation_threshold", 0.6))

        self._loiter = {}       # tid -> {anchor, first_seen, last}
        self._abandon = {}      # tid -> {anchor, unattended_since, last}
        self._kp_hist = {}      # tid -> deque[(pts, torso, t)]
        self._fallen_since = {}
        self._fight_since = {}  # frozenset(pair) -> t

    # ------------------------------------------------------------------ step
    def step(self, detections: list[dict]) -> tuple[dict, dict]:
        """
        Returns (strengths, aux) where
            strengths = {cue: s in [0,1]}
            aux       = {"agitation": {tid: a}, "posture": {tid: str},
                         "fallen_ids": set, "fight_ids": set,
                         "loiter_ids": set, "abandoned_ids": set, "people": int}
        """
        self.t += self.dt
        now = self.t
        persons = [d for d in detections if d.get("is_person", True)]
        objects = [d for d in detections if not d.get("is_person", True)]
        s = {k: 0.0 for k in CUE_NAMES}

        # ---- behaviour: posture + agitation per person -------------------
        agit, posture, fallen_ids = {}, {}, set()
        for d in persons:
            tid = d["track_id"]
            kp = d.get("keypoints")
            kp = None if kp is None else np.asarray(kp, float).reshape(-1, 3)
            posture[tid] = self._posture(kp, d["bbox"])
            agit[tid] = self._agitation(tid, kp, d["bbox"], now)
            if posture[tid] == "fallen":
                self._fallen_since.setdefault(tid, now)
                if now - self._fallen_since[tid] >= self.fall_confirm:
                    fallen_ids.add(tid)
            else:
                self._fallen_since.pop(tid, None)
        if fallen_ids:
            held = max(now - self._fallen_since[t] for t in fallen_ids)
            s["FALL"] = _ramp(held, self.fall_confirm, self.fall_confirm + 2.0)

        # ---- fight: close pair, both agitated, held --------------------
        fight_ids = set()
        active_pairs = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a, b = persons[i], persons[j]
                ca, cb = _centroid(a["bbox"]), _centroid(b["bbox"])
                dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                if dist > self.proximity:
                    continue
                if agit.get(a["track_id"], 0) < self.fight_agit_th:
                    continue
                if agit.get(b["track_id"], 0) < self.fight_agit_th:
                    continue
                pair = frozenset((a["track_id"], b["track_id"]))
                active_pairs.add(pair)
                self._fight_since.setdefault(pair, now)
                if now - self._fight_since[pair] >= self.fight_confirm:
                    fight_ids.update(pair)
        for p in [p for p in self._fight_since if p not in active_pairs]:
            del self._fight_since[p]
        if fight_ids:
            held = max((now - self._fight_since[p]) for p in active_pairs) if active_pairs else self.fight_confirm
            s["FIGHT"] = _ramp(held, self.fight_confirm, self.fight_confirm + 2.0)

        # ---- loitering ------------------------------------------------
        loiter_ids = set()
        seen = set()
        for d in persons:
            tid = d["track_id"]; seen.add(tid)
            cx, cy = _centroid(d["bbox"])
            tr = self._loiter.get(tid)
            if tr is None:
                self._loiter[tid] = {"anchor": (cx, cy), "first": now, "last": now}
            else:
                tr["last"] = now
                if math.hypot(cx - tr["anchor"][0], cy - tr["anchor"][1]) > self.loiter_tol:
                    tr["anchor"] = (cx, cy); tr["first"] = now
                elif now - tr["first"] >= self.dwell:
                    loiter_ids.add(tid)
        for tid in [t for t in self._loiter if t not in seen and now - self._loiter[t]["last"] > 3]:
            del self._loiter[tid]
        if loiter_ids:
            d = max(now - self._loiter[t]["first"] for t in loiter_ids)
            s["LOITERING"] = _ramp(d, self.dwell, self.dwell * 2.5)

        # ---- abandoned object --------------------------------------
        abandoned_ids = set()
        seen_o = set()
        pcents = [_centroid(p["bbox"]) for p in persons]
        for o in objects:
            tid = o["track_id"]; seen_o.add(tid)
            ox, oy = _centroid(o["bbox"])
            attended = any(math.hypot(ox - px, oy - py) <= self.owner_radius
                           for (px, py) in pcents)
            tr = self._abandon.get(tid)
            if tr is None:
                tr = self._abandon[tid] = {"anchor": (ox, oy),
                                           "unattended_since": None, "last": now}
            tr["last"] = now
            if math.hypot(ox - tr["anchor"][0], oy - tr["anchor"][1]) > self.obj_tol:
                tr["anchor"] = (ox, oy); tr["unattended_since"] = None
                continue
            if attended:
                tr["unattended_since"] = None
            else:
                tr["unattended_since"] = tr["unattended_since"] or now
                if now - tr["unattended_since"] >= self.unattended:
                    abandoned_ids.add(tid)
        for tid in [t for t in self._abandon if t not in seen_o and now - self._abandon[t]["last"] > 3]:
            del self._abandon[tid]
        if abandoned_ids:
            u = max(now - self._abandon[t]["unattended_since"] for t in abandoned_ids)
            s["ABANDONED"] = _ramp(u, self.unattended, self.unattended * 2.5)

        # ---- crowd --------------------------------------------------
        people = len(persons)
        if people >= self.crowd_th:
            s["CROWD"] = _ramp(people, self.crowd_th, self.crowd_th * 2.0)

        aux = {"agitation": agit, "posture": posture, "fallen_ids": fallen_ids,
               "fight_ids": fight_ids, "loiter_ids": loiter_ids,
               "abandoned_ids": abandoned_ids, "people": people}
        return s, aux

    # ------------------------------------------------------------------ posture / agitation
    def _posture(self, kp, bbox) -> str:
        if kp is not None and kp[[5, 6, 11, 12], 2].min() >= 0.3:
            sh = kp[[5, 6], :2].mean(axis=0)
            hp = kp[[11, 12], :2].mean(axis=0)
            dx, dy = abs(sh[0] - hp[0]), abs(sh[1] - hp[1])
            return "fallen" if dx > dy * 1.2 else "upright"
        x1, y1, x2, y2 = bbox
        h = max(y2 - y1, 1.0)
        return "fallen" if (x2 - x1) / h >= 1.15 else "unknown"

    def _agitation(self, tid, kp, bbox, now) -> float:
        if kp is not None:
            torso = _torso_len(kp)
            pts = [tuple(kp[i, :2]) for i in (9, 10, 15, 16) if kp[i, 2] >= 0.3]
        else:
            torso, pts = 100.0, []
        if not pts:
            pts = [_centroid(bbox)]
        hist = self._kp_hist.setdefault(tid, deque(maxlen=self.hist_len))
        hist.append((pts, torso, now))
        if len(hist) < 2:
            return 0.0
        speeds = []
        for k in range(1, len(hist)):
            p0, _, _ = hist[k - 1]
            p1, tl, _ = hist[k]
            n = min(len(p0), len(p1))
            if n == 0:
                continue
            d = sum(math.hypot(p1[j][0] - p0[j][0], p1[j][1] - p0[j][1])
                    for j in range(n)) / n
            speeds.append(d / tl)
        return round(sum(speeds) / len(speeds), 3) if speeds else 0.0
