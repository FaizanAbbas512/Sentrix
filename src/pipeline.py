"""
pipeline.py
-----------
Sab kuch yahan jurta hai. Frame andar -> annotated frame + stats bahar.

Ab POSE-based: har person ka skeleton nikalta hai, behaviour analyzer usse
posture (khada/gira) + agitation (limb speed) nikalta hai. Yeh box nahi,
ASAL body dekhta hai.

Flow:
  frame -> Detector(pose) -> Behaviour(posture+agitation)
        -> [Fall(from posture), Fight(from agitation), Loitering, Crowd]
        -> Fusion -> Alerts -> draw skeleton + boxes + threat banner
"""

import json
import os
import time
import cv2

from src.detector import Detector
from src.analyzers import (
    LoiteringAnalyzer, AbandonedAnalyzer, CrowdAnalyzer,
    FightAnalyzer, BehaviourAnalyzer,
)
from src.fusion import ThreatFusion
from src.alerts import AlertManager

# CALM-VAD decision layer (calm/) — optional, enabled via config: calm.enabled
try:
    from calm import CalmVAD, ReliabilityEstimator
    from calm.cues import strength_from_pipeline
    _CALM_AVAILABLE = True
except Exception:
    _CALM_AVAILABLE = False

COL_NORMAL = (120, 200, 90)
COL_OBJECT = (200, 160, 90)
COL_ALERT = (60, 60, 235)
COL_FALL = (60, 130, 240)
COL_FIGHT = (60, 60, 235)
COL_SKELETON = (230, 200, 120)

LEVEL_COLOR = {
    "NONE": (120, 200, 90), "LOW": (200, 200, 90),
    "MEDIUM": (60, 180, 235), "HIGH": (60, 60, 235),
}

# COCO skeleton connections (kaunsa point kaunse se juda)
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.behaviour = BehaviourAnalyzer(cfg)
        self.loiter = LoiteringAnalyzer(cfg)
        self.abandon = AbandonedAnalyzer(cfg)
        self.crowd = CrowdAnalyzer(cfg)
        self.fight = FightAnalyzer(cfg)
        self.fusion = ThreatFusion(cfg)
        self.alerts = AlertManager(cfg)

        # ---- CALM-VAD decision layer (optional) ----
        self.calm_on = bool(cfg.get("calm", {}).get("enabled", False)) and _CALM_AVAILABLE
        if self.calm_on:
            self.calm = CalmVAD(cfg)
            self.rel = ReliabilityEstimator(cfg)
            self._evidence_log = os.path.join(
                os.path.dirname(cfg["alerts"]["log_file"]), "evidence.jsonl")
        elif cfg.get("calm", {}).get("enabled") and not _CALM_AVAILABLE:
            print("[pipeline] calm.enabled set but the calm package failed to import")

        self.detect_every = 2
        self._frame_no = 0
        self._cached = []
        self._cached_behav = {}

        self.stats = {
            "people": 0, "crowded": False, "loitering": 0, "abandoned": 0,
            "fall": 0, "fight": 0, "threat": "NONE", "threat_score": 0,
            "fps": 0.0, "totals": self.alerts.total_counts,
            "calm": None,
        }
        self._t_prev = time.time()

    def process(self, frame):
        now = time.time()
        self._frame_no += 1

        if self._frame_no % self.detect_every == 0 or not self._cached:
            self._cached = self.detector.detect(frame)
            self._cached_behav = self.behaviour.update(self._cached, now)
        detections = self._cached
        behav = self._cached_behav

        # fall ids from behaviour (posture-based)
        fall_ids = {tid for tid, b in behav.items() if b["fallen"]}
        fight_ids = self.fight.update(detections, behav, now)
        loiter_ids = self.loiter.update(detections, now)
        abandon_ids = self.abandon.update(detections, now)
        people, is_crowd = self.crowd.update(detections)

        signals = {
            "FALL": len(fall_ids), "FIGHT": len(fight_ids),
            "ABANDONED": len(abandon_ids), "LOITERING": len(loiter_ids),
            "CROWD": 1 if is_crowd else 0,
        }
        level, score, _ = self.fusion.assess(signals)

        # ---- CALM-VAD: reliability-discounted, calibrated, budgeted decision ----
        if self.calm_on:
            self._calm_step(detections, fall_ids, fight_ids, loiter_ids,
                            abandon_ids, people, is_crowd, frame, now)

        for tid in fall_ids:
            self.alerts.fire("FALL", tid, "person gira", frame, now)
        for tid in fight_ids:
            self.alerts.fire("FIGHT", tid, "aggression", frame, now)
        for tid in abandon_ids:
            self.alerts.fire("ABANDONED", tid,
                             f"{self.abandon.unattended_time(tid, now):.0f}s lawaaris",
                             frame, now)
        for tid in loiter_ids:
            self.alerts.fire("LOITERING", tid,
                             f"{self.loiter.dwell_time(tid, now):.0f}s ek jagah",
                             frame, now)
        if is_crowd:
            self.alerts.fire("CROWD", -1, f"{people} log", frame, now)

        self._draw(frame, detections, behav, loiter_ids, abandon_ids,
                   fall_ids, fight_ids, level, now)

        dt = now - self._t_prev
        self._t_prev = now
        self.stats.update({
            "people": people, "crowded": is_crowd,
            "loitering": len(loiter_ids), "abandoned": len(abandon_ids),
            "fall": len(fall_ids), "fight": len(fight_ids),
            "threat": level, "threat_score": score,
            "fps": round(1.0 / dt if dt > 0 else 0.0, 1),
            "totals": self.alerts.total_counts,
        })
        return frame

    def _calm_step(self, detections, fall_ids, fight_ids, loiter_ids,
                   abandon_ids, people, is_crowd, frame, now):
        """Run the M1..M4 chain and stash the evidence record in self.stats."""
        # M1: per-frame pose reliability, averaged over the people driving a cue
        drivers = set(fall_ids) | set(fight_ids) | set(loiter_ids)
        persons = [d for d in detections if d.is_person]
        pool = [d for d in persons if d.track_id in drivers] or persons
        rs = []
        for d in pool:
            r, _ = self.rel.reliability(d.track_id, d.keypoints, d.bbox,
                                        frame.shape)
            rs.append(r)
        r_pose = sum(rs) / len(rs) if rs else 1.0

        strengths = strength_from_pipeline(
            loiter_ids, abandon_ids, fall_ids, fight_ids, people,
            self.cfg["crowd"]["threshold"],
            loiter_obj=self.loiter, abandon_obj=self.abandon, now=now)

        decision = self.calm.assess(strengths, pose_reliability=r_pose, now=now)
        self.stats["calm"] = decision.record

        if decision.alarm:
            try:
                with open(self._evidence_log, "a", encoding="utf-8") as f:
                    f.write(decision.to_json() + "\n")
            except OSError:
                pass

    def _draw_skeleton(self, frame, kp):
        if kp is None:
            return
        for (a, b) in SKELETON:
            if a < len(kp) and b < len(kp) and kp[a][2] > 0.3 and kp[b][2] > 0.3:
                pa = (int(kp[a][0]), int(kp[a][1]))
                pb = (int(kp[b][0]), int(kp[b][1]))
                cv2.line(frame, pa, pb, COL_SKELETON, 2)
        for i in range(len(kp)):
            if kp[i][2] > 0.3:
                cv2.circle(frame, (int(kp[i][0]), int(kp[i][1])), 3,
                           COL_SKELETON, -1)

    def _draw(self, frame, detections, behav, loiter_ids, abandon_ids,
              fall_ids, fight_ids, level, now):
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            if det.is_person:
                self._draw_skeleton(frame, det.keypoints)
                b = behav.get(det.track_id, {})
                agit = b.get("agitation", 0.0)

                if det.track_id in fall_ids:
                    color, tag = COL_FALL, f"FALL #{det.track_id}"
                elif det.track_id in fight_ids:
                    color, tag = COL_FIGHT, f"FIGHT #{det.track_id}"
                elif det.track_id in loiter_ids:
                    secs = self.loiter.dwell_time(det.track_id, now)
                    color, tag = COL_ALERT, f"LOITER #{det.track_id} {secs:.0f}s"
                else:
                    color = COL_NORMAL
                    post = b.get("posture", "?")
                    tag = f"#{det.track_id} {post} a:{agit:.1f}"
            else:
                if det.track_id in abandon_ids:
                    secs = self.abandon.unattended_time(det.track_id, now)
                    color, tag = COL_ALERT, f"ABANDONED {det.label} {secs:.0f}s"
                else:
                    color, tag = COL_OBJECT, f"{det.label} #{det.track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

        if level != "NONE":
            bar = LEVEL_COLOR.get(level, COL_ALERT)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), bar, -1)
            cv2.putText(frame, f"THREAT: {level}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
