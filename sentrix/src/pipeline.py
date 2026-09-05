"""
pipeline.py
-----------
Sab kuch yahan jurta hai. Ek frame andar jata hai, annotated frame +
stats bahar aate hain.

Flow:
  frame -> Detector -> [Loitering, Abandoned, Crowd] analyzers
        -> AlertManager -> draw boxes/labels -> output frame + stats
"""

import time
import cv2

from src.detector import Detector
from src.analyzers import LoiteringAnalyzer, AbandonedAnalyzer, CrowdAnalyzer
from src.alerts import AlertManager

# Command-center color scheme (BGR for OpenCV)
COL_NORMAL = (120, 200, 90)    # calm green
COL_WATCH = (60, 180, 235)     # amber
COL_ALERT = (60, 60, 235)      # red
COL_OBJECT = (200, 160, 90)    # blue-ish for bags
COL_TEXT = (240, 240, 240)


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.loiter = LoiteringAnalyzer(cfg)
        self.abandon = AbandonedAnalyzer(cfg)
        self.crowd = CrowdAnalyzer(cfg)
        self.alerts = AlertManager(cfg)

        # live stats (dashboard padhta hai)
        self.stats = {
            "people": 0,
            "crowded": False,
            "loitering": 0,
            "abandoned": 0,
            "fps": 0.0,
            "totals": self.alerts.total_counts,
        }
        self._t_prev = time.time()

    def process(self, frame):
        now = time.time()
        detections = self.detector.detect(frame)

        loiter_ids = self.loiter.update(detections, now)
        abandon_ids = self.abandon.update(detections, now)
        people, is_crowd = self.crowd.update(detections)

        # ---- alerts fire karo ----
        for tid in loiter_ids:
            secs = self.loiter.dwell_time(tid, now)
            self.alerts.fire("LOITERING", tid,
                             f"{secs:.0f}s ek jagah", frame, now)
        for tid in abandon_ids:
            secs = self.abandon.unattended_time(tid, now)
            self.alerts.fire("ABANDONED", tid,
                             f"{secs:.0f}s lawaaris", frame, now)
        if is_crowd:
            self.alerts.fire("CROWD", -1, f"{people} log", frame, now)

        # ---- draw ----
        self._draw(frame, detections, loiter_ids, abandon_ids, now)

        # ---- stats update ----
        dt = now - self._t_prev
        self._t_prev = now
        fps = 1.0 / dt if dt > 0 else 0.0
        self.stats.update({
            "people": people,
            "crowded": is_crowd,
            "loitering": len(loiter_ids),
            "abandoned": len(abandon_ids),
            "fps": round(fps, 1),
            "totals": self.alerts.total_counts,
        })
        return frame

    def _draw(self, frame, detections, loiter_ids, abandon_ids, now):
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            if det.is_person:
                if det.track_id in loiter_ids:
                    color = COL_ALERT
                    secs = self.loiter.dwell_time(det.track_id, now)
                    tag = f"LOITER #{det.track_id}  {secs:.0f}s"
                else:
                    color = COL_NORMAL
                    tag = f"person #{det.track_id}"
            else:
                if det.track_id in abandon_ids:
                    color = COL_ALERT
                    secs = self.abandon.unattended_time(det.track_id, now)
                    tag = f"ABANDONED {det.label} {secs:.0f}s"
                else:
                    color = COL_OBJECT
                    tag = f"{det.label} #{det.track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # label background
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1,
                        cv2.LINE_AA)

        # top banner agar koi active alert hai
        if loiter_ids or abandon_ids or self.stats.get("crowded"):
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), COL_ALERT, -1)
            cv2.putText(frame, "! ACTIVE ALERT", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2, cv2.LINE_AA)
