"""
behaviour.py
------------
YEH hai woh cheez jo tu chahta tha — system box nahi, ASAL BODY dekhta hai.

Pose model har person ka skeleton deta hai (17 points: kandhe, kohni,
kalai, kulhe, ghutne, takhne). In points se hum samajhte hain:

  1. POSTURE  — banda khada hai ya GIRA hua (torso ka angle dekh ke)
  2. AGITATION — haath/paaon kitni tezi se chal rahe (asal limb speed)

Aur yeh imaandaar hai: yeh ACTION padhta hai (kya kar raha hai), niyat/dar
NAHI (woh kisi se nahi hota). Box-based fall se yeh bohot behtar hai kyunki
yeh body ka actual shape dekhta hai, sirf rectangle nahi.

COCO keypoints index:
  5,6 = shoulders   7,8 = elbows   9,10 = wrists
  11,12 = hips      13,14 = knees  15,16 = ankles
"""

import time
import math
from collections import deque

KP_CONF = 0.3   # is se kam confidence wale points ignore


def _valid(kp, i):
    return kp is not None and i < len(kp) and kp[i][2] >= KP_CONF


def _mid(kp, a, b):
    """Do points ka beech ka point (agar dono valid)."""
    if _valid(kp, a) and _valid(kp, b):
        return ((kp[a][0] + kp[b][0]) / 2, (kp[a][1] + kp[b][1]) / 2)
    if _valid(kp, a):
        return (kp[a][0], kp[a][1])
    if _valid(kp, b):
        return (kp[b][0], kp[b][1])
    return None


class BehaviourAnalyzer:
    """
    Har frame: per-person posture + agitation nikaalta hai.
    History rakhta hai taake limb speed (agitation) nap sake.
    """

    def __init__(self, cfg):
        b = cfg.get("behaviour", {})
        self.fall_confirm = b.get("fall_confirm_seconds", 1.2)
        self.agitation_th = b.get("agitation_threshold", 0.5)  # body-size normalized
        self.history_len = b.get("history_frames", 5)

        self.kp_hist = {}      # id -> deque of (wrist/ankle points, torso_len, t)
        self.fallen_since = {} # id -> time
        self.last_seen = {}

    def update(self, detections, now=None):
        """
        return: dict id -> {"posture": "upright"/"fallen"/"unknown",
                            "agitation": float, "fallen": bool}
        """
        now = now or time.time()
        out = {}
        seen = set()

        for det in detections:
            if not det.is_person:
                continue
            kp = det.keypoints
            seen.add(det.track_id)
            self.last_seen[det.track_id] = now

            posture = self._posture(kp, det.bbox)
            agit = self._agitation(det.track_id, kp, det.bbox, now)

            # fall = "fallen" posture sustained
            fallen = False
            if posture == "fallen":
                if det.track_id not in self.fallen_since:
                    self.fallen_since[det.track_id] = now
                elif now - self.fallen_since[det.track_id] >= self.fall_confirm:
                    fallen = True
            else:
                self.fallen_since.pop(det.track_id, None)

            out[det.track_id] = {"posture": posture, "agitation": agit,
                                 "fallen": fallen}

        # cleanup
        for tid in [t for t in self.last_seen if t not in seen]:
            if now - self.last_seen[tid] > 3:
                self.last_seen.pop(tid, None)
                self.kp_hist.pop(tid, None)
                self.fallen_since.pop(tid, None)

        return out

    def _torso_len(self, kp, bbox):
        sh = _mid(kp, 5, 6)
        hp = _mid(kp, 11, 12)
        if sh and hp:
            d = math.hypot(sh[0] - hp[0], sh[1] - hp[1])
            if d > 1:
                return d
        # fallback: bbox height ka aadha
        x1, y1, x2, y2 = bbox
        return max((y2 - y1) / 2, 1)

    def _posture(self, kp, bbox):
        """Torso vertical = upright, horizontal = fallen."""
        sh = _mid(kp, 5, 6)
        hp = _mid(kp, 11, 12)
        if sh and hp:
            dx = abs(sh[0] - hp[0])
            dy = abs(sh[1] - hp[1])
            if dx > dy * 1.2:      # zyada horizontal = gira hua
                return "fallen"
            return "upright"
        # keypoints na ho to box shape se andaza (fallback)
        x1, y1, x2, y2 = bbox
        w, h = (x2 - x1), (y2 - y1)
        if h > 0 and w / h >= 1.15:
            return "fallen"
        return "unknown" if not sh else "upright"

    def _agitation(self, tid, kp, bbox, now):
        """Wrists+ankles ki speed, body-size se normalize."""
        torso = self._torso_len(kp, bbox)
        pts = []
        for i in (9, 10, 15, 16):   # wrists, ankles
            if _valid(kp, i):
                pts.append((kp[i][0], kp[i][1]))
        if not pts:
            cx, cy = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            pts = [(cx, cy)]

        hist = self.kp_hist.setdefault(tid, deque(maxlen=self.history_len))
        hist.append((pts, torso, now))

        if len(hist) < 2:
            return 0.0

        # consecutive frames ke beech average normalized displacement
        speeds = []
        for k in range(1, len(hist)):
            prev_pts, _, _ = hist[k - 1]
            cur_pts, t_len, _ = hist[k]
            n = min(len(prev_pts), len(cur_pts))
            if n == 0:
                continue
            d = sum(math.hypot(cur_pts[j][0] - prev_pts[j][0],
                               cur_pts[j][1] - prev_pts[j][1])
                    for j in range(n)) / n
            speeds.append(d / t_len)   # body-size normalized
        return round(sum(speeds) / len(speeds), 3) if speeds else 0.0
