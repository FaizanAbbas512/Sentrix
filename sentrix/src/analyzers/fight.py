"""
fight.py
--------
Fight / aggression — ab POSE ke saath (behtar).

Intelligent rule (jaisa tune kaha):
  - Akela banda haath hilaye = NAHI (paas koi nahi)
  - Do log paas khade baat karein, hilein na = NAHI
  - Do (ya zyada) log PAAS + DONO ke limbs sach mein TEZ + DER tak = FIGHT

Pehle motion box ke center se napa jata tha (crude). Ab agar pose available
ho to ASLI limb agitation (haath-paaon ki speed) use hoti hai. Pose na ho
to centroid motion fallback.

Note: yeh phir bhi 100% nahi — do log gale milein ya naachein toh false
positive aa sakta hai. Paper mein imaandaari se likhna.
"""

import time
import math


class FightAnalyzer:
    def __init__(self, cfg):
        c = cfg["fight"]
        self.proximity = c["proximity"]
        self.motion_th = c["motion_threshold"]
        self.agit_th = c.get("agitation_threshold", 0.6)
        self.confirm = c["confirm_seconds"]
        self.prev_center = {}
        self.engaged_since = {}

    def update(self, detections, behaviour=None, now=None):
        now = now or time.time()
        persons = [d for d in detections if d.is_person]

        cur_center = {}
        motion = {}
        for p in persons:
            cx, cy = p.center
            cur_center[p.track_id] = (cx, cy)
            if p.track_id in self.prev_center:
                px, py = self.prev_center[p.track_id]
                motion[p.track_id] = math.hypot(cx - px, cy - py)
            else:
                motion[p.track_id] = 0.0

        def is_agitated(tid):
            if behaviour and tid in behaviour:
                return behaviour[tid]["agitation"] >= self.agit_th
            return motion.get(tid, 0.0) >= self.motion_th

        flagged = set()
        active_pairs = set()

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a, b = persons[i], persons[j]
                ax, ay = cur_center[a.track_id]
                bx, by = cur_center[b.track_id]
                dist = math.hypot(ax - bx, ay - by)

                close = dist <= self.proximity
                both_fast = is_agitated(a.track_id) and is_agitated(b.track_id)

                pair = frozenset((a.track_id, b.track_id))
                if close and both_fast:
                    active_pairs.add(pair)
                    if pair not in self.engaged_since:
                        self.engaged_since[pair] = now
                    elif now - self.engaged_since[pair] >= self.confirm:
                        flagged.add(a.track_id)
                        flagged.add(b.track_id)

        for pair in [p for p in self.engaged_since if p not in active_pairs]:
            del self.engaged_since[pair]

        self.prev_center = cur_center
        return flagged
