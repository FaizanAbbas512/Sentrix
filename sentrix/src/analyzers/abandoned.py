"""
abandoned.py
------------
Abandoned object = koi bag/suitcase/backpack jo:
  1. ek jagah static pada hai (hila nahi), AUR
  2. aas paas koi insaan nahi (owner_radius ke andar), AUR
  3. yeh haalat unattended_seconds se zyada der rahi

Yeh classic airport/station security scenario hai (bomb threat).

Logic:
  - Har object ID ki position track karo
  - Har frame check karo: koi person nazdeek hai?
  - Agar nahi -> "unattended" timer chalao
  - Timer threshold paar -> ABANDONED alert
  - Koi person paas aa jaye -> timer reset (attended)
"""

import time
import math


class AbandonedAnalyzer:
    def __init__(self, cfg):
        c = cfg["abandoned"]
        self.unattended = c["unattended_seconds"]
        self.radius = c["owner_radius"]
        self.tol = c["movement_tolerance"]
        self.tracks = {}   # obj_id -> {anchor, unattended_since, last_seen}

    def update(self, detections, now=None):
        """Abandoned object IDs ka set return karta hai."""
        now = now or time.time()
        flagged = set()
        seen_ids = set()

        persons = [d for d in detections if d.is_person]
        objects = [d for d in detections if not d.is_person]

        for obj in objects:
            seen_ids.add(obj.track_id)
            ox, oy = obj.center

            # koi person owner_radius ke andar hai?
            attended = any(
                math.hypot(ox - px, oy - py) <= self.radius
                for (px, py) in (p.center for p in persons)
            )

            if obj.track_id not in self.tracks:
                self.tracks[obj.track_id] = {
                    "anchor": (ox, oy),
                    "unattended_since": None,
                    "last_seen": now,
                }

            t = self.tracks[obj.track_id]
            t["last_seen"] = now

            # object khud hil to nahi gaya? (uthaya gaya / move hua)
            ax, ay = t["anchor"]
            if math.hypot(ox - ax, oy - ay) > self.tol:
                t["anchor"] = (ox, oy)
                t["unattended_since"] = None   # moving object abandoned nahi
                continue

            if attended:
                # owner paas hai -> reset
                t["unattended_since"] = None
            else:
                # akela hai -> timer
                if t["unattended_since"] is None:
                    t["unattended_since"] = now
                elif now - t["unattended_since"] >= self.unattended:
                    flagged.add(obj.track_id)

        # gaye hue objects clean karo
        gone = [tid for tid in self.tracks if tid not in seen_ids]
        for tid in gone:
            if now - self.tracks[tid]["last_seen"] > 3:
                del self.tracks[tid]

        return flagged

    def unattended_time(self, track_id, now=None):
        now = now or time.time()
        t = self.tracks.get(track_id)
        if t and t["unattended_since"]:
            return now - t["unattended_since"]
        return 0.0
