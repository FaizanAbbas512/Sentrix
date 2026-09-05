"""
loitering.py
------------
Loitering = koi banda ek hi jagah bahut der tak khada/ghoom raha hai.

Logic:
  - Har person ID ka "pehli baar dekha" time yaad rakho
  - Uski position history rakho
  - Agar banda dwell_seconds se zyada der se hai AUR zyada hila nahi
    (movement_tolerance ke andar) -> LOITERING

Sirf "der se hai" kaafi nahi — warna har baitha hua banda flag hoga.
Isliye stationary check zaroori hai: jo guzar raha hai woh loiter nahi.
"""

import time
import math


class LoiteringAnalyzer:
    def __init__(self, cfg):
        c = cfg["loitering"]
        self.dwell = c["dwell_seconds"]
        self.tol = c["movement_tolerance"]
        self.tracks = {}   # track_id -> {first_seen, anchor, last_seen}

    def update(self, detections, now=None):
        """Loitering kar rahe person IDs ka set return karta hai."""
        now = now or time.time()
        flagged = set()
        seen_ids = set()

        for det in detections:
            if not det.is_person:
                continue
            seen_ids.add(det.track_id)
            cx, cy = det.center

            if det.track_id not in self.tracks:
                # naya banda
                self.tracks[det.track_id] = {
                    "first_seen": now,
                    "anchor": (cx, cy),   # jahan se "khada" hona shuru hua
                    "last_seen": now,
                }
            else:
                t = self.tracks[det.track_id]
                t["last_seen"] = now
                ax, ay = t["anchor"]
                dist = math.hypot(cx - ax, cy - ay)

                if dist > self.tol:
                    # banda hil gaya -> anchor reset, timer dobara shuru
                    t["anchor"] = (cx, cy)
                    t["first_seen"] = now
                else:
                    # ek hi jagah hai — kitni der ho gayi?
                    if now - t["first_seen"] >= self.dwell:
                        flagged.add(det.track_id)

        # jo log frame se chale gaye unko bhula do
        gone = [tid for tid in self.tracks if tid not in seen_ids]
        for tid in gone:
            # thodi grace di jaa sakti hai, lekin simple rakhte hain
            if now - self.tracks[tid]["last_seen"] > 3:
                del self.tracks[tid]

        return flagged

    def dwell_time(self, track_id, now=None):
        """Kisi ID ka current dwell time (seconds) — UI labels ke liye."""
        now = now or time.time()
        if track_id in self.tracks:
            return now - self.tracks[track_id]["first_seen"]
        return 0.0
