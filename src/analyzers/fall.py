"""
fall.py
-------
Fall (girna) detection — bina pose model ke, sirf bounding box ke shape se.
Yeh weak laptop pe chalta hai kyunki extra model nahi chahiye.

Idea:
  - Khada insaan: box LAMBA hota hai (height > width)
  - Gira hua insaan: box CHAURA hota hai (width > height)
  - Agar kisi person ka box achanak chaura ho jaye AUR thodi der waisa
    hi rahe -> fall confirm

"Thodi der rahe" check isliye taake jhukna/baithna ko fall na samjhe —
woh transient hota hai, gira hua banda wahi pada rehta hai.
"""

import time


class FallAnalyzer:
    def __init__(self, cfg):
        c = cfg["fall"]
        self.ar = c["aspect_ratio"]
        self.confirm = c["confirm_seconds"]
        self.tracks = {}   # id -> {fallen_since, last_seen}

    def update(self, detections, now=None):
        now = now or time.time()
        flagged = set()
        seen = set()

        for det in detections:
            if not det.is_person:
                continue
            seen.add(det.track_id)
            x1, y1, x2, y2 = det.bbox
            w, h = (x2 - x1), (y2 - y1)
            if h <= 0:
                continue
            aspect = w / h

            t = self.tracks.setdefault(det.track_id,
                                       {"fallen_since": None, "last_seen": now})
            t["last_seen"] = now

            if aspect >= self.ar:
                # box chaura hai -> possibly fallen
                if t["fallen_since"] is None:
                    t["fallen_since"] = now
                elif now - t["fallen_since"] >= self.confirm:
                    flagged.add(det.track_id)
            else:
                t["fallen_since"] = None   # khada hai, reset

        # gaye hue clean karo
        for tid in [k for k in self.tracks if k not in seen]:
            if now - self.tracks[tid]["last_seen"] > 3:
                del self.tracks[tid]

        return flagged
