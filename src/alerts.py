"""
alerts.py
---------
Alerts ko manage karta hai:
  - Cooldown: same alert baar baar fire na ho (har frame pe spam na ho)
  - CSV log: har alert timestamp ke saath save (paper ke results ke liye)
  - Snapshot: alert ke waqt ka frame image save (paper mein dikhane ke liye)
  - Recent alerts list: dashboard pe live feed dikhane ke liye
"""

import os
import csv
import time
from collections import deque
from datetime import datetime

import cv2


class AlertManager:
    def __init__(self, cfg):
        c = cfg["alerts"]
        self.cooldown = c["cooldown_seconds"]
        self.save_snaps = c["save_snapshots"]
        self.log_file = c["log_file"]
        self.snap_dir = c["snapshot_dir"]

        self._last_fired = {}            # (type, id) -> time
        self.recent = deque(maxlen=50)   # dashboard feed
        self.total_counts = {"LOITERING": 0, "ABANDONED": 0, "CROWD": 0,
                             "FALL": 0, "FIGHT": 0}

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(self.snap_dir, exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "type", "track_id", "detail", "snapshot"]
                )

    def fire(self, alert_type, track_id, detail="", frame=None, now=None):
        """Ek alert raise karo (cooldown respect karte hue)."""
        now = now or time.time()
        key = (alert_type, track_id)

        if key in self._last_fired and now - self._last_fired[key] < self.cooldown:
            return None   # abhi cooldown mein hai

        self._last_fired[key] = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        snap_path = ""
        if self.save_snaps and frame is not None:
            fname = f"{alert_type}_{track_id}_{int(now)}.jpg"
            snap_path = os.path.join(self.snap_dir, fname)
            cv2.imwrite(snap_path, frame)

        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([ts, alert_type, track_id, detail, snap_path])

        self.total_counts[alert_type] = self.total_counts.get(alert_type, 0) + 1
        event = {
            "timestamp": ts,
            "type": alert_type,
            "track_id": track_id,
            "detail": detail,
        }
        self.recent.appendleft(event)
        return event
