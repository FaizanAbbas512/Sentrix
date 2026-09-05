"""
detector.py
-----------
YOLO26 ke upar ek saaf wrapper. Har frame pe detection + tracking
chalata hai aur structured results return karta hai.

Tracking isliye zaroori hai: bina persistent ID ke hum nahi keh sakte
ke "yeh wahi banda hai jo 15 second se khada hai". ByteTrack har object
ko ek sticky ID deta hai jo frames ke beech chipka rehta hai.
"""

from dataclasses import dataclass, field
from ultralytics import YOLO


@dataclass
class Detection:
    """Ek detected object — person ya bag."""
    track_id: int           # persistent ID (frames ke beech same rehta hai)
    label: str              # "person", "backpack", etc.
    class_id: int
    confidence: float
    bbox: tuple             # (x1, y1, x2, y2)
    keypoints: object = None  # pose model se: numpy array (17, 3) = x,y,conf

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def is_person(self):
        return self.label == "person"


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg["model"]
        self.model = YOLO(m["weights"])
        self.conf = m["conf"]
        self.iou = m["iou"]
        self.tracker = m["tracker"]
        self.imgsz = m.get("imgsz", 640)

        # Class ID -> readable name ka reverse map banao
        self.id_to_name = {cfg["classes"]["person"]: "person"}
        for name, cid in cfg["classes"]["objects"].items():
            self.id_to_name[cid] = name
        self.wanted_ids = list(self.id_to_name.keys())

    def detect(self, frame):
        """Ek frame lo, Detection objects ki list do."""
        results = self.model.track(
            frame,
            imgsz=self.imgsz,        # chhoti inference resolution = tez
            persist=True,            # IDs ko frames ke beech yaad rakho
            classes=self.wanted_ids, # sirf person + bags
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.id is None:        # tracker ne abhi IDs assign nahi kiye
            return detections

        # pose model ho to keypoints bhi nikaalo
        kpts = None
        if getattr(results[0], "keypoints", None) is not None:
            try:
                xy = results[0].keypoints.xy.cpu().numpy()      # (N,17,2)
                cf = results[0].keypoints.conf
                cf = cf.cpu().numpy() if cf is not None else None  # (N,17)
                kpts = (xy, cf)
            except Exception:
                kpts = None

        for idx, (box, tid, cls, conf) in enumerate(zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        )):
            x1, y1, x2, y2 = map(int, box)

            person_kp = None
            if kpts is not None and idx < len(kpts[0]):
                xy, cf = kpts
                import numpy as np
                k = xy[idx]                                   # (17,2)
                c = cf[idx] if cf is not None else np.ones(len(k))
                person_kp = np.concatenate([k, c[:, None]], axis=1)  # (17,3)

            detections.append(Detection(
                track_id=int(tid),
                label=self.id_to_name.get(int(cls), str(cls)),
                class_id=int(cls),
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                keypoints=person_kp,
            ))
        return detections
