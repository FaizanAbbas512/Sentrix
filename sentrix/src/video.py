"""
video.py
--------
Video source ko handle karta hai — chahe webcam ho ya video file.
Frame ko config width pe resize karta hai (chhota frame = tez inference).
File khatam ho jaye to loop kar deta hai (demo ke liye useful).
"""

import cv2


class VideoSource:
    def __init__(self, cfg):
        self.source = cfg["source"]
        self.target_w = cfg["frame_width"]
        self.is_file = isinstance(self.source, str)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video source khul nahi paya: {self.source}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.is_file:
                # file khatam -> dobara shuru (demo loop)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return None
        return self._resize(frame)

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w == self.target_w:
            return frame
        scale = self.target_w / w
        return cv2.resize(frame, (self.target_w, int(h * scale)))

    def release(self):
        self.cap.release()
