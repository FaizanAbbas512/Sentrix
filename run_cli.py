"""
run_cli.py
----------
Web dashboard ke baghair quick test ke liye. Seedha ek OpenCV window
kholta hai. Browser nahi chahiye.

Chalao:   python run_cli.py
Band karo: window pe 'q' dabao
"""

import cv2
import yaml

from src.video import VideoSource
from src.pipeline import Pipeline


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    src = VideoSource(cfg)
    pipe = Pipeline(cfg)

    print("SENTRIX (CLI mode) — 'q' dabao band karne ke liye")
    while True:
        frame = src.read()
        if frame is None:
            break
        out = pipe.process(frame)

        s = pipe.stats
        hud = f"People:{s['people']}  Loiter:{s['loitering']}  Abandon:{s['abandoned']}  {s['fps']}fps"
        cv2.putText(out, hud, (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.imshow("SENTRIX", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
