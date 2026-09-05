"""
app.py
------
SENTRIX web dashboard ka entry point.

Chalao:   python app.py
Phir browser mein kholo:  http://127.0.0.1:5000

Yeh ek background thread mein video process karta hai, aur:
  /              -> dashboard page
  /video_feed    -> live annotated MJPEG stream
  /api/stats     -> current numbers (JSON)
  /api/alerts    -> recent alerts feed (JSON)
"""

import threading
import time

import cv2
import yaml
from flask import Flask, Response, render_template, jsonify

from src.video import VideoSource
from src.pipeline import Pipeline

# ---- config load ----
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

app = Flask(__name__,
            template_folder="dashboard/templates",
            static_folder="dashboard/static")

# ---- shared state between worker thread and web ----
_lock = threading.Lock()
_latest_frame = None
_pipeline = Pipeline(CFG)
_running = True


def _worker():
    """Background loop: frame padho, process karo, latest store karo."""
    global _latest_frame
    src = VideoSource(CFG)
    while _running:
        frame = src.read()
        if frame is None:
            time.sleep(0.05)
            continue
        out = _pipeline.process(frame)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _latest_frame = buf.tobytes()
    src.release()


def _mjpeg():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with _lock:
            frame = _latest_frame
        if frame is not None:
            yield boundary + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html", cfg=CFG)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    return jsonify(_pipeline.stats)


@app.route("/api/alerts")
def api_alerts():
    return jsonify(list(_pipeline.alerts.recent))


if __name__ == "__main__":
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    print("\n  SENTRIX chal raha hai ->  http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
