"""
evaluate.py
-----------
Paper ke liye numbers generate karta hai. Ek test video pe poori pipeline
chala ke yeh measure karta hai:

  - Average FPS (real-time chal raha hai ya nahi)
  - Total frames, total detections
  - Per-class detection counts
  - Total alerts fire hue (loitering / abandoned / crowd)
  - Latency per frame (ms)

Chalao:
   python evaluate.py --video data/videos/test.mp4
   python evaluate.py --video data/videos/test.mp4 --frames 500

Output: results/eval_report.txt  (paper mein paste karne layak table)
"""

import argparse
import time
import os

import yaml
import numpy as np

from src.video import VideoSource
from src.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="test video path (warna config source)")
    ap.add_argument("--frames", type=int, default=0, help="kitne frames (0=poora)")
    args = ap.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.video:
        cfg["source"] = args.video

    src = VideoSource(cfg)
    pipe = Pipeline(cfg)

    latencies = []
    det_counts = {}
    n = 0
    t_start = time.time()

    while True:
        frame = src.read()
        if frame is None:
            break

        t0 = time.perf_counter()
        dets = pipe.detector.detect(frame)
        pipe.loiter.update(dets)
        pipe.abandon.update(dets)
        pipe.crowd.update(dets)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms
        for d in dets:
            det_counts[d.label] = det_counts.get(d.label, 0) + 1

        n += 1
        if args.frames and n >= args.frames:
            break
        if n % 50 == 0:
            print(f"  processed {n} frames...")

    src.release()
    elapsed = time.time() - t_start

    lat = np.array(latencies) if latencies else np.array([0])
    avg_fps = n / elapsed if elapsed > 0 else 0

    report = []
    report.append("=" * 52)
    report.append("  SENTRIX — Evaluation Report")
    report.append("=" * 52)
    report.append(f"  Model            : {cfg['model']['weights']}")
    report.append(f"  Frame width      : {cfg['frame_width']} px")
    report.append(f"  Frames processed : {n}")
    report.append(f"  Wall time        : {elapsed:.1f} s")
    report.append("-" * 52)
    report.append(f"  Average FPS      : {avg_fps:.2f}")
    report.append(f"  Latency mean     : {lat.mean():.1f} ms")
    report.append(f"  Latency p95      : {np.percentile(lat,95):.1f} ms")
    report.append(f"  Latency max      : {lat.max():.1f} ms")
    report.append("-" * 52)
    report.append("  Detections by class:")
    for k, v in sorted(det_counts.items(), key=lambda x: -x[1]):
        report.append(f"    {k:<12} : {v}")
    report.append("-" * 52)
    report.append("  Alerts fired:")
    for k, v in pipe.alerts.total_counts.items():
        report.append(f"    {k:<12} : {v}")
    report.append("=" * 52)

    text = "\n".join(report)
    print("\n" + text)

    os.makedirs("results", exist_ok=True)
    with open("results/eval_report.txt", "w") as f:
        f.write(text)
    print("\n  -> results/eval_report.txt mein save ho gaya")


if __name__ == "__main__":
    main()
