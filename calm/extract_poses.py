"""
extract_poses.py  —  videos  ->  the generic-JSON pose schema
============================================================

Turnkey step for Colab / Kaggle: point it at a folder of videos, get back the
`{fps, clips:[{name, n_frames, split, gt, frames:[[det,...],...]}]}` file that
`calm.datasets.load_generic_json` (and therefore `calm.harness`) consumes.

    python -m calm.extract_poses \
        --videos  data/benchmarks/shanghaitech/testing/videos \
        --gt      data/benchmarks/shanghaitech/testing/frame_masks \
        --out     data/pose/shanghaitech.json \
        --split   test \
        --weights yolo11n-pose.pt --imgsz 640 --device 0

Ground truth (`--gt`) is optional and accepts, per clip (matched by filename stem):
    *.npy   frame mask, 1 = anomalous            (ShanghaiTech / NWPU style)
    *.txt   one "start end" interval per line    (frame indices)
    *.json  {"gt": [[start,end], ...]}

Run once per split (train/calib/test). To build a cross-dataset file, run for
each dataset and merge the "clips" lists, setting "split" per clip.

Needs `ultralytics` + `opencv-python` (already in requirements.txt). Everything
downstream of this file is CPU-only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _load_gt(gt_dir, stem, n_frames):
    if not gt_dir:
        return []
    for ext in (".npy", ".txt", ".json"):
        p = os.path.join(gt_dir, stem + ext)
        if not os.path.exists(p):
            continue
        if ext == ".npy":
            mask = np.load(p).astype(int).ravel()
            return _mask_to_intervals(mask)
        if ext == ".txt":
            out = []
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                a, b = line.replace(",", " ").split()[:2]
                out.append([int(float(a)), int(float(b))])
            return out
        if ext == ".json":
            d = json.load(open(p))
            return [list(map(int, g)) for g in d.get("gt", [])]
    return []


def _mask_to_intervals(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append([i, j - 1])
            i = j
        else:
            i += 1
    return out


def extract_one(model, path, classes, imgsz, conf, iou, tracker, stride=1, device=None):
    """Return (n_frames, frames) for one video. frames = list[list[det dict]]."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if stride > 1 and (fi % stride):
            frames.append([])           # keep index alignment
            continue
        res = model.track(frame, imgsz=imgsz, conf=conf, iou=iou,
                          persist=True, tracker=tracker, classes=classes,
                          device=device, verbose=False)
        dets = []
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            kp_xy = kp_cf = None
            if getattr(res[0], "keypoints", None) is not None:
                kp_xy = res[0].keypoints.xy.cpu().numpy()          # (N,17,2)
                c = res[0].keypoints.conf
                kp_cf = c.cpu().numpy() if c is not None else None  # (N,17)
            for k in range(len(ids)):
                is_person = int(cls[k]) == 0
                kp = None
                if is_person and kp_xy is not None and k < len(kp_xy):
                    cf = kp_cf[k] if kp_cf is not None else np.ones(17)
                    kp = np.concatenate([kp_xy[k], cf[:, None]], axis=1).round(2).tolist()
                dets.append({
                    "id": int(ids[k]),
                    "keypoints": kp,
                    "bbox": [round(float(v), 1) for v in xyxy[k]],
                    "is_person": bool(is_person),
                    "label": "person" if is_person else str(int(cls[k])),
                })
        frames.append(dets)
    cap.release()
    return len(frames), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of video files")
    ap.add_argument("--gt", default=None, help="folder of per-clip ground truth (optional)")
    ap.add_argument("--out", required=True, help="output .json path")
    ap.add_argument("--split", default="test", choices=["train", "calib", "test"])
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU")
    ap.add_argument("--fps", type=float, default=None, help="override; else read from video")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--ext", default="mp4,avi,mov,mkv")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    # NOTE: pass device to model.track() (it accepts '0', 'cpu', 'cuda:0');
    # model.to('0') would fail -- torch wants 'cuda:0', not a bare index.
    # person + common bag classes (COCO) so the abandoned-object cue still works
    classes = [0, 24, 26, 28]

    exts = tuple("." + e.strip().lower() for e in args.ext.split(","))
    vids = sorted(p for p in glob.glob(os.path.join(args.videos, "*"))
                  if p.lower().endswith(exts))
    if not vids:
        raise SystemExit(f"no videos with {exts} in {args.videos}")

    import cv2
    clips = []
    fps_seen = []
    for vp in vids:
        stem = os.path.splitext(os.path.basename(vp))[0]
        cap = cv2.VideoCapture(vp)
        vfps = args.fps or (cap.get(cv2.CAP_PROP_FPS) or 20.0)
        cap.release()
        fps_seen.append(vfps)
        print(f"  [{len(clips)+1}/{len(vids)}] {stem}  (fps~{vfps:.1f})")
        n, frames = extract_one(model, vp, classes, args.imgsz, args.conf,
                                args.iou, args.tracker, args.stride, args.device)
        clips.append({
            "name": stem, "n_frames": n, "split": args.split,
            "gt": _load_gt(args.gt, stem, n), "frames": frames,
        })

    out = {"fps": args.fps or float(np.median(fps_seen)), "clips": clips}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    sz = os.path.getsize(args.out) / 1e6
    n_gt = sum(len(c["gt"]) for c in clips)
    print(f"\n  -> {args.out}  ({sz:.1f} MB, {len(clips)} clips, {n_gt} gt events)")
    print(f"     next:  python -m calm.harness --generic {args.out} --tag {args.split}")


if __name__ == "__main__":
    main()
