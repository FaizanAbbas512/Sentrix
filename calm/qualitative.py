"""
qualitative.py  --  reproduce the paper's Fig. 3 evidence records + snapshots
==============================================================================

Everything in paper/figures/qual/ (the two PNGs used in Fig. 3) and the two
evidence records quoted in figures/qual_examples.tex come from this script,
run once against the real Avenue pose JSON (data/pose/avenue.json) and the
real Avenue test videos (data/raw/avenue/.../testing_videos/). Nothing here
is synthetic or hand-picked without a documented search: `search` below
scans every frame of every one of the 21 real Avenue test clips for (a) the
strongest true alarm inside a labelled anomaly window and (b) the largest
M1-discounting effect / highest evidence-fusion conflict, then prints the
winners with their full record. `snapshot` decodes the exact winning frame
from the exact matching video and saves it as a PNG.

    python -m calm.qualitative search
    python -m calm.qualitative snapshot --clip 01 --frame 965 --out paper/figures/qual/clip01_f965.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from .cues import CueBank, CUE_NAMES
from .reliability import ReliabilityEstimator
from .evidence import EvidenceFusion
from .datasets import load_generic_json


def run_clip(clip, cfg, use_reliability=True):
    """Same per-frame loop as harness.score_clip, but also returns the full
    FusionResult per frame (contributions) so a specific frame's evidence
    record can be printed in full."""
    cb = CueBank(cfg, fps=clip.fps)
    rel = ReliabilityEstimator(cfg)
    fus = EvidenceFusion(cfg)
    n = clip.n_frames
    belief = np.zeros(n); r_series = np.ones(n); conflict = np.zeros(n)
    strength_mat = np.zeros((n, len(CUE_NAMES)))
    records = [None] * n
    for t, dets in enumerate(clip.frames):
        strengths, aux = cb.step(dets)
        if use_reliability:
            rs = []
            drivers = (aux["fallen_ids"] | aux["fight_ids"] | aux["loiter_ids"] | aux["abandoned_ids"])
            persons = [p for p in dets if p.get("is_person", True)]
            pool = [p for p in persons if p["track_id"] in drivers] or persons
            for p in pool:
                r, _ = rel.reliability(p["track_id"], p.get("keypoints"), p["bbox"], (480, 640))
                rs.append(r)
            r_pose = float(np.mean(rs)) if rs else 1.0
        else:
            r_pose = 1.0
        fr = fus.fuse(strengths, r_pose=r_pose)
        belief[t] = fr.bel_A
        r_series[t] = r_pose
        conflict[t] = fr.conflict
        for i, name in enumerate(CUE_NAMES):
            strength_mat[t, i] = strengths[name]
        records[t] = fr
    return belief, r_series, conflict, strength_mat, records


def in_gt(gt, t):
    return any(a <= t <= b for a, b in gt)


def search(pose_json="data/pose/avenue.json", config_path="config.yaml"):
    """Exhaustive real-data search used for Fig. 3: for every test clip,
    find the strongest true alarm and the largest M1-discounting gap /
    highest conflict, then print the two global winners with full detail."""
    cfg = yaml.safe_load(open(config_path))
    clips = {c.name: c for c in load_generic_json(pose_json)}

    best_ta = (-1.0, None, None)      # (belief, clip, frame) inside GT
    best_gap = (-1.0, None, None)      # (belief_noM1 - belief, clip, frame)
    best_conflict = (-1.0, None, None)

    for name, c in clips.items():
        belief, r_series, conflict, strength_mat, _ = run_clip(c, cfg, use_reliability=True)
        belief_nom1, *_ = run_clip(c, cfg, use_reliability=False)
        gt = c.gt_intervals

        in_gt_mask = np.array([in_gt(gt, t) for t in range(c.n_frames)])
        if in_gt_mask.any():
            t_ta = int(np.argmax(np.where(in_gt_mask, belief, -1)))
            if belief[t_ta] > best_ta[0]:
                best_ta = (float(belief[t_ta]), name, t_ta)

        gap = belief_nom1 - belief
        t_gap = int(np.argmax(gap))
        if gap[t_gap] > best_gap[0]:
            best_gap = (float(gap[t_gap]), name, t_gap)

        t_conf = int(np.argmax(conflict))
        if conflict[t_conf] > best_conflict[0]:
            best_conflict = (float(conflict[t_conf]), name, t_conf)

    for label, (val, name, t) in [("true alarm", best_ta), ("M1 gap", best_gap), ("conflict", best_conflict)]:
        c = clips[name]
        belief, r_series, conflict, strength_mat, records = run_clip(c, cfg, use_reliability=True)
        belief_nom1, *_ = run_clip(c, cfg, use_reliability=False)
        fr = records[t]
        print(f"\n=== {label} winner: clip {name} frame {t} (value={val:.4f}) ===")
        print("strengths     :", dict(zip(CUE_NAMES, strength_mat[t].tolist())))
        print("r_pose        :", r_series[t])
        print("belief (M1 on):", belief[t], " belief (M1 off):", belief_nom1[t])
        print("plausibility  :", fr.pl_A, " conflict K:", fr.conflict)
        print("in_gt         :", in_gt(c.gt_intervals, t))
        print("contributions :", fr.contributions)


def snapshot(video_path, frame_idx, out_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video_path} (total={total})")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"saved {out_path}  ({frame.shape}, video has {int(total)} frames)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="exhaustive real-data search over all Avenue test clips")
    p_search.add_argument("--pose-json", default="data/pose/avenue.json")
    p_search.add_argument("--config", default="config.yaml")

    p_snap = sub.add_parser("snapshot", help="save one real video frame as a PNG")
    p_snap.add_argument("--clip", required=True, help='clip name, e.g. "01"')
    p_snap.add_argument("--frame", type=int, required=True)
    p_snap.add_argument("--videos-root", default="data/raw/avenue/Avenue Dataset/testing_videos")
    p_snap.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "search":
        search(args.pose_json, args.config)
    elif args.cmd == "snapshot":
        video_path = os.path.join(args.videos_root, f"{args.clip}.avi")
        snapshot(video_path, args.frame, args.out)


if __name__ == "__main__":
    main()
