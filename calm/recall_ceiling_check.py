"""
recall_ceiling_check.py -- diagnostic, not a paper result.

Question: paper/tables report FAPH@recall=0.9 as "--" (never reached) on
every benchmark. calm/harness.py's eval_stream() sweeps tau over
np.linspace(0.98, 0.02, 60) -- i.e. it never tries tau below 0.02. Is the
"0.9 recall never reached" finding a real, data-driven ceiling, or an
artifact of that 0.02 floor (i.e. would recall keep climbing if we swept
lower)?

Method: reuse calm.harness's own score_clip() + the identical calib/test
split + the identical isotonic Calibrator fit (so the calibrated
probabilities are byte-for-byte what the paper's numbers are built from),
then sweep tau on a much wider grid -- down to 0.0, not just 0.02 -- and
report achieved global recall at each point. If recall saturates well
below 0.9 even at tau=0, the ceiling is real (some GT events never get any
meaningful belief at all, at any frame). If recall keeps climbing past what
the 0.02-floor grid showed, the "--" cells are partly a grid artifact.

    python -m calm.recall_ceiling_check --generic data/pose/shanghaitech.json --tag shanghaitech
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import yaml

from .datasets import load_generic_json
from .calibration import Calibrator
from . import metrics as M
from .harness import score_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    clips = load_generic_json(args.generic)
    empty = [c.name for c in clips if c.n_frames == 0]
    if empty:
        clips = [c for c in clips if c.n_frames > 0]

    # ---- identical calib/test split logic to calm/harness.py:141-152 ----
    calib = [c for c in clips if c.split == "calib"]
    test = [c for c in clips if c.split == "test"]
    if not calib:
        test = [c for c in clips if c.split != "calib"] or clips
    if not calib or not any(c.gt_intervals for c in calib) or not any(not c.gt_intervals for c in calib):
        norm = [c for c in test if not c.gt_intervals]
        anom = [c for c in test if c.gt_intervals]
        carve = norm[: max(1, len(norm) // 3)] + anom[: max(1, len(anom) // 3)]
        calib = list({c.name: c for c in (calib + carve)}.values())
        calib_names = {c.name for c in calib}
        test = [c for c in test if c.name not in calib_names] or test

    print(f"[{args.tag}] calib={len(calib)} test={len(test)}")

    # ---- score every clip (identical to harness.py) ----
    scored = {c.name: score_clip(c, cfg) for c in (calib + test)}

    bel_cal = np.concatenate([scored[c.name]["belief"] for c in calib])
    y_cal = np.concatenate([M.frame_labels(c.n_frames, c.gt_intervals) for c in calib])

    method = cfg.get("calm", {}).get("calibration_method", "isotonic")
    n_pos_cal = int(y_cal.sum())
    if method == "isotonic" and (len(y_cal) < 2000 or n_pos_cal < 200):
        method = "platt"
    cal = Calibrator(method).fit(bel_cal, y_cal)

    prob = {c.name: cal.predict(scored[c.name]["belief"]) for c in test}

    anom_clips = [c for c in test if c.gt_intervals]
    total_gt = sum(len(c.gt_intervals) for c in anom_clips)
    print(f"[{args.tag}] {len(anom_clips)} anomalous test clips, {total_gt} GT events")

    # global max prob ever reached, per anomalous clip, to see if any clip's
    # belief simply never rises no matter what
    never_rises = []
    for c in anom_clips:
        pmax = float(prob[c.name].max()) if len(prob[c.name]) else 0.0
        if pmax < 0.02:
            never_rises.append((c.name, pmax))
    print(f"[{args.tag}] anomalous clips whose calibrated prob NEVER exceeds 0.02 "
          f"anywhere: {len(never_rises)}/{len(anom_clips)}")
    if never_rises[:10]:
        print("  examples:", never_rises[:10])

    # ---- extended grid: original harness grid (0.98->0.02) plus a much
    # lower tail (0.02 -> 0.0), same recall computation as harness.py ----
    grid = np.concatenate([np.linspace(0.98, 0.02, 60), np.linspace(0.018, 0.0, 40)])
    print(f"\n{'tau':>8} {'global_recall':>14}")
    prev_bucket = None
    for tau in grid:
        caught = 0
        for c in anom_clips:
            preds = M.score_to_intervals(prob[c.name], float(tau), c.fps)
            _, _, rec = M.event_f1(preds, c.gt_intervals, tiou=0.1)
            caught += rec * len(c.gt_intervals)
        recall = caught / total_gt if total_gt else 0.0
        bucket = round(tau, 3)
        # only print every ~10th grid point plus anything near the old floor / new max
        if prev_bucket is None or abs(bucket - prev_bucket) >= 0.05 or tau in (grid[59], grid[-1]):
            print(f"{tau:8.4f} {recall:14.4f}")
            prev_bucket = bucket

    # final: recall at tau=0 exactly (every frame with score>=0 "fires" --
    # the absolute most permissive threshold possible)
    caught = 0
    for c in anom_clips:
        preds = M.score_to_intervals(prob[c.name], 0.0, c.fps)
        _, _, rec = M.event_f1(preds, c.gt_intervals, tiou=0.1)
        caught += rec * len(c.gt_intervals)
    recall_at_zero = caught / total_gt if total_gt else 0.0
    print(f"\n[{args.tag}] recall at tau=0.0 (most permissive threshold possible): "
          f"{recall_at_zero:.4f}")
    print(f"[{args.tag}] old grid's floor (tau=0.02) recall was printed above -- "
          f"compare the two.")


if __name__ == "__main__":
    main()
