"""
external_scores.py  --  score an EXTERNAL, precomputed per-frame anomaly
score (e.g. a trained pose-density baseline such as STG-NF) through the exact
same evaluation protocol calm.harness.run() uses for CALM-VAD / B0-B4:

    event-F1 @ tIoU {0.2..0.5}, FAPH @ fixed recall {0.7,0.8,0.9}, ECE /
    adaptive-ECE / Brier (raw vs. calibrated), frame AUC.

Why this exists (see paper/main.tex Baselines paragraph, B3): the paper
deliberately did not reproduce a published pose-density baseline like STG-NF,
citing published numbers instead, because "a fair run would mean cloning and
training that method's full codebase... a different scope of work". This
module is the other half of closing that gap once the codebase actually IS
cloned and trained (see colab/stgnf_baseline_colab.ipynb): it takes whatever
per-frame anomaly score the external method produces on our test clips and
puts it through the identical M3-calibration + event/FAPH/ECE pipeline, so
the resulting numbers are comparable to the rest of the paper's tables, not
just a frame-AUC citation.

It deliberately does NOT run CueBank / EvidenceFusion / M1 / M4 -- those are
CALM-VAD's own front end and have no equivalent for an opaque external score.
B0/B1/B2/B4 (which need the cue-strength vector) are also out of scope here.
What IS reused, by hand-kept-in-sync design, from harness.py's run(): the
calib/test split rule and the M3 calibration-method selection rule, so a
difference in the numbers reflects the METHOD, not a difference in protocol.

Usage:
    python -m calm.external_scores \
        --clips data/pose/shanghaitech.json \
        --scores results/stgnf_raw_scores_shanghaitech.json \
        --tag stgnf_shanghaitech --label "STG-NF (B3, reproduced)"

Score-file format (produced by the Colab notebook's conversion cell):
    {"<clip_name>": [s_0, s_1, ..., s_{n_frames-1}], ...}
One flat float per frame, same frame count and clip-name convention as the
matching data/pose/<dataset>.json (so clips line up by name). Convention:
HIGHER = MORE ANOMALOUS (STG-NF's own raw output is a normality
log-likelihood -- high means normal -- so the notebook flips its sign before
writing this file; see the notebook's scoring cell for the exact flip).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from .calibration import Calibrator, expected_calibration_error
from .risk_control import count_alarm_events
from . import metrics as M


def _split_like_harness(clips):
    """
    Mirrors calm/harness.py run()'s calib/test carve-out exactly. Kept as a
    separate copy (not imported) so this module has no dependency on
    CueBank/EvidenceFusion; if that logic changes in harness.py, mirror the
    change here too.
    """
    calib = [c for c in clips if c.split == "calib"]
    test = [c for c in clips if c.split == "test"]
    if not test:
        test = [c for c in clips if c.split != "calib"] or clips
    if (not calib or not any(c.gt_intervals for c in calib)
            or not any(not c.gt_intervals for c in calib)):
        norm = [c for c in test if not c.gt_intervals]
        anom = [c for c in test if c.gt_intervals]
        carve = norm[: max(1, len(norm) // 3)] + anom[: max(1, len(anom) // 3)]
        calib = list({c.name: c for c in (calib + carve)}.values())
        calib_names = {c.name for c in calib}
        test = [c for c in test if c.name not in calib_names] or test
    return calib, test


def run_external(clips, scores_by_clip, cfg=None, tag="external",
                  label="external baseline", budgets=(2.0, 5.0, 10.0),
                  recalls=(0.7, 0.8, 0.9), delta=0.05, outdir="results"):
    cfg = cfg or {}
    empty = [c.name for c in clips if c.n_frames == 0]
    if empty:
        print(f"[external_scores] dropping {len(empty)} zero-frame clip(s): "
              f"{empty[:5]}{' ...' if len(empty) > 5 else ''}")
        clips = [c for c in clips if c.n_frames > 0]
    missing = [c.name for c in clips if c.name not in scores_by_clip]
    if missing:
        raise SystemExit(
            f"[external_scores] {len(missing)} clip(s) have no external score "
            f"(name mismatch between --clips and --scores?): {missing[:10]}")
    bad_len = [c.name for c in clips if len(scores_by_clip[c.name]) != c.n_frames]
    if bad_len:
        raise SystemExit(
            f"[external_scores] {len(bad_len)} clip(s) have a score array whose "
            f"length != n_frames (a silent misalignment would corrupt every "
            f"metric below, so this is fatal, not a warning): {bad_len[:10]}")
    if not clips:
        raise SystemExit("[external_scores] no clips with frames left to evaluate")

    fps = clips[0].fps
    calib, test = _split_like_harness(clips)

    def stack_raw(split_clips):
        b, y = [], []
        for c in split_clips:
            b.append(np.asarray(scores_by_clip[c.name], float))
            y.append(M.frame_labels(c.n_frames, c.gt_intervals))
        return np.concatenate(b), np.concatenate(y)

    raw_cal, y_cal = stack_raw(calib)
    raw_te, y_te = stack_raw(test)

    # ---- rank-preserving normalization to [0,1] before calibration --------
    # Calibrator's platt/temperature methods assume x is roughly in [0,1]
    # (they logit-clip it); an external method's raw score (e.g. STG-NF's
    # log-likelihood) is an unbounded real. Isotonic doesn't need this, but
    # we do it unconditionally so behaviour doesn't depend on which method
    # the set-size rule below picks. It changes nothing about AUC / event-F1
    # / FAPH, since those are computed from the post-calibration probability
    # (rank-preserving monotone map of a rank-preserving monotone map).
    lo, hi = float(raw_cal.min()), float(raw_cal.max())
    span = (hi - lo) if hi > lo else 1.0
    norm_cal = np.clip((raw_cal - lo) / span, 0.0, 1.0)
    norm_te = np.clip((raw_te - lo) / span, 0.0, 1.0)

    method = cfg.get("calm", {}).get("calibration_method", "isotonic")
    n_pos_cal = int((y_cal == 1).sum())
    if method == "isotonic" and (len(y_cal) < 2000 or n_pos_cal < 200):
        method = "platt"
        print(f"[external_scores] small calibration set (n={len(y_cal)}, "
              f"pos={n_pos_cal}) -> using Platt instead of isotonic")
    cal = Calibrator(method).fit(norm_cal, y_cal)
    cal_report = cal.report(norm_te, y_te)
    cal_report["region"] = f"all test frames; n={len(y_te)}, pos={int((y_te == 1).sum())}"
    ece_noCal = expected_calibration_error(norm_te, y_te)

    prob_by_clip = {}
    for c in test:
        raw = np.asarray(scores_by_clip[c.name], float)
        norm = np.clip((raw - lo) / span, 0.0, 1.0)
        prob_by_clip[c.name] = cal.predict(norm)

    # ---- frame AUC + event-F1 + FAPH, identical recipe to harness.eval_stream
    allp = np.concatenate([prob_by_clip[c.name] for c in test])
    auc = M.roc_auc(allp, y_te)

    anom_clips = [c for c in test if c.gt_intervals]
    norm_clips = [c for c in test if not c.gt_intervals]
    ev_rows = []
    for c in test:
        preds = M.score_to_intervals(prob_by_clip[c.name], 0.5, c.fps)
        ev_rows.append(M.event_f1_multi(preds, c.gt_intervals))
    ev = {k: float(np.mean([e[k] for e in ev_rows])) for k in ev_rows[0]}

    norm_hours = sum(c.n_frames for c in norm_clips) / fps / 3600.0
    total_gt = sum(len(c.gt_intervals) for c in anom_clips)
    faph = {}
    for r in recalls:
        hit_faph = None
        for tau in np.linspace(0.98, 0.02, 60):
            caught = 0
            for c in anom_clips:
                preds = M.score_to_intervals(prob_by_clip[c.name], tau, c.fps)
                _, _, rec = M.event_f1(preds, c.gt_intervals, tiou=0.1)
                caught += rec * len(c.gt_intervals)
            recall = caught / total_gt if total_gt else 0.0
            if recall >= r:
                fa = sum(count_alarm_events(prob_by_clip[c.name], tau, c.fps)
                         for c in norm_clips)
                hit_faph = fa / norm_hours if norm_hours > 0 else float("inf")
                break
        faph[f"faph@recall={r}"] = None if hit_faph is None else round(float(hit_faph), 2)

    stream = {"label": label, "frame_auc": auc, "event": ev, **faph}

    report = {
        "tag": tag, "n_clips": len(clips), "n_calib": len(calib), "n_test": len(test),
        "fps": fps,
        "calibration": {**cal_report, "ece_no_calibration": ece_noCal},
        "streams": [stream],
        "note": ("M1/M4/B0-B2 are CALM-VAD-specific and not evaluated for an opaque "
                 "external score; only frame AUC / event-F1 / FAPH / ECE are reported, "
                 "computed with the same calib/test split and M3 calibration rule as "
                 "calm.harness.run() so these numbers are comparable to its output."),
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"calm_report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    txt = _render(report)
    with open(os.path.join(outdir, f"calm_report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n  -> results/calm_report_{tag}.json / .txt")
    return report


def _render(r) -> str:
    L = []
    L.append("=" * 66)
    L.append(f"  External-baseline deployable-evaluation report   [{r['tag']}]")
    L.append("=" * 66)
    L.append(f"  clips: {r['n_clips']}  (calib {r['n_calib']} / test {r['n_test']})   fps {r['fps']}")
    c = r["calibration"]
    L.append("-" * 66)
    L.append("  CALIBRATION (test split)")
    L.append(f"    method              : {c['method']}")
    L.append(f"    region              : {c.get('region', 'all frames')}")
    L.append(f"    ECE  raw -> cal      : {c['ece_raw']:.4f} -> {c['ece_cal']:.4f}")
    L.append(f"    adaptive-ECE r -> c  : {c['aece_raw']:.4f} -> {c['aece_cal']:.4f}")
    L.append(f"    Brier raw -> cal     : {c['brier_raw']:.4f} -> {c['brier_cal']:.4f}")
    L.append("-" * 66)
    L.append("  DETECTION + ALARM LOAD (mean over test clips)")
    for s in r["streams"]:
        L.append(f"    {s['label']}")
        L.append(f"       frame AUC        : {s['frame_auc']:.3f}")
        L.append(f"       event F1 avg     : {s['event']['f1_avg']:.3f}   "
                 f"(@0.2 {s['event']['f1@0.2']:.2f} / @0.5 {s['event']['f1@0.5']:.2f})")
        for k in s:
            if k.startswith("faph@"):
                v = s[k]
                L.append(f"       {k:<16}: " + ("n/a" if v is None else f"{v:.1f} /hour"))
    L.append("-" * 66)
    L.append(f"  NOTE: {r['note']}")
    L.append("=" * 66)
    return "\n".join(L)


def _load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: np.asarray(arr, float) for name, arr in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True,
                    help="generic-JSON pose+gt file (calm.datasets.load_generic_json)")
    ap.add_argument("--scores", required=True,
                    help='JSON {"clip_name": [scores...]}, higher = more anomalous')
    ap.add_argument("--tag", default=None)
    ap.add_argument("--label", default="external baseline")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    from . import datasets as D
    clips = D.load_generic_json(args.clips)
    scores_by_clip = _load_scores(args.scores)
    tag = args.tag or os.path.splitext(os.path.basename(args.scores))[0]

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    run_external(clips, scores_by_clip, cfg, tag=tag, label=args.label)


if __name__ == "__main__":
    main()
