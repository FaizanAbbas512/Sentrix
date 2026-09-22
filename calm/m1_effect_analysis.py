"""
m1_effect_analysis.py  —  does M1 (reliability discounting) do anything real
=============================================================================

A targeted, honest investigation of M1's real-data effect, run on the four
non-ShanghaiTech real benchmarks (UBnormal, CHAD, Avenue, NWPU-Campus).
ShanghaiTech is deliberately excluded here (a concurrent edit is touching its
pose file / report).

Re-uses the SAME machinery the paper's numbers come from:
  * calm.cues.CueBank            — identical cue-strength computation to harness.py
  * calm.reliability.ReliabilityEstimator — identical r_pose computation to harness.py
  * calm.harness.score_clip      — identical M1-on / M1-off belief scoring
  * calm.metrics.roc_auc / event_f1 / score_to_intervals
  * calm.risk_control.count_alarm_events

Nothing here re-derives r_pose or belief with new math; it only *instruments*
the existing per-frame loop (calm/harness.py's score_clip, calm/cues.py's
CueBank) to record what r_pose and the cue strengths were on every frame, and
reuses harness.py's own FAPH/AUC building blocks for the before/after check.

Usage:
    python -m calm.m1_effect_analysis
    python -m calm.m1_effect_analysis --low-cutoff 0.5

Writes:
    results/m1_reliability_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

from .cues import CueBank, CUE_NAMES
from .reliability import ReliabilityEstimator
from . import datasets as D
from . import metrics as M
from .risk_control import count_alarm_events
from .harness import score_clip

# Same hardcoded frame_shape harness.py's score_clip uses for every real
# dataset today (see calm/harness.py score_clip). We replicate it exactly so
# r_pose here is IDENTICAL to what the shipped harness computes -- not a
# re-derivation with different assumptions.
FRAME_SHAPE = (480, 640)

DATASETS = {
    "ubnormal": "data/pose/ubnormal.json",
    "chad": "data/pose/chad.json",
    "avenue": "data/pose/avenue.json",
    "nwpucampus": "nwpu results/nwpucampus.json",
}


# --------------------------------------------------------------------------- #
#  instrumented per-frame pass (reuses CueBank + ReliabilityEstimator as-is)  #
# --------------------------------------------------------------------------- #
def instrument_clip(clip, cfg):
    """
    One pass over a clip's frames using the exact same CueBank +
    ReliabilityEstimator objects/calls harness.score_clip uses, but also
    recording per-person reliability + features so we can characterise the
    real distribution (step 1/2/4), not just the single pipeline r_pose
    scalar. Each person is scored with rel.reliability(...) EXACTLY ONCE per
    frame (the estimator keeps per-track state -- e.g. track_stability advances
    a frame counter on every call -- so calling it twice per person per frame
    would corrupt that internal state and silently diverge from harness.py's
    real numbers).
    """
    cb = CueBank(cfg, fps=clip.fps)
    rel = ReliabilityEstimator(cfg)

    n = clip.n_frames
    r_pipeline = np.ones(n, float)       # what M1 actually feeds to M2 (driver-pool-or-all)
    r_all_mean = np.ones(n, float)       # mean r over every person in the frame
    r_all_min = np.ones(n, float)        # worst person in the frame
    vis_frac_mean = np.ones(n, float)    # mean visible_joint_frac over persons (occlusion proxy)
    n_persons = np.zeros(n, int)
    any_cue = np.zeros(n, bool)
    cue_active = np.zeros((n, len(CUE_NAMES)), bool)
    crowd_active = np.zeros(n, bool)

    for t, dets in enumerate(clip.frames):
        strengths, aux = cb.step(dets)
        persons = [d for d in dets if d.get("is_person", True)]

        rs, vfs = [], []
        for d in persons:
            r, feat = rel.reliability(d["track_id"], d.get("keypoints"),
                                      d["bbox"], FRAME_SHAPE)
            rs.append(r)
            vfs.append(feat.visible_joint_frac)

        drivers = (aux["fallen_ids"] | aux["fight_ids"]
                   | aux["loiter_ids"] | aux["abandoned_ids"])
        if rs:
            idx_pool = [i for i, d in enumerate(persons) if d["track_id"] in drivers]
            pool_rs = [rs[i] for i in idx_pool] if idx_pool else rs
            r_pipeline[t] = float(np.mean(pool_rs))
            r_all_mean[t] = float(np.mean(rs))
            r_all_min[t] = float(np.min(rs))
            vis_frac_mean[t] = float(np.mean(vfs))
        # else: leave at the "fully trusted, nobody to discount" defaults (1.0)

        n_persons[t] = len(persons)
        for i, name in enumerate(CUE_NAMES):
            cue_active[t, i] = strengths[name] > 0.0
        any_cue[t] = bool(cue_active[t].any())
        crowd_active[t] = strengths["CROWD"] > 0.0

    return dict(r_pipeline=r_pipeline, r_all_mean=r_all_mean, r_all_min=r_all_min,
                vis_frac_mean=vis_frac_mean, n_persons=n_persons,
                any_cue=any_cue, cue_active=cue_active, crowd_active=crowd_active)


# --------------------------------------------------------------------------- #
#  step 1 : real reliability distribution                                    #
# --------------------------------------------------------------------------- #
def distribution_stats(r_pipeline, n_persons):
    has_person = n_persons > 0
    r_wp = r_pipeline[has_person]   # only frames that actually had someone to score
    out = {
        "n_frames_total": int(len(r_pipeline)),
        "n_frames_with_person": int(has_person.sum()),
        "frac_frames_with_person": float(has_person.mean()) if len(r_pipeline) else 0.0,
    }
    if len(r_wp) == 0:
        out["note"] = "no frames with a detected person"
        return out
    out["mean_r_pipeline_when_person_present"] = float(r_wp.mean())
    out["median_r_pipeline_when_person_present"] = float(np.median(r_wp))
    for thr in (0.9, 0.7, 0.5, 0.3, 0.1):
        out[f"frac_below_{thr}"] = float((r_wp < thr).mean())
    out["p10"] = float(np.percentile(r_wp, 10))
    out["p5"] = float(np.percentile(r_wp, 5))
    out["p1"] = float(np.percentile(r_wp, 1))
    return out


# --------------------------------------------------------------------------- #
#  step 2 : overlap between low reliability and active cues                  #
# --------------------------------------------------------------------------- #
def overlap_stats(r_pipeline, any_cue, n_persons, cutoff):
    has_person = n_persons > 0
    low = (r_pipeline < cutoff) & has_person
    n_low = int(low.sum())
    n_low_and_cue = int((low & any_cue).sum())
    n_cue = int(any_cue.sum())
    n_cue_and_low = int((any_cue & low).sum())
    return {
        "cutoff": cutoff,
        "n_frames_low_reliability": n_low,
        "frac_low_reliability_of_all": float(low.mean()) if len(low) else 0.0,
        "n_frames_active_cue": n_cue,
        "frac_active_cue_of_all": float(any_cue.mean()) if len(any_cue) else 0.0,
        "n_low_reliability_AND_active_cue": n_low_and_cue,
        "frac_of_low_reliability_frames_with_active_cue": (
            n_low_and_cue / n_low if n_low else 0.0),
        "frac_of_active_cue_frames_that_are_low_reliability": (
            n_cue_and_low / n_cue if n_cue else 0.0),
    }


# --------------------------------------------------------------------------- #
#  step 3 : targeted before/after on the overlap subset                      #
# --------------------------------------------------------------------------- #
def score_all_clips(clips, cfg):
    """Score every clip once with M1 on and once with M1 off (calm.harness.score_clip,
    unmodified) so step3/step4 can reuse the same belief arrays instead of re-scoring
    the whole dataset for every subset check."""
    scored = {}
    for c in clips:
        s_calm = score_clip(c, cfg, use_reliability=True)
        s_noM1 = score_clip(c, cfg, use_reliability=False)
        scored[c.name] = (s_calm, s_noM1)
    return scored


def targeted_comparison(clips, cfg, low_mask_by_clip, any_cue_by_clip, scored=None):
    """
    For the given dataset's clips, using the already-scored M1-on / M1-off
    raw beliefs (calm.harness.score_clip, unmodified), compare:
      (a) frame AUC restricted to exactly the low-reliability+active-cue
          frame subset (well-defined for any non-contiguous frame set).
      (b) event-level FAPH before/after, restricted to whichever CLIPS
          contain a non-trivial number of overlap frames, PROVIDED that
          subset still has at least one normal (gt-free) clip and one
          anomalous (gt) clip -- otherwise FAPH is not computable and we
          say so rather than force a number.
    """
    all_belief_calm, all_belief_noM1, all_y, all_mask = [], [], [], []
    per_clip_overlap_frames = {}

    if scored is None:
        scored = score_all_clips(clips, cfg)
    for c in clips:
        s_calm, s_noM1 = scored[c.name]
        y = M.frame_labels(c.n_frames, c.gt_intervals)
        mask = low_mask_by_clip[c.name] & any_cue_by_clip[c.name]
        n_ov = int(mask.sum())
        per_clip_overlap_frames[c.name] = n_ov
        if n_ov:
            all_belief_calm.append(s_calm["belief"][mask])
            all_belief_noM1.append(s_noM1["belief"][mask])
            all_y.append(y[mask])

    result = {"per_clip_overlap_frame_count": per_clip_overlap_frames}

    total_overlap = sum(per_clip_overlap_frames.values())
    result["total_overlap_frames"] = total_overlap
    if total_overlap == 0:
        result["frame_auc_on_overlap_subset"] = None
        result["note"] = "zero low-reliability+active-cue frames in this dataset; no subset to evaluate"
        return result

    bc = np.concatenate(all_belief_calm)
    bn = np.concatenate(all_belief_noM1)
    y = np.concatenate(all_y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    result["overlap_subset_n_pos_frames"] = n_pos
    result["overlap_subset_n_neg_frames"] = n_neg

    auc_calm = M.roc_auc(bc, y) if n_pos and n_neg else None
    auc_noM1 = M.roc_auc(bn, y) if n_pos and n_neg else None
    result["frame_auc_on_overlap_subset"] = {
        "M1_on": auc_calm, "M1_off": auc_noM1,
        "delta_M1_on_minus_off": (None if auc_calm is None or auc_noM1 is None
                                  else float(auc_calm - auc_noM1)),
        "note": None if (n_pos and n_neg) else
                "overlap subset has only one class of frames (all-anomaly or all-normal); AUC undefined",
    }
    result["belief_mean_on_overlap_subset"] = {
        "M1_on_mean_belief": float(bc.mean()), "M1_off_mean_belief": float(bn.mean()),
        "M1_on_mean_belief_pos_frames": float(bc[y == 1].mean()) if n_pos else None,
        "M1_off_mean_belief_pos_frames": float(bn[y == 1].mean()) if n_pos else None,
        "M1_on_mean_belief_neg_frames": float(bc[y == 0].mean()) if n_neg else None,
        "M1_off_mean_belief_neg_frames": float(bn[y == 0].mean()) if n_neg else None,
    }

    # event-level FAPH restricted to clips that contain overlap frames
    overlap_clip_names = {name for name, k in per_clip_overlap_frames.items() if k > 0}
    sub_clips = [c for c in clips if c.name in overlap_clip_names]
    norm_sub = [c for c in sub_clips if not c.gt_intervals]
    anom_sub = [c for c in sub_clips if c.gt_intervals]
    if not norm_sub or not anom_sub:
        result["event_faph_on_overlap_clips"] = None
        result["event_faph_note"] = (
            f"{len(sub_clips)} clip(s) contain overlap frames "
            f"({len(norm_sub)} normal / {len(anom_sub)} anomalous); FAPH needs at "
            "least one of each to define both recall (on anomalous clips) and "
            "false-alarm rate (on normal clips) -- not computable on this subset.")
    else:
        fps = clips[0].fps
        norm_hours = sum(c.n_frames for c in norm_sub) / fps / 3600.0
        total_gt = sum(len(c.gt_intervals) for c in anom_sub)
        faph_rows = {}
        for label, key in (("M1_on", "calm"), ("M1_off", "noM1")):
            prob_by_clip = {c.name: scored[c.name][0 if key == "calm" else 1]["belief"]
                            for c in sub_clips}
            per_recall = {}
            for r in (0.7, 0.8, 0.9):
                hit = None
                for tau in np.linspace(0.98, 0.02, 60):
                    caught = 0
                    for c in anom_sub:
                        preds = M.score_to_intervals(prob_by_clip[c.name], tau, c.fps)
                        _, _, rec = M.event_f1(preds, c.gt_intervals, tiou=0.1)
                        caught += rec * len(c.gt_intervals)
                    recall = caught / total_gt if total_gt else 0.0
                    if recall >= r:
                        fa = sum(count_alarm_events(prob_by_clip[c.name], tau, c.fps)
                                 for c in norm_sub)
                        hit = fa / norm_hours if norm_hours > 0 else float("inf")
                        break
                per_recall[f"faph@recall={r}"] = None if hit is None else round(float(hit), 3)
            faph_rows[label] = per_recall
        result["event_faph_on_overlap_clips"] = {
            "clips_used": sorted(overlap_clip_names),
            "n_normal_clips": len(norm_sub), "n_anomalous_clips": len(anom_sub),
            "normal_hours": round(norm_hours, 4),
            **faph_rows,
        }
    return result


# --------------------------------------------------------------------------- #
#  step 4 : occlusion / crowd as an alternative real-world proxy             #
# --------------------------------------------------------------------------- #
def occlusion_lens(clips, cfg, crowd_by_clip, lowvis_by_clip, any_cue_by_clip,
                   low_mask_by_clip, scored=None):
    """
    Two alternative "naturally occluded" masks, independent of r_pose:
      * crowd_active   : CROWD cue firing (many people -> partial occlusion likely)
      * low_visible_joint_frac : < 0.5 of the 17 joints seen, regardless of r_pose
    Compares each against the r_pose-based low-reliability mask (step 3's cut)
    to see if they pick out the same frames or a different population, and runs
    the same frame-AUC before/after check on the crowd/occlusion-defined subset.
    """
    if scored is None:
        scored = score_all_clips(clips, cfg)
    out = {}
    for lens_name, mask_by_clip in (("crowd_active", crowd_by_clip),
                                    ("low_visible_joint_frac<0.5", lowvis_by_clip)):
        n_total = sum(len(m) for m in mask_by_clip.values())
        n_lens = sum(int(m.sum()) for m in mask_by_clip.values())
        n_lens_and_cue = sum(int((mask_by_clip[c.name] & any_cue_by_clip[c.name]).sum())
                             for c in clips)
        n_lens_and_rlow = sum(int((mask_by_clip[c.name] & low_mask_by_clip[c.name]).sum())
                              for c in clips)
        overlap_mask_by_clip = {c.name: mask_by_clip[c.name] & any_cue_by_clip[c.name]
                                for c in clips}
        cmp_ = targeted_comparison(clips, cfg, overlap_mask_by_clip,
                                   {c.name: np.ones(c.n_frames, bool) for c in clips},
                                   scored=scored)
        out[lens_name] = {
            "n_frames_in_lens": n_lens,
            "frac_of_all_frames": (n_lens / n_total) if n_total else 0.0,
            "n_lens_AND_active_cue": n_lens_and_cue,
            "n_lens_AND_r_pose_low": n_lens_and_rlow,
            "frac_lens_frames_that_are_also_r_pose_low": (
                n_lens_and_rlow / n_lens if n_lens else 0.0),
            "before_after_on_lens_AND_active_cue_subset": cmp_,
        }
    return out


# --------------------------------------------------------------------------- #
def run_dataset(name, path, cfg, low_cutoff):
    t0 = time.time()
    print(f"[{name}] loading {path} ...")
    clips = D.load_generic_json(path)
    clips = [c for c in clips if c.n_frames > 0]
    print(f"[{name}] {len(clips)} clips, "
          f"{sum(c.n_frames for c in clips)} frames, "
          f"{sum(len(c.gt_intervals) for c in clips)} gt events "
          f"-- loaded in {time.time()-t0:.1f}s")

    per_clip = {}
    t1 = time.time()
    for c in clips:
        per_clip[c.name] = instrument_clip(c, cfg)
    print(f"[{name}] instrumented pass done in {time.time()-t1:.1f}s")

    r_pipeline_all = np.concatenate([per_clip[c.name]["r_pipeline"] for c in clips])
    n_persons_all = np.concatenate([per_clip[c.name]["n_persons"] for c in clips])
    any_cue_all = np.concatenate([per_clip[c.name]["any_cue"] for c in clips])

    step1 = distribution_stats(r_pipeline_all, n_persons_all)

    # Two cutoffs: the fixed "natural threshold" requested (--low-cutoff, default
    # 0.5), and an adaptive per-dataset bottom-decile cutoff (step1's own p10) --
    # in case (as with Avenue) the dataset's real r_pose never gets anywhere near
    # 0.5, so a fixed cutoff would trivially yield an empty subset and tell us
    # nothing. Both are reported; neither is hidden if it comes back empty.
    p10 = step1.get("p10")
    cutoffs = {"fixed": low_cutoff}
    if p10 is not None:
        cutoffs["adaptive_p10"] = float(p10)

    t2 = time.time()
    scored = score_all_clips(clips, cfg)
    print(f"[{name}] scored all clips (M1 on/off) in {time.time()-t2:.1f}s")

    crowd_by_clip = {c.name: per_clip[c.name]["crowd_active"] for c in clips}
    lowvis_by_clip = {c.name: (per_clip[c.name]["vis_frac_mean"] < 0.5)
                              & (per_clip[c.name]["n_persons"] > 0)
                      for c in clips}
    any_cue_by_clip = {c.name: per_clip[c.name]["any_cue"] for c in clips}

    step2 = {}
    step3 = {}
    step4 = {}
    for cutoff_label, cutoff_val in cutoffs.items():
        step2[cutoff_label] = overlap_stats(r_pipeline_all, any_cue_all, n_persons_all, cutoff_val)

        low_mask_by_clip = {c.name: (per_clip[c.name]["r_pipeline"] < cutoff_val)
                                    & (per_clip[c.name]["n_persons"] > 0)
                            for c in clips}

        t2b = time.time()
        step3[cutoff_label] = targeted_comparison(clips, cfg, low_mask_by_clip,
                                                   any_cue_by_clip, scored=scored)
        print(f"[{name}] step3 ({cutoff_label}={cutoff_val:.3f}) done in {time.time()-t2b:.1f}s")

        t3 = time.time()
        step4[cutoff_label] = occlusion_lens(clips, cfg, crowd_by_clip, lowvis_by_clip,
                                             any_cue_by_clip, low_mask_by_clip, scored=scored)
        print(f"[{name}] step4 ({cutoff_label}) done in {time.time()-t3:.1f}s")

    return {
        "dataset": name, "n_clips": len(clips),
        "n_frames": int(sum(c.n_frames for c in clips)),
        "n_gt_events": int(sum(len(c.gt_intervals) for c in clips)),
        "cutoffs_used": cutoffs,
        "step1_reliability_distribution": step1,
        "step2_overlap_with_active_cue": step2,
        "step3_targeted_before_after": step3,
        "step4_occlusion_crowd_lens": step4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-cutoff", type=float, default=0.5,
                    help="r_pose cutoff for 'low reliability' (default 0.5; see step1 before picking)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", default=None, help="comma-separated subset of dataset names to run")
    ap.add_argument("--out", default="results/m1_reliability_analysis.json")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = list(DATASETS)
    if args.only:
        names = [n.strip() for n in args.only.split(",")]

    report = {"low_cutoff": args.low_cutoff, "datasets": {}}
    for name in names:
        path = DATASETS[name]
        report["datasets"][name] = run_dataset(name, path, cfg, args.low_cutoff)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"  -> partial results written to {args.out}")

    print(f"\nDone. Full results in {args.out}")


if __name__ == "__main__":
    main()
