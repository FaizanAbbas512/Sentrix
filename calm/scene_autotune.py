"""
scene_autotune.py  —  scene-adaptive cue thresholds, learned from
CONFIRMED-NORMAL footage statistics only (no anomaly labels)
===========================================================================

Problem this addresses: the five heuristic cues in calm/cues.py (FALL /
FIGHT / ABANDONED / LOITERING / CROWD) read their thresholds straight out of
config.yaml as FIXED constants -- e.g. `loitering.dwell_seconds: 15`,
`fight.proximity: 130`, `crowd.threshold: 8`. Those numbers were picked "by
feel" against one camera setup and stay the same regardless of camera
distance, scene density, or site. A crowd of 8 means something very
different in a small shop than on a train-station concourse; 130px of
"proximity" means something very different on a close indoor camera than a
wide outdoor one.

This module derives site-specific replacements for a subset of those
constants from statistics of a site's own CONFIRMED-NORMAL footage --
nothing here ever looks at an anomaly label. That is a hard design
constraint, not a convenience: it is what makes this tuner usable live,
during deployment warm-up, exactly the way calm/detector.py's M4 already
self-calibrates the alarm budget from normal-only footage (see config.yaml
`calm.auto_calibrate` / `warmup_hours`). A tuner that needed anomaly labels
would just be a smaller version of the "M3 needs labels" problem this
project already solved around differently for M4 -- undeployable outside a
labelled research benchmark.

What gets tuned, and the statistic behind each (see the four `_tune_*`
functions below for the exact formulas):

  * crowd.threshold              <- a high percentile of the site's own
                                     normal per-frame person count.
  * loitering.dwell_seconds      <- a high percentile of how long people, at
                                     the site, normally pause in one place
                                     before moving on.
  * loitering.movement_tolerance <- normalised by the site's own typical
                                     person size in frame (see
                                     _person_scale_px below), not left as a
                                     fixed pixel constant.
  * fight.proximity              <- also normalised by the site's own
                                     typical person size in frame.
  * abandoned.unattended_seconds
    abandoned.owner_radius
    abandoned.movement_tolerance <- normal-footage statistics of how objects
                                     and their owners actually behave at this
                                     site (object jitter, person-to-object
                                     proximity, how long objects are ever
                                     briefly left alone). Requires the site's
                                     normal footage to actually contain
                                     object (non-person) detections; if it
                                     doesn't, these fall back to the fixed
                                     config values (see MIN_OBJECT_* guards).

What is deliberately left FIXED, and why (not every parameter honestly fits
this scheme -- see `LEFT_FIXED_REASONS` at the bottom): agitation_threshold
(behaviour + fight), fight.motion_threshold, the confirm_seconds /
fall_confirm_seconds latency constants, fall.aspect_ratio, and
behaviour.history_frames. Every one of those describes what ABNORMAL motion
or an already-flagged state looks like, or is a pure latency/smoothing
knob -- not a property of normal footage. Forcing a "normal-derived"
threshold onto them would mean inventing an arbitrary multiple of normal
behaviour dressed up as data-derived, which is exactly the kind of dishonest
tuning this module is built to avoid.

Two design-constant caveats, stated plainly (not hidden in the formulas):
  1. Every tuned value is `(data-derived reference statistic) x (a fixed
     multiplier)`. The reference statistic (a percentile of normal
     occupancy / dwell time / person size / proximity) is genuinely learned
     from this site's normal footage and is what makes the threshold
     scene-adaptive. The multiplier (e.g. "2.0x person scale" for FIGHT
     proximity, "1.3x the P90 dwell" for LOITERING) is a fixed judgment call
     we picked and documented, the same way the existing strength ramps in
     cues.py already hardcode e.g. `dwell * 2.5`. We do not claim the
     multiplier itself is learned -- only that the reference unit it scales
     now adapts per site instead of being an arbitrary global pixel/second
     constant.
  2. Every statistic has a minimum-sample guard. Below it we fall back to
     the fixed config value and say so explicitly in the returned report --
     we would rather ship a known-fixed constant than a percentile computed
     from a handful of frames.

Usage:

    from calm.scene_autotune import tune_cue_thresholds
    tuned_cfg = tune_cue_thresholds(normal_clips, base_cfg)
    # tuned_cfg is base_cfg with crowd/loitering/fight/abandoned thresholds
    # replaced where a normal-footage statistic honestly supports it, plus
    # a tuned_cfg["scene_autotune"] diagnostics block (per-parameter formula,
    # sample sizes, tuned-vs-fixed values) for logging / the eval report.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .cues import _centroid, _torso_len, CueBank


# --------------------------------------------------------------------------- #
#  design constants (fixed multipliers layered on top of data-derived        #
#  reference statistics -- see module docstring caveat (1)). Named and       #
#  centralised here so the honesty constraint is auditable in one place.     #
# --------------------------------------------------------------------------- #
CROWD_PERCENTILE = 99            # "busiest it normally gets"
CROWD_MARGIN_PEOPLE = 1          # + this many people before it's a "crowd"
MIN_NORMAL_FRAMES_FOR_CROWD = 200

LOITER_TOL_FRAC_OF_SCALE = 0.5   # "stationary" = within half a torso-length
LOITER_DWELL_PERCENTILE = 90     # top decile of normal per-track pause length
LOITER_DWELL_MARGIN = 1.3        # then go 30% past that before flagging
MIN_TRACKS_FOR_LOITER_DWELL = 5

FIGHT_PROXIMITY_FRAC_OF_SCALE = 2.0   # "close enough to grapple"

ABANDONED_TOL_FRAC_OF_SCALE = 0.35    # objects jitter less than people sway
ABANDONED_OWNER_RADIUS_PERCENTILE = 90
ABANDONED_OWNER_RADIUS_MARGIN = 1.1
ABANDONED_UNATTENDED_PERCENTILE = 90
ABANDONED_UNATTENDED_MARGIN = 1.3
MIN_OBJECT_DETECTIONS_FOR_ABANDONED = 50
MIN_OBJECT_PERSON_COOCCURRENCE_FRAMES = 30
MIN_OBJECT_TRACKS_FOR_UNATTENDED = 5

LEFT_FIXED_REASONS = {
    "behaviour.agitation_threshold":
        "defines what ABNORMAL limb speed looks like; cannot be derived from "
        "normal-only statistics without inventing an arbitrary multiple of "
        "normal speed, which would just re-encode the same judgment call the "
        "fixed constant already makes.",
    "fight.agitation_threshold":
        "same reasoning as behaviour.agitation_threshold, applied to the "
        "fight-specific agitation gate.",
    "fight.motion_threshold":
        "a centroid-fallback motion threshold used only when pose is "
        "unavailable; same 'what does abnormal motion look like' problem as "
        "agitation_threshold.",
    "behaviour.fall_confirm_seconds":
        "a confirmation-latency constant (how long an already-flagged fallen "
        "posture must hold), not a description of normal footage.",
    "fight.confirm_seconds":
        "confirmation-latency constant, not a normal-footage statistic.",
    "fall.confirm_seconds":
        "confirmation-latency constant, not a normal-footage statistic.",
    "fall.aspect_ratio":
        "the upright/fallen silhouette boundary is set by human body "
        "proportions, not by scene geometry or site-specific normal "
        "statistics -- a site's normal footage (people standing, walking) "
        "gives no signal about where that boundary should sit.",
    "behaviour.history_frames":
        "a smoothing-window length, not a threshold.",
}


# --------------------------------------------------------------------------- #
#  shared reference statistic: how big is a person in THIS camera's frame?   #
# --------------------------------------------------------------------------- #
def _person_scale_px(clips) -> float | None:
    """
    Median torso length (shoulder-midpoint to hip-midpoint) over every
    person detection in `clips`, reusing cues._torso_len exactly as
    CueBank._agitation already does (same fallback: 100.0px for a
    degenerate/zero-length torso reading). Detections with no keypoints
    fall back to half the bbox height.

    This is the site's own pixel reference unit for "how big is a person in
    this camera's frame" -- used to turn fixed pixel constants (FIGHT
    proximity, the movement-tolerance spatial params) into values that scale
    with camera distance/geometry instead of meaning something different on
    every install. Returns None if no person detections are available at
    all (so callers can fall back to the fixed config cleanly).
    """
    scales = []
    for c in clips:
        for frame in c.frames:
            for d in frame:
                if not d.get("is_person", True):
                    continue
                kp = d.get("keypoints")
                if kp is not None:
                    kp = np.asarray(kp, dtype=float).reshape(-1, 3)
                    scales.append(_torso_len(kp))
                else:
                    x1, y1, x2, y2 = d["bbox"]
                    scales.append(max(float(y2) - float(y1), 1.0) * 0.5)
    if not scales:
        return None
    return float(np.median(scales))


# --------------------------------------------------------------------------- #
#  CROWD                                                                     #
# --------------------------------------------------------------------------- #
def _tune_crowd(clips, base_cfg):
    fixed = float(base_cfg.get("crowd", {}).get("threshold", 8))
    counts = [sum(1 for d in frame if d.get("is_person", True))
              for c in clips for frame in c.frames]
    if len(counts) < MIN_NORMAL_FRAMES_FOR_CROWD:
        return fixed, {
            "tuned": False,
            "reason": (f"only {len(counts)} normal frames available -- need "
                       f">= {MIN_NORMAL_FRAMES_FOR_CROWD} to estimate a normal "
                       "occupancy distribution"),
            "n_frames": len(counts), "fixed_value": fixed,
        }
    counts = np.asarray(counts, dtype=float)
    p99 = float(np.percentile(counts, CROWD_PERCENTILE))
    p95 = float(np.percentile(counts, 95))
    new_th = max(2.0, math.ceil(p99) + CROWD_MARGIN_PEOPLE)
    return new_th, {
        "tuned": True,
        "formula": f"ceil(P{CROWD_PERCENTILE}(normal per-frame person count)) + {CROWD_MARGIN_PEOPLE}",
        "n_frames": len(counts),
        f"p{CROWD_PERCENTILE}_normal_occupancy": p99, "p95_normal_occupancy": p95,
        "max_normal_occupancy": float(counts.max()),
        "new_value": new_th, "fixed_value": fixed,
    }


# --------------------------------------------------------------------------- #
#  LOITERING                                                                 #
# --------------------------------------------------------------------------- #
def _tune_loitering(clips, base_cfg, scale_px):
    lo = base_cfg.get("loitering", {})
    fixed_dwell = float(lo.get("dwell_seconds", 15))
    fixed_tol = float(lo.get("movement_tolerance", 60))

    if scale_px is None:
        return fixed_dwell, fixed_tol, {
            "tuned": False,
            "reason": "no person detections in normal footage -- cannot "
                      "derive a person-scale reference for movement_tolerance",
        }

    new_tol = float(max(5.0, LOITER_TOL_FRAC_OF_SCALE * scale_px))

    # Probe pass: replay normal footage through the SAME anchor/reset
    # tracking CueBank already uses live (reused, not reimplemented), with
    # dwell_seconds set high enough it never fires, movement_tolerance set
    # to the just-derived new_tol. For every track, record the longest
    # continuous "hasn't moved past new_tol from its anchor" streak observed
    # -- i.e. the longest normal pause, per person, at this site.
    probe_cfg = {**base_cfg, "loitering": {**lo, "dwell_seconds": 1.0e9,
                                           "movement_tolerance": new_tol}}
    track_max = {}
    for c in clips:
        cb = CueBank(probe_cfg, fps=c.fps)
        for dets in c.frames:
            cb.step(dets)
            for tid, tr in cb._loiter.items():
                d = cb.t - tr["first"]
                if d > track_max.get(tid, 0.0):
                    track_max[tid] = d

    if len(track_max) < MIN_TRACKS_FOR_LOITER_DWELL:
        return fixed_dwell, new_tol, {
            "tuned": False, "movement_tolerance_tuned": True,
            "movement_tolerance_new": new_tol,
            "movement_tolerance_formula":
                f"{LOITER_TOL_FRAC_OF_SCALE} * median normal person torso length (px)",
            "reason": (f"only {len(track_max)} distinct tracks observed in normal "
                       f"footage -- need >= {MIN_TRACKS_FOR_LOITER_DWELL} to estimate "
                       "a dwell-time distribution"),
        }

    vals = np.asarray(list(track_max.values()), dtype=float)
    p90 = float(np.percentile(vals, LOITER_DWELL_PERCENTILE))
    new_dwell = max(2.0, p90 * LOITER_DWELL_MARGIN)
    return new_dwell, new_tol, {
        "tuned": True,
        "scale_px": scale_px, "n_tracks": len(track_max),
        "movement_tolerance_new": new_tol,
        "movement_tolerance_formula":
            f"{LOITER_TOL_FRAC_OF_SCALE} * median normal person torso length (px)",
        f"p{LOITER_DWELL_PERCENTILE}_track_max_stationary_s": p90,
        "dwell_seconds_formula":
            f"{LOITER_DWELL_MARGIN} * P{LOITER_DWELL_PERCENTILE}(per-track longest "
            "continuous stationary streak in normal footage)",
        "new_value": new_dwell, "fixed_dwell_seconds": fixed_dwell,
        "fixed_movement_tolerance": fixed_tol,
    }


# --------------------------------------------------------------------------- #
#  FIGHT (proximity only -- see LEFT_FIXED_REASONS for agitation_threshold)  #
# --------------------------------------------------------------------------- #
def _tune_fight_proximity(base_cfg, scale_px):
    fi = base_cfg.get("fight", {})
    fixed = float(fi.get("proximity", 130))
    if scale_px is None:
        return fixed, {
            "tuned": False,
            "reason": "no person detections in normal footage -- cannot "
                      "derive a person-scale reference for proximity",
        }
    new_prox = float(max(10.0, FIGHT_PROXIMITY_FRAC_OF_SCALE * scale_px))
    return new_prox, {
        "tuned": True, "scale_px": scale_px,
        "formula": f"{FIGHT_PROXIMITY_FRAC_OF_SCALE} * median normal person torso "
                   "length (px) -- 'close enough to grapple', scaled by this "
                   "camera's own pixel-per-person geometry instead of a fixed "
                   "global pixel constant",
        "new_value": new_prox, "fixed_value": fixed,
    }


# --------------------------------------------------------------------------- #
#  ABANDONED                                                                 #
# --------------------------------------------------------------------------- #
def _tune_abandoned(clips, base_cfg, scale_px):
    ab = base_cfg.get("abandoned", {})
    fixed_unatt = float(ab.get("unattended_seconds", 12))
    fixed_orad = float(ab.get("owner_radius", 150))
    fixed_atol = float(ab.get("movement_tolerance", 40))

    n_obj = sum(1 for c in clips for fr in c.frames for d in fr
                if not d.get("is_person", True))
    n_frames = sum(c.n_frames for c in clips)

    if scale_px is None or n_obj < MIN_OBJECT_DETECTIONS_FOR_ABANDONED:
        return fixed_unatt, fixed_orad, fixed_atol, {
            "tuned": False,
            "reason": (f"only {n_obj} object (non-person) detections found across "
                       f"{n_frames} normal frames -- need >= "
                       f"{MIN_OBJECT_DETECTIONS_FOR_ABANDONED} to derive owner-"
                       "proximity / unattended-duration statistics. ABANDONED's "
                       "thresholds fall back to the fixed config values; this is "
                       "a data-availability limit (this site's/dataset's normal "
                       "footage never shows a tracked object), not a modelling "
                       "choice -- tuning is skipped rather than faked."),
            "n_object_detections": n_obj, "n_frames": n_frames,
        }

    new_atol = float(max(3.0, ABANDONED_TOL_FRAC_OF_SCALE * scale_px))

    # owner_radius: distance from each object to its NEAREST person, in every
    # normal frame where both an object and at least one person are present.
    dists = []
    for c in clips:
        for fr in c.frames:
            persons = [d for d in fr if d.get("is_person", True)]
            objs = [d for d in fr if not d.get("is_person", True)]
            if not objs or not persons:
                continue
            pc = [_centroid(p["bbox"]) for p in persons]
            for o in objs:
                ox, oy = _centroid(o["bbox"])
                dists.append(min(math.hypot(ox - px, oy - py) for px, py in pc))

    if len(dists) < MIN_OBJECT_PERSON_COOCCURRENCE_FRAMES:
        return fixed_unatt, fixed_orad, new_atol, {
            "tuned": False, "movement_tolerance_tuned": True,
            "movement_tolerance_new": new_atol,
            "movement_tolerance_formula":
                f"{ABANDONED_TOL_FRAC_OF_SCALE} * median normal person torso length (px)",
            "reason": (f"only {len(dists)} normal frames had both an object and a "
                       f"person -- need >= {MIN_OBJECT_PERSON_COOCCURRENCE_FRAMES} to "
                       "estimate a normal owner-proximity distribution"),
            "n_object_detections": n_obj,
        }

    dists = np.asarray(dists, dtype=float)
    p90 = float(np.percentile(dists, ABANDONED_OWNER_RADIUS_PERCENTILE))
    new_orad = float(max(10.0, p90 * ABANDONED_OWNER_RADIUS_MARGIN))

    # Probe pass for unattended_seconds, same pattern as loitering's dwell
    # probe: replay through CueBank with unattended_seconds set high enough
    # it never fires, owner_radius/movement_tolerance set to the just-derived
    # values, and read off the longest continuous "unattended so far" streak
    # per object track.
    probe_cfg = {**base_cfg, "abandoned": {**ab, "unattended_seconds": 1.0e9,
                                           "owner_radius": new_orad,
                                           "movement_tolerance": new_atol}}
    track_max = {}
    for c in clips:
        cb = CueBank(probe_cfg, fps=c.fps)
        for dets in c.frames:
            cb.step(dets)
            for tid, tr in cb._abandon.items():
                if tr["unattended_since"] is None:
                    continue
                d = cb.t - tr["unattended_since"]
                if d > track_max.get(tid, 0.0):
                    track_max[tid] = d

    info = {
        "tuned": True, "n_object_detections": n_obj,
        "n_owner_proximity_samples": len(dists),
        "movement_tolerance_new": new_atol,
        "movement_tolerance_formula":
            f"{ABANDONED_TOL_FRAC_OF_SCALE} * median normal person torso length (px)",
        f"p{ABANDONED_OWNER_RADIUS_PERCENTILE}_nearest_person_dist": p90,
        "owner_radius_new": new_orad,
        "owner_radius_formula":
            f"{ABANDONED_OWNER_RADIUS_MARGIN} * P{ABANDONED_OWNER_RADIUS_PERCENTILE}"
            "(distance from object to nearest person, normal frames with an "
            "object present)",
        "fixed_unattended_seconds": fixed_unatt, "fixed_owner_radius": fixed_orad,
        "fixed_movement_tolerance": fixed_atol,
    }

    if len(track_max) < MIN_OBJECT_TRACKS_FOR_UNATTENDED:
        info["unattended_seconds_tuned"] = False
        info["unattended_seconds_reason"] = (
            f"only {len(track_max)} object tracks were ever unattended in normal "
            f"footage -- need >= {MIN_OBJECT_TRACKS_FOR_UNATTENDED} to estimate a "
            "duration distribution")
        return fixed_unatt, new_orad, new_atol, info

    vals = np.asarray(list(track_max.values()), dtype=float)
    p90u = float(np.percentile(vals, ABANDONED_UNATTENDED_PERCENTILE))
    new_unatt = max(2.0, p90u * ABANDONED_UNATTENDED_MARGIN)
    info["unattended_seconds_tuned"] = True
    info["n_object_tracks"] = len(track_max)
    info[f"p{ABANDONED_UNATTENDED_PERCENTILE}_track_max_unattended_s"] = p90u
    info["unattended_seconds_formula"] = (
        f"{ABANDONED_UNATTENDED_MARGIN} * P{ABANDONED_UNATTENDED_PERCENTILE}"
        "(per-object longest continuous unattended streak in normal footage)")
    return new_unatt, new_orad, new_atol, info


# --------------------------------------------------------------------------- #
#  orchestrator                                                              #
# --------------------------------------------------------------------------- #
def tune_cue_thresholds(normal_clips, base_cfg: dict) -> dict:
    """
    normal_clips : list[calm.datasets.Clip], CONFIRMED NORMAL (empty
                   gt_intervals) -- e.g. the normal-labelled subset of
                   calm.harness.split_calib_test(...)'s calib split, or a
                   live warm-up buffer collected the same way M4's
                   auto_calibrate already does. No clip here may carry an
                   anomaly label or interval; this function never reads
                   gt_intervals, but callers must not pass it any clip that
                   hasn't already been confirmed normal, or the "no anomaly
                   labels needed" deployability claim stops being true.
    base_cfg     : the full config dict (as loaded from config.yaml).

    Returns a NEW cfg dict: a deep copy of base_cfg with
    crowd.threshold / loitering.dwell_seconds / loitering.movement_tolerance /
    fight.proximity / abandoned.unattended_seconds / abandoned.owner_radius /
    abandoned.movement_tolerance replaced by scene-adaptive values wherever a
    normal-footage statistic honestly supports it (see the `_tune_*`
    functions), and left at the original fixed value otherwise. config.yaml
    itself is never modified -- this is purely a new, opt-in cfg dict the
    caller chooses to use.

    A diagnostics block is attached at cfg["scene_autotune"]: which
    parameters were tuned vs. left fixed and why, the exact formulas, sample
    sizes, and old-vs-new values, plus cfg["scene_autotune"]["left_fixed"]
    listing every parameter this module does not attempt to tune at all
    (agitation_threshold and friends -- see LEFT_FIXED_REASONS) with the
    reason. This key is additive; nothing in calm/cues.py or calm/harness.py
    reads it, so it never affects scoring.
    """
    cfg = copy.deepcopy(base_cfg)
    n_frames = sum(c.n_frames for c in normal_clips)
    scale_px = _person_scale_px(normal_clips)

    report = {
        "n_normal_clips": len(normal_clips),
        "n_normal_frames": n_frames,
        "person_scale_px": scale_px,
        "person_scale_formula": "median torso length (px) over every person "
                                "detection in the confirmed-normal clips "
                                "passed in (reuses cues._torso_len)",
    }

    new_crowd, crowd_info = _tune_crowd(normal_clips, base_cfg)
    cfg.setdefault("crowd", {})["threshold"] = new_crowd
    report["crowd"] = crowd_info

    new_dwell, new_ltol, loiter_info = _tune_loitering(normal_clips, base_cfg, scale_px)
    cfg.setdefault("loitering", {})
    cfg["loitering"]["dwell_seconds"] = new_dwell
    cfg["loitering"]["movement_tolerance"] = new_ltol
    report["loitering"] = loiter_info

    new_prox, fight_info = _tune_fight_proximity(base_cfg, scale_px)
    cfg.setdefault("fight", {})["proximity"] = new_prox
    report["fight"] = fight_info

    new_unatt, new_orad, new_atol, aband_info = _tune_abandoned(
        normal_clips, base_cfg, scale_px)
    cfg.setdefault("abandoned", {})
    cfg["abandoned"]["unattended_seconds"] = new_unatt
    cfg["abandoned"]["owner_radius"] = new_orad
    cfg["abandoned"]["movement_tolerance"] = new_atol
    report["abandoned"] = aband_info

    report["left_fixed"] = dict(LEFT_FIXED_REASONS)

    cfg["scene_autotune"] = report
    return cfg


if __name__ == "__main__":
    # Tiny smoke test on the synthetic dataset (no download needed): tunes
    # against the synthetic normal clips and prints what changed. This is
    # NOT the offline validation -- see calm/run_scene_autotune_eval.py for
    # that (the real 5-benchmark, calib/test-separated comparison).
    import yaml
    from . import datasets as D

    with open("config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    clips = D.synthetic()
    normal = [c for c in clips if not c.gt_intervals]
    tuned = tune_cue_thresholds(normal, base_cfg)
    import json
    print(json.dumps(tuned["scene_autotune"], indent=2, default=float))
