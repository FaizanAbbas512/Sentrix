"""
run_scene_autotune_eval.py  —  end-to-end, rerunnable validation of
calm/scene_autotune.py against the fixed config.yaml cue thresholds
===========================================================================

Answers one question honestly, per dataset: does replacing the fixed cue
thresholds (loitering.dwell_seconds, loitering.movement_tolerance,
fight.proximity, crowd.threshold, abandoned.unattended_seconds/owner_radius/
movement_tolerance) with calm.scene_autotune's normal-footage-derived values
measurably change frame AUC / event F1 / FAPH on a TEST split the tuner
never saw?

Methodology (identical for all 5 real benchmarks, no per-dataset tuning):

  1. Load the dataset's real generic-JSON pose file with
     calm.datasets.load_generic_json -- the same loader used to produce
     every existing results/calm_report_<dataset>.json baseline.
  2. Split into calib/test with calm.harness.split_calib_test, the exact
     function calm.harness.run() itself calls -- so "calib" and "test" here
     are IDENTICAL to the splits underlying the existing baseline reports.
  3. Take only the CALIB-split clips with an EMPTY gt_intervals list
     (confirmed-normal by construction). Feed ONLY those to
     calm.scene_autotune.tune_cue_thresholds(normal_calib, base_cfg). This
     function never reads gt_intervals or any label, and normal_calib
     clips carry none anyway -- but the point of doing the split first and
     filtering is to also guarantee no TEST-split clip, normal or
     anomalous, is ever passed to the tuner. Asserted explicitly below.
  4. Run calm.harness.run() TWICE on the SAME calib+test clip list in the
     SAME process: once with the untouched base_cfg (fixed thresholds,
     "fresh-fixed"), once with tuned_cfg (scene-adaptive thresholds). Both
     go through harness.run()'s own score_clip/CueBank path completely
     unmodified; the only difference between the two runs is the cue-
     threshold numbers under cfg["crowd"] / ["loitering"] / ["fight"] /
     ["abandoned"].

     IMPORTANT, found while building this: comparing the tuned run against
     the *committed* results/calm_report_<dataset>.json is NOT reliable --
     a determinism check (run harness.run() twice in one process on chad
     with byte-identical cfg) confirms harness.run() itself is perfectly
     reproducible (frame AUC identical to 16 decimal places across the two
     runs), but re-running the SAME fixed cfg fresh in this environment
     gives a small but real difference from the committed chad report
     (0.5896 fresh vs. 0.5889 committed) even though config_digest matches.
     That gap is environment/code drift since the committed report was
     generated (most likely an sklearn version difference in the isotonic
     regression -- config_digest does not even capture the sklearn
     version), NOT anything to do with cue tuning. So the only honest
     comparison is fresh-fixed vs. tuned, computed back-to-back in the same
     process/environment -- which is what this script does. The committed
     baseline is still loaded and reported alongside, purely so this drift
     is visible rather than hidden.
  5. Compare the "CALM-VAD (M1+M2+M3)" stream's frame_auc / event.f1_avg /
     event['f1@0.2'] / event['f1@0.5'] / faph@recall={0.7,0.8,0.9} between
     the fresh-fixed and tuned runs -- both computed in this same process,
     on the SAME test split, by the SAME code path, differing only in cue
     thresholds.

config.yaml is never written to. Nothing under paper/ is touched. Results
are written to results/calm_report_<dataset>_scene_autotuned.json/.txt and
results/calm_report_<dataset>_fresh_fixed.json/.txt (via the normal
harness.run() side effect) and a single comparison summary to
results/scene_autotune_eval.json / .txt.

Run:
    python -m calm.run_scene_autotune_eval
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from . import datasets as D
from .harness import run, split_calib_test
from .scene_autotune import tune_cue_thresholds

POSE_PATHS = {
    "shanghaitech": "data/pose/shanghaitech.json",
    "ubnormal": "data/pose/ubnormal.json",
    "chad": "data/pose/chad.json",
    "avenue": "data/pose/avenue.json",
    "nwpucampus": "nwpu results/nwpucampus.json",
}

# The in-domain baseline report for nwpucampus was written next to its pose
# file (in "nwpu results/") rather than in results/ like the other four --
# same layout this whole project already uses (see calm/m1_effect_analysis.py
# / calm/run_all_cross_pairs.py). Looked up explicitly so the comparison
# reads the SAME baseline file the paper's other numbers came from, instead
# of silently missing it because outdir defaults to "results/".
BASELINE_REPORT_PATHS = {
    "nwpucampus": "nwpu results/calm_report_nwpucampus.json",
}


def _calm_stream(report):
    return next(s for s in report["streams"] if s["label"].startswith("CALM-VAD"))


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def evaluate_one(name, path, base_cfg, outdir="results"):
    print(f"\n{'='*70}\n{name}  ({path})\n{'='*70}")
    clips = D.load_generic_json(path)
    clips = [c for c in clips if c.n_frames > 0]

    calib, test = split_calib_test(clips)
    normal_calib = [c for c in calib if not c.gt_intervals]
    test_names = {c.name for c in test}

    # Hard check: the tuner must never see a test-split clip, and every
    # clip handed to it must be confirmed-normal (no gt intervals).
    leaked = [c.name for c in normal_calib if c.name in test_names]
    mislabeled = [c.name for c in normal_calib if c.gt_intervals]
    assert not leaked, f"[LEAK] calib-normal clip(s) also in test split: {leaked}"
    assert not mislabeled, f"[LEAK] non-normal clip(s) passed to tuner: {mislabeled}"

    print(f"  clips: {len(clips)}  calib: {len(calib)} (normal-only used for "
          f"tuning: {len(normal_calib)})  test: {len(test)}")

    tuned_cfg = tune_cue_thresholds(normal_calib, base_cfg)
    tune_report = tuned_cfg["scene_autotune"]
    print("  scene_autotune diagnostics:")
    for cue in ("crowd", "loitering", "fight", "abandoned"):
        d = tune_report[cue]
        print(f"    {cue:<10} tuned={d.get('tuned')}"
              + (f"  reason={d['reason']}" if not d.get("tuned") and "reason" in d else ""))

    # Committed baseline: loaded and reported for transparency only (see the
    # module docstring's determinism-check note) -- NOT used as the primary
    # comparison, because it may have been generated in a different
    # environment (e.g. a different sklearn version) than this run.
    baseline_path = BASELINE_REPORT_PATHS.get(name, os.path.join(outdir, f"calm_report_{name}.json"))
    committed_stream = None
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            committed_stream = _calm_stream(json.load(f))
    else:
        print(f"  (no committed baseline at {baseline_path} -- skipping that comparison)")

    # The only apples-to-apples comparison: fixed and tuned, scored back to
    # back in THIS process on the SAME clip list.
    fixed_report = run(clips, base_cfg, tag=f"{name}_fresh_fixed", outdir=outdir)
    fixed_stream = _calm_stream(fixed_report)
    tuned_report = run(clips, tuned_cfg, tag=f"{name}_scene_autotuned", outdir=outdir)
    tuned_stream = _calm_stream(tuned_report)

    assert fixed_report["n_test"] == tuned_report["n_test"], (
        f"[run_scene_autotune_eval] test-split size differs between fresh-fixed "
        f"({fixed_report['n_test']}) and tuned ({tuned_report['n_test']}) runs "
        f"for {name} -- comparison would not be apples-to-apples")

    if committed_stream is not None:
        drift = tuned_stream["frame_auc"] - fixed_stream["frame_auc"]
        stale_drift = fixed_stream["frame_auc"] - committed_stream["frame_auc"]
        if abs(stale_drift) > 1e-9:
            print(f"  [environment drift] committed baseline frame_auc "
                  f"{committed_stream['frame_auc']:.4f} != fresh-fixed rerun "
                  f"{fixed_stream['frame_auc']:.4f} (same cfg) -- {abs(stale_drift):.4f} "
                  "gap is environment/code drift since the committed report was made, "
                  "not cue tuning; using fresh-fixed as the true baseline.")

    faph_keys = [k for k in fixed_stream if k.startswith("faph@")]
    row = {
        "dataset": name,
        "n_calib": tuned_report["n_calib"], "n_test": tuned_report["n_test"],
        "n_normal_calib_used_for_tuning": len(normal_calib),
        "frame_auc": {"fixed": fixed_stream["frame_auc"], "tuned": tuned_stream["frame_auc"],
                      "delta": tuned_stream["frame_auc"] - fixed_stream["frame_auc"],
                      "committed_baseline": committed_stream["frame_auc"] if committed_stream else None},
        "event_f1_avg": {"fixed": fixed_stream["event"]["f1_avg"],
                         "tuned": tuned_stream["event"]["f1_avg"],
                         "delta": tuned_stream["event"]["f1_avg"] - fixed_stream["event"]["f1_avg"],
                         "committed_baseline": committed_stream["event"]["f1_avg"] if committed_stream else None},
        "event_f1@0.2": {"fixed": fixed_stream["event"]["f1@0.2"],
                         "tuned": tuned_stream["event"]["f1@0.2"]},
        "event_f1@0.5": {"fixed": fixed_stream["event"]["f1@0.5"],
                         "tuned": tuned_stream["event"]["f1@0.5"]},
        "faph": {k: {"fixed": fixed_stream[k], "tuned": tuned_stream[k]} for k in faph_keys},
        "scene_autotune": tune_report,
    }

    print(f"  frame AUC        : fresh-fixed {fixed_stream['frame_auc']:.4f}  ->  "
          f"tuned {tuned_stream['frame_auc']:.4f}  (delta {row['frame_auc']['delta']:+.4f})")
    print(f"  event F1 avg     : fresh-fixed {fixed_stream['event']['f1_avg']:.4f}  ->  "
          f"tuned {tuned_stream['event']['f1_avg']:.4f}  (delta {row['event_f1_avg']['delta']:+.4f})")
    for k in faph_keys:
        b, t = fixed_stream[k], tuned_stream[k]
        print(f"  {k:<16} : fresh-fixed {_fmt(b)}  ->  tuned {_fmt(t)}")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of dataset names to run (default: all 5)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    names = args.only or list(POSE_PATHS)
    rows = []
    failures = []
    for name in names:
        path = POSE_PATHS[name]
        try:
            rows.append(evaluate_one(name, path, base_cfg, outdir=args.outdir))
        except Exception as e:
            print(f"  !! FAILED on {name}: {type(e).__name__}: {e}")
            failures.append((name, str(e)))

    print(f"\n{'='*70}\nSUMMARY  ({len(rows)}/{len(names)} datasets evaluated)\n{'='*70}")
    print("(fixed = fresh rerun of the unmodified config.yaml thresholds in THIS "
          "process, not the possibly environment-drifted committed baseline -- see "
          "module docstring)")
    header = f"{'dataset':<14}{'AUC fixed':>11}{'AUC tuned':>11}{'d_AUC':>9}" \
             f"{'F1 fixed':>10}{'F1 tuned':>10}{'d_F1':>8}"
    print(header)
    for r in rows:
        print(f"{r['dataset']:<14}"
              f"{r['frame_auc']['fixed']:>11.4f}{r['frame_auc']['tuned']:>11.4f}"
              f"{r['frame_auc']['delta']:>+9.4f}"
              f"{r['event_f1_avg']['fixed']:>10.4f}{r['event_f1_avg']['tuned']:>10.4f}"
              f"{r['event_f1_avg']['delta']:>+8.4f}")
    for name, err in failures:
        print(f"{name:<14} FAILED: {err}")

    os.makedirs(args.outdir, exist_ok=True)
    summary = {"rows": rows, "failures": failures}
    with open(os.path.join(args.outdir, "scene_autotune_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(args.outdir, "scene_autotune_eval.txt"), "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(f"{r['dataset']:<14}"
                   f"{r['frame_auc']['fixed']:>11.4f}{r['frame_auc']['tuned']:>11.4f}"
                   f"{r['frame_auc']['delta']:>+9.4f}"
                   f"{r['event_f1_avg']['fixed']:>10.4f}{r['event_f1_avg']['tuned']:>10.4f}"
                   f"{r['event_f1_avg']['delta']:>+8.4f}\n")
    print(f"\n  -> {args.outdir}/scene_autotune_eval.json / .txt")


if __name__ == "__main__":
    main()
