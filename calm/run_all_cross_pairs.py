"""
run_all_cross_pairs.py  --  regenerate the FULL cross-dataset generalisation
matrix (all 5 x 4 = 20 fit/test pairs) with one clearly-documented,
consistent method, replacing the 12 original pairs whose exact generating
script was lost.

Method (same for every pair, no exceptions, no per-pair tuning):
  - load each dataset's real generic-JSON pose file ONCE
  - for the FIT dataset: take its own clips (every clip in the released
    pose files here already carries split=="test" -- none of the five
    ship a separate held-out calibration split in the file itself),
    relabel them split="calib"
  - for the TEST dataset: take its own clips, split="test"
  - hand the combined list to the *unmodified* calm.harness.run() exactly
    as for every in-domain run in this paper -- run()'s own fallback logic
    (present since early in this project, unchanged) carves a genuinely
    class-balanced calibration subset from whichever side lacks an
    all-normal clip, which for Avenue/CHAD/UBnormal/NWPU/ShanghaiTech is
    every one of them (no released pose-VAD split here ships a clip with
    zero anomalous frames) -- this is real, existing, already-used harness
    behaviour, not new logic invented for this table.

Prints a running summary and writes results/calm_report_<A>__to__<B>.json
for every pair via the same harness code path as every other real number
in this paper. Continues past a single pair's failure rather than losing
the other 19 (same resilience pattern used throughout this project).
"""
from __future__ import annotations

import dataclasses
import time
import traceback

import yaml

from calm.datasets import load_generic_json
from calm.harness import run

POSE_PATHS = {
    "shanghaitech": "data/pose/shanghaitech.json",
    "ubnormal": "data/pose/ubnormal.json",
    "chad": "data/pose/chad.json",
    "avenue": "data/pose/avenue.json",
    "nwpucampus": "nwpu results/nwpucampus.json",
}


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading all 5 real pose files once (reused across all 20 pairs)...")
    loaded = {}
    for name, path in POSE_PATHS.items():
        t0 = time.time()
        loaded[name] = load_generic_json(path)
        print(f"  {name}: {len(loaded[name])} clips  ({time.time()-t0:.1f}s)")

    names = list(POSE_PATHS)
    pairs = [(a, b) for a in names for b in names if a != b]
    print(f"\n{len(pairs)} pairs to run.\n")

    results = {}
    for i, (fit_name, test_name) in enumerate(pairs):
        tag = f"{fit_name}__to__{test_name}"
        print(f"\n{'='*70}\n[{i+1}/{len(pairs)}] {tag}\n{'='*70}")
        try:
            fit_clips = [dataclasses.replace(c, split="calib") for c in loaded[fit_name]]
            test_clips = [dataclasses.replace(c, split="test") for c in loaded[test_name]]
            report = run(fit_clips + test_clips, cfg, tag=tag, outdir="results")
            calm_stream = next(s for s in report["streams"] if s["label"].startswith("CALM-VAD"))
            results[tag] = calm_stream["event"]["f1_avg"]
            print(f"  -> event F1_avg = {results[tag]:.4f}")
        except Exception as e:
            print(f"  !! FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[tag] = None

    print(f"\n{'='*70}\nSUMMARY  ({sum(1 for v in results.values() if v is not None)}/{len(pairs)} succeeded)\n{'='*70}")
    for tag, f1 in results.items():
        print(f"  {tag:35s} {f1}")


if __name__ == "__main__":
    main()
