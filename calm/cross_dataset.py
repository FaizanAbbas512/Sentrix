"""
cross_dataset.py  --  fit M3/M4 on one benchmark's clips, evaluate on another's

Rebuilds the cross-dataset harness invocation used to produce
paper/tables/cross_dataset.tex's 12 original pairs (that ad-hoc script was
not preserved from earlier in the project). Mechanically simple: load both
generic-JSON pose files, force every "fit" clip's split to "calib" and every
"test" clip's split to "test", concatenate, and hand the combined list to
the existing calm.harness.run() -- it already derives the calib/test split
purely from each Clip's .split attribute, so no changes to harness.py are
needed.

    python -m calm.cross_dataset --fit data/pose/nwpucampus.json \
        --test data/pose/shanghaitech.json \
        --fit-name nwpucampus --test-name shanghaitech
"""
from __future__ import annotations

import argparse
import dataclasses

import yaml

from .datasets import load_generic_json
from .harness import run


def cross_run(fit_path, test_path, fit_name, test_name, cfg, outdir="results"):
    """Matches paper/tables/cross_dataset.tex's documented protocol: fit
    M3/M4 using the FIT dataset's own test-split clips (not its calib
    clips, and not the whole dataset) as the calibration source, evaluate
    on the TEST dataset's own test-split clips. Both datasets' generic-JSON
    files already carry a real calib/test split per clip (from when they
    were built), so this only needs to filter to split=="test" on each side
    and relabel for the combined run -- no new split logic invented here."""
    fit_clips = [c for c in load_generic_json(fit_path) if c.split == "test"]
    test_clips = [c for c in load_generic_json(test_path) if c.split == "test"]
    if not fit_clips:
        raise SystemExit(f"cross_run: no split=='test' clips found in {fit_path}")
    if not test_clips:
        raise SystemExit(f"cross_run: no split=='test' clips found in {test_path}")
    fit_clips = [dataclasses.replace(c, split="calib") for c in fit_clips]
    test_clips = [dataclasses.replace(c, split="test") for c in test_clips]
    tag = f"{fit_name}__to__{test_name}"
    return run(fit_clips + test_clips, cfg, tag=tag, outdir=outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="generic-JSON path: fit M3/M4 on this dataset")
    ap.add_argument("--test", required=True, help="generic-JSON path: evaluate on this dataset")
    ap.add_argument("--fit-name", required=True)
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cross_run(args.fit, args.test, args.fit_name, args.test_name, cfg, args.outdir)


if __name__ == "__main__":
    main()
