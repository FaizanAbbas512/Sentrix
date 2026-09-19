"""
nwpu_gt.py  --  explode NWPU-Campus's single NWPU_Campus_gt.npz into the
per-clip .npy frame-mask files calm.extract_poses already knows how to read
("*.npy frame mask, 1 = anomalous (ShanghaiTech / NWPU style)").

The official release ships one npz keyed by clip name (e.g. "D001_01") with
a 0/1-per-frame array (0=normal, 1=abnormal -- confirmed by the dataset's
own read_gt_file.py, same convention as ShanghaiTech, no polarity surprise
like UBnormal). This just re-shapes that into one .npy per key so the
existing, already-tested extract_poses.py --gt loader works unmodified.

    python -m calm.nwpu_gt --npz data/raw/nwpu/groundtruth/NWPU_Campus_gt.npz \
        --out data/raw/nwpu/gt_npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def explode(npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(npz_path)
    n = 0
    for key in d.keys():
        arr = d[key].astype(int).ravel()
        np.save(os.path.join(out_dir, f"{key}.npy"), arr)
        n += 1
    print(f"  -> {n} per-clip .npy files written to {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    explode(args.npz, args.out)


if __name__ == "__main__":
    main()
