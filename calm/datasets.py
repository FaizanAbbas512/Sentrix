"""
datasets.py  —  skeleton streams in, (detections-per-frame, gt-intervals) out
===========================================================================

The harness consumes a `Clip`:

    Clip(name, fps, n_frames, frames, gt_intervals, split)
      frames        : list[ list[detection_dict] ]   one list per frame
      detection_dict: {track_id, keypoints (17,3) | None, bbox, is_person, label}
      gt_intervals  : list[(start_frame, end_frame)]   ground-truth anomalies

Loaders:
  * synthetic(...)          — no download; scripted normal + anomaly clips.
                              Enough to exercise the whole M1..M4 chain and the
                              metrics, and to smoke-test before touching real data.
  * load_shanghaitech_hr()  — HR-ShanghaiTech / Alphapose-style per-frame JSON.
  * load_generic_json()     — a documented flat schema you can convert any
                              pose-VAD benchmark into (see README_CALM.md).

The real-benchmark loaders are written against the common
"{video: {frame: [ {keypoints:[x,y,c,...], bbox:[...], idx:int} ] }}" layout used
by STG-NF / MoCoDAD releases. Point them at your extracted-pose folder; if a
benchmark ships a different layout, convert it to `generic_json` once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import glob
import pickle
import numpy as np


@dataclass
class Clip:
    name: str
    fps: float
    n_frames: int
    frames: list                       # list[list[dict]]
    gt_intervals: list                 # list[(start, end)]
    split: str = "test"                # "train" | "calib" | "test"
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  synthetic                                                                 #
# --------------------------------------------------------------------------- #
_COCO_NEUTRAL = np.array([
    [0, -60], [3, -63], [-3, -63], [7, -60], [-7, -60],      # nose, eyes, ears
    [14, -40], [-14, -40], [20, -12], [-20, -12], [22, 12], [-22, 12],  # sh/elb/wri
    [10, 0], [-10, 0], [11, 40], [-11, 40], [12, 78], [-12, 78],        # hip/knee/ankle
], dtype=float)


def _person(cx, cy, scale=1.0, pose="upright", agitate=0.0, conf=0.9, rng=None):
    rng = rng or np.random
    k = _COCO_NEUTRAL.copy() * scale
    if pose == "fallen":                       # rotate ~90 deg -> torso horizontal
        k = k[:, ::-1].copy()
        k[:, 1] *= -1
    k[:, 0] += cx
    k[:, 1] += cy + 60 * scale
    if agitate > 0:                            # jitter the limbs (wrists/ankles)
        for i in (9, 10, 15, 16):
            k[i] += rng.normal(0, agitate * 25, size=2)
    kp = np.hstack([k, np.full((17, 1), conf)])
    xs, ys = kp[:, 0], kp[:, 1]
    bbox = (float(xs.min() - 8), float(ys.min() - 8), float(xs.max() + 8), float(ys.max() + 8))
    return kp, bbox


def _corrupt_person(cx, cy, rng):
    """A broken skeleton: low keypoint confidence + torso reading horizontal.
    This is what occlusion / low-res / a bad detection looks like to the cues,
    and it false-fires the FALL posture check unless M1 discounts it."""
    k = _COCO_NEUTRAL.copy()
    k = k[:, ::-1].copy(); k[:, 1] *= -1          # torso -> horizontal
    k[:, 0] += cx + rng.normal(0, 6, size=17)
    k[:, 1] += cy + 60 + rng.normal(0, 6, size=17)
    conf = np.full((17, 1), rng.uniform(0.12, 0.28))   # below KP_CONF mostly
    kp = np.hstack([k, conf])
    xs, ys = kp[:, 0], kp[:, 1]
    bbox = (float(xs.min() - 8), float(ys.min() - 8),
            float(xs.max() + 8), float(ys.max() + 8))
    return kp, bbox


def _mk_frame(persons, objects=None):
    dets = []
    for tid, (kp, bbox) in persons.items():
        dets.append(dict(track_id=tid, keypoints=kp, bbox=bbox,
                         is_person=True, label="person"))
    for tid, bbox in (objects or {}).items():
        dets.append(dict(track_id=tid, keypoints=None, bbox=bbox,
                         is_person=False, label="backpack"))
    return dets


def synthetic(n_normal_clips=10, n_anom_clips=6, fps=20, seconds=75, seed=0,
              degrade=False):
    """
    Returns list[Clip]. Normal clips: people walking through. Anomaly clips:
    one scripted event (fall / fight / loiter / abandoned) in the middle third.

    degrade=True lowers keypoint confidence and adds ID-switch noise on a subset
    of frames — use it to check M1 actually rescues the false-alarm rate.
    """
    rng = np.random.default_rng(seed)
    N = int(fps * seconds)
    clips = []

    def walk_path(t, lane):
        return (40 + 900 * (t / N) % 1.0 * 0 + 60 + t * 3.0) % 620 + 20, 250 + lane * 40

    for c in range(n_normal_clips):
        frames = []
        n_people = rng.integers(1, 4)
        # On degraded runs, half the normal test clips get a sustained
        # "corrupted skeleton" episode: low keypoint confidence + a torso that
        # reads as horizontal. Without M1 this false-fires the FALL cue (the
        # clip is normal, so every such alarm is a false alarm). With M1 the
        # low confidence drives r -> 0 and M2 discounts the cue away.
        corrupt = (degrade and c % 2 == 0)
        ca, cb = int(N * 0.45), int(N * 0.75)
        for t in range(N):
            persons = {}
            for pid in range(n_people):
                cx = (80 + t * (2.0 + 0.5 * pid) + pid * 130) % 600 + 20
                cy = 240 + pid * 35
                if corrupt and pid == 0 and ca <= t <= cb:
                    persons[pid] = _corrupt_person(cx, cy, rng)
                else:
                    conf = 0.9
                    if degrade and rng.random() < 0.15:
                        conf = rng.uniform(0.15, 0.45)
                    persons[pid] = _person(cx, cy, scale=rng.uniform(0.85, 1.1),
                                           agitate=rng.uniform(0, 0.12),
                                           conf=conf, rng=rng)
            frames.append(_mk_frame(persons))
        split = "calib" if c < max(2, n_normal_clips // 3) else "test"
        clips.append(Clip(f"normal_{c:02d}", fps, N, frames, [], split=split))

    events = ["fall", "fight", "loiter", "abandoned", "fall", "fight"]
    for c in range(n_anom_clips):
        ev = events[c % len(events)]
        frames = []
        a, b = int(N * 0.38), int(N * 0.62)
        for t in range(N):
            persons = {}
            objs = {}
            inside = a <= t <= b
            # background walkers
            for pid in range(rng.integers(0, 2) + 1):
                cx = (100 + t * 2.5 + pid * 200) % 600 + 20
                persons[10 + pid] = _person(cx, 300 + pid * 20, rng=rng,
                                            agitate=rng.uniform(0, 0.1))
            if ev == "fall":
                persons[1] = _person(320, 250, pose="fallen" if inside else "upright",
                                     conf=0.9, rng=rng)
            elif ev == "fight":
                ag = 1.4 if inside else 0.05
                persons[1] = _person(300, 250, agitate=ag, rng=rng)
                persons[2] = _person(345, 252, agitate=ag, rng=rng)
            elif ev == "loiter":
                persons[1] = _person(300, 250, agitate=0.05, rng=rng)  # stays put whole clip
            elif ev == "abandoned":
                persons[1] = _person(300 if t < a else 520, 250, rng=rng)  # owner leaves
                objs[99] = (295.0, 300.0, 330.0, 335.0)
            frames.append(_mk_frame(persons, objs))
        # loiter/abandoned mature over time; label the window from when the cue
        # strength ramp completes (system *should* be confident) to the end, so
        # score and label agree -- otherwise the ramp-up frames are a built-in
        # calibration mismatch. 15 s dwell / 12 s unattended = config defaults;
        # the strength ramp runs to 2.5x that.
        if ev == "loiter":
            gt = [(min(int(15 * 2.5 * fps), N - 2), N - 1)]
        elif ev == "abandoned":
            gt = [(min(a + int(12 * 2.5 * fps), N - 2), N - 1)]
        else:
            gt = [(a, b)]
        clips.append(Clip(f"anom_{ev}_{c:02d}", fps, N, frames, gt, split="test"))

    return clips


# --------------------------------------------------------------------------- #
#  real-benchmark loaders                                                    #
# --------------------------------------------------------------------------- #
def _norm_kp(raw, n=17):
    """Accept [x1,y1,c1, x2,y2,c2, ...] or [[x,y,c], ...] -> (17,3) float array."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[0] < n:
        pad = np.zeros((n - arr.shape[0], 3))
        arr = np.vstack([arr, pad])
    return arr[:n, :3]


def _bbox_from_kp(kp):
    v = kp[kp[:, 2] > 0.05]
    if len(v) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    return (float(v[:, 0].min()), float(v[:, 1].min()),
            float(v[:, 0].max()), float(v[:, 1].max()))


def load_generic_json(path, fps=20, split="test"):
    """
    Flat schema (convert any benchmark to this once):

    {
      "fps": 20,
      "clips": [
        {"name": "01_014", "n_frames": 900, "split": "test",
         "gt": [[120, 240]],
         "frames": [
            [ {"id": 3, "keypoints": [x,y,c, ...17...], "bbox": [x1,y1,x2,y2],
               "is_person": true, "label": "person"} ],
            ...
         ]}
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = float(data.get("fps", fps))
    clips = []
    for cd in data["clips"]:
        frames = []
        for fr in cd["frames"]:
            dets = []
            for d in fr:
                kp = d.get("keypoints")
                kp = _norm_kp(kp) if kp is not None else None
                bbox = tuple(d["bbox"]) if d.get("bbox") else \
                    (_bbox_from_kp(kp) if kp is not None else (0, 0, 1, 1))
                dets.append(dict(track_id=int(d.get("id", d.get("track_id", -1))),
                                 keypoints=kp, bbox=bbox,
                                 is_person=bool(d.get("is_person", True)),
                                 label=d.get("label", "person")))
            frames.append(dets)
        clips.append(Clip(cd["name"], fps, int(cd.get("n_frames", len(frames))),
                          frames, [tuple(g) for g in cd.get("gt", [])],
                          split=cd.get("split", split)))
    return clips


def _kp_idx(pp):
    """AlphaPose stores person id as 'idx' which may be int, str, or [id]."""
    v = pp.get("idx", pp.get("track_id", pp.get("id", -1)))
    if isinstance(v, (list, tuple)):
        v = v[0] if v else -1
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


def load_shanghaitech_hr(pose_dir, gt_dir=None, fps=24, split="test"):
    """
    HR-ShanghaiTech / HR-Avenue style (STG-NF, GEPC, MoCoDAD releases):
    one JSON per clip named like '01_0014.json'. Two layouts are accepted:
      * dict  {frame_idx(str): [ {"keypoints":[...51...], "idx":id}, ... ]}
      * list  [ {"image_id":"...", "keypoints":[...], "idx":id}, ... ]  (raw AlphaPose)
    Ground truth: a per-clip .npy frame mask (1 = anomalous) in gt_dir, same stem
    (also tries '<stem>.npy' inside gt_dir, or a single 'gt.npy'/'frame_labels.npy').
    """
    clips = []
    jsons = sorted(glob.glob(os.path.join(pose_dir, "*.json")))
    if not jsons:
        raise SystemExit(f"no *.json under {pose_dir}")

    # optional single combined gt file (dict clip->mask) as a fallback
    combined_gt = {}
    if gt_dir:
        for cand in ("gt.npy", "frame_labels.npy", "test_frame_mask.npy"):
            cp = os.path.join(gt_dir, cand)
            if os.path.exists(cp):
                try:
                    combined_gt = dict(np.load(cp, allow_pickle=True).item())
                except Exception:
                    pass

    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        with open(jp, "r", encoding="utf-8") as f:
            raw = json.load(f)

        per_frame = {}
        if isinstance(raw, dict):
            for k, ppl in raw.items():
                per_frame.setdefault(_frame_no(k), []).extend(ppl)
        else:                                    # list of AlphaPose entries
            for e in raw:
                per_frame.setdefault(_frame_no(e.get("image_id", e.get("frame", 0))),
                                     []).append(e)

        n = (max(per_frame) + 1) if per_frame else 0
        frames = [[] for _ in range(n)]
        for fi, people in per_frame.items():
            for pp in people:
                kp = _norm_kp(pp["keypoints"])
                frames[fi].append(dict(track_id=_kp_idx(pp), keypoints=kp,
                                       bbox=_bbox_from_kp(kp), is_person=True,
                                       label="person"))
        gt = []
        if stem in combined_gt:
            gt = _mask_to_intervals(np.asarray(combined_gt[stem]).astype(int).ravel())
        elif gt_dir:
            for cand in (stem + ".npy", stem.replace("_", "") + ".npy"):
                gp = os.path.join(gt_dir, cand)
                if os.path.exists(gp):
                    gt = _mask_to_intervals(np.load(gp).astype(int).ravel())
                    break
        clips.append(Clip(stem, float(fps), n, frames, gt, split=split))

    got = sum(len(c.gt_intervals) for c in clips)
    print(f"[load_shanghaitech_hr] {len(clips)} clips, {got} gt events"
          + ("  (no GT found -- check gt_dir)" if got == 0 else ""))
    return clips


def _frame_no(k):
    """'0014' / '01_0014.jpg' / 14 / '14.png' -> 14 (last integer run)."""
    import re
    s = str(k)
    m = re.findall(r"\d+", s)
    return int(m[-1]) if m else 0


def _mask_to_intervals(mask):
    intervals = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            intervals.append((i, j - 1))
            i = j
        else:
            i += 1
    return intervals


# ========================================================================= #
#  Real benchmark loaders — the community pose-VAD releases                  #
#  (STG-NF: GEPC-style JSON ; MoCoDAD: Morais-style trajectory CSV)          #
# ========================================================================= #
def _parse_pose_json(raw):
    """
    Any of these -> {frame_idx: [ {track_id, kp(17,3)} ]}:
      * person-major nested  {pid: {fid: {"keypoints":[51], "scores":...}}}   (GEPC / STG-NF)
      * frame-major          {fid: [ {"keypoints":[...], "idx":id}, ... ]}
      * flat AlphaPose list  [ {"image_id":..., "keypoints":[...], "idx":id}, ... ]
    """
    per_frame = {}
    if isinstance(raw, list):
        for e in raw:
            per_frame.setdefault(_frame_no(e.get("image_id", e.get("frame", 0))),
                                 []).append((_kp_idx(e), _norm_kp(e["keypoints"])))
    elif isinstance(raw, dict) and raw:
        v0 = next(iter(raw.values()))
        person_major = isinstance(v0, dict) and v0 and \
            isinstance(next(iter(v0.values())), dict)
        if person_major:
            for pid, fdict in raw.items():
                tid = _kp_idx({"idx": pid})
                for fid, fr in fdict.items():
                    per_frame.setdefault(_frame_no(fid), []).append(
                        (tid, _norm_kp(fr["keypoints"])))
        else:                                       # frame-major
            for fid, ppl in raw.items():
                for pp in ppl:
                    per_frame.setdefault(_frame_no(fid), []).append(
                        (_kp_idx(pp), _norm_kp(pp["keypoints"])))
    return per_frame


def _clips_from_perframe(name, per_frame, gt, fps, split):
    n = (max(per_frame) + 1) if per_frame else 0
    frames = [[] for _ in range(n)]
    for fi, ppl in per_frame.items():
        for tid, kp in ppl:
            frames[fi].append(dict(track_id=tid, keypoints=kp,
                                   bbox=_bbox_from_kp(kp), is_person=True,
                                   label="person"))
    return Clip(name, float(fps), n, frames, gt, split=split)


def _find_gt(gt_dir, stem):
    if not gt_dir:
        return []
    for cand in (stem + ".npy", stem.replace("_", "") + ".npy",
                 stem.split("_")[0] + "_" + stem.split("_")[-1] + ".npy"):
        p = os.path.join(gt_dir, cand)
        if os.path.exists(p):
            return _mask_to_intervals(np.load(p).astype(int).ravel())
    return []


def load_gepc_json(pose_dir, gt_dir=None, fps=24, split="test"):
    """STG-NF / GEPC layout: pose_dir/*.json (person-major), gt_dir/<stem>.npy."""
    clips = []
    for jp in sorted(glob.glob(os.path.join(pose_dir, "*.json"))):
        base = os.path.splitext(os.path.basename(jp))[0]
        stem = "_".join(base.split("_")[:2])          # 01_0014_alphapose_... -> 01_0014
        with open(jp, "r", encoding="utf-8") as f:
            per_frame = _parse_pose_json(json.load(f))
        clips.append(_clips_from_perframe(stem, per_frame, _find_gt(gt_dir, stem),
                                          fps, split))
    _report("load_gepc_json", clips)
    return clips


def load_ubnormal_stgnf(pose_dir, gt_dir, fps=30, split="test"):
    """
    UBnormal as released in the STG-NF data bundle. NOT auto-detected by
    load_any -- called explicitly (--ubnormal-stgnf) because its ground-truth
    convention is the OPPOSITE of ShanghaiTech's and needs the correct stem
    rule, and getting either wrong would silently corrupt every number.

      pose_dir/<abnormal|normal>_scene_<N>_scenario<...>_alphapose_tracked_person.json
      gt_dir/<same stem>_tracks.txt   -- an .npy array (despite the .txt name)
                                          of PER-FRAME LABELS WHERE 1 = NORMAL,
                                          0 = ANOMALOUS (verified empirically:
                                          every 'normal_*' file is uniformly 1.0
                                          across all frames and all clips; every
                                          'abnormal_*' file is a 0/1 mix with a
                                          duration that varies clip to clip --
                                          the opposite polarity from ShanghaiTech's
                                          test_frame_mask, where 1 = anomalous).
    We invert (anomaly = 1 - label) so downstream code has one convention.
    """
    clips = []
    for jp in sorted(glob.glob(os.path.join(pose_dir, "*.json"))):
        base = os.path.splitext(os.path.basename(jp))[0]
        stem = base.replace("_alphapose_tracked_person", "")
        with open(jp, "r", encoding="utf-8") as f:
            per_frame = _parse_pose_json(json.load(f))

        gt = []
        gp = os.path.join(gt_dir, stem + "_tracks.txt")
        if os.path.exists(gp):
            try:
                raw = np.load(gp).astype(float).ravel()
                gt = _mask_to_intervals((1.0 - raw).astype(int))
            except Exception as e:
                print(f"[load_ubnormal_stgnf] could not read {gp}: {e}")

        clips.append(_clips_from_perframe(stem, per_frame, gt, fps, split))
    _report("load_ubnormal_stgnf", clips)
    return clips


def load_ubnormal_stgnf_auto(root, fps=30, split="test"):
    """
    Auto-detect UBnormal's pose_dir/gt_dir under `root` and load them
    through load_ubnormal_stgnf so the ground-truth inversion is always
    applied.

    Does NOT reuse load_any's generic GEPC/gt ranking: the real STG-NF
    Drive bundle ships ShanghaiTech and UBnormal side by side in the SAME
    zip, so "the GEPC dir with the most files" or "the .npy gt dir" can
    silently resolve to ShanghaiTech's own directories instead of
    UBnormal's (verified while wiring this: on the combined bundle, that
    naive ranking picked ShanghaiTech/pose/test and ShanghaiTech/gt/
    test_frame_mask outright -- 0 UBnormal ground truth, pos=0 everywhere,
    the same silent-corruption failure mode this function exists to avoid).
    Instead we identify UBnormal's own directories by their distinctive
    file-naming convention: pose files named
    "<abnormal|normal>_..._alphapose_tracked_person.json", and gt files
    named "<same stem>_tracks.txt" (an .npy array despite the .txt name,
    see load_ubnormal_stgnf) -- so an .npy-only gt scan never finds it either.
    """
    pose_dir = None
    for d in _all_dirs(root):
        names = [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.json"))]
        matches = sum(1 for n in names
                      if n.startswith(("abnormal_", "normal_"))
                      and "alphapose_tracked_person" in n)
        if matches >= 2:
            if pose_dir is None or matches > pose_dir[1]:
                pose_dir = (d, matches)
    if pose_dir is None:
        raise SystemExit(
            f"load_ubnormal_stgnf_auto: no UBnormal-shaped pose dir found under {root} "
            "(expected <abnormal|normal>_..._alphapose_tracked_person.json files)")
    pose_dir = pose_dir[0]

    gt_dir = None
    for d in _all_dirs(root):
        n_tracks = len(glob.glob(os.path.join(d, "*_tracks.txt")))
        if n_tracks >= 2:
            if gt_dir is None or n_tracks > gt_dir[1]:
                gt_dir = (d, n_tracks)
    if gt_dir is None:
        raise SystemExit(
            f"load_ubnormal_stgnf_auto: no UBnormal-shaped gt dir found under {root} "
            "(expected <stem>_tracks.txt files)")
    gt_dir = gt_dir[0]

    print(f"[load_ubnormal_stgnf_auto] pose_dir={os.path.relpath(pose_dir, root)}  "
          f"gt_dir={os.path.relpath(gt_dir, root)}  (inverted GT applied)")
    return load_ubnormal_stgnf(pose_dir, gt_dir, fps=fps, split=split)


def convert_avenue_mat_gt(mask_dir, out_dir):
    """
    CUHK Avenue ground truth ships as ground_truth_demo/testing_label_mask/
    '<N>_label.mat', each holding a MATLAB 'volLabel' cell array: one
    (H,W) uint8 pixel mask per frame. Convert to one <video-stem>.npy
    per-frame binary array (1=anomalous) per clip, named to match the video
    files ('01.avi' <-> '1_label.mat', note the video has a leading zero the
    mask filename does not).
    """
    import re
    from scipy.io import loadmat
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for mp in sorted(glob.glob(os.path.join(mask_dir, "*_label.mat"))):
        m = re.search(r"(\d+)_label", os.path.basename(mp))
        if not m:
            continue
        stem = f"{int(m.group(1)):02d}"
        a = loadmat(mp)["volLabel"]
        if a.dtype == object:
            mask = np.array([int(np.any(f)) for f in a.ravel()])
        else:
            arr = np.asarray(a)
            mask = ((arr.reshape(-1, arr.shape[-1]).sum(0) > 0).astype(int)
                    if arr.ndim >= 3 else (arr.ravel() > 0).astype(int))
        np.save(os.path.join(out_dir, stem + ".npy"), mask)
        n += 1
    print(f"[convert_avenue_mat_gt] {n} mask files -> {out_dir}")
    return n


def load_chad(root, fps=30, split="test", split_file="splits/test_split_1.txt"):
    """
    CHAD -- Charlotte Anomaly Dataset (TeCSAR-UNCC), the metadata release
    (poses ship natively, no video/GPU needed):
      root/annotations/<stem>.pkl       {frame: {person_id: [bbox_xywh(4,),
                                          keypoints(17,3) x,y,conf]}}  (verified
                                          directly against a real file)
      root/anomaly_labels/<stem>.npy    per-frame binary, 1 = anomalous
                                         (verified: normal-suffixed stems are
                                         all-zero, abnormal-suffixed are 0/1 mix
                                         -- same polarity as ShanghaiTech)
      root/splits/{train,test}_split_{1,2}.txt   two INDEPENDENT partitions
                                         (test_split_1's clips do not appear in
                                         train_split_1, but do overlap
                                         train_split_2 -- they are alternates,
                                         not to be merged). Default: split 1's
                                         official test list.
    fps=30: the paper states all four cameras are recorded at 30 fps
    (1920x1080 for cams 1-3, 1280x720 for cam 4; arXiv:2212.09258).
    Pass split_file=None to load every annotated clip instead.
    """
    ann_dir = os.path.join(root, "annotations")
    lab_dir = os.path.join(root, "anomaly_labels")
    if split_file:
        with open(os.path.join(root, split_file)) as f:
            stems = [ln.strip() for ln in f if ln.strip()]
    else:
        stems = [os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".pkl")]

    clips = []
    for stem in stems:
        pp = os.path.join(ann_dir, stem + ".pkl")
        if not os.path.exists(pp):
            continue
        with open(pp, "rb") as f:
            raw = pickle.load(f)
        n = (max(raw.keys()) + 1) if raw else 0
        frames = [[] for _ in range(n)]
        for fi, people in raw.items():
            for pid, (bbox, kp) in people.items():
                # real detector output: occluded/undetected joints and (rarely)
                # a whole-person bbox can come through as NaN. Force confidence
                # to 0 (not just zero the coordinate) on any row touched by NaN,
                # so downstream code -- which already treats low-confidence
                # keypoints as "not seen" -- correctly skips it, instead of a
                # NaN silently propagating into every later computation.
                kp = np.asarray(kp, float)
                bad = np.isnan(kp).any(axis=1)
                kp[bad] = 0.0
                bbox = np.nan_to_num(np.asarray(bbox, float), nan=0.0)
                x, y, w, h = [float(v) for v in bbox]
                frames[fi].append(dict(track_id=int(pid), keypoints=kp,
                                       bbox=(x, y, x + max(w, 0), y + max(h, 0)),
                                       is_person=True, label="person"))
        gt = []
        lp = os.path.join(lab_dir, stem + ".npy")
        if os.path.exists(lp):
            gt = _mask_to_intervals(np.load(lp).astype(int).ravel())
        clips.append(Clip(stem, float(fps), n, frames, gt, split=split))
    _report("load_chad", clips)
    return clips


def load_known_normal_pool(pose_dir, fps=30, split="calib", require_prefix="normal_"):
    """
    Load a pose folder that is normal *by construction* (e.g. UBnormal's
    'train' split, which STG-NF ships as 186 'normal_scene_*' clips with zero
    'abnormal_*' files) as confirmed-normal-stream hours -- gt is forced to
    [] regardless of any gt_dir, so there is no chance of cross-contamination
    from another split's ground truth. Use this to give M4 more normal hours
    to certify a budget against, on top of whatever the test/calib split
    already supplies.

    require_prefix guards against silently mixing in a non-normal clip if the
    folder's contents ever change; set to None to disable the check.
    """
    clips = []
    skipped = 0
    for jp in sorted(glob.glob(os.path.join(pose_dir, "*.json"))):
        base = os.path.splitext(os.path.basename(jp))[0]
        if require_prefix and not os.path.basename(jp).startswith(require_prefix):
            skipped += 1
            continue
        stem = base.replace("_alphapose_tracked_person", "")
        with open(jp, "r", encoding="utf-8") as f:
            per_frame = _parse_pose_json(json.load(f))
        clips.append(_clips_from_perframe(stem, per_frame, [], fps, split))
    if skipped:
        print(f"[load_known_normal_pool] skipped {skipped} file(s) not matching "
              f"prefix {require_prefix!r} (not assumed normal)")
    _report("load_known_normal_pool", clips)
    return clips


def load_trajectory_csv(traj_root, gt_dir=None, fps=24, split="test"):
    """
    MoCoDAD / Morais layout:
      traj_root/<scene>_<clip>/<person_id>.csv   rows: frame, x0,y0, x1,y1, ... (17 joints)
                                                 (35 cols) or frame + x,y,c*17 (52 cols)
      gt_dir/<scene>_<clip>.npy
    """
    clips = []
    for cdir in sorted(glob.glob(os.path.join(traj_root, "*"))):
        if not os.path.isdir(cdir):
            continue
        stem = os.path.basename(cdir)
        per_frame = {}
        for cp in glob.glob(os.path.join(cdir, "*.csv")):
            pid = _frame_no(os.path.basename(cp))
            try:
                arr = np.loadtxt(cp, delimiter=",", ndmin=2)
            except Exception:
                arr = np.genfromtxt(cp, delimiter=",")
                arr = np.atleast_2d(arr)
            if arr.size == 0:
                continue
            ncol = arr.shape[1]
            stepper = 3 if ncol >= 1 + 17 * 3 else 2   # x,y,c  vs  x,y
            for row in arr:
                fi = int(row[0])
                body = row[1:]
                kp = np.zeros((17, 3), float)
                for j in range(17):
                    o = j * stepper
                    if o + 1 < len(body):
                        kp[j, 0] = body[o]
                        kp[j, 1] = body[o + 1]
                        kp[j, 2] = body[o + 2] if stepper == 3 else 1.0
                per_frame.setdefault(fi, []).append((pid, kp))
        clips.append(_clips_from_perframe(stem, per_frame, _find_gt(gt_dir, stem),
                                          fps, split))
    _report("load_trajectory_csv", clips)
    return clips


def _report(who, clips):
    got = sum(len(c.gt_intervals) for c in clips)
    nf = sum(c.n_frames for c in clips)
    print(f"[{who}] {len(clips)} clips, {nf} frames, {got} gt events"
          + ("  -- NO GT FOUND (check gt_dir)" if got == 0 else ""))


# ------------------------------------------------------------------------- #
#  one entry point: point it at a dataset root, it figures out the layout   #
# ------------------------------------------------------------------------- #
def _all_dirs(root):
    return [d for d, _, _ in os.walk(root)]


def _rank_test(path):
    """higher = more likely the TEST split."""
    p = path.lower()
    return (("test" in p) * 2 + ("testing" in p) * 2
            - ("train" in p) * 3 - ("val" in p) * 2)


def inspect(root):
    """Print what pose / gt folders load_any would pick under `root`."""
    dirs = _all_dirs(root)
    gepc = [(d, len(glob.glob(f"{d}/*.json"))) for d in dirs
            if len(glob.glob(f"{d}/*.json")) >= 2]
    traj = [d for d in dirs if os.path.basename(d).lower() == "trajectories"
            and glob.glob(f"{d}/*/*.csv")]
    gt = [(d, len(glob.glob(f"{d}/*.npy"))) for d in dirs
          if len(glob.glob(f"{d}/*.npy")) >= 1]
    print(f"[inspect] {root}")
    print("  GEPC json dirs :", [(os.path.relpath(d, root), n) for d, n in
                                 sorted(gepc, key=lambda x: -x[1])[:6]] or "none")
    print("  trajectory dirs:", [os.path.relpath(d, root) for d in traj][:6] or "none")
    print("  .npy gt dirs   :", [(os.path.relpath(d, root), n) for d, n in
                                 sorted(gt, key=lambda x: -x[1])[:6]] or "none")
    return gepc, traj, gt


def load_any(root, fps=24, split="test"):
    """
    Auto-detect and load a pose-VAD dataset anywhere under `root`. Handles:
      * a single generic-schema .json file              -> load_generic_json
      * GEPC / STG-NF : any dir with many per-clip *.json (+ a *.npy gt dir)
      * MoCoDAD       : any '*/trajectories/<clip>/<pid>.csv' (+ a *.npy gt dir)
    Picks the TEST split when several candidates exist.
    """
    if os.path.isfile(root) and root.endswith(".json"):
        return load_generic_json(root, fps=fps)
    if not os.path.isdir(root):
        raise SystemExit(f"load_any: {root} is not a folder")

    gepc, traj, gt = inspect(root)

    # best ground-truth dir: prefer 'test_frame_mask', then 'test' in path, then most files
    gt_dir = None
    if gt:
        gt.sort(key=lambda x: (("frame_mask" in x[0].lower()) * 4 + _rank_test(x[0]), x[1]),
                reverse=True)
        gt_dir = gt[0][0]

    # GEPC: the json dir with the most files, preferring a 'test' path
    if gepc:
        gepc.sort(key=lambda x: (_rank_test(x[0]), x[1]), reverse=True)
        pose_dir = gepc[0][0]
        names = [os.path.basename(p) for p in glob.glob(os.path.join(pose_dir, "*.json"))]
        if any(n.startswith(("abnormal_", "normal_")) and "alphapose_tracked_person" in n
               for n in names):
            raise SystemExit(
                f"load_any: {os.path.relpath(pose_dir, root)} looks like UBnormal's "
                "STG-NF release (abnormal_*/normal_*_alphapose_tracked_person.json), "
                "not a generic GEPC dataset. --auto would load it with the ground-truth "
                "polarity backwards (UBnormal's convention is the OPPOSITE of "
                "ShanghaiTech's -- see load_ubnormal_stgnf's docstring). "
                "Use --ubnormal-auto <root> instead of --auto for UBnormal.")
        print(f"[load_any] GEPC  pose_dir={os.path.relpath(pose_dir, root)}  "
              f"gt_dir={os.path.relpath(gt_dir, root) if gt_dir else None}")
        return load_gepc_json(pose_dir, gt_dir, fps=fps, split=split)

    # MoCoDAD trajectory folders
    if traj:
        traj.sort(key=_rank_test, reverse=True)
        traj_dir = traj[0]
        print(f"[load_any] MoCoDAD  traj_dir={os.path.relpath(traj_dir, root)}  "
              f"gt_dir={os.path.relpath(gt_dir, root) if gt_dir else None}")
        return load_trajectory_csv(traj_dir, gt_dir, fps=fps, split=split)

    # CHAD: a folder with sibling annotations/*.pkl + anomaly_labels/*.npy
    for d, _, _ in os.walk(root):
        if (os.path.basename(d) == "annotations" and glob.glob(f"{d}/*.pkl")
                and os.path.isdir(os.path.join(os.path.dirname(d), "anomaly_labels"))):
            chad_root = os.path.dirname(d)
            print(f"[load_any] CHAD  root={os.path.relpath(chad_root, root) or '.'}")
            sf = "splits/test_split_1.txt"
            if not os.path.exists(os.path.join(chad_root, sf)):
                sf = None
            return load_chad(chad_root, fps=fps, split=split, split_file=sf)

    raise SystemExit(
        f"load_any: no pose data found under {root}.\n"
        "  expected: a folder of per-clip *.json (GEPC/STG-NF), or\n"
        "            */trajectories/<clip>/<pid>.csv (MoCoDAD), or\n"
        "            a single generic-schema .json file.\n"
        "  run  python -m calm.datasets <root>  to see the tree.")


def clips_to_generic(clips, fps=24, path=None):
    """Serialise loaded Clips to the generic schema (for reuse / cross-dataset merge)."""
    import dataclasses
    obj = {"fps": float(fps), "clips": [dataclasses.asdict(c) for c in clips]}
    for c in obj["clips"]:
        # load_generic_json reads the ground-truth key as "gt", not the
        # dataclass field name "gt_intervals" -- rename on the way out or the
        # round trip silently loses every ground-truth interval.
        c["gt"] = c.pop("gt_intervals")
        for fr in c["frames"]:
            for d in fr:
                kp = d.get("keypoints")
                if kp is not None:
                    d["keypoints"] = np.asarray(kp).round(2).tolist()
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    return obj


if __name__ == "__main__":
    # python -m calm.datasets <root>   -> show what load_any would pick + a tree
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m calm.datasets <dataset_root>")
        raise SystemExit(1)
    r = sys.argv[1]
    inspect(r)
    print("\n  first 3 levels of the tree:")
    base = r.rstrip("/\\")
    for d, subs, files in os.walk(r):
        depth = d[len(base):].count(os.sep)
        if depth > 2:
            subs[:] = []
            continue
        j = len(glob.glob(f"{d}/*.json"))
        c = len(glob.glob(f"{d}/*.csv"))
        n = len(glob.glob(f"{d}/*.npy"))
        tags = " ".join(t for t in (f"{j} json" if j else "", f"{c} csv" if c else "",
                                    f"{n} npy" if n else "") if t)
        print("   " + "  " * depth + os.path.basename(d) + ("/  " + tags if tags else "/"))
