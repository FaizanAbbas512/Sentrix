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
