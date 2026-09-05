"""
reliability.py  —  M1 : per-frame / per-track reliability estimator
==================================================================

Skeleton-based cues are only as good as the skeleton. Occlusion, small person
scale, truncation at the frame edge, motion blur and ID switches all corrupt
keypoints BEFORE any analyzer sees them — and a corrupted skeleton often
"looks anomalous" (limbs fly around, torso angle jumps). Prior pose-VAD work
mostly ignores this; the closest work (Reliability-Aware Prototype Calibration,
arXiv:2606.20312) only down-weights a single model's score by mean keypoint
confidence.

We compute a scalar reliability  r in [0, 1]  from cheap features that are
already available in the SENTRIX pipeline, and feed it to M2 (evidence.py) as
a *discount rate*: an unreliable frame pushes belief mass to "unknown" instead
of to "anomaly".

Two modes:
  * heuristic (default) — a transparent weighted blend of the features.
  * fitted             — logistic regression, trained via `fit()` when you can
                          supply a proxy quality label (e.g. IoU of the
                          predicted skeleton against a reference, or agreement
                          between the pose cue and a box-trajectory cue).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

# COCO-17 keypoint indices we care about for "is this a usable skeleton".
_TORSO_IDX = (5, 6, 11, 12)          # shoulders + hips
_LIMB_IDX = (7, 8, 9, 10, 13, 14, 15, 16)  # elbows, wrists, knees, ankles


@dataclass
class ReliabilityFeatures:
    """Everything M1 looks at for one person on one frame. All in [0, 1]."""
    mean_kp_conf: float          # mean confidence over the 17 keypoints
    torso_kp_conf: float         # mean confidence over shoulders + hips
    visible_joint_frac: float    # fraction of the 17 joints above KP_CONF
    scale_norm: float            # person pixel height / frame height, capped
    untruncated: float           # 1 - (how much the bbox is clipped by frame edge)
    track_stability: float       # 1 on a settled track, dips right after an ID switch
    temporal_smooth: float       # 1 - normalised keypoint jitter vs previous frame

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.mean_kp_conf, self.torso_kp_conf, self.visible_joint_frac,
            self.scale_norm, self.untruncated, self.track_stability,
            self.temporal_smooth,
        ], dtype=float)

    @staticmethod
    def names() -> list[str]:
        return list(asdict(ReliabilityFeatures(0, 0, 0, 0, 0, 0, 0)).keys())


# Heuristic weights — deliberately readable, sum to 1. Torso visibility and
# keypoint confidence dominate; jitter and truncation are smaller corrections.
_HEURISTIC_W = np.array([0.28, 0.24, 0.16, 0.10, 0.09, 0.07, 0.06])

KP_CONF = 0.30   # a keypoint below this is treated as "not seen"


class ReliabilityEstimator:
    def __init__(self, cfg: dict | None = None):
        c = (cfg or {}).get("reliability", {}) if cfg else {}
        self.min_r = float(c.get("floor", 0.05))       # never fully trust or distrust
        self.max_r = float(c.get("ceil", 0.99))
        self.scale_cap = float(c.get("scale_cap", 0.45))  # person taller than this -> full scale score
        self.jitter_cap = float(c.get("jitter_cap", 0.15))  # torso-normalised jitter that maps to 0
        self._w = _HEURISTIC_W.copy()
        self._b = 0.0
        self._fitted = False
        # per-track memory for stability / smoothness features
        self._prev_kp: dict[int, np.ndarray] = {}
        self._age: dict[int, int] = {}
        self._last_switch: dict[int, int] = {}
        self._frame_no = 0

    # ------------------------------------------------------------------ features
    def features(self, track_id: int, keypoints, bbox, frame_shape,
                 id_switched: bool = False) -> ReliabilityFeatures:
        """
        keypoints : (17, 3) array of (x, y, conf) or None
        bbox      : (x1, y1, x2, y2)
        frame_shape : (H, W) or (H, W, C)
        """
        self._frame_no += 1
        H = float(frame_shape[0])
        W = float(frame_shape[1])
        x1, y1, x2, y2 = [float(v) for v in bbox]

        if keypoints is None:
            kp = np.zeros((17, 3), dtype=float)
        else:
            kp = np.asarray(keypoints, dtype=float).reshape(-1, 3)

        conf = kp[:, 2]
        mean_kp_conf = float(np.clip(conf.mean(), 0, 1)) if conf.size else 0.0
        torso_kp_conf = float(np.clip(conf[list(_TORSO_IDX)].mean(), 0, 1)) if conf.size else 0.0
        visible_joint_frac = float((conf >= KP_CONF).mean()) if conf.size else 0.0

        person_h = max(y2 - y1, 1.0)
        scale_norm = float(np.clip((person_h / H) / self.scale_cap, 0, 1))

        # how much of the bbox sits outside the frame -> truncation
        vis_w = max(min(x2, W) - max(x1, 0.0), 0.0)
        vis_h = max(min(y2, H) - max(y1, 0.0), 0.0)
        box_area = max((x2 - x1) * (y2 - y1), 1.0)
        untruncated = float(np.clip((vis_w * vis_h) / box_area, 0, 1))

        # track stability: 0 on the frame of a switch, recovers over ~15 frames
        if id_switched or track_id not in self._age:
            self._age[track_id] = 0
            self._last_switch[track_id] = self._frame_no
        self._age[track_id] += 1
        since_switch = self._frame_no - self._last_switch.get(track_id, self._frame_no)
        track_stability = float(np.clip(since_switch / 15.0, 0, 1))

        # temporal smoothness: torso-normalised mean shift of visible joints
        temporal_smooth = 1.0
        prev = self._prev_kp.get(track_id)
        if prev is not None and conf.size:
            torso_len = _torso_len(kp)
            both_vis = (conf >= KP_CONF) & (prev[:, 2] >= KP_CONF)
            if both_vis.any() and torso_len > 1:
                d = np.linalg.norm(kp[both_vis, :2] - prev[both_vis, :2], axis=1).mean()
                jitter = float(d / torso_len)
                temporal_smooth = float(np.clip(1.0 - jitter / self.jitter_cap, 0, 1))
        self._prev_kp[track_id] = kp.copy()

        return ReliabilityFeatures(
            mean_kp_conf, torso_kp_conf, visible_joint_frac, scale_norm,
            untruncated, track_stability, temporal_smooth,
        )

    # ------------------------------------------------------------------ predict
    def predict(self, feats: ReliabilityFeatures) -> float:
        x = feats.as_vector()
        if self._fitted:
            z = float(self._w @ x + self._b)
            r = 1.0 / (1.0 + math.exp(-z))
        else:
            r = float(self._w @ x)          # weights sum to 1 -> already in [0, 1]
        return float(np.clip(r, self.min_r, self.max_r))

    def reliability(self, track_id, keypoints, bbox, frame_shape,
                    id_switched: bool = False) -> tuple[float, ReliabilityFeatures]:
        f = self.features(track_id, keypoints, bbox, frame_shape, id_switched)
        return self.predict(f), f

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X : (n, 7) feature matrix from ReliabilityFeatures.as_vector()
        y : (n,)   proxy quality label in [0, 1] (soft) or {0, 1} (hard)

        Trains logistic regression. Falls back to the heuristic if sklearn is
        missing or the label has one class only.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 7:
            raise ValueError("X must be (n, 7); use ReliabilityFeatures.as_vector()")
        y_bin = (y >= 0.5).astype(int)
        if y_bin.min() == y_bin.max():
            print("[reliability] single-class proxy label - keeping heuristic weights")
            return self
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            print("[reliability] scikit-learn not installed - keeping heuristic weights")
            return self
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y_bin)
        self._w = clf.coef_.ravel().astype(float)
        self._b = float(clf.intercept_[0])
        self._fitted = True
        return self

    def reset_tracks(self):
        self._prev_kp.clear()
        self._age.clear()
        self._last_switch.clear()


def _torso_len(kp: np.ndarray) -> float:
    """Shoulder-midpoint to hip-midpoint distance; fallback to a constant."""
    sh = kp[[5, 6], :2].mean(axis=0)
    hp = kp[[11, 12], :2].mean(axis=0)
    d = float(np.linalg.norm(sh - hp))
    return d if d > 1 else 100.0
