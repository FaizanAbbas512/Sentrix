"""
metrics.py  —  the deployable evaluation axes
============================================

Frame-level AUC is kept for continuity with prior work, but the paper's story
is told by:

  * event-level F1 at temporal IoU        (operators act on events, not frames)
  * false alarms per hour at fixed recall  (the production-failure metric)
  * calibration error                      (in calibration.py)
  * cross-dataset drop                     (run the harness twice, subtract)
  * CPU latency                            (measured live in harness.py)

Ground truth is a list of (start_frame, end_frame) anomaly intervals.
Predictions are either a per-frame score array or a list of predicted intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
#  frame-level                                                               #
# --------------------------------------------------------------------------- #
def frame_labels(n_frames: int, gt_intervals) -> np.ndarray:
    y = np.zeros(int(n_frames), dtype=int)
    for a, b in gt_intervals:
        y[max(0, int(a)):min(n_frames, int(b) + 1)] = 1
    return y


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney). No sklearn needed."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    all_s = np.concatenate([pos, neg])
    _tie_correct(all_s, ranks)
    r_pos = ranks[:len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _tie_correct(values, ranks):
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    i = 0
    n = len(v)
    while i < n:
        j = i
        while j < n and v[j] == v[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j


# --------------------------------------------------------------------------- #
#  score stream  ->  predicted intervals                                     #
# --------------------------------------------------------------------------- #
def score_to_intervals(scores, tau: float, fps: float,
                       min_gap_s: float = 3.0, min_len_s: float = 0.3):
    scores = np.asarray(scores, float)
    above = scores >= tau
    gap = max(int(round(min_gap_s * fps)), 1)
    min_len = max(int(round(min_len_s * fps)), 1)
    intervals = []
    i, n = 0, len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        k = j
        while k < n and (k - j) < gap:
            if above[k]:
                j = k + 1
                while j < n and above[j]:
                    j += 1
                k = j
            else:
                k += 1
        if (j - i) >= min_len:
            intervals.append((i, j - 1))
        i = j
    return intervals


# --------------------------------------------------------------------------- #
#  event-level F1 at temporal IoU                                            #
# --------------------------------------------------------------------------- #
def _tiou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def event_f1(pred_intervals, gt_intervals, tiou: float = 0.3):
    """Greedy one-to-one matching at the given tIoU. Returns (f1, precision, recall)."""
    preds = list(pred_intervals)
    gts = list(gt_intervals)
    if not preds and not gts:
        return 1.0, 1.0, 1.0
    if not preds:
        return 0.0, 0.0, 0.0
    if not gts:
        return 0.0, 0.0, 0.0

    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            o = _tiou(p, g)
            if o >= tiou:
                pairs.append((o, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for o, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def event_f1_multi(pred_intervals, gt_intervals, tious=(0.2, 0.3, 0.4, 0.5)):
    """Average event-F1 over several tIoU thresholds (the [2] protocol)."""
    out = {}
    vals = []
    for t in tious:
        f1, p, r = event_f1(pred_intervals, gt_intervals, t)
        out[f"f1@{t}"] = f1
        vals.append(f1)
    out["f1_avg"] = float(np.mean(vals)) if vals else 0.0
    return out


# --------------------------------------------------------------------------- #
#  false alarms per hour at fixed recall                                     #
# --------------------------------------------------------------------------- #
@dataclass
class FaphAtRecall:
    target_recall: float
    achieved_recall: float
    tau: float
    faph: float
    n_false_events: int
    normal_hours: float


def faph_at_recall(scores, gt_intervals, fps: float, target_recall: float = 0.9,
                   n_frames: int | None = None, min_gap_s: float = 3.0):
    """
    Sweep tau; find the lowest tau (most alarms) whose EVENT recall >= target;
    at that tau, report false-alarm events per hour of NON-anomalous stream.
    """
    scores = np.asarray(scores, float)
    n = int(n_frames or len(scores))
    y = frame_labels(n, gt_intervals)
    normal_frames = int((y == 0).sum())
    normal_hours = normal_frames / fps / 3600.0

    best = None
    for tau in np.linspace(scores.max() if len(scores) else 1.0,
                           scores.min() if len(scores) else 0.0, 60):
        preds = score_to_intervals(scores, tau, fps, min_gap_s)
        _, _, rec = event_f1(preds, gt_intervals, tiou=0.1)   # loose: "did we catch it at all"
        # count false-alarm events = predicted intervals overlapping no GT
        fe = sum(1 for p in preds if all(_tiou(p, g) == 0 and
                                         not (p[0] <= g[1] and g[0] <= p[1])
                                         for g in gt_intervals)) if gt_intervals \
             else len(preds)
        faph = fe / normal_hours if normal_hours > 0 else float("inf")
        if rec >= target_recall:
            cand = FaphAtRecall(target_recall, rec, float(tau), float(faph),
                                int(fe), float(normal_hours))
            if best is None or cand.faph < best.faph:
                best = cand
    if best is None:      # target recall never reached
        preds = score_to_intervals(scores, float(scores.min() if len(scores) else 0.0),
                                   fps, min_gap_s)
        _, _, rec = event_f1(preds, gt_intervals, tiou=0.1)
        best = FaphAtRecall(target_recall, rec, float(scores.min() if len(scores) else 0.0),
                            float("inf"), len(preds), float(normal_hours))
    return best
