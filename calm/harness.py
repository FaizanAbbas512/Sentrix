"""
harness.py  —  the deployable-evaluation harness (paper's Section 5-6 engine)
==========================================================================

Given a set of Clips (calm/datasets.py) it runs the full CALM-VAD chain and
produces one report covering all five axes:

    detection    : frame AUC, event-F1 @ tIoU {0.2..0.5}
    alarm load   : false alarms / hour @ recall {0.7, 0.8, 0.9}, and @ budget
    calibration  : ECE / adaptive-ECE / Brier, raw vs calibrated
    generalization: (run twice on A->A and A->B, subtract; helper provided)
    cost         : ms/frame, FPS on THIS machine, memory footprint

It also runs the baselines the paper needs:
    B0 weighted-sum (SENTRIX as-is)   B1 logistic over cue strengths
    B2 gradient-boosted trees          + ablations (-M1, -M3, -M4, fusion swap)

Everything is pure-Python + numpy (+ optional sklearn for B1/B2). No OpenCV,
no camera, no GPU. Results are written to results/calm_report_<tag>.json and a
human-readable results/calm_report_<tag>.txt.

    python -m calm.harness --synthetic
    python -m calm.harness --generic path/to/benchmark.json --tag shanghaitech
"""

from __future__ import annotations

import argparse
import json
import os
import time
import platform
from dataclasses import asdict

import numpy as np
import yaml

from .cues import CueBank, CUE_NAMES
from .reliability import ReliabilityEstimator
from .evidence import EvidenceFusion
from .calibration import Calibrator, expected_calibration_error, brier_score
from .risk_control import LearnThenTest
from . import metrics as M


# --------------------------------------------------------------------------- #
#  score one clip -> per-frame raw belief + per-frame features for baselines  #
# --------------------------------------------------------------------------- #
def score_clip(clip, cfg, use_reliability=True):
    cb = CueBank(cfg, fps=clip.fps)
    rel = ReliabilityEstimator(cfg)
    fus = EvidenceFusion(cfg)

    belief = np.zeros(clip.n_frames, float)
    belief_noisyor = np.zeros(clip.n_frames, float)   # fusion-variant ablation
    r_series = np.ones(clip.n_frames, float)
    strength_mat = np.zeros((clip.n_frames, len(CUE_NAMES)), float)
    conflict = np.zeros(clip.n_frames, float)
    lat_ms = np.zeros(clip.n_frames, float)

    for t, dets in enumerate(clip.frames):
        t0 = time.perf_counter()
        strengths, aux = cb.step(dets)

        # per-frame pose reliability = mean over people who actually drove a cue
        if use_reliability:
            rs = []
            drivers = (aux["fallen_ids"] | aux["fight_ids"]
                       | aux["loiter_ids"] | aux["abandoned_ids"])
            persons = [d for d in dets if d.get("is_person", True)]
            pool = [d for d in persons if d["track_id"] in drivers] or persons
            for d in pool:
                r, _ = rel.reliability(d["track_id"], d.get("keypoints"),
                                       d["bbox"], (480, 640))
                rs.append(r)
            r_pose = float(np.mean(rs)) if rs else 1.0
        else:
            r_pose = 1.0

        fr = fus.fuse(strengths, r_pose=r_pose)
        belief_noisyor[t] = fus.fuse_noisy_or(strengths, r_pose=r_pose)
        lat_ms[t] = (time.perf_counter() - t0) * 1000.0

        belief[t] = fr.bel_A
        r_series[t] = r_pose
        conflict[t] = fr.conflict
        for i, name in enumerate(CUE_NAMES):
            strength_mat[t, i] = strengths[name]

    return {
        "belief": belief, "belief_noisyor": belief_noisyor,
        "r": r_series, "strengths": strength_mat,
        "conflict": conflict, "latency_ms": lat_ms,
    }


# --------------------------------------------------------------------------- #
#  baselines over the same per-frame cue-strength features                   #
# --------------------------------------------------------------------------- #
def _weighted_sum_scores(strength_mat, cfg):
    w = cfg.get("fusion", {}).get("weights", {})
    vec = np.array([w.get(n, 1) for n in CUE_NAMES], float)
    vec = vec / max(vec.sum(), 1e-9)
    return np.clip(strength_mat @ vec, 0, 1)


def _max_cue_scores(strength_mat):
    """B4: the single strongest cue, no fusion at all."""
    return np.clip(strength_mat.max(axis=1), 0, 1)


def _fit_sklearn(kind, X, y):
    try:
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=1000)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=120, max_depth=3)
        m.fit(X, y)
        return m
    except Exception as e:
        print(f"[harness] {kind} baseline skipped ({e})")
        return None


# --------------------------------------------------------------------------- #
#  run                                                                       #
# --------------------------------------------------------------------------- #
def run(clips, cfg, tag="synthetic", budgets=(2.0, 5.0, 10.0),
        recalls=(0.7, 0.8, 0.9), delta=0.05, outdir="results"):
    empty = [c.name for c in clips if c.n_frames == 0]
    if empty:
        print(f"[harness] dropping {len(empty)} zero-frame clip(s) "
              f"(empty pose file?): {empty[:5]}{' ...' if len(empty) > 5 else ''}")
        clips = [c for c in clips if c.n_frames > 0]
    if not clips:
        raise SystemExit("[harness] no clips with frames left to evaluate")

    fps = clips[0].fps
    calib = [c for c in clips if c.split == "calib"]
    test = [c for c in clips if c.split == "test"]
    if not test:
        test = [c for c in clips if c.split != "calib"] or clips
    # M3 needs BOTH classes in the calibration split; M4 needs normal-only.
    if not calib or not any(c.gt_intervals for c in calib) or not any(not c.gt_intervals for c in calib):
        norm = [c for c in test if not c.gt_intervals]
        anom = [c for c in test if c.gt_intervals]
        carve = norm[: max(1, len(norm) // 3)] + anom[: max(1, len(anom) // 3)]
        calib = list({c.name: c for c in (calib + carve)}.values())
        calib_names = {c.name for c in calib}          # Clip holds numpy arrays,
        test = [c for c in test if c.name not in calib_names] or test  # so compare by name, not `in`

    # ---- score every clip, with and without M1 ----------------------------
    scored = {c.name: {"clip": c,
                       "calm": score_clip(c, cfg, use_reliability=True),
                       "noM1": score_clip(c, cfg, use_reliability=False)}
              for c in clips}

    # ---- build calibration arrays (window = frame) ----------------------
    def stack(split_clips, key, field):
        b, y, S = [], [], []
        for c in split_clips:
            s = scored[c.name][key]
            yl = M.frame_labels(c.n_frames, c.gt_intervals)
            b.append(s[field]); y.append(yl); S.append(s["strengths"])
        return (np.concatenate(b), np.concatenate(y), np.vstack(S))

    bel_cal, y_cal, X_cal = stack(calib, "calm", "belief")
    bel_te, y_te, X_te = stack(test, "calm", "belief")
    bel_cal_n, _, _ = stack(calib, "noM1", "belief")
    bel_te_n, _, _ = stack(test, "noM1", "belief")

    # ---- M3: calibrate ------------------------------------------------
    # Frame-level calibration over the full test stream (matches how prior
    # audits report ECE). isotonic needs a few hundred positives or it
    # overfits, so fall back to Platt (2 params) on small calibration sets --
    # relevant for the smaller benchmarks (e.g. HR-Avenue) and for synthetic.
    method = cfg.get("calm", {}).get("calibration_method", "isotonic")
    n_pos_cal = int((y_cal == 1).sum())
    if method == "isotonic" and (len(y_cal) < 2000 or n_pos_cal < 200):
        method = "platt"
        print(f"[harness] small calibration set (n={len(y_cal)}, pos={n_pos_cal})"
              f" -> using Platt instead of isotonic")
    cal = Calibrator(method).fit(bel_cal, y_cal)
    cal_report = cal.report(bel_te, y_te)
    cal_report["region"] = f"all test frames; n={len(y_te)}, pos={int((y_te==1).sum())}"
    p_te = cal.predict(bel_te)

    # ablation: no calibration
    ece_noCal = expected_calibration_error(bel_te, y_te)

    # ---- baselines on the test frames -------------------------------
    ws_cal = Calibrator(method).fit(_weighted_sum_scores(X_cal, cfg), y_cal)
    p_ws = ws_cal.predict(_weighted_sum_scores(X_te, cfg))

    b4_cal = Calibrator(method).fit(_max_cue_scores(X_cal), y_cal)      # B4: strongest cue

    bel_cal_nor, _, _ = stack(calib, "calm", "belief_noisyor")          # fusion-variant ablation
    bel_te_nor, _, _ = stack(test, "calm", "belief_noisyor")
    nor_cal = Calibrator(method).fit(bel_cal_nor, y_cal)

    lr = _fit_sklearn("logistic", X_cal, y_cal)
    gb = _fit_sklearn("gbt", X_cal, y_cal)
    p_lr = lr.predict_proba(X_te)[:, 1] if lr is not None else None
    p_gb = gb.predict_proba(X_te)[:, 1] if gb is not None else None

    # ---- per-clip event metrics + FAPH ------------------------------
    def eval_stream(prob_by_clip, label):
        # frame AUC over concatenated test
        allp = np.concatenate([prob_by_clip[c.name] for c in test])
        auc = M.roc_auc(allp, y_te)
        # ---- event-F1: per-clip at a global tau, then averaged ----
        anom_clips = [c for c in test if c.gt_intervals]
        norm_clips = [c for c in test if not c.gt_intervals]
        ev_rows = []
        for c in test:
            preds = M.score_to_intervals(prob_by_clip[c.name], 0.5, c.fps)
            ev_rows.append(M.event_f1_multi(preds, c.gt_intervals))
        ev = {k: float(np.mean([e[k] for e in ev_rows])) for k in ev_rows[0]}

        # ---- FAPH at fixed recall, measured GLOBALLY ----
        # sweep one tau; recall = caught GT events over all anomaly clips;
        # false alarms = predicted events on NORMAL clips that hit no GT.
        from .risk_control import count_alarm_events
        norm_hours = sum(c.n_frames for c in norm_clips) / fps / 3600.0
        total_gt = sum(len(c.gt_intervals) for c in anom_clips)
        faph = {}
        for r in recalls:
            hit_faph = None
            for tau in np.linspace(0.98, 0.02, 60):
                caught = 0
                for c in anom_clips:
                    preds = M.score_to_intervals(prob_by_clip[c.name], tau, c.fps)
                    _, _, rec = M.event_f1(preds, c.gt_intervals, tiou=0.1)
                    caught += rec * len(c.gt_intervals)
                recall = caught / total_gt if total_gt else 0.0
                if recall >= r:
                    fa = sum(count_alarm_events(prob_by_clip[c.name], tau, c.fps)
                             for c in norm_clips)
                    hit_faph = fa / norm_hours if norm_hours > 0 else float("inf")
                    break
            faph[f"faph@recall={r}"] = (None if hit_faph is None
                                        else round(float(hit_faph), 2))
        return {"label": label, "frame_auc": auc, "event": ev, **faph}

    cal_n = Calibrator(method).fit(bel_cal_n, y_cal)          # noM1 calibrator, fit once
    prob_calm = {c.name: cal.predict(scored[c.name]["calm"]["belief"]) for c in test}
    prob_noM1 = {c.name: cal_n.predict(scored[c.name]["noM1"]["belief"]) for c in test}
    prob_ws = {c.name: ws_cal.predict(_weighted_sum_scores(scored[c.name]["calm"]["strengths"], cfg))
               for c in test}
    prob_b4 = {c.name: b4_cal.predict(_max_cue_scores(scored[c.name]["calm"]["strengths"]))
               for c in test}
    prob_nor = {c.name: nor_cal.predict(scored[c.name]["calm"]["belief_noisyor"]) for c in test}

    streams = [eval_stream(prob_calm, "CALM-VAD (M1+M2+M3)"),
               eval_stream(prob_noM1, "ablation -M1"),
               eval_stream(prob_nor, "ablation: fusion=noisy-OR"),
               eval_stream(prob_ws, "B0 weighted-sum (SENTRIX)"),
               eval_stream(prob_b4, "B4 strongest cue")]
    if p_lr is not None:
        off = 0
        prob_lr = {}
        for c in test:
            prob_lr[c.name] = p_lr[off:off + c.n_frames]; off += c.n_frames
        streams.append(eval_stream(prob_lr, "B1 logistic"))
    if p_gb is not None:
        off = 0
        prob_gb = {}
        for c in test:
            prob_gb[c.name] = p_gb[off:off + c.n_frames]; off += c.n_frames
        streams.append(eval_stream(prob_gb, "B2 gradient-boosted"))

    # ---- calibration-variant ablation: same belief arrays, 3 calibrators ----
    calib_variants = {}
    for cm in ("isotonic", "platt", "temperature"):
        try:
            cv = Calibrator(cm).fit(bel_cal, y_cal)
            rep = cv.report(bel_te, y_te)
            calib_variants[cm] = {"ece": rep["ece_cal"], "aece": rep["aece_cal"],
                                  "brier": rep["brier_cal"]}
        except Exception as e:
            calib_variants[cm] = {"error": str(e)}

    # ---- M4: pick tau on the NORMAL calibration stream, verify on held-out normal
    from .risk_control import count_alarm_events
    norm_calib_clips = [c for c in calib if not c.gt_intervals]
    norm_test_clips = [c for c in test if not c.gt_intervals]
    normal_calib = (np.concatenate([cal.predict(scored[c.name]["calm"]["belief"])
                                    for c in norm_calib_clips])
                    if norm_calib_clips else cal.predict(bel_cal))
    normal_test = (np.concatenate([prob_calm[c.name] for c in norm_test_clips])
                   if norm_test_clips else np.zeros(1))
    calib_hours = len(normal_calib) / fps / 3600.0
    held_hours = len(normal_test) / fps / 3600.0
    m4 = {"calib_normal_hours": round(calib_hours, 3),
          "held_out_normal_hours": round(held_hours, 3),
          "min_hours_to_certify": {b: round(0.5 * 5.9915 / b, 2) for b in budgets},
          "by_budget": {}}
    for beta in budgets:
        ltt = LearnThenTest(beta=beta, delta=delta, fps=fps)
        res = ltt.select(normal_calib)
        certified = len(res.admissible_taus) > 0        # did LTT admit any threshold?
        held_events = count_alarm_events(normal_test, res.tau, fps)
        held_faph = held_events / held_hours if held_hours > 0 else None
        m4["by_budget"][f"beta={beta}"] = {
            "tau": res.tau,
            "certified": certified,
            "faph_upper_bound_calib": res.faph_hi_at_tau,
            "held_out_normal_faph": held_faph,
            "held_out_holds": (held_faph is not None and held_faph <= beta),
            "note": res.guarantee,
        }

    # ---- cost -------------------------------------------------------
    all_lat = np.concatenate([scored[c.name]["calm"]["latency_ms"] for c in clips])
    cost = {
        "machine": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "decision_layer_ms_per_frame_mean": float(all_lat.mean()),
        "decision_layer_ms_per_frame_p95": float(np.percentile(all_lat, 95)),
        "decision_layer_fps_headroom": float(1000.0 / max(all_lat.mean(), 1e-6)),
        "note": "excludes pose extraction; that is the front-end cost, measured separately",
    }

    report = {
        "tag": tag,
        "n_clips": len(clips), "n_calib": len(calib), "n_test": len(test),
        "fps": fps,
        "calibration": {**cal_report, "ece_no_calibration": ece_noCal},
        "calibration_variants": calib_variants,
        "streams": streams,
        "risk_control_M4": m4,
        "cost": cost,
        "config_digest": {k: cfg.get(k) for k in ("fusion", "calm", "loitering",
                                                  "abandoned", "crowd", "behaviour", "fight")},
    }

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"calm_report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    txt = _render(report)
    with open(os.path.join(outdir, f"calm_report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n  -> results/calm_report_{tag}.json / .txt")
    return report


def _render(r) -> str:
    L = []
    L.append("=" * 66)
    L.append(f"  CALM-VAD deployable-evaluation report   [{r['tag']}]")
    L.append("=" * 66)
    L.append(f"  clips: {r['n_clips']}  (calib {r['n_calib']} / test {r['n_test']})   fps {r['fps']}")
    c = r["calibration"]
    L.append("-" * 66)
    L.append("  CALIBRATION (test split)")
    L.append(f"    method              : {c['method']}")
    L.append(f"    region              : {c.get('region', 'all frames')}")
    L.append(f"    ECE  raw -> cal      : {c['ece_raw']:.4f} -> {c['ece_cal']:.4f}")
    L.append(f"    adaptive-ECE r -> c  : {c['aece_raw']:.4f} -> {c['aece_cal']:.4f}")
    L.append(f"    Brier raw -> cal     : {c['brier_raw']:.4f} -> {c['brier_cal']:.4f}")
    if r.get("calibration_variants"):
        L.append("    calibration-variant ablation (same belief, different map):")
        for cm, v in r["calibration_variants"].items():
            if "error" in v:
                L.append(f"      {cm:<12}: skipped ({v['error']})")
            else:
                L.append(f"      {cm:<12}: ECE {v['ece']:.4f}  aECE {v['aece']:.4f}  Brier {v['brier']:.4f}")
    L.append("-" * 66)
    L.append("  DETECTION + ALARM LOAD (mean over test clips)")
    for s in r["streams"]:
        L.append(f"    {s['label']}")
        L.append(f"       frame AUC        : {s['frame_auc']:.3f}")
        L.append(f"       event F1 avg     : {s['event']['f1_avg']:.3f}   "
                 f"(@0.2 {s['event']['f1@0.2']:.2f} / @0.5 {s['event']['f1@0.5']:.2f})")
        for k in s:
            if k.startswith("faph@"):
                v = s[k]
                L.append(f"       {k:<16}: " + ("n/a" if v is None else f"{v:.1f} /hour"))
    L.append("-" * 66)
    m4 = r["risk_control_M4"]
    L.append("  RISK CONTROL  M4  (Learn-then-Test)")
    L.append(f"    normal-stream hours : calib {m4['calib_normal_hours']}  /  "
             f"held-out {m4['held_out_normal_hours']}")
    L.append(f"    hours needed to certify: " +
             "  ".join(f"b={b}->{h}h" for b, h in m4["min_hours_to_certify"].items()))
    for beta, d in m4["by_budget"].items():
        if d["certified"]:
            tag_ = "CERTIFIED"
        else:
            tag_ = "not certified (need more normal hours)"
        hf = d["held_out_normal_faph"]
        L.append(f"    {beta:<10} tau={d['tau']:.3f}  [{tag_}]  "
                 f"held-out {'n/a' if hf is None else f'{hf:.2f}/h'}")
    L.append("-" * 66)
    co = r["cost"]
    L.append("  COST  (decision layer only; pose extraction measured separately)")
    L.append(f"    {co['machine']}")
    L.append(f"    ms/frame mean/p95   : {co['decision_layer_ms_per_frame_mean']:.3f}"
             f" / {co['decision_layer_ms_per_frame_p95']:.3f}")
    L.append("=" * 66)
    L.append("  NOTE: numbers above depend entirely on the input clips. On the")
    L.append("  synthetic set they only prove the pipeline runs end to end.")
    L.append("=" * 66)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use the built-in synthetic clips")
    ap.add_argument("--degrade", action="store_true", help="synthetic: add pose degradation")
    ap.add_argument("--generic", default=None, help="path to a generic-JSON benchmark (datasets.load_generic_json)")
    ap.add_argument("--auto", default=None, metavar="ROOT",
                    help="dataset root; auto-detects GEPC/STG-NF json, MoCoDAD csv, or a generic json")
    ap.add_argument("--shanghaitech", nargs=2, metavar=("POSE_DIR", "GT_DIR"),
                    help="HR-ShanghaiTech pose dir + gt dir (GEPC-style json)")
    ap.add_argument("--ubnormal-stgnf", nargs=2, metavar=("POSE_DIR", "GT_DIR"),
                    help="UBnormal as released in the STG-NF bundle (inverted "
                         "gt polarity + different stem rule -- not auto-detected)")
    ap.add_argument("--ubnormal-auto", default=None, metavar="ROOT",
                    help="UBnormal root dir: auto-detects pose_dir/gt_dir like --auto, "
                         "but always applies the ground-truth inversion. Use this "
                         "instead of --auto for UBnormal -- --auto silently loads it "
                         "with the wrong gt polarity (it looks like a normal GEPC/"
                         "STG-NF dataset to the generic detector).")
    ap.add_argument("--chad", default=None, metavar="ROOT",
                    help="CHAD dataset root (contains annotations/, anomaly_labels/, splits/)")
    ap.add_argument("--fps", type=float, default=24.0, help="stream fps for real datasets")
    ap.add_argument("--save-generic", default=None,
                    help="also write the loaded clips to this generic-JSON path (for reuse / cross-dataset)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from . import datasets as D
    if args.generic:
        clips = D.load_generic_json(args.generic)
        tag = args.tag or os.path.splitext(os.path.basename(args.generic))[0]
    elif args.auto:
        clips = D.load_any(args.auto, fps=args.fps)
        tag = args.tag or os.path.basename(os.path.normpath(args.auto))
    elif args.shanghaitech:
        clips = D.load_gepc_json(args.shanghaitech[0], args.shanghaitech[1], fps=args.fps)
        tag = args.tag or "shanghaitech_hr"
    elif args.ubnormal_stgnf:
        clips = D.load_ubnormal_stgnf(args.ubnormal_stgnf[0], args.ubnormal_stgnf[1], fps=args.fps)
        tag = args.tag or "ubnormal"
    elif args.ubnormal_auto:
        clips = D.load_ubnormal_stgnf_auto(args.ubnormal_auto, fps=args.fps)
        tag = args.tag or "ubnormal"
    elif args.chad:
        clips = D.load_chad(args.chad, fps=args.fps)
        tag = args.tag or "chad"
    else:
        clips = D.synthetic(degrade=args.degrade)
        tag = args.tag or ("synthetic_degraded" if args.degrade else "synthetic")

    if args.save_generic and not args.generic:
        D.clips_to_generic(clips, fps=args.fps, path=args.save_generic)
        print(f"  -> saved generic copy: {args.save_generic}")

    run(clips, cfg, tag=tag)


if __name__ == "__main__":
    main()
