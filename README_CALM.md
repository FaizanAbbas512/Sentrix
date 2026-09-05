# CALM-VAD — the research layer on top of SENTRIX

**C**alibrated, **A**uditable, **L**ow-latency **M**ulti-cue **V**ideo **A**nomaly **D**etection.

SENTRIX (`src/`) gives perception + 5 behavioural cues, then ends in a
hand-weighted sum and a hand-tuned threshold. `calm/` replaces those two weak
links with a 4-part chain and evaluates it the way a control-room operator
would, not the way a leaderboard does.

Full rationale, the gap analysis and the proof-of-novelty are in the
**research dossier** (published artifact). This file is the *engineering* guide.

---

## The chain

| Step | File | What it does | Paper section |
|---|---|---|---|
| **M1** reliability | `calm/reliability.py` | scalar `r∈[0,1]` — how much to trust this frame's pose/tracking (keypoint conf, visible joints, scale, truncation, ID-switch, jitter) | Method 4.1 |
| **M2** evidence fusion | `calm/evidence.py` | each cue → Dempster–Shafer mass over `{anomaly, normal, unknown}`, **discounted by `r`**, combined with Dempster's rule; keeps the conflict mass `K` | Method 4.2 |
| **M3** calibration | `calm/calibration.py` | isotonic / Platt / temperature map so fused belief becomes a probability with low ECE | Method 4.3 |
| **M4** risk control | `calm/risk_control.py` | Learn-then-Test: pick `τ` so `P(false alarms/hour ≤ β) ≥ 1−δ` (exact Poisson bound, fixed-sequence testing) | Method 4.4 |
| output | `calm/detector.py` | `CalmDecision.record` — the auditable JSON: `p`, per-cue mass, `K`, `r`, `τ`, budget | Method 4.5 |

Adapters and evaluation:

| File | Role |
|---|---|
| `calm/cues.py` | SENTRIX analyzer state → cue strength `s∈[0,1]`. `CueBank` re-implements the 5 timers for offline/benchmark use (no OpenCV). |
| `calm/metrics.py` | frame AUC, **event-F1 @ tIoU**, **FAPH @ fixed recall**, score→interval debouncing |
| `calm/datasets.py` | `synthetic()` (no download) · `load_generic_json()` (documented flat schema) · `load_shanghaitech_hr()` |
| `calm/harness.py` | runs the whole thing → `results/calm_report_<tag>.json` + `.txt` (all 5 axes + baselines + ablations) |
| `calm/selftest.py` | 22 invariant checks on M1–M4, no data needed |
| `calm/extract_poses.py` | videos → the generic-JSON pose schema (the Colab/Kaggle step) |

---

## Quick start

```bash
pip install -r requirements.txt          # adds scikit-learn + scipy

python -m calm.selftest                  # 22 checks, ~2 s — proves the maths
python -m calm.harness --synthetic       # end-to-end on scripted clips, ~40 s
python -m calm.harness --synthetic --degrade   # adds pose degradation (tests M1)
```

> The synthetic numbers are **not publication results** — the clips are scripted.
> But `--degrade` is a *controlled demonstration of the mechanism*: it injects
> sustained corrupted-skeleton episodes into otherwise-normal clips, and the
> ablation shows M1 and M3 doing their jobs (venv run, CPU):
>
> | (synthetic `--degrade`) | frame AUC | FAPH @ recall 0.9 | ECE raw→cal |
> |---|---|---|---|
> | **CALM-VAD (M1+M2+M3)** | **0.96** | **0.0 / h** | **0.089 → 0.071** |
> | ablation $-$M1          | 0.94 | **19.2 / h** | — |
> | B0 weighted-sum         | 0.94 | **19.2 / h** | — |
>
> M1's reliability discount removes the corrupted-skeleton false alarms
> (19.2 → 0 /h); M3 lowers calibration error. On clean synthetic
> (`--synthetic` without `--degrade`) all methods are equivalent — M1 only
> matters when the front end is actually unreliable, which is the point.
> `python -m calm.selftest` → 22/22. Real magnitudes come from the benchmark
> runs below.

---

## Running it for real (the paper's experiments)

### 1. Get skeletons

Pick benchmarks: **ShanghaiTech**, **NWPU-Campus**, **UBnormal**, **HR-Avenue**,
**CHAD** (+ a **UCF-Crime** subset for the cross-domain drop). Extract 17-keypoint
COCO poses once on Colab/Kaggle GPU and **cache them** — everything after is
CPU-light. The bundled script does this and writes the schema directly:

```bash
python -m calm.extract_poses \
    --videos data/benchmarks/shanghaitech/testing/videos \
    --gt     data/benchmarks/shanghaitech/testing/frame_masks \
    --out    data/pose/shanghaitech.json --split test --device 0
```

`--gt` accepts per-clip `*.npy` frame masks, `*.txt` (one `start end` per line),
or `*.json` (`{"gt": [[s,e],...]}`), matched by filename stem. Run once per split.

### 2. Convert to the generic schema (once per benchmark)

```json
{
  "fps": 20,
  "clips": [
    {"name": "01_0014", "n_frames": 900, "split": "test",
     "gt": [[420, 530]],
     "frames": [
       [ {"id": 3, "keypoints": [x,y,c, ...17 triples...],
          "bbox": [x1,y1,x2,y2], "is_person": true, "label": "person"} ],
       []                                    // empty frame = nobody detected
     ]}
  ]
}
```

- `split`: `"calib"` for the calibration slice (M3 needs both classes; M4 needs
  the normal clips in it), `"test"` for evaluation. If you don't set splits the
  harness carves a calibration set automatically.
- `gt`: list of `[start_frame, end_frame]` anomaly intervals (frame-level GT is
  converted for you).
- HR-ShanghaiTech pose dumps: use `load_shanghaitech_hr(pose_dir, gt_dir)`
  directly (expects `<clip>.json` + `<clip>.npy` frame mask).

### 3. Run

```bash
python -m calm.harness --generic data/pose/shanghaitech.json --tag shanghaitech
python -m calm.harness --generic data/pose/nwpu.json         --tag nwpu
```

### 4. Cross-dataset drop (generalization axis)

Fit M3 + M4 on dataset A, evaluate on B. Simplest path: build one generic-JSON
where A's clips are `split:"calib"` and B's clips are `split:"test"`, then run
once. Report the absolute drop vs the A→A run.

### 5. Read the report

`results/calm_report_<tag>.txt` covers all five axes; the `.json` has every
number for your tables. The rows you paste into the paper:

- **Detection** — frame AUC (continuity) + event-F1 avg / @0.2 / @0.5
- **Alarm load** — FAPH @ recall 0.7 / 0.8 / 0.9, per stream
- **Calibration** — ECE / adaptive-ECE / Brier, raw → calibrated, + `ece_no_calibration`
- **Risk control (M4)** — per budget β: `τ`, `certified?`, held-out FAPH, whether the guarantee held
- **Cost** — decision-layer ms/frame on your machine (pose extraction measured separately with SENTRIX `evaluate.py`)

### 6. Headline claims the harness is built to test

1. M1+M2 cut FAPH at fixed recall vs B0 (weighted-sum) / B1 (logistic) / B2 (GBT)
2. M3 cuts ECE sharply with no recall loss (`ece_raw` → `ece_cal`)
3. M4 holds the FAPH budget on held-out normal streams (`held_out_holds` = true)
4. End-to-end runs > 15 FPS on CPU (decision layer + SENTRIX pose front-end)
5. Cross-dataset drop is smaller for the calibrated chain than for uncalibrated baselines

---

## Using CALM-VAD live (in the SENTRIX pipeline)

```python
from calm import CalmVAD
from calm.cues import strength_from_pipeline

calm = CalmVAD(CFG)
# once, offline: calm.fit_calibration(bel, labels); calm.set_threshold_from_normal_stream(p_normal)

# per frame, after the analyzers have run:
strengths = strength_from_pipeline(loiter_ids, abandon_ids, fall_ids, fight_ids,
                                   people, CFG["crowd"]["threshold"],
                                   loiter_obj=pipe.loiter, abandon_obj=pipe.abandon, now=now)
r_pose = ...  # mean M1 reliability over the people who drove the cues
decision = calm.assess(strengths, pose_reliability=r_pose, now=now)
if decision.alarm:
    log_json(decision.record)     # the auditable evidence record
```

**This is already wired in.** Set `calm.enabled: true` in `config.yaml` and the
live pipeline (`app.py`, `run_cli.py`) runs M1–M4 every frame, puts the
evidence record in `pipeline.stats["calm"]`, and appends fired alarms to
`data/logs/evidence.jsonl`. It defaults to `false` so nothing changes unless
you ask for it. Calibration/budget are the config `fallback` values until you
call `fit_calibration()` / `set_threshold_from_normal_stream()` with real data.

---

## Config

All CALM-VAD knobs live in the `calm:` / `reliability:` / `evidence:` blocks of
`config.yaml`. `fusion.weights` is kept — it's the B0 baseline.

## Honesty notes

- `calm/` contains **no results**. Every number is produced by *your* harness run.
- `scikit-learn` is required for isotonic calibration and the B1/B2 baselines;
  without it M3 falls back to Platt and the baselines are skipped.
- M4 needs enough **confirmed-normal** calibration video: ≈ `0.5 · χ²₀.₉₅(2) / β`
  hours (≈ 0.6 h for β=5, ≈ 1.5 h for β=2). The report prints this and marks
  each budget `certified` / `not certified`.
