# CALM-VAD — manuscript

LaTeX source **and a compiled `main.pdf`** (9 pages, IEEEtran). Method,
related work, problem formulation, and analysis are fully written. Four of
five planned benchmarks have **real results** in the text and tables
(ShanghaiTech, UBnormal, CHAD, Avenue), including B4 and the fusion- and
calibration-variant ablation sweeps (Tables VII-VIII, §VI-E) and two real
qualitative evidence-record panels with real Avenue video snapshots
(Fig. 3). Only NWPU-Campus remains genuinely pending — search `\pend{` in
`main.tex` for the exact remaining spots. `main.pdf` was compiled and
visually checked on this machine (MiKTeX), not fabricated.

## Files

```
main.tex / main.pdf      the manuscript + its compiled PDF (IEEEtran conference)
refs.bib                 bibliography (verified Sept 2026; check page nos before camera-ready)
figures/architecture.tex baseline vs CALM-VAD back end   (TikZ, \input by main)
figures/protocol.tex     the 5 evaluation axes           (TikZ, \input by main)
tables/positioning.tex        Table I   — capability matrix — complete
tables/main_results.tex       Table II  — ShanghaiTech detection + alarm load — real
tables/calibration.tex        Table III — ShanghaiTech calibration — real
tables/risk_control.tex       Table IV  — ShanghaiTech M4 — real
tables/risk_control_ubnormal.tex Table V — UBnormal M4, boosted normal hours (certified!) — real
tables/cross_benchmark.tex    Table VI   — all 4 datasets side by side — real
tables/cross_dataset.tex      Table IX   — 12-pair cross-dataset generalisation — real
tables/ablation_fusion.tex    Table VII  — Dempster vs. noisy-OR vs. B4, all 4 datasets — real
tables/ablation_calibration.tex Table VIII — isotonic vs. Platt vs. temperature ECE, all 4 datasets — real
tables/cost.tex               Table X    — CPU latency incl. real front-end (YOLO11n-pose) timing — real
figures/qual_examples.tex     Fig. 3     — 2 real evidence records + real Avenue snapshots — real
```

## Compile

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Works on Overleaf (New Project → Upload Project → zip `paper/`) or any TeX
Live / MiKTeX install with the packages below. If you change `main.tex`,
recompile and re-check `main.pdf` before committing it (a stale PDF next to
changed `.tex` is worse than no PDF).

Needs: `IEEEtran`, `cite`, `amsmath/amssymb`, `algorithm`, `algpseudocode`,
`graphicx`, `booktabs`, `multirow`, `xcolor`, `tikz` (libs `arrows.meta`,
`positioning`, `fit`, `backgrounds`, `calc`), `hyperref`, `cleveref`. All are
in a standard TeX Live / MiKTeX / Overleaf install.

## Draft switch

`main.tex` line ~27: `\draftmodetrue`.

- `true`  — shows the "WORKING DRAFT" banner, blue `pending` text, red `[TODO]` notes.
  **Currently true, deliberately** — one real gap remains (NWPU-Campus),
  and hiding that would misrepresent the paper's state.
- `false` — clean build once that is gone (or accepted as a stated limitation
  rather than a to-do); flip it and recompile before submission.

## What's left before this is submission-ready

1. NWPU-Campus (needs a GPU pose-extraction pass — 76.6 GB raw video, no pose
   release). Not attempted — genuinely out of scope for a CPU-only session;
   flagged rather than silently skipped.
2. ~~Run B3 (published pose-density baseline) and B4 (single strongest cue)~~
   B4 is run on all 4 datasets (Table VII). B3 is a deliberate, explained
   decline, not an oversight: a fair B3 needs cloning and training STG-NF's
   own normalizing-flow model end to end, a different scope of work from
   evaluating our own back end (see the Baselines paragraph in §V); we cite
   its published numbers rather than approximate a re-implementation.
3. ~~Run the fusion-variant and calibration-variant ablation sweeps~~ Done
   on all 4 datasets, real numbers (§VI-E, Tables VII-VIII). Headline
   findings: Dempster vs. noisy-OR is inconsequential everywhere
   (≤0.002 AUC); isotonic calibration beats Platt and temperature scaling
   on every one of the four benchmarks, sometimes by 10x.
4. ~~2–3 qualitative figure panels (evidence records + snapshots)~~ Done:
   Fig. 3, 2 panels, real Avenue video frames + real evidence records found
   by an exhaustive search of all 21 test clips, reproducible via
   `python -m calm.qualitative search` (prints the winning clip/frame/record)
   and `python -m calm.qualitative snapshot --clip 01 --frame 965 --out ...`
   (decodes the exact real video frame).
5. Re-read the four closest papers cited in the dossier before claiming
   novelty in a submission — still an editorial task for a human pass,
   not done in this session.
6. `\draftmodefalse`, recompile, proofread — hold until (1) is closed or
   explicitly accepted as a stated limitation, and (5) is done.

## Target venues (see the research dossier for the full argument)

arXiv (cs.CV) first, then *MDPI Sensors* / *IEEE Access*; workshop fallback
IEEE AVSS or a CVPR/ICCV/WACV trustworthy-vision workshop.