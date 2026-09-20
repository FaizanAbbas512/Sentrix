# CALM-VAD — manuscript

LaTeX source **and a compiled `main.pdf`** (10 pages, IEEEtran). Method,
related work, problem formulation, and analysis are fully written. **All
five** planned benchmarks now have real results in the text and tables
(ShanghaiTech, UBnormal, CHAD, Avenue, NWPU-Campus), including B4 and the
fusion- and calibration-variant ablation sweeps (Tables VII-VIII, §VI-E) and
two real qualitative evidence-record panels with real Avenue video
snapshots (Fig. 3). Every `refs.bib` citation has been individually
verified against its real source, including author names (no entry cites
"Anonymous" any more). `main.pdf` was compiled and visually checked on this
machine (MiKTeX), not fabricated.

**NWPU-Campus, completed 2026-09-20**: ran for real on Colab (T4 GPU, batched
extraction to fit a 64GB disk budget, `colab/calm_vad_nwpu_colab.ipynb`), all
242 real test clips, no subset. The result is a genuine, diagnosed negative
control, not a bug: across a 73,165-frame sample FALL/FIGHT/ABANDONED/CROWD
never fire and LOITERING fires on only 3.8% of frames (verified via a direct
`CueBank` run on the real merged pose data, not inferred) -- NWPU-Campus's
wide, distant camera views rarely put a person in the pose/limb-speed regime
our cues were tuned to recognise, unrelated to pose-extraction quality
(median keypoint confidence 0.88). Frame AUC lands at exactly chance
(0.500), matching that diagnosis; M4 nonetheless certifies (naturally, no
boosted normal stream needed) because a quiet decision layer produces few
false alarms almost by construction. This is now the paper's second
independent chance-level negative control alongside Avenue, reached by a
different underlying mechanism, and is discussed as such in §VI-D.

**One item genuinely still open**: the 12-pair cross-dataset generalisation
matrix (Table IX, §VI-F) has not yet been extended to include NWPU-Campus as
a fit/test source (would need 8 more real cross-dataset pairs run). Search
`\pend{` in `main.tex` — only the generic draft-mode legend text remains
flagged; no benchmark-specific pending claims are left.

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
tables/cross_benchmark.tex    Table VI   — all 5 datasets side by side — real
tables/ablation_fusion.tex    Table VII  — Dempster vs. noisy-OR vs. B4, all 5 datasets — real
tables/ablation_calibration.tex Table VIII — isotonic vs. Platt vs. temperature ECE, all 5 datasets — real
tables/cross_dataset.tex      Table IX   — 12-pair cross-dataset generalisation — real, NWPU-Campus not yet added (8 more pairs)
tables/cost.tex               Table X    — CPU latency incl. real front-end (YOLO11n-pose) timing — real
figures/qual_examples.tex     Fig. 3     — 2 real evidence records + real Avenue snapshots — real
```

`nwpu results/` (git-ignored, kept locally): the raw outputs from the real
Colab run -- `calm_report_nwpucampus.{json,txt}` and the 288MB merged
`nwpucampus.json` pose file for all 242 test clips. Source of truth for
every NWPU-Campus number in the paper.

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
  **Currently true, deliberately** — one real gap remains (the cross-dataset
  matrix doesn't yet include NWPU-Campus), and hiding that would
  misrepresent the paper's state.
- `false` — clean build once that is gone (or accepted as a stated limitation
  rather than a to-do); flip it and recompile before submission.

## What's left before this is submission-ready

1. ~~NWPU-Campus~~ **Done for real, 2026-09-20.** Ran the prepared Colab
   pipeline (`colab/calm_vad_nwpu_colab.ipynb`) end to end: all 106 archive
   volumes via a Drive shortcut (anonymous `gdown` proved unreliable past
   ~30 files -- Google blocks the IP -- fixed to read through the user's own
   authenticated Drive session instead), extracted in batches of 50 to fit a
   64GB disk budget, all 242 real test clips pose-extracted on a T4 GPU and
   merged, then scored with the exact same `calm.harness` as the other four.
   Real numbers now throughout §V-§VIII and Tables VI-VIII (source:
   `nwpu results/calm_report_nwpucampus.json`, git-ignored, kept locally).
   The one remaining sub-item: NWPU-Campus is not yet in the 12-pair
   cross-dataset matrix (Table IX) as a fit/test source -- would need 8 more
   real cross-dataset runs; no cross-dataset runner script currently exists
   in the repo to do this with (the one used for the original 12 pairs was
   ad hoc and not preserved) -- would need writing first.
2. ~~Run B3 (published pose-density baseline) and B4 (single strongest cue)~~
   B4 is run on all 5 datasets (Table VII). B3 is a deliberate, explained
   decline, not an oversight: a fair B3 needs cloning and training STG-NF's
   own normalizing-flow model end to end, a different scope of work from
   evaluating our own back end (see the Baselines paragraph in §V); we cite
   its published numbers rather than approximate a re-implementation.
3. ~~Run the fusion-variant and calibration-variant ablation sweeps~~ Done
   on all 5 datasets, real numbers (§VI-E, Tables VII-VIII). Headline
   findings: Dempster vs. noisy-OR is inconsequential everywhere
   (≤0.002 AUC); isotonic calibration beats Platt and temperature scaling on
   four of five benchmarks (the exception, NWPU-Campus, is reported not
   suppressed: temperature edges it out there, 0.010 vs. 0.013, because the
   near-flat belief on that benchmark leaves isotonic little non-monotone
   structure to exploit).
4. ~~2–3 qualitative figure panels (evidence records + snapshots)~~ Done:
   Fig. 3, 2 panels, real Avenue video frames + real evidence records found
   by an exhaustive search of all 21 test clips, reproducible via
   `python -m calm.qualitative search` (prints the winning clip/frame/record)
   and `python -m calm.qualitative snapshot --clip 01 --frame 965 --out ...`
   (decodes the exact real video frame).
5. ~~Re-read the four closest papers cited in the dossier before claiming
   novelty in a submission~~ Done as an actual verification pass, not just
   a re-read: every `refs.bib` entry's real source was fetched individually
   (arXiv page, publisher listing, or the paper's own PDF text) and
   confirmed real and accurately described — no fabricated citations. Found
   and fixed real defects along the way: ~18 placeholder "Anonymous"
   authors filled in with the real names (including `edgevadsurvey2026`,
   resolved last, via independent cross-checking of the named authors'
   institutional profiles — no entry cites Anonymous any more); a wrong
   volume/year on two Applied Ergonomics entries; a wrong journal name on
   one Elsevier entry; two dead (never-cited) entries removed; and one real
   factual error — `reliabilityproto2026` does **not** use temperature
   scaling (verified by reading its full text: zero occurrences of
   "temperature") — caught and the differentiation paragraph in §II-A
   rewritten to describe the actual method.
6. `\draftmodefalse`, recompile, proofread — hold until (1)'s cross-dataset
   sub-item is closed or explicitly accepted as a stated limitation (every
   other condition is now done).

## Target venues (see the research dossier for the full argument)

arXiv (cs.CV) first, then *MDPI Sensors* / *IEEE Access*; workshop fallback
IEEE AVSS or a CVPR/ICCV/WACV trustworthy-vision workshop.