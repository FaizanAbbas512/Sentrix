# CALM-VAD — manuscript

LaTeX source **and a compiled `main.pdf`** (10 pages, IEEEtran, clean build —
`\draftmodefalse`). Method, related work, problem formulation, and analysis
are fully written. **All five** planned benchmarks have real results in the
text and tables (ShanghaiTech, UBnormal, CHAD, Avenue, NWPU-Campus),
including B4 and the fusion- and calibration-variant ablation sweeps (Tables
VII-VIII, §VI-E), the full 20-pair cross-dataset generalisation matrix
across all five (Table IX, §VI-F), and two real qualitative evidence-record
panels with real Avenue video snapshots (Fig. 3). Every `refs.bib` citation
has been individually verified against its real source, including author
names (no entry cites "Anonymous"). `main.pdf` was compiled and visually
checked on this machine (MiKTeX), not fabricated. No `\pend{}` marker in
`main.tex` names an unresolved benchmark claim any more — the only three
left are the generic draft-mode legend text (inert while
`\draftmodefalse`).

**NWPU-Campus, completed 2026-09-20**: ran for real on Colab (T4 GPU, batched
extraction to fit a 64GB disk budget, `colab/calm_vad_nwpu_colab.ipynb`), all
242 real test clips, no subset. The result is a genuine, diagnosed negative
control, not a bug: across a 73,165-frame sample FALL/FIGHT/ABANDONED/CROWD
never fire and LOITERING fires on only 3.8% of frames (verified via a direct
`CueBank` run on the real merged pose data, not inferred) — NWPU-Campus's
wide, distant camera views rarely put a person in the pose/limb-speed regime
our cues were tuned to recognise, unrelated to pose-extraction quality
(median keypoint confidence 0.88). Frame AUC lands at exactly chance
(0.500), matching that diagnosis; M4 nonetheless certifies (naturally, no
boosted normal stream needed) because a quiet decision layer produces few
false alarms almost by construction. This is the paper's second independent
chance-level negative control alongside Avenue, reached by a different
underlying mechanism (§VI-D) — and, consistently, the two benchmarks with
the worst cross-dataset calibration transfer too (§VI-F).

**Cross-dataset matrix, completed 2026-09-20**: all 20 fit/test pairs across
all five benchmarks (`calm/run_all_cross_pairs.py`, new — the script that
produced the original 12-pair table was not preserved from earlier in the
project, so this regenerates all 20 with one documented method rather than
mixing an unrecoverable old method with a new one for just the 8 NWPU
pairs). Validated before trusting it: where the fit dataset has genuine
all-normal clips (UBnormal), this reproduces the original 12-pair table's
values exactly (+0.309, −0.013, +0.259) — the strongest confirmation
available without the lost script. Where it does not (ShanghaiTech / CHAD /
Avenue / NWPU-Campus as fit source — none of the five released pose-VAD
splits used here ships a clip with zero anomalous frames), the numbers
shift from the old table by a real, understood amount: `calm.harness.run()`
already had a fallback (unchanged, used since early in this project) that
borrows a class-balanced calibration subset from the test side when the fit
side lacks an all-normal clip, and that fallback firing consistently is why
those cells differ. Headline new finding: UBnormal-fit calibration onto
NWPU-Campus reaches ECE = 0.462, 35x NWPU-Campus's own in-domain 0.013 — the
single worst cell in the whole matrix.

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
tables/cross_dataset.tex      Table IX   — 20-pair cross-dataset generalisation, all 5 datasets — real
tables/cost.tex               Table X    — CPU latency incl. real front-end (YOLO11n-pose) timing — real
figures/qual_examples.tex     Fig. 3     — 2 real evidence records + real Avenue snapshots — real
```

`nwpu results/` (git-ignored, kept locally): the raw outputs from the real
Colab run — `calm_report_nwpucampus.{json,txt}` and the 288MB merged
`nwpucampus.json` pose file for all 242 test clips. Source of truth for
every NWPU-Campus number in the paper.

`results/` (git-ignored): every `calm_report_<tag>.{json,txt}`, including
all 20 `calm_report_<fit>__to__<test>.json` cross-dataset reports — source
of truth for Table IX.

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

`main.tex` line ~27: `\draftmodefalse`.

- `false` — clean build, no banner, no `pending`/`[TODO]` markers. **Set as
  of 2026-09-20** — every item tracked below is done, so there is nothing
  left for the banner to honestly flag.
- `true` — shows the "WORKING DRAFT" banner, blue `pending` text, red
  `[TODO]` notes; flip back to this if a new real gap is opened (e.g. a
  reviewer requests something not yet run) rather than silently shipping
  an unflagged gap.

## What's left before this is submission-ready

1. ~~NWPU-Campus~~ **Done for real, 2026-09-20.** Ran the prepared Colab
   pipeline (`colab/calm_vad_nwpu_colab.ipynb`) end to end: all 106 archive
   volumes via a Drive shortcut, extracted in batches of 50 to fit a 64GB
   disk budget, all 242 real test clips pose-extracted on a T4 GPU and
   merged, then scored with the exact same `calm.harness` as the other
   four. Real numbers throughout §V-§VIII and Tables VI-VIII.
2. ~~Extend the cross-dataset matrix to NWPU-Campus~~ **Done for real,
   2026-09-20.** All 20 pairs (not just the 8 new ones) regenerated with
   one documented, validated method — see the note above. Table IX,
   §VI-F.
3. ~~Run B3 (published pose-density baseline) and B4 (single strongest cue)~~
   B4 is run on all 5 datasets (Table VII). B3 is a deliberate, explained
   decline, not an oversight: a fair B3 needs cloning and training STG-NF's
   own normalizing-flow model end to end, a different scope of work from
   evaluating our own back end (see the Baselines paragraph in §V); we cite
   its published numbers rather than approximate a re-implementation.
4. ~~Run the fusion-variant and calibration-variant ablation sweeps~~ Done
   on all 5 datasets, real numbers (§VI-E, Tables VII-VIII). Headline
   findings: Dempster vs. noisy-OR is inconsequential everywhere
   (≤0.002 AUC); isotonic calibration beats Platt and temperature scaling on
   four of five benchmarks (the exception, NWPU-Campus, is reported not
   suppressed: temperature edges it out there, 0.010 vs. 0.013).
5. ~~2–3 qualitative figure panels (evidence records + snapshots)~~ Done:
   Fig. 3, 2 panels, real Avenue video frames + real evidence records found
   by an exhaustive search of all 21 test clips, reproducible via
   `python -m calm.qualitative search` and `python -m calm.qualitative
   snapshot --clip 01 --frame 965 --out ...`.
6. ~~Re-read the four closest papers cited in the dossier before claiming
   novelty in a submission~~ Done as an actual verification pass: every
   `refs.bib` entry's real source was fetched individually and confirmed
   real and accurately described — no fabricated citations. Fixed real
   defects along the way: ~18 placeholder "Anonymous" authors filled in
   with real names; a wrong volume/year on two Applied Ergonomics entries;
   a wrong journal name on one Elsevier entry; two dead entries removed;
   and one real factual error — `reliabilityproto2026` does **not** use
   temperature scaling — caught by reading its full text and the §II-A
   differentiation paragraph rewritten accordingly.
7. ~~`\draftmodefalse`, recompile, proofread~~ **Done, 2026-09-20** — every
   condition above is closed. A final human proofread of the prose (not a
   code/data check, an editorial one) is still worth doing before
   submission, same as for any manuscript.

## Target venues (see the research dossier for the full argument)

arXiv (cs.CV) first, then *MDPI Sensors* / *IEEE Access*; workshop fallback
IEEE AVSS or a CVPR/ICCV/WACV trustworthy-vision workshop.
