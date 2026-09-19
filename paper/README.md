# CALM-VAD — manuscript

LaTeX source **and a compiled `main.pdf`** (9 pages, IEEEtran). Method,
related work, problem formulation, and analysis are fully written. Four of
five planned benchmarks have **real results** in the text and tables
(ShanghaiTech, UBnormal, CHAD, Avenue), including B4 and the fusion- and
calibration-variant ablation sweeps (Tables VII-VIII, §VI-E) and two real
qualitative evidence-record panels with real Avenue video snapshots
(Fig. 3). Every `refs.bib` citation has been individually verified against
its real source (not just re-read) — see item 5 below. Only NWPU-Campus
remains genuinely pending, and that gap was investigated rather than
assumed (item 1 below) — search `\pend{` in `main.tex` for the exact
remaining spots. `main.pdf` was compiled and visually checked on this
machine (MiKTeX), not fabricated.

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

1. NWPU-Campus — investigated (both plausible shortcuts around a GPU pass
   checked and ruled out with evidence, see main.tex §V), and the GPU pass
   itself is now fully prepared: **`colab/calm_vad_nwpu_colab.ipynb`**, a
   dedicated, resumable, ready-to-run notebook covering the real, complete
   dataset (all 242 real test clips, not a subset) —
     - all 106 real archive-volume Drive IDs hardcoded, MD5-verified against
       the dataset's own checksums, auto-retries a corrupt/partial download
       (this exact download+verify+retry logic was run for real against
       volume 1 of the actual dataset while building this: fresh download
       verified correct, a corrupted copy was correctly detected and
       re-fetched);
     - the real ground truth (`NWPU_Campus_gt.npz`, `0=normal/1=anomaly`,
       confirmed against the dataset's own `read_gt_file.py`) converted via
       the new `calm/nwpu_gt.py` module (tested against the real file: 242
       clips, sane real anomaly-interval durations);
     - pose-extraction is resumable per clip (`extract_poses.py --resume-dir`,
       new) and saved to the user's Drive as it goes, so a Colab disconnect
       -- likely, given this is hours of GPU work -- costs at most a
       re-download of the archive, never already-finished GPU compute;
     - ends by running the exact same `calm.harness` used for the other
       four benchmarks and printing the real numbers to paste into the
       paper.
   Not run yet (needs the user's own GPU quota) -- this is prepared
   infrastructure, not a result; `\draftmodetrue` stays true until it's
   actually run.
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
5. ~~Re-read the four closest papers cited in the dossier before claiming
   novelty in a submission~~ Done as an actual verification pass, not just
   a re-read: every `refs.bib` entry's real source was fetched individually
   (arXiv page, publisher listing, or the paper's own PDF text) and
   confirmed real and accurately described — no fabricated citations. Found
   and fixed real defects along the way: ~18 placeholder "Anonymous"
   authors filled in with the real names; a wrong volume/year on two
   Applied Ergonomics entries; a wrong journal name on one Elsevier entry;
   two dead (never-cited) entries removed; and one real factual error —
   `reliabilityproto2026` does **not** use temperature scaling (verified by
   reading its full text: zero occurrences of "temperature") — caught and
   the differentiation paragraph in §II-A rewritten to describe the actual
   method. One entry (`edgevadsurvey2026`) still has a placeholder author
   field — its title/venue/DOI are confirmed but the real author list is
   behind a paywalled abstract; resolve before camera-ready.
6. `\draftmodefalse`, recompile, proofread — hold until (1) is closed or
   explicitly accepted as a stated limitation (now the only remaining
   condition — (5) is done).

## Target venues (see the research dossier for the full argument)

arXiv (cs.CV) first, then *MDPI Sensors* / *IEEE Access*; workshop fallback
IEEE AVSS or a CVPR/ICCV/WACV trustworthy-vision workshop.