# CALM-VAD — manuscript

LaTeX source **and a compiled `main.pdf`** (8 pages, IEEEtran). Method,
related work, problem formulation, and analysis are fully written. Four of
five planned benchmarks have **real results** in the text and tables
(ShanghaiTech, UBnormal, CHAD, Avenue); NWPU-Campus and a few extras (B3/B4
baselines, the fusion/calibration ablation sweep, qualitative figure panels)
are still marked `pending` in the text — search `\pend{` in `main.tex` for
the exact list. `main.pdf` was compiled and visually checked on this machine
(MiKTeX), not fabricated.

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
tables/cross_benchmark.tex    Table VI  — all 4 datasets side by side — real
tables/cross_dataset.tex      Table VII — 12-pair cross-dataset generalisation — real
tables/cost.tex               Table  -- CPU latency, front-end row still \pend{}
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
  **Currently true, deliberately** — real placeholders remain (NWPU, B3/B4,
  qualitative figures), and hiding that would misrepresent the paper's state.
- `false` — clean build once those are gone; flip it and recompile before submission.

## What's left before this is submission-ready

1. NWPU-Campus (needs a GPU pose-extraction pass — 76.6 GB raw video, no pose release)
2. Run B3 (published pose-density baseline) and B4 (single strongest cue)
3. Run the fusion-variant and calibration-variant ablation sweeps
4. 2–3 qualitative figure panels (evidence records + snapshots)
5. Re-read the four closest papers cited in the dossier before claiming novelty in a submission
6. `\draftmodefalse`, recompile, proofread

## Target venues (see the research dossier for the full argument)

arXiv (cs.CV) first, then *MDPI Sensors* / *IEEE Access*; workshop fallback
IEEE AVSS or a CVPR/ICCV/WACV trustworthy-vision workshop.