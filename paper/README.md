# CALM-VAD — manuscript

LaTeX source for the paper. Method, related work, problem formulation and
analysis are written; the result tables are stubs that you fill from the
evaluation harness.

## Files

```
main.tex                 the manuscript (IEEEtran conference)
refs.bib                 bibliography (verified Sept 2026; check page nos before camera-ready)
figures/architecture.tex baseline vs CALM-VAD back end   (TikZ, \input by main)
figures/protocol.tex     the 5 evaluation axes           (TikZ, \input by main)
tables/positioning.tex   capability matrix (Table I) — already complete
tables/main_results.tex  Table II  — detection + alarm load   <- fill from harness
tables/calibration.tex   Table III — ECE / Brier              <- fill from harness
tables/risk_control.tex  Table IV  — M4 budget adherence      <- fill from harness
tables/cross_dataset.tex Table V   — cross-dataset drop       <- fill from harness
tables/cost.tex          Table VI  — CPU latency              <- fill from harness
```

## Compile

Local install was not available here, so **do the first compile on Overleaf**
(New Project → Upload Project → zip this `paper/` folder) or with a full TeX
Live:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Needs: `IEEEtran`, `cite`, `amsmath/amssymb`, `algorithm`, `algpseudocode`,
`graphicx`, `booktabs`, `multirow`, `xcolor`, `tikz` (libs `arrows.meta`,
`positioning`, `fit`, `backgrounds`, `calc`), `hyperref`, `cleveref`. All are
in a standard TeX Live / Overleaf.

## Draft switch

`main.tex` line ~27: `\draftmodetrue`.

- `true`  — shows the "WORKING DRAFT" banner, blue `pending` text, red `[TODO]` notes.
- `false` — clean build for submission.

## Filling the tables

1. Run the harness on each benchmark:
   `python -m calm.harness --generic data/pose/<name>.json --tag <name>`
2. Open `results/calm_report_<name>.json`. The mapping:

   | JSON path | Table |
   |---|---|
   | `streams[*]` (`frame_auc`, `event.f1_avg`, `faph@recall=*`) | `tables/main_results.tex` (Table II) |
   | `calibration` (`ece_raw`/`ece_cal`, `aece_*`, `brier_*`, `ece_no_calibration`) | `tables/calibration.tex` (Table III) |
   | `risk_control_M4.by_budget` (`tau`, `certified`, `held_out_normal_faph`) | `tables/risk_control.tex` (Table IV) |
   | two runs A→A vs A→B, subtract `event.f1_avg` | `tables/cross_dataset.tex` (Table V) |
   | `cost` + SENTRIX `evaluate.py` for the pose front end | `tables/cost.tex` (Table VI) |

3. Replace every `--` / `\pend{...}` with the number. Turn `\draftmodefalse`.
4. Re-run the analysis paragraphs in Section VI against what you actually got —
   they currently state the *hypothesis* each table tests, in blue.

## Target venues (see the research dossier for the full argument)

arXiv (cs.CV) first, then *MDPI Sensors* / *IEEE Access*; workshop fallback
IEEE AVSS or a CVPR/ICCV/WACV trustworthy-vision workshop.
