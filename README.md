# Shuttlecock PINN adaptive-collocation reproducibility package

This repository accompanies the manuscript:

**Sample-Efficient Residual-Adaptive Collocation for Physics-Informed Neural Networks: A 270-Run Study of Shuttlecock Flipping Dynamics**

Target journal: *International Journal of Applied and Computational Mathematics* (Springer Nature).

## Purpose

This package supports the controlled numerical benchmark comparing:

- Monte Carlo resampling (MC-R),
- Latin hypercube resampling (LHS-R), and
- residual-adaptive Latin hypercube resampling (ALHS-R)

for a physics-informed neural network applied to nonlinear shuttlecock angular dynamics.

The publication experiment uses three collocation budgets (`N_f = 50, 100, 200`) and 30 paired random seeds, for 270 total training runs.

## Repository structure

- `code/` - frozen Python implementation and the 270-run publication preset.
- `data/` - run-level results, summaries, integrity check, stage-selection table, and sanitized experiment manifest.
- `supplementary/` - human-readable statistical supplementary PDF.
- `journal_supplementary/` - exact `ESM_1.pdf`, `ESM_2.zip`, and `ESM_3.zip` files intended for Springer supplementary submission.
- `docs/` - data dictionary, reproducibility instructions, online-resource mapping, and upload guides.
- `CITATION.cff` - machine-readable citation metadata for GitHub.
- `zenodo_metadata_template.json` - metadata template to copy into Zenodo or rename to `.zenodo.json` only if using the Zenodo-GitHub release workflow.
- `requirements.txt` / `environment.yml` - computational environment information.

## Reproduce the publication experiment

The frozen study environment recorded in the public manifest was:

- Python 3.9.16
- PyTorch 2.8.0
- NumPy 1.25.2
- pandas 2.0.2
- macOS 15.7.9, Apple MPS, float32

Create an environment, then run:

```bash
cd code
python run_main_270_v3_fairness.py
```

A small pipeline check can be run first with:

```bash
python run_main_270_v3_fairness.py --dry-run
```

The dry run is **not** part of the publication experiment.

## Integrity target

The publication dataset should report:

- expected runs: 270
- unique run keys: 270
- duplicate rows: 0
- missing runs: 0
- `complete = true`

See `data/main_v3_fairness_270_integrity.json`.

## Scope boundary

This repository supports the controlled numerical benchmark in the manuscript. It does **not** contain the newer experimentally observed shuttlecock trajectories being developed by the research team for subsequent studies.

## Citation and DOI

Before the first public release, connect the GitHub repository to Zenodo or create a manual Zenodo deposit. After Zenodo assigns a DOI, add the DOI badge and citation here and, if desired, add the DOI to `CITATION.cff`.

## Licensing

See `LICENSES_TO_CONFIRM.md`. The team should approve licenses before public release.
