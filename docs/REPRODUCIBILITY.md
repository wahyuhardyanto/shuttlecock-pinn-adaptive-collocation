# Reproducibility instructions

## 1. Environment

The publication run was recorded with Python 3.9.16, PyTorch 2.8.0, NumPy 1.25.2, and pandas 2.0.2.

Using conda:

```bash
conda env create -f environment.yml
conda activate shuttlecock-pinn-alhs
```

or using pip in a suitable Python 3.9 environment:

```bash
pip install -r requirements.txt
```

## 2. Optional pipeline test

```bash
cd code
python run_main_270_v3_fairness.py --dry-run
```

This verifies the pipeline only. It is not part of the publication results.

## 3. Publication experiment

```bash
cd code
python run_main_270_v3_fairness.py
```

The preset is frozen as:

- methods: MC-R, LHS-R, ALHS-R;
- collocation budgets: 50, 100, 200;
- seeds: 0 through 29;
- Adam: 5000 epochs, learning rate 1e-3;
- common resampling every 250 epochs;
- physics validation every 250 epochs on 501 points;
- L-BFGS: maximum 300 iterations on the union of the latest four collocation sets;
- RK4: evaluation only.

## 4. Expected outputs

The runner creates a `main_v3_fairness_270/` directory containing the run-level CSV files, summaries, stage selection, integrity report, and manifests.

## 5. Verify integrity

Confirm that `main_v3_fairness_270_integrity.json` reports:

```json
{
  "expected_runs": 270,
  "observed_rows": 270,
  "unique_run_keys": 270,
  "duplicate_key_rows": 0,
  "missing_run_count": 0,
  "complete": true
}
```

## 6. Platform note

The publication experiment was executed with Apple MPS in float32. Small runtime and floating-point differences are expected across hardware and software stacks. The primary reproducibility target is the frozen experimental protocol and the qualitative/statistical findings, not bitwise-identical runtime values.
