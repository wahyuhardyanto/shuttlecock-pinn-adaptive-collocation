"""Frozen main-experiment preset for the shuttlecock PINN V3 fairness study.

Main experiment (publication run)
---------------------------------
- Methods: MC-R, LHS-R, ALHS-R
- Collocation budgets: N_f = 50, 100, 200
- Paired seeds: 0..29
- Total runs: 3 x 3 x 30 = 270
- Adam: 5000 epochs, lr=1e-3
- Common periodic resampling: every 250 epochs
- Physics validation: every 250 epochs on 501 fixed points
- L-BFGS: max_iter=300 on union of the last 4 collocation sets
- Endpoint physics anchors at t=0 and t=T for every method
- RK4: evaluation only
- Model checkpoints: NOT saved for the 270-run main experiment
- Resume: enabled by default

The numerical/algorithmic implementation lives in
``shuttlecock_pinn_diagnostic_v3_fairness.py`` and is intentionally not
modified by this preset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd
import torch

import shuttlecock_pinn_diagnostic_v3_fairness as v3


PRESET_VERSION = "main270_v3_fairness_1.0"
METHODS: Tuple[str, ...] = ("mc", "lhs", "alhs")
N_VALUES: Tuple[int, ...] = (50, 100, 200)
SEEDS: Tuple[int, ...] = tuple(range(30))
EXPECTED_RUNS = len(METHODS) * len(N_VALUES) * len(SEEDS)


def make_main_config() -> v3.TrainConfig:
    """Return the frozen V3 configuration used for the main experiment."""
    return v3.TrainConfig(
        n_collocation=100,  # overwritten per budget by v3.run_experiment
        epochs=5000,
        lr=1e-3,
        resample_every=250,
        validation_every=250,
        validation_points=501,
        hidden=32,
        depth=3,
        alhs_mode="residual",
        alhs_n_grid=2001,
        alhs_uniform_mix=0.10,
        alhs_power=1.0,
        lbfgs_max_iter=300,
        lbfgs_union_sets=4,
        restore_best_adam_before_lbfgs=True,
        select_final_by_validation=True,
        print_every=0,
    )


def make_dry_config() -> v3.TrainConfig:
    """Tiny configuration used only to verify the execution pipeline."""
    return v3.TrainConfig(
        n_collocation=30,
        epochs=30,
        lr=1e-3,
        resample_every=10,
        validation_every=10,
        validation_points=101,
        hidden=16,
        depth=2,
        alhs_mode="residual",
        alhs_n_grid=401,
        alhs_uniform_mix=0.10,
        alhs_power=1.0,
        lbfgs_max_iter=5,
        lbfgs_union_sets=2,
        restore_best_adam_before_lbfgs=True,
        select_final_by_validation=True,
        print_every=10,
    )


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clear_accelerator_cache() -> None:
    """Release cached accelerator memory between collocation-budget blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def expected_keys(
    methods: Sequence[str], n_values: Sequence[int], seeds: Iterable[int]
) -> set[tuple[str, int, int]]:
    return {
        (str(method).lower(), int(n), int(seed))
        for n in n_values
        for seed in seeds
        for method in methods
    }


def integrity_report(
    df: pd.DataFrame,
    methods: Sequence[str],
    n_values: Sequence[int],
    seeds: Sequence[int],
) -> Dict[str, object]:
    expected = expected_keys(methods, n_values, seeds)

    if df.empty:
        observed: set[tuple[str, int, int]] = set()
        duplicate_count = 0
    else:
        keys_df = df[["method", "n_collocation", "seed"]].copy()
        keys_df["method"] = keys_df["method"].astype(str).str.lower()
        keys_df["n_collocation"] = keys_df["n_collocation"].astype(int)
        keys_df["seed"] = keys_df["seed"].astype(int)
        observed = set(map(tuple, keys_df.itertuples(index=False, name=None)))
        duplicate_count = int(keys_df.duplicated().sum())

    missing = sorted(expected - observed, key=lambda x: (x[1], x[2], x[0]))
    unexpected = sorted(observed - expected, key=lambda x: (x[1], x[2], x[0]))

    return {
        "expected_runs": len(expected),
        "observed_rows": int(len(df)),
        "unique_run_keys": int(len(observed)),
        "duplicate_key_rows": duplicate_count,
        "missing_run_count": len(missing),
        "unexpected_run_count": len(unexpected),
        "complete": (
            len(observed) == len(expected)
            and duplicate_count == 0
            and not missing
            and not unexpected
        ),
        "missing_runs": [
            {"method": m, "n_collocation": n, "seed": s} for m, n, s in missing
        ],
        "unexpected_runs": [
            {"method": m, "n_collocation": n, "seed": s}
            for m, n, s in unexpected
        ],
    }


def write_manifest(
    output_dir: Path,
    params: v3.ShuttleParams,
    problem: v3.ProblemConfig,
    cfg: v3.TrainConfig,
    methods: Sequence[str],
    n_values: Sequence[int],
    seeds: Sequence[int],
    dry_run: bool,
) -> Path:
    module_path = Path(v3.__file__).resolve()
    runner_path = Path(__file__).resolve()
    manifest = {
        "preset_version": PRESET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "runtime": v3.runtime_info(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "params": {**asdict(params), "a": params.a, "b": params.b},
        "problem": asdict(problem),
        "train_config": asdict(cfg),
        "methods": list(methods),
        "method_labels": {"mc": "MC-R", "lhs": "LHS-R", "alhs": "ALHS-R"},
        "n_values": list(n_values),
        "seeds": list(seeds),
        "expected_runs": len(methods) * len(n_values) * len(seeds),
        "save_models": False,
        "resume": True,
        "module_file": str(module_path),
        "module_sha256": sha256_file(module_path),
        "runner_file": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "fairness_policy": {
            "periodic_refresh_all_methods": True,
            "endpoint_anchors": [problem.t0, problem.tf],
            "fixed_total_collocation_budget": True,
            "lbfgs_union_last_k_sets": cfg.lbfgs_union_sets,
            "physics_validation_checkpointing": True,
            "rk4_used_for_training_or_selection": False,
        },
    }
    path = output_dir / "main_v3_fairness_270_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_main_experiment(
    output_dir: str | Path = "main_v3_fairness_270",
    device: str = "auto",
    dtype: str = "auto",
    resume: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the frozen main experiment, split into one resumable CSV per N_f."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v3.set_runtime(device, dtype)
    params = v3.ShuttleParams()
    problem = v3.ProblemConfig()

    if dry_run:
        cfg = make_dry_config()
        methods = METHODS
        n_values = (30,)
        seeds = (0,)
        n_eval = 201
    else:
        cfg = make_main_config()
        methods = METHODS
        n_values = N_VALUES
        seeds = SEEDS
        n_eval = 2001

    print("=" * 78)
    print("Shuttlecock PINN V3 fairness - main experiment preset")
    print("Preset:", PRESET_VERSION)
    print("Runtime:", v3.runtime_info())
    print("Methods:", methods)
    print("N_f:", n_values)
    print("Seeds:", f"{seeds[0]}..{seeds[-1]}" if len(seeds) > 1 else seeds)
    print("Expected runs:", len(methods) * len(n_values) * len(seeds))
    print("SAVE_MODELS: False")
    print("RESUME:", resume)
    print("Output directory:", output_dir.resolve())
    print("=" * 78)

    manifest_path = write_manifest(
        output_dir, params, problem, cfg, methods, n_values, seeds, dry_run=dry_run
    )

    # Also write the module-native metadata for exact compatibility with V3.
    metadata_path = output_dir / "main_v3_fairness_270.config.json"
    v3.save_run_metadata(
        metadata_path,
        params,
        problem,
        cfg,
        methods=methods,
        n_values=n_values,
        seeds=seeds,
    )

    frames = []
    for budget_index, ncol in enumerate(n_values, start=1):
        print("\n" + "#" * 78)
        print(f"BUDGET BLOCK {budget_index}/{len(n_values)}: N_f={ncol}")
        print("#" * 78)

        raw_csv = output_dir / f"main_v3_fairness_N{ncol}_30seeds.csv"
        # In dry-run mode keep the filename truthful.
        if dry_run:
            raw_csv = output_dir / f"dry_run_v3_fairness_N{ncol}.csv"

        df_n = v3.run_experiment(
            methods=methods,
            n_values=(ncol,),
            seeds=seeds,
            params=params,
            problem=problem,
            base_config=cfg,
            output_csv=raw_csv,
            artifacts_dir=None,
            save_models=False,
            resume=resume,
            n_eval=n_eval,
        )
        frames.append(df_n)

        summary_n = v3.summarize_results(df_n)
        summary_n.to_csv(
            output_dir / f"summary_N{ncol}.csv", index=False
        )

        print(f"Completed rows for N_f={ncol}: {len(df_n)}")
        clear_accelerator_cache()

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = df.sort_values(["n_collocation", "seed", "method"]).reset_index(drop=True)

    combined_csv = output_dir / "main_v3_fairness_270_combined.csv"
    summary_csv = output_dir / "main_v3_fairness_270_summary.csv"
    flags_csv = output_dir / "main_v3_fairness_270_flags.csv"
    stages_csv = output_dir / "main_v3_fairness_270_stage_selection.csv"
    integrity_json = output_dir / "main_v3_fairness_270_integrity.json"

    df.to_csv(combined_csv, index=False)
    v3.summarize_results(df).to_csv(summary_csv, index=False)
    v3.flag_diagnostic_runs(df).to_csv(flags_csv, index=False)
    v3.stage_selection_summary(df).to_csv(stages_csv, index=False)

    report = integrity_report(df, methods, n_values, tuple(seeds))
    report.update(
        {
            "preset_version": PRESET_VERSION,
            "manifest": str(manifest_path),
            "combined_csv": str(combined_csv),
        }
    )
    integrity_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("FINAL INTEGRITY CHECK")
    print(json.dumps({k: v for k, v in report.items() if k not in {"missing_runs", "unexpected_runs"}}, indent=2))
    if report["missing_run_count"]:
        print("Missing runs:", report["missing_runs"][:20])
    if report["unexpected_run_count"]:
        print("Unexpected runs:", report["unexpected_runs"][:20])
    print("\nSaved combined results:", combined_csv)
    print("Saved summary:", summary_csv)
    print("Saved flags:", flags_csv)
    print("Saved stage selection:", stages_csv)
    print("Saved integrity report:", integrity_json)
    print("=" * 78)

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--dtype", default="auto", help="auto, float32, or float64")
    parser.add_argument("--output-dir", default="main_v3_fairness_270")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="start output CSVs from scratch instead of skipping completed run keys",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a tiny 3-method pipeline test instead of the 270-run experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_main_experiment(
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
