# Data dictionary

The primary run-level file is `data/main_v3_fairness_270_combined.csv`. Each row is one `(method, collocation budget, seed)` run.

## Identification and design

- `method` - internal method key: `mc`, `lhs`, or `alhs`.
- `method_label` - publication label: MC-R, LHS-R, or ALHS-R.
- `n_collocation` - total collocation budget, including two endpoint physics anchors.
- `seed` - paired random seed.
- `epochs` - Adam epoch count.
- `resample_every` - common collocation refresh interval.
- `validation_every` - physics-validation interval.
- `validation_points` - number of fixed validation points.
- `lbfgs_union_sets` - number of recent collocation sets unioned for L-BFGS.
- `alhs_mode` - adaptive indicator mode for ALHS-R; blank for nonadaptive baselines.
- `device`, `dtype` - computational device and floating-point precision.
- `resample_count` - number of periodic resampling events.

## Optimization and checkpoint selection

- `loss_adam_final_trainset` - normalized physics loss on the final Adam training set.
- `adam_final_val_loss` - validation physics loss at the final Adam epoch.
- `adam_best_val_loss` - best physics-validation loss observed during Adam.
- `adam_best_epoch` - epoch of the best Adam checkpoint.
- `lbfgs_n_points` - number of unique collocation points used by L-BFGS.
- `loss_lbfgs_union_before`, `loss_lbfgs_union_after` - normalized physics loss on the L-BFGS union set before/after refinement.
- `lbfgs_val_loss_after` - validation physics loss after L-BFGS.
- `selected_stage` - final accepted stage (`adam_best` or `lbfgs`).
- `selected_val_loss` - validation loss of the accepted model.

## Runtime

- `runtime_sampling_s` - time spent generating collocation samples.
- `runtime_validation_s` - validation-grid evaluation time.
- `runtime_adam_s` - Adam training time.
- `runtime_lbfgs_s` - L-BFGS refinement time.
- `runtime_total_s` - total measured training workflow time before final reference diagnostics.

## Trajectory-error metrics

- `phi_mse`, `phi_rmse` - angle mean-squared and root-mean-squared errors versus RK4 reference.
- `phi_rel_l2` - relative L2 angle error; primary trajectory metric.
- `phi_linf` - angle L-infinity error.
- `omega_mse`, `omega_rmse` - angular-velocity mean-squared and root-mean-squared errors.
- `omega_rel_l2` - relative L2 angular-velocity error.
- `omega_linf` - angular-velocity L-infinity error.
- `phi_linf_t`, `omega_linf_t` - times at which the corresponding maximum absolute errors occur.

## Physics-residual metrics

- `residual_rms_raw`, `residual_mae_raw` - RMS and MAE of the unnormalized ODE residual.
- `residual_rms_norm`, `residual_mae_norm` - RMS and MAE after normalization by coefficient `b`.
- `residual_p95_norm`, `residual_p99_norm`, `residual_max_norm` - upper quantiles and maximum normalized residual.
- `residual_max_t` - time at which the maximum normalized residual occurs.
- `dense_physics_loss` - mean squared normalized physics residual on the common dense evaluation grid.

## Initial-condition diagnostics

- `phi0_pred`, `omega0_pred` - predicted initial angle and angular velocity.
- `phi0_abs_error`, `omega0_abs_error` - absolute initial-condition errors; the hard output transform makes these numerically negligible.

## Other files

- `main_v3_fairness_270_summary.csv` - descriptive summaries by method and collocation budget.
- `main_v3_fairness_270_flags.csv` - diagnostic flag output from the frozen pipeline.
- `main_v3_fairness_270_stage_selection.csv` - counts of selected Adam-best vs L-BFGS checkpoints.
- `main_v3_fairness_270_integrity.json` - completeness and uniqueness checks.
- `main_v3_fairness_270_manifest_public.json` - sanitized experiment configuration, runtime versions, hashes, and fairness policy.
