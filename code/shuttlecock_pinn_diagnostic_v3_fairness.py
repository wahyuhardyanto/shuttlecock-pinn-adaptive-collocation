"""Diagnostic V3 / fairness PINN workflow for shuttlecock flipping dynamics.

This module implements the fairness corrections identified after the V2
3-method x 5-seed screening.

Core fairness changes
---------------------
1. MC, LHS, and ALHS ALL refresh their collocation sets on the same schedule.
2. t=t0 and t=tf are always included as physics anchor points while keeping
   the total collocation budget fixed.
3. L-BFGS is trained on the de-duplicated union of the last K collocation sets
   (same K for every method), rather than only the final set.
4. A common physics-validation grid, not used for optimizer updates, is used
   for Adam checkpoint selection and post-L-BFGS model selection.
5. RK4 remains evaluation-only. It is never used to choose checkpoints or to
   update model parameters.

The corrected governing ODE is

    phi_tt + a phi_t + b sin(phi) = 0,

with

    a = rho S C_D U / [2 M_B (1 + M_B/M_C)]
    b = rho S C_D U^2 / [2 (M_C + M_B) l_GC].

The network enforces both initial conditions exactly by construction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# 1. Runtime and reproducibility
# -----------------------------------------------------------------------------
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def set_runtime(device_name: str = "auto", dtype_name: str = "auto") -> Tuple[torch.device, torch.dtype]:
    """Configure global PyTorch device and dtype.

    auto device order: CUDA -> MPS -> CPU.
    auto dtype: float32 on CUDA/MPS, float64 on CPU.
    """
    global DEVICE, DTYPE

    name = str(device_name).lower()
    if name == "auto":
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(name)

    dname = str(dtype_name).lower()
    if dname == "auto":
        DTYPE = torch.float32 if DEVICE.type in {"cuda", "mps"} else torch.float64
    elif dname in {"float32", "fp32", "32"}:
        DTYPE = torch.float32
    elif dname in {"float64", "fp64", "64", "double"}:
        if DEVICE.type == "mps":
            raise ValueError("Apple MPS does not support float64 reliably; use float32 or CPU.")
        DTYPE = torch.float64
    else:
        raise ValueError("dtype_name must be auto, float32, or float64")

    return DEVICE, DTYPE


def runtime_info() -> Dict[str, str]:
    return {
        "device": str(DEVICE),
        "dtype": str(DTYPE).replace("torch.", ""),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_runtime("auto", "auto")


# -----------------------------------------------------------------------------
# 2. Physical model and RK4 reference
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ShuttleParams:
    C_D: float = 0.44
    rho: float = 1.2       # kg/m^3
    S: float = 0.0028      # 28 cm^2
    M_B: float = 0.002     # skirt mass, kg
    M_C: float = 0.003     # cork mass, kg
    l_GC: float = 0.02     # m
    U: float = 8.0         # m/s

    @property
    def a(self) -> float:
        return (self.rho * self.S * self.C_D * self.U) / (
            2.0 * self.M_B * (1.0 + self.M_B / self.M_C)
        )

    @property
    def b(self) -> float:
        return (self.rho * self.S * self.C_D * self.U**2) / (
            2.0 * (self.M_C + self.M_B) * self.l_GC
        )


@dataclass(frozen=True)
class ProblemConfig:
    t0: float = 0.0
    tf: float = 0.20
    phi0: float = math.pi - 0.1
    omega0: float = 0.0
    dt_ref: float = 2e-5


@dataclass
class ReferenceSolution:
    t: np.ndarray
    phi: np.ndarray
    omega: np.ndarray


def ode_rhs(t: float, y: np.ndarray, params: ShuttleParams) -> np.ndarray:
    del t
    phi, omega = y
    return np.array(
        [omega, -params.a * omega - params.b * np.sin(phi)],
        dtype=float,
    )


def rk4_step(f, t: float, y: np.ndarray, dt: float, params: ShuttleParams) -> np.ndarray:
    k1 = f(t, y, params)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, params)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, params)
    k4 = f(t + dt, y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(problem: ProblemConfig, params: ShuttleParams) -> ReferenceSolution:
    n = int(round((problem.tf - problem.t0) / problem.dt_ref))
    t = np.linspace(problem.t0, problem.tf, n + 1)
    y = np.zeros((n + 1, 2), dtype=float)
    y[0] = np.array([problem.phi0, problem.omega0], dtype=float)
    for i in range(n):
        y[i + 1] = rk4_step(ode_rhs, t[i], y[i], problem.dt_ref, params)
    return ReferenceSolution(t=t, phi=y[:, 0], omega=y[:, 1])


# -----------------------------------------------------------------------------
# 3. PINN with exact IC enforcement
# -----------------------------------------------------------------------------
class ShuttlePINN(nn.Module):
    """Hard-constrained PINN.

    phi_hat(t) = phi0 + omega0*(t-t0) + (t-t0)^2 N_theta(tau),
    tau in [-1,1].
    """

    def __init__(
        self,
        hidden: int = 32,
        depth: int = 3,
        t0: float = 0.0,
        tf: float = 0.20,
        phi0: float = math.pi - 0.1,
        omega0: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.t0 = float(t0)
        self.tf = float(tf)
        self.phi0 = float(phi0)
        self.omega0 = float(omega0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tau = 2.0 * (t - self.t0) / (self.tf - self.t0) - 1.0
        raw = self.net(tau)
        dt = t - self.t0
        return self.phi0 + self.omega0 * dt + dt**2 * raw


def _ensure_grad_time(t: torch.Tensor) -> torch.Tensor:
    if t.requires_grad:
        return t
    return t.clone().detach().requires_grad_(True)


def derivatives_phi(
    model: ShuttlePINN,
    t: torch.Tensor,
    create_graph_second: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = _ensure_grad_time(t)
    phi = model(t)
    dphi = torch.autograd.grad(
        phi, t, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    ddphi = torch.autograd.grad(
        dphi,
        t,
        torch.ones_like(dphi),
        create_graph=create_graph_second,
        retain_graph=True,
    )[0]
    return t, phi, dphi, ddphi


def physics_residual(
    model: ShuttlePINN,
    t: torch.Tensor,
    params: ShuttleParams,
    create_graph: bool = True,
) -> torch.Tensor:
    _, phi, dphi, ddphi = derivatives_phi(model, t, create_graph_second=create_graph)
    return ddphi + params.a * dphi + params.b * torch.sin(phi)


def physics_loss(model: ShuttlePINN, t: torch.Tensor, params: ShuttleParams) -> torch.Tensor:
    r = physics_residual(model, t, params, create_graph=True)
    return torch.mean((r / params.b) ** 2)


# -----------------------------------------------------------------------------
# 4. Fair sampling: same refresh schedule + fixed endpoint anchors
# -----------------------------------------------------------------------------
def _check_budget(n_total: int) -> int:
    if n_total < 3:
        raise ValueError("n_collocation must be >= 3 because two endpoint anchors are reserved")
    return n_total - 2


def _with_endpoint_anchors(interior: np.ndarray, t0: float, tf: float) -> np.ndarray:
    interior = np.asarray(interior, dtype=float).reshape(-1, 1)
    return np.vstack(([[t0]], interior, [[tf]])).astype(float)


def sample_mc_fair(n_total: int, t0: float, tf: float, rng: np.random.Generator) -> np.ndarray:
    n_interior = _check_budget(n_total)
    interior = rng.uniform(t0, tf, size=(n_interior, 1))
    return _with_endpoint_anchors(interior, t0, tf)


def sample_lhs_fair(n_total: int, t0: float, tf: float, rng: np.random.Generator) -> np.ndarray:
    n_interior = _check_budget(n_total)
    u = (np.arange(n_interior) + rng.random(n_interior)) / n_interior
    rng.shuffle(u)
    interior = (t0 + (tf - t0) * u).reshape(-1, 1)
    return _with_endpoint_anchors(interior, t0, tf)


def adaptive_indicator(
    model: ShuttlePINN,
    grid_np: np.ndarray,
    params: ShuttleParams,
    mode: str = "residual",
) -> np.ndarray:
    t = torch.tensor(
        np.asarray(grid_np).reshape(-1, 1),
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    need_gradient = mode == "residual_gradient"
    r = physics_residual(model, t, params, create_graph=need_gradient)

    if mode == "residual":
        score = torch.abs(r)
    elif mode == "residual_gradient":
        drdt = torch.autograd.grad(
            r, t, torch.ones_like(r), create_graph=False, retain_graph=False
        )[0]
        score = torch.abs(drdt)
    else:
        raise ValueError("mode must be 'residual' or 'residual_gradient'")

    return score.detach().cpu().numpy().ravel()


def sample_adaptive_lhs_fair(
    model: ShuttlePINN,
    n_total: int,
    t0: float,
    tf: float,
    rng: np.random.Generator,
    params: ShuttleParams,
    mode: str = "residual",
    n_grid: int = 2001,
    uniform_mix: float = 0.10,
    power: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-weighted LHS with the same two endpoint anchors as other methods."""
    if not (0.0 < uniform_mix <= 1.0):
        raise ValueError("uniform_mix must be in (0,1]")
    n_interior = _check_budget(n_total)

    grid = np.linspace(t0, tf, n_grid)
    score = adaptive_indicator(model, grid, params, mode=mode)
    score = np.maximum(score, 0.0) ** power

    if np.max(score) <= 1e-15:
        density = np.ones_like(score)
    else:
        scaled = score / (np.mean(score) + 1e-15)
        density = uniform_mix + (1.0 - uniform_mix) * scaled

    dx = np.diff(grid)
    area = 0.5 * (density[:-1] + density[1:]) * dx
    cdf = np.concatenate([[0.0], np.cumsum(area)])
    cdf /= cdf[-1]

    u = (np.arange(n_interior) + rng.random(n_interior)) / n_interior
    rng.shuffle(u)
    interior = np.interp(u, cdf, grid).reshape(-1, 1)
    return _with_endpoint_anchors(interior, t0, tf), grid, density


def refresh_collocation_set(
    method: str,
    model: ShuttlePINN,
    cfg: "TrainConfig",
    problem: ProblemConfig,
    params: ShuttleParams,
    rng: np.random.Generator,
    initial: bool = False,
) -> np.ndarray:
    """Create one collocation set under the common V3 fairness policy.

    ALHS starts from ordinary LHS before a trained residual field exists.
    Thereafter all three methods refresh on the same schedule.
    """
    method = method.lower()
    if method == "mc":
        return sample_mc_fair(cfg.n_collocation, problem.t0, problem.tf, rng)
    if method == "lhs":
        return sample_lhs_fair(cfg.n_collocation, problem.t0, problem.tf, rng)
    if method == "alhs":
        if initial:
            return sample_lhs_fair(cfg.n_collocation, problem.t0, problem.tf, rng)
        samples, _, _ = sample_adaptive_lhs_fair(
            model=model,
            n_total=cfg.n_collocation,
            t0=problem.t0,
            tf=problem.tf,
            rng=rng,
            params=params,
            mode=cfg.alhs_mode,
            n_grid=cfg.alhs_n_grid,
            uniform_mix=cfg.alhs_uniform_mix,
            power=cfg.alhs_power,
        )
        return samples
    raise ValueError("method must be mc, lhs, or alhs")


# -----------------------------------------------------------------------------
# 5. Physics-validation grid and checkpoint helpers
# -----------------------------------------------------------------------------
def make_physics_validation_grid(problem: ProblemConfig, n_points: int = 501) -> np.ndarray:
    """Fixed validation grid not used for optimizer updates.

    The interior locations are cell midpoints rather than the final evaluation
    linspace, reducing overlap between checkpoint selection and final reporting.
    Endpoints are included to monitor terminal physics behavior.
    """
    if n_points < 3:
        raise ValueError("validation grid needs at least 3 points")
    n_interior = n_points - 2
    u = (np.arange(n_interior) + 0.5) / n_interior
    interior = problem.t0 + (problem.tf - problem.t0) * u
    return np.concatenate(([problem.t0], interior, [problem.tf]))


def residual_on_grid(
    model: ShuttlePINN,
    t_np: np.ndarray,
    params: ShuttleParams,
    batch_size: int = 1024,
) -> np.ndarray:
    t_np = np.asarray(t_np, dtype=float).reshape(-1)
    parts = []
    model.eval()
    for start in range(0, len(t_np), batch_size):
        chunk = t_np[start : start + batch_size]
        t = torch.tensor(
            chunk.reshape(-1, 1), dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        r = physics_residual(model, t, params, create_graph=False)
        parts.append(r.detach().cpu().numpy().ravel())
    return np.concatenate(parts)


def physics_validation_loss(
    model: ShuttlePINN,
    validation_grid: np.ndarray,
    params: ShuttleParams,
) -> float:
    r = residual_on_grid(model, validation_grid, params)
    return float(np.mean((r / params.b) ** 2))


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def _unique_union(sample_sets: Sequence[np.ndarray]) -> np.ndarray:
    """De-duplicated 1-D union; repeated endpoint anchors are counted once."""
    if not sample_sets:
        raise ValueError("sample_sets must not be empty")
    values = np.concatenate([np.asarray(x).reshape(-1) for x in sample_sets])
    unique = np.unique(values)
    return unique.reshape(-1, 1)


# -----------------------------------------------------------------------------
# 6. Training
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    n_collocation: int = 100
    epochs: int = 5000
    lr: float = 1e-3
    resample_every: int = 250
    validation_every: int = 250
    validation_points: int = 501
    hidden: int = 32
    depth: int = 3
    alhs_mode: str = "residual"
    alhs_n_grid: int = 2001
    alhs_uniform_mix: float = 0.10
    alhs_power: float = 1.0
    lbfgs_max_iter: int = 300
    lbfgs_union_sets: int = 4
    restore_best_adam_before_lbfgs: bool = True
    select_final_by_validation: bool = True
    print_every: int = 0


@dataclass
class TrainResult:
    model: ShuttlePINN
    loss_history: np.ndarray
    validation_history: pd.DataFrame
    sample_snapshots: Dict[int, np.ndarray]
    loss_adam_final_trainset: float
    adam_final_val_loss: float
    adam_best_val_loss: float
    adam_best_epoch: int
    lbfgs_n_points: int
    loss_lbfgs_union_before: float
    loss_lbfgs_union_after: float
    lbfgs_val_loss_after: float
    selected_stage: str
    selected_val_loss: float
    resample_count: int
    runtime_sampling_s: float
    runtime_validation_s: float
    runtime_adam_s: float
    runtime_lbfgs_s: float
    runtime_total_s: float


def _loss_on_points(model: ShuttlePINN, t_np: np.ndarray, params: ShuttleParams) -> float:
    t = torch.tensor(
        np.asarray(t_np).reshape(-1, 1),
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    model.eval()
    return float(physics_loss(model, t, params).detach().cpu())


def train_one(
    method: str,
    seed: int,
    params: ShuttleParams,
    problem: ProblemConfig,
    cfg: TrainConfig,
) -> TrainResult:
    """Train one fair-resampling MC/LHS/ALHS run."""
    method = method.lower()
    if method not in {"mc", "lhs", "alhs"}:
        raise ValueError("method must be mc, lhs, or alhs")
    if cfg.resample_every <= 0 or cfg.validation_every <= 0:
        raise ValueError("resample_every and validation_every must be positive")
    if cfg.lbfgs_union_sets < 1:
        raise ValueError("lbfgs_union_sets must be >= 1")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    total_start = time.perf_counter()
    model = ShuttlePINN(
        hidden=cfg.hidden,
        depth=cfg.depth,
        t0=problem.t0,
        tf=problem.tf,
        phi0=problem.phi0,
        omega0=problem.omega0,
    ).to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    validation_grid = make_physics_validation_grid(problem, cfg.validation_points)

    sampling_timer = 0.0
    validation_timer = 0.0
    t0_sample = time.perf_counter()
    t_np = refresh_collocation_set(
        method, model, cfg, problem, params, rng, initial=True
    )
    sampling_timer += time.perf_counter() - t0_sample

    recent_sets = deque(maxlen=cfg.lbfgs_union_sets)
    recent_sets.append(t_np.copy())
    sample_snapshots: Dict[int, np.ndarray] = {0: np.sort(t_np.ravel())}
    loss_history = []
    val_records = []
    resample_count = 0

    best_adam_state = None
    best_adam_val = float("inf")
    best_adam_epoch = -1

    adam_start = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        # A common refresh schedule for MC, LHS, and ALHS.
        # The refreshed set is used for the next block of epochs.
        if ep > 1 and ((ep - 1) % cfg.resample_every == 0):
            ts = time.perf_counter()
            model.eval()
            t_np = refresh_collocation_set(
                method, model, cfg, problem, params, rng, initial=False
            )
            sampling_timer += time.perf_counter() - ts
            recent_sets.append(t_np.copy())
            sample_snapshots[ep - 1] = np.sort(t_np.ravel())
            resample_count += 1

        t_col = torch.tensor(t_np, dtype=DTYPE, device=DEVICE, requires_grad=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = physics_loss(model, t_col, params)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))

        needs_validation = (ep % cfg.validation_every == 0) or (ep == cfg.epochs)
        if needs_validation:
            tv = time.perf_counter()
            val_loss = physics_validation_loss(model, validation_grid, params)
            validation_timer += time.perf_counter() - tv
            val_records.append({"epoch": ep, "physics_val_loss": val_loss})
            if val_loss < best_adam_val:
                best_adam_val = val_loss
                best_adam_epoch = ep
                best_adam_state = _clone_state_dict(model)

        if cfg.print_every and (ep == 1 or ep % cfg.print_every == 0):
            latest_val = val_records[-1]["physics_val_loss"] if val_records else float("nan")
            print(
                f"{method.upper():4s} seed={seed:3d} epoch={ep:5d}/{cfg.epochs} "
                f"L_train={loss_history[-1]:.3e} L_val={latest_val:.3e}"
            )

    runtime_adam_s = time.perf_counter() - adam_start

    # Final Adam diagnostics BEFORE restoring the best checkpoint.
    loss_adam_final_trainset = _loss_on_points(model, t_np, params)
    tv = time.perf_counter()
    adam_final_val_loss = physics_validation_loss(model, validation_grid, params)
    validation_timer += time.perf_counter() - tv

    if best_adam_state is None:
        best_adam_state = _clone_state_dict(model)
        best_adam_val = adam_final_val_loss
        best_adam_epoch = cfg.epochs

    # Build the same type of last-K-set union for every method.
    lbfgs_points = _unique_union(list(recent_sets))
    lbfgs_n_points = int(lbfgs_points.shape[0])

    if cfg.restore_best_adam_before_lbfgs:
        _load_state_dict(model, best_adam_state)

    loss_lbfgs_union_before = _loss_on_points(model, lbfgs_points, params)

    runtime_lbfgs_s = 0.0
    if cfg.lbfgs_max_iter > 0:
        t_lbfgs = torch.tensor(
            lbfgs_points, dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=cfg.lbfgs_max_iter,
            tolerance_grad=1e-10 if DTYPE == torch.float64 else 1e-7,
            tolerance_change=1e-12 if DTYPE == torch.float64 else 1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            value = physics_loss(model, t_lbfgs, params)
            value.backward()
            return value

        tl = time.perf_counter()
        lbfgs.step(closure)
        runtime_lbfgs_s = time.perf_counter() - tl

    loss_lbfgs_union_after = _loss_on_points(model, lbfgs_points, params)
    tv = time.perf_counter()
    lbfgs_val_loss_after = physics_validation_loss(model, validation_grid, params)
    validation_timer += time.perf_counter() - tv
    lbfgs_state = _clone_state_dict(model)

    if cfg.select_final_by_validation:
        if lbfgs_val_loss_after <= best_adam_val:
            selected_stage = "lbfgs"
            selected_val_loss = lbfgs_val_loss_after
            _load_state_dict(model, lbfgs_state)
        else:
            selected_stage = "adam_best"
            selected_val_loss = best_adam_val
            _load_state_dict(model, best_adam_state)
    else:
        selected_stage = "lbfgs" if cfg.lbfgs_max_iter > 0 else "adam_final"
        selected_val_loss = lbfgs_val_loss_after
        _load_state_dict(model, lbfgs_state)

    runtime_total_s = time.perf_counter() - total_start
    sample_snapshots[cfg.epochs] = np.sort(t_np.ravel())

    return TrainResult(
        model=model,
        loss_history=np.asarray(loss_history, dtype=float),
        validation_history=pd.DataFrame(val_records),
        sample_snapshots=sample_snapshots,
        loss_adam_final_trainset=loss_adam_final_trainset,
        adam_final_val_loss=adam_final_val_loss,
        adam_best_val_loss=best_adam_val,
        adam_best_epoch=best_adam_epoch,
        lbfgs_n_points=lbfgs_n_points,
        loss_lbfgs_union_before=loss_lbfgs_union_before,
        loss_lbfgs_union_after=loss_lbfgs_union_after,
        lbfgs_val_loss_after=lbfgs_val_loss_after,
        selected_stage=selected_stage,
        selected_val_loss=selected_val_loss,
        resample_count=resample_count,
        runtime_sampling_s=sampling_timer,
        runtime_validation_s=validation_timer,
        runtime_adam_s=runtime_adam_s,
        runtime_lbfgs_s=runtime_lbfgs_s,
        runtime_total_s=runtime_total_s,
    )


# -----------------------------------------------------------------------------
# 7. Dense diagnostic evaluation against RK4 (evaluation only)
# -----------------------------------------------------------------------------
def _safe_rel_l2(error: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.linalg.norm(truth))
    return float(np.linalg.norm(error) / denom) if denom > 0 else float("nan")


def error_metrics(truth: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - truth
    abs_err = np.abs(err)
    return {
        f"{prefix}_mse": float(np.mean(err**2)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_rel_l2": _safe_rel_l2(err, truth),
        f"{prefix}_linf": float(np.max(abs_err)),
    }


def predict_phi_omega(
    model: ShuttlePINN,
    t_np: np.ndarray,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    t_np = np.asarray(t_np, dtype=float).reshape(-1)
    phi_parts = []
    omega_parts = []
    model.eval()
    for start in range(0, len(t_np), batch_size):
        chunk = t_np[start : start + batch_size]
        t = torch.tensor(
            chunk.reshape(-1, 1), dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        phi = model(t)
        omega = torch.autograd.grad(
            phi, t, torch.ones_like(phi), create_graph=False, retain_graph=False
        )[0]
        phi_parts.append(phi.detach().cpu().numpy().ravel())
        omega_parts.append(omega.detach().cpu().numpy().ravel())
    return np.concatenate(phi_parts), np.concatenate(omega_parts)


def hard_ic_diagnostics(model: ShuttlePINN, problem: ProblemConfig) -> Dict[str, float]:
    t = torch.tensor([[problem.t0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    phi = model(t)
    omega = torch.autograd.grad(phi, t, torch.ones_like(phi), create_graph=False)[0]
    return {
        "phi0_pred": float(phi.detach().cpu()),
        "omega0_pred": float(omega.detach().cpu()),
        "phi0_abs_error": abs(float(phi.detach().cpu()) - problem.phi0),
        "omega0_abs_error": abs(float(omega.detach().cpu()) - problem.omega0),
    }


def evaluate_model(
    model: ShuttlePINN,
    params: ShuttleParams,
    problem: ProblemConfig,
    reference: ReferenceSolution,
    n_eval: int = 2001,
    batch_size: int = 1024,
    return_curves: bool = False,
):
    t_eval = np.linspace(problem.t0, problem.tf, n_eval)
    phi_true = np.interp(t_eval, reference.t, reference.phi)
    omega_true = np.interp(t_eval, reference.t, reference.omega)
    phi_pred, omega_pred = predict_phi_omega(model, t_eval, batch_size=batch_size)
    r_raw = residual_on_grid(model, t_eval, params, batch_size=batch_size)
    r_norm = r_raw / params.b

    metrics: Dict[str, float] = {}
    metrics.update(error_metrics(phi_true, phi_pred, "phi"))
    metrics.update(error_metrics(omega_true, omega_pred, "omega"))

    phi_abs = np.abs(phi_pred - phi_true)
    omega_abs = np.abs(omega_pred - omega_true)
    r_abs = np.abs(r_norm)
    metrics.update(
        {
            "phi_linf_t": float(t_eval[int(np.argmax(phi_abs))]),
            "omega_linf_t": float(t_eval[int(np.argmax(omega_abs))]),
            "residual_rms_raw": float(np.sqrt(np.mean(r_raw**2))),
            "residual_mae_raw": float(np.mean(np.abs(r_raw))),
            "residual_rms_norm": float(np.sqrt(np.mean(r_norm**2))),
            "residual_mae_norm": float(np.mean(r_abs)),
            "residual_p95_norm": float(np.quantile(r_abs, 0.95)),
            "residual_p99_norm": float(np.quantile(r_abs, 0.99)),
            "residual_max_norm": float(np.max(r_abs)),
            "residual_max_t": float(t_eval[int(np.argmax(r_abs))]),
            "dense_physics_loss": float(np.mean(r_norm**2)),
        }
    )
    metrics.update(hard_ic_diagnostics(model, problem))

    if not return_curves:
        return metrics

    curves = pd.DataFrame(
        {
            "t": t_eval,
            "phi_true": phi_true,
            "phi_pred": phi_pred,
            "omega_true": omega_true,
            "omega_pred": omega_pred,
            "residual_raw": r_raw,
            "residual_norm": r_norm,
        }
    )
    return metrics, curves


# -----------------------------------------------------------------------------
# 8. Repeated experiment, artifacts, metadata
# -----------------------------------------------------------------------------
def _save_run_artifacts(
    result: TrainResult,
    method: str,
    n_collocation: int,
    seed: int,
    params: ShuttleParams,
    problem: ProblemConfig,
    cfg: TrainConfig,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_N{n_collocation}_seed{seed}"

    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "method": method,
            "n_collocation": n_collocation,
            "seed": seed,
            "params": asdict(params),
            "problem": asdict(problem),
            "train_config": asdict(cfg),
            "runtime": runtime_info(),
            "selected_stage": result.selected_stage,
            "selected_val_loss": result.selected_val_loss,
            "adam_best_epoch": result.adam_best_epoch,
        },
        artifacts_dir / f"{stem}.pt",
    )

    np.savez_compressed(
        artifacts_dir / f"{stem}_samples.npz",
        **{f"epoch_{ep}": pts for ep, pts in result.sample_snapshots.items()},
    )
    result.validation_history.to_csv(
        artifacts_dir / f"{stem}_validation_history.csv", index=False
    )


def save_run_metadata(
    path: str | Path,
    params: ShuttleParams,
    problem: ProblemConfig,
    cfg: TrainConfig,
    methods: Sequence[str],
    n_values: Sequence[int],
    seeds: Sequence[int],
) -> None:
    payload = {
        "runtime": runtime_info(),
        "params": {**asdict(params), "a": params.a, "b": params.b},
        "problem": asdict(problem),
        "base_train_config": asdict(cfg),
        "methods": list(methods),
        "method_labels": {"mc": "MC-R", "lhs": "LHS-R", "alhs": "ALHS-R"},
        "fairness_policy": {
            "periodic_refresh_all_methods": True,
            "endpoint_anchors": [problem.t0, problem.tf],
            "fixed_total_collocation_budget": True,
            "lbfgs_union_last_k_sets": cfg.lbfgs_union_sets,
            "physics_validation_checkpointing": True,
            "rk4_used_for_training_or_selection": False,
        },
        "n_values": list(n_values),
        "seeds": list(seeds),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(
    methods: Sequence[str] = ("mc", "lhs", "alhs"),
    n_values: Sequence[int] = (100,),
    seeds: Iterable[int] = range(5),
    params: Optional[ShuttleParams] = None,
    problem: Optional[ProblemConfig] = None,
    base_config: Optional[TrainConfig] = None,
    output_csv: str | Path = "screening_diagnostic_v3_fairness_3x5.csv",
    artifacts_dir: Optional[str | Path] = None,
    save_models: bool = False,
    resume: bool = True,
    n_eval: int = 2001,
) -> pd.DataFrame:
    params = params or ShuttleParams()
    problem = problem or ProblemConfig()
    base_config = base_config or TrainConfig()
    methods = tuple(m.lower() for m in methods)
    n_values = tuple(int(n) for n in n_values)
    seeds = tuple(int(s) for s in seeds)
    output_csv = Path(output_csv)
    artifact_path = Path(artifacts_dir) if artifacts_dir is not None else None

    reference = solve_rk4(problem, params)

    if resume and output_csv.exists():
        df_old = pd.read_csv(output_csv)
        rows = df_old.to_dict("records")
        completed = {
            (str(r["method"]).lower(), int(r["n_collocation"]), int(r["seed"]))
            for r in rows
            if "method" in r and "n_collocation" in r and "seed" in r
        }
        print(f"Resume: found {len(completed)} completed runs in {output_csv}")
    else:
        rows = []
        completed = set()

    total_requested = len(methods) * len(n_values) * len(seeds)
    run_counter = 0

    for ncol in n_values:
        for seed in seeds:
            for method in methods:
                run_counter += 1
                key = (method, ncol, seed)
                if key in completed:
                    print(f"[{run_counter}/{total_requested}] SKIP {method} N={ncol} seed={seed}")
                    continue

                cfg = replace(base_config, n_collocation=ncol)
                print(
                    f"[{run_counter}/{total_requested}] RUN method={method} N={ncol} seed={seed} "
                    f"epochs={cfg.epochs} refresh={cfg.resample_every} LBFGS={cfg.lbfgs_max_iter}"
                )

                result = train_one(method, seed, params, problem, cfg)
                metrics = evaluate_model(
                    result.model, params, problem, reference, n_eval=n_eval, return_curves=False
                )

                row = {
                    "method": method,
                    "method_label": {"mc": "MC-R", "lhs": "LHS-R", "alhs": "ALHS-R"}[method],
                    "n_collocation": ncol,
                    "seed": seed,
                    "epochs": cfg.epochs,
                    "resample_every": cfg.resample_every,
                    "validation_every": cfg.validation_every,
                    "validation_points": cfg.validation_points,
                    "lbfgs_union_sets": cfg.lbfgs_union_sets,
                    "alhs_mode": cfg.alhs_mode if method == "alhs" else "NA",
                    "device": str(DEVICE),
                    "dtype": str(DTYPE).replace("torch.", ""),
                    "resample_count": result.resample_count,
                    "loss_adam_final_trainset": result.loss_adam_final_trainset,
                    "adam_final_val_loss": result.adam_final_val_loss,
                    "adam_best_val_loss": result.adam_best_val_loss,
                    "adam_best_epoch": result.adam_best_epoch,
                    "lbfgs_n_points": result.lbfgs_n_points,
                    "loss_lbfgs_union_before": result.loss_lbfgs_union_before,
                    "loss_lbfgs_union_after": result.loss_lbfgs_union_after,
                    "lbfgs_val_loss_after": result.lbfgs_val_loss_after,
                    "selected_stage": result.selected_stage,
                    "selected_val_loss": result.selected_val_loss,
                    "runtime_sampling_s": result.runtime_sampling_s,
                    "runtime_validation_s": result.runtime_validation_s,
                    "runtime_adam_s": result.runtime_adam_s,
                    "runtime_lbfgs_s": result.runtime_lbfgs_s,
                    "runtime_total_s": result.runtime_total_s,
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_csv, index=False)

                if save_models and artifact_path is not None:
                    _save_run_artifacts(
                        result, method, ncol, seed, params, problem, cfg, artifact_path
                    )

                print(
                    "    "
                    f"stage={result.selected_stage:9s} "
                    f"phi_L2={metrics['phi_rel_l2']:.3e} "
                    f"omega_L2={metrics['omega_rel_l2']:.3e} "
                    f"Rrms={metrics['residual_rms_norm']:.3e} "
                    f"L_dense={metrics['dense_physics_loss']:.3e} "
                    f"time={result.runtime_total_s:.1f}s"
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["n_collocation", "seed", "method"]).reset_index(drop=True)
        df.to_csv(output_csv, index=False)
    return df


def summarize_results(
    df: pd.DataFrame,
    metrics: Sequence[str] = (
        "phi_rel_l2",
        "omega_rel_l2",
        "residual_rms_norm",
        "residual_p99_norm",
        "dense_physics_loss",
        "selected_val_loss",
        "runtime_total_s",
        "runtime_sampling_s",
    ),
) -> pd.DataFrame:
    out = []
    for (method, ncol), g in df.groupby(["method", "n_collocation"]):
        for metric in metrics:
            if metric not in g.columns:
                continue
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            if x.empty:
                continue
            out.append(
                {
                    "method": method,
                    "n_collocation": int(ncol),
                    "metric": metric,
                    "count": int(x.size),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
                    "median": float(x.median()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }
            )
    return pd.DataFrame(out)


def flag_diagnostic_runs(
    df: pd.DataFrame,
    phi_rel_l2_max: float = 0.01,
    residual_rms_norm_min: float = 0.05,
) -> pd.DataFrame:
    mask = (
        (pd.to_numeric(df["phi_rel_l2"], errors="coerce") <= phi_rel_l2_max)
        & (pd.to_numeric(df["residual_rms_norm"], errors="coerce") >= residual_rms_norm_min)
    )
    cols = [
        "method",
        "method_label",
        "n_collocation",
        "seed",
        "selected_stage",
        "phi_rel_l2",
        "omega_rel_l2",
        "residual_rms_norm",
        "residual_p99_norm",
        "residual_max_norm",
        "residual_max_t",
        "adam_best_val_loss",
        "lbfgs_val_loss_after",
        "selected_val_loss",
        "dense_physics_loss",
    ]
    return df.loc[mask, [c for c in cols if c in df.columns]].copy()


def stage_selection_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "selected_stage" not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(["method", "n_collocation", "selected_stage"])
        .size()
        .rename("count")
        .reset_index()
    )
    return out


# -----------------------------------------------------------------------------
# 9. CLI
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "screening"], default="smoke")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--dtype", default="auto", help="auto, float32, or float64")
    parser.add_argument("--output", default="screening_diagnostic_v3_fairness_3x5.csv")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--artifacts-dir", default="screening_diagnostic_v3_fairness_artifacts")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_runtime(args.device, args.dtype)

    params = ShuttleParams()
    problem = ProblemConfig()
    print("Runtime:", runtime_info())
    print("Parameters:", {**asdict(params), "a": params.a, "b": params.b})
    print("Problem:", asdict(problem))

    if args.mode == "smoke":
        cfg = TrainConfig(
            n_collocation=30,
            epochs=30,
            lr=1e-3,
            resample_every=10,
            validation_every=10,
            validation_points=101,
            hidden=16,
            depth=2,
            lbfgs_max_iter=5,
            lbfgs_union_sets=2,
            print_every=10,
        )
        df = run_experiment(
            methods=("mc", "lhs", "alhs"),
            n_values=(30,),
            seeds=(0,),
            params=params,
            problem=problem,
            base_config=cfg,
            output_csv=args.output,
            artifacts_dir=args.artifacts_dir,
            save_models=args.save_models,
            resume=not args.no_resume,
            n_eval=201,
        )
    else:
        cfg = TrainConfig(
            n_collocation=100,
            epochs=5000,
            lr=1e-3,
            resample_every=250,
            validation_every=250,
            validation_points=501,
            hidden=32,
            depth=3,
            alhs_mode="residual",
            lbfgs_max_iter=300,
            lbfgs_union_sets=4,
            restore_best_adam_before_lbfgs=True,
            select_final_by_validation=True,
            print_every=0,
        )
        metadata_path = Path(args.output).with_suffix(".config.json")
        save_run_metadata(
            metadata_path,
            params,
            problem,
            cfg,
            methods=("mc", "lhs", "alhs"),
            n_values=(100,),
            seeds=range(5),
        )
        df = run_experiment(
            methods=("mc", "lhs", "alhs"),
            n_values=(100,),
            seeds=range(5),
            params=params,
            problem=problem,
            base_config=cfg,
            output_csv=args.output,
            artifacts_dir=args.artifacts_dir,
            save_models=args.save_models,
            resume=not args.no_resume,
            n_eval=2001,
        )

    print("\nResults:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summarize_results(df).to_string(index=False))
    flagged = flag_diagnostic_runs(df)
    if not flagged.empty:
        print("\nFlagged trajectory-good / physics-poor runs:")
        print(flagged.to_string(index=False))
    print("\nSelected-stage counts:")
    print(stage_selection_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
