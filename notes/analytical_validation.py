"""Analytical validation of the Crank-Nicolson reference solver and the PINN.

Two benchmark problems with exact closed-form solutions:

  Benchmark 1 (steady state, the REAL boundary conditions):
      constant flux q on the hot side, Robin convection on the coolant side.
      Exact:  T(x) = T_cool + q/h_cool + q*(L - x)/k
      Tests: spatial discretization + the actual Neumann/Robin ghost-point BCs.

  Benchmark 2 (fully transient, insulated slab, single cosine mode):
      q_hot = 0 and h_cool = 0 (zero flux both sides),
      initial condition T(x,0) = T_mean + A*cos(pi*x/L).
      Exact:  T(x,t) = T_mean + A*cos(pi*x/L)*exp(-alpha*(pi/L)^2 * t)
      Tests: the time-stepping, and (separately) the PINN implementation.

Run from the repo root:  PYTHONPATH=src python analytical_validation.py
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_pinn.config import ExperimentConfig
from thermal_pinn.reference import generate_reference_solution, _second_derivative_matrix, _boundary_source
from thermal_pinn.model import pinn_model
from thermal_pinn.losses import pde_residual, boundary_losses

RESULTS: dict[str, float] = {}

# ----------------------------------------------------------------------------
# Benchmark 1: steady state with the real (Neumann + Robin) boundary conditions
# ----------------------------------------------------------------------------

def benchmark1_steady_state():
    cfg = ExperimentConfig()
    cfg.q_amp = 0.0          # constant flux -> a steady state exists
    cfg.t_final = 50.0       # long enough to fully settle (time scale L^2/alpha ~ 0.22 s)
    cfg.nt = 2001
    ref = generate_reference_solution(cfg)

    q = cfg.q_base
    T_exact = cfg.T_cool + q / cfg.h_cool + q * (cfg.L_wall - ref.x) / cfg.k
    err = np.abs(ref.T[-1, :] - T_exact)
    RESULTS["b1_max_err_K"] = float(err.max())
    return cfg, ref.x, ref.T[-1, :], T_exact


# ----------------------------------------------------------------------------
# Benchmark 2: transient cosine mode in an insulated slab
# ----------------------------------------------------------------------------

def make_cfg_b2() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.q_base = 0.0
    cfg.q_amp = 0.0
    cfg.h_cool = 0.0          # zero flux on both sides -> insulated slab
    cfg.T_cool = 300.0        # only used for nondimensionalization here
    cfg.t_final = 0.05        # decay rate alpha*(pi/L)^2 ~ 44 1/s -> e^-2.2 by t_final
    return cfg


def exact_b2(cfg: ExperimentConfig, x: np.ndarray, t: np.ndarray,
             T_mean: float = 400.0, A: float = 100.0) -> np.ndarray:
    lam = cfg.alpha * (np.pi / cfg.L_wall) ** 2
    X, TT = np.meshgrid(x, t)
    return T_mean + A * np.cos(np.pi * X / cfg.L_wall) * np.exp(-lam * TT)


def cn_solve_with_ic(cfg: ExperimentConfig, T_init: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The repo's Crank-Nicolson scheme, with a custom initial profile.

    Identical matrices/BC handling to reference.generate_reference_solution;
    only the initial condition differs (the repo version hardcodes uniform T0).
    """
    x = np.linspace(0.0, cfg.L_wall, cfg.nx)
    t = np.linspace(0.0, cfg.t_final, cfg.nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    A = _second_derivative_matrix(cfg, dx)
    I = np.eye(cfg.nx)
    lhs = I - 0.5 * dt * cfg.alpha * A
    rhs_matrix = I + 0.5 * dt * cfg.alpha * A

    T = np.empty((cfg.nt, cfg.nx))
    T[0, :] = T_init
    for n in range(cfg.nt - 1):
        b_now = _boundary_source(cfg, float(t[n]), dx)
        b_next = _boundary_source(cfg, float(t[n + 1]), dx)
        rhs = rhs_matrix @ T[n, :] + 0.5 * dt * cfg.alpha * (b_now + b_next)
        T[n + 1, :] = np.linalg.solve(lhs, rhs)
    return x, t, T


def benchmark2_cn():
    cfg = make_cfg_b2()
    cfg.nx, cfg.nt = 81, 301
    x = np.linspace(0.0, cfg.L_wall, cfg.nx)
    T_init = 400.0 + 100.0 * np.cos(np.pi * x / cfg.L_wall)
    x, t, T_num = cn_solve_with_ic(cfg, T_init)
    T_ex = exact_b2(cfg, x, t)
    err = np.abs(T_num - T_ex)
    RESULTS["b2_cn_max_err_K"] = float(err.max())
    rel = np.linalg.norm(T_num - T_ex) / np.linalg.norm(T_ex - 400.0)
    RESULTS["b2_cn_rel_l2"] = float(rel)
    return cfg, x, t, T_num, T_ex


def benchmark2_grid_convergence():
    """Halve dx and dt together; CN is 2nd order so error should drop ~4x per step."""
    rows = []
    for nx, nt in ((11, 26), (21, 51), (41, 101), (81, 201), (161, 401)):
        cfg = make_cfg_b2()
        cfg.nx, cfg.nt = nx, nt
        x = np.linspace(0.0, cfg.L_wall, nx)
        T_init = 400.0 + 100.0 * np.cos(np.pi * x / cfg.L_wall)
        x, t, T_num = cn_solve_with_ic(cfg, T_init)
        T_ex = exact_b2(cfg, x, t)
        rows.append((nx - 1, float(np.max(np.abs(T_num - T_ex)))))
    orders = [np.log2(rows[i - 1][1] / rows[i][1]) for i in range(1, len(rows))]
    RESULTS["b2_cn_observed_order"] = float(np.mean(orders[-2:]))
    return rows, orders


# ----------------------------------------------------------------------------
# PINN validation on Benchmark 2
# ----------------------------------------------------------------------------

def benchmark2_pinn(seed: int = 7, adam_epochs: int = 5000):
    cfg = make_cfg_b2()
    # network/training settings: the repo's "full"-size defaults
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = pinn_model(cfg)

    # Exact solution in nondimensional variables:
    #   theta(x_hat, t_hat) = 1 + cos(pi x_hat) exp(-beta pi^2 t_hat)
    # (T_mean=400, A=100, T_cool=300, delta_T=100)
    def theta0(x_hat: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.cos(np.pi * x_hat)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(adam_epochs):
        opt.zero_grad(set_to_none=True)
        x_p, t_p = torch.rand(cfg.n_pde, 1), torch.rand(cfg.n_pde, 1)
        r = pde_residual(model, x_p, t_p, cfg)
        loss_pde = torch.mean(r ** 2)
        t_b = torch.rand(cfg.n_bc, 1)
        lh, lc = boundary_losses(model, t_b, cfg)   # with q=0, h=0 these are zero-flux BCs
        x_i = torch.rand(cfg.n_ic, 1)
        loss_ic = torch.mean((model(x_i, torch.zeros_like(x_i)) - theta0(x_i)) ** 2)
        loss = cfg.w_pde * loss_pde + cfg.w_bc * (lh + lc) + cfg.w_ic * loss_ic
        loss.backward()
        opt.step()

    # LBFGS polish on a fixed collocation set
    torch.manual_seed(0)
    x_p, t_p = torch.rand(2000, 1), torch.rand(2000, 1)
    t_b, x_i = torch.rand(400, 1), torch.rand(400, 1)
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=300, history_size=50,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        r = pde_residual(model, x_p, t_p, cfg)
        lh, lc = boundary_losses(model, t_b, cfg)
        li = torch.mean((model(x_i, torch.zeros_like(x_i)) - theta0(x_i)) ** 2)
        loss = cfg.w_pde * torch.mean(r ** 2) + cfg.w_bc * (lh + lc) + cfg.w_ic * li
        loss.backward()
        return loss

    lbfgs.step(closure)

    # Evaluate against the exact solution on a fine grid
    x = np.linspace(0.0, cfg.L_wall, 201)
    t = np.linspace(0.0, cfg.t_final, 201)
    T_ex = exact_b2(cfg, x, t)
    X, TT = np.meshgrid(x / cfg.L_wall, t / cfg.t_final)
    pts = torch.as_tensor(np.stack([X.ravel(), TT.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        theta = model(pts[:, 0:1], pts[:, 1:2]).numpy().reshape(T_ex.shape)
    T_pinn = cfg.T_cool + cfg.delta_T * theta

    err = T_pinn - T_ex
    RESULTS["b2_pinn_rel_l2"] = float(np.linalg.norm(err) / np.linalg.norm(T_ex - 400.0))
    RESULTS["b2_pinn_max_err_K"] = float(np.max(np.abs(err)))
    return cfg, x, t, T_pinn, T_ex


# ----------------------------------------------------------------------------

def main():
    cfg1, x1, T_cn1, T_ex1 = benchmark1_steady_state()
    cfg2, x2, t2, T_cn2, T_ex2 = benchmark2_cn()
    rows, orders = benchmark2_grid_convergence()
    cfgp, xp_, tp_, T_pinn, T_exp = benchmark2_pinn()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(x1 * 1e3, T_ex1 - 273.15, "k-", lw=2, label="exact")
    ax.plot(x1 * 1e3, T_cn1 - 273.15, "o", ms=4, mfc="none", color="tab:blue",
            markevery=5, label="Crank-Nicolson")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("T (C)")
    ax.set_title(f"(a) Benchmark 1: steady state, real BCs\nmax error = {RESULTS['b1_max_err_K']:.1e} K")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for frac, c in ((0.0, "tab:blue"), (0.2, "tab:orange"), (0.5, "tab:green"), (1.0, "tab:red")):
        i = int(frac * (len(t2) - 1))
        ax.plot(x2 * 1e3, T_ex2[i] - 273.15, "-", color=c, lw=2,
                label=f"exact, t={t2[i]*1e3:.0f} ms")
        ax.plot(x2 * 1e3, T_cn2[i] - 273.15, "o", ms=4, mfc="none", color=c, markevery=8)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("T (C)")
    ax.set_title(f"(b) Benchmark 2: decaying cosine mode, CN (circles) vs exact\nmax error = {RESULTS['b2_cn_max_err_K']:.2e} K")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    n = np.array([r[0] for r in rows], dtype=float)
    e = np.array([r[1] for r in rows])
    ax.loglog(n, e, "o-", color="tab:blue", label="CN max error")
    ax.loglog(n, e[0] * (n[0] / n) ** 2, "k--", lw=1, label="slope -2 (2nd order)")
    ax.set_xlabel("grid intervals N (dx and dt halved together)")
    ax.set_ylabel("max |T_num - T_exact| (K)")
    ax.set_title(f"(c) CN grid convergence\nobserved order ~ {RESULTS['b2_cn_observed_order']:.2f}")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    for frac, c in ((0.0, "tab:blue"), (0.2, "tab:orange"), (0.5, "tab:green"), (1.0, "tab:red")):
        i = int(frac * (len(tp_) - 1))
        ax.plot(xp_ * 1e3, T_exp[i] - 273.15, "-", color=c, lw=2,
                label=f"exact, t={tp_[i]*1e3:.0f} ms")
        ax.plot(xp_ * 1e3, T_pinn[i] - 273.15, "o", ms=4, mfc="none", color=c, markevery=20)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("T (C)")
    ax.set_title(f"(d) Benchmark 2: PINN (circles) vs exact\nrel L2 = {RESULTS['b2_pinn_rel_l2']*100:.2f}%, max error = {RESULTS['b2_pinn_max_err_K']:.2f} K")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("analytical_validation.png", dpi=150)

    print("\n================ VALIDATION SUMMARY ================")
    print(f"Benchmark 1 (steady, real BCs):   CN max error = {RESULTS['b1_max_err_K']:.2e} K")
    print(f"Benchmark 2 (transient cosine):   CN max error = {RESULTS['b2_cn_max_err_K']:.2e} K"
          f"  (rel L2 = {RESULTS['b2_cn_rel_l2']:.2e})")
    print("Grid convergence (dx, dt halved together):")
    for (N, e_), o in zip(rows, [None] + orders):
        tail = f"   observed order = {o:.2f}" if o is not None else ""
        print(f"    N = {N:4d}   max error = {e_:.3e} K{tail}")
    print(f"PINN on Benchmark 2:              rel L2 = {RESULTS['b2_pinn_rel_l2']*100:.2f}%,"
          f"  max error = {RESULTS['b2_pinn_max_err_K']:.2f} K")
    print("====================================================")


if __name__ == "__main__":
    main()
