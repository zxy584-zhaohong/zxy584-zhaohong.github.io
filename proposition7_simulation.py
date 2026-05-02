"""
Simulation of Proposition 7
============================
Compares the Utility Loss dynamics of 4 heterogeneous agents (God,
Aggressive short-term trader, Rational long-term trader, Bounded
Rational retail investor) across 3 scenarios (Steady State, Single
Shock, Double Shock).

The recursive system follows Section 9 of the paper. All agents share
the same recursive equations and differ only in the parameter values
that enter those equations (kappa, the m* rule). Agent 0 (God) is the
omniscient benchmark with zero loss.

Author: Quant Econ / Finance simulation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =====================================================================
# 0. PARAMETER BLOCK (user-adjustable)
# =====================================================================
T = 100             # total periods (= total dimensions that arrive)
mu_i = 1.0          # decision coefficient, same for all dimensions
tau2_x = 0.5        # observation noise variance for x_{i,t}
kappa = 0.3         # cognitive cost (Agent 3 only)
m_bar = 0.05        # minimum passive attention floor
nu0_hat = 0.5       # initial prior mean of nu_i
P0 = 1.0            # initial prior variance of nu_i
eta2_z = 1.0        # diagnostic signal noise variance (before scaling by m*)
nu_steady = 1.0     # true nu_i in steady state
nu_shock = 5.0      # true nu_i after a structural shock
t_shock = 30        # period at which the shock occurs (0-indexed period)
N_sim = 200         # number of Monte Carlo replications for smoothing

SEED = 20260502
OUT_DIR = "/workspace"

# Plot styling
sns.set_style("whitegrid")

AGENT_COLOR = {
    "god":               "black",
    "aggressive":        "tab:blue",
    "rational":          "tab:green",
    "bounded_rational":  "tab:red",
}
AGENT_LINESTYLE = {
    "god":               "--",
    "aggressive":        "-",
    "rational":          "-",
    "bounded_rational":  "-",
}
AGENT_LABEL = {
    "god":              f"God (omniscient)",
    "aggressive":       f"Aggressive (m=1, $\\kappa$=0)",
    "rational":         f"Rational ($\\kappa$=0, $\\tau^2$={tau2_x})",
    "bounded_rational": f"Bounded Rational ($\\kappa$={kappa}, $\\tau^2$={tau2_x})",
}


# =====================================================================
# 1. CORE SIMULATOR
# =====================================================================
def simulate_agent(
    agent_type: str,
    nu_schedule: np.ndarray,
    *,
    T: int = T,
    N_sim: int = N_sim,
    mu_i: float = mu_i,
    tau2_x: float = tau2_x,
    kappa: float = kappa,
    m_bar: float = m_bar,
    nu0_hat: float = nu0_hat,
    P0: float = P0,
    eta2_z: float = eta2_z,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the per-dimension recursive system of Section 9 for a given
    agent and return Monte-Carlo averaged paths of:

        L_t        : instantaneous utility loss at period t
        cumL_t     : cumulative loss sum_{s=0}^{t} L_s

    Parameters
    ----------
    agent_type   : one of {'god', 'aggressive', 'rational', 'bounded_rational'}
    nu_schedule  : np.ndarray, shape (T,)
                   nu_schedule[i] = true nu of dimension i (constant across t).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    # --- Agent 0 : omniscient -> zero loss always ----------------------
    if agent_type == "god":
        return np.zeros(T), np.zeros(T)

    # --- Agent-specific cognitive cost ---------------------------------
    if agent_type == "bounded_rational":
        kappa_eff = kappa
    else:                                # aggressive, rational
        kappa_eff = 0.0
    C_i = kappa_eff + mu_i ** 2 * tau2_x

    force_full_attention = (agent_type == "aggressive")

    # --- State arrays, shape (N_sim, T) --------------------------------
    # x_s_prev[s, i] : agent's perceived state of dimension i at end of t-1
    # nu_hat[s, i]   : agent's posterior mean of nu_i
    # P[s, i]        : agent's posterior variance of nu_i
    x_s_prev = np.zeros((N_sim, T))
    nu_hat   = np.full((N_sim, T), nu0_hat, dtype=float)
    P        = np.full((N_sim, T), P0,      dtype=float)

    L_t = np.zeros(T)

    # Per-dimension std-dev of the true x_{i,t}
    sigma_x_dim = np.sqrt(nu_schedule)            # shape (T,)

    for t in range(T):
        # Dimensions active at period t: i = 0,1,...,t  (dim i arrives at t=i)
        n_active = t + 1
        sl = slice(0, n_active)

        # ---------- m*_{i,t} via the attention rule -------------------
        # nu_hat estimates a variance (>=0); clip to enforce that.
        nu_hat_pos = np.maximum(nu_hat[:, sl], 0.0)
        if force_full_attention:
            m_star = np.ones((N_sim, n_active))
        else:
            A = mu_i ** 2 * (nu_hat_pos + x_s_prev[:, sl] ** 2)
            m_star = m_bar + (1.0 - m_bar) * A / (A + C_i)
            # numerical safety
            m_star = np.clip(m_star, m_bar, 1.0)

        # ---------- True x_{i,t} and observation noise ----------------
        x_true = rng.standard_normal((N_sim, n_active)) * sigma_x_dim[sl][None, :]
        xi     = rng.standard_normal((N_sim, n_active)) * np.sqrt(tau2_x)

        # ---------- Perception update --------------------------------
        x_s_new = (1.0 - m_star) * x_s_prev[:, sl] + m_star * (x_true + xi)

        # ---------- Action error and instantaneous loss --------------
        err_per_dim = mu_i * (x_true - x_s_new)         # (N_sim, n_active)
        err_total   = err_per_dim.sum(axis=1)           # (N_sim,)
        L_t[t]      = 0.5 * np.mean(err_total ** 2)

        # ---------- Kalman update of nu_hat using diagnostic signal ---
        # z_{i,t+1} = nu_i + u,  u ~ N(0, eta2_z / m*)
        u = rng.standard_normal((N_sim, n_active)) * np.sqrt(eta2_z / m_star)
        z = nu_schedule[sl][None, :] + u
        lam = m_star * P[:, sl] / (m_star * P[:, sl] + eta2_z)
        nu_hat[:, sl] = nu_hat[:, sl] + lam * (z - nu_hat[:, sl])
        # nu is a variance: enforce non-negativity of the posterior mean
        nu_hat[:, sl] = np.maximum(nu_hat[:, sl], 0.0)
        P[:, sl]      = P[:, sl] * (1.0 - lam)

        # Carry the perception state forward (dim t+1 stays at init=0).
        x_s_prev[:, sl] = x_s_new

    cum_L = np.cumsum(L_t)
    return L_t, cum_L


# =====================================================================
# 2. SCENARIO BUILDERS
# =====================================================================
def nu_schedule_steady() -> np.ndarray:
    return np.full(T, nu_steady, dtype=float)

def nu_schedule_single_shock() -> np.ndarray:
    nu = np.full(T, nu_steady, dtype=float)
    nu[t_shock] = nu_shock
    return nu

def nu_schedule_double_shock() -> np.ndarray:
    nu = np.full(T, nu_steady, dtype=float)
    nu[t_shock]     = nu_shock
    nu[t_shock + 1] = nu_shock
    return nu


@dataclass
class ScenarioResult:
    name: str
    title: str
    nu_schedule: np.ndarray
    has_shock: bool
    losses: dict        # agent_type -> (L_t, cumL_t)


def run_scenario(name: str, title: str, nu_schedule: np.ndarray,
                 has_shock: bool) -> ScenarioResult:
    """Run all 4 agents for a given nu schedule and return the losses."""
    losses = {}
    # Use the SAME random stream resetting per agent, so Monte-Carlo draws
    # of (x, xi, u) are common across agents in expectation. This makes
    # the comparison cleaner.
    for agent in ("god", "aggressive", "rational", "bounded_rational"):
        rng = np.random.default_rng(SEED)
        losses[agent] = simulate_agent(agent, nu_schedule, rng=rng)
    return ScenarioResult(name=name, title=title, nu_schedule=nu_schedule,
                          has_shock=has_shock, losses=losses)


# =====================================================================
# 3. PLOTTING
# =====================================================================
def plot_scenario(result: ScenarioResult, fname: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    t_axis = np.arange(1, T + 1)

    for agent in ("god", "aggressive", "rational", "bounded_rational"):
        L_t, cumL = result.losses[agent]
        axes[0].plot(t_axis, L_t,
                     color=AGENT_COLOR[agent],
                     linestyle=AGENT_LINESTYLE[agent],
                     linewidth=2.0,
                     label=AGENT_LABEL[agent])
        axes[1].plot(t_axis, cumL,
                     color=AGENT_COLOR[agent],
                     linestyle=AGENT_LINESTYLE[agent],
                     linewidth=2.0,
                     label=AGENT_LABEL[agent])

    if result.has_shock:
        for ax in axes:
            ax.axvline(t_shock + 1, color="gray", linestyle="--",
                       linewidth=1.2, alpha=0.7,
                       label=f"shock @ t={t_shock + 1}")

    axes[0].set_title("(A) Instantaneous Utility Loss  $L_t$")
    axes[1].set_title("(B) Cumulative Utility Loss  $\\sum_{s\\leq t} L_s$")

    for ax in axes:
        ax.set_xlabel("Period $t$")
        ax.set_ylabel("Utility Loss")
        ax.legend(loc="best", fontsize=9, framealpha=0.85)

    fig.suptitle(result.title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {fname}")


# =====================================================================
# 4. MAIN
# =====================================================================
def main() -> None:
    common = (f"$\\nu_{{steady}}$={nu_steady}, $\\nu_{{shock}}$={nu_shock}, "
              f"$\\tau^2$={tau2_x}, $\\kappa$={kappa}, "
              f"$\\bar m$={m_bar}, $T$={T}, $N_{{sim}}$={N_sim}")

    scenarios = [
        ("scenario1_steady",
         f"Scenario 1: Steady State  ({common})",
         nu_schedule_steady(),  False),
        ("scenario2_single_shock",
         f"Scenario 2: Single Shock at t={t_shock + 1}  ({common})",
         nu_schedule_single_shock(),  True),
        ("scenario3_double_shock",
         f"Scenario 3: Double Shock at t={t_shock + 1}, {t_shock + 2}  ({common})",
         nu_schedule_double_shock(),  True),
    ]

    for name, title, nu_sched, has_shock in scenarios:
        print(f"\n>>> Running {name} ...")
        res = run_scenario(name, title, nu_sched, has_shock)

        # Quick console summary
        for agent in ("god", "aggressive", "rational", "bounded_rational"):
            L_t, cumL = res.losses[agent]
            print(f"    {agent:18s}  L_T={L_t[-1]:9.3f}   cumL_T={cumL[-1]:10.2f}")

        out_path = os.path.join(OUT_DIR, f"{name}.png")
        plot_scenario(res, out_path)


if __name__ == "__main__":
    main()
