"""
Simulation and visualization of Proposition 7 from
"Sticky Mental Models with Sequential Sparse Attention" (Zhaohong Yao).

The script reproduces, *exactly* and without any modification of the recursive
system in Section 9 of the proposal, the utility-loss dynamics of four
heterogeneous agents across three scenarios.

The four agents are:
    Agent 0 - God (Full Information Rational)
    Agent 1 - Aggressive / Short-term Trader   (forces m_{i,t} = 1)
    Agent 2 - Rational / Long-term Trader      (kappa = 0)
    Agent 3 - Bounded Rational / Retail        (kappa > 0)

The three scenarios are:
    Scenario 1 - Steady State (no shock)
    Scenario 2 - Single shock at t_shock
    Scenario 3 - Double shock at t_shock and t_shock + 1

Per-scenario Monte Carlo: N_sim replications with shared shock draws across
the four agents within each replication (so the agent comparison is fair).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 0. PARAMETER BLOCK (user-adjustable)
# =============================================================================

T          = 100        # total periods (= total dimensions that arrive)
MU_I       = 1.0        # decision coefficient mu_i, identical across dimensions
TAU2_X     = 0.5        # observation-noise variance for x_{i,t}
KAPPA      = 0.3        # cognitive cost (Agent 3 only)
M_BAR      = 0.05       # minimum passive attention floor
NU0_HAT    = 0.5        # initial prior mean of nu_i for all agents
P0         = 1.0        # initial prior variance of nu_i
ETA2_Z     = 1.0        # diagnostic-signal noise variance (before /m*)
NU_STEADY  = 1.0        # true nu_i in steady state
NU_SHOCK   = 5.0        # true nu_i after a structural shock
T_SHOCK    = 30         # period at which the shock arrives
N_SIM      = 200        # number of Monte Carlo replications
MASTER_SEED = 20260502  # master RNG seed for reproducibility


# =============================================================================
# 1. SHOCK GENERATION (shared across agents within a replication)
# =============================================================================

def make_nu_schedule(scenario: str) -> np.ndarray:
    """Return an array nu[1..T] giving Var(x_{i,t}) for each dimension i.

    The variance of dimension i is constant in t (the proposal assumes
    nu_i is time-invariant; only the realization x_{i,t} is redrawn).
    """
    nu = np.full(T + 1, NU_STEADY, dtype=float)
    if scenario == "scenario_1":
        return nu
    if scenario == "scenario_2":
        nu[T_SHOCK] = NU_SHOCK
        return nu
    if scenario == "scenario_3":
        nu[T_SHOCK]     = NU_SHOCK
        nu[T_SHOCK + 1] = NU_SHOCK
        return nu
    raise ValueError(f"Unknown scenario: {scenario}")


def draw_shocks(rng: np.random.Generator,
                nu_schedule: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-draw all stochastic primitives for one replication.

    Returns three (T+1, T+1) matrices indexed [t, i] for t in 1..T, i in 1..t:
        x_true[t, i] ~ N(0, nu_i)             realization of x_{i,t}
        xi[t, i]     ~ N(0, tau2_x)           perception noise xi_{i,t}
        u_std[t, i]  ~ N(0, 1)                standard-normal seed for the
                                              diagnostic noise u_{i,t+1};
                                              the agent-specific scaling by
                                              sqrt(eta2_z / m*_{i,t}) happens
                                              inside simulate_agent.
    Entries with i > t are left at zero and never read.
    """
    x_true = np.zeros((T + 1, T + 1))
    xi     = np.zeros((T + 1, T + 1))
    u_std  = np.zeros((T + 1, T + 1))
    for t in range(1, T + 1):
        sd_x = np.sqrt(nu_schedule[1:t + 1])
        x_true[t, 1:t + 1] = rng.normal(0.0, sd_x)
        xi[t, 1:t + 1]     = rng.normal(0.0, np.sqrt(TAU2_X), size=t)
        u_std[t, 1:t + 1]  = rng.standard_normal(size=t)
    return x_true, xi, u_std


# =============================================================================
# 2. AGENT SIMULATION (Section 9 recursive system)
# =============================================================================

def simulate_agent(agent_type: str,
                   nu_schedule: np.ndarray,
                   x_true: np.ndarray,
                   xi: np.ndarray,
                   u_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one agent over T periods with pre-drawn shocks.

    Implements *exactly* the recursive system of Section 9:

        A_{i,t}      = mu_i^2 * (hat_nu_{i,t} + (x^s_{i,t-1})^2)
        C_i          = kappa_i + mu_i^2 * tau2_x
        m*_{i,t}     = m_bar + (1 - m_bar) * A_{i,t} / (A_{i,t} + C_i)
        x^s_{i,t}    = (1 - m*) x^s_{i,t-1} + m* (x_{i,t} + xi_{i,t})
        z_{i,t+1}    = nu_i + u_{i,t+1},   u ~ N(0, eta2_z / m*_{i,t})
        lambda_{i,t} = m* P / (m* P + eta2_z)
        hat_nu update: hat_nu + lambda * (z - hat_nu)
        P update:      P * (1 - lambda)

    Initialization at i = t (the first period dimension i exists):
        x^s_{i,i-1} = 0,  hat_nu_{i,i} = NU0_HAT,  P_{i,i} = P0.

    Returns (L_inst, L_cum), each of length T + 1 (index 0 unused, t = 1..T).
    """

    L_inst = np.zeros(T + 1)

    if agent_type == "god":
        # Full-information rational benchmark: x^s_{i,t} = x_{i,t}, error = 0.
        return L_inst, np.cumsum(L_inst)

    # Agent-specific cognitive cost.
    if agent_type == "bounded_rational":
        kappa_i = KAPPA
    else:
        kappa_i = 0.0
    C_i = kappa_i + MU_I ** 2 * TAU2_X  # scalar (mu_i and tau2_x are common)

    # Per-dimension state vectors of length T + 1 (1..T are used).
    x_perc = np.zeros(T + 1)            # x^s_{i, t-1}
    nu_hat = np.full(T + 1, NU0_HAT)    # hat_nu_{i, t}
    P_arr  = np.full(T + 1, P0)         # P_{i, t}

    for t in range(1, T + 1):
        # Slice over arrived dimensions 1..t.
        sl = slice(1, t + 1)

        x_true_t = x_true[t, sl]
        xi_t     = xi[t, sl]
        u_std_t  = u_std[t, sl]
        nu_i_t   = nu_schedule[sl]

        # --- attention rule ---------------------------------------------------
        if agent_type == "aggressive":
            # Force m_{i,t} = 1 for every arrived dimension.
            m_star = np.ones(t)
        else:
            A = MU_I ** 2 * (nu_hat[sl] + x_perc[sl] ** 2)
            # Closed-form interior solution from Section 6.  The proposal
            # explicitly constrains m \in [0, 1]; the affine floor m_bar then
            # gives m* \in [m_bar, 1].  We enforce this feasibility set so
            # the rare cases where the Gaussian belief on nu_i drifts
            # negative (a limitation acknowledged in Section 4 of the
            # proposal) do not break the recursion.
            bar_m = np.clip(A / (A + C_i), 0.0, 1.0)
            m_star = M_BAR + (1.0 - M_BAR) * bar_m

        # --- perception update ------------------------------------------------
        x_s_new = (1.0 - m_star) * x_perc[sl] + m_star * (x_true_t + xi_t)

        # --- action error and instantaneous utility loss ---------------------
        err_t = np.sum(MU_I * (x_true_t - x_s_new))
        L_inst[t] = 0.5 * err_t ** 2

        # --- diagnostic signal and Kalman update -----------------------------
        # u_{i,t+1} ~ N(0, eta2_z / m*_{i,t})
        u_t = u_std_t * np.sqrt(ETA2_Z / m_star)
        z_t = nu_i_t + u_t
        lam = (m_star * P_arr[sl]) / (m_star * P_arr[sl] + ETA2_Z)
        nu_hat[sl] = nu_hat[sl] + lam * (z_t - nu_hat[sl])
        P_arr[sl]  = P_arr[sl] * (1.0 - lam)

        # --- carry perceived state forward to t+1 ----------------------------
        x_perc[sl] = x_s_new

    return L_inst, np.cumsum(L_inst)


# =============================================================================
# 3. MONTE CARLO DRIVER
# =============================================================================

AGENT_TYPES = ("god", "aggressive", "rational", "bounded_rational")


def run_scenario(scenario: str) -> dict[str, dict[str, np.ndarray]]:
    """Run N_SIM replications for the given scenario; return mean L_inst / L_cum."""
    nu_schedule = make_nu_schedule(scenario)
    master_rng = np.random.default_rng(MASTER_SEED + hash(scenario) % (2 ** 31))

    # Accumulators of shape (T+1,) for each agent.
    sums_inst = {a: np.zeros(T + 1) for a in AGENT_TYPES}
    sums_cum  = {a: np.zeros(T + 1) for a in AGENT_TYPES}

    for _ in range(N_SIM):
        # Shared shocks across the four agents within this replication.
        rng = np.random.default_rng(master_rng.integers(0, 2 ** 63 - 1))
        x_true, xi, u_std = draw_shocks(rng, nu_schedule)
        for agent in AGENT_TYPES:
            l_inst, l_cum = simulate_agent(agent, nu_schedule, x_true, xi, u_std)
            sums_inst[agent] += l_inst
            sums_cum[agent]  += l_cum

    return {
        agent: {
            "L_inst": sums_inst[agent] / N_SIM,
            "L_cum":  sums_cum[agent]  / N_SIM,
        }
        for agent in AGENT_TYPES
    }


# =============================================================================
# 4. PLOTTING
# =============================================================================

AGENT_STYLE = {
    "god": {
        "label": "God (omniscient)",
        "color": "black",
        "linestyle": "--",
        "linewidth": 1.8,
    },
    "aggressive": {
        "label": f"Aggressive (m=1, $\\kappa$=0)",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "rational": {
        "label": f"Rational ($\\kappa$=0, $\\tau^2$={TAU2_X})",
        "color": "tab:green",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "bounded_rational": {
        "label": f"Bounded Rational ($\\kappa$={KAPPA}, $\\tau^2$={TAU2_X})",
        "color": "tab:red",
        "linestyle": "-",
        "linewidth": 1.8,
    },
}

SCENARIO_TITLES = {
    "scenario_1": (
        f"Scenario 1: Steady State "
        f"($\\nu$={NU_STEADY}, $\\tau^2$={TAU2_X}, $\\kappa$={KAPPA}, "
        f"$\\bar m$={M_BAR}, T={T}, N={N_SIM})"
    ),
    "scenario_2": (
        f"Scenario 2: Single Shock at t={T_SHOCK} "
        f"($\\nu_{{shock}}$={NU_SHOCK}, $\\nu$={NU_STEADY}, "
        f"$\\tau^2$={TAU2_X}, $\\kappa$={KAPPA}, $\\bar m$={M_BAR}, "
        f"T={T}, N={N_SIM})"
    ),
    "scenario_3": (
        f"Scenario 3: Double Shock at t={T_SHOCK},{T_SHOCK + 1} "
        f"($\\nu_{{shock}}$={NU_SHOCK}, $\\nu$={NU_STEADY}, "
        f"$\\tau^2$={TAU2_X}, $\\kappa$={KAPPA}, $\\bar m$={M_BAR}, "
        f"T={T}, N={N_SIM})"
    ),
}

SCENARIO_FILE = {
    "scenario_1": "proposition7_scenario1_steady.png",
    "scenario_2": "proposition7_scenario2_single_shock.png",
    "scenario_3": "proposition7_scenario3_double_shock.png",
}


def plot_scenario(scenario: str, results: dict[str, dict[str, np.ndarray]]) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    t_axis = np.arange(1, T + 1)

    # --- (A) Instantaneous loss ----------------------------------------------
    ax = axes[0]
    for agent in AGENT_TYPES:
        style = AGENT_STYLE[agent]
        ax.plot(t_axis, results[agent]["L_inst"][1:], **style)
    if scenario in ("scenario_2", "scenario_3"):
        ax.axvline(T_SHOCK, color="gray", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"shock at t={T_SHOCK}")
        if scenario == "scenario_3":
            ax.axvline(T_SHOCK + 1, color="gray", linestyle=":",
                       linewidth=1.0, alpha=0.8,
                       label=f"shock at t={T_SHOCK + 1}")
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Utility Loss")
    ax.set_title("(A) Instantaneous Utility Loss $L_t$")
    ax.legend(loc="best", fontsize=9)

    # --- (B) Cumulative loss --------------------------------------------------
    ax = axes[1]
    for agent in AGENT_TYPES:
        style = AGENT_STYLE[agent]
        ax.plot(t_axis, results[agent]["L_cum"][1:], **style)
    if scenario in ("scenario_2", "scenario_3"):
        ax.axvline(T_SHOCK, color="gray", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"shock at t={T_SHOCK}")
        if scenario == "scenario_3":
            ax.axvline(T_SHOCK + 1, color="gray", linestyle=":",
                       linewidth=1.0, alpha=0.8,
                       label=f"shock at t={T_SHOCK + 1}")
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Utility Loss")
    ax.set_title(r"(B) Cumulative Utility Loss $\sum_{s=1}^{t} L_s$")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(SCENARIO_TITLES[scenario], fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(SCENARIO_FILE[scenario], dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. MAIN
# =============================================================================

def main() -> None:
    for scenario in ("scenario_1", "scenario_2", "scenario_3"):
        print(f"[run] {scenario} ...")
        results = run_scenario(scenario)
        plot_scenario(scenario, results)
        print(f"[done] saved {SCENARIO_FILE[scenario]}")


if __name__ == "__main__":
    main()
