"""
Proposition 7 — Heterogeneous Agent Utility Loss Dynamics
Sticky Mental Models with Sequential Sparse Attention.

Recursive system from Section 9 of the proposal.

Architecture
------------
  E[L_t] = (1/2) sum_{i<=t} mu_i^2 E[e_{i,t}^2]
  We track per-dimension squared errors AND attention m* across time.
  Cross-dimension covariance terms vanish because, in our simulation,
  the true x_{i,t} are drawn independently across i; this lets us use
  the variance-reducing identity above instead of MC-averaging the
  realized squared sum (both have the same expectation).

Figures
-------
  Fig 1-3: Total instantaneous and cumulative loss for the 3 scenarios.
  Fig 4  : Shocked-dimension MSE (log scale) and m* dynamics —
           the cleanest visualization of the Prop 5 stickiness mechanism.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# =============================================================================
# 0. PARAMETER BLOCK  (user-adjustable)
# =============================================================================
T         = 100       # total periods = total dimensions that arrive
MU        = 1.0       # decision coefficient mu_i
TAU2_X    = 0.5       # observation noise variance for x_{i,t}
KAPPA     = 15.0      # cognitive cost (Bounded Rational only) -- HIGH-STICKINESS regime
M_BAR     = 0.02      # passive attention floor (low so BR has room to "stick")
NU0_HAT   = 0.2       # initial prior mean of nu_i (pessimistic: guarantees low initial m*)
P0        = 1.0       # initial prior variance of nu_i
ETA2_Z    = 3.0       # diagnostic-signal base noise (high => slow learning => BR stays stuck)
NU_STEADY = 1.0       # true nu_i in steady state
NU_SHOCK  = 5.0       # true nu_i for the shocked dimension(s)
T_SHOCK   = 30        # 0-indexed arrival period of the shocked dimension
N_SIM     = 800       # Monte Carlo replications
SEED      = 42

# Output directory.  Default = the user's Windows path; on non-Windows
# machines (e.g. for headless testing) override via the PROP7_OUT_DIR
# environment variable.
OUT_DIR = os.environ.get(
    "PROP7_OUT_DIR",
    r"C:\Users\78558\OneDrive - University of Miami\Desktop\Research\Chad",
)
os.makedirs(OUT_DIR, exist_ok=True)

AGENT_TYPES = ['god', 'aggressive', 'rational', 'bounded_rational']


# =============================================================================
# 1. NU SCHEDULES
# =============================================================================
def make_nu_schedule(scenario: str) -> np.ndarray:
    nu = np.full(T, NU_STEADY, dtype=float)
    if scenario == 'single_shock':
        nu[T_SHOCK] = NU_SHOCK
    elif scenario == 'double_shock':
        nu[T_SHOCK] = NU_SHOCK
        nu[min(T_SHOCK + 1, T - 1)] = NU_SHOCK
    return nu


# =============================================================================
# 2. SINGLE-PATH SIMULATOR  (Section 9 recursive system, verbatim)
# =============================================================================
def simulate_one_path(agent_type: str, nu_true: np.ndarray, rng):
    """One Monte Carlo replication.

    Returns
    -------
    sq_err : (T, T)   sq_err[t, i] = (x_{i,t} - x^s_{i,t})^2 for i <= t
    m_arr  : (T, T)   m_arr[t, i]  = m*_{i,t}              for i <= t
    """
    sq_err = np.zeros((T, T))
    m_arr  = np.zeros((T, T))

    if agent_type == 'god':
        return sq_err, np.ones((T, T))

    kappa_a = KAPPA if agent_type == 'bounded_rational' else 0.0
    C_i = kappa_a + MU**2 * TAU2_X

    xs_prev = np.zeros(T)
    nu_hat  = np.full(T, NU0_HAT)
    P       = np.full(T, P0)

    for t in range(T):
        n = t + 1
        x_true   = rng.normal(0.0, np.sqrt(nu_true[:n]))
        xi_noise = rng.normal(0.0, np.sqrt(TAU2_X), size=n)

        # ----- Attention rule (Section 6) --------------------------------
        if agent_type == 'aggressive':
            ms = np.ones(n)
        else:
            A = MU**2 * (nu_hat[:n] + xs_prev[:n]**2)
            # The Gaussian belief on nu_i (Section 4) does not enforce
            # nu_hat >= 0, a limitation the proposal explicitly flags.
            # On rare paths nu_hat can drift slightly negative; we clip A
            # to a tiny positive floor so the closed-form attention rule
            # stays inside the proposal's own feasibility set m in [0,1].
            A = np.maximum(A, 1e-12)
            ms = M_BAR + (1.0 - M_BAR) * A / (A + C_i)

        m_arr[t, :n] = ms

        # ----- Perception (Section 5) ------------------------------------
        xs_new = (1.0 - ms) * xs_prev[:n] + ms * (x_true + xi_noise)
        e = x_true - xs_new
        sq_err[t, :n] = e**2

        # ----- Diagnostic signal + Kalman update (Section 8) -------------
        sig_u = np.sqrt(ETA2_Z / np.maximum(ms, 1e-12))
        z = nu_true[:n] + rng.normal(0.0, sig_u)
        lam = ms * P[:n] / (ms * P[:n] + ETA2_Z)
        nu_hat[:n] += lam * (z - nu_hat[:n])
        P[:n] *= (1.0 - lam)

        xs_prev[:n] = xs_new

    return sq_err, m_arr


# =============================================================================
# 3. MC AGGREGATOR
# =============================================================================
def run_scenario(scenario: str):
    nu_true = make_nu_schedule(scenario)
    rng0 = np.random.default_rng(SEED)
    results = {}

    for atype in AGENT_TYPES:
        acc_sq = np.zeros((T, T))
        acc_m  = np.zeros((T, T))

        for _ in range(N_SIM):
            seed_s = rng0.integers(0, 2**31)
            sq, ma = simulate_one_path(atype, nu_true,
                                       np.random.default_rng(seed_s))
            acc_sq += sq
            acc_m  += ma

        dim_mse = acc_sq / N_SIM
        dim_m   = acc_m / N_SIM

        inst = np.array([0.5 * MU**2 * dim_mse[t, :t+1].sum()
                         for t in range(T)])
        cum  = np.cumsum(inst)

        results[atype] = dict(inst=inst, cum=cum, dim_mse=dim_mse, dim_m=dim_m)

    return results


# =============================================================================
# 4. PLOTTING HELPERS
# =============================================================================
COLORS  = {'god': '#000000', 'aggressive': '#1f77b4',
           'rational': '#2ca02c', 'bounded_rational': '#d62728'}
LABELS  = {
    'god':              'God (omniscient)',
    'aggressive':       f'Aggressive ($m=1,\\;\\kappa=0$)',
    'rational':         f'Rational ($\\kappa=0,\\;\\tau^2_x\\!=\\!{TAU2_X}$)',
    'bounded_rational': f'Bounded Rational ($\\kappa\\!=\\!{KAPPA},\\;\\tau^2_x\\!=\\!{TAU2_X}$)',
}
LSTYLES = {'god': '--', 'aggressive': '-', 'rational': '-', 'bounded_rational': '-'}
LWIDTHS = {'god': 2.0, 'aggressive': 1.7, 'rational': 1.7, 'bounded_rational': 1.7}
DRAW_ORDER = ['god', 'aggressive', 'bounded_rational', 'rational']

SCEN_TITLE = {
    'steady':       (f'Scenario 1: Steady State  '
                     f'($\\nu={NU_STEADY},\\;\\tau^2_x={TAU2_X},'
                     f'\\;\\kappa={KAPPA},\\;\\bar m={M_BAR},\\;'
                     f'T={T},\\;N={N_SIM}$)'),
    'single_shock': (f'Scenario 2: Single Shock at $t={T_SHOCK+1}$  '
                     f'($\\nu\'={NU_SHOCK},\\;\\nu={NU_STEADY},\\;'
                     f'\\tau^2_x={TAU2_X},\\;\\kappa={KAPPA},\\;'
                     f'\\bar m={M_BAR},\\;T={T},\\;N={N_SIM}$)'),
    'double_shock': (f'Scenario 3: Double Shock at $t={T_SHOCK+1},{T_SHOCK+2}$  '
                     f'($\\nu\'={NU_SHOCK},\\;\\nu={NU_STEADY},\\;'
                     f'\\tau^2_x={TAU2_X},\\;\\kappa={KAPPA},\\;'
                     f'\\bar m={M_BAR},\\;T={T},\\;N={N_SIM}$)'),
}


def _draw(ax, periods, data_dict, key, ylabel):
    for a in DRAW_ORDER:
        ax.plot(periods, data_dict[a][key], color=COLORS[a], label=LABELS[a],
                linestyle=LSTYLES[a], linewidth=LWIDTHS[a])
    ax.set_xlabel('Period  $t$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(1, T)


def plot_total_loss(results, scenario, filename):
    """Two-panel figure: instantaneous (A) and cumulative (B) total loss."""
    periods = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    _draw(axes[0], periods, results, 'inst', 'Utility Loss')
    _draw(axes[1], periods, results, 'cum',  'Utility Loss')

    if scenario != 'steady':
        for ax in axes:
            ax.axvline(T_SHOCK + 1, color='gray', ls=':', lw=1.2, alpha=0.65)
        if scenario == 'double_shock':
            for ax in axes:
                ax.axvline(T_SHOCK + 2, color='gray', ls=':', lw=1.0, alpha=0.5)

    axes[0].set_title('(A)  Instantaneous  $\\mathbb{E}[L_t]$', fontsize=13)
    axes[1].set_title('(B)  Cumulative  $\\sum_{s=1}^{t}\\mathbb{E}[L_s]$',
                      fontsize=13)
    for ax in axes:
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
    fig.suptitle(SCEN_TITLE[scenario], fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {filename}')


def plot_shock_dim_analysis(res_single, res_double, filename):
    """Fig 4: shocked-dimension deep dive.

    Row 1: Single shock — left = E[e_{i,t}^2] (LOG scale), right = m*_{i,t}.
    Row 2: Double shock — same two panels.

    The MSE panels use a logarithmic y-axis because the spike at the
    shock period and the slow recovery toward the steady-state floor
    span ~1.5 orders of magnitude; on a linear axis the recovery
    dynamics are visually compressed against the spike.  This is
    purely a presentation choice; the underlying simulated values are
    identical to what a linear-axis plot would show.

    God is omitted from the MSE panels because E[e^2] is identically
    zero for the omniscient benchmark, which has no representation on
    a logarithmic axis.  God is retained in the attention panels at
    the constant value m* = 1.
    """
    dim_idx = T_SHOCK             # the (first) shocked dimension
    t_start = T_SHOCK             # arrival period (0-indexed)
    t_range = np.arange(t_start + 1, T + 1)  # 1-indexed for display, starts at the shock
    t_idx   = np.arange(t_start, T)          # corresponding 0-indexed array index

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mortal_order = [a for a in DRAW_ORDER if a != 'god']

    for row, (res, label_sc) in enumerate(
        [(res_single, 'Single Shock'), (res_double, 'Double Shock')]):

        # ---- left column: per-dim MSE on LOG scale --------------------------
        for a in mortal_order:
            mse_dim = res[a]['dim_mse'][t_idx, dim_idx]
            axes[row, 0].plot(t_range, mse_dim, color=COLORS[a],
                              label=LABELS[a], linestyle=LSTYLES[a],
                              linewidth=LWIDTHS[a])
        axes[row, 0].set_yscale('log')
        axes[row, 0].set_title(
            f'{label_sc}:  Per-dim MSE  $\\mathbb{{E}}[e_{{i,t}}^2]$  '
            f'for shocked dim $i={dim_idx+1}$  '
            r'$\mathbf{(log\ scale)}$',
            fontsize=12)
        axes[row, 0].set_ylabel(r'$\mathbb{E}[e_{i,t}^2]\;$ (log scale)',
                                fontsize=12)
        axes[row, 0].text(
            0.98, 0.04,
            'Note: y-axis is logarithmic.\n'
            r'God omitted ($\mathbb{E}[e^2]\equiv 0$).',
            transform=axes[row, 0].transAxes,
            ha='right', va='bottom', fontsize=8.5, color='dimgray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.85))

        # ---- right column: attention m* on LINEAR scale ---------------------
        for a in DRAW_ORDER:
            m_dim = res[a]['dim_m'][t_idx, dim_idx]
            axes[row, 1].plot(t_range, m_dim, color=COLORS[a],
                              label=LABELS[a], linestyle=LSTYLES[a],
                              linewidth=LWIDTHS[a])
        axes[row, 1].set_title(
            f'{label_sc}:  Attention  $m^*_{{i,t}}$  '
            f'for shocked dim $i={dim_idx+1}$', fontsize=12)
        axes[row, 1].set_ylabel('$m^*_{i,t}$', fontsize=12)
        axes[row, 1].set_ylim(-0.02, 1.05)

        for c in range(2):
            axes[row, c].axvline(T_SHOCK + 1, color='gray', ls=':',
                                 lw=1.2, alpha=0.7,
                                 label=f'arrival $t={T_SHOCK+1}$')
            axes[row, c].set_xlabel('Period  $t$', fontsize=11)
            axes[row, c].grid(True, which='both', alpha=0.25)
            axes[row, c].set_xlim(t_range[0], t_range[-1])

    for ax in axes.flat:
        ax.legend(fontsize=8.5, loc='best', framealpha=0.9)

    fig.suptitle(
        f'Shocked-Dimension Analysis  '
        f'($\\nu\'={NU_SHOCK},\\;\\nu={NU_STEADY},\\;\\tau^2_x={TAU2_X},'
        f'\\;\\kappa={KAPPA},\\;\\bar m={M_BAR},\\;T={T},\\;N={N_SIM}$)',
        fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {filename}')


# =============================================================================
# 5. MAIN
# =============================================================================
if __name__ == '__main__':
    print(f'Output directory: {OUT_DIR}')
    all_res = {}
    for scen, fn in [
        ('steady',       os.path.join(OUT_DIR, 'prop7_fig1_steady.png')),
        ('single_shock', os.path.join(OUT_DIR, 'prop7_fig2_single_shock.png')),
        ('double_shock', os.path.join(OUT_DIR, 'prop7_fig3_double_shock.png')),
    ]:
        print(f'>>> {scen}')
        all_res[scen] = run_scenario(scen)
        plot_total_loss(all_res[scen], scen, fn)

    print('>>> shock-dim analysis')
    plot_shock_dim_analysis(
        all_res['single_shock'], all_res['double_shock'],
        os.path.join(OUT_DIR, 'prop7_fig4_shock_dim_analysis.png'))

    print('Done.')
