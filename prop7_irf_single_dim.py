"""
Proposition 7 — Single-Dimension IRF Visualization
===================================================
Heterogeneous-agent impulse response to a volatility regime shift,
implemented as the **single long-lived dimension** degenerate case
of the model in Section 9 of "Sticky Mental Models with Sequential
Sparse Attention".

Why this degenerate case is interesting
---------------------------------------
The full proposal model is multi-dimensional (dimension i arrives at
period i; the agent tracks all arrived dimensions simultaneously).  In
that setting the aggregate utility loss aggregates across all arrived
dimensions, which visually dilutes the contribution of any single
shocked dimension.

When only ONE long-lived dimension exists, the aggregate loss equals
the per-dimension loss and the impulse response to a regime shift in
nu_1 becomes visually clean.  The proposal's Section 9 recursive
system applies VERBATIM:

    A_{1,t}      = mu^2 * (nu_hat_t + (x^s_{t-1})^2)
    C            = kappa + mu^2 * tau2_x
    m*_t         = m_bar + (1 - m_bar) * A_{1,t} / (A_{1,t} + C)
    x^s_t        = (1 - m*_t) x^s_{t-1} + m*_t (x_t + xi_t)
    z_{t+1}      = nu_{1,t} + u_{t+1},  u ~ N(0, eta2_z / m*_t)
    lambda_t     = m*_t P_t / (m*_t P_t + eta2_z)
    nu_hat_{t+1} = nu_hat_t + lambda_t (z_{t+1} - nu_hat_t)
    P_{t+1}      = P_t (1 - lambda_t)

The only thing that distinguishes this experiment from Proposition 6's
long-run-learning environment is that nu_{1,t} is time-varying:

    nu_{1,t} = NU_LOW   for t < T_SHOCK
    nu_{1,t} = NU_HIGH  for t >= T_SHOCK

The Gaussian belief on nu is mis-specified relative to this regime
shift: the agent treats nu as a constant unknown and learns it from
diagnostic signals.  This mis-specification is the source of the
underreaction we visualize.

Three agents, identical recursive system, differ only in (m, kappa):
  - Aggressive            : m_t = 1            (kappa = 0)
  - Rational              : full attention rule (kappa = 0)
  - Bounded Rational (BR) : full attention rule (kappa = KAPPA_BR > 0)

Aggregate "price" for Panel C (a market-clearing extension that goes
slightly beyond the proposal proper):
    P_t = w_A * x^s_A,t + w_R * x^s_R,t + w_B * x^s_B,t
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 0. PARAMETER BLOCK
# =====================================================
T          = 150        # total periods
T_SHOCK    = 50         # period at which nu_{1,t} jumps
NU_LOW     = 1.0        # pre-shock variance of x_{1,t}
NU_HIGH    = 20.0       # post-shock variance of x_{1,t}
TAU2_X     = 5.0        # observation noise variance
MU         = 1.0        # decision coefficient mu_1
KAPPA_BR   = 50.0       # cognitive cost for BR (Rational uses 0)
M_BAR      = 0.05       # passive attention floor
NU0_HAT    = 1.0        # initial prior mean of nu_1
P0         = 5.0        # initial prior variance of nu_1
ETA2_Z     = 1.0        # diagnostic-signal base noise
N_SIM      = 800        # Monte Carlo replications
SEED       = 42

# Market weights for the aggregate price (must sum to 1)
W_AGG, W_RAT, W_BR = 0.2, 0.4, 0.4

# Output directory.  Default = the user's Windows OneDrive path.
# Override on other machines via the PROP7_OUT_DIR environment variable.
OUT_DIR = os.environ.get(
    "PROP7_OUT_DIR",
    r"C:\Users\78558\OneDrive - University of Miami\Desktop\Research\Chad",
)
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================
# 1. NU PATH
# =====================================================
def make_nu_path() -> np.ndarray:
    nu = np.full(T, NU_LOW)
    nu[T_SHOCK:] = NU_HIGH
    return nu


# =====================================================
# 2. SINGLE-PATH SIMULATOR (Section 9 recursive system, verbatim)
# =====================================================
def simulate_replication(rng, nu_path):
    """One Monte Carlo replication.  All three agents face the same
    realization of x_{1,t} and xi_{1,t} (the diagnostic noise u_{1,t+1}
    is drawn separately per agent because its variance eta2_z/m* depends
    on the agent-specific attention).
    """
    x_true = rng.normal(0.0, np.sqrt(nu_path))
    xi     = rng.normal(0.0, np.sqrt(TAU2_X), size=T)

    results = {}

    # ---------- Aggressive: m_t = 1 always ------------------------------
    # x^s_t = (1 - 1)*x^s_{t-1} + 1*(x_t + xi_t) = x_t + xi_t.
    # The agent still runs the Kalman update on nu_hat with m*=1, but
    # nu_hat plays no role in the perceived state because m is forced to 1.
    results['aggressive'] = {
        'x_perc': x_true + xi,
        'm':      np.ones(T),
    }

    # ---------- Rational and Bounded Rational ---------------------------
    for agent in ('rational', 'bounded_rational'):
        kappa = KAPPA_BR if agent == 'bounded_rational' else 0.0
        C_i = kappa + MU**2 * TAU2_X

        x_perc = np.zeros(T)
        m_arr  = np.zeros(T)
        nu_hat = NU0_HAT
        P      = P0
        xs_prev = 0.0

        for t in range(T):
            # Section 6 attention rule (with [0,1] feasibility floor)
            A = MU**2 * (nu_hat + xs_prev**2)
            A = max(A, 1e-12)
            m_star = M_BAR + (1.0 - M_BAR) * A / (A + C_i)
            m_arr[t] = m_star

            # Section 5 perception update
            xs_new = (1.0 - m_star) * xs_prev + m_star * (x_true[t] + xi[t])
            x_perc[t] = xs_new

            # Section 8 diagnostic signal + Kalman update on nu_hat
            sig_u = np.sqrt(ETA2_Z / max(m_star, 1e-12))
            u_t = rng.normal(0.0, sig_u)
            z_t = nu_path[t] + u_t
            lam = m_star * P / (m_star * P + ETA2_Z)
            nu_hat = nu_hat + lam * (z_t - nu_hat)
            P = P * (1.0 - lam)

            xs_prev = xs_new

        results[agent] = {'x_perc': x_perc, 'm': m_arr}

    return x_true, results


# =====================================================
# 3. MONTE CARLO AGGREGATOR
# =====================================================
def run():
    nu_path = make_nu_path()
    rng_master = np.random.default_rng(SEED)

    agents = ('aggressive', 'rational', 'bounded_rational')
    mse        = {a: np.zeros(T) for a in agents}
    m_avg      = {a: np.zeros(T) for a in agents}
    price_err  = np.zeros(T)

    for _ in range(N_SIM):
        seed = rng_master.integers(0, 2**31)
        rng = np.random.default_rng(seed)
        x_true, res = simulate_replication(rng, nu_path)

        for a in agents:
            mse[a]   += (x_true - res[a]['x_perc'])**2 / N_SIM
            m_avg[a] += res[a]['m'] / N_SIM

        P = (W_AGG * res['aggressive']['x_perc']
             + W_RAT * res['rational']['x_perc']
             + W_BR  * res['bounded_rational']['x_perc'])
        price_err += np.abs(P - x_true) / N_SIM

    return mse, m_avg, price_err


# =====================================================
# 4. PLOTTING (style adopted from the user's reduced-form IRF figure)
# =====================================================
def plot_irf(mse, m_avg, price_err, filename):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 13.5,
        "legend.fontsize": 10.5,
        "figure.dpi": 110,
    })

    C_AGG   = "#2E5EAA"   # deep blue
    C_RAT   = "#1B7F3B"   # forest green
    C_BR    = "#C8252C"   # crimson red
    C_PRICE = "#5B2C82"   # purple
    C_SHOCK = "#D62728"

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 13.5), sharex=True)
    t_axis = np.arange(T)

    # Theoretical static-MMSE attention bounds
    m_low_static  = NU_LOW  / (NU_LOW  + TAU2_X)
    m_high_static = NU_HIGH / (NU_HIGH + TAU2_X)
    m_low_full    = M_BAR + (1.0 - M_BAR) * m_low_static
    m_high_full   = M_BAR + (1.0 - M_BAR) * m_high_static

    def shock_line(ax):
        ax.axvline(T_SHOCK, color=C_SHOCK, ls="--", lw=1.6, alpha=0.85)

    # ============ Panel A: Tracking error ============
    ax = axes[0]
    ax.plot(t_axis, mse['aggressive'], color=C_AGG, lw=2.4,
            label=r"Aggressive  ($m_t \equiv 1,\;\kappa=0$)")
    ax.plot(t_axis, mse['rational'], color=C_RAT, lw=2.4,
            label=r"Rational  ($\kappa=0,\;\tau^2_x=$" + f"{TAU2_X})")
    ax.plot(t_axis, mse['bounded_rational'], color=C_BR, lw=2.4,
            label=fr"Bounded Rational  ($\kappa={int(KAPPA_BR)},\;\tau^2_x={TAU2_X}$)")
    shock_line(ax)
    ax.set_ylabel(r"$\mathbb{E}\!\left[(x_t - \hat{x}_t)^2\right]$")
    ax.set_title("Panel A   ·   Tracking Error  (Per-Period Utility Loss)")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.7")

    # Annotate BR's spike
    peak_idx = np.argmax(mse['bounded_rational'])
    peak_val = mse['bounded_rational'][peak_idx]
    ax.annotate("BR underreaction:\nstale low-$\\nu$ prior\nmeets new high-$\\nu$ regime",
                xy=(peak_idx, peak_val),
                xytext=(peak_idx + 18, peak_val * 0.78),
                fontsize=9.5, color=C_BR, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_BR,
                                connectionstyle="arc3,rad=-0.2", lw=1.2))

    # ============ Panel B: Endogenous attention m*_t ============
    ax = axes[1]
    ax.plot(t_axis, m_avg['aggressive'], color=C_AGG, lw=2.4,
            label="Aggressive")
    ax.plot(t_axis, m_avg['rational'], color=C_RAT, lw=2.4,
            label="Rational")
    ax.plot(t_axis, m_avg['bounded_rational'], color=C_BR, lw=2.4,
            label="Bounded Rational")
    shock_line(ax)

    ax.axhline(m_low_full,  color="gray", ls=":", lw=1, alpha=0.7)
    ax.axhline(m_high_full, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(T - 3, m_low_full + 0.025,
            rf"$m^*_{{\rm low}}({{\rm Rat}}) \approx {m_low_full:.2f}$",
            fontsize=9, color="0.35", ha="right")
    ax.text(T - 3, m_high_full + 0.025,
            rf"$m^*_{{\rm high}}({{\rm Rat}}) \approx {m_high_full:.2f}$",
            fontsize=9, color="0.35", ha="right")

    ax.set_ylabel(r"Attention  $m^*_t$")
    ax.set_title("Panel B   ·   Endogenous Attention "
                 "(continuous Section-6 closed form, no discrete trigger)")
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="center right", framealpha=0.95, edgecolor="0.7")

    # Annotate BR's slow continuous ascent (no FOMO jump — that would
    # be the misspecified discrete-trigger model that this script is
    # explicitly NOT implementing).
    br_post = m_avg['bounded_rational'][T_SHOCK:]
    half_post = T_SHOCK + np.argmin(np.abs(br_post - br_post[-1] / 2))
    ax.annotate("BR salience channel:\ncontinuous, slow rise\n(stickiness)",
                xy=(half_post, m_avg['bounded_rational'][half_post]),
                xytext=(half_post + 12, 0.40),
                fontsize=9.5, color=C_BR, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_BR,
                                connectionstyle="arc3,rad=0.2", lw=1.2))

    rat_half = T_SHOCK + np.argmin(np.abs(m_avg['rational'][T_SHOCK:]
                                          - m_high_full * 0.95))
    ax.annotate("Rational: rapid recovery\nto $m^*_{\\rm high}$",
                xy=(rat_half, m_avg['rational'][rat_half]),
                xytext=(rat_half + 12, 0.95),
                fontsize=9.5, color=C_RAT, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_RAT,
                                connectionstyle="arc3,rad=-0.15", lw=1.1))

    # ============ Panel C: Aggregate pricing error ============
    ax = axes[2]
    ax.plot(t_axis, price_err, color=C_PRICE, lw=2.6, zorder=3)
    ax.fill_between(t_axis, 0, price_err, color=C_PRICE, alpha=0.16, zorder=2)
    shock_line(ax)

    ax.set_ylabel(r"$\mathbb{E}\,|P_t - x_t|$")
    ax.set_xlabel(r"Time  $t$")
    ax.set_title("Panel C   ·   Aggregate Pricing Error  "
                 fr"($P_t = {W_AGG}\,\hat{{x}}^A_t + {W_RAT}\,\hat{{x}}^R_t + "
                 fr"{W_BR}\,\hat{{x}}^B_t$)")

    pre_baseline = price_err[max(0, T_SHOCK - 30):T_SHOCK].mean()
    post_steady  = price_err[-30:].mean()
    ymax = price_err.max()
    ax.set_ylim(0, ymax * 1.45)

    # Two horizontal reference lines: pre-shock baseline and new
    # permanent post-shock steady state.  The wedge between them is
    # the permanent excess pricing error caused by BR stickiness.
    ax.axhline(pre_baseline, color="0.30", ls="--", lw=1.1, alpha=0.7)
    ax.axhline(post_steady,  color=C_BR,  ls="--", lw=1.1, alpha=0.7)
    ax.text(2, pre_baseline + 0.05,
            fr"pre-shock baseline $\approx {pre_baseline:.2f}$",
            fontsize=9, color="0.25", ha="left", va="bottom")
    ax.text(T - 2, post_steady + 0.05,
            fr"post-shock steady state $\approx {post_steady:.2f}$",
            fontsize=9, color=C_BR, ha="right", va="bottom")

    ax.text(T_SHOCK - 1.5, ymax * 1.32, "Volatility\nregime shift",
            color=C_SHOCK, fontsize=10, fontweight="bold",
            ha="right", va="center")

    # Permanent-wedge annotation: BR's stickiness produces a permanent
    # increase in aggregate pricing error of roughly post_steady -
    # pre_baseline, which never decays within the simulation horizon.
    wedge = post_steady - pre_baseline
    bracket_x = T - 8
    ax.annotate(
        "", xy=(bracket_x, post_steady), xytext=(bracket_x, pre_baseline),
        arrowprops=dict(arrowstyle="<->", color="0.20", lw=1.4),
    )
    ax.text(bracket_x - 1.2, (pre_baseline + post_steady) / 2,
            f"permanent\nwedge\n$\\approx {wedge:.2f}$",
            fontsize=9.5, color="0.15", ha="right", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    # Brief impact-window annotation just over the spike
    spike_end = T_SHOCK + 8
    y_arrow = ymax * 1.13
    ax.annotate("", xy=(spike_end, y_arrow),
                xytext=(T_SHOCK + 0.3, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="0.30", lw=1.2))
    ax.text((T_SHOCK + spike_end) / 2, y_arrow + ymax * 0.04,
            "impact spike", fontsize=9.5, color="0.25",
            ha="center", va="bottom", style="italic")

    plt.suptitle(
        "Heterogeneous-Agent Impulse Response to a Volatility Regime Shift\n"
        rf"(single long-lived dimension, $\nu$: {NU_LOW}$\to${NU_HIGH} at "
        rf"$t={T_SHOCK}$, $\tau^2_x={TAU2_X}$, $\kappa_{{\rm BR}}={int(KAPPA_BR)}$, "
        rf"$\bar m={M_BAR}$, $T={T}$, $N={N_SIM}$)",
        fontsize=13.5, y=1.005, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {filename}")


# =====================================================
# 5. MAIN
# =====================================================
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    print(f"Running {N_SIM} Monte Carlo paths...")
    mse, m_avg, price_err = run()

    print("\nLong-run MSE (avg over last 30 periods):")
    for a in ('aggressive', 'rational', 'bounded_rational'):
        print(f"  {a:<20s}: {mse[a][-30:].mean():>7.3f}")

    print("\nLong-run attention m* (avg over last 30 periods):")
    for a in ('aggressive', 'rational', 'bounded_rational'):
        print(f"  {a:<20s}: {m_avg[a][-30:].mean():>7.3f}")

    print(f"\nPeak BR MSE: {mse['bounded_rational'].max():.3f}  "
          f"(at t={np.argmax(mse['bounded_rational'])})")
    print(f"Peak aggregate pricing error: {price_err.max():.3f}  "
          f"(at t={np.argmax(price_err)})")

    plot_irf(mse, m_avg, price_err,
             os.path.join(OUT_DIR, "prop7_irf_single_dim.png"))
    print("Done.")
