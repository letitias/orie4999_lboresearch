"""
Monte Carlo simulation of LBO equity returns.

Structure
---------
1. INPUTS      : each uncertain assumption gets a distribution (low / most likely / high)
2. SAMPLING    : draw N joint scenarios (optionally correlated via a Gaussian copula)
3. LBO ENGINE  : run the same deterministic LBO model on every scenario (vectorized)
4. OUTPUTS     : IRR / MOIC distribution, outcome probabilities, variance attribution
"""

import numpy as np
import pandas as pd
from scipy import stats

N_SIMS = 10_000
SEED = 42
TAX_RATE = 0.25          # held fixed (not in the tornado ranges)
EBITDA_0 = 100.0         # scale-free: returns don't depend on company size

# ---------------------------------------------------------------------------
# 1. INPUTS: (low, most likely, high), triangular, same ranges as tornado deck
# ---------------------------------------------------------------------------
INPUTS = {
    "entry_mult":  (10.0,  12.0,  15.5),
    "exit_mult":   (10.0,  12.0,  15.5),
    "ebitda_g":    (0.02,  0.06,  0.10),
    "leverage":    (4.6,   5.2,   5.9),     # Debt / EBITDA at entry
    "rate":        (0.064, 0.0925, 0.101),
    "capex_nwc":   (0.15,  0.25,  0.35),    # % of EBITDA
    "da":          (0.10,  0.20,  0.35),    # % of EBITDA
    "fees":        (0.01,  0.02,  0.03),    # % of entry EV
}
HOLD_YEARS = np.array([5, 6, 7])
HOLD_PROBS = np.array([0.25, 0.50, 0.25])

BASE = {k: v[1] for k, v in INPUTS.items()} | {"hold": 6}

# Optional correlation between inputs (Spearman-ish, applied on normal scores).
# Empty dict = all inputs independent.
CORRELATIONS = {
    # ("entry_mult", "exit_mult"): 0.6,   # expensive markets at entry tend to persist
    # ("ebitda_g", "leverage"):    0.3,   # lenders lend more to faster growers
}


# ---------------------------------------------------------------------------
# 2. SAMPLING
# ---------------------------------------------------------------------------
def triangular_ppf(u, low, mode, high):
    """Map uniform(0,1) draws to a triangular distribution (inverse CDF)."""
    c = (mode - low) / (high - low)
    return stats.triang.ppf(u, c, loc=low, scale=high - low)


def sample_inputs(n=N_SIMS, correlations=CORRELATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    names = list(INPUTS) + ["hold"]
    k = len(names)

    # Correlation matrix on standard-normal scores (Gaussian copula)
    corr = np.eye(k)
    for (a, b), rho in correlations.items():
        i, j = names.index(a), names.index(b)
        corr[i, j] = corr[j, i] = rho
    z = rng.standard_normal((n, k)) @ np.linalg.cholesky(corr).T
    u = stats.norm.cdf(z)                       # correlated uniforms

    draws = {name: triangular_ppf(u[:, i], *INPUTS[name])
             for i, name in enumerate(INPUTS)}
    # Holding period: discrete 5/6/7 with 25/50/25 weights
    cum = np.cumsum(HOLD_PROBS)
    draws["hold"] = HOLD_YEARS[np.searchsorted(cum, u[:, -1])]
    return pd.DataFrame(draws)

EMPIRICAL_PATH = "data/brightquery_company_params.csv"
EMPIRICAL_COLS = ["ebitda_g", "da", "capex_nwc"]

def sample_inputs_empirical(n=N_SIMS, correlations=CORRELATIONS, seed=SEED):
    """Same as sample_inputs, but operating inputs are drawn jointly from real companies."""
    draws = sample_inputs(n, correlations, seed)      # multiples, leverage, rate, fees, hold
    co = pd.read_csv(EMPIRICAL_PATH)
    picked = co.sample(n=n, replace=True, random_state=seed + 1)[EMPIRICAL_COLS]
    draws[EMPIRICAL_COLS] = picked.to_numpy()
    return draws


# ---------------------------------------------------------------------------
# 3. LBO ENGINE (vectorized: every array has one entry per scenario)
# ---------------------------------------------------------------------------
def run_lbo(d):
    entry_ev = d["entry_mult"] * EBITDA_0
    debt = d["leverage"] * EBITDA_0
    equity_in = entry_ev + d["fees"] * entry_ev - debt   # sources = uses
    cash = np.zeros(len(entry_ev))

    exit_equity = np.zeros(len(entry_ev))
    for t in range(1, HOLD_YEARS.max() + 1):
        ebitda = EBITDA_0 * (1 + d["ebitda_g"]) ** t
        interest = d["rate"] * debt                        # on opening balance
        ebt = ebitda - d["da"] * ebitda - interest
        tax = np.maximum(ebt, 0) * TAX_RATE
        fcf = ebitda - interest - tax - d["capex_nwc"] * ebitda

        paydown = np.clip(fcf, 0, debt)                   # 100% cash sweep
        debt = debt - paydown
        cash = cash + (fcf - paydown)                     # excess (or shortfall) to cash

        exits_now = d["hold"] == t
        exit_ev = d["exit_mult"] * ebitda
        exit_equity = np.where(exits_now, np.maximum(exit_ev - debt + cash, 0), exit_equity)

    moic = exit_equity / equity_in
    irr = np.where(moic > 0, moic ** (1 / d["hold"]) - 1, -1.0)   # total loss = -100%
    return pd.DataFrame({"irr": irr, "moic": moic})


# ---------------------------------------------------------------------------
# 4. OUTPUTS
# ---------------------------------------------------------------------------
def summarize(draws, results):
    irr = results["irr"]
    print(f"Runs: {len(irr):,}")
    print(f"Mean IRR    {irr.mean():.1%}   Median IRR {irr.median():.1%}")
    print(f"5th–95th    {irr.quantile(.05):.1%} to {irr.quantile(.95):.1%}")
    print(f"Std dev     {irr.std()*100:.1f} pts   Median MOIC {results['moic'].median():.2f}x")
    print(f"P(IRR < 8%)  {(irr < .08).mean():.1%}")
    print(f"P(IRR > 15%) {(irr > .15).mean():.1%}")
    print(f"P(IRR > 20%) {(irr > .20).mean():.1%}")
    print(f"P(MOIC < 1)  {(results['moic'] < 1).mean():.1%}")

    # Variance attribution: squared Spearman rank correlation, normalized
    rho = draws.apply(lambda col: stats.spearmanr(col, irr)[0])
    share = (rho**2 / (rho**2).sum()).sort_values(ascending=False)
    print("\nShare of IRR variance (squared Spearman, normalized):")
    print((share * 100).round(1).astype(str).add("%").to_string())

    # Convergence check: does the mean IRR stabilize as runs increase?
    print("\nConvergence (mean IRR by number of runs):")
    for n in [100, 1_000, 5_000, len(irr)]:
        print(f"  {n:>6,}: {irr[:n].mean():.2%}")


# ---------------------------------------------------------------------------
# 5. USER INPUT (terminal prompts; press Enter to keep the default)
# ---------------------------------------------------------------------------
# name -> (label shown to user, unit, scale). Percent inputs are typed as 6 for 6%.
LABELS = {
    "entry_mult": ("Entry EV/EBITDA",        "x", 1),
    "exit_mult":  ("Exit EV/EBITDA",         "x", 1),
    "ebitda_g":   ("EBITDA growth",          "%", 100),
    "leverage":   ("Debt/EBITDA",            "x", 1),
    "rate":       ("Interest rate",          "%", 100),
    "capex_nwc":  ("Capex + NWC (% EBITDA)", "%", 100),
    "da":         ("D&A (% EBITDA)",         "%", 100),
    "fees":       ("Fees (% entry EV)",      "%", 100),
}


def ask(prompt, default, cast=float):
    """Ask for one value; Enter keeps the default; re-ask on bad input."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  Not a number, try again.")


def prompt_inputs():
    """Overwrite the module-level settings with the user's values."""
    global INPUTS, BASE, TAX_RATE, HOLD_PROBS, N_SIMS

    print("Enter each input's low / most likely / high. Press Enter to keep the default.")
    print("Percentages are typed as plain numbers, e.g. 6 for 6%.\n")
    new_inputs = {}
    for name, (label, unit, scale) in LABELS.items():
        lo, mode, hi = (round(v * scale, 4) for v in INPUTS[name])
        while True:
            print(f"{label} ({unit})")
            lo_ = ask("  low        ", lo)
            mode_ = ask("  most likely", mode)
            hi_ = ask("  high       ", hi)
            if lo_ <= mode_ <= hi_ and lo_ < hi_:
                break
            print("  Need low <= most likely <= high, with low < high. Try again.")
        new_inputs[name] = (lo_ / scale, mode_ / scale, hi_ / scale)
    INPUTS = new_inputs

    print("Holding period: probability of exiting in year 5 / 6 / 7 (must sum to 100)")
    while True:
        p = np.array([ask(f"  year {y} %", round(d * 100), float)
                      for y, d in zip(HOLD_YEARS, HOLD_PROBS)])
        if abs(p.sum() - 100) < 1e-6 and (p >= 0).all():
            HOLD_PROBS = p / 100
            break
        print("  Probabilities must be non-negative and sum to 100.")

    TAX_RATE = ask("Tax rate (%)", TAX_RATE * 100) / 100
    N_SIMS = ask("Number of runs", N_SIMS, int)

    corr = {}
    rho = ask("Correlation between entry and exit multiple (-0.9 to 0.9, 0 = none)", 0.0)
    if rho:
        corr[("entry_mult", "exit_mult")] = rho
    rho = ask("Correlation between EBITDA growth and leverage (-0.9 to 0.9, 0 = none)", 0.0)
    if rho:
        corr[("ebitda_g", "leverage")] = rho

    BASE = {k: v[1] for k, v in INPUTS.items()} | {
        "hold": int(HOLD_YEARS[np.argmax(HOLD_PROBS)])}
    print()
    return corr


if __name__ == "__main__":
    import sys
    correlations = CORRELATIONS if "--defaults" in sys.argv else prompt_inputs()

    base = run_lbo(pd.DataFrame([BASE]))
    print(f"Base case IRR {base.irr[0]:.1%}, MOIC {base.moic[0]:.2f}x\n")

    print("=== Assumed (triangular) inputs ===")
    draws = sample_inputs(n=N_SIMS, correlations=correlations)
    summarize(draws, run_lbo(draws))

    print("\n=== Operating inputs from BrightQuery companies ===")
    draws_emp = sample_inputs_empirical(n=N_SIMS, correlations=correlations)
    summarize(draws_emp, run_lbo(draws_emp))
