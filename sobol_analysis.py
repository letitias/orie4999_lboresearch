"""
Sobol sensitivity analysis for the LBO Monte Carlo model (uses MonteCarloRun.py unchanged).

Question answered: which uncertain inputs drive the spread in IRR, and do they matter alone or
through interactions with other inputs?

What it computes, per input (and for the holding period):
  S1  first-order index   : share of IRR variance explained by the input alone
  ST  total-order index   : share explained by the input alone PLUS every interaction it is part of
  ST - S1                 : how much of that input's influence comes only through interactions
and, for the top inputs, second-order indices S_ij (how much the PAIR explains beyond the two alone).

Method: Saltelli (2010) first-order and Jansen total-order estimators on scrambled Sobol sequences,
closed-form second-order from the Sobol (1993) estimator, bootstrap 95% intervals.

IMPORTANT LIMIT: Sobol indices assume the inputs are INDEPENDENT. So the run uses rho = 0 for the
entry/exit multiples. It cannot say how much of the variance is "due to" a correlated pair.

Run (from the folder that holds MonteCarloRun.py and data/):
    python sobol_analysis.py
Needs: numpy, pandas, scipy >= 1.7 (for scipy.stats.qmc).
Reads : data/multiple_params.csv                (low / mode / high EV/EBITDA, via MonteCarloRun.py)
        data/brightquery_company_params.csv     (optional; real company inputs from the Dewey notebook)
Writes: sobol_<config>_indices.csv and sobol_<config>_pairs.csv for each configuration it runs.
Takes a couple of minutes: each configuration is about 32,768 x 11 model runs plus bootstrap.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

import MonteCarloRun as mc   # the LBO model: INPUTS, triangular_ppf, run_lbo, HOLD_YEARS / HOLD_PROBS

N_LOG2 = 15           # base sample N = 2**15 = 32,768 (model evaluations = N * (k + 2) + extras)
N_BOOT = 500          # bootstrap resamples used for the 95% intervals
SEED = 42             # same seed as the deck, so results are reproducible


# ---------------------------------------------------------------- estimators
def sobol_indices(fA, fB, fAB, fC_single=None, fC_pair=None, boot=0, rng=None):
    """
    Turn model outputs on the Saltelli sample matrices into Sobol indices.

    fA, fB : model outputs at the A and B base matrices            (N,)
    fAB    : dict i -> outputs at A with column i taken from B     (N,)  [first / total order]
    fC_single : dict i -> outputs at B with column i taken from A  (N,)  [closed second-order]
    fC_pair   : dict (i,j) -> outputs at B with cols i,j from A    (N,)
    Returns (point, ci): point is a dict of index dicts; ci is the same shape holding (lo, hi) 95% bounds,
    or None when boot == 0.

    Intuition: A and B are two independent random samples of the inputs. AB_i is A with only input i
    re-drawn (taken from B). If input i matters, changing it changes the output.
    """
    def est(idx):
        # `idx` selects which rows to use, so the same code serves the point estimate (all rows) and
        # each bootstrap resample (rows drawn with replacement).
        a, b = fA[idx], fB[idx]
        var = np.var(np.concatenate([a, b]), ddof=1)      # total output variance
        out = {"S1": {}, "ST": {}}
        for i, ab in fAB.items():
            ab = ab[idx]
            out["S1"][i] = np.mean(b * (ab - a)) / var                 # Saltelli 2010: variance from input i alone
            out["ST"][i] = 0.5 * np.mean((a - ab) ** 2) / var          # Jansen: alone + all interactions
        if fC_single is not None and fC_pair is not None:
            # Second-order: variance explained by the pair {i, j} together, minus what i and j explain alone.
            ab_mean = np.mean(a * b)                                     # estimates f0^2 (squared mean output)
            vc1 = {i: (np.mean(a * c[idx]) - ab_mean) / var for i, c in fC_single.items()}
            out["S2"] = {}
            for (i, j), c in fC_pair.items():
                vcij = (np.mean(a * c[idx]) - ab_mean) / var
                out["S2"][(i, j)] = vcij - vc1[i] - vc1[j]
        return out

    n = len(fA)
    point = est(np.arange(n))
    if not boot:
        return point, None
    # Bootstrap: recompute every index on `boot` resamples of the N rows, then take the 2.5th and 97.5th
    # percentiles as the 95% interval.
    draws = [est(rng.integers(0, n, n)) for _ in range(boot)]
    ci = {}
    for key in point:
        ci[key] = {k: (np.percentile([d[key][k] for d in draws], 2.5),
                       np.percentile([d[key][k] for d in draws], 97.5)) for k in point[key]}
    return point, ci


def saltelli_run(model, k, n_log2=N_LOG2, seed=SEED, pairs=(), boot=0):
    """
    Build the Saltelli sample matrices, run `model` on each, and return the indices.

    model: function taking an (N, k) matrix of uniforms in [0,1) and returning (N,) outputs.
    k    : number of independent factors (columns).
    pairs: list of (i, j) column pairs to also compute second-order indices for.
    Returns ((point, ci), (fA, fB)); fA and fB are returned so callers can check the output mean / spread.
    """
    # One scrambled Sobol sequence with 2k columns, split into two independent N x k matrices A and B.
    sob = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    M = sob.random_base2(m=n_log2)
    A, B = M[:, :k], M[:, k:]
    fA, fB = model(A), model(B)
    fAB, fC1, fCp = {}, {}, {}
    for i in range(k):
        X = A.copy(); X[:, i] = B[:, i]; fAB[i] = model(X)          # A with column i from B
    if pairs:
        # Extra matrices for second-order indices: B with one or two columns taken from A.
        singles = sorted({i for p in pairs for i in p})
        for i in singles:
            X = B.copy(); X[:, i] = A[:, i]; fC1[i] = model(X)       # B with column i from A
        for (i, j) in pairs:
            X = B.copy(); X[:, [i, j]] = A[:, [i, j]]; fCp[(i, j)] = model(X)
    rng = np.random.default_rng(seed + 1)   # separate stream for the bootstrap
    return sobol_indices(fA, fB, fAB, fC1 or None, fCp or None, boot=boot, rng=rng), (fA, fB)


# ---------------------------------------------------------------- validation: Ishigami function
def _ishigami(U, a=7.0, b=0.1):
    """Standard sensitivity-analysis test function with known Sobol indices (used only by validate())."""
    X = -np.pi + 2 * np.pi * U      # map uniforms to [-pi, pi]
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def validate():
    """
    Check the estimators against a function whose true indices are known, so we know the code is right
    before trusting it on the LBO model.
    Known answers: S1 = [0.3139, 0.4424, 0], ST = [0.5576, 0.4424, 0.2437], S13 = 0.2437.
    """
    (pt, _), _ = saltelli_run(_ishigami, 3, n_log2=16, pairs=[(0, 2)])
    print("Estimator check on the Ishigami function (known S1 = 0.314 / 0.442 / 0.000,"
          " ST = 0.558 / 0.442 / 0.244, S_13 = 0.244):")
    print("  S1 :", [round(pt["S1"][i], 3) for i in range(3)])
    print("  ST :", [round(pt["ST"][i], 3) for i in range(3)])
    print("  S13:", round(pt["S2"][(0, 2)], 3))


# ---------------------------------------------------------------- LBO model as a function of uniforms
# Factor order = the 8 inputs in MonteCarloRun.INPUTS order, then the holding period.
NAMES = list(mc.INPUTS) + ["hold"]
LABEL = {"entry_mult": "Entry multiple", "exit_mult": "Exit multiple", "ebitda_g": "EBITDA growth",
         "leverage": "Leverage", "rate": "Interest rate", "capex_nwc": "Capex + NWC",
         "da": "Depreciation", "fees": "Fees", "hold": "Hold period"}


def make_model(inputs, output="irr", empirical=None):
    """
    Wrap the LBO as a function of uniforms, which is what the Saltelli sampler needs.

    Factors are the 8 inputs (in mc.INPUTS order) + hold.
    inputs   : dict name -> (low, mode, high) triangular parameters (usually mc.INPUTS, possibly with
               the multiples overridden).
    output   : "irr" returns IRR; anything else returns the 0/1 indicator MOIC < 1.
    empirical: optional DataFrame of real companies (columns ebitda_g, da, capex_nwc). If given, those
    three inputs use the companies' EMPIRICAL marginal distributions (still drawn independently).
    """
    def model(U):
        draws = {}
        for i, name in enumerate(mc.INPUTS):
            if empirical is not None and name in mc.EMPIRICAL_COLS:
                # empirical quantile function: uniform u -> the u-th quantile of the real companies' values
                draws[name] = np.quantile(empirical[name].to_numpy(), U[:, i])
            else:
                draws[name] = mc.triangular_ppf(U[:, i], *inputs[name])
        # Holding period is discrete (5 / 6 / 7 years); map the last uniform column through the cumulative probabilities.
        cum = np.cumsum(mc.HOLD_PROBS)
        draws["hold"] = mc.HOLD_YEARS[np.searchsorted(cum, U[:, -1])]
        res = mc.run_lbo(pd.DataFrame(draws))
        return res["irr"].to_numpy() if output == "irr" else (res["moic"].to_numpy() < 1).astype(float)
    return model


# Grouped version: the three operating inputs become ONE factor ("operating"), so a real company's
# growth, capex + NWC and D&A stay together, as in the Monte Carlo runs.
GROUP_NAMES = ["entry_mult", "exit_mult", "leverage", "rate", "fees", "hold", "operating"]
LABEL["operating"] = "Operating bundle (one real company: growth + capex/NWC + D&A)"


def make_grouped_model(inputs, empirical):
    """The three operating inputs become ONE factor: pick a real company, keep its three values together."""
    def model(U):
        # first five columns: entry, exit, leverage, rate, fees (triangular)
        draws = {n: mc.triangular_ppf(U[:, i], *inputs[n]) for i, n in enumerate(GROUP_NAMES[:5])}
        cum = np.cumsum(mc.HOLD_PROBS)
        draws["hold"] = mc.HOLD_YEARS[np.searchsorted(cum, U[:, 5])]
        # last column picks one company row; min() guards the u == 1 edge case
        row = np.minimum((U[:, 6] * len(empirical)).astype(int), len(empirical) - 1)
        pick = empirical.iloc[row]
        for c in mc.EMPIRICAL_COLS:
            draws[c] = pick[c].to_numpy()
        return mc.run_lbo(pd.DataFrame(draws))["irr"].to_numpy()
    return model


def table(point, ci, names=NAMES):
    """Tidy table of S1 / ST (with 95% bounds) per input, sorted by total-order index."""
    rows = []
    for i, n in enumerate(names):
        rows.append({"input": LABEL[n], "S1": point["S1"][i], "S1_lo": ci["S1"][i][0], "S1_hi": ci["S1"][i][1],
                     "ST": point["ST"][i], "ST_lo": ci["ST"][i][0], "ST_hi": ci["ST"][i][1],
                     "ST_minus_S1": point["ST"][i] - point["S1"][i]})
    return pd.DataFrame(rows).sort_values("ST", ascending=False).reset_index(drop=True)


def spearman_share(inputs):
    """
    The deck's existing 'share of IRR variance' (squared Spearman, normalized), same inputs, rho = 0.
    Included so Sobol can be compared with the method used on earlier slides.
    """
    old = mc.INPUTS
    mc.INPUTS = inputs                      # temporarily swap the model's inputs...
    try:
        draws = mc.sample_inputs(correlations={})
        irr = mc.run_lbo(draws)["irr"]
    finally:
        mc.INPUTS = old                     # ...and always restore them
    rho = draws.apply(lambda c: stats.spearmanr(c, irr)[0])
    return (rho ** 2 / (rho ** 2).sum()).rename(index=LABEL)


def run_config(label, inputs, out_csv, empirical=None, grouped=False):
    """
    Run and print one full Sobol analysis, and save its tables.

    label    : heading to print
    inputs   : triangular (low, mode, high) dict for the model
    out_csv  : filename prefix for the two output csvs
    empirical: optional real-company DataFrame (see make_model)
    grouped  : True = treat the three operating inputs as one company-draw factor
    """
    print(f"\n=================== {label} ===================")
    names = GROUP_NAMES if grouped else NAMES
    k = len(names)
    model = make_grouped_model(inputs, empirical) if grouped else make_model(inputs, empirical=empirical)
    # Quick, smaller run (N = 2**13) only to rank the inputs, so the expensive second-order indices
    # are computed for the 4 most influential ones (6 pairs) rather than all 36 pairs.
    (pt0, _), _ = saltelli_run(model, k, n_log2=13)
    top = sorted(range(k), key=lambda i: -pt0["ST"][i])[:4]
    pairs = [(a, b) for ai, a in enumerate(top) for b in top[ai + 1:]]

    # Full run with second-order pairs and bootstrap intervals.
    (pt, ci), (fA, fB) = saltelli_run(model, k, pairs=pairs, boot=N_BOOT)
    t = table(pt, ci, names)
    if empirical is None:
        # only meaningful for the triangular configs (the old method draws operating inputs from triangles)
        t["old_squared_spearman_share"] = t["input"].map(spearman_share(inputs))
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    # If the model were purely additive, the S1 values would sum to 1; the shortfall is interactions.
    print(f"\nSum of S1 = {sum(pt['S1'].values()):.3f}   (1 - sum S1 = {1 - sum(pt['S1'].values()):.3f}"
          " of variance comes from interactions)")
    print("Sum of ST = %.3f" % sum(pt["ST"].values()))
    print(f"IRR variance check: mean {np.mean(np.r_[fA, fB]):.3%}, sd {np.std(np.r_[fA, fB]) * 100:.2f} pts")

    p2 = pd.DataFrame([{"pair": f"{LABEL[names[i]]} x {LABEL[names[j]]}", "S_ij": v,
                        "lo": ci["S2"][(i, j)][0], "hi": ci["S2"][(i, j)][1]}
                       for (i, j), v in pt["S2"].items()]).sort_values("S_ij", ascending=False)
    print("\nSecond-order (pair) indices among the top-4 inputs:")
    print(p2.round(3).to_string(index=False))
    t.to_csv(out_csv + "_indices.csv", index=False)
    p2.to_csv(out_csv + "_pairs.csv", index=False)
    return t, p2


if __name__ == "__main__":
    validate()                                              # prove the estimators work before using them
    dam = dict(mc.INPUTS)                                   # picks up data/multiple_params.csv (Damodaran)
    orig = dict(mc.INPUTS); orig["entry_mult"] = orig["exit_mult"] = (10.0, 12.0, 15.5)   # the deck's original guess
    run_config("Original triangular multiples (10 / 12 / 15.5x)", orig, "sobol_original")
    run_config("Damodaran multiples (8.14 / 12.47 / 24.11x)", dam, "sobol_damodaran")

    # Optional: real BrightQuery operating inputs (needs data/brightquery_company_params.csv from the notebook)
    import os
    if os.path.exists(mc.EMPIRICAL_PATH):
        co = pd.read_csv(mc.EMPIRICAL_PATH, index_col=0)
        print(f"\nLoaded {len(co):,} companies from {mc.EMPIRICAL_PATH}")
        # (a) each operating input drawn independently from its own real distribution
        run_config("Damodaran multiples + REAL operating inputs (empirical marginals, drawn independently)",
                   dam, "sobol_dewey_marginals", empirical=co)
        # (b) one real company supplies all three together (matches how the Monte Carlo runs use the data)
        run_config("Damodaran multiples + REAL operating inputs (one company draw keeps the 3 inputs together)",
                   dam, "sobol_dewey_grouped", empirical=co, grouped=True)
    else:
        print(f"\n(skipped Dewey operating-input runs: {mc.EMPIRICAL_PATH} not found)")
