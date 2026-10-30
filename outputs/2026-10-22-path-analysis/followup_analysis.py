"""Follow-up to path_analysis.py: the three next steps from slide 23.

1. Cash needs with cleaner reinvestment. A year in which gross PP&E jumps by more than 50% is treated as
   non-operating spending (an acquisition or the 2018-19 lease-accounting change). That year's capex is
   replaced by the company's typical capex rate (median capex / EBITDA over its other positive-EBITDA years).
2. Pre-purchase risk model. Features use only years up to the purchase year. A logistic regression is fit on
   purchase years 2013-2015 and evaluated on 2016-2018, against the team's profitability screen.
3. Real price moves. Damodaran's archived US EV/EBITDA files (Jan 2001 - Jan 2026, data/damodaran_archive)
   give each industry's actual multiple change over 5-7 years. Exit multiple = entry multiple x that change,
   compared with the assumed copula correlation (0, 0.6, 0.9).

Everything else (prices, leverage, rate, fees, hold, seed 42) is drawn exactly as in path_analysis.py.
Requirements: numpy, pandas, scipy, xlrd, scikit-learn.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DAM_DIR = ROOT / "data/damodaran_archive"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("path_analysis", OUT / "path_analysis.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)
ra = pa.ra
SEED, H = pa.SEED, pa.H
JUMP = 0.5            # gross PP&E growth above this in one year = non-operating (acquisition / accounting)
K = 1000              # financing draws per window for window-level loss probabilities
B = 500               # bootstrap replicates
TRAIN, TEST = range(2013, 2016), range(2016, 2019)
FEATURES = {
    "margin": "EBITDA margin in the purchase year",
    "margin_change_3y": "Change in EBITDA margin over the 3 years before purchase",
    "margin_volatility": "Standard deviation of EBITDA margin, purchase year and 3 years before",
    "revenue_growth_3y": "Annual revenue growth over the 3 years before purchase",
    "loss_year_before": "Any loss-making year in the 3 years before purchase (0/1)",
    "log_revenue": "Size: log of purchase-year revenue",
    "profit_spike": "Purchase-year EBITDA minus prior 3-year average, as a share of revenue",
    "reinvestment_rate": "Median reinvestment / EBITDA up to purchase",
    "depreciation_rate": "Median depreciation / EBITDA up to purchase",
}


# ---------------------------------------------------------------- shared data
def panel_parts():
    d = pa.panel()
    prev = d.groupby("id")[["year", "gross_ppe", "nwc"]].shift(1)
    ok = prev.year == d.year - 1
    d["capex"] = np.where(ok, d.gross_ppe - prev.gross_ppe, np.nan)
    d["dnwc"] = np.where(ok, d.nwc - prev.nwc, np.nan)
    d["ppe_jump"] = ok & (prev.gross_ppe > 0) & (d.gross_ppe / prev.gross_ppe - 1 > JUMP)
    usable = (d.ebitda > 0) & d.capex.notna() & ~d.ppe_jump
    typical = (d.capex / d.ebitda).where(usable).groupby(d.id).median()
    d["typical_capex_rate"] = d.id.map(typical)
    return d


def window_arrays(d, w):
    """(windows x 7) capex, change in NWC (both / purchase EBITDA), jump flags, typical capex rate."""
    piv = {c: d.pivot(index="id", columns="year", values=c) for c in ["ebitda", "capex", "dnwc", "ppe_jump"]}
    rate = d.groupby("id").typical_capex_rate.first()
    CI, NI, JF = [], [], []
    for i, s in w.itertuples(index=False):
        yrs = list(range(s + 1, s + H + 1))
        e0 = piv["ebitda"].at[i, s]
        CI.append(piv["capex"].loc[i, yrs].to_numpy(dtype=float) / e0)
        NI.append(piv["dnwc"].loc[i, yrs].to_numpy(dtype=float) / e0)
        JF.append(piv["ppe_jump"].loc[i, yrs].fillna(False).to_numpy(dtype=bool))
    return np.array(CI), np.array(NI), np.array(JF), rate.reindex(w.id).to_numpy()


def load_damodaran():
    """Industry x year EV/EBITDA panel. File vebitdaYY.xls is dated January 20YY+1 (checked on every file
    that carries a date); vebitda.xls in the project root is January 2026."""
    norm = lambda x: " ".join(str(x).split())
    excl = r"Bank|Brokerage|Financial Svcs|Insurance|Reinsurance|Investments|R\.E\.I\.T|REIT|Real Estate|Total Market|Grand Total|^Market$"
    files = [(2000 + int(p.stem[-2:]) + 1, p) for p in sorted(DAM_DIR.glob("vebitda[0-9][0-9].xls"))]
    files.append((2026, ROOT / "vebitda.xls"))
    rows = []
    for year, f in files:
        xl = pd.ExcelFile(f)
        sheet = "Industry Averages" if "Industry Averages" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(f, sheet_name=sheet, header=None)
        h = df.index[df[0].map(norm).isin(["Industry Name", "Industry"])][0]
        hdr = [norm(x) for x in df.iloc[h]]
        col = next(i for i, c in enumerate(hdr) if c in ("EV/EBITDA", "Value/EBITDA"))  # positive-EBITDA firms from 2018
        t = df.iloc[h + 1:, [0, col]].copy()
        t.columns = ["industry", "ev_ebitda"]
        t["industry"] = t.industry.map(norm)
        t["ev_ebitda"] = pd.to_numeric(t.ev_ebitda, errors="coerce")
        t = t[(t.industry != "nan") & ~t.industry.str.contains(excl) & (t.ev_ebitda > 0)]
        t["year"] = year
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def multiple_pairs(panel):
    """Same industry name at entry year Y and exit year Y + h, h = 5, 6, 7."""
    wide = panel.pivot_table(index="industry", columns="year", values="ev_ebitda", aggfunc="first")
    out = []
    for h in (5, 6, 7):
        for y in wide.columns:
            if y + h in wide.columns:
                both = wide[[y, y + h]].dropna()
                out += [(ind, y, h, a, b) for ind, (a, b) in zip(both.index, both.to_numpy())]
    return pd.DataFrame(out, columns=["industry", "entry_year", "hold", "entry", "exit"])


def window_loss_probability(fin_k, EI, RI, DI, rows):
    """For each window row: share of the K financing draws that lose money (MOIC < 1)."""
    k = len(fin_k)
    out = np.empty(len(rows))
    for start in range(0, len(rows), 500):
        r_ = rows[start:start + 500]
        fin = pd.concat([fin_k] * len(r_), ignore_index=True)
        idx = np.repeat(r_, k)
        res, _ = pa.path_engine(fin, EI[idx], RI[idx], DI[idx])
        out[start:start + len(r_)] = (res.moic.to_numpy() < 1).reshape(len(r_), k).mean(axis=1)
    return out


def company_mean(values, ids):
    """Company-equal average: average within each company first, then across companies."""
    return float(pd.Series(values).groupby(np.asarray(ids)).mean().mean())


def features(d, w):
    g = d.set_index(["id", "year"]).sort_index()
    rows = []
    for i, s in w.itertuples(index=False):
        h = g.loc[i].loc[s - 3:s]
        if len(h) < 4 or list(h.index) != list(range(s - 3, s + 1)) or (h.revenue <= 0).any() or h[["ebitda", "revenue"]].isna().any().any():
            rows.append([np.nan] * len(FEATURES))
            continue
        assert h.index.max() <= s, "features must use pre-purchase years only"
        m = h.ebitda / h.revenue
        hist = g.loc[i].loc[:s]
        rows.append([m.iloc[-1], m.iloc[-1] - m.iloc[0], m.std(ddof=0),
                     (h.revenue.iloc[-1] / h.revenue.iloc[0]) ** (1 / 3) - 1, float((h.ebitda.iloc[:-1] <= 0).any()),
                     np.log(h.revenue.iloc[-1]), (h.ebitda.iloc[-1] - h.ebitda.iloc[:-1].mean()) / h.revenue.iloc[-1],
                     hist.rv_ratio.median(), hist.da_ratio.median()])
    return pd.DataFrame(rows, columns=list(FEATURES), index=w.index)


# ---------------------------------------------------------------- main
def main():
    co, _, _ = pa.main_sample()
    _, mult = ra.multiples()
    d = panel_parts()
    w, arrays, _ = pa.build_windows(d, pa.ENTRY_ALL)
    EI, RI, DI = arrays
    CI, NI, JF, TR = window_arrays(d, w)
    assert np.allclose(CI + NI, RI, rtol=1e-12, atol=1e-12), "capex + change in NWC must equal reinvestment"
    in_main = w.id.isin(co.index).to_numpy()
    g1 = pa.groups_of(w, in_main)
    checks, rows = {}, []

    # ============ 1. Cash needs with cleaner reinvestment (experiment-1 companies, combined spec)
    fin = pa.financing("combined", mult=mult)
    rng = np.random.default_rng(SEED + 1)
    picked = pa.pick(g1, len(fin), rng)            # identical picks to path_analysis.run_paths
    ei, ri, di = EI[picked], RI[picked], DI[picked]
    prof = np.maximum(ei, 0)
    rate = TR[picked][:, None]
    capex_clean = np.where(JF[picked], rate * prof, CI[picked])
    ri_clean = capex_clean + NI[picked]
    cid = w.id.to_numpy()[picked]
    ri_median = co.capex_nwc.reindex(cid).to_numpy()[:, None] * prof
    dd = ra.draw(co, mult, .6, SEED)
    t = np.arange(1, H + 1)
    grow = (1 + dd.ebitda_g.to_numpy()[:, None]) ** t
    variants = [
        ("Constant growth (main sample)", dd, (grow, dd.capex_nwc.to_numpy()[:, None] * grow, dd.da.to_numpy()[:, None] * grow)),
        ("Year-by-year, reinvestment as reported", fin, (ei, ri, di)),
        ("Year-by-year, acquisition and accounting jumps removed", fin, (ei, ri_clean, di)),
        ("Year-by-year, reinvestment at company's typical rate", fin, (ei, ri_median, di)),
    ]
    for label, f_, (a, b, c) in variants:
        r, s, mc_ = pa.path_engine(f_, a, b, c, return_min_cash=True)
        need = -mc_[s]
        rows.append({"analysis": "1_cash_needs", "case": label, "loss_pct": float((r.moic < 1).mean() * 100),
                     "total_loss_pct": float((r.moic == 0).mean() * 100), "median_irr_pct": float(r.irr.median() * 100),
                     "cash_shortfall_pct": float(s.mean() * 100),
                     "median_peak_need_pct_of_purchase_ebitda": float(np.median(need)) if len(need) else 0.0,
                     "share_need_over_half_purchase_ebitda_pct": float((need > 50).sum() / len(s) * 100)})
    r0, s0 = ra.funding_engine(dd)
    assert np.isclose((r0.moic < 1).mean() * 100, rows[0]["loss_pct"]), "constant-growth check must match review_analysis"
    yrs_main = JF[in_main]
    checks["cash_needs"] = {"share_window_years_flagged_pct": float(yrs_main.mean() * 100),
                            "share_windows_with_a_flagged_year_pct": float(yrs_main.any(axis=1).mean() * 100),
                            "flag_rate_by_calendar_year_pct": {int(y): float(v * 100) for y, v in
                                                               d[d.id.isin(co.index) & d.capex.notna()].groupby("year").ppe_jump.mean().items()}}
    print("CASH NEEDS done", flush=True)

    # ============ 2. Pre-purchase risk model
    pool = w.s.isin(list(TRAIN) + list(TEST)).to_numpy()
    hist = pa.history_stats(d, w[pool])
    keep_idx = hist.index[(hist.n_years >= 4) & hist.da_med.between(*pa.DA_BOUNDS) & hist.rv_med.between(*pa.RV_BOUNDS)]
    X = features(d, w.loc[keep_idx])
    X = X.dropna()
    idx = X.index.to_numpy()
    fin_k = fin.iloc[:K].reset_index(drop=True)
    p = window_loss_probability(fin_k, EI, RI, DI, idx)
    s_year = w.s.to_numpy()[idx]
    ids = w.id.to_numpy()[idx]
    tr, te = np.isin(s_year, list(TRAIN)), np.isin(s_year, list(TEST))
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    Xc = X.clip(lo, hi, axis=1)
    mu, sd = Xc[tr].mean(), Xc[tr].std(ddof=0)
    Z = ((Xc - mu) / sd).to_numpy()
    # fractional outcome: each window enters once as a loss (weight p) and once as a gain (weight 1 - p)
    Xtr = np.vstack([Z[tr], Z[tr]])
    ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
    wtr = np.r_[p[tr], 1 - p[tr]]
    model = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr, sample_weight=wtr)
    risk = model.predict_proba(Z)[:, 1]
    coef = pd.DataFrame({"feature": list(FEATURES), "description": list(FEATURES.values()),
                         "standardized_coefficient": model.coef_[0]}).sort_values("standardized_coefficient", key=np.abs, ascending=False)
    coef.to_csv(OUT / "prediction_coefficients.csv", index=False)
    # screen on the same windows (pre-purchase years only), as in path_analysis experiment 2
    hs = hist.loc[idx]
    screen_pass = ((hs.n_years >= 5) & hs.always_pos & (hs.margin_med >= .10)).to_numpy()
    te_idx = np.flatnonzero(te)
    cuts = np.quantile(risk[te], [1 / 3, 2 / 3])
    tercile = np.digitize(risk, cuts)          # 0 = lowest predicted risk
    groups = {"All test windows": te, "Model: lowest-risk third": te & (tercile == 0),
              "Model: middle third": te & (tercile == 1), "Model: highest-risk third": te & (tercile == 2),
              "Screen: pass": te & screen_pass, "Screen: fail": te & ~screen_pass}
    for name, m in groups.items():
        rows.append({"analysis": "2_prediction", "case": name, "windows": int(m.sum()), "companies": int(len(np.unique(ids[m]))),
                     "loss_pct": company_mean(p[m] * 100, ids[m]), "predicted_pct": company_mean(risk[m] * 100, ids[m])})
    # out-of-sample checks
    rho_s = stats.spearmanr(risk[te], p[te]).statistic
    train_ids = set(ids[tr])
    unseen = te & ~np.isin(ids, list(train_ids))
    unseen_rows = {}
    for name, m in [("lowest", unseen & (tercile == 0)), ("highest", unseen & (tercile == 2)), ("all", unseen)]:
        unseen_rows[name] = {"windows": int(m.sum()), "loss_pct": company_mean(p[m] * 100, ids[m])}
    # bootstrap by company on the test set
    rng_b = np.random.default_rng(2027)
    te_ids = np.unique(ids[te])
    by_id = pd.Series(np.arange(len(idx))).groupby(ids).apply(lambda s_: s_.to_numpy()).to_dict()
    boot = []
    for _ in range(B):
        sample = rng_b.choice(te_ids, size=len(te_ids), replace=True)
        rr = np.concatenate([by_id[c] for c in sample])
        rr = rr[te[rr]]
        rid = np.concatenate([[k_] * int(te[by_id[c]].sum()) for k_, c in enumerate(sample)])
        def gm(mask):
            return company_mean(p[rr][mask] * 100, rid[mask])
        hi_ = gm(tercile[rr] == 2); lo_ = gm(tercile[rr] == 0)
        fa_ = gm(~screen_pass[rr]); pa_ = gm(screen_pass[rr])
        boot.append({"model_gap": hi_ - lo_, "screen_gap": fa_ - pa_, "model_minus_screen_gap": (hi_ - lo_) - (fa_ - pa_)})
    boot = pd.DataFrame(boot)
    pred_ci = boot.quantile([.025, .5, .975]).T
    pred_ci.columns = ["p2.5", "p50", "p97.5"]
    pred_ci.to_csv(OUT / "prediction_bootstrap_intervals.csv")
    # cross-check: window-level probabilities vs a 10,000-deal simulation of the test set
    r_mc, _, _ = pa.run_paths(fin, arrays, pa.groups_of(w, np.isin(np.arange(len(w)), idx[te])), SEED)
    checks["prediction"] = {
        "windows_train": int(tr.sum()), "windows_test": int(te.sum()),
        "companies_train": int(len(np.unique(ids[tr]))), "companies_test": int(len(np.unique(ids[te]))),
        "dropped_missing_features": int(len(keep_idx) - len(idx)),
        "spearman_predicted_vs_actual_test": float(rho_s), "unseen_company_test": unseen_rows,
        "test_loss_window_level_pct": company_mean(p[te] * 100, ids[te]),
        "test_loss_10000_deal_simulation_pct": float((r_mc.moic < 1).mean() * 100),
        "screen_pass_share_test_pct": float(screen_pass[te].mean() * 100),
    }
    pd.DataFrame({"id": ids, "purchase_year": s_year, "loss_probability": p, "predicted_risk": risk,
                  "screen_pass": screen_pass, "set": np.where(tr, "train", "test"), **{k_: X[k_].to_numpy() for k_ in FEATURES}}
                 ).to_csv(OUT / "prediction_windows.csv", index=False)
    print("PREDICTION done", flush=True)

    # ============ 3. Real price moves
    panel = load_damodaran()
    t26 = panel[panel.year == 2026].set_index("industry").ev_ebitda.sort_index()
    ref, _ = ra.multiples()
    ref = ref.set_index("industry").ev_ebitda.sort_index()
    assert t26.index.equals(ref.index) and np.allclose(t26.to_numpy(), ref.to_numpy()), "2026 parse must match review_analysis"
    pairs = multiple_pairs(panel)
    pairs["ratio"] = pairs.exit / pairs.entry
    pairs["period"] = np.where(pairs.entry_year <= 2013, "2001-2013 entries", "2014-2021 entries")
    pairs.to_csv(OUT / "multiple_pairs.csv", index=False)
    summ = pairs.groupby(["period", "hold"]).agg(pairs=("ratio", "size"), industries=("industry", "nunique"),
                                                 median_ratio=("ratio", "median"), p10_ratio=("ratio", lambda x: x.quantile(.1)),
                                                 p90_ratio=("ratio", lambda x: x.quantile(.9)),
                                                 sd_log_ratio=("ratio", lambda x: np.log(x).std()))
    summ.to_csv(OUT / "multiple_pairs_summary.csv")
    # copula: dispersion of log(exit / entry) and rank correlation implied by each rho, on Damodaran 2026 triangle
    ra.mc.INPUTS = dict(ra.ORIGINAL)
    ra.mc.INPUTS["entry_mult"] = ra.mc.INPUTS["exit_mult"] = mult
    grid = np.round(np.arange(0, .96, .05), 2)
    sim_sd, sim_sp = [], []
    for rho in grid:
        dz = ra.mc.sample_inputs(n=200_000, correlations={("entry_mult", "exit_mult"): rho}, seed=SEED)
        sim_sd.append(np.log(dz.exit_mult / dz.entry_mult).std())
        sim_sp.append(stats.spearmanr(dz.entry_mult, dz.exit_mult).statistic)
    sim_sd, sim_sp = np.array(sim_sd), np.array(sim_sp)

    def implied_rho(sd):  # sd falls as rho rises
        return float(np.interp(sd, sim_sd[::-1], grid[::-1]))

    hist_sd = float(np.log(pairs.ratio).std())
    imp = {"pooled": implied_rho(hist_sd)}
    for per, g in pairs.groupby("period"):
        imp[per] = implied_rho(np.log(g.ratio).std())
    rng_i = np.random.default_rng(2028)
    inds = pairs.industry.unique()
    by_ind = {k_: v.ratio.to_numpy() for k_, v in pairs.groupby("industry")}
    boot_rho = [implied_rho(np.log(np.concatenate([by_ind[c] for c in rng_i.choice(inds, len(inds))])).std()) for _ in range(B)]
    checks["price_moves"] = {
        "years": sorted(int(y) for y in panel.year.unique()), "pairs": int(len(pairs)), "industries": int(pairs.industry.nunique()),
        "sd_log_ratio_hist": hist_sd, "implied_rho": imp,
        "hist_sd_within_copula_range": bool(sim_sd.min() <= hist_sd <= sim_sd.max()),
        "implied_rho_bootstrap_95": [float(np.quantile(boot_rho, .025)), float(np.quantile(boot_rho, .975))],
        "copula_sd_log_ratio_by_rho": dict(zip(map(float, grid), map(float, sim_sd))),
        "copula_spearman_at_0.6": float(sim_sp[list(grid).index(.6)]),
        "hist_spearman_entry_exit": float(stats.spearmanr(pairs.entry, pairs.exit).statistic),
        "share_exit_2_turns_below_entry_hist_pct": float(((pairs.entry - pairs.exit) > 2).mean() * 100),
        "median_ratio": float(pairs.ratio.median()),
    }
    pd.DataFrame({"rho": grid, "sd_log_exit_over_entry": sim_sd, "spearman_entry_exit": sim_sp}).to_csv(OUT / "copula_dispersion.csv", index=False)

    # model runs: constant growth (main sample) and year-by-year paths (experiment-1 companies)
    ratio_by_hold = {h: pairs.loc[pairs.hold == h, "ratio"].to_numpy() for h in (5, 6, 7)}

    def hist_exit(df, seed, pool=None):
        out = df.copy()
        rng_h = np.random.default_rng(seed)
        r_ = np.empty(len(out))
        for h in (5, 6, 7):
            m = out.hold.to_numpy() == h
            src = ratio_by_hold[h] if pool is None else pool[h]
            r_[m] = src[rng_h.integers(len(src), size=m.sum())]
        out["exit_mult"] = out.entry_mult.to_numpy() * r_
        return out

    for model_name in ["Constant growth (main sample)", "Year-by-year paths"]:
        for label, rho in [("rho = 0", 0.), ("rho = 0.3", .3), ("rho = 0.6", .6), ("rho = 0.9", .9), ("Historical industry moves", None),
                           ("Historical, 2001-2013 entries", "old"), ("Historical, 2014-2021 entries", "new")]:
            base = ra.draw(co if model_name.startswith("Constant") else None, mult, 0. if rho is None or isinstance(rho, str) else rho, SEED)
            if rho is None:
                base = hist_exit(base, SEED + 3)
            elif isinstance(rho, str):
                per = "2001-2013 entries" if rho == "old" else "2014-2021 entries"
                pool_ = {h: pairs.loc[(pairs.hold == h) & (pairs.period == per), "ratio"].to_numpy() for h in (5, 6, 7)}
                base = hist_exit(base, SEED + 3, pool_)
            if model_name.startswith("Constant"):
                r, s = ra.funding_engine(base)
            else:
                r, s, _ = pa.run_paths(base, arrays, g1, SEED)
            rows.append({"analysis": "3_price_moves", "model": model_name, "case": label,
                         "loss_pct": float((r.moic < 1).mean() * 100), "total_loss_pct": float((r.moic == 0).mean() * 100),
                         "median_irr_pct": float(r.irr.median() * 100), "p05_irr_pct": float(r.irr.quantile(.05) * 100),
                         "p95_irr_pct": float(r.irr.quantile(.95) * 100),
                         "sd_log_exit_over_entry": float(np.log(base.exit_mult / base.entry_mult).std())})
    print("PRICE MOVES done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "followup_results.csv", index=False)
    (OUT / "followup_checks.json").write_text(json.dumps(checks, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(res.round(2).to_string(index=False))
    print(coef.round(3).to_string(index=False))
    print(pred_ci.round(2).to_string())
    print(summ.round(3).to_string())
    print(json.dumps({k_: v for k_, v in checks.items()}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
