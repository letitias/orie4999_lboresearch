"""Reproduce the supplied LBO analysis and run presentation experiments.

Requirements: numpy, pandas, scipy, xlrd. Run from any directory.
Original notebooks, source files, and MonteCarloRun.py remain unchanged.
"""
from pathlib import Path
import copy
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("original_mc", ROOT / "MonteCarloRun.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)
ORIGINAL = copy.deepcopy(mc.INPUTS)
ORIGINAL["entry_mult"] = ORIGINAL["exit_mult"] = (10., 12., 15.5)
SEED, N = 42, 10_000


def prepare():
    files = sorted(ROOT.glob("financials-annual*.csv"))
    raw = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    d = raw.rename(columns=lambda c: c.lower().replace("bq_", "")).copy()
    cols = ["year", "ebitda", "operating_income", "revenue", "buildings_and_other_depreciable_assets",
            "depletable_assets", "land", "less_accumulated_depreciation", "current_assets", "cash",
            "current_liabilities", "short_term_debt"]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    duplicate_keys = int(d.duplicated(["id", "year"]).sum())
    missing_years = int(d.year.isna().sum())
    d = d.dropna(subset=["id", "year"]).sort_values(["id", "year"]).drop_duplicates(["id", "year"])
    d["gross_ppe"] = d.buildings_and_other_depreciable_assets + d.depletable_assets.fillna(0) + d.land.fillna(0)
    d["acc_dep"] = d.less_accumulated_depreciation.abs()
    d["nwc"] = (d.current_assets - d.cash.fillna(0)) - (d.current_liabilities - d.short_term_debt.fillna(0))
    prev = d.groupby("id")[["year", "gross_ppe", "acc_dep", "nwc"]].shift(1)
    ok = prev.year == d.year - 1
    for dest, source in [("capex", "gross_ppe"), ("depr", "acc_dep"), ("d_nwc", "nwc")]:
        d[dest] = np.where(ok, d[source] - prev[source], np.nan)
    pos = d[d.ebitda > 0].copy()
    pos["da_pct"] = pos.depr / pos.ebitda
    pos["capex_nwc_pct"] = (pos.capex + pos.d_nwc) / pos.ebitda
    all_co = pos.groupby("id")[["da_pct", "capex_nwc_pct"]].median()
    e = d[["id", "year", "ebitda"]].dropna()
    pairs = e.merge(e.assign(year=e.year - 5), on=["id", "year"], suffixes=("", "_fwd"))
    pairs = pairs[(pairs.ebitda > 0) & (pairs.ebitda_fwd > 0)].copy()
    pairs["g"] = (pairs.ebitda_fwd / pairs.ebitda) ** .2 - 1
    bounded_g = pairs[pairs.g.between(-.30, .50)].groupby("id").g.median().rename("ebitda_g")

    def cohort(bounds=(-.5, 1.5), da_high=.8):
        p = all_co[all_co.da_pct.between(0, da_high) & all_co.capex_nwc_pct.between(*bounds)]
        return p.rename(columns={"da_pct": "da", "capex_nwc_pct": "capex_nwc"}).join(bounded_g, how="inner").dropna()

    co = cohort()
    d["margin"] = d.ebitda / d.revenue
    screen = d.groupby("id").agg(n_years=("year", "count"), always_pos=("ebitda", lambda s: (s > 0).all()),
                                 margin=("margin", "median"))
    keep = screen[(screen.n_years >= 5) & screen.always_pos & (screen.margin >= .10)].index
    pe = co.loc[co.index.intersection(keep)]
    complete = all_co.dropna()
    filtered = all_co[all_co.da_pct.between(0, .8) & all_co.capex_nwc_pct.between(-.5, 1.5)]
    audit = {"source_files": [p.name for p in files], "source_rows": len(raw), "source_companies": int(raw.BQ_ID.nunique()),
             "year_min": int(d.year.min()), "year_max": int(d.year.max()), "duplicate_keys": duplicate_keys,
             "missing_year_rows": missing_years, "positive_ebitda_companies": len(all_co),
             "missing_operating_ratios": int(all_co.isna().any(axis=1).sum()),
             "outlier_companies_removed": len(complete)-len(filtered), "after_ratio_bounds": len(filtered),
             "final_company_sample": len(co), "pe_screened_companies": len(pe),
             "growth_positive_endpoint_windows": len(pairs), "growth_bounded_windows": int(pairs.g.between(-.3,.5).sum())}
    return co, pe, {"narrow_bounds": cohort((0, .75)), "wide_bounds": cohort((-1, 2), 1.)}, audit


def multiples():
    raw = pd.read_excel(ROOT / "vebitda.xls", sheet_name="Industry Averages", header=None)
    header = raw.index[raw[0].eq("Industry Name")][0]
    t = raw.iloc[header + 1:, [0, 1, 3]].dropna().copy()
    t.columns = ["industry", "n_firms", "ev_ebitda"]
    t["industry"] = t.industry.str.replace(r"\s+", " ", regex=True).str.strip()
    t["ev_ebitda"] = pd.to_numeric(t.ev_ebitda, errors="coerce")
    exclude = r"Bank|Brokerage|Financial Svcs|Insurance|Reinsurance|Investments|R\.E\.I\.T|REIT|Real Estate|Total Market"
    t = t[~t.industry.str.contains(exclude) & (t.ev_ebitda > 0)]
    return t, tuple(t.ev_ebitda.quantile([.1,.5,.9]).round(2))


def draw(co=None, mult=None, rho=0., seed=SEED, independent=False, only=None):
    mc.INPUTS = copy.deepcopy(ORIGINAL)
    if mult is not None:
        mc.INPUTS["entry_mult"] = mc.INPUTS["exit_mult"] = mult
    d = mc.sample_inputs(n=N, correlations={("entry_mult", "exit_mult"): rho}, seed=seed)
    if co is not None:
        cols = [only] if only else mc.EMPIRICAL_COLS
        picked = co.sample(n=N, replace=True, random_state=seed+1)[cols].reset_index(drop=True)
        if independent:
            rng = np.random.default_rng(seed+2)
            picked = pd.DataFrame({c: rng.permutation(picked[c].to_numpy()) for c in cols})
        d[cols] = picked.to_numpy()
    return d


def funding_engine(d, borrow=False):
    """Original equations, optionally financing deficits by extra interest-bearing debt.

    This is a sensitivity case with unlimited borrowing, not a default model.
    No borrowing capacity, fees, or borrowing spread is assumed.
    """
    x = {c:d[c].to_numpy() for c in d.columns}
    debt = x["leverage"]*mc.EBITDA_0
    cash = np.zeros(len(d)); exit_equity = np.zeros(len(d)); shortfall = np.zeros(len(d),dtype=bool)
    equity_in = x["entry_mult"]*mc.EBITDA_0*(1+x["fees"]) - debt
    assert np.all(equity_in > 0), "Invalid initial equity"
    for year in range(1,8):
        ebitda = mc.EBITDA_0*(1+x["ebitda_g"])**year
        interest = x["rate"]*debt
        tax = np.maximum(ebitda*(1-x["da"])-interest,0)*mc.TAX_RATE
        fcf = ebitda-interest-tax-x["capex_nwc"]*ebitda
        active = year <= x["hold"]
        if borrow:
            available = cash+fcf
            shortfall |= active & (available < 0)
            extra_debt = np.maximum(-available,0)
            debt += extra_debt
            available = np.maximum(available,0)
            paydown = np.minimum(available,debt)
            debt -= paydown
            cash = available-paydown
        else:
            paydown = np.clip(fcf,0,debt)
            debt -= paydown
            cash += fcf-paydown
            shortfall |= active & (cash < 0)
        exit_equity = np.where(x["hold"]==year,np.maximum(x["exit_mult"]*ebitda-debt+cash,0),exit_equity)
    moic = exit_equity/equity_in
    return pd.DataFrame({"irr":moic**(1/x["hold"])-1,"moic":moic}), shortfall


def metrics(r, shortfall):
    return {"mean_irr_pct":r.irr.mean()*100,"median_irr_pct":r.irr.median()*100,
            "p05_irr_pct":r.irr.quantile(.05)*100,"p95_irr_pct":r.irr.quantile(.95)*100,
            "std_irr_pp":r.irr.std()*100,"median_moic":r.moic.median(),
            "loss_pct":(r.moic<1).mean()*100,"under8_pct":(r.irr<.08).mean()*100,
            "over20_pct":(r.irr>.20).mean()*100,"cash_shortfall_pct":shortfall.mean()*100}


def main():
    OUT.mkdir(exist_ok=True)
    co, pe, bounds, audit = prepare()
    t, mult = multiples()
    print("DATA AUDIT",json.dumps(audit),flush=True)
    print("MULTIPLES",len(t),mult,flush=True)
    co.to_csv(OUT/"brightquery_company_params.csv")
    pe.to_csv(OUT/"brightquery_company_params_pe.csv")
    co.describe(percentiles=[.1,.25,.5,.75,.9]).to_csv(OUT/"company_input_summary.csv")
    co.corr(method="spearman").to_csv(OUT/"operating_correlations.csv")
    t.to_csv(OUT/"industry_multiples.csv",index=False)
    rows=[]
    scores=[]

    def experiment(label, companies=None, multiple=None, rho=0., independent=False, only=None, borrow=False, seed=SEED):
        d=draw(companies,multiple,rho,seed,independent,only)
        r,shortfall=funding_engine(d,borrow)
        if not borrow:
            np.testing.assert_allclose(r.to_numpy(),mc.run_lbo(d).to_numpy(),rtol=1e-12,atol=1e-12)
        row={"case":label,"n":N,"seed":seed,"rho":rho,"funding":"extra debt" if borrow else "original",
             "company_count":len(companies) if companies is not None else 0,**metrics(r,shortfall)}
        rows.append(row)
        rank_rho=d.rank().corrwith(r.irr.rank())
        score=(rank_rho**2/(rank_rho**2).sum()).sort_values(ascending=False)
        for name,value in score.items():scores.append({"case":label,"rho":rho,"input":name,"rank_score_pct":value*100})
        return row

    experiment("Original triangular")
    experiment("Dewey operating inputs",co)
    experiment("Dewey operating inputs shuffled",co,independent=True)
    for col in mc.EMPIRICAL_COLS:
        experiment("Only empirical "+col,co,only=col)
    for rho in [0,.3,.6,.9]:
        experiment("Damodaran only",multiple=mult,rho=rho)
        experiment("Dewey + Damodaran",co,mult,rho)
        experiment("Screened Dewey + Damodaran",pe,mult,rho)
    experiment("Screened Dewey operating inputs",pe)
    for name,c in bounds.items():
        experiment("Dewey "+name,c)
    experiment("Dewey operating inputs funded",co,borrow=True)
    experiment("Dewey + Damodaran funded",co,mult,.6,borrow=True)
    pd.DataFrame(rows).to_csv(OUT/"experiment_results.csv",index=False)
    pd.DataFrame(scores).to_csv(OUT/"rank_correlation_scores.csv",index=False)
    convergence=[]
    for seed in range(100,120):
        d=draw(co,mult,.6,seed)
        r,s=funding_engine(d)
        convergence.append({"seed":seed,**metrics(r,s)})
    pd.DataFrame(convergence).to_csv(OUT/"seed_stability.csv",index=False)
    source_q=t.ev_ebitda.quantile([.1,.5,.9]).to_numpy()
    sim_q=mc.triangular_ppf(np.array([.1,.5,.9]),*mult)
    quantiles=pd.DataFrame({"percentile":[10,50,90],"source_multiple":source_q,"triangular_multiple":sim_q})
    quantiles.to_csv(OUT/"multiple_quantile_comparison.csv",index=False)
    (OUT/"data_audit.json").write_text(json.dumps({**audit,"industry_count":len(t),"multiple_params":list(mult)},indent=2))
    frame=pd.DataFrame(rows)
    print(frame[["case","rho","company_count","median_irr_pct","p05_irr_pct","p95_irr_pct","loss_pct","cash_shortfall_pct"]].round(2).to_string(index=False),flush=True)
    print("CO SUMMARY",co.describe().round(4).to_string(),flush=True)
    print("SEED STABILITY",pd.DataFrame(convergence)[["median_irr_pct","loss_pct"]].agg(["min","max"]).round(2).to_string(),flush=True)
    print("QUANTILES",quantiles.round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
