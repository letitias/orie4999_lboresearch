"""Year-by-year operating paths and a screen that uses only pre-purchase information.

What this script does
---------------------
1. Rebuilds the company-year panel exactly as review_analysis.py does, but drops the "2026" rows:
   they are a latest-available snapshot (PP&E, accumulated depreciation and working-capital fields are
   empty; 69% repeat the 2025 EBITDA).
2. Builds historical deal windows: a real company, a real purchase year s (2010-2018), and that
   company's actual EBITDA, reinvestment (change in gross PP&E + change in NWC) and depreciation
   (change in accumulated depreciation) for each of the next 7 years, all divided by EBITDA in year s.
3. Runs the same LBO cash-flow engine as review_analysis.funding_engine, but year by year from the
   window instead of one constant growth rate and constant ratios per deal.
4. Experiment 1: constant model vs. year-by-year paths on the same companies.
   Experiment 2: a profitability screen applied with pre-purchase data only (no hindsight) vs. the
   same screen applied with each company's full history (hindsight), on the same windows.
5. Sense checks: engine equivalence, seed stability, company-cluster bootstrap intervals,
   sensitivity to extreme values / depreciation treatment / weighting, survivorship counts.

Prices, leverage, interest rate, fees and holding period are drawn exactly as before (same seed),
so every comparison uses identical financing draws. Requirements: numpy, pandas, scipy, xlrd.
Run from any directory; results are written beside this script.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("review_analysis", ROOT / "outputs/2026-10-15-analysis/review_analysis.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

SEED, N, H = 42, 10_000, 7          # simulations per case; every window carries 7 forward years
LAST_YEAR = 2025                    # last real fiscal year (2026 rows are a snapshot)
ENTRY_ALL = range(2010, LAST_YEAR - H + 1)       # 2010-2018: purchase years with 7 complete years after
ENTRY_SCREEN = range(2014, LAST_YEAR - H + 1)    # 2014-2018: also allows 5 years of history before purchase
DA_BOUNDS, RV_BOUNDS = (0, .8), (-.5, 1.5)      # slide 8 plausible-range bounds
B = 500                                          # bootstrap replicates
SPECS = {  # name -> (multiples, entry/exit correlation)
    "combined": ("damodaran", .6),   # slides 15, 17, 18
    "original": (None, 0.),          # slides 10, 16
}


# ---------------------------------------------------------------- data
_READ_CSV = pd.read_csv  # keep the original: main_sample() temporarily replaces pandas.read_csv


def read_without_snapshot(*args, **kwargs):
    df = _READ_CSV(*args, **kwargs)
    if "BQ_YEAR" in df.columns:
        df = df[pd.to_numeric(df["BQ_YEAR"], errors="coerce") <= LAST_YEAR]
    return df


def main_sample():
    """review_analysis.prepare() unchanged, run on the files without the 2026 snapshot rows."""
    original = ra.pd.read_csv
    ra.pd.read_csv = read_without_snapshot
    try:
        co, pe, _, audit = ra.prepare()
    finally:
        ra.pd.read_csv = original
    return co, pe, audit


def snapshot_audit():
    raw = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(ROOT.glob("financials-annual*.csv"))],
                    ignore_index=True)
    y = pd.to_numeric(raw.BQ_YEAR, errors="coerce")
    e = pd.to_numeric(raw.BQ_EBITDA, errors="coerce")
    bs = pd.to_numeric(raw.BQ_LESS_ACCUMULATED_DEPRECIATION, errors="coerce")
    w = pd.DataFrame({"id": raw.BQ_ID, "y": y, "e": e}).pivot_table(index="id", columns="y", values="e", aggfunc="first")
    pair = w[[2025.0, 2026.0]].dropna()
    ctrl = w[[2024.0, 2025.0]].dropna()
    return {"rows_2026": int((y == 2026).sum()),
            "rows_2026_with_accumulated_depreciation": int(((y == 2026) & bs.notna()).sum()),
            "share_2026_ebitda_equal_2025": float((pair[2025.0] == pair[2026.0]).mean()),
            "share_2025_ebitda_equal_2024": float((ctrl[2024.0] == ctrl[2025.0]).mean())}


def panel():
    """Company-year panel with the same derived items as review_analysis.prepare()."""
    raw = pd.concat([read_without_snapshot(p, low_memory=False) for p in sorted(ROOT.glob("financials-annual*.csv"))],
                    ignore_index=True)
    d = raw.rename(columns=lambda c: c.lower().replace("bq_", "")).copy()
    cols = ["year", "ebitda", "revenue", "buildings_and_other_depreciable_assets", "depletable_assets", "land",
            "less_accumulated_depreciation", "current_assets", "cash", "current_liabilities", "short_term_debt"]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["id", "year"]).sort_values(["id", "year"]).drop_duplicates(["id", "year"])
    d["year"] = d.year.astype(int)
    d["gross_ppe"] = d.buildings_and_other_depreciable_assets + d.depletable_assets.fillna(0) + d.land.fillna(0)
    d["acc_dep"] = d.less_accumulated_depreciation.abs()
    d["nwc"] = (d.current_assets - d.cash.fillna(0)) - (d.current_liabilities - d.short_term_debt.fillna(0))
    prev = d.groupby("id")[["year", "gross_ppe", "acc_dep", "nwc"]].shift(1)
    ok = prev.year == d.year - 1
    d["reinv"] = np.where(ok, (d.gross_ppe - prev.gross_ppe) + (d.nwc - prev.nwc), np.nan)
    d["depr"] = np.where(ok, d.acc_dep - prev.acc_dep, np.nan)
    d["flow_ok"] = d.ebitda.notna() & d.reinv.notna() & d.depr.notna()
    pos = d.ebitda > 0
    d["da_ratio"] = np.where(pos, d.depr / d.ebitda, np.nan)
    d["rv_ratio"] = np.where(pos, d.reinv / d.ebitda, np.nan)
    d["margin"] = d.ebitda / d.revenue
    return d


def build_windows(d, entry_years):
    """Every (company, purchase year) with positive EBITDA at purchase and 7 complete years after it."""
    E = d.pivot(index="id", columns="year", values="ebitda")
    R = d.pivot(index="id", columns="year", values="reinv")
    D = d.pivot(index="id", columns="year", values="depr")
    F = d.pivot(index="id", columns="year", values="flow_ok").reindex(columns=range(2010, LAST_YEAR + 1))
    F = F.fillna(False).astype(bool)
    keys, ei, ri, di = [], [], [], []
    candidates = complete = dropouts = 0
    last_year = d.groupby("id").year.max()
    for s in entry_years:
        yrs = list(range(s + 1, s + H + 1))
        e0 = E[s] if s in E.columns else pd.Series(dtype=float)
        pos = e0 > 0
        candidates += int(pos.sum())
        ok = pos & F[yrs].all(axis=1)
        dropouts += int((pos & ~ok & (last_year.reindex(pos.index) < s + H)).sum())
        ids = ok[ok].index
        complete += len(ids)
        base = e0[ids].to_numpy()[:, None]
        keys += [(i, s) for i in ids]
        ei.append(E.loc[ids, yrs].to_numpy() / base)
        ri.append(R.loc[ids, yrs].to_numpy() / base)
        di.append(D.loc[ids, yrs].to_numpy() / base)
    w = pd.DataFrame(keys, columns=["id", "s"])
    arrays = (np.vstack(ei), np.vstack(ri), np.vstack(di))
    assert np.isfinite(np.hstack(arrays)).all(), "window arrays must be complete"
    return w, arrays, {"candidates": candidates, "complete": complete, "dropouts": dropouts}


def history_stats(d, w):
    """Screen inputs computed from years up to and including the purchase year only."""
    g = d.set_index(["id", "year"]).sort_index()
    rows = []
    for i, s in w.itertuples(index=False):
        h = g.loc[i].loc[:s]
        rows.append((len(h), bool((h.ebitda > 0).all()), h.margin.median(), h.da_ratio.median(), h.rv_ratio.median()))
    return pd.DataFrame(rows, columns=["n_years", "always_pos", "margin_med", "da_med", "rv_med"], index=w.index)


def full_history_screen(d):
    """The team's screen on each company's full history (2010-2025): uses years after purchase."""
    s = d.groupby("id").agg(n_years=("year", "count"), always_pos=("ebitda", lambda x: bool((x > 0).all())),
                            margin=("margin", "median"))
    return s.index[(s.n_years >= 5) & s.always_pos & (s.margin >= .10)]


# ---------------------------------------------------------------- engine
def path_engine(fin, EI, RI, DI, borrow=False, clip_depr=True, return_min_cash=False):
    """review_analysis.funding_engine with year-by-year operating values.

    EI, RI, DI: (deals x 7) EBITDA, reinvestment and depreciation, each divided by purchase-year EBITDA.
    return_min_cash: also return the lowest cumulative cash balance before the sale (original treatment only).
    """
    x = {c: fin[c].to_numpy() for c in ["entry_mult", "exit_mult", "leverage", "rate", "fees", "hold"]}
    n, e0 = len(fin), ra.mc.EBITDA_0
    debt = x["leverage"] * e0
    cash = np.zeros(n)
    exit_equity = np.zeros(n)
    shortfall = np.zeros(n, dtype=bool)
    min_cash = np.zeros(n)
    equity_in = x["entry_mult"] * e0 * (1 + x["fees"]) - debt
    assert np.all(equity_in > 0), "Invalid initial equity"
    for year in range(1, H + 1):
        ebitda = e0 * EI[:, year - 1]
        depr = e0 * DI[:, year - 1]
        if clip_depr:
            depr = np.maximum(depr, 0)   # a fall in accumulated depreciation is a disposal, not negative depreciation
        reinv = e0 * RI[:, year - 1]
        interest = x["rate"] * debt
        tax = np.maximum(ebitda - depr - interest, 0) * ra.mc.TAX_RATE
        fcf = ebitda - interest - tax - reinv
        active = year <= x["hold"]
        if borrow:
            available = cash + fcf
            shortfall |= active & (available < 0)
            debt = debt + np.maximum(-available, 0)
            available = np.maximum(available, 0)
            paydown = np.minimum(available, debt)
            debt = debt - paydown
            cash = available - paydown
        else:
            paydown = np.clip(fcf, 0, debt)
            debt = debt - paydown
            cash = cash + fcf - paydown
            shortfall |= active & (cash < 0)
            min_cash = np.where(active, np.minimum(min_cash, cash), min_cash)
        exit_equity = np.where(x["hold"] == year, np.maximum(x["exit_mult"] * ebitda - debt + cash, 0), exit_equity)
    moic = exit_equity / equity_in
    out = pd.DataFrame({"irr": moic ** (1 / x["hold"]) - 1, "moic": moic})
    return (out, shortfall, min_cash) if return_min_cash else (out, shortfall)


def pick(groups, n, rng):
    """Company-equal weighting: pick a company uniformly, then one of its windows uniformly.

    groups: list of integer arrays, one per company, holding that company's window row numbers.
    """
    c = rng.integers(len(groups), size=n)
    sizes = np.array([len(g) for g in groups])
    k = (rng.random(n) * sizes[c]).astype(int)
    return np.array([groups[ci][ki] for ci, ki in zip(c, k)])


def groups_of(w, mask):
    sub = w[mask]
    return [idx.to_numpy() for _, idx in sub.groupby("id").groups.items()]


def financing(spec, seed=SEED, mult=None):
    m, rho = SPECS[spec]
    return ra.draw(None, mult if m == "damodaran" else None, rho, seed)


def run_paths(fin, arrays, groups, seed, borrow=False, clip_depr=True, weights="company"):
    rng = np.random.default_rng(seed + 1)
    if weights == "company":
        rows = pick(groups, len(fin), rng)
    else:  # every window equally likely
        allrows = np.concatenate(groups)
        rows = allrows[rng.integers(len(allrows), size=len(fin))]
    EI, RI, DI = (a[rows] for a in arrays)
    r, s = path_engine(fin, EI, RI, DI, borrow=borrow, clip_depr=clip_depr)
    return r, s, rows


def run_constant(fin_spec, companies, seed=SEED, mult=None, borrow=False):
    m, rho = SPECS[fin_spec]
    d = ra.draw(companies, mult if m == "damodaran" else None, rho, seed)
    r, s = ra.funding_engine(d, borrow)
    r.attrs["exit_below_entry_pct"] = float((d.ebitda_g < 0).mean() * 100)  # constant growth < 0 <=> exit EBITDA below purchase
    return r, s


def exit_below_entry(EI, rows, fin):
    return float((EI[rows, fin.hold.to_numpy() - 1] < 1).mean() * 100)


def rebase(arrays, w, d, years=3):
    """Same windows, but purchase EBITDA = median of the purchase year and the 2 years before (all positive)."""
    E = d.pivot(index="id", columns="year", values="ebitda")
    hist = np.vstack([[E.at[i, s - k] if (s - k) in E.columns else np.nan for i, s in zip(w.id, w.s)]
                      for k in range(years)])
    ok = (hist > 0).all(axis=0)
    base = np.where(ok, np.median(hist, axis=0), np.nan)
    scale = (hist[0] / base)[:, None]
    return tuple(a * scale for a in arrays), ok


# ---------------------------------------------------------------- main
def main():
    co, pe, audit_main = main_sample()
    _, mult = ra.multiples()
    d = panel()
    snap = snapshot_audit()
    print("SNAPSHOT", json.dumps(snap), flush=True)
    print("MAIN SAMPLE without 2026 rows:", len(co), "companies;", len(pe), "after full-history screen", flush=True)

    w, arrays, wcount = build_windows(d, ENTRY_ALL)
    EI, RI, DI = arrays
    extreme = ((np.abs(EI) > 20) | (np.abs(RI) > 20) | (np.abs(DI) > 20)).any(axis=1)
    print("WINDOWS", json.dumps(wcount), flush=True)

    rows, checks = [], {"snapshot_rows": snap, "window_counts": wcount}

    def record(experiment, case, spec, r, s, n_companies, n_windows, **extra):
        rows.append({"experiment": experiment, "case": case, "spec": spec, "companies": n_companies,
                     "windows": n_windows, **ra.metrics(r, s),
                     "total_loss_pct": float((r.moic.to_numpy() == 0).mean() * 100), **extra})

    # ---------------- Check 1: the year-by-year engine reproduces the constant engine on constant paths
    eq = {}
    for spec in SPECS:
        m, rho = SPECS[spec]
        dd = ra.draw(co, mult if m == "damodaran" else None, rho, SEED)
        t = np.arange(1, H + 1)
        grow = (1 + dd.ebitda_g.to_numpy()[:, None]) ** t
        syn = (grow, dd.capex_nwc.to_numpy()[:, None] * grow, dd.da.to_numpy()[:, None] * grow)
        for borrow in (False, True):
            r0, s0 = ra.funding_engine(dd, borrow)
            r1, s1 = path_engine(dd, *syn, borrow=borrow)
            diff = float(np.max(np.abs(r0.to_numpy() - r1.to_numpy())))
            assert diff < 1e-9 and (s0 == s1).all(), (spec, borrow, diff)
            eq[f"{spec}_{'borrow' if borrow else 'original'}"] = diff
    checks["engine_equivalence_max_abs_diff"] = eq
    print("ENGINE EQUIVALENCE max abs diff:", eq, flush=True)

    # ---------------- Experiment 1: constant vs. year-by-year, same companies
    in_main = w.id.isin(co.index).to_numpy()
    cos_with_windows = co.loc[co.index.intersection(w.id[in_main].unique())]
    g1 = groups_of(w, in_main)
    for spec in SPECS:
        fin = financing(spec, mult=mult)
        r, s = run_constant(spec, co, mult=mult)
        record("1_paths", "Constant model, main sample", spec, r, s, len(co), 0,
               exit_below_entry_pct=r.attrs["exit_below_entry_pct"])
        r, s = run_constant(spec, cos_with_windows, mult=mult)
        record("1_paths", "Constant model, companies with a 7-year window", spec, r, s, len(cos_with_windows), 0,
               exit_below_entry_pct=r.attrs["exit_below_entry_pct"])
        r, s, picked = run_paths(fin, arrays, g1, SEED)
        record("1_paths", "Year-by-year paths, same companies", spec, r, s, len(g1), int(in_main.sum()),
               exit_below_entry_pct=exit_below_entry(EI, picked, fin),
               share_paths_with_loss_year=float((EI[picked] <= 0).any(axis=1).mean() * 100))
        # Breakdown: real profit path only; reinvestment and depreciation stay at each company's median ratio
        cid = w.id.to_numpy()[picked]
        prof = np.maximum(EI[picked], 0)
        ra_ = co.capex_nwc.reindex(cid).to_numpy()[:, None] * prof
        da_ = co.da.reindex(cid).to_numpy()[:, None] * prof
        rp, sp = path_engine(fin, EI[picked], ra_, da_)
        record("1_paths", "Year-by-year profit only (reinvestment, depreciation at company medians)", spec, rp, sp,
               len(g1), int(in_main.sum()), exit_below_entry_pct=exit_below_entry(EI, picked, fin))
        rb, sb, _ = run_paths(fin, arrays, g1, SEED, borrow=True)
        record("1_paths", "Year-by-year paths, deficits funded with debt", spec, rb, sb, len(g1), int(in_main.sum()))
        if spec == "combined":
            exit_year = w.s.to_numpy()[picked] + fin.hold.to_numpy()
            by_exit = pd.DataFrame({"sale_year": exit_year, "loss": r.moic.to_numpy() < 1})
            by_exit = by_exit.groupby("sale_year").loss.agg(deals="size", loss_pct="mean")
            by_exit["loss_pct"] *= 100
            by_exit.to_csv(OUT / "loss_by_sale_year.csv")
            checks["loss_excluding_2020_sales_pct"] = float((r.moic.to_numpy()[exit_year != 2020] < 1).mean() * 100)
    print("EXPERIMENT 1 done", flush=True)

    # ---------------- Experiment 2: screen with pre-purchase data only vs. full history (hindsight)
    w2_mask = w.s.isin(list(ENTRY_SCREEN)).to_numpy()
    hist = history_stats(d, w[w2_mask])
    base_idx = hist.index[(hist.n_years >= 5) & hist.da_med.between(*DA_BOUNDS) & hist.rv_med.between(*RV_BOUNDS)]
    base = np.zeros(len(w), dtype=bool)
    base[base_idx] = True
    pre_pass = np.zeros(len(w), dtype=bool)
    hb = hist.loc[base_idx]
    pre_pass[hb.index[hb.always_pos & (hb.margin_med >= .10)]] = True
    hind_ids = full_history_screen(d)
    hind_pass = base & w.id.isin(hind_ids).to_numpy()
    groups2 = {
        "All eligible windows": base,
        "Pass: screen on pre-purchase years only": pre_pass,
        "Fail: screen on pre-purchase years only": base & ~pre_pass,
        "Pass: screen on full history (hindsight)": hind_pass,
        "Fail: screen on full history (hindsight)": base & ~hind_pass,
    }
    checks["experiment2_classification"] = {
        "windows": int(base.sum()), "companies": int(w.id[base].nunique()),
        "pre_pass": int(pre_pass.sum()), "hindsight_pass": int(hind_pass.sum()),
        "both_pass": int((pre_pass & hind_pass).sum()), "pre_only": int((pre_pass & ~hind_pass).sum()),
        "hindsight_only": int((hind_pass & ~pre_pass).sum()),
    }
    realized = {}
    for name, mask in groups2.items():
        e = EI[mask]
        realized[name] = {"share_ebitda_year6_below_entry": float((e[:, 5] < 1).mean() * 100),
                          "share_any_loss_making_year": float((e <= 0).any(axis=1).mean() * 100),
                          "median_ebitda_year6_vs_entry": float(np.median(e[:, 5]))}
    checks["experiment2_realized_paths"] = realized
    for spec in SPECS:
        fin = financing(spec, mult=mult)
        for name, mask in groups2.items():
            g = groups_of(w, mask)
            r, s, picked = run_paths(fin, arrays, g, SEED)
            record("2_screen", name, spec, r, s, len(g), int(mask.sum()),
                   exit_below_entry_pct=exit_below_entry(EI, picked, fin))
    print("EXPERIMENT 2 done", flush=True)

    # cohort robustness: does the pre-purchase screen separate outcomes in every purchase year?
    cohort = []
    fin = financing("combined", mult=mult)
    for s_year in ENTRY_SCREEN:
        ym = (w.s == s_year).to_numpy()
        for name, mask in [("pass", pre_pass & ym), ("fail", base & ~pre_pass & ym), ("all", base & ym)]:
            g = groups_of(w, mask)
            r, s, _ = run_paths(fin, arrays, g, SEED)
            cohort.append({"entry_year": s_year, "group": name, "windows": int(mask.sum()), "companies": len(g),
                           "loss_pct": (r.moic < 1).mean() * 100, "median_irr_pct": r.irr.median() * 100})
    pd.DataFrame(cohort).to_csv(OUT / "cohort_results.csv", index=False)

    # ---------------- Check 2: company-cluster bootstrap intervals (combined spec)
    fin = financing("combined", mult=mult)
    rng = np.random.default_rng(2026)
    ids1 = cos_with_windows.index.to_numpy()
    rows_by_id = {i: idx.to_numpy() for i, idx in w.groupby("id").groups.items()}
    boot = []
    for b in range(B):
        # experiment 1: resample companies; constant uses their parameters, paths use their windows
        cb = rng.choice(ids1, size=len(ids1), replace=True)
        rc, _ = run_constant("combined", cos_with_windows.loc[cb], mult=mult, seed=SEED)
        gb = [rows_by_id[i][in_main[rows_by_id[i]]] for i in cb]
        rp, _, _ = run_paths(fin, arrays, gb, SEED + 1000 + b)
        # experiment 2: resample companies from the eligible pool, keep each one's eligible windows
        ids2 = np.unique(w.id[base])
        cb2 = rng.choice(ids2, size=len(ids2), replace=True)
        gp = [rows_by_id[i][pre_pass[rows_by_id[i]]] for i in cb2]
        gf = [rows_by_id[i][(base & ~pre_pass)[rows_by_id[i]]] for i in cb2]
        gh = [rows_by_id[i][hind_pass[rows_by_id[i]]] for i in cb2]
        ga = [rows_by_id[i][base[rows_by_id[i]]] for i in cb2]
        out = {}
        for name, g in [("pre_pass", gp), ("pre_fail", gf), ("hind_pass", gh), ("all", ga)]:
            g = [x for x in g if len(x)]
            r, _, _ = run_paths(fin, arrays, g, SEED + 5000 + b)
            out[name] = (r.moic < 1).mean() * 100
        boot.append({"const_loss": (rc.moic < 1).mean() * 100, "path_loss": (rp.moic < 1).mean() * 100,
                     "const_median": rc.irr.median() * 100, "path_median": rp.irr.median() * 100,
                     **{f"{k}_loss": v for k, v in out.items()}})
    boot = pd.DataFrame(boot)
    boot["path_minus_const_loss"] = boot.path_loss - boot.const_loss
    boot["fail_minus_pass_loss"] = boot.pre_fail_loss - boot.pre_pass_loss
    boot["all_minus_prepass_loss"] = boot.all_loss - boot.pre_pass_loss
    boot["all_minus_hindpass_loss"] = boot.all_loss - boot.hind_pass_loss
    boot.to_csv(OUT / "bootstrap_replicates.csv", index=False)
    ci = boot.quantile([.025, .5, .975]).T
    ci.columns = ["p2.5", "p50", "p97.5"]
    ci.to_csv(OUT / "bootstrap_intervals.csv")
    print("BOOTSTRAP\n", ci.round(2).to_string(), flush=True)

    # ---------------- Check 3: seed stability (combined spec, 20 other seeds: new financing and window picks)
    seeds = []
    for sd in range(100, 120):
        fin_s = financing("combined", seed=sd, mult=mult)
        r1, _, _ = run_paths(fin_s, arrays, g1, sd)
        rp_, _, _ = run_paths(fin_s, arrays, groups_of(w, pre_pass), sd)
        rf_, _, _ = run_paths(fin_s, arrays, groups_of(w, base & ~pre_pass), sd)
        seeds.append({"seed": sd, "paths_loss": (r1.moic < 1).mean() * 100, "pre_pass_loss": (rp_.moic < 1).mean() * 100,
                      "pre_fail_loss": (rf_.moic < 1).mean() * 100})
    seeds = pd.DataFrame(seeds)
    seeds.to_csv(OUT / "seed_stability_paths.csv", index=False)
    checks["seed_stability"] = seeds.drop(columns="seed").agg(["min", "max"]).round(3).to_dict()

    # ---------------- Check 4: sensitivity of headline numbers (combined spec)
    sens = []
    fin = financing("combined", mult=mult)
    rebased, has3 = rebase(arrays, w, d)
    variants = [("as specified", {}, None, arrays),
                ("exclude windows with any value > 20x purchase EBITDA", {}, ~extreme, arrays),
                ("no floor on depreciation", {"clip_depr": False}, None, arrays),
                ("every window equally likely", {"weights": "window"}, None, arrays),
                ("windows with 3 years of profit history, purchase-year base", {}, has3, arrays),
                ("windows with 3 years of profit history, 3-year median base", {}, has3, rebased)]
    for label, kw, keep, arr in variants:
        keep = np.ones(len(w), dtype=bool) if keep is None else keep
        res = {"variant": label}
        for key, mask in [("paths_same_companies", in_main), ("pre_pass", pre_pass), ("pre_fail", base & ~pre_pass),
                          ("hind_pass", hind_pass), ("all_eligible", base)]:
            r, s, _ = run_paths(fin, arr, groups_of(w, mask & keep), SEED, **kw)
            res[f"{key}_loss_pct"] = (r.moic < 1).mean() * 100
        sens.append(res)
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)
    checks["extreme_windows"] = {"share_all_windows_pct": float(extreme.mean() * 100),
                                 "share_experiment1_pct": float(extreme[in_main].mean() * 100),
                                 "share_experiment2_pct": float(extreme[base].mean() * 100)}

    # ---------------- Worked example for the speaker notes: a typical real window, base-case financing
    base_fin = pd.DataFrame([{"entry_mult": mult[1], "exit_mult": mult[1], "leverage": ra.ORIGINAL["leverage"][1],
                              "rate": ra.ORIGINAL["rate"][1], "fees": ra.ORIGINAL["fees"][1], "hold": 6}])
    cand = np.flatnonzero(pre_pass)
    r_all, _ = path_engine(pd.concat([base_fin] * len(cand), ignore_index=True), EI[cand], RI[cand], DI[cand])
    m_all = r_all.moic.to_numpy()
    ex_row = int(cand[np.argmin(np.abs(m_all - np.median(m_all)))])  # the window with the median outcome
    ex = w.iloc[ex_row]
    steps = []
    debt0 = base_fin.leverage[0] * 100
    equity_in = base_fin.entry_mult[0] * 100 * (1 + base_fin.fees[0]) - debt0
    debt, cash = debt0, 0.0
    for t in range(1, 7):
        eb, rv, dp = 100 * EI[ex_row, t - 1], 100 * RI[ex_row, t - 1], max(100 * DI[ex_row, t - 1], 0)
        interest = base_fin.rate[0] * debt
        tax = max(eb - dp - interest, 0) * ra.mc.TAX_RATE
        fcf = eb - interest - tax - rv
        pay = min(max(fcf, 0), debt)
        debt -= pay
        cash += fcf - pay
        steps.append({"year": t, "ebitda": eb, "interest": interest, "tax": tax, "reinvestment": rv,
                      "free_cash_flow": fcf, "debt_end": debt, "cash_end": cash})
    ex_df = pd.DataFrame(steps)
    ex_df.to_csv(OUT / "worked_example.csv", index=False)
    r_ex, _ = path_engine(base_fin, EI[[ex_row]], RI[[ex_row]], DI[[ex_row]])
    exit_eq = max(mult[1] * ex_df.ebitda.iloc[-1] - ex_df.debt_end.iloc[-1] + ex_df.cash_end.iloc[-1], 0)
    assert abs(exit_eq / equity_in - r_ex.moic[0]) < 1e-9, "hand calculation must match the engine"
    checks["worked_example"] = {"entry_year": int(ex.s), "hold": 6, "moic": float(r_ex.moic[0]),
                                "irr_pct": float(r_ex.irr[0] * 100), "equity_in": float(equity_in),
                                "exit_equity": float(exit_eq)}

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "path_results.csv", index=False)
    (OUT / "path_checks.json").write_text(json.dumps(checks, indent=2, default=float))
    (OUT / "data_audit.json").write_text(json.dumps({**audit_main, "multiple_params": list(mult)}, indent=2))
    show = ["experiment", "case", "spec", "companies", "windows", "median_irr_pct", "p05_irr_pct", "p95_irr_pct",
            "loss_pct", "cash_shortfall_pct"]
    print(res[show].round(2).to_string(index=False), flush=True)
    print(json.dumps({k: checks[k] for k in ["experiment2_classification", "experiment2_realized_paths",
                                             "seed_stability", "extreme_windows", "worked_example"]}, indent=1,
                     default=float), flush=True)
    print(pd.DataFrame(sens).round(2).to_string(index=False), flush=True)
    print(pd.DataFrame(cohort).round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
