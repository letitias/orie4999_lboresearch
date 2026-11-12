"""11/12 follow-up to the 11/05 ORIE 4999 slides.

Run with Python, numpy, pandas and scipy. Reads the existing BrightQuery CSVs
and the 10/15 calibration, and writes results beside this file.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "path_analysis", OUT.parent / "2026-10-22-path-analysis/path_analysis.py"
)
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)
mc = pa.ra.mc

HOLDS = tuple(int(h) for h in mc.HOLD_YEARS)
WEIGHTS = dict(zip(HOLDS, mc.HOLD_PROBS))
SIZE_CUTS = [0, 10e6, 25e6, 100e6, 500e6, np.inf]
SIZE_LABELS = ["<$10M", "$10–25M", "$25–100M", "$100–500M", "$500M+"]
FIELDS = {
    "EBITDA": "ebitda",
    "Gross PP&E input": "buildings_and_other_depreciable_assets",
    "Accumulated D&A": "acc_dep",
    "Current assets": "current_assets",
    "Current liabilities": "current_liabilities",
}


def panel():
    d = pa.panel()
    prior = d.groupby("id")[["year", "gross_ppe", "nwc"]].shift()
    consecutive = d.year.eq(prior.year + 1)
    d["capex"] = (d.gross_ppe - prior.gross_ppe).where(consecutive)
    d["delta_nwc"] = (d.nwc - prior.nwc).where(consecutive)
    jump = consecutive & prior.gross_ppe.gt(0) & d.gross_ppe.gt(1.5 * prior.gross_ppe)
    normal = d.ebitda.gt(0) & ~jump
    capex_rate = (d.capex / d.ebitda).where(normal).groupby(d.id).transform("median")
    capex_rate = capex_rate.fillna((d.capex / d.ebitda).where(normal).median())
    d["clean_capex"] = d.capex.where(~jump, capex_rate * d.ebitda.clip(lower=0))
    np.testing.assert_allclose(d.capex + d.delta_nwc, d.reinv, equal_nan=True)
    return d


def build_candidates(d):
    """Same positive-entry universe and vintages as the 11/05 audit."""
    year_max = d.groupby("id").year.max()
    w = d.loc[d.year.isin(pa.ENTRY_ALL) & d.ebitda.gt(0), ["id", "year", "ebitda"]].copy()
    w.columns = ["id", "entry", "ebitda"]
    w = w.reset_index(drop=True)
    w["size"] = pd.cut(w.ebitda, SIZE_CUTS, labels=SIZE_LABELS, right=False)
    w["last_year"] = w.id.map(year_max)
    p = d.set_index(["id", "year"])
    cols = ["ebitda", "capex", "clean_capex", "delta_nwc", "depr"]
    mat = []
    raw_present = []
    raw_null = []
    for age in range(8):
        index = pd.MultiIndex.from_arrays([w.id, w.entry + age])
        values = p.reindex(index)
        present = index.isin(p.index)
        raw_present.append(present)
        raw_null.append(values[list(FIELDS.values())].isna().to_numpy() & present[:, None])
        if age:
            mat.append(values[cols].to_numpy() / w.ebitda.to_numpy()[:, None])
    paths = np.stack(mat, axis=1)
    required = np.isfinite(paths[:, :, [0, 1, 3, 4]]).all(axis=2)
    for h in HOLDS:
        w[f"complete{h}"] = required[:, :h].all(axis=1)
        w[f"dropout{h}"] = w.last_year.lt(w.entry + h)
        assert not (w[f"complete{h}"] & w[f"dropout{h}"]).any()
    absent_years = (~np.stack(raw_present, axis=1)).sum(axis=1)
    raw_null[0][:, 0] = False  # Entry EBITDA is required by construction.
    fields = np.stack(raw_null, axis=1).sum(axis=1)
    w["field_types_missing"] = (fields > 0).sum(axis=1)
    w["field_cells_missing"] = fields.sum(axis=1)
    w["absent_years"] = absent_years
    return w, paths, fields


def screens(d):
    """Reproduce 10/15 company eligibility and vary one filter at a time."""
    pos = d.loc[d.ebitda.gt(0)]
    ratios = pos.groupby("id")[["da_ratio", "rv_ratio"]].median()
    ratio_ids = ratios.index[
        ratios.da_ratio.between(*pa.DA_BOUNDS) & ratios.rv_ratio.between(*pa.RV_BOUNDS)
    ]
    e = d[["id", "year", "ebitda"]].dropna()
    ends = e.merge(e.assign(year=e.year - 5), on=["id", "year"], suffixes=("", "_exit"))
    positive = ends.loc[ends.ebitda.gt(0) & ends.ebitda_exit.gt(0)].copy()
    positive["cagr"] = (positive.ebitda_exit / positive.ebitda) ** 0.2 - 1
    cagr = positive.loc[positive.cagr.between(-0.30, 0.50)].groupby("id").cagr.median()
    base_ids = ratio_ids.intersection(cagr.index)
    assert len(base_ids) == 2041
    return base_ids, ratio_ids.intersection(positive.id.unique()), ratio_ids, ratios.join(cagr)


def audit(w, fields, base_ids, paths):
    complete = w.complete7
    drop = w.dropout7
    missing = ~complete & ~drop
    assert (len(w), int(complete.sum()), int(drop.sum()), int(missing.sum())) == (26326, 12867, 1134, 12325)
    assert (w.loc[missing, "absent_years"] + w.loc[missing, "field_cells_missing"]).gt(0).all()
    s = w.assign(incomplete=missing).groupby("size", observed=True).agg(
        candidates=("id", "size"), complete=("complete7", "sum"),
        dropouts=("dropout7", "sum"), missing=("incomplete", "sum")
    )
    for col in ["complete", "missing", "dropouts"]:
        s[f"{col}_pct"] = 100 * s[col] / s.candidates
    observed5 = np.isfinite(paths[:, 4, 0])
    s["ebitda_y5_observed"] = w.loc[observed5].groupby("size", observed=True).size()
    s["ebitda_y5_nonpositive"] = w.loc[observed5 & (paths[:, 4, 0] <= 0)].groupby("size", observed=True).size()
    s["ebitda_y5_nonpositive_pct"] = 100 * s.ebitda_y5_nonpositive / s.ebitda_y5_observed
    s.to_csv(OUT / "coverage_by_size.csv")

    naics = pd.read_csv(ROOT / "benefit-plan-data-sample.csv")
    assert naics.bq_id.is_unique
    mapped = w.assign(sector=w.id.map(naics.set_index("bq_id").bq_naics_sector_name))
    sectors = mapped.assign(incomplete=missing).groupby("sector").agg(
        companies=("id", "nunique"), candidates=("id", "size"),
        complete=("complete7", "sum"), missing=("incomplete", "sum"), dropouts=("dropout7", "sum")
    )
    sectors["complete_pct"] = 100 * sectors.complete / sectors.candidates
    sectors.to_csv(OUT / "coverage_by_sector_sample.csv")

    hold = []
    for label, mask in [("All candidates", w.id.notna()), ("Legacy cohort", w.id.isin(base_ids))]:
        for h in HOLDS:
            n = int(mask.sum())
            c = int(w.loc[mask, f"complete{h}"].sum())
            k = int(w.loc[mask, f"dropout{h}"].sum())
            hold.append(dict(scope=label, hold=h, candidates=n, complete=c,
                             dropouts=k, missing=n-c-k, complete_pct=100*c/n))
    pd.DataFrame(hold).to_csv(OUT / "coverage_by_hold.csv", index=False)
    result = {
        "missing_windows": int(missing.sum()),
        "missing_with_row_gaps": int((missing & w.absent_years.gt(0)).sum()),
        "missing_with_cell_gaps": int((missing & w.field_cells_missing.gt(0)).sum()),
        "missing_with_both": int((missing & w.absent_years.gt(0) & w.field_cells_missing.gt(0)).sum()),
        "field_types_missing": {str(k): int(v) for k, v in w.loc[missing, "field_types_missing"].value_counts().sort_index().items()},
        "missing_by_field": [
            dict(field=field, windows=int((fields[missing, i] > 0).sum()),
                 cells=int(fields[missing, i].sum())) for i, field in enumerate(FIELDS)
        ],
        "sector_windows": int(mapped.sector.notna().sum()),
        "sector_companies": int(mapped.loc[mapped.sector.notna(), "id"].nunique()),
        "dropout_companies_7y": int(w.loc[drop, "id"].nunique()),
    }
    return result


def financing(seed, n=10_000):
    """Use prior Damodaran quantiles rounded as in review_analysis.multiples."""
    source = pd.read_csv(OUT.parent / "2026-10-15-analysis/multiple_quantile_comparison.csv")
    quantiles = tuple(source.source_multiple.round(2))
    assert quantiles == (8.14, 12.47, 24.11)
    parameters = {**pa.ra.ORIGINAL, "entry_mult": quantiles, "exit_mult": quantiles}
    return draw_financing(parameters, seed, n)


def draw_financing(parameters, seed, n):
    old = mc.INPUTS
    try:
        mc.INPUTS = parameters
        return mc.sample_inputs(n=n, correlations={("entry_mult", "exit_mult"): .6}, seed=seed)
    finally:
        mc.INPUTS = old


def operating(paths, capex_col=1):
    return paths[:, :, 0], paths[:, :, capex_col] + paths[:, :, 3], paths[:, :, 4]


def metrics(irr, moic):
    return dict(loss_pct=100 * np.mean(moic < 1), total_loss_pct=100 * np.mean(moic == 0),
                median_irr_pct=100 * np.median(irr), median_moic=np.median(moic))


def path_result(fin, paths):
    r, _ = pa.path_engine(fin, *operating(paths))
    return metrics(r.irr, r.moic)


def pick_eligible(fin, w, seed, mask, by_hold, company_equal=True):
    rng = np.random.default_rng(seed + 1)
    rows = np.empty(len(fin), dtype=int)
    for h in HOLDS:
        target = fin.hold.to_numpy() == h
        eligible = mask & w[f"complete{h if by_hold else 7}"]
        if company_equal:
            groups = pa.groups_of(w, eligible)
            rows[target] = pa.pick(groups, int(target.sum()), rng)
        else:
            rows[target] = rng.choice(w.index[eligible], size=int(target.sum()))
    return rows


def selection(w, paths, ids):
    rows = []
    for seed in [42, 51, 64, 83, 96]:
        fin = financing(seed)
        for name, companies in ids.items():
            mask = w.id.notna() if companies is None else w.id.isin(companies)
            for by_hold in [False, True]:
                selected = pick_eligible(fin, w, seed, mask, by_hold)
                available = mask & w.complete5 if by_hold else mask & w.complete7
                rows.append(dict(case=name, seed=seed, eligibility="By hold" if by_hold else "7 years",
                                 companies=w.loc[available, "id"].nunique(), windows=int(available.sum()),
                                 **path_result(fin, paths[selected])))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "selection_results.csv", index=False)
    return result


def matched_bounds(w, paths, base_ids):
    """Exact eligible-window denominators. Integrate each observed window
    over the same 256 financing draws per hold, then apply missing outcomes.
    """
    k = 256
    fin = financing(42, 8 * k)
    matched_ids = w.loc[w.id.isin(base_ids) & w.complete7, "id"].unique()
    assert len(matched_ids) == 1687
    scopes = [
        ("11/05 matched companies", w.id.isin(matched_ids), True),
        ("All candidates", w.id.notna(), False),
    ]
    rows = []
    for h in HOLDS:
        f = fin.loc[fin.hold.eq(h)].head(k).reset_index(drop=True)
        assert len(f) == k
        f = f.iloc[np.tile(np.arange(k), 250)].reset_index(drop=True)
        obs_ids = w.index[w[f"complete{h}"]].to_numpy()
        p = np.full(len(w), np.nan)
        for batch in np.array_split(obs_ids, int(np.ceil(len(obs_ids) / 250))):
            t = paths[np.repeat(batch, k)]
            r, _ = pa.path_engine(f.iloc[:len(t)], *operating(t))
            p[batch] = (r.moic.to_numpy() < 1).reshape(-1, k).mean(axis=1)
        for scope, mask, company_equal in scopes:
            observed = mask & w[f"complete{h}"]
            drop = mask & w[f"dropout{h}"]
            unknown = mask & ~observed
            values = np.nan_to_num(p)

            def rate(x, in_scope):
                if company_equal:
                    return 100 * pd.Series(x[in_scope]).groupby(w.id[in_scope].to_numpy()).mean().mean()
                return 100 * x[in_scope].mean()

            rows.append(dict(scope=scope, hold=h, observed=int(observed.sum()),
                             dropouts=int(drop.sum()), missing=int((unknown & ~drop).sum()),
                             observed_pct=rate(values, observed),
                             visible_low_pct=rate(values, observed | drop),
                             visible_high_pct=rate(values + drop.to_numpy(), observed | drop),
                             visible_half_pct=rate(values + .5 * drop.to_numpy(), observed | drop),
                             all_low_pct=rate(values, mask),
                             all_high_pct=rate(values + unknown.to_numpy(), mask)))
    out = pd.DataFrame(rows)
    for scope, group in out.groupby("scope"):
        pct = group.filter(regex="_pct$")
        weights = group.hold.map(WEIGHTS).to_numpy()
        total = (pct.to_numpy() * weights[:, None]).sum(axis=0)
        out.loc[len(out), ["scope", "hold", *pct.columns]] = [scope, 0, *total]
    out.to_csv(OUT / "matched_bounds.csv", index=False)
    return out


def verify_legacy_baseline(d, base_ids):
    fin = financing(42)
    w, arrays, _ = pa.build_windows(d, pa.ENTRY_ALL)
    r, _, _ = pa.run_paths(fin, arrays, pa.groups_of(w, w.id.isin(base_ids)), 42)
    baseline = metrics(r.irr, r.moic)
    assert round(baseline["loss_pct"], 1) == 32.5, baseline
    return baseline


def ebitda_proxy_test(fin, w, paths):
    """Reinstate cases with nonpositive EBITDA at year five as zero equity.
    This is a scenario on observed year-five sign, not a measured default.
    """
    rng = np.random.default_rng(43)
    observed = w.complete7.to_numpy()
    distressed = paths[:, 4, 0] <= 0
    rows = []
    for size in ["All", "$10–25M", "$25–100M"]:
        mask = observed & ((w["size"] == size).to_numpy() if size != "All" else observed)
        ok = np.flatnonzero(mask & ~distressed)
        draw = rng.choice(ok, size=len(fin))
        r, _ = pa.path_engine(fin, *operating(paths[draw]))
        proxy_rate = np.mean(distressed[mask])
        drop_loss = np.mean(r.moic < 1)
        rows.append(dict(size=size, complete_windows=int(mask.sum()), proxy_pct=100*proxy_rate,
                         dropped_loss_pct=100*drop_loss,
                         zero_equity_loss_pct=100*(proxy_rate + (1-proxy_rate)*drop_loss)))
    pd.DataFrame(rows).to_csv(OUT / "ebitda_proxy_sensitivity.csv", index=False)


def debt_model(fin, paths, limit=1.0, reserve=.5, amort=.05, capex_col=2, recovery=0.0,
               detail=False):
    """Annual waterfall. All balances and limits use entry EBITDA units."""
    ebitda, reinvest, dep = operating(paths, capex_col)
    hold = fin.hold.to_numpy()
    original_term = fin.leverage.to_numpy()
    term = original_term.copy()
    rate = fin.rate.to_numpy()
    cash = np.full(len(fin), reserve)
    line = np.zeros(len(fin))
    contributed = fin.entry_mult.to_numpy() * (1 + fin.fees.to_numpy()) - term + reserve
    assert np.all(contributed > 0)
    default = np.zeros(len(fin), dtype=bool)
    cash_gap = default.copy()
    interest_gap = default.copy()
    used_line = default.copy()
    equity = np.zeros(len(fin))
    for i in range(7):
        active = (i < hold) & ~default
        interest = term * rate + line * (rate + .02)
        tax = np.maximum(ebitda[:, i] - np.maximum(dep[:, i], 0) - interest, 0) * mc.TAX_RATE
        principal = np.minimum(term, amort * original_term)
        before_debt_service = ebitda[:, i] - reinvest[:, i] - tax
        available = cash + before_debt_service - interest - principal
        interest_gap |= active & (cash + before_debt_service < interest)
        cash_gap |= active & (available < 0)
        draw = np.minimum(np.maximum(-available, 0), np.maximum(limit - line, 0))
        used_line |= active & (draw > 0)
        available += draw
        line += np.where(active, draw, 0)
        fail = active & (available < -1e-10)
        default |= fail
        equity[fail] = recovery * contributed[fail]
        paying = active & ~fail
        term -= np.where(paying, principal, 0)
        surplus = np.maximum(available - reserve, 0)
        repay_line = np.minimum(surplus, line)
        line -= np.where(paying, repay_line, 0)
        repay_term = np.minimum(surplus - repay_line, term)
        term -= np.where(paying, repay_term, 0)
        cash = np.where(paying, available - repay_line - repay_term, cash)
        exit_now = paying & (hold == i + 1)
        proceeds = np.maximum(fin.exit_mult.to_numpy() * ebitda[:, i] - term - line + cash, 0)
        equity[exit_now] = proceeds[exit_now]
        assert np.all(line <= limit + 1e-10) and np.all(term >= -1e-10)
    moic = equity / contributed
    irr = moic ** (1 / hold) - 1
    summary = {**metrics(irr, moic), "default_pct": 100*default.mean(),
               "cash_gap_pct": 100*cash_gap.mean(), "interest_gap_pct": 100*interest_gap.mean(),
               "draw_pct": 100*used_line.mean()}
    return (summary, default, term + line) if detail else summary


def gaussian_paths(history, n, seed):
    """Empirical marginal distributions by deal year and rank dependence
    within and across years. Work in levels to keep negative EBITDA paths.
    """
    keys = [0, 2, 3, 4]
    sample = history[:, :, keys].reshape(-1, 28)
    scores = norm.ppf(rankdata(sample, axis=0) / (len(sample) + 1))
    corr = np.corrcoef(scores, rowvar=False)
    same_year = np.repeat(np.arange(7), 4)
    block = corr * (same_year[:, None] == same_year)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 28))
    result = []
    for cov in [block, corr]:
        u = norm.cdf(z @ np.linalg.cholesky(cov).T)
        values = np.column_stack([np.quantile(sample[:, j], u[:, j]) for j in range(28)])
        arr = np.zeros((n, 7, 5))
        arr[:, :, keys] = values.reshape(-1, 7, 4)
        arr[:, :, 1] = arr[:, :, 2]
        result.append(arr)
    return (*result, corr)


def path_diagnostics(arr):
    e = arr[:, :, 0]
    delta = np.diff(e, axis=1, prepend=np.ones((len(e), 1)))
    return dict(change_p10=np.quantile(delta, .10), change_p50=np.quantile(delta, .50),
                change_p90=np.quantile(delta, .90),
                two_declines_pct=100*np.mean(((delta[:, 1:] < 0) & (delta[:, :-1] < 0)).any(axis=1)),
                down_y5_pct=100*np.mean(e[:, 4] < 1),
                negative_ebitda_pct=100*np.mean((e <= 0).any(axis=1)))


def waterfall_and_copula(w, paths, ratios):
    fin = financing(42, 50_000)
    history_mask = w.complete7.to_numpy()
    history = paths[history_mask]
    sample = np.random.default_rng(43).choice(w.index[history_mask], size=len(fin))
    historical = paths[sample]
    independent, dependent, corr = gaussian_paths(history, len(fin), 44)
    result = []
    for label, arr in [("Historical paths", historical), ("Independent years", independent),
                       ("Serial Gaussian copula", dependent)]:
        result.append(dict(case=label, **path_diagnostics(arr), **debt_model(fin, arr)))
    # Constant and annual paired on precisely the legacy cohort and draw indices.
    legacy_mask = w.id.isin(ratios.cagr.dropna().index) & w.id.isin(
        ratios.index[ratios.da_ratio.between(*pa.DA_BOUNDS) & ratios.rv_ratio.between(*pa.RV_BOUNDS)]
    )
    rows = pick_eligible(fin, w, 42, legacy_mask, by_hold=False)
    param = ratios.loc[w.loc[rows, "id"]]
    constant = np.zeros_like(paths[rows])
    constant[:, :, 0] = (1 + param.cagr.to_numpy()[:, None]) ** np.arange(1, 8)
    constant[:, :, 2] = constant[:, :, 0] * param.rv_ratio.to_numpy()[:, None]
    constant[:, :, 4] = constant[:, :, 0] * param.da_ratio.to_numpy()[:, None]
    result += [
        dict(case="Legacy constant paths", **path_diagnostics(constant), **debt_model(fin, constant)),
        dict(case="Legacy historical paths", **path_diagnostics(paths[rows]), **debt_model(fin, paths[rows])),
    ]
    pd.DataFrame(result).to_csv(OUT / "copula_and_paths.csv", index=False)
    tests = []
    for name, args in [
        ("No revolver", dict(limit=0)), ("0.5x revolver", dict(limit=.5)),
        ("1.0x revolver", {}), ("2.0x revolver", dict(limit=2)),
        ("No starting reserve", dict(reserve=0)), ("No scheduled amortization", dict(amort=0)),
        ("Raw PP&E changes", dict(capex_col=1)), ("25% equity recovery", dict(recovery=.25)),
    ]:
        tests.append(dict(case=name, **debt_model(fin, historical, **args)))
    pd.DataFrame(tests).to_csv(OUT / "distress_sensitivity.csv", index=False)
    base, base_default, base_debt = debt_model(fin, historical, detail=True)
    shock_paths = historical.copy()
    shock_paths[:, 1:3, 0] -= .2 * np.abs(shock_paths[:, 1:3, 0])
    shock, shock_default, shock_debt = debt_model(fin, shock_paths, detail=True)
    both_survive = ~base_default & ~shock_default
    shock_rows = []
    for name, result, debt in [("Observed paths", base, base_debt), ("EBITDA shock", shock, shock_debt)]:
        shock_rows.append(dict(case=name, **result, matched_survivors=int(both_survive.sum()),
                               median_exit_debt_x=np.median(debt[both_survive])))
    pd.DataFrame(shock_rows).to_csv(OUT / "ebitda_shock.csv", index=False)
    value = []
    for label, changes in [
        ("Base", {}), ("Entry −2.0x", {"entry_mult": fin.entry_mult - 2}),
        ("Entry +2.0x", {"entry_mult": fin.entry_mult + 2}),
        ("Leverage −1.0x", {"leverage": fin.leverage - 1}),
        ("Leverage +1.0x", {"leverage": fin.leverage + 1}),
        ("Exit multiple = entry", {"exit_mult": fin.entry_mult}),
    ]:
        value.append(dict(case=label, **debt_model(fin.assign(**changes), historical)))
    fixed_ebitda = historical.copy()
    fixed_ebitda[:, :, 0] = 1
    value.append(dict(case="Flat EBITDA only", **debt_model(fin, fixed_ebitda)))
    pd.DataFrame(value).to_csv(OUT / "value_sensitivity.csv", index=False)
    return {"paths": len(history), "simulations": len(fin),
            "lag1_ebitda_rank_normal_corr": np.diag(corr, k=4)[::4].tolist()}


def main():
    d = panel()
    w, paths, fields = build_candidates(d)
    legacy, positive, ratio_ids, ratios = screens(d)
    summary = audit(w, fields, legacy, paths)
    summary["candidate_companies"] = int(w.id.nunique())
    summary["legacy_baseline"] = verify_legacy_baseline(d, legacy)
    selection(w, paths, {"Legacy filters": legacy, "No CAGR limits": positive,
                         "No outcome filters": ratio_ids, "No company filters": None})
    matched_bounds(w, paths, legacy)
    ebitda_proxy_test(financing(42), w, paths)
    summary["copula"] = waterfall_and_copula(w, paths, ratios)
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
