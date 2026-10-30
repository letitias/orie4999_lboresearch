"""Verification cross-checks for the October 15 deck update.

Reuses review_analysis.py (same seed, draws and engine); writes supplementary_checks.json.
Run after review_analysis.py. Requirements: numpy, pandas, scipy, xlrd.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("review", OUT / "review_analysis.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)


def pct(x):
    return float(np.mean(x) * 100)


def main():
    co, pe, _, _ = ra.prepare()
    t, mult = ra.multiples()
    checks = {}

    # 1. Liquidity: overlap between cash shortfall and loss, combined model at rho 0.6.
    d = ra.draw(co, mult, .6)
    r, short = ra.funding_engine(d)
    rf, short_f = ra.funding_engine(d, borrow=True)
    loss = r.moic.to_numpy() < 1
    checks["combined_rho06"] = {
        "loss_pct": pct(loss), "shortfall_pct": pct(short),
        "loss_given_shortfall_pct": pct(loss[short]), "loss_given_no_shortfall_pct": pct(loss[~short]),
        "shortfall_given_loss_pct": pct(short[loss]),
        "shortfall_sets_identical_with_funding": bool((short == short_f).all()),
        "funded_loss_pct": pct(rf.moic < 1),
        "paths_flipped_to_loss_by_funding": int(((rf.moic < 1) & ~loss).sum()),
        "shortfall_given_capex_nwc_above_50pct": pct(short[d.capex_nwc.to_numpy() > .5]),
        "shortfall_given_capex_nwc_below_50pct": pct(short[d.capex_nwc.to_numpy() <= .5]),
    }

    # 2. Damodaran-only rho 0: 95th percentile with rounded vs unrounded parameters.
    unrounded = tuple(t.ev_ebitda.quantile([.1, .5, .9]))
    out = {}
    for name, m in [("rounded", mult), ("unrounded", unrounded)]:
        r0, s0 = ra.funding_engine(ra.draw(None, m, 0.))
        out[name] = {"params": [float(v) for v in m], "p95_irr_pct": float(r0.irr.quantile(.95) * 100),
                     "p05_irr_pct": float(r0.irr.quantile(.05) * 100)}
    checks["damodaran_only_rho0"] = out

    # 3. Deterministic base cases (all inputs at mode, 6-year hold).
    base = {k: v[1] for k, v in ra.ORIGINAL.items()} | {"hold": 6}
    for name, m in [("original_multiples", None), ("damodaran_mode", mult)]:
        b = dict(base)
        if m is not None:
            b["entry_mult"] = b["exit_mult"] = m[1]
        res = ra.mc.run_lbo(pd.DataFrame([b]))
        checks[f"base_case_{name}"] = {"irr_pct": float(res.irr[0] * 100), "moic": float(res.moic[0])}

    # 4. Screen composition: how the 919 differ from the 2,061 on operating inputs.
    checks["sample_medians"] = {
        "all_2061": co.median().to_dict(), "screened_919": pe.median().to_dict(),
        "all_share_negative_growth_pct": pct(co.ebitda_g < 0), "screened_share_negative_growth_pct": pct(pe.ebitda_g < 0),
    }
    (OUT / "supplementary_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
