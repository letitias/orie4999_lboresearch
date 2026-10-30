# Year-by-year paths and a screen without hindsight

`path_analysis.py` produces every number below; it imports `../2026-10-15-analysis/review_analysis.py` unchanged. All cases use 10,000 simulations, seed 42, and identical draws for prices, leverage, interest rate, fees and holding period. "Combined" means Damodaran multiples with an assumed entry/exit correlation of 0.6, as on slides 15, 17 and 18.

## What the code does

1. **Panel.** Reads the three BrightQuery files and drops the `2026` rows. Those rows are a latest-available snapshot, not fiscal 2026:
   - the fields the model needs (PP&E, accumulated depreciation, current assets and liabilities, cash) are empty in all but 2–3 of the 4,086 rows;
   - 69% repeat the company's 2025 EBITDA exactly, versus 0.05% for 2024 to 2025.

   Derived items are the same as in `review_analysis.prepare()`:
   - reinvestment = change in gross PP&E + change in non-cash working capital;
   - depreciation = change in accumulated depreciation;
   - both are computed only for consecutive years.
2. **Deal windows.** A window is a real company, a purchase year *s* from 2010 to 2018, and that company's actual EBITDA, reinvestment and depreciation in each of the next 7 years, all divided by EBITDA in year *s*. Requirements:
   - EBITDA in year *s* is positive;
   - all 7 following years are complete.

   This gives 12,867 windows from 26,326 candidates.
3. **Engine.** `path_engine` is `review_analysis.funding_engine` with one change: each year uses the window's actual values instead of `(1 + g)^t` and constant ratios. Negative yearly depreciation (a fall in accumulated depreciation, meaning disposals) is floored at zero. The sale value is the exit multiple × EBITDA in the sale year, and equity cannot go below zero.
4. **Sampling.** Each simulated deal picks a company at random, then one of that company's windows at random. This weights every company equally, as the constant model does.

## Experiment 1: constant growth vs. year-by-year paths

The comparison uses the same 1,687 companies: the main-sample companies (2,041 once the 2026 rows are removed) that have at least one complete window. Combined specification:

| Case | P(MOIC < 1) | Total loss | Median IRR | 5th / 95th pct. IRR | Sells below purchase EBITDA |
|---|---:|---:|---:|---:|---:|
| Constant model, all 2,041 companies | 16.8% | 1.9% | 12.6% | −15.5% / 39.2% | 18.5% |
| Constant model, same 1,687 companies | 16.0% | 1.6% | 12.2% | −13.3% / 36.8% | 18.4% |
| Year-by-year EBITDA only (reinvestment and depreciation at company medians) | 32.2% | 15.6% | 10.0% | −100% / 53.4% | 34.1% |
| Year-by-year EBITDA, reinvestment and depreciation | 32.5% | 15.9% | 9.7% | −100% / 54.2% | 34.1% |

- **Difference, year-by-year minus constant:** +16.5 points. A company-cluster bootstrap with 500 resamples gives a 95% interval of +13.4 to +17.0.
- **Why it happens:** almost all of the increase comes from the real EBITDA paths. The constant model gives each company one median growth rate, built only from 5-year windows with positive endpoints, so profit can never collapse. In real paths, 34% of deals sell at lower EBITDA than at purchase, and about half of all losses are total losses.
- **Robustness:** losses stay between 28.7% and 33.4% across all of these:
  - excluding windows with any value above 20× purchase EBITDA;
  - removing the depreciation floor;
  - a 3-year median purchase base;
  - weighting every window equally;
  - funding deficits with debt;
  - 20 other seeds.

  Excluding sales in 2020 leaves 31.6%.
- **Cash shortfalls:** they rise to 74%, but we don't report that as a finding. Yearly reinvestment is lumpy, with acquisitions and a 2019 jump in PP&E consistent with lease accounting. With reinvestment held at company medians the rate is 46%.

## Experiment 2: the profitability screen, with and without hindsight

**Windows.** Purchase years 2014–2018, with at least 5 observed years up to the purchase year. Slide 8's ratio bounds are applied to pre-purchase medians only. That leaves 5,078 windows from 1,364 companies.

**Screen rule (the team's).** At least 5 years, positive EBITDA in every year, median EBITDA margin of at least 10%. Two versions:
- **Pre-purchase:** applied to years up to the purchase year.
- **Hindsight:** applied to 2010–2025 (full history), as on slide 17.

Combined specification:

| Group | Windows | P(MOIC < 1) | Total loss | Median IRR |
|---|---:|---:|---:|---:|
| All eligible windows | 5,078 | 31.6% | 15.8% | 10.2% |
| Pass, pre-purchase years only | 3,499 | 30.5% | 13.0% | 9.6% |
| Fail, pre-purchase years only | 1,579 | 35.8% | 21.3% | 10.3% |
| Pass, full history (hindsight) | 2,705 | 17.2% | 2.8% | 13.2% |
| Fail, full history (hindsight) | 2,373 | 45.5% | 27.6% | 3.6% |

Bootstrap 95% intervals:
- all minus pre-purchase pass: −0.3 to +3.5 points;
- pre-purchase fail minus pass: +1.6 to +9.5 points;
- all minus hindsight pass: +11.9 to +16.3 points.

Interpretation:
- **Without hindsight**, the screen separates outcomes a little: deals that pass beat deals that fail in every purchase year from 2014 to 2018, and there are fewer total losses (13.0% vs 21.3%). But passing barely improves on not screening at all.
- **With hindsight**, the same rule appears to halve losses because it removes, by construction, every company that later made a loss (0% of hindsight-pass windows contain a loss-making year).
- **Slide 17:** the 17.4% → 10.4% improvement shown there is mostly this hindsight effect.

## Data issue affecting slides 14–18

Those slides use the team pipeline, which includes the 2026 snapshot rows in 5-year growth windows. Removing them changes the main results only slightly:

| Result | With 2026 rows | Without |
|---|---:|---:|
| Main-sample companies | 2,061 | 2,041 |
| Combined loss | 17.4% | 16.8% |
| Screened loss | 10.4% | 10.2% |
| Cash shortfall | 18.6% | 18.0% |

## Limitations

- **Survivorship.** Windows need 7 complete years of data. 1,134 candidate windows (4.3%) belong to companies that leave the data before then and cannot be followed. Failures among them would make real losses higher than shown.
- **Public companies only.** The data covers US public companies 2010–2025, not buyout targets; sale years include 2020.
- **Assumed inputs.** The holding period is drawn independently of the path, so no sponsor exit timing is modeled. Prices, leverage and rates are still assumptions, and ρ = 0.6 is assumed.
- **Balance-sheet proxies.** Reinvestment and depreciation are inferred from balance-sheet changes.
- **Rank scores.** Squared Spearman rank scores are not variance shares (not used here).
