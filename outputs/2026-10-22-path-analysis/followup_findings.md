# Follow-up: prediction, cash needs, real price moves

`followup_analysis.py` produces every number below. It reuses `path_analysis.py` and `review_analysis.py` unchanged. All cases use Damodaran entry multiples, seed 42 and the same financing draws as the earlier slides, and P(MOIC < 1) weights every company equally.

## 1. Can pre-purchase information predict losing deals?

**Setup**
- **Windows:** purchase years 2013–2018, at least 4 years observed up to the purchase year, and slide 8's ratio bounds applied to pre-purchase years.
- **Features:** nine, all built from the purchase year and the 3 years before it:
  - EBITDA margin, its 3-year change and its volatility;
  - 3-year revenue growth;
  - a prior loss-making year (yes/no);
  - size (log revenue);
  - purchase-year profit spike;
  - median reinvestment and depreciation rates.
- **Outcome:** each window's loss probability across 1,000 financing draws.
- **Model:** logistic regression, fit on purchases from 2013–2015 (3,102 windows, 1,238 companies) and tested on 2016–2018 (2,988 windows, 1,225 companies).

**Results on the test set, P(MOIC < 1)**

| Group | Loss |
|---|---:|
| All test windows | 28.6% |
| Model: lowest-risk third | 21.1% |
| Model: middle third | 29.2% |
| Model: highest-risk third | 35.3% |
| Screen: pass (pre-purchase years) | 27.4% |
| Screen: fail | 31.1% |

**Gaps, with company bootstrap 95% intervals (500 resamples)**

| Gap | Size | 95% interval |
|---|---:|---:|
| Model: highest minus lowest third | 14.2 pts | 10.1 to 18.4 |
| Screen: fail minus pass | 3.6 pts | −0.1 to 7.5 |
| Model gap minus screen gap | | +5.8 to +15.6 |

**Companies never seen in training:** the lowest-risk third loses 18.8% and the highest-risk third 39.7% (62 and 170 windows).

**What raises predicted risk** (standardized coefficients, largest first):
- higher depreciation relative to EBITDA, i.e. capital intensity (+0.20);
- smaller size (−0.14 on log revenue);
- volatile margins (+0.13);
- faster revenue growth before purchase (+0.09);
- a purchase-year profit spike (+0.07).

**Caveats**
- Window-level ranking is noisy: the Spearman correlation between predicted and actual is 0.14, so the model separates groups, not individual deals.
- Training outcome years overlap test outcome years, so both share the same economy.
- Predicted levels run about 3 points high on the test set.

## 2. Cash needs with cleaner reinvestment

**The data problem.** In other years, 4–9% of companies' gross PP&E rises by more than 50% in a year. That share jumps to 19% in 2018 and 25% in 2019, with no rise in revenue jumps. This fits the lease-accounting change that put leases on the balance sheet. Across all years, 35% of such jumps come with revenue growth above 25%, which is the signature of an acquisition.

**The rule.** Any year where gross PP&E grows by more than 50% is treated as non-operating spending. Its capex is replaced by the company's typical capex rate: the median of capex / EBITDA over its other positive-EBITDA years. This flags 8.8% of window-years and 45.6% of windows.

| Year-by-year model (1,687 companies, ρ = 0.6) | Deals with a cash-negative year | P(MOIC < 1) |
|---|---:|---:|
| Constant growth, for reference | 18.0% | 16.8% |
| Reinvestment as reported | 73.7% | 32.5% |
| Acquisition and accounting jumps removed | 66.1% | 31.0% |
| Reinvestment at each company's typical rate | 45.6% | 32.1% |

**Reading it**
- With real profit paths, roughly half to two-thirds of deals have at least one year in which cash turns negative, versus 18% under constant growth.
- Cleaning the data barely changes losses.
- The size of the cash gaps is not reported: the original model sends later positive cash to debt rather than back to the cash balance, so measured deficits never shrink.
- A cash shortfall is not a default.

## 3. Real price moves

**Data.** Damodaran's US EV/EBITDA files for January 2001 to January 2026. The 2001–2024 archive files are in `data/damodaran_archive`. The file labeled YY is dated January 20YY+1, verified on every file that carries a date; the file labeled 17 has a stale date cell but different values. The 2026 file parses to exactly the table used by `review_analysis.py`.

**Pairs.** The same industry name at entry year Y and exit year Y + h, for h = 5, 6, 7. That gives 3,807 pairs. Damodaran renamed most industries in 2014, and pairs that cross the rename are lost.

**What history shows**
- The spread of log(exit / entry) is 0.48. The copula gives 0.20 at ρ = 0.6 and 0.32 even with fully independent draws. Real industry moves were wider than any version of our assumption.
- The median ratio is 1.07: multiples drifted up on average. 25% of pairs exit more than 2 turns below entry.
- The rank correlation between entry and exit multiples is 0.46, versus 0.58 implied by ρ = 0.6.

**Model runs** (exit = entry × a historical ratio for the same holding period):

| P(MOIC < 1) | ρ = 0.9 | ρ = 0.6 | ρ = 0.3 | ρ = 0 | Historical | Historical, 2001–13 entries | Historical, 2014–21 entries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Year-by-year paths | 31.6% | 32.5% | 33.5% | 34.3% | 33.3% | 34.9% | 30.6% |
| Constant growth (main sample) | 15.1% | 16.8% | 18.3% | 19.7% | 18.6% | 20.8% | 15.7% |

**Reading it**
- History implies more price risk than ρ = 0.6. Because multiples also drifted upward, losses land near ρ = 0.3.
- Across every price assumption tested, losses range over 4.3 points with year-by-year paths (30.6–34.9%) and 5.8 points with constant growth (15.1–20.8%). Real profit paths moved them by about 16 points (path_findings.md).

**Caveats**
- Industry averages move less than single companies, which understates deal-level price risk.
- Industry multiple changes partly reflect industry profit cycles (multiples rise when profits dip), while the model draws them independently of the company's own profit path. That overstates risk.

## Sense checks
- Capex + change in NWC reproduces reinvestment exactly.
- Constant growth run through the new engine matches `review_analysis.funding_engine`.
- The 2026 Damodaran parse matches the team's table exactly.
- Every feature uses only years up to the purchase year; this is enforced in code.
- Window-level loss probabilities (28.1–28.7% across blocks of financing draws) agree with full 10,000-deal simulations (27.2–28.4% across seeds).
- The copula reproduces a rank correlation of 0.58 at ρ = 0.6.
