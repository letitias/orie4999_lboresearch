# Findings for the October 15 presentation

These results use the three annual-financial CSV partitions supplied in this project and the supplied Damodaran workbook. The original source files, notebooks, simulation script, and presentation remain unchanged. `review_analysis.py` reproduces the preparation logic and runs the experiments below.

## 1. Refresh the empirical calibration slides

The current export contains 68,003 observations from 5,080 companies, covering 2010–2026. One observation has no year and is excluded. There are no duplicate company/year keys.

| Preparation step | Companies |
|---|---:|
| At least one positive-EBITDA observation | 4,644 |
| Both company-median operating ratios available | 3,168 |
| After depreciation and reinvestment bounds | 2,247 |
| After joining eligible five-year growth observations | 2,061 |
| After the notebook's additional PE-style screen | 919 |

The ratio bounds remove 921 of 3,168 complete companies, or 29.1%. This replaces the old approximately 37% figure. The current data do not reproduce the deck's sample of 2,470 companies. Without the earlier export, the exact reason for the difference cannot be isolated.

Company medians in the final sample: annual EBITDA growth 6.45%, depreciation/EBITDA 10.06%, and reinvestment/EBITDA 14.40%. Spearman correlations are approximately 0.23 for depreciation versus reinvestment, -0.06 for depreciation versus growth, and 0.15 for reinvestment versus growth. Update the input table and correlation heatmap as well as the sample count.

Five-year growth is annualized CAGR between positive endpoints five years apart. The code does not require all intervening years. Depreciation and reinvestment remain balance-sheet-change proxies. The original treatment fills missing cash, short-term debt, land, and depletable assets with zero. Company-level medians may combine different periods.

## 2. Add a combined operating and valuation experiment

All headline cases use 10,000 simulations and seed 42. Damodaran low/mode/high parameters are 8.14x, 12.47x, and 24.11x across 80 industries. The correlation parameter is the Gaussian copula's latent-normal correlation, not an estimated historical correlation.

| Entry/exit correlation parameter | Loss probability with triangular operating inputs | Loss probability with Dewey operating inputs |
|---:|---:|---:|
| 0.0 | 9.26% | 20.25% |
| 0.3 | 5.87% | 19.09% |
| 0.6 | 2.18% | 17.38% |
| 0.9 | 0.09% | 15.77% |

Suggested title: **Correlation reduces valuation risk while operating downside remains**.

At correlation 0.6, the combined model has median IRR 12.06%, mean IRR 10.60%, and 5th–95th percentile IRR of -16.24% to 37.81%. The strong correlation case does not make the combined simulation nearly riskless. These are modeled outcomes conditional on the chosen input distributions.

## 3. Add experiments identifying the source of operating downside

Hold the old multiple distributions and all financing draws fixed. Replace one operating assumption at a time with draws from the final company sample.

| Operating assumptions | Median IRR | 5th–95th percentile IRR | Loss probability |
|---|---:|---:|---:|
| Original triangular assumptions | 12.17% | 5.10% to 19.51% | 0.09% |
| Empirical growth only | 12.86% | -16.06% to 39.09% | 16.06% |
| Empirical reinvestment only | 12.69% | 2.38% to 20.92% | 2.98% |
| Empirical depreciation only | 11.96% | 4.86% to 19.36% | 0.18% |
| All empirical operating inputs, sampled jointly | 13.22% | -17.72% to 38.31% | 15.64% |

Suggested title: **The empirical growth distribution drives most of the increase in modeled downside**.

The experiments are controlled replacements, not additive contributions. Do not sum the loss probabilities. Holding each sampled growth rate constant for the full investment makes persistent growth and decline influential. A future extension can simulate matched annual company trajectories to distinguish persistent business differences from annual shocks.

Permuting the three empirical columns independently while preserving their exact sampled marginal distributions increases loss probability from 15.64% to 17.05% in the controlled seed-42 run. This is evidence that the retained company-level relationships moderately reduce downside in this experiment. It does not estimate all economic dependencies.

The existing normalized squared Spearman score for growth is 92.34% in the Dewey-only case and 69.22% in the combined independent-multiples case. Label these rank-correlation scores, not formal shares of output variance.

## 4. Add the completed sample-screening comparison

The corrected execution order constructs `co` before applying the screen. The screening rules are at least five observed years, positive EBITDA in every observed year, and median EBITDA/revenue of at least 10%.

| Combined model at correlation 0.6 | Companies | Median IRR | 5th–95th percentile IRR | Loss probability |
|---|---:|---:|---:|---:|
| Unscreened eligible sample | 2,061 | 12.06% | -16.24% to 37.81% | 17.38% |
| Additional PE-style screen | 919 | 13.35% | -5.45% to 35.63% | 10.44% |

Suggested title: **Company selection changes the modeled downside materially**.

The screen selects consistently profitable public companies. It does not identify actual buyout targets or validate private-equity returns. Screening on the full observed history also uses future profitability relative to a historical entry date, so this is a sample sensitivity experiment, not an out-of-sample backtest.

## 5. Add liquidity outcomes and a funding sensitivity

The original engine allows cash to fall below zero and subtracts the accumulated deficit from exit equity. It does not finance that deficit or charge interest on it. The new sensitivity uses available cash first, then funds deficits with additional debt at the scenario's interest rate. It assumes unlimited borrowing and does not represent a realistic revolver limit or a default model.

| Combined model at correlation 0.6 | Loss probability | 5th percentile IRR | Cash-shortfall probability |
|---|---:|---:|---:|
| Original cash-deficit treatment | 17.38% | -16.24% | 18.58% |
| Extra debt funds deficits | 17.57% | -16.58% | 18.58% |

Suggested title: **Terminal returns do not capture interim funding needs**.

In this specification, financing deficits changes aggregate loss probability by only 0.19 percentage points. The material new metric is the 18.58% probability of needing funding during the holding period. This is not a default probability. Borrowing capacity and lender terms are a future extension.

## 6. Show the distribution-calibration choice explicitly

| Multiple statistic | Damodaran data | Implied triangular distribution |
|---|---:|---:|
| 10th percentile | 8.142x | 10.770x |
| Median | 12.466x | 14.469x |
| 90th percentile | 24.115x | 19.798x |

The code treats the source p10/p50/p90 as minimum/mode/maximum. It therefore truncates source tails and does not preserve the source quantiles. This can be an intentional approximation, but the slides should label it accurately. Comparing empirical multiple resampling with the current triangular approximation is additional work; it has not been performed in these results. Historical changes within matched industries are still needed to calibrate entry/exit dynamics.

## Verification and reproducibility

- Reexecuting the original data-preparation notebook cells, with the screen moved after company construction, reproduces the analysis runner's 2,061 company parameter rows to numerical tolerance.
- Every original-funding scenario matches the supplied `run_lbo` engine to a numerical tolerance of 1e-12.
- The original base case is 12.1818% IRR and 1.9931x MOIC.
- Across 20 additional random seeds, the combined correlation-0.6 model's loss probability ranges from 16.79% to 18.13%, and its median IRR from 11.70% to 12.28%. This is simulation variability conditional on the input sample, not a confidence interval for real-world risk or historical-sample uncertainty.
- The notebook's loader currently looks for compressed CSVs under `data/`, while the supplied files are uncompressed CSVs in the project root. The separate runner reads the actual files without moving them.

Install numpy, pandas, scipy, and xlrd in a Python environment, then run `review_analysis.py`. It finds the source files relative to its own location and exports all comparison results beside this report. `experiment_results.csv` contains the main outputs, `seed_stability.csv` contains the seed experiment, and `data_audit.json` records sample construction.

Prioritize the combined experiment, the controlled input replacements, and the corrected sample-screening comparison for the next slides. Update existing empirical tables to the current export before adding new claims.
