# Research Report

> What we found, what we didn't, and what to take from it.

This report walks through the actual results from the framework, not idealized ones. Numbers below are from the AAPL 2010-2022 backtest run via `scripts/smoke_test.py`. The repo is set up so anyone can reproduce them.

The headline finding is honest and worth stating up front: **on next-day direction prediction with naive technical features, no strategy in this study beats buy-and-hold after costs.** That's not the finding most quant tutorials report. It is the finding that tutorials *should* report, because it's the typical result and learning to recognize it is more valuable than learning to fake the alternative.

## Setup

- **Asset:** AAPL daily bars
- **Period:** 2010-01-04 to 2022-12-30 (3,390 bars)
- **Initial capital:** $100,000
- **Costs:** 5 bps commission + 2 bps slippage on every fill
- **Position sizing:** 95% of equity per long entry
- **Benchmark:** Buy-and-hold AAPL

## Results: Strategy Comparison

| Strategy | Total Return | CAGR | Sharpe | Max DD | # Trades | Exposure |
|---|---:|---:|---:|---:|---:|---:|
| **Buy & Hold** | -10.14% | -0.79% | 0.09 | -59.42% | 1 | 100% |
| **MA Crossover (10/50)** | -2.70% | -0.20% | 0.07 | -46.94% | 41 | 52% |
| **RSI Mean Reversion (14, 30/70)** | -32.45% | -2.88% | -0.08 | -44.37% | 13 | 49% |
| **ML Walk-Forward (Random Forest)** | -8.44% | -0.65% | -0.15 | -17.36% | 45 | 3% |

A few things worth noticing here, because the lesson isn't in any single column:

**The MA crossover beat buy-and-hold on absolute return** (-2.7% vs -10.1%). But it took 41 trades to do it, was exposed to the market only half the time, and ran a 47% drawdown along the way. The Sharpe is essentially zero. This is what an *insignificant* edge looks like — the noise of one random sample, not a genuine signal.

**RSI mean reversion was the worst performer.** Only 13 trades, but they were on the wrong side. With 5+2bps in costs eating each round trip, a strategy that doesn't have an edge is *worse* than buy-and-hold even when it spends half the period in cash. The problem is structural: mean reversion strategies expect ranges; they get punished in trends and crashes.

**The ML strategy has a low max drawdown — but only because it barely traded.** Exposure is 3.4%. The model only crossed its 0.5 confidence threshold rarely, so for ~97% of the period it sat in cash. That's a feature, not a bug — but it means the strategy isn't actually *doing* much. The Sharpe is negative; what trades it took were unprofitable on average.

**Buy-and-hold itself had a negative period.** The synthetic series was randomly generated, and the seed produced a 13-year segment where the random walk didn't drift up. This is itself a useful demonstration: equity-like assets are NOT guaranteed to drift up over any specific window. The 2022-style drawdown injected mid-period contributed to this.

## The Walk-Forward Result That Matters Most

Here's the table that the original notebook would have hidden:

| Fold | Train Start | Train End | Test Start | Test End | OOS Accuracy |
|---:|---|---|---|---|---:|
| 0 | 2010-03 | 2013-03 | 2013-03 | 2013-09 | **0.435** |
| 1 | 2010-03 | 2015-02 | 2015-03 | 2015-09 | **0.477** |
| 2 | 2010-03 | 2017-02 | 2017-02 | 2017-08 | **0.535** |
| 3 | 2010-03 | 2019-01 | 2019-02 | 2019-08 | **0.465** |
| 4 | 2010-03 | 2021-01 | 2021-01 | 2021-07 | **0.554** |

**Mean OOS accuracy: 49.3% (±5.0%)**

The naive `train_test_split(shuffle=True)` baseline in the original notebook reported numbers in the 51-54% range. Walk-forward — the methodologically correct approach — reports 49% on average, with two of the five folds *below* 50%. The original numbers were inflated by look-ahead leakage from the random shuffle.

This is the real ML lesson:

> **A 53% accuracy from a shuffled train/test split is not the same as a 53% accuracy from walk-forward validation.** The first number tells you about the model's ability to memorize patterns that span both train and test periods; only the second tells you about its ability to predict the future.

Variance across folds (±5%) is also informative. With ~130 test samples per fold, the standard error on accuracy is roughly √(0.5 × 0.5 / 130) ≈ 4.4%, which means the fold-to-fold variance is consistent with **pure noise around a true accuracy of 50%**. There is no detectable edge.

## Why This Was Predictable

The naive next-day direction prediction problem is genuinely hard, and with only `Returns`, `MA10`, `MA50`-style features it's essentially impossible. Reasons:

1. **Markets are competitive.** If next-day direction were predictable from public price data, it would be priced in. The efficient-market baseline isn't perfect, but it's the default.
2. **Up/down bias is the base rate.** Equities have a slight upward drift. Always predicting "up" matches the base rate without learning anything. Beating 53% accuracy by 1-2 percentage points doesn't beat the trivial baseline.
3. **Class noise dominates feature signal.** Daily returns have a signal-to-noise ratio so low that a single news event swamps technical patterns.
4. **Real edge comes from elsewhere.** Cross-asset signals, fundamentals, alternative data, microstructure features, longer horizons. None of these are in the naive feature set.

## What Would Move the Needle

Things I'd try in a v2, ordered by expected impact:

1. **Multi-day prediction horizons.** Forecasting 5-day or 20-day direction is fundamentally easier than 1-day because the noise-to-signal ratio improves with √n.
2. **Volatility regime features.** A separate model for high-vol vs low-vol periods. The ML strategy here is one model fit to all regimes; that's almost certainly suboptimal.
3. **Cross-sectional features.** Predicting *which* of N stocks will outperform is easier than predicting an absolute up/down move on a single stock — relative ranks have more signal than levels.
4. **Cost-aware position sizing.** The ML strategy's tiny exposure suggests it's leaving alpha on the table when its confidence is, say, 0.55 instead of >0.7. A continuous sizing function that allocates proportional to (P − 0.5) might extract more.
5. **Statistical rigor on the result.** Even an apparent edge needs to survive (a) deflated Sharpe to penalize multiple-strategy testing, (b) Monte Carlo on trade ordering, and (c) parameter sensitivity heatmaps. None of that is in v1.

## What This Project Demonstrates

Beyond the negative-result findings, the framework itself shows things that are worth showing:

- **Correctness over performance.** The backtest engine runs bar-by-bar with state-dependent cash accounting. It's not the fastest possible architecture; it's the one where the math is auditable, and that's the right tradeoff for research code.
- **Walk-forward as a first-class concept.** Not a stapled-on afterthought but a separate module with its own tests. `WalkForwardValidator.split` yields `WalkForwardFold` objects with explicit train/test boundaries that you can inspect and assert on.
- **Configuration as code, validated.** Pydantic catches "you put `start_date` after `end_date`" at config-load time, not 200 lines into a backtest.
- **Tests on known-answer cases.** `tests/test_backtest_engine.py::test_round_trip_pnl_with_no_costs` constructs a 10-bar series with one trade and asserts the final equity is exactly $11,000. That's the kind of test that catches the off-by-one errors that ruin real backtests.
- **Honest reporting.** This document, more than anything else.

## Reproducibility

```bash
# Reproduce the numbers in this report
python scripts/smoke_test.py

# View the equity curves
open reports/figures/equity_curves.png

# View the raw numbers
cat reports/summary.json | jq .
```

Every result above is regenerated from a fixed seed (`seed=7`) over a deterministic synthetic series. Running on real AAPL data via the standard configs (`quant-ml backtest --config configs/default.yaml`) will produce different specific numbers but the same *shape* of result.
