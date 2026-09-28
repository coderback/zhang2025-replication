> **Provenance.** This file was written and committed in the author's private research repository (commit `260681b`, 2026-09-28) before any GDELT sentiment or model result was computed. Paths such as `eod_swing/rF/` and `research/run_F1.py` refer to that repository. In this repository the code lives in `zhang_replication/` and `scripts/`. "The project's hold-out" refers to that wider research programme, which reserves data from 2024-10 onwards.

# Pre-registration F1: replicating "Interpretable ML for Macro Alpha" (FinBERT + GDELT + XGBoost)

**Written:** 2026-09-28, before any GDELT sentiment or model result was computed.

## The paper and why it's worth replicating
- **Paper:** Zhang (2025), arXiv 2505.16136v1 (`docs/2505.16136v1.pdf`).
- **Claim:** FinBERT sentiment of GDELT headlines drives an XGBoost classifier for next-day direction, with
  out-of-sample (c. 2017 → Apr 2025), cost-adjusted Sharpe ratios of **5.87 (EUR/USD), 4.65 (USD/JPY) and 4.65
  (10-year Treasury futures, ZN)**, and daily win rates of 66–73%.
- **Why this is implausible:** results like these are orders of magnitude beyond published daily-FX
  predictability, and our own GDELT tone test (L2, strict timing) had a Sharpe of −0.35.
- **Code:** the paper's repository (github.com/yukepenn/macro-news-sentiment-trading) is not public, so the
  pipeline is rebuilt from the paper's text.

**Hypothesis (stated before the data):** the reported performance comes from feature/return timing, not from
sentiment. The method as described, with correct alignment, will not reproduce it, while a one-day look-ahead
misalignment will.

## Pipeline (following the paper; our choices where it is silent are marked ◆)
**Events**
- GDELT 1.0 daily event exports, 2015-01-01 → 2024-10-02.
- Filter: three-digit EventBaseCode 100–199. ◆ The paper says "EventCode 100–199"; CAMEO 10–19 are demand,
  disapprove, reject, threaten, protest, force posture, reduce relations, coerce, assault and fight, not
  "consultations and statements" as the paper describes.
- Per day: the top 100 events by NumArticles, no de-duplication (as in the paper).

**Headlines** ◆
- Taken from each event's SOURCEURL slug: the longest path segment of at least 3 hyphen- or underscore-separated
  words, with ids and extensions removed. Events without a usable slug are dropped (the paper drops failed
  extractions).
- **Why:** the paper scraped each headline over HTTP. Scraping about 370k URLs from 2015 onward is impractical and
  suffers heavy link rot.
- **Check:** titles are scraped for a random sample of 1,000 URLs. Reported: the share of usable slugs, and the
  correlation between FinBERT scores of the slug text and the real title.

**Sentiment**
- ProsusAI/finbert on CPU, max 64 tokens, score s = P(pos) − P(neg).

**Features (as in §3.3–3.4)**
- mean S, std σ, volume N, log(1+N), article impact S·log(1+N), Goldstein mean and std
- lags 1–3 of S, σ, N, G; MA5 and MA20 of S; ΔS = MA5 − MA20
- 5- and 10-day rolling std of S; 5- and 10-day rolling sums of N
- lagged return r_t; 20-day volatility

**Prices and target**
- Yahoo Finance EURUSD=X, USDJPY=X, ZN=F daily closes (as in the paper).
- Target: y_t = 1[log(P_{t+1}/P_t) > 0].
- Sentiment is computed per UTC calendar day and joined to trading days. Lags and rolling windows are taken on
  the trading-day series. ◆

**Models**
- Logistic regression (L2; C ∈ {0.01, 0.1, 1, 10}). ◆
- XGBoost (depth ∈ {2, 3, 4}, learning rate ∈ {0.03, 0.1}, λ ∈ {1, 5}, up to 500 trees with early stopping on
  the last 20% of the training window). ◆
- Both tuned by an inner TimeSeriesSplit(3) on the training data only.

**Out-of-sample protocol**
- Outer TimeSeriesSplit(n_splits=5) over the sample after a 20-day warm-up.
- Daily position +1 if p > 0.5, else −1.
- Costs: 0.02% (FX) or 0.05% (ZN) per round trip, i.e. half of that per unit of position change.
- Metrics on the pooled out-of-sample series: Sharpe, CAGR, win %, one-sided block-bootstrap p (21-day blocks),
  and 20 random-sign twins.

**Window** ◆
- The paper runs to April 2025. Our primary window ends **2024-09-30**, because 2024-10 onward is this project's
  locked hold-out.
- Extending to April 2025 for exact comparability needs the user's go-ahead and would be reported separately.

## Variants (all pre-registered; identical except for the news window)
| Variant | News used for the prediction made at the close of day t | Purpose |
|---|---|---|
| **P** (paper, as described) | Events dated UTC day t (SQLDATE = t) | Faithful replication |
| **S** (strict) | Events added to GDELT on day t−1 or earlier (DATEADDED ≤ t−1), so all news predates the close of t | No-leak benchmark |
| **L** (leak diagnostic) | Events dated day t+1, the same day as the return being predicted | Shows what a one-day misalignment produces |

## Verdicts (XGBoost, pooled out-of-sample, primary window)
1. **Replicated:** P has Sharpe ≥ 2 on at least 2 of 3 assets. Then S decides:
   - S also ≥ 2: a genuine signal, which enters the programme as test 23 (BH) and is then costed at CFD rates.
   - S < 1: a same-day timing effect.
2. **Not replicated, leakage explains it:** P has Sharpe < 1 on all assets, and L has Sharpe ≥ 2 on at least
   2 of 3.
3. **Not replicated, unexplained:** P < 1 on all assets and L < 2. The described method yields no edge, and the
   source of the paper's numbers can't be identified without its code.

Logistic results, per-fold results, win rates and the slug-validation statistics are reported as descriptive.

## Outputs
- Code: `eod_swing/rF/` (GDELT download, slugs, FinBERT, features, models).
- Runner: `research/run_F1.py`. Results: `research/RESULTS_F1_finbert_replication.md`.
- Dependencies: optional extra `nlp` (xgboost, torch CPU, transformers).
