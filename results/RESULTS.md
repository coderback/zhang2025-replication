> **Provenance.** This file was written and committed in the author's private research repository (commit `260681b`, 2026-09-28) before any GDELT sentiment or model result was computed. Paths such as `eod_swing/rF/` and `research/run_F1.py` refer to that repository. In this repository the code lives in `zhang_replication/` and `scripts/`. "The project's hold-out" refers to that wider research programme, which reserves data from 2024-10 onwards.

# F1: replication of Zhang (2025), "Interpretable Machine Learning for Macro Alpha": results (2026-09-28)

- **Pre-registration:** 260681b.
- **Code:** `eod_swing/rF/` and `research/run_F1.py` (9ef94de; memory fix f21e5a1). Tests in `tests/test_rF.py`.
- **Logs:** `data/logs/run_F1.log`, `data/logs/run_F1_slugs.log`. Table in `data/checks/F1_results.csv`.

## Verdict: NOT REPLICATED (pre-registered category 3)
The method described in the paper has no edge on any asset, under any timing, with either model.

| | Paper (Table 1, XGBoost) | This replication, P (as described), XGBoost |
|---|---|---|
| EUR/USD Sharpe | **5.87** (win 72.7%) | **−0.43** (win 49.4%) |
| USD/JPY Sharpe | **4.65** (win 72.1%) | **−0.05** (win 51.9%) |
| ZN Sharpe | **4.65** (win 66.1%) | **−0.73** (win 48.8%) |

All 18 combinations (3 assets × 3 timing variants × 2 models), pooled out of sample c. mid-2016 → 2024-09, net of the
paper's costs:

| Asset | Variant | Logistic Sharpe | XGBoost Sharpe | XGBoost p | Random-sign twins' p95 |
|---|---|---|---|---|---|
| EUR/USD | P (paper) | −0.09 | −0.43 | 0.89 | +0.30 |
| EUR/USD | S (strict) | −0.06 | −0.66 | 0.97 | +0.09 |
| EUR/USD | L (one-day look-ahead) | −0.50 | −0.31 | 0.84 | +0.30 |
| USD/JPY | P | −0.28 | −0.05 | 0.57 | +0.31 |
| USD/JPY | S | −0.51 | +0.02 | 0.48 | +0.15 |
| USD/JPY | L | −0.43 | −0.29 | 0.82 | +0.31 |
| ZN | P | −0.77 | −0.73 | 0.99 | −0.67 |
| ZN | S | −0.85 | −1.00 | 0.997 | −0.50 |
| ZN | L | −0.48 | −0.71 | 0.98 | −0.67 |

- **Win rates:** 47–52% everywhere.
- **Against the random-sign twins:** no XGBoost row beats their 95th percentile at identical costs. The one
  logistic exception, ZN L (−0.48 vs −0.67), changes position on 43% of days against the twins' 50%, so it
  pays less in costs; before costs its Sharpe (+0.49) is below the twins' 95th percentile (+0.55, 200 draws).
- **Why the ZN twins are negative:** daily sign flips at 0.05% per round trip cost several percent a year.

## What this means
1. **The timing hypothesis is rejected.** Variant L feeds the model a full day of *future* news and still earns
   nothing. These headlines, top GDELT events in CAMEO codes 100–199 (demands, threats, protests, coercion,
   violence), carry no usable information about next-day FX or Treasury moves, even with look-ahead. The paper's
   Sharpe of about 5 therefore can't come from a simple news-timing leak.
2. **So the reported performance probably comes from something outside the described method.** Candidates:
   - an off-by-one between positions and returns in the backtest (e.g. trading day t's already-known return)
   - label leakage in the CV folds or tuning
   - a data or merge error

   None can be verified, because the linked repository (github.com/yukepenn/macro-news-sentiment-trading) is not
   public.
3. **Other warning signs in the paper:**
   - Win rates of 66–73% on daily FX and Treasury direction.
   - A leftover drafting note under Table 1 ("Numerical values in this table must accurately reflect results…").
   - CAMEO codes 100–199 described as "consultations and statements" (they are conflict and coercion codes).
   - Inconsistent trade counts (USD/JPY logistic: 215 trades; XGBoost: 917).
4. **It agrees with our L2 test** (GDELT tone → US500, Sharpe −0.35).

## Deviations and data checks
- **Headlines:** taken from URL slugs, not scraped pages.
  - 93% of the top-100 events have a usable slug.
  - On a random sample of 1,000, only **389 live titles** could still be retrieved, so link rot would hit the
    paper's scraping approach today too.
  - On those 389, FinBERT scores of slug and title correlate **0.66**, with **84.8% sign agreement**. The
    daily index averages about 93 headlines, so its noise from slug text is far smaller than per headline.
- **Data volume:** 3,561 of 3,563 GDELT daily files (2 missing at source), with 365,026 (SQLDATE) and 356,100
  (DATEADDED) top-100 rows. 237,618 unique headlines were scored with ProsusAI/finbert.
- **Window:** ends 2024-09-30 (the project's hold-out starts 2024-10), not April 2025 as in the paper. The first
  out-of-sample day is mid-2016, earlier than the paper's c. 2017, because TimeSeriesSplit(5) splits the shorter
  sample differently.
- **Hyperparameter grids:** our choice (the paper gives none). The logistic baseline, which has almost nothing to
  tune, is also negative.

## Prior independent replication (found after our run)
Andrews (2026) also failed to replicate the paper:
- **Publication:** "A Replication Study of Zhang (2025): Why News Sentiment Fails to Predict Asset Returns", SSRN
  6426038, 16 Mar 2026 ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6426038),
  [code](https://github.com/jandrews65/zhang2025)).
- **Result:** mean AUC 0.50 and accuracy 49.5% across EUR/USD, SPY, USO, GLD and SLV.
- **Author contact:** he contacted the author on 15 Feb 2026 and had no reply at submission.

Our replication agrees and adds three things:

| | Andrews (2026) | This replication (F1) |
|---|---|---|
| Event filter | Manuscript: all US-related event types, the paper's CAMEO 100–199 filter **not applied** (event codes discarded at ingestion). A later repository commit (1 Mar 2026) adds an EUR/USD-only run with the filter (AUC 0.51) | The paper's filter (EventBaseCode 100–199), top 100 per day |
| Assets | EUR/USD plus SPY, USO, GLD, SLV (not the paper's USD/JPY or ZN) | The paper's three: EUR/USD, USD/JPY, ZN |
| Timing diagnosis | Discusses leakage as a possible cause | **Tests it:** strict (S) and a one-day look-ahead (L) both give ~0, so a news-timing leak can't explain the paper |
| Headlines | Scraped (79% success) | URL slugs (93%), validated against 389 live titles (corr 0.66) |
| Protocol | Walk-forward with anti-leakage checks | Pre-registered, with random-sign twins at identical costs |
| Consistency | Its comparison table lists a Sharpe of 2.20* alongside AUC 0.478, and it reports its own backtest bugs (Sharpe 11–16 before fixes) | All 18 combinations between −1.00 and +0.02 |

Andrews also repeats the paper's description of codes 100–199 as "cooperation". In CAMEO they are demand, threat,
protest, coercion and violence codes.

**Implication for publishing:** a replication note from us would be the *second* independent failure. It should cite
Andrews (2026) and lead with what's new: the paper's own filter and assets, plus the timing test that rules out a
simple look-ahead leak.

## Programme status
F1 is a failed replication, not a new signal. The programme's tally stays at 22 tests, and nothing is confirmed.
The result is suitable for a short replication note, e.g. an SSRN working paper or Critical Finance Review /
Finance Research Letters. It should first request the author's code or data, and cite our L variant as evidence
that a news-timing leak alone can't explain the paper's numbers.
