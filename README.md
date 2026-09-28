# Replication of Zhang (2025), "Interpretable Machine Learning for Macro Alpha"

A pre-registered replication of Zhang, Y. (2025), *Interpretable Machine Learning for Macro Alpha: A News Sentiment
Case Study*, [arXiv:2505.16136](https://arxiv.org/abs/2505.16136). The paper reports cost-adjusted out-of-sample
Sharpe ratios of 5.87 (EUR/USD), 4.65 (USD/JPY) and 4.65 (10-year Treasury futures) from FinBERT sentiment of GDELT
headlines and an XGBoost classifier.

**Result: not replicated.** Following the paper's event filter, assets, features, models, validation and costs:

| | Paper | This replication (XGBoost, paper's timing) |
|---|---|---|
| EUR/USD | Sharpe 5.87, win 72.7% | Sharpe −0.43, win 49.4% |
| USD/JPY | Sharpe 4.65, win 72.1% | Sharpe −0.05, win 51.9% |
| ZN | Sharpe 4.65, win 66.1% | Sharpe −0.73, win 48.8% |

- **Every variant fails.** All 18 asset × timing × model combinations lie between −1.00 and +0.02, and none beats
  random-sign trading at the same costs.
- **News timing isn't the explanation.** A variant fed the *next* day's news also earns nothing.
- **A price-feature shift is a candidate.** Shifting the lagged-return feature by one row (post hoc illustration)
  gives Sharpe 15–18.
- **Details:** [`results/RESULTS.md`](results/RESULTS.md) and the note in [`paper/`](paper/).
- **Prior work:** Andrews (2026, [SSRN 6426038](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6426038)).

## Repository
| Path | Contents |
|---|---|
| `preregistration/PREREGISTRATION.md` | Design and verdict rules, committed before any result |
| `zhang_replication/` | GDELT download and filter, URL-slug headlines, FinBERT scoring, features, models, backtest |
| `scripts/run_replication.py` | Full replication (`--smoke` for a code check, `--validate-slugs` for the headline check) |
| `scripts/diagnostics.py` | Post hoc: saves daily returns, runs the one-row shift illustration |
| `results/` | Results write-up, all-combination table, daily returns, **FinBERT scores for 237,618 headlines** |
| `paper/` | LaTeX replication note and figure (compile with Overleaf or `pdflatex`) |
| `tests/` | Unit tests for the timing alignment, slug parsing and cost model |

## Reproduce
```bash
pip install -r requirements.txt
python -m zhang_replication.gdelt_events       # ~45 GB streamed, ~300 MB kept in data/gdelt_events (resumable)
python scripts/run_replication.py              # uses results/finbert_scores.parquet, so no GPU is needed
python scripts/run_replication.py --validate-slugs
python scripts/diagnostics.py && python paper/make_figures.py
pytest
```
Yahoo Finance prices are downloaded on first run. The shipped FinBERT scores are loaded automatically; delete or
rename `results/finbert_scores.parquet` to rescore (about 5 hours on a laptop CPU).

## Citation
Ojebiyi, T. O. (2026). *Timing cannot explain it: A pre-registered replication of Zhang (2025), "Interpretable
Machine Learning for Macro Alpha"*. Working paper. Contact: tobiojebiyi@gmail.com

License: MIT (code). GDELT data are subject to the GDELT Project's terms; FinBERT is ProsusAI/finbert.
