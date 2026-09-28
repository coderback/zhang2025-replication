"""F1 diagnostics for the replication note (POST HOC, not part of the pre-registered verdict).

1. Saves the out-of-sample daily strategy returns of the pre-registered P-variant XGBoost for the figures.
2. Illustration M ("return-feature leak"): the same pipeline, except the lagged-return feature r_t is replaced by the
   next-day return r_{t+1}, a one-row shift error. This shows the magnitude a single off-by-one in a price feature
   produces; it is not a claim about what the original code did.
Uses the cached FinBERT scores (no model loading)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import numpy as np
import pandas as pd

from zhang_replication.stats import one_sided_p
from zhang_replication.replicate import COST_RT, backtest, daily_index, features, load_prices, load_top, metrics, walk_forward
from zhang_replication.sentiment import load_cache, slug_text

END = "2024-09-30"
top = load_top(end="2024-10-02")["sqldate"]
top["text"] = top["url"].map(slug_text)
idx = daily_index(top, load_cache())
prices = load_prices()
daily, rows = {}, {}
for asset, close in prices.items():
    close = close[close.index <= END]
    f = features(idx, close, "P")
    for label, frame in (("P", f), ("M", f.assign(ret=f["fwd"]))):
        prob = walk_forward(frame, "xgb")
        r = backtest(prob, frame["fwd"], COST_RT[asset])
        m = metrics(r)
        m["p"] = float(one_sided_p(r.to_numpy(), block=21))
        rows[(asset, label)] = m
        daily[f"{asset}_{label}"] = r
        print(f"{asset} {label}: Sharpe {m['sharpe']:+.2f}, CAGR {m['cagr']:+.1%}, win {m['win']:.1%}", flush=True)
pd.DataFrame(daily).to_parquet("results/F1_daily_returns.parquet")
print(pd.DataFrame(rows).T[["sharpe", "cagr", "vol", "max_dd", "win", "p"]].astype(float).round(3).to_string())
