"""F1: replication of Zhang (2025) "Interpretable ML for Macro Alpha" (arXiv 2505.16136), exactly as pre-registered
(preregistration/PREREGISTRATION.md, 260681b). Primary window ends 2024-09-30 (the project's
hold-out starts 2024-10). Stages: slugs -> FinBERT (cached, resumable) -> daily indices -> P/S/L variants ->
logistic + XGBoost walk-forward -> backtest vs random-sign twins -> verdict.

    python scripts/run_replication.py            full run
    python scripts/run_replication.py --smoke    code check with random scores instead of FinBERT (no information)
    python scripts/run_replication.py --validate-slugs   scrape 1,000 real titles and compare with slug sentiment
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import sys

import numpy as np
import pandas as pd

from zhang_replication.stats import one_sided_p
from zhang_replication.replicate import (COST_RT, backtest, daily_index, features, load_prices, load_top, metrics,
                                    random_sign_sharpes, walk_forward)
from zhang_replication.sentiment import slug_text

pd.set_option("display.width", 220)
SPEC = "preregistration/PREREGISTRATION.md"
END = "2024-09-30"
SMOKE = "--smoke" in sys.argv

top = load_top(end="2024-10-02")                               # memory-light top-100 per news day (two passes)
for k in top:
    top[k]["text"] = top[k]["url"].map(slug_text)
print(f"top-100 rows: SQLDATE {len(top['sqldate']):,} (usable slugs {top['sqldate'].text.notna().mean():.1%}), "
      f"DATEADDED {len(top['dateadded']):,} (usable slugs {top['dateadded'].text.notna().mean():.1%})", flush=True)

if "--validate-slugs" in sys.argv:
    import concurrent.futures as cf
    import html
    import re
    import urllib.request
    from zhang_replication.sentiment import FinBert
    sample = top["sqldate"].dropna(subset=["text"]).sample(1000, random_state=7)

    def title(url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=10).read(200_000).decode("utf-8", "ignore")
            m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
            return html.unescape(m.group(1)).strip() if m else None
        except Exception:
            return None
    with cf.ThreadPoolExecutor(16) as ex:
        sample["title"] = list(ex.map(title, sample["url"]))
    ok = sample.dropna(subset=["title"])
    ok = ok[ok.title.str.len() > 10]
    fb = FinBert()
    a, b = fb.score(ok.text.tolist()), fb.score(ok.title.tolist())
    print(f"SLUG VALIDATION: {len(ok)}/1000 titles retrieved; corr(FinBERT slug, FinBERT title) = "
          f"{np.corrcoef(a, b)[0, 1]:.3f}; sign agreement {np.mean(np.sign(a) == np.sign(b)):.1%}")
    sys.exit()

texts = pd.concat([top["sqldate"]["text"], top["dateadded"]["text"]]).dropna().unique().tolist()
if SMOKE:
    scores = pd.Series(np.random.default_rng(0).uniform(-1, 1, len(texts)), index=texts)
else:
    from zhang_replication.sentiment import score_all
    scores = score_all(texts)
idx = {"P": daily_index(top["sqldate"], scores), "S": daily_index(top["dateadded"], scores)}
idx["L"] = idx["P"]
prices = load_prices()

rows = {}
for asset, close in prices.items():
    close = close[close.index <= END]
    for variant in ("P", "S", "L"):
        f = features(idx[variant], close, variant)
        for model in ("logit", "xgb"):
            prob = walk_forward(f, model)
            r = backtest(prob, f["fwd"], COST_RT[asset])
            m = metrics(r)
            m["p"] = float(one_sided_p(r.to_numpy(), block=21))
            tw = random_sign_sharpes(f["fwd"], prob.index, COST_RT[asset])
            m["twin_p95"] = float(np.quantile(tw, 0.95))
            m["oos_from"] = str(prob.index.min().date())
            rows[(asset, variant, model)] = m
            print(f"  {asset} {variant} {model}: Sharpe {m['sharpe']:+.2f}, CAGR {m['cagr']:+.1%}, win {m['win']:.1%}, "
                  f"p {m['p']:.3f}, twins p95 {m['twin_p95']:+.2f} (OOS from {m['oos_from']})", flush=True)

res = pd.DataFrame(rows).T
res.index.names = ["asset", "variant", "model"]
if not SMOKE:
    res.to_csv("results/F1_results.csv")
print("\n", res[["sharpe", "cagr", "vol", "max_dd", "win", "p", "twin_p95", "days"]].astype(float).round(3).to_string())

x = res.xs("xgb", level="model")["sharpe"].astype(float)
P, S, L = (x.xs(v, level="variant") for v in ("P", "S", "L"))
if (P >= 2).sum() >= 2:
    verdict = "1 REPLICATED; " + ("genuine (S also >= 2): enters the programme as test 23" if (S >= 2).sum() >= 2
                                   else "same-day timing effect (S < 2)")
elif (P < 1).all() and (L >= 2).sum() >= 2:
    verdict = "2 NOT REPLICATED - a one-day look-ahead misalignment reproduces the paper's numbers"
elif (P < 1).all():
    verdict = "3 NOT REPLICATED - the described method has no edge; the paper's numbers are unexplained without its code"
else:
    verdict = "outside the pre-registered categories (P between 1 and 2 on some asset): reported as is"
print(f"\nVERDICT ({'SMOKE TEST, meaningless' if SMOKE else SPEC}): {verdict}")
