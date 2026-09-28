"""Figure and placeholder numbers for the replication note, from results/F1_daily_returns.parquet
(written by scripts/diagnostics.py). Run from the repository root."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
r = pd.read_parquet("results/F1_daily_returns.parquet")
PAPER_CAGR = {"EURUSD": 0.554, "USDJPY": 0.532, "ZN": 0.221}
NAMES = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY", "ZN": "10y Treasury (ZN)"}


def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252)


fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
for ax, a in zip(axes, PAPER_CAGR):
    s = r[f"{a}_P"].dropna()
    eq = (1 + s).cumprod()
    years = np.arange(len(s)) / 252
    ax.plot(s.index, (1 + PAPER_CAGR[a]) ** years, "--", color="0.45", lw=1.2, label="Original (reported CAGR)")
    ax.plot(s.index, eq.to_numpy(), color="C0", lw=1.4, label="Replication P, XGBoost")
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(NAMES[a], fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Growth of $1 (log scale)", fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_cumulative.pdf")
fig.savefig(OUT / "fig_cumulative.png", dpi=160)

print("M illustration (lagged-return feature shifted to r_{t+1}):")
for a in PAPER_CAGR:
    m = r[f"{a}_M"].dropna()
    print(f"  {a}: Sharpe {sharpe(m):.2f}, win {(m > 0).mean():.1%}")
print("P from 2017-01-01:")
for a in PAPER_CAGR:
    p = r[f"{a}_P"].dropna()
    p = p[p.index >= "2017-01-01"]
    print(f"  {a}: Sharpe {sharpe(p):+.2f}, win {(p > 0).mean():.1%}")

# Partial leak (post hoc): each day independently uses the M position with probability q, else the P position,
# with q set so that the expected win rate equals the original's; costs recomputed on the mixed positions.
from zhang_replication.replicate import COST_RT, backtest, load_prices  # noqa: E402

PAPER_WIN = {"EURUSD": 0.727, "USDJPY": 0.721, "ZN": 0.661}
prices = load_prices()
print("Partial leak, 200 draws (mean [5%, 95%]):")
for a in PAPER_CAGR:
    c = prices[a][prices[a].index <= "2024-09-30"]
    fwd = np.log(c).diff().shift(-1)
    P, M = r[f"{a}_P"].dropna(), r[f"{a}_M"].dropna()
    idx = P.index.intersection(M.index)
    ret = np.expm1(fwd.reindex(idx))

    def pos(s):  # the sign whose gross return is closest to the net strategy return
        s = s.reindex(idx)
        return np.where(np.abs(s - ret) <= np.abs(s + ret), 1.0, -1.0)

    pP, pM = pos(P), pos(M)
    q = (PAPER_WIN[a] - 0.5) / ((M.reindex(idx) > 0).mean() - 0.5)
    rng = np.random.default_rng(1)
    sh, win = [], []
    for _ in range(200):
        mix = np.where(rng.random(len(idx)) < q, pM, pP)
        x = backtest(pd.Series(np.where(mix > 0, 0.9, 0.1), idx), fwd, COST_RT[a])
        sh.append(sharpe(x))
        win.append((x > 0).mean())
    print(f"  {a}: q={q:.2f}, win {np.mean(win):.1%}, Sharpe {np.mean(sh):.2f} "
          f"[{np.percentile(sh, 5):.2f}, {np.percentile(sh, 95):.2f}]")
