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
