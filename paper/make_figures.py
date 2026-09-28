"""Figures and placeholder numbers for the replication note, from results/F1_daily_returns.parquet
(written by scripts/diagnostics.py) and results/F1_results.csv (scripts/run_replication.py). Run from the repository root.

fig_sharpe      Sharpe ratio of every explanation per asset, against the random-sign range (Figure 1)
fig_cumulative  growth of $1, replication P vs constant growth at the reported CAGR (Figure 2)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: F401
import matplotlib.patches  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zhang_replication.replicate import COST_RT, backtest, load_prices

OUT = Path(__file__).parent
r = pd.read_parquet("results/F1_daily_returns.parquet")
res = pd.read_csv("results/F1_results.csv")
ASSETS = ["EURUSD", "USDJPY", "ZN"]
NAMES = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY", "ZN": "10y Treasury (ZN)"}
PAPER_CAGR = {"EURUSD": 0.554, "USDJPY": 0.532, "ZN": 0.221}
PAPER_SHARPE = {"EURUSD": 5.87, "USDJPY": 4.65, "ZN": 4.65}
PAPER_WIN = {"EURUSD": 0.727, "USDJPY": 0.721, "ZN": 0.661}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False})


def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252)


def as_prob(pos, idx):
    return pd.Series(np.where(pos > 0, 0.9, 0.1), idx)


# ------------------------------------------------------------------ numbers
prices = load_prices()
leak, twins = {}, {}
print("M illustration (lagged-return feature shifted to r_{t+1}):")
for a in ASSETS:
    m = r[f"{a}_M"].dropna()
    print(f"  {a}: Sharpe {sharpe(m):.2f}, win {(m > 0).mean():.1%}")
print("P from 2017-01-01:")
for a in ASSETS:
    p = r[f"{a}_P"].dropna()
    p = p[p.index >= "2017-01-01"]
    print(f"  {a}: Sharpe {sharpe(p):+.2f}, win {(p > 0).mean():.1%}")

# Partial leak (post hoc): each day independently takes the M position with probability q, else the P position, with q
# set so that the expected win rate equals the original's; costs are recomputed on the mixed positions.
# Random-sign range: 200 random-sign strategies on the P days at the same costs (the pre-registered test used 20).
print("Partial leak, 200 draws (mean [5%, 95%]); random-sign 5-95%:")
for a in ASSETS:
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
        x = backtest(as_prob(np.where(rng.random(len(idx)) < q, pM, pP), idx), fwd, COST_RT[a])
        sh.append(sharpe(x))
        win.append((x > 0).mean())
    leak[a] = dict(q=q, mean=np.mean(sh), lo=np.percentile(sh, 5), hi=np.percentile(sh, 95))
    rng = np.random.default_rng(0)
    tw = [sharpe(backtest(pd.Series(rng.random(len(idx)), idx), fwd, COST_RT[a])) for _ in range(200)]
    twins[a] = (np.percentile(tw, 5), np.percentile(tw, 95))
    print(f"  {a}: q={q:.2f}, win {np.mean(win):.1%}, Sharpe {leak[a]['mean']:.2f} "
          f"[{leak[a]['lo']:.2f}, {leak[a]['hi']:.2f}]; random-sign [{twins[a][0]:+.2f}, {twins[a][1]:+.2f}]")

# ------------------------------------------------------------------ Figure 1: Sharpe by explanation
NEWS, SHIFT, REPORTED, BAND = "#1f77b4", "#d62728", "black", "0.88"
rows = [  # label, kind (top to bottom)
    ("Reported (Zhang 2025)", "reported"),
    ("Price-feature shift, all days", "full"),
    ("Price-feature shift, some days", "partial"),
    ("L: next day's news (look-ahead)", "L"),
    ("P: same-day news (as described)", "P"),
    ("S: previous day's news (strict)", "S"),
]
xgb = res[res.model == "xgb"].set_index(["asset", "variant"])["sharpe"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, a in zip(axes, ASSETS):
    ax.axvspan(*twins[a], color=BAND, lw=0, zorder=0)
    ax.axvline(0, color="0.6", lw=0.6, zorder=1)
    for y, (label, kind) in enumerate(reversed(rows)):
        if kind in ("P", "S", "L"):
            v = xgb[(a, kind)]
            ax.plot(v, y, "o", color=NEWS, ms=5, zorder=3)
            ax.annotate(f"{v:+.2f}".replace("-", "−"), (v, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7)
        elif kind == "partial":
            v = leak[a]["mean"]
            ax.errorbar(v, y, xerr=[[v - leak[a]["lo"]], [leak[a]["hi"] - v]], fmt="o", color=SHIFT, ms=5,
                        capsize=2, lw=1, zorder=3)
            ax.annotate(f"{v:.1f}  ({leak[a]['q']:.0%} of days)", (leak[a]["hi"], y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=SHIFT)
        elif kind == "full":
            v = sharpe(r[f"{a}_M"])
            ax.plot(v, y, "o", color=SHIFT, ms=5, zorder=3)
            ax.annotate(f"{v:.1f}", (v, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7,
                        color=SHIFT)
        else:
            v = PAPER_SHARPE[a]
            ax.plot(v, y, "D", color=REPORTED, ms=5, zorder=3)
            ax.annotate(f"{v:.2f}", (v, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7)
    ax.set_title(NAMES[a])
    ax.set_xlim(-2.5, 20)
    ax.set_xlabel("Annualised Sharpe ratio, net of costs")
    ax.grid(axis="x", alpha=0.3, lw=0.5)
axes[0].set_yticks(range(len(rows)), [lbl for lbl, _ in reversed(rows)])
axes[0].set_ylim(-0.6, len(rows) - 0.3)
handles = [
    matplotlib.lines.Line2D([], [], marker="D", ls="", color=REPORTED, ms=5, label="Reported by the original"),
    matplotlib.lines.Line2D([], [], marker="o", ls="", color=NEWS, ms=5, label="News timing variants (pre-registered)"),
    matplotlib.lines.Line2D([], [], marker="o", ls="", color=SHIFT, ms=5, label="Price-feature shift (post hoc)"),
    matplotlib.patches.Patch(color=BAND, label="Random-sign strategies, 5–95%"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_sharpe.pdf")
fig.savefig(OUT / "fig_sharpe.png", dpi=160)

# ------------------------------------------------------------------ Figure 2: growth of $1
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
for ax, a in zip(axes, ASSETS):
    s = r[f"{a}_P"].dropna()
    years = np.arange(len(s)) / 252
    ax.plot(s.index, (1 + PAPER_CAGR[a]) ** years, "--", color="0.45", lw=1.2,
            label="Constant growth at the reported CAGR")
    ax.plot(s.index, (1 + s).cumprod().to_numpy(), color=NEWS, lw=1.3, label="Replication (P, XGBoost)")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(NAMES[a])
    ax.grid(alpha=0.3, which="major", lw=0.5)
axes[0].set_ylim(0.5, 60)
axes[0].set_ylabel("Growth of $1 (log scale)")
axes[2].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_cumulative.pdf")
fig.savefig(OUT / "fig_cumulative.png", dpi=160)
