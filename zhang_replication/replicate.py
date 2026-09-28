"""F1 replication engine (preregistration/PREREGISTRATION.md): daily sentiment indices per news day,
the paper's feature set on the trading-day calendar, logistic and XGBoost classifiers in a 5-fold expanding window
with inner time-series tuning, and the long/short backtest with the paper's costs.

Variants set which news day feeds the prediction made at the close of trading day t (for the return t -> t+1):
    P  events dated t (SQLDATE = t)                    the paper as described
    S  events added on the calendar day before t       strict: all news predates the close of t
    L  events dated on the next trading day (t+1)      a one-day look-ahead misalignment (diagnostic)
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd

from zhang_replication.config import DATA_DIR

TOP_N = 100
PRICES = DATA_DIR / "prices_yahoo"
TICKERS = {"EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "ZN": "ZN=F"}
COST_RT = {"EURUSD": 0.0002, "USDJPY": 0.0002, "ZN": 0.0005}      # per round trip; half per unit of position change
LOGIT_C = (0.01, 0.1, 1.0, 10.0)
XGB_GRID = [dict(max_depth=d, learning_rate=lr, reg_lambda=lam)
            for d, lr, lam in itertools.product((2, 3, 4), (0.03, 0.1), (1.0, 5.0))]


# --------------------------------------------------------------------------------------------- prices
def load_prices(start="2014-11-01", end="2024-10-01") -> dict[str, pd.Series]:
    """Yahoo daily closes (the paper's source), cached as CSV."""
    PRICES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, tk in TICKERS.items():
        p = PRICES / f"{name}.csv"
        if not p.exists():
            import yfinance as yf
            df = yf.download(tk, start=start, end=end, auto_adjust=False, progress=False)
            close = df["Close"]
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            close.rename("close").to_csv(p)
        s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        out[name] = s[~s.index.duplicated()].dropna().astype(float)
    return out


# --------------------------------------------------------------------------------------------- sentiment indices
def top_events(ev: pd.DataFrame, key: str) -> pd.DataFrame:
    """Top ``TOP_N`` events by NumArticles per news day, with the day taken from ``key`` (sqldate or dateadded)."""
    e = ev.assign(day=pd.to_datetime(ev[key], format="%Y%m%d", errors="coerce")).dropna(subset=["day"])
    e = e.sort_values(["day", "num_articles"], ascending=[True, False])
    return e.groupby("day", group_keys=False).head(TOP_N)


def load_top(end: str = "2024-10-02") -> dict[str, pd.DataFrame]:
    """Memory-light equivalent of ``top_events(load(), key)`` for key in (sqldate, dateadded): pass 1 reads only the
    small ranking columns of every daily file, pass 2 reads url and goldstein for the selected rows only."""
    from zhang_replication.gdelt_events import OUT
    files = [f for f in sorted(OUT.glob("*.parquet")) if f.stem <= pd.Timestamp(end).strftime("%Y%m%d")]
    parts = []
    for i, f in enumerate(files):
        d = pd.read_parquet(f, columns=["sqldate", "dateadded", "num_articles"])
        parts.append(pd.DataFrame({"file": np.int32(i), "row": np.arange(len(d), dtype=np.int32),
                                   "sqldate": pd.to_numeric(d["sqldate"], errors="coerce"),
                                   "dateadded": pd.to_numeric(d["dateadded"], errors="coerce"),
                                   "num_articles": d["num_articles"].to_numpy()}))
    rank = pd.concat(parts, ignore_index=True)
    picks = {}
    for key in ("sqldate", "dateadded"):
        r = rank.dropna(subset=[key]).sort_values([key, "num_articles"], ascending=[True, False], kind="stable")
        picks[key] = r.groupby(key, group_keys=False).head(TOP_N)[["file", "row", key, "num_articles"]]
    need = pd.concat([p[["file", "row"]] for p in picks.values()]).drop_duplicates()
    detail = []
    for i, g in need.groupby("file"):
        d = pd.read_parquet(files[i], columns=["url", "goldstein"]).iloc[g["row"].to_numpy()]
        detail.append(d.assign(file=np.int32(i), row=g["row"].to_numpy()))
    detail = pd.concat(detail, ignore_index=True)
    out = {}
    for key, p in picks.items():
        t = p.merge(detail, on=["file", "row"], how="left")
        t["day"] = pd.to_datetime(t[key].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
        out[key] = t.dropna(subset=["day"]).drop(columns=["file", "row"])
    return out


def daily_index(top: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Per news day: mean S, std sigma, N, log(1+N), article impact, Goldstein mean and std (paper §3.3)."""
    t = top.assign(s=top["text"].map(scores)).dropna(subset=["s"])
    g = t.groupby("day")
    d = pd.DataFrame({"S": g["s"].mean(), "sigma": g["s"].std(ddof=0), "N": g["s"].size(),
                      "G": g["goldstein"].mean(), "sigmaG": g["goldstein"].std(ddof=0)})
    d["logN"] = np.log1p(d["N"])
    d["AI"] = d["S"] * d["logN"]
    return d


def features(idx: pd.DataFrame, close: pd.Series, variant: str) -> pd.DataFrame:
    """The paper's features on the trading-day calendar of ``close``, for the given timing variant, plus the
    target y = 1[r_{t+1} > 0] and the realised next-day log return."""
    days = close.index
    if variant == "P":
        news_day = days
    elif variant == "S":
        news_day = days - pd.Timedelta(days=1)
    elif variant == "L":
        news_day = pd.DatetimeIndex(np.r_[days[1:], [pd.NaT]])
    else:
        raise ValueError(variant)
    f = idx.reindex(news_day).set_axis(days)
    for c in ("S", "sigma", "N", "G"):
        for k in (1, 2, 3):
            f[f"{c}_lag{k}"] = f[c].shift(k)
    f["S_ma5"], f["S_ma20"] = f["S"].rolling(5).mean(), f["S"].rolling(20).mean()
    f["dS"] = f["S_ma5"] - f["S_ma20"]
    f["S_sd5"], f["S_sd10"] = f["S"].rolling(5).std(), f["S"].rolling(10).std()
    f["N_sum5"], f["N_sum10"] = f["N"].rolling(5).sum(), f["N"].rolling(10).sum()
    lr = np.log(close).diff()
    f["ret"] = lr
    f["vol20"] = lr.rolling(20).std() * math.sqrt(252)
    f["fwd"] = lr.shift(-1)
    f["y"] = (f["fwd"] > 0).astype(float)
    return f.iloc[20:].dropna()


# --------------------------------------------------------------------------------------------- models
def _inner_splits(n: int, k: int = 3):
    from sklearn.model_selection import TimeSeriesSplit
    return list(TimeSeriesSplit(n_splits=k).split(np.arange(n)))


def fit_logit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best, best_c = np.inf, LOGIT_C[0]
    for c in LOGIT_C:
        losses = []
        for tr, va in _inner_splits(len(y)):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(X[tr], y[tr])
            losses.append(log_loss(y[va], m.predict_proba(X[va])[:, 1], labels=[0, 1]))
        if np.mean(losses) < best:
            best, best_c = np.mean(losses), c
    return make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=2000)).fit(X, y)


def _xgb(params, X, y, n_trees=None):
    from xgboost import XGBClassifier
    cut = int(len(y) * 0.8)
    if n_trees is None:                                             # early stopping on the last 20% of the window
        m = XGBClassifier(n_estimators=500, early_stopping_rounds=30, eval_metric="logloss", verbosity=0,
                          n_jobs=4, **params)
        m.fit(X[:cut], y[:cut], eval_set=[(X[cut:], y[cut:])], verbose=False)
        return m, max(10, (m.best_iteration or 0) + 1)
    m = XGBClassifier(n_estimators=n_trees, eval_metric="logloss", verbosity=0, n_jobs=4, **params)
    return m.fit(X, y), n_trees


def fit_xgb(X: np.ndarray, y: np.ndarray):
    from sklearn.metrics import log_loss
    best, choice = np.inf, None
    for params in XGB_GRID:
        losses = []
        for tr, va in _inner_splits(len(y)):
            m, _ = _xgb(params, X[tr], y[tr])
            losses.append(log_loss(y[va], m.predict_proba(X[va])[:, 1], labels=[0, 1]))
        if np.mean(losses) < best:
            best, choice = np.mean(losses), params
    _, n = _xgb(choice, X, y)
    return _xgb(choice, X, y, n_trees=n)[0]


def walk_forward(f: pd.DataFrame, model: str, n_splits: int = 5) -> pd.Series:
    """Out-of-sample P(up) from an outer TimeSeriesSplit(5): each fold trains on everything before it."""
    from sklearn.model_selection import TimeSeriesSplit
    cols = [c for c in f.columns if c not in ("fwd", "y")]
    X, y = f[cols].to_numpy(float), f["y"].to_numpy(int)
    p = pd.Series(np.nan, f.index)
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit_logit(X[tr], y[tr]) if model == "logit" else fit_xgb(X[tr], y[tr])
        p.iloc[te] = m.predict_proba(X[te])[:, 1]
    return p.dropna()


# --------------------------------------------------------------------------------------------- backtest
def backtest(prob: pd.Series, fwd: pd.Series, cost_rt: float) -> pd.Series:
    """Daily strategy return: +1 if P(up) > 0.5 else -1, times the next-day simple return, less half the round-trip
    cost per unit of position change."""
    pos = np.where(prob > 0.5, 1.0, -1.0)
    ret = np.expm1(fwd.reindex(prob.index).to_numpy())
    dpos = np.abs(np.diff(np.r_[0.0, pos]))
    return pd.Series(pos * ret - dpos * cost_rt / 2, prob.index)


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"sharpe": float(r.mean() / r.std() * math.sqrt(252)), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "vol": float(r.std() * math.sqrt(252)), "max_dd": float((1 - eq / eq.cummax()).max()),
            "win": float((r > 0).mean()), "days": len(r)}


def random_sign_sharpes(fwd: pd.Series, index: pd.Index, cost_rt: float, n: int = 20, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        prob = pd.Series(rng.random(len(index)), index)
        out.append(metrics(backtest(prob, fwd, cost_rt))["sharpe"])
    return out
