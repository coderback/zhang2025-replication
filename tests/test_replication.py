import numpy as np
import pandas as pd

from zhang_replication.replicate import backtest, daily_index, features, top_events
from zhang_replication.sentiment import slug_text


def test_slug_text_keeps_headline_words_and_drops_ids():
    assert slug_text("https://x.com/2019/03/12/fed-holds-rates-steady.html") == "fed holds rates steady"
    assert slug_text("https://x.co.uk/news/17493833.ethiopia-airlines-crash-grounded/") == "ethiopia airlines crash grounded"
    assert slug_text("https://x.com/article/1234567") is None
    assert slug_text(None) is None


def test_top_events_ranks_by_articles_per_day_key():
    ev = pd.DataFrame({"sqldate": ["20200101"] * 3 + ["20200102"], "dateadded": ["20200102"] * 4,
                       "num_articles": [5, 50, 20, 1], "goldstein": 0.0, "text": list("abcd")})
    top = top_events(ev, "sqldate")
    assert top[top.day == "2020-01-01"].text.tolist()[0] == "b"
    assert set(top_events(ev, "dateadded").day) == {pd.Timestamp("2020-01-02")}


def _index(days):
    idx = pd.DataFrame({"S": np.arange(len(days), dtype=float), "sigma": 0.1, "N": 100, "G": 0.0, "sigmaG": 1.0},
                       index=days)
    idx["logN"], idx["AI"] = np.log1p(idx.N), idx.S * np.log1p(idx.N)
    return idx


def test_variants_align_news_to_the_right_day():
    cal = pd.date_range("2020-01-01", periods=80, freq="D")
    idx = _index(cal)                                         # S on news day d = its position in the calendar
    trade = pd.bdate_range("2020-01-06", periods=40)
    close = pd.Series(np.linspace(1, 2, len(trade)), trade)
    pos = {d: i for i, d in enumerate(cal)}
    for variant, expect in (("P", lambda t: pos[t]), ("S", lambda t: pos[t - pd.Timedelta(days=1)])):
        f = features(idx, close, variant)
        t = f.index[5]
        assert f.loc[t, "S"] == expect(t), variant
    f = features(idx, close, "L")
    t = f.index[5]
    nxt = trade[trade.get_loc(t) + 1]                         # L = news of the next trading day (the leak)
    assert f.loc[t, "S"] == pos[nxt]


def test_target_is_the_next_day_return_and_no_feature_uses_it():
    cal = pd.date_range("2020-01-01", periods=80, freq="D")
    trade = pd.bdate_range("2020-01-06", periods=40)
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(trade)))), trade)
    f = features(_index(cal), close, "P")
    lr = np.log(close).diff()
    t = f.index[3]
    assert np.isclose(f.loc[t, "fwd"], lr.shift(-1).loc[t]) and np.isclose(f.loc[t, "ret"], lr.loc[t])
    assert f.loc[t, "y"] == float(lr.shift(-1).loc[t] > 0)


def test_backtest_costs_half_round_trip_per_unit_change():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prob = pd.Series([0.9, 0.9, 0.1], idx)
    fwd = pd.Series([0.0, 0.0, 0.0], idx)
    r = backtest(prob, fwd, 0.0002)
    assert np.allclose(r.to_numpy(), [-0.0001, 0.0, -0.0002])    # open 1 unit, hold, flip 2 units
