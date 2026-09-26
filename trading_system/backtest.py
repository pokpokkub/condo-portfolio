"""Vectorised monthly backtester and performance metrics."""
import numpy as np
import pandas as pd


def run(weights, rets, lag=2, cost=0.001):
    """Backtest target weights against asset returns.

    weights : DataFrame of target weights decided with data up to the row's month.
    lag     : months between the decision and the first return earned. Shiller
              prices are monthly *averages*, so lag=2 is used to remove any
              look-ahead from averaging (decide at month t, earn t+1 -> t+2).
    cost    : one-way transaction cost per unit of turnover (0.1% default).
    """
    w = weights.reindex(rets.index).shift(lag).fillna(0.0)
    r = rets[w.columns].fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    port = (w * r).sum(axis=1) - turnover * cost
    first = w.abs().sum(axis=1).ne(0).idxmax()
    return port.loc[first:], w.loc[first:]


def trades(w, rets):
    """Per-asset round trips: contiguous months with weight > 0. Returns list of trade returns."""
    out = []
    for col in w.columns:
        if col == "cash":
            continue
        held = w[col] > 0
        ret = rets[col].reindex(w.index).fillna(0.0)
        grp = (held != held.shift()).cumsum()
        for _, idx in held[held].groupby(grp[held]).groups.items():
            out.append((1 + ret.loc[idx]).prod() - 1)
    return np.array(out)


def metrics(port, w=None, rets=None):
    eq = (1 + port).cumprod()
    years = len(port) / 12
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = port.std() * np.sqrt(12)
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    pos, neg = port[port > 0].sum(), -port[port < 0].sum()
    m = {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe(0%)": port.mean() * 12 / vol if vol else np.nan,
        "MaxDD": mdd,
        "Calmar": cagr / -mdd if mdd else np.nan,
        "Ulcer": np.sqrt((dd ** 2).mean()),
        "PF(monthly)": pos / neg if neg else np.inf,
        "WinMonths": (port > 0).mean(),
        "Worst12m": ((1 + port).rolling(12).apply(np.prod, raw=True) - 1).min(),
        "Years": years,
    }
    if w is not None:
        t = trades(w, rets)
        if len(t):
            m["Trades"] = len(t)
            m["WinRate"] = (t > 0).mean()
            m["PF(trades)"] = t[t > 0].sum() / -t[t < 0].sum() if (t < 0).any() else np.inf
    return m


def table(results):
    df = pd.DataFrame(results).T
    fmt = {c: "{:.1%}" for c in ["CAGR", "Vol", "MaxDD", "Ulcer", "WinMonths", "Worst12m", "WinRate"]}
    fmt.update({c: "{:.2f}" for c in ["Sharpe(0%)", "Calmar", "PF(monthly)", "PF(trades)"]})
    fmt.update({"Years": "{:.0f}", "Trades": "{:.0f}"})
    out = df.copy().astype(object)
    for c, f in fmt.items():
        if c in df:
            out[c] = df[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    return out
