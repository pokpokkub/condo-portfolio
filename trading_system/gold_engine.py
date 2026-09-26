"""Bar-by-bar backtester for short-term XAUUSD CFD strategies on daily bars.

Execution model (deliberately conservative):
  - Signals are computed on a bar's close; orders fill at the NEXT bar's open.
  - Stop-loss / take-profit are checked on each bar's high/low. If both could have
    been hit inside the same bar, the stop is assumed to fill first. A gap through
    the stop fills at the open (worse than the stop).
  - Every fill pays half the spread plus slippage (USD per oz), and each night a
    position is held pays swap (annual % of notional, long and short separately).
  - Position size risks a fixed fraction of equity on the distance to the stop.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load_d1():
    return pd.read_csv(DATA / "xauusd_d1_oanda.csv", parse_dates=["date"], index_col="date")


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class Costs:
    spread: float = 0.35      # USD/oz round-trip spread (typical retail XAUUSD CFD: 0.20-0.50)
    slippage: float = 0.10    # USD/oz per fill
    swap_long: float = -0.04  # annual financing on notional for longs
    swap_short: float = -0.01  # annual financing for shorts (conservative: also a cost)


@dataclass
class Signals:
    """Per-bar arrays produced by a strategy (all evaluated at the bar's close)."""
    entry: pd.Series            # +1 long, -1 short, 0 none
    stop_dist: pd.Series        # price distance from entry to stop (> 0)
    target_dist: pd.Series = None  # optional take-profit distance
    exit_long: pd.Series = None    # close-based exit condition for longs
    exit_short: pd.Series = None
    max_bars: int = 0              # time stop (0 = none)
    trail_mult: pd.Series = None   # optional trailing stop distance (price units)


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def backtest(df, sig: Signals, risk=0.01, costs=Costs(), start_equity=10_000.0, max_lev=10.0):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    idx = df.index
    n = len(df)
    entry = sig.entry.reindex(idx).fillna(0).values
    sd = sig.stop_dist.reindex(idx).values
    td = sig.target_dist.reindex(idx).values if sig.target_dist is not None else np.full(n, np.nan)
    xl = sig.exit_long.reindex(idx).fillna(False).values if sig.exit_long is not None else np.zeros(n, bool)
    xs = sig.exit_short.reindex(idx).fillna(False).values if sig.exit_short is not None else np.zeros(n, bool)
    tr = sig.trail_mult.reindex(idx).values if sig.trail_mult is not None else None
    fill_cost = costs.spread / 2 + costs.slippage

    eq = start_equity
    curve = np.empty(n)
    trades = []
    pos = 0; size = 0.0; px = stop = tgt = np.nan; bars = 0; ent_i = 0; pend_exit = False; best = np.nan

    for i in range(n):
        # 1) execute pending orders at this bar's open
        if pos != 0 and pend_exit:
            fill = o[i] - pos * fill_cost
            pnl = pos * (fill - px) * size
            eq += pnl
            trades.append((idx[ent_i], idx[i], pos, px, fill, pnl, "exit"))
            pos = 0; pend_exit = False
        if pos == 0 and i > 0 and entry[i - 1] != 0 and np.isfinite(sd[i - 1]) and sd[i - 1] > 0:
            pos = int(entry[i - 1])
            px = o[i] + pos * fill_cost
            size = min(eq * risk / sd[i - 1], eq * max_lev / px)
            stop = px - pos * sd[i - 1]
            tgt = px + pos * td[i - 1] if np.isfinite(td[i - 1]) else np.nan
            bars = 0; ent_i = i; best = px

        # 2) intrabar stop / target
        if pos != 0:
            hit = None
            if pos > 0:
                if l[i] <= stop:
                    hit = min(o[i], stop) if i != ent_i else stop
                elif np.isfinite(tgt) and h[i] >= tgt:
                    hit = max(o[i], tgt) if i != ent_i else tgt
            else:
                if h[i] >= stop:
                    hit = max(o[i], stop) if i != ent_i else stop
                elif np.isfinite(tgt) and l[i] <= tgt:
                    hit = min(o[i], tgt) if i != ent_i else tgt
            if hit is not None:
                fill = hit - pos * fill_cost
                pnl = pos * (fill - px) * size
                eq += pnl
                trades.append((idx[ent_i], idx[i], pos, px, fill, pnl, "stop/target"))
                pos = 0

        # 3) end-of-bar bookkeeping: swap, trailing stop, close-based exits
        if pos != 0:
            swap = costs.swap_long if pos > 0 else costs.swap_short
            nights = 3 if idx[i].dayofweek == 2 else 1  # triple swap Wednesday
            eq += swap / 365 * nights * c[i] * size
            bars += 1
            if tr is not None and np.isfinite(tr[i]):
                best = max(best, h[i]) if pos > 0 else min(best, l[i])
                new = best - pos * tr[i]
                stop = max(stop, new) if pos > 0 else min(stop, new)
            if (pos > 0 and xl[i]) or (pos < 0 and xs[i]) or (sig.max_bars and bars >= sig.max_bars):
                pend_exit = True
        curve[i] = eq + (pos * (c[i] - px) * size if pos != 0 else 0.0)

    tdf = pd.DataFrame(trades, columns=["entry_date", "exit_date", "side", "entry", "exit", "pnl", "reason"])
    equity = pd.Series(curve, index=idx)
    return Result(equity, tdf, stats(equity, tdf))


def stats(equity, trades):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = equity / equity.cummax() - 1
    rets = equity.pct_change().dropna()
    wins, losses = trades.pnl[trades.pnl > 0], trades.pnl[trades.pnl <= 0]
    return {
        "CAGR": cagr,
        "MaxDD": dd.min(),
        "Ret/DD": cagr / -dd.min() if dd.min() < 0 else np.nan,
        "Sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() else np.nan,
        "PF": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.inf,
        "Trades": len(trades),
        "Trades/yr": len(trades) / years,
        "WinRate": len(wins) / len(trades) if len(trades) else np.nan,
        "AvgWin/AvgLoss": wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan,
        "Exposure": np.nan,
    }


def fmt(rows):
    df = pd.DataFrame(rows).T.drop(columns=["Exposure"], errors="ignore")
    out = df.astype(object)
    for col in df:
        if col in ("CAGR", "MaxDD", "WinRate"):
            out[col] = df[col].map(lambda v: f"{v:.1%}")
        elif col in ("Trades",):
            out[col] = df[col].map(lambda v: f"{v:.0f}")
        else:
            out[col] = df[col].map(lambda v: f"{v:.2f}")
    return out
