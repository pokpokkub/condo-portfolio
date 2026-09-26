"""Intraday (flat-by-close) XAUUSD simulator on 1-minute bars, London local time.

Opening-range breakout (ORB), both directions:
  - Range = high/low of minutes [start, start + or_len) (London time).
  - After the range, a buy-stop at range high + buffer and a sell-stop at range low - buffer.
    The first one touched is taken (one trade per day).
  - Stop-loss: stop_frac * range width from the entry (1.0 = the other side of the range).
  - Take-profit: target_r * stop distance (0 = none).
  - Anything still open is closed at the `exit` minute. Never held overnight -> no swap.

Fills are conservative: stop entries fill at the worse of the level and the bar open, plus
half-spread + slippage; if the stop and target could both be touched inside one minute
the stop is assumed first, and a stop touched on the entry minute counts as hit.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ORB:
    start: int = 8 * 60        # range start, minutes after London midnight
    or_len: int = 60           # range length in minutes
    exit: int = 16 * 60        # flat time
    buffer: float = 0.0        # extra USD beyond the range to trigger
    stop_frac: float = 1.0     # stop distance as a fraction of range width
    target_r: float = 0.0      # take-profit in R (0 = hold to exit time)
    min_w: float = 0.0         # skip if range width < min_w * ATR(daily)
    max_w: float = 9.9         # skip if range width > max_w * ATR(daily)
    long: bool = True
    short: bool = True
    fade: bool = False         # True: sell the break above the range / buy the break below (limit orders)
    trend: int = 0             # >0: breakout mode trades only with the daily SMA(trend) direction;
                               #     fade mode trades only against extensions opposite the trend
    spread: float = 0.35
    slippage: float = 0.10
    cost_bp: float = 0.0       # if > 0: cost per fill in basis points of price (overrides spread/slippage)


def load_m1(path):
    z = np.load(path)
    t = pd.to_datetime(z["t"], unit="m")
    return pd.DataFrame({"open": z["o"], "high": z["h"], "low": z["l"], "close": z["c"]}, index=t)


def split_days(m1):
    """Group minute bars by London calendar day -> list of (date, minute-of-day, o, h, l, c)."""
    day = m1.index.normalize()
    mod = (m1.index.hour * 60 + m1.index.minute).values
    o, h, l, c = (m1[k].values for k in ["open", "high", "low", "close"])
    bounds = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        d = day[a]
        if d.dayofweek < 5:
            out.append((d, mod[a:b], o[a:b], h[a:b], l[a:b], c[a:b]))
    return out


def daily_atr(days, n=14):
    rng = pd.Series({d: h.max() - l.min() for d, _, _, h, l, _ in days})
    return rng.rolling(n).mean().shift()  # yesterday's value: known before today opens


def run_orb(days, p: ORB, atr=None, trend=None):
    rows = []
    for d, mod, o, h, l, c in days:
        cost = o[0] * p.cost_bp / 1e4 if p.cost_bp else p.spread / 2 + p.slippage
        in_or = (mod >= p.start) & (mod < p.start + p.or_len)
        if in_or.sum() < p.or_len * 0.5:
            continue
        hi, lo = h[in_or].max(), l[in_or].min()
        w = hi - lo
        if w <= 0:
            continue
        if atr is not None:
            a = atr.get(d, np.nan)
            if not np.isfinite(a) or w < p.min_w * a or w > p.max_w * a:
                continue
        live = np.flatnonzero((mod >= p.start + p.or_len) & (mod < p.exit))
        if len(live) == 0:
            continue
        up, dn = hi + p.buffer, lo - p.buffer
        # which extension may be traded: in fade mode a break up gives a SELL
        allow_up = p.short if p.fade else p.long
        allow_dn = p.long if p.fade else p.short
        if p.trend and trend is not None:
            tu = trend.get(d, np.nan)
            if not np.isfinite(tu):
                continue
            if p.fade:   # fade only counter-trend spikes: sell spikes up in downtrend, buy dips in uptrend
                allow_up &= tu < 0
                allow_dn &= tu > 0
            else:
                allow_up &= tu > 0
                allow_dn &= tu < 0
        side = 0
        for j in live:
            hit_up = allow_up and h[j] >= up
            hit_dn = allow_dn and l[j] <= dn
            if hit_up and hit_dn:  # both in one minute: ambiguous -> skip the day
                break
            if hit_up:
                side, entry, k = (-1, up - cost, j) if p.fade else (1, max(up, o[j]) + cost, j)
                break
            if hit_dn:
                side, entry, k = (1, dn + cost, j) if p.fade else (-1, min(dn, o[j]) - cost, j)
                break
        if side == 0:
            continue
        sd = p.stop_frac * w
        stop = entry - side * sd
        tgt = entry + side * p.target_r * sd if p.target_r > 0 else np.nan
        exit_px, why = None, "time"
        after = live[live >= k]
        for j in after:
            if side > 0:
                if l[j] <= stop:
                    exit_px, why = (min(stop, o[j]) if j != k else stop), "stop"
                elif np.isfinite(tgt) and h[j] >= tgt and j != k:
                    exit_px, why = max(tgt, o[j]), "target"
            else:
                if h[j] >= stop:
                    exit_px, why = (max(stop, o[j]) if j != k else stop), "stop"
                elif np.isfinite(tgt) and l[j] <= tgt and j != k:
                    exit_px, why = min(tgt, o[j]), "target"
            if exit_px is not None:
                break
        if exit_px is None:
            exit_px = c[after[-1]]
        exit_px -= side * cost
        pnl = side * (exit_px - entry)
        rows.append((d, side, entry, exit_px, pnl, pnl / sd, pnl / entry, why))
    return pd.DataFrame(rows, columns=["date", "side", "entry", "exit", "pnl", "R", "ret", "reason"])


def daily_trend(days, n):
    """+1 / -1 if yesterday's session close is above / below its SMA(n); known before today."""
    close = pd.Series({d: c[-1] for d, _, _, _, _, c in days})
    return np.sign(close - close.rolling(n).mean()).shift()


def side_stats(t):
    def one(x):
        if len(x) == 0:
            return {"Trades": 0}
        g, b = x.R[x.R > 0].sum(), -x.R[x.R <= 0].sum()
        return {"Trades": len(x), "WinRate": (x.R > 0).mean(), "PF": g / b if b else np.inf,
                "AvgR": x.R.mean(), "Ret/trade(bp)": x.ret.mean() * 1e4}
    return {"All": one(t), "Buy": one(t[t.side > 0]), "Sell": one(t[t.side < 0])}
