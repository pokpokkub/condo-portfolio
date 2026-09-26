"""Short-term daily XAUUSD strategy candidates. Each returns engine.Signals."""
import numpy as np
import pandas as pd

from .gold_engine import Signals, atr


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def breakout(df, n=20, stop_atr=2.0, trail_atr=3.0, trend=200, shorts=True):
    """Donchian breakout in the direction of the long-term trend, ATR stop + ATR trailing stop."""
    a = atr(df)
    c = df["close"]
    hi = df["high"].rolling(n).max().shift()
    lo = df["low"].rolling(n).min().shift()
    up = c > c.rolling(trend).mean() if trend else pd.Series(True, index=df.index)
    long_ = (c > hi) & up
    short = (c < lo) & ~up if shorts else pd.Series(False, index=df.index)
    entry = long_.astype(int) - short.astype(int)
    return Signals(entry=entry, stop_dist=stop_atr * a, trail_mult=trail_atr * a)


def pullback(df, rsi_n=2, lo=10, hi=90, trend=200, exit_sma=5, stop_atr=3.0, max_bars=7, shorts=True):
    """Buy short-term oversold dips in an uptrend (and sell overbought rallies in a downtrend).

    Exit on the close crossing the short SMA, a time stop, or an ATR stop.
    """
    a = atr(df)
    c = df["close"]
    r = rsi(c, rsi_n)
    up = c > c.rolling(trend).mean()
    long_ = (r < lo) & up
    short = (r > hi) & ~up if shorts else pd.Series(False, index=df.index)
    ma = c.rolling(exit_sma).mean()
    return Signals(entry=long_.astype(int) - short.astype(int), stop_dist=stop_atr * a,
                   exit_long=c > ma, exit_short=c < ma, max_bars=max_bars)


def vol_breakout_daytrade(df, k=0.7, trend=200, spread=0.35, slippage=0.10):
    """Intraday volatility breakout, long only, flat every night (no swap).

    Buy-stop at today's open + k * yesterday's range, only when yesterday's close is
    above its SMA(trend). Exit at the day's close. Returns per-day return on notional
    (0 on days without a trade). A gap through the level fills at the open.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).shift()
    up = (c > c.rolling(trend).mean()).shift().fillna(False).astype(bool) if trend else pd.Series(True, index=df.index)
    level = o + k * rng
    hit = (h >= level) & up
    fill = np.maximum(level, o) + spread / 2 + slippage
    exit_ = c - spread / 2 - slippage
    r = pd.Series(0.0, index=df.index)
    r[hit] = ((exit_ - fill) / fill)[hit]
    return r
