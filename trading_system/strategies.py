"""Strategy weight generators. Each returns a DataFrame of target weights per month."""
import numpy as np
import pandas as pd

GOLD_FLOAT = pd.Period("1971-08", "M")  # gold price was fixed before Bretton Woods ended


def _frame(levels, cols):
    return pd.DataFrame(0.0, index=levels.index, columns=cols)


def buy_hold(levels, mix):
    w = _frame(levels, list(mix))
    for k, v in mix.items():
        w[k] = v
    return w


def in_trend(levels, asset, sma):
    lv = levels[asset]
    sig = lv > lv.rolling(sma).mean()
    if asset == "gold":
        sig &= levels.index >= GOLD_FLOAT + sma
    return sig.fillna(False)


def stock_timing(levels, sma=10):
    """Hold stocks when above their SMA, otherwise 10y bonds."""
    w = _frame(levels, ["stocks", "bonds"])
    s = in_trend(levels, "stocks", sma)
    w["stocks"] = s.astype(float)
    w["bonds"] = 1 - w["stocks"]
    return w


def tfmr(levels, sma=10, assets=("stocks", "gold", "bonds"), vol_window=12, target_vol=None, rets=None,
         fallback=None):
    """Trend-Filtered Multi-asset Rotation.

    Each asset has a base weight (equal, or inverse-volatility when vol_window is set).
    An asset keeps its slice only while its total-return index is above its SMA;
    otherwise that slice goes to `fallback` (if in trend) or cash. Optional target_vol scales gross exposure
    (capped at 100%, no leverage).
    """
    cols = list(assets) + ["cash"]
    w = _frame(levels, cols)
    avail = pd.DataFrame({a: levels[a].notna() & (levels.index >= (GOLD_FLOAT if a == "gold" else levels.index[0]))
                          for a in assets})
    if vol_window and rets is not None:
        vol = rets[list(assets)].rolling(vol_window, min_periods=vol_window).std()
        base = (1 / vol).where(avail)
    else:
        base = avail.astype(float).where(avail)
    base = base.div(base.sum(axis=1), axis=0).fillna(0.0)
    for a in assets:
        w[a] = base[a] * in_trend(levels, a, sma).astype(float)
    if fallback:
        # Slices of risk assets that are out of trend move to the fallback asset
        # (e.g. bonds) while it is itself in trend; otherwise they stay in cash.
        risk = [a for a in assets if a != fallback]
        idle = sum(base[a] - w[a] for a in risk)
        w[fallback] += idle * in_trend(levels, fallback, sma).astype(float)

    if target_vol and rets is not None:
        cov_ret = (rets[list(assets)].fillna(0.0) * w[list(assets)].shift(0)).sum(axis=1)
        realised = cov_ret.rolling(vol_window or 12).std() * np.sqrt(12)
        scale = (target_vol / realised).clip(upper=1.0).fillna(1.0)
        for a in assets:
            w[a] *= scale
    w["cash"] = 1 - w[list(assets)].sum(axis=1)
    return w
