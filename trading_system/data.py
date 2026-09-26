"""Load and prepare monthly total-return series for stocks, bonds, gold and cash.

Sources (all public, downloaded from github.com/datasets):
  - sp500_shiller_monthly.csv : Robert Shiller S&P 500 data (monthly average price,
    trailing dividends, 10-year Treasury yield), 1871 onwards.
  - gold_monthly.csv          : monthly gold price (USD/oz), 1833 onwards.
  - sp500_daily_1999_2018.csv : daily S&P 500 OHLC (from the `arch` package) used
    for a daily-close validation of the trend filter.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _bond_return(y_prev, y_now, maturity=10.0):
    """Monthly total return of a 10y par bond bought at yield y_prev and re-priced at y_now.

    Semiannual coupon = y_prev; after one month the bond has maturity - 1/12 years left.
    """
    c = y_prev / 2
    r = y_now / 2
    n = 2 * (maturity - 1 / 12)
    price = np.where(r > 0, c / r * (1 - (1 + r) ** (-n)) + (1 + r) ** (-n), 1.0)
    return price - 1 + y_prev / 12


def load_monthly():
    """Return a DataFrame of monthly total returns: stocks, bonds, gold, cash.

    Row t holds the return from month t-1 to month t. Also returns price-index
    levels (for trend signals) as a second DataFrame.
    """
    sp = pd.read_csv(DATA_DIR / "sp500_shiller_monthly.csv", parse_dates=["Date"])
    sp["Date"] = sp["Date"].dt.to_period("M")
    sp = sp.set_index("Date")

    # Shiller's file stops updating dividends / rates in 2023; carry the last
    # known dividend yield and 10y yield forward (documented approximation).
    dy = (sp["Dividend"] / sp["SP500"]).replace(0, np.nan).ffill()
    y10 = (sp["Long Interest Rate"] / 100).replace(0, np.nan).ffill()

    stock_ret = sp["SP500"].pct_change() + dy.shift(1) / 12
    bond_ret = pd.Series(_bond_return(y10.shift(1).values, y10.values), index=sp.index)

    gold = pd.read_csv(DATA_DIR / "gold_monthly.csv")
    gold["Date"] = pd.PeriodIndex(gold["Date"], freq="M")
    gold = gold.set_index("Date")["Price"]
    gold_ret = gold.pct_change()

    rets = pd.DataFrame({"stocks": stock_ret, "bonds": bond_ret, "gold": gold_ret})
    rets["cash"] = 0.0  # no T-bill series available -> conservative 0% on cash
    rets = rets.dropna(subset=["stocks", "bonds"])

    levels = (1 + rets.fillna(0)).cumprod()
    levels.loc[rets["gold"].isna(), "gold"] = np.nan
    return rets, levels


def load_sp500_daily():
    d = pd.read_csv(DATA_DIR / "sp500_daily_1999_2018.csv", parse_dates=["Date"], index_col="Date")
    return d["Adj Close"]
