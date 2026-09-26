"""Run the full research report for the TFMR trading system.

    python run_backtest.py            # prints tables, writes reports/ (markdown + chart)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_system import backtest as B
from trading_system import strategies as S
from trading_system.data import load_monthly, load_sp500_daily

OUT = Path(__file__).resolve().parent / "reports"
SMA = 10  # fixed a priori (classic 10-month trend filter); NOT optimised on this data


def section(title, df, lines):
    txt = df.to_markdown() if hasattr(df, "to_markdown") else str(df)
    print(f"\n=== {title} ===\n{df.to_string()}")
    lines += [f"\n## {title}\n", txt, ""]


def evaluate(specs, rets, start, end=None, cost=0.001):
    start = pd.Period(start, "M")
    res, curves = {}, {}
    for name, w in specs.items():
        p, ww = B.run(w, rets, cost=cost)
        p, ww = p.loc[start:end], ww.loc[start:end]
        res[name] = B.metrics(p, ww, rets)
        curves[name] = (1 + p).cumprod()
    return B.table(res).drop(columns=["Years"]), pd.DataFrame(curves)


def main():
    OUT.mkdir(exist_ok=True)
    rets, lv = load_monthly()
    lines = ["# TFMR — Trend-Filtered Multi-asset Rotation: backtest report\n"]
    system = S.tfmr(lv, SMA, vol_window=None, fallback="bonds")

    specs = {
        "Buy&Hold S&P500": S.buy_hold(lv, {"stocks": 1}),
        "Buy&Hold 60/40": S.buy_hold(lv, {"stocks": .6, "bonds": .4}),
        "Buy&Hold 1/3 each": S.buy_hold(lv, {"stocks": 1 / 3, "bonds": 1 / 3, "gold": 1 / 3}),
        "Stock timing SMA10": S.stock_timing(lv, SMA),
        "TFMR (system)": system,
    }
    tab, curves = evaluate(specs, rets, "1973-01")
    section("1. Main test 1973-01 → 2026-08 (cost 0.1%/side)", tab, lines)

    # 2. Parameter robustness: the edge should hold on a plateau, not one lucky value.
    rob = {f"SMA {n}": S.tfmr(lv, n, vol_window=None, fallback="bonds") for n in [6, 8, 9, 10, 11, 12, 14, 16]}
    section("2. Parameter robustness (SMA months)", evaluate(rob, rets, "1973-01")[0], lines)

    # 3. Out-of-sample history before gold floated: stocks/bonds only, 1881-1972.
    oos = {"Buy&Hold S&P500": S.buy_hold(lv, {"stocks": 1}),
           "Buy&Hold 60/40": S.buy_hold(lv, {"stocks": .6, "bonds": .4}),
           "TFMR stocks+bonds": S.tfmr(lv, SMA, assets=("stocks", "bonds"), vol_window=None, fallback="bonds")}
    section("3. Out-of-sample 1881-01 → 1972-12 (no gold)", evaluate(oos, rets, "1881-01", "1972-12")[0], lines)

    # 4. Sub-periods.
    sub = {}
    for a, b in [("1973", "1982"), ("1983", "1992"), ("1993", "2002"), ("2003", "2012"), ("2013", "2026")]:
        p, _ = B.run(system, rets)
        p = p.loc[pd.Period(a + "-01", "M"):pd.Period(b + "-12", "M")]
        bh = rets["stocks"].loc[p.index]
        m, mb = B.metrics(p), B.metrics(bh)
        sub[f"{a}-{b}"] = {"TFMR CAGR": m["CAGR"], "TFMR MaxDD": m["MaxDD"], "TFMR PF": m["PF(monthly)"],
                           "S&P CAGR": mb["CAGR"], "S&P MaxDD": mb["MaxDD"]}
    sub = pd.DataFrame(sub).T
    sub_fmt = sub.copy().astype(object)
    for c in sub:
        sub_fmt[c] = sub[c].map(("{:.2f}" if "PF" in c else "{:.1%}").format)
    section("4. Sub-periods (TFMR vs S&P500)", sub_fmt, lines)

    # 5. Cost stress.
    cost = {f"cost {c:.2%}": evaluate({"x": system}, rets, "1973-01", cost=c)[0].iloc[0] for c in [0, .001, .0025, .005, .01]}
    section("5. Transaction-cost stress", pd.DataFrame(cost).T, lines)

    # 6. Daily-close validation, 1999-2018: real closes instead of monthly averages.
    #    Same rule as the system: check at each month-end close vs the 10 month-end SMA,
    #    execute on the next day's close. Also shown: a daily SMA200 variant.
    px = load_sp500_daily()
    r = px.pct_change().fillna(0)
    me = px.groupby(px.index.to_period("M")).tail(1)
    me_sig = (me > me.rolling(SMA).mean()).astype(float)
    monthly_sig = me_sig.reindex(px.index).ffill().shift(1).fillna(0)
    sma200_sig = (px > px.rolling(200).mean()).astype(float).shift(1).fillna(0)
    daily = {}
    start = px.index[220]
    for name, sig in [("S&P500 buy&hold (price only)", pd.Series(1.0, index=px.index)),
                      ("Month-end > SMA10m, else cash", monthly_sig),
                      ("Daily > SMA200, else cash", sma200_sig)]:
        s = (sig * r - sig.diff().abs().fillna(0) * 0.001).loc[start:]
        eq = (1 + s).cumprod()
        yrs = len(s) / 252
        daily[name] = {"CAGR": f"{eq.iloc[-1] ** (1 / yrs) - 1:.1%}",
                       "MaxDD": f"{(eq / eq.cummax() - 1).min():.1%}",
                       "PF(daily)": f"{s[s > 0].sum() / -s[s < 0].sum():.2f}",
                       "Switches": int(sig.diff().abs().loc[start:].sum())}
    section("6. Daily-close validation 1999-2018 (S&P500 only, cash 0%, next-day execution)",
            pd.DataFrame(daily).T, lines)

    # 7. Current signal.
    last = lv.index[-1]
    trend = {a: bool(S.in_trend(lv, a, SMA).iloc[-1]) for a in ["stocks", "gold", "bonds"]}
    cur = system.iloc[-1]
    sig_df = pd.DataFrame({"in trend": trend, "target weight": {a: f"{cur[a]:.0%}" for a in trend}})
    section(f"7. Latest signal (data through {last})", sig_df, lines)

    # Chart.
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for c in ["Buy&Hold S&P500", "Buy&Hold 60/40", "TFMR (system)"]:
        eq = curves[c]
        x = eq.index.to_timestamp()
        ax[0].plot(x, eq, label=c, lw=1.8 if "TFMR" in c else 1.1)
        ax[1].plot(x, eq / eq.cummax() - 1, lw=1.8 if "TFMR" in c else 1.1)
    ax[0].set_yscale("log"); ax[0].set_title("Growth of $1 (log), 1973-2026"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_title("Drawdown"); ax[1].grid(alpha=.3)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown.png", dpi=110)

    lines.append("\n![equity](equity_drawdown.png)\n")
    (OUT / "backtest_report.md").write_text("\n".join(lines))
    print(f"\nWrote {OUT / 'backtest_report.md'} and equity_drawdown.png")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
