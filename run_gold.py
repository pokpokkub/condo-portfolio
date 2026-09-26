"""XAUUSD short-term daily system (GDB): research report.

    python run_gold.py     # writes reports/gold_report.md + reports/gold_equity.png

Design protocol: parameters were chosen on 2006-2014 only (in-sample). 2015-2020-05
was held out and is reported as out-of-sample (OOS).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_system import gold_strategies as G
from trading_system.gold_engine import Costs, backtest, load_d1

OUT = Path(__file__).resolve().parent / "reports"
PERIODS = {"In-sample 2006-2014": ("2006", "2014"), "Out-of-sample 2015-2020": ("2015", "2020"),
           "Full 2006-2020": ("2006", "2020")}

# Final parameters (fixed from the in-sample study)
VB_K, TREND = 0.7, 200
BO_N, BO_STOP, BO_TRAIL = 20, 2.0, 2.0


def module_returns(df, spread=0.35, slippage=0.10, vb_lev=1.0, bo_risk=0.01):
    """Daily returns of module A (day trade) and module B (swing), as fractions of equity."""
    a = G.vol_breakout_daytrade(df, VB_K, TREND, spread, slippage) * vb_lev
    res = backtest(df, G.breakout(df, BO_N, BO_STOP, BO_TRAIL, trend=TREND, shorts=False), risk=bo_risk,
                   costs=Costs(spread=spread, slippage=slippage))
    b = res.equity.pct_change().fillna(0.0)
    return a, b, res.trades


def summarize(r, trade_pnls):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    t = np.asarray(trade_pnls)
    return {"CAGR": cagr, "MaxDD": dd, "CAGR/MaxDD": cagr / -dd if dd < 0 else np.nan,
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
            "PF": t[t > 0].sum() / -t[t < 0].sum(), "Trades": len(t), "WinRate": (t > 0).mean(),
            "Worst month": r.groupby(r.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1).min()}


def fmt(rows):
    df = pd.DataFrame(rows).T
    out = df.astype(object)
    for c in df:
        f = "{:.0f}" if c == "Trades" else "{:.2f}" if c in ("CAGR/MaxDD", "Sharpe", "PF") else "{:.1%}"
        out[c] = df[c].map(f.format)
    return out


def evaluate(df, a0, b0, **kw):
    a, b, tr = module_returns(df, **kw)
    out = {}
    for name, (s, e) in PERIODS.items():
        aa, bb = a.loc[s:e], b.loc[s:e]
        bt = tr[(tr.entry_date >= s) & (tr.entry_date <= e + "-12-31")]
        # trade P&L as a fraction of equity at entry, so both modules are comparable
        eq_b = (1 + b).cumprod()
        b_tr = [p / (eq_b.asof(d) * 10_000) for p, d in zip(bt.pnl, bt.entry_date)]
        out[name] = {"A": summarize(aa, aa[aa != 0]), "B": summarize(bb, b_tr),
                     "A+B": summarize(aa + bb, list(aa[aa != 0]) + b_tr)}
    return out, a, b


def main():
    OUT.mkdir(exist_ok=True)
    df = load_d1()
    lines = ["# GDB — Gold Daily Breakout system (XAUUSD CFD): backtest report\n",
             "Data: Oanda XAU/USD 1-minute bars aggregated to daily (NY 17:00 close), 2006-03 → 2020-05.",
             "Costs: spread 0.35 USD/oz + slippage 0.10 USD/oz per fill; swap long -4%/yr, triple on Wednesday.",
             "Parameters chosen on 2006-2014 only; 2015-2020 is untouched out-of-sample.\n"]

    def emit(title, table):
        print(f"\n=== {title} ===\n{table.to_string()}")
        lines.extend([f"\n## {title}\n", table.to_markdown(), ""])

    res, a, b = evaluate(df, None, None)
    for name in PERIODS:
        rows = {"A: vol-breakout day trade (1x)": res[name]["A"], "B: Donchian-20 swing (1% risk)": res[name]["B"],
                "GDB = A + B": res[name]["A+B"]}
        bh = df.close.loc[PERIODS[name][0]:PERIODS[name][1]].pct_change().fillna(0)
        rows["Buy & hold gold (no swap)"] = summarize(bh, bh[bh != 0])
        emit(f"{name}", fmt(rows))

    # Parameter plateau for module A (PF by k and period)
    plateau = {}
    for k in [0.5, 0.6, 0.7, 0.8, 1.0]:
        r = G.vol_breakout_daytrade(df, k, TREND)
        for name, (s, e) in list(PERIODS.items())[:2]:
            t = r.loc[s:e]; t = t[t != 0]
            plateau.setdefault(f"k={k}", {})[name] = f"{t[t > 0].sum() / -t[t < 0].sum():.2f}"
    emit("Module A robustness: PF for different k", pd.DataFrame(plateau).T)

    plateau = {}
    for n in [10, 20, 40, 55]:
        for trail in [2.0, 3.0]:
            res_b = backtest(df, G.breakout(df, n, BO_STOP, trail, trend=TREND, shorts=False))
            t = res_b.trades
            for name, (s, e) in list(PERIODS.items())[:2]:
                x = t[(t.entry_date >= s) & (t.entry_date <= e + "-12-31")].pnl
                plateau.setdefault(f"N={n}, trail={trail}ATR", {})[name] = f"{x[x > 0].sum() / -x[x < 0].sum():.2f}"
    emit("Module B robustness: PF for different channel / trail", pd.DataFrame(plateau).T)

    # Cost stress on the combined system, OOS
    cost_rows = {}
    for spread in [0.20, 0.35, 0.50, 0.80]:
        r, _, _ = evaluate(df, None, None, spread=spread)
        cost_rows[f"spread {spread:.2f}"] = r["Out-of-sample 2015-2020"]["A+B"]
    emit("Cost stress — GDB out-of-sample", fmt(cost_rows))

    # Leverage / risk scaling (full period)
    lev_rows = {}
    for mult in [1, 2, 3, 4]:
        r, _, _ = evaluate(df, None, None, vb_lev=1.0 * mult, bo_risk=0.01 * mult)
        lev_rows[f"A {mult}x notional + B {mult}% risk"] = r["Full 2006-2020"]["A+B"]
    emit("Position-size scaling (full period)", fmt(lev_rows))

    # Yearly
    ab = a + b
    yearly = pd.DataFrame({"A": a.groupby(a.index.year).apply(lambda x: (1 + x).prod() - 1),
                           "B": b.groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1),
                           "GDB": ab.groupby(ab.index.year).apply(lambda x: (1 + x).prod() - 1),
                           "Gold": df.close.groupby(df.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)})
    emit("Calendar-year returns", yearly.map("{:.1%}".format))

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, r in [("A: day trade", a), ("B: swing", b), ("GDB (A+B)", ab)]:
        eq = (1 + r.loc["2006-03-20":]).cumprod()
        ax[0].plot(eq.index, eq, label=name, lw=2 if "GDB" in name else 1)
        ax[1].plot(eq.index, eq / eq.cummax() - 1, lw=2 if "GDB" in name else 1)
    for x in ax:
        x.axvline(pd.Timestamp("2015-01-01"), color="grey", ls="--", lw=1)
        x.grid(alpha=.3)
    ax[0].text(pd.Timestamp("2015-02-01"), ax[0].get_ylim()[1] * 0.97, "out-of-sample →", va="top", color="grey")
    ax[0].set_title("GDB on XAUUSD — growth of 1 (A 1x notional, B 1% risk per trade)"); ax[0].legend()
    ax[1].set_title("Drawdown"); ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout(); fig.savefig(OUT / "gold_equity.png", dpi=110)
    lines.append("\n![equity](gold_equity.png)\n")
    (OUT / "gold_report.md").write_text("\n".join(lines))
    print("\nWrote reports/gold_report.md")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
