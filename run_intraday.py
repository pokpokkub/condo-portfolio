"""LORB — London Opening-Range Breakout on XAUUSD, flat every day, buys AND sells.

    python tools/build_xau_m1.py   # once: fetch 1-minute data
    python run_intraday.py         # writes reports/intraday_report.md + reports/intraday_equity.png

Protocol: ~150 intraday variants (ORB/fade, 07:00/08:00/13:30 London, stop, target, trend filter)
were screened on 2006-2014 only; the variant with the best *weaker side* PF was fixed and then
run once on 2015-2020-05 (out-of-sample).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_system.intraday import ORB, daily_atr, daily_trend, load_m1, run_orb, side_stats, split_days

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
# Final system: range 07:00-08:00 London, trade the first break until 20:00, stop = other side of
# range, no target, only in the direction of the daily SMA(20) trend.
SYSTEM = dict(start=7 * 60, or_len=60, exit=20 * 60, stop_frac=1.0, target_r=0.0, trend=20)
COST_BP = 0.5625  # per fill = (0.35 spread / 2 + 0.05 slippage) USD at a $4,000 gold price
RISK = 0.005      # 0.5% of equity risked per trade
MAX_LEV = 10.0    # notional cap (x equity) for very narrow ranges


def equity(t, risk=RISK, max_lev=MAX_LEV):
    """Compound trades: each risks `risk` of equity on the stop distance (capped by leverage)."""
    stop_pct = (t.pnl / t.R).abs() / t.entry  # stop distance as a fraction of price
    lev = np.minimum(risk / stop_pct, max_lev)
    r = pd.Series((lev * t.ret).values, index=pd.to_datetime(t.date))
    return (1 + r).cumprod(), r


def perf(eq, r):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": f"{cagr:.1%}", "MaxDD": f"{dd:.1%}", "CAGR/MaxDD": f"{cagr / -dd:.2f}"}


def fmt_sides(s):
    row = {}
    for side in ["All", "Buy", "Sell"]:
        row[f"{side} trades"] = s[side]["Trades"]
        row[f"{side} PF"] = f"{s[side]['PF']:.2f}"
        row[f"{side} win%"] = f"{s[side]['WinRate']:.0%}"
    row["Avg R"] = f"{s['All']['AvgR']:.3f}"
    return row


def main():
    OUT.mkdir(exist_ok=True)
    days = split_days(load_m1(ROOT / "data" / "xauusd_m1_london.npz"))
    atr = daily_atr(days)
    trends = {n: daily_trend(days, n) for n in [0, 20, 50, 200] if n}
    IS = [x for x in days if x[0].year <= 2014]
    OOS = [x for x in days if x[0].year > 2014]
    lines = ["# LORB — London Opening-Range Breakout (XAUUSD, intraday, buy & sell)\n",
             "Oanda XAU/USD 1-minute bars 2006-03 → 2020-05, London time. Costs per fill "
             f"{COST_BP} bp of price (≈ $0.35 spread + $0.05 slippage at $4,000 gold). "
             "No overnight positions → no swap.\n"]

    def emit(title, df):
        print(f"\n=== {title} ===\n{df.to_string()}")
        lines.extend([f"\n## {title}\n", df.to_markdown(), ""])

    def trades(ds, **kw):
        p = ORB(**{**SYSTEM, "cost_bp": COST_BP, **kw})
        return run_orb(ds, p, atr, trends.get(p.trend))

    t_is, t_oos, t_all = trades(IS), trades(OOS), trades(days)
    main_tab = {}
    for name, t in [("In-sample 2006-2014", t_is), ("Out-of-sample 2015-2020", t_oos), ("Full 2006-2020", t_all)]:
        eq, r = equity(t)
        main_tab[name] = {**fmt_sides(side_stats(t)), **perf(eq, r)}
    emit("Main result (0.5% risk per trade)", pd.DataFrame(main_tab).T)

    rob = {}
    for label, kw in [("system", {}), ("range from 08:00", {"start": 8 * 60}), ("no trend filter", {"trend": 0}),
                      ("trend SMA50", {"trend": 50}), ("trend SMA200", {"trend": 200}),
                      ("exit 16:00", {"exit": 16 * 60}), ("range 45 min", {"or_len": 45}),
                      ("range 90 min", {"or_len": 90})]:
        row = {}
        for per, ds in [("IS", IS), ("OOS", OOS)]:
            s = side_stats(trades(ds, **kw))
            row[f"{per} Buy PF"] = f"{s['Buy']['PF']:.2f}"
            row[f"{per} Sell PF"] = f"{s['Sell']['PF']:.2f}"
        rob[label] = row
    emit("Robustness: neighbouring settings (PF per side)", pd.DataFrame(rob).T)

    cost = {}
    for bp in [0.25, 0.5625, 0.75, 1.0, 1.5]:
        s = side_stats(trades(OOS, cost_bp=bp))
        cost[f"{bp} bp/fill (≈ ${bp * 4000 / 1e4 * 2:.2f} round trip at $4,000)"] = {
            "OOS Buy PF": f"{s['Buy']['PF']:.2f}", "OOS Sell PF": f"{s['Sell']['PF']:.2f}",
            "OOS Avg R": f"{s['All']['AvgR']:.3f}"}
    emit("Cost stress (out-of-sample)", pd.DataFrame(cost).T)

    t_all = t_all.assign(year=pd.to_datetime(t_all.date).dt.year)
    eq, r = equity(t_all)
    yearly = pd.DataFrame({
        "Buy PF": t_all[t_all.side > 0].groupby("year").R.apply(lambda x: x[x > 0].sum() / -x[x < 0].sum()),
        "Sell PF": t_all[t_all.side < 0].groupby("year").R.apply(lambda x: x[x > 0].sum() / -x[x < 0].sum()),
        "Return": r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)})
    yearly = yearly.assign(**{"Buy PF": yearly["Buy PF"].map("{:.2f}".format),
                              "Sell PF": yearly["Sell PF"].map("{:.2f}".format),
                              "Return": yearly["Return"].map("{:.1%}".format)})
    emit("Calendar years (0.5% risk)", yearly)

    size = {}
    for rk in [0.0025, 0.005, 0.01]:
        size[f"risk {rk:.2%}/trade"] = perf(*equity(t_all, risk=rk))
    emit("Risk per trade vs return / drawdown (full period)", pd.DataFrame(size).T)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, sub in [("Buy trades only", t_all[t_all.side > 0]), ("Sell trades only", t_all[t_all.side < 0]),
                      ("LORB (both)", t_all)]:
        e, _ = equity(sub)
        ax[0].plot(e.index, e, label=name, lw=2 if "both" in name else 1)
        ax[1].plot(e.index, e / e.cummax() - 1, lw=2 if "both" in name else 1)
    for x in ax:
        x.axvline(pd.Timestamp("2015-01-01"), color="grey", ls="--", lw=1)
        x.grid(alpha=.3)
    ax[0].set_title("LORB on XAUUSD — growth of 1 at 0.5% risk per trade (dashed line: out-of-sample starts)")
    ax[0].legend(); ax[1].set_title("Drawdown")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout(); fig.savefig(OUT / "intraday_equity.png", dpi=110)
    lines.append("\n![equity](intraday_equity.png)\n")
    (OUT / "intraday_report.md").write_text("\n".join(lines))
    print("\nWrote reports/intraday_report.md")


if __name__ == "__main__":
    main()
