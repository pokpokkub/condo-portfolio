"""Download Oanda XAU/USD 1-minute bars (2006-2020) and cache them for the intraday system.

    python tools/build_xau_m1.py        # -> data/xauusd_m1_london.npz (not committed, ~200MB)

Source: github.com/FutureSharks/financial-data (Oanda data, sparse git checkout).
Timestamps are converted from UTC to Europe/London local time (handles DST).
"""
import glob
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "xauusd_m1_london.npz"
SUB = "pyfinancialdata/data/currencies/oanda/XAU_USD"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "fd"
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/FutureSharks/financial-data.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", SUB], check=True)
        files = sorted(glob.glob(str(repo / SUB / "*" / "*.csv")))
        d = pd.concat(pd.read_csv(f, parse_dates=["time"]) for f in files)
    d = d.drop_duplicates("time").set_index("time").sort_index()
    ts = d.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)
    np.savez(OUT, t=ts.values.astype("datetime64[m]").astype(np.int64),
             o=d.open.values, h=d.high.values, l=d.low.values, c=d.close.values)
    print(f"wrote {OUT} ({len(d):,} bars)")


if __name__ == "__main__":
    main()
