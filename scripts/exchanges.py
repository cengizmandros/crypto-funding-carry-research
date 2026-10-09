"""Cross-exchange funding comparison (Binance vs Bybit vs OKX) and market-wide funding regime."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from farb import data as D

REPORTS = D.ROOT / "reports"


def ann_daily(df: pd.DataFrame) -> pd.Series:
    """Sum of funding per UTC day * 365 = annualised carry for a 1x short."""
    return df.set_index("ts")["rate"].resample("D").sum() * 365


def main():
    out = {"yearly": {}, "monthly": {}}
    syms = sorted(p.stem for p in (D.DATA / "bybit_funding").glob("*.parquet"))
    okx = pd.read_parquet(D.DATA / "okx_daily.parquet") if (D.DATA / "okx_daily.parquet").exists() else pd.DataFrame()
    rows = []
    for s in syms:
        b = D.load("funding", s)
        y = pd.read_parquet(D.DATA / "bybit_funding" / f"{s}.parquet")
        if not len(b):
            continue
        bn, by = ann_daily(b), ann_daily(y)
        inst = s[:-4] + "-USDT-SWAP"
        ok = okx[okx["inst"] == inst].set_index("date")["rate"] if len(okx) else pd.Series(dtype=float)
        # OKX archive is a daily snapshot of the current period's realised rate -> x3 periods/day
        ok = ok.groupby(level=0).mean() * 3 * 365
        df = pd.DataFrame({"binance": bn, "bybit": by, "okx": ok})
        for yr, g in df.groupby(df.index.year):
            rows.append((s, yr, *[float(g[c].mean()) if g[c].notna().sum() > 30 else np.nan for c in df.columns],
                         float((g["binance"] < 0).mean())))
    t = pd.DataFrame(rows, columns=["symbol", "year", "binance", "bybit", "okx", "binance_neg_day_share"])
    t.to_csv(REPORTS / "exchanges_yearly.csv", index=False)
    piv = t.groupby("year")[["binance", "bybit", "okx", "binance_neg_day_share"]].median()
    out["yearly_median_across_coins"] = piv.round(4).to_dict()
    btc = t[t.symbol == "BTCUSDT"].set_index("year")[["binance", "bybit", "okx"]]
    out["btc"] = btc.round(4).to_dict()
    # market-wide: universe-median funding on Binance, by month
    u = pd.read_parquet(D.DATA / "universe.parquet")
    med = {}
    for m, g in u.groupby("month"):
        vals = []
        for s in g["symbol"]:
            f = D.load("funding", s)
            if len(f):
                x = f[(f["ts"] >= m) & (f["ts"] < m + pd.offsets.MonthBegin(1))]
                if len(x):
                    vals.append(x["rate"].sum() * 365 / max((x["ts"].max() - x["ts"].min()).days + 1, 1))
        if vals:
            med[str(m.date())[:7]] = {"median": float(np.median(vals)), "p75": float(np.quantile(vals, 0.75)),
                                      "share_neg": float(np.mean(np.array(vals) < 0))}
    out["universe_monthly"] = med
    (REPORTS / "exchanges.json").write_text(json.dumps(out, indent=1, default=str))
    print(piv.round(4).to_string())
    print(btc.round(4).to_string())


if __name__ == "__main__":
    main()
