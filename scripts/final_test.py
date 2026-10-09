"""ONE-SHOT holdout (2025-10-01 .. 2026-09-30) for both strategy families + benchmarks.
Alt-selection: each leverage uses the combo it would have picked at the holdout start
(best trailing 12 months). Majors: fixed rules. Also re-computes the WF-period benchmarks.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from farb import data as D
from farb.metrics import summary

sys.path.insert(0, str(D.ROOT / "scripts"))
from research import GRID, HOLDOUT_START, LEVS, REPORTS, SIM_END, WF_START  # noqa: E402

LOCK = REPORTS / "HOLDOUT_USED.lock"
TB = Path.home() / "trader-bot" / "reports"   # read-only


def best_combo(runs, until, months=12):
    t1 = pd.Timestamp(until, tz="UTC")
    t0 = t1 - pd.DateOffset(months=months)
    best, bv = None, -np.inf
    for r in runs:
        x = r["ret"][(r["ret"].index >= t0) & (r["ret"].index < t1)]
        v = (1 + x).prod() - 1 if len(x) > 30 else -np.inf
        if v > bv:
            best, bv = r, v
    return best


def benchmarks(start, end) -> dict[str, pd.Series]:
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    idx = pd.date_range(s, e, freq="D", tz="UTC")
    out = {}
    btc = D.load("spot", "BTCUSDT", "1h").set_index("ts")["close"].resample("D").last()
    out["btc_buy_hold"] = btc.pct_change().reindex(idx).fillna(0)
    y = pd.read_parquet(D.DATA / "stable_yields.parquet")
    out["tbill_3m"] = (y["tbill"].ffill().reindex(idx, method="ffill") / 100 / 365).fillna(0)
    a = y["aave_usdt"].ffill().reindex(idx, method="ffill") / 100 / 365
    if a.notna().mean() > 0.9:
        out["aave_usdt"] = a.fillna(0)
    tb = []
    for f in ("wf_equity.parquet", "holdout_equity.parquet"):
        p = TB / f
        if p.exists():
            df = pd.read_parquet(p)
            if "ensemble|regime" in df.columns:
                tb.append(df["ensemble|regime"])
    if tb:
        x = pd.concat(tb).sort_index()
        x = x[~x.index.duplicated()]
        x.index = x.index.normalize()
        x = x.reindex(idx)
        if x.notna().mean() > 0.8:
            out["trader_bot_system"] = x.fillna(0)
    return out


def main():
    if LOCK.exists():
        print("holdout already used:", LOCK.read_text())
        return 1
    sel = json.loads((REPORTS / "selected.json").read_text())
    LOCK.write_text(f"used {pd.Timestamp.now(tz='UTC').isoformat()} recommended={sel['family']}|{sel['recommended']}\n")
    hs = pd.Timestamp(HOLDOUT_START, tz="UTC")
    res = {"alt_selection": {}, "majors": {}, "benchmarks": {}, "wf_benchmarks": {}}
    runs = pickle.load(open(REPORTS / "runs_main.pkl", "rb"))
    for n, _ in LEVS:
        b = best_combo([r for r in runs if r["lev"] == n], HOLDOUT_START)
        x = b["ret"][b["ret"].index >= hs]
        s = summary(x)
        s["liquidations"] = sum(1 for t in b["liq_times"] if t >= hs)
        s["combo"] = GRID[b["g"]]
        res["alt_selection"][n] = s
    maj = pd.read_parquet(REPORTS / "majors_holdout_daily.parquet")
    mj = json.loads((REPORTS / "majors.json").read_text())
    for n in maj.columns:
        s = summary(maj[n].dropna())
        s["liquidations"] = mj[n]["base"].get("_holdout_liq")
        res["majors"][n] = s
    end = str(maj.index.max().date())
    for k, v in benchmarks(HOLDOUT_START, end).items():
        res["benchmarks"][k] = summary(v)
    for k, v in benchmarks(WF_START, "2025-09-30").items():
        res["wf_benchmarks"][k] = summary(v)
    (REPORTS / "holdout.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"HOLDOUT {HOLDOUT_START} .. {end}")
    print("%-22s %8s %8s %8s %5s" % ("", "total", "maxDD", "worstM", "liq"))
    for fam in ("majors", "alt_selection"):
        for n, s in res[fam].items():
            print("%-22s %7.2f%% %7.2f%% %7.2f%% %5s" % (f"{fam[:6]}|{n}", 100 * s["total"], 100 * s["max_dd"], 100 * s["worst_month"], s["liquidations"]))
    for k, s in res["benchmarks"].items():
        print("%-22s %7.2f%% %7.2f%% %7.2f%%" % (k, 100 * s["total"], 100 * s["max_dd"], 100 * s["worst_month"]))
    print("WF-period benchmarks (CAGR):", {k: round(100 * v["cagr"], 2) for k, v in res["wf_benchmarks"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
