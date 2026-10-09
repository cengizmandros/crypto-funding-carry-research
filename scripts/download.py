"""Download / update all data.  Stages:
 1. daily futures klines for every USDT perp ever listed (for point-in-time liquidity ranking)
 2. monthly universe: top 30 perps by trailing 30d futures volume that also have a spot pair
 3. for every symbol ever in the universe: funding, 1h futures, 1h spot, 5m mark price
 4. cross-exchange comparison: Bybit funding (paged API), OKX daily archive snapshots
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from farb import data as D

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(D.ROOT / "logs" / "download.log"), logging.StreamHandler()])
log = logging.getLogger("download")
TOP_N = 30


def universe(perps, spot):
    qv = {}
    for s in perps:
        df = D.load("fut", s, "1d")
        if len(df):
            qv[s] = df.set_index("ts")["quote_volume"]
    qv = pd.DataFrame(qv).sort_index()
    first = qv.notna().idxmax()
    rows = []
    for m in pd.date_range(pd.Timestamp(D.START, tz="UTC") + pd.offsets.MonthBegin(1),
                           qv.index.max() + pd.offsets.MonthBegin(1), freq="MS"):
        hist = qv.loc[: m - pd.Timedelta(seconds=1)]
        if len(hist) < 30:
            continue
        alive = hist.iloc[-1].notna()
        aged = (m - first) >= pd.Timedelta(days=30)
        has_spot = pd.Series([s in spot for s in qv.columns], index=qv.columns)
        score = hist.iloc[-30:].sum(min_count=15)[alive & aged & has_spot].dropna()
        for r, s in enumerate(score.sort_values(ascending=False).head(TOP_N).index, 1):
            rows.append((m, s, r))
    return pd.DataFrame(rows, columns=["month", "symbol", "rank"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="daily paper-trading refresh: only current universe")
    a = ap.parse_args()
    sym_file = D.DATA / "symbols.json"
    if not sym_file.exists() or not a.quick:
        perps, spot = D.perp_symbols(), sorted(D.spot_symbols())
        sym_file.write_text(json.dumps({"perps": perps, "spot": spot}))
    js = json.loads(sym_file.read_text())
    perps, spot = js["perps"], set(js["spot"])
    if not a.quick:
        log.info("stage1: %d perps daily klines", len(perps))
        r = D.run_many(D.update_klines, [("fut", s, "1d") for s in perps])
        log.info("stage1 done, failed=%s", [k for k, v in r.items() if v < 0][:10])
        u = universe(perps, spot)
        u.to_parquet(D.DATA / "universe.parquet", index=False)
    u = pd.read_parquet(D.DATA / "universe.parquet")
    cands = sorted(u["symbol"].unique()) if not a.quick else sorted(u[u["month"] == u["month"].max()]["symbol"])
    log.info("stage3: %d candidates", len(cands))
    jobs = [("funding", s) for s in cands]
    r = D.run_many(lambda kind, s: D.update_funding(s), jobs)
    jobs = [(m, s, iv) for s in cands for m, iv in (("fut", "1h"), ("spot", "1h"), ("mark", "5m"))]
    r2 = D.run_many(D.update_klines, jobs)
    bad = [k for k, v in {**r, **r2}.items() if v < 0]
    log.info("stage3 done, failed=%s", bad)
    if a.quick:
        return 0
    log.info("stage4: bybit + okx")
    top = list(u[u["month"] == u["month"].max()]["symbol"])[:20]
    for s in sorted(set(top) | {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"}):
        df = D.bybit_funding(s)
        if len(df):
            df.to_parquet(D.path("bybit_funding", s), index=False)
    days = pd.date_range(pd.Timestamp(D.START, tz="UTC"), pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2), freq="D")
    okx = D.run_many(lambda d: D.okx_daily(d), [(d,) for d in days], workers=12)
    okx = pd.concat([v for v in okx.values() if isinstance(v, pd.DataFrame)], ignore_index=True)
    okx.to_parquet(D.DATA / "okx_daily.parquet", index=False)
    log.info("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
