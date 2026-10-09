"""Historical data for funding-rate carry research. Public sources only, no keys.

Binance (primary): data.binance.vision archive (incl. delisted perps) + public REST top-up
  - futures (USD-M) funding rates, 1d/1h futures klines, 5m mark-price klines
  - spot 1h klines for the cash leg
Bybit: public v5 funding history (paged).   OKX: daily archive snapshots (static.okx.com).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(os.environ.get("FA_HOME", Path.home() / "funding-arb"))
DATA = ROOT / "data"
START = "2021-01-01"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE = "https://data.binance.vision"
log = logging.getLogger(__name__)
NO_REST = os.environ.get("FA_NO_REST") == "1"   # research downloads: archive (CDN) only

_s = requests.Session()
_s.headers["User-Agent"] = "funding-arb-research/1.0"
_s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
STABLE = {"USDC", "BUSD", "TUSD", "USDP", "FDUSD", "DAI", "USDE", "USD1", "BFUSD", "XUSD", "RLUSD", "PYUSD", "EUR"}

# market -> (archive prefix, REST kline url)
MARKETS = {
    "spot": ("data/spot/monthly/klines", "https://api.binance.com/api/v3/klines"),
    "fut": ("data/futures/um/monthly/klines", "https://fapi.binance.com/fapi/v1/klines"),
    "mark": ("data/futures/um/monthly/markPriceKlines", "https://fapi.binance.com/fapi/v1/markPriceKlines"),
}
IV_MS = {"5m": 300_000, "1h": 3_600_000, "1d": 86_400_000}


import threading

# Per-host throttle. Binance REST weight limits are per IP: fapi 2400/min (klines with
# limit>=1000 cost 10), fundingRate shares 500 req / 5 min. Exceeding them -> 429, then a
# 418 IP ban. The static archive (data.binance.vision) is a CDN and is not throttled.
_MIN_GAP = {"fapi.binance.com": 0.35, "api.binance.com": 0.12, "api.bybit.com": 0.1}
_lock = threading.Lock()
_next_ok: dict[str, float] = {}


def _throttle(url):
    host = url.split("/")[2]
    if url.endswith("/fundingRate"):
        host, gap = "fapi-funding", 0.7      # separate 500 req / 5 min bucket
    else:
        gap = _MIN_GAP.get(host)
    if gap is None:
        return
    with _lock:
        t = time.time()
        wait = _next_ok.get(host, 0) - t
        _next_ok[host] = max(t, _next_ok.get(host, 0)) + gap
    if wait > 0:
        time.sleep(wait)


def get(url, params=None, tries=6):
    host = url.split("/")[2]
    for i in range(tries):
        _throttle(url)
        try:
            r = _s.get(url, params=params, timeout=30)
            if r.status_code in (418, 429):
                pause = float(r.headers.get("retry-after", 60))
                with _lock:   # pause every thread hitting this host
                    _next_ok[host] = max(_next_ok.get(host, 0), time.time() + pause + 1)
                log.warning("%s %s: backing off %.0fs", host, r.status_code, pause)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** i)
                continue
            return r
        except requests.RequestException:
            time.sleep(2 ** i)
    raise RuntimeError(url)


def s3_list(prefix, delimiter="/"):
    pre, keys, marker = [], [], ""
    while True:
        p = {"prefix": prefix, "marker": marker}
        if delimiter:
            p["delimiter"] = delimiter
        t = get(S3, p).text
        pp = re.findall(r"<Prefix>([^<]*)</Prefix>", t)[1:]
        kk = re.findall(r"<Key>([^<]*)</Key>", t)
        pre += pp
        keys += kk
        if "<IsTruncated>true" not in t:
            return pre, keys
        nm = re.search(r"<NextMarker>([^<]*)</NextMarker>", t)
        marker = nm.group(1) if nm else (kk[-1] if kk else pp[-1])


def perp_symbols() -> list[str]:
    pre, _ = s3_list("data/futures/um/monthly/fundingRate/")
    out = []
    for p in pre:
        s = p.rstrip("/").split("/")[-1]
        if s.endswith("USDT") and "_" not in s and s[:-4] not in STABLE:
            out.append(s)
    return sorted(out)


def spot_symbols() -> set[str]:
    pre, _ = s3_list("data/spot/monthly/klines/")
    return {p.rstrip("/").split("/")[-1] for p in pre}


def _zip_csv(key):
    r = get(f"{ARCHIVE}/{key}")
    if r.status_code != 200:
        return None
    c = get(f"{ARCHIVE}/{key}.CHECKSUM")
    if c.status_code == 200 and hashlib.sha256(r.content).hexdigest() != c.text.split()[0]:
        raise ValueError(f"checksum {key}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.open(z.namelist()[0]).read()
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if isinstance(df.iloc[0, 0], str) and not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    return df


def _norm_ts(v):
    v = pd.to_numeric(v).astype("int64")
    return v.where(v < 10 ** 14, v // 1000)


def _parse_klines(df):
    df = df.iloc[:, :12].copy()
    df.columns = KLINE_COLS
    df["open_time"] = _norm_ts(df["open_time"])
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["ts", "open", "high", "low", "close", "volume", "quote_volume"]]


def path(kind, sym, iv=None):
    d = DATA / (f"{kind}_{iv}" if iv else kind)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sym}.parquet"


def load(kind, sym, iv=None) -> pd.DataFrame:
    p = path(kind, sym, iv)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def update_klines(market: str, sym: str, iv: str, start=START) -> int:
    prefix, rest = MARKETS[market]
    old = load(market, sym, iv)
    if iv == "1d" and not len(old) and not NO_REST:
        # one REST call covers ~4 years of daily bars for listed symbols; archive only if delisted
        r = get(rest, {"symbol": sym, "interval": iv, "startTime": int(pd.Timestamp(start, tz="UTC").timestamp() * 1000),
                       "limit": 1500})
        if r.status_code == 200 and r.json():
            rows = r.json()
            df = _parse_klines(pd.DataFrame(rows))
            if len(rows) == 1500:
                r2 = get(rest, {"symbol": sym, "interval": iv, "startTime": rows[-1][0] + IV_MS[iv], "limit": 1500})
                if r2.status_code == 200 and r2.json():
                    df = pd.concat([df, _parse_klines(pd.DataFrame(r2.json()))])
            df = df[df["ts"] + pd.Timedelta(days=1) <= pd.Timestamp.now(tz="UTC")]
            df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            df.to_parquet(path(market, sym, iv), index=False)
            return len(df)
    last = old["ts"].max() if len(old) else None
    start_ts = pd.Timestamp(start, tz="UTC")
    frames = [old] if len(old) else []
    _, keys = s3_list(f"{prefix}/{sym}/{iv}/", None)
    for k in sorted(x for x in keys if x.endswith(".zip")):
        m = pd.Timestamp(re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) + "-01", tz="UTC")
        end = m + pd.offsets.MonthBegin(1)
        if end <= start_ts or (last is not None and end - pd.Timedelta(milliseconds=IV_MS[iv]) <= last):
            continue
        df = _zip_csv(k)
        if df is not None:
            frames.append(_parse_klines(df))
    cur = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    resume = cur["ts"].max() + pd.Timedelta(milliseconds=IV_MS[iv]) if len(cur) else start_ts
    now_ms = int(time.time() * 1000)
    st = int(resume.timestamp() * 1000)
    tail = []
    while st < now_ms and not NO_REST:
        r = get(rest, {"symbol": sym, "interval": iv, "startTime": st, "limit": 1500 if market != "spot" else 1000})
        if r.status_code != 200 or not r.json():
            break
        rows = r.json()
        tail.append(_parse_klines(pd.DataFrame(rows)))
        st = rows[-1][0] + IV_MS[iv]
        if len(rows) < 1000:
            break
    if tail:
        cur = pd.concat([cur] + tail, ignore_index=True)
    if not len(cur):
        return 0
    cur = cur[(cur["ts"] >= start_ts) & (cur["ts"] + pd.Timedelta(milliseconds=IV_MS[iv]) <= pd.Timestamp.now(tz="UTC"))]
    cur = cur.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    cur.to_parquet(path(market, sym, iv), index=False)
    return len(cur)


def update_funding(sym: str, start=START) -> int:
    old = load("funding", sym)
    last = old["ts"].max() if len(old) else None
    start_ts = pd.Timestamp(start, tz="UTC")
    frames = [old] if len(old) else []
    _, keys = s3_list(f"data/futures/um/monthly/fundingRate/{sym}/", None)
    for k in sorted(x for x in keys if x.endswith(".zip")):
        m = pd.Timestamp(re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) + "-01", tz="UTC")
        end = m + pd.offsets.MonthBegin(1)
        if end <= start_ts or (last is not None and end <= last):
            continue
        df = _zip_csv(k)
        if df is not None:
            df = df.iloc[:, :3]
            df.columns = ["calc_time", "interval_h", "rate"]
            frames.append(pd.DataFrame({"ts": pd.to_datetime(_norm_ts(df["calc_time"]), unit="ms", utc=True),
                                        "rate": pd.to_numeric(df["rate"]),
                                        "interval_h": pd.to_numeric(df["interval_h"], errors="coerce")}))
    cur = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts", "rate", "interval_h"])
    st = int((cur["ts"].max() + pd.Timedelta(minutes=1)).timestamp() * 1000) if len(cur) else int(start_ts.timestamp() * 1000)
    while not NO_REST:
        r = get("https://fapi.binance.com/fapi/v1/fundingRate", {"symbol": sym, "startTime": st, "limit": 1000})
        if r.status_code != 200 or not r.json():
            break
        rows = pd.DataFrame(r.json())
        cur = pd.concat([cur, pd.DataFrame({"ts": pd.to_datetime(rows["fundingTime"].astype("int64"), unit="ms", utc=True),
                                            "rate": rows["fundingRate"].astype(float), "interval_h": float("nan")})])
        st = int(rows["fundingTime"].astype("int64").max()) + 60_000
        if len(rows) < 1000:
            break
    if not len(cur):
        return 0
    # funding settles on the hour; archive calc_time can carry a few ms of jitter
    cur["ts"] = cur["ts"].dt.round("min")
    cur = cur[cur["ts"] >= start_ts].drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    gap_h = cur["ts"].diff().dt.total_seconds() / 3600
    cur["interval_h"] = cur["interval_h"].fillna(gap_h).fillna(8.0)
    cur.to_parquet(path("funding", sym), index=False)
    return len(cur)


def run_many(fn, items, workers=32):
    res = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *it): it for it in items}
        for f in as_completed(futs):
            try:
                res[futs[f]] = f.result()
            except Exception as e:
                log.warning("%s failed: %s", futs[f], e)
                res[futs[f]] = -1
    return res


# ---------------------------------------------------------------- Bybit / OKX

def bybit_funding(sym: str, start=START) -> pd.DataFrame:
    out, end = [], int(time.time() * 1000)
    stop = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    while end > stop:
        r = get("https://api.bybit.com/v5/market/funding/history",
                {"category": "linear", "symbol": sym, "limit": 200, "endTime": end})
        lst = r.json().get("result", {}).get("list", []) if r.status_code == 200 else []
        if not lst:
            break
        out += lst
        end = int(lst[-1]["fundingRateTimestamp"]) - 1
        time.sleep(0.05)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df = pd.DataFrame({"ts": pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True),
                       "rate": df["fundingRate"].astype(float)})
    return df[df["ts"] >= pd.Timestamp(start, tz="UTC")].drop_duplicates("ts").sort_values("ts")


def okx_daily(day: pd.Timestamp) -> pd.DataFrame | None:
    d = day.strftime("%Y-%m-%d")
    url = f"https://static.okx.com/cdn/okex/traderecords/swaprate/monthly/{day.strftime('%Y%m')}/allswaprate-swaprate-{d}.zip"
    r = get(url)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), encoding="gbk", encoding_errors="replace")
    df = df.iloc[:, [0, 3, 4]]
    df.columns = ["inst", "rate", "next_ts"]
    df = df[df["inst"].str.endswith("-USDT-SWAP")]
    df["date"] = day.normalize()
    return df
