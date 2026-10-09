"""Daily funding-regime monitor (read-only, public endpoints, no keys, no trading).

Once a day (after the 00:00 UTC funding print) it:
  1. ranks USDT perps by 24h futures quote volume, keeps the top 30 that also trade on spot
  2. fetches each coin's settled funding prints for the missing UTC day(s)
  3. annualised funding per coin per day = sum(prints in that day) * 365
     (correct for 8h, 4h and 1h funding intervals alike)
  4. appends the cross-coin median to logs/funding_monitor.csv
  5. if the median is > 10% for 14 consecutive calendar days -> writes ~/funding-arb/ALARM.txt

API budget per normal run: 1x /fapi/v1/ticker/24hr (weight 40), 1x /api/v3/ticker/price (weight 4),
30x /fapi/v1/fundingRate (limit-500-per-5-min bucket) spaced 0.7 s apart. On any 418/429 it stops
immediately, records the retry-after time and does not call again before it has passed.
Missed days (PC off) are back-filled from the same fundingRate calls (up to 30 days back).
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from farb import data as D
from farb import notify

ROOT = D.ROOT
LOG_CSV = ROOT / "logs" / "funding_monitor.csv"
DETAIL = ROOT / "logs" / "funding_monitor_detail.jsonl"
STATE = ROOT / "logs" / "funding_monitor_state.json"
ALARM = ROOT / "ALARM.txt"
FAPI, SAPI = "https://fapi.binance.com/fapi/v1", "https://api.binance.com/api/v3"
TOP_N, THRESHOLD, STREAK_DAYS, MAX_BACKFILL = 30, 0.10, 14, 30
FIELDS = ["date", "median_apr", "p75_apr", "share_neg", "n_coins", "above_threshold", "streak", "source", "computed_at"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(ROOT / "logs" / "monitor.log"), logging.StreamHandler()])
log = logging.getLogger("monitor")


class RateLimited(Exception):
    pass


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(st: dict):
    STATE.write_text(json.dumps(st, indent=1, default=str))


def call(url: str, params: dict | None = None):
    """One throttled request. Never retries into a rate limit: 418/429 -> stop the run."""
    D._throttle(url)
    r = D._s.get(url, params=params, timeout=30)
    if r.status_code in (418, 429):
        raise RateLimited(float(r.headers.get("retry-after", 300)))
    r.raise_for_status()
    return r.json()


def top_perps() -> list[str]:
    fut = pd.DataFrame(call(f"{FAPI}/ticker/24hr"))
    spot = {x["symbol"] for x in call(f"{SAPI}/ticker/price")}
    fut = fut[fut["symbol"].str.endswith("USDT") & fut["symbol"].isin(spot)
              & ~fut["symbol"].str[:-4].isin(D.STABLE)]
    fut["qv"] = fut["quoteVolume"].astype(float)
    return list(fut.sort_values("qv", ascending=False)["symbol"].head(TOP_N))


def read_log() -> pd.DataFrame:
    if not LOG_CSV.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(LOG_CSV)


def streak_of(df: pd.DataFrame) -> int:
    """Consecutive calendar days, ending at the latest logged day, with median > threshold.
    A missing day breaks the streak."""
    if df.empty:
        return 0
    d = df.assign(date=pd.to_datetime(df["date"])).sort_values("date").drop_duplicates("date", keep="last")
    n, prev = 0, None
    for _, row in d.iloc[::-1].iterrows():
        if prev is not None and (prev - row["date"]).days != 1:
            break
        if not row["median_apr"] > THRESHOLD:
            break
        n += 1
        prev = row["date"]
    return n


def main() -> int:
    st = load_state()
    now = pd.Timestamp.now(tz="UTC")
    ban_until = pd.Timestamp(st["ban_until"]) if st.get("ban_until") else None
    if ban_until is not None and now < ban_until:
        log.warning("rate-limit cool-down until %s; skipping this run", ban_until)
        return 0

    last_full_day = now.normalize() - pd.Timedelta(days=1)          # yesterday (complete UTC day)
    hist = read_log()
    done = set(hist["date"].astype(str)) if len(hist) else set()
    first = max(last_full_day - pd.Timedelta(days=MAX_BACKFILL - 1),
                pd.Timestamp(max(done), tz="UTC") + pd.Timedelta(days=1) if done else last_full_day - pd.Timedelta(days=STREAK_DAYS - 1))
    todo = [d for d in pd.date_range(first, last_full_day, freq="D") if str(d.date()) not in done]
    if not todo:
        log.info("nothing to do: %s already logged", last_full_day.date())
        return 0

    try:
        syms = top_perps()
        start_ms = int(todo[0].timestamp() * 1000)
        prints = {}
        for s in syms:
            rows = call(f"{FAPI}/fundingRate", {"symbol": s, "startTime": start_ms, "limit": 1000})
            prints[s] = pd.DataFrame(rows)
    except RateLimited as e:
        st["ban_until"] = (now + pd.Timedelta(seconds=e.args[0] + 60)).isoformat()
        st["last_error"] = f"{now.isoformat()} rate limited, retry-after {e.args[0]:.0f}s"
        save_state(st)
        log.error("rate limited; stopping until %s", st["ban_until"])
        return 1
    except requests.RequestException as e:
        st["last_error"] = f"{now.isoformat()} {e!r}"
        save_state(st)
        log.error("request failed: %r", e)
        return 1

    new_rows = []
    with DETAIL.open("a") as det:
        for day in todo:
            apr = {}
            for s, df in prints.items():
                if df.empty:
                    continue
                ts = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True).dt.round("min")
                m = (ts >= day) & (ts < day + pd.Timedelta(days=1))
                if m.any():
                    apr[s] = float(df.loc[m, "fundingRate"].astype(float).sum() * 365)
            if len(apr) < TOP_N // 2:
                log.warning("%s: only %d coins with prints, skipped", day.date(), len(apr))
                continue
            v = np.array(list(apr.values()))
            new_rows.append({"date": str(day.date()), "median_apr": round(float(np.median(v)), 5),
                             "p75_apr": round(float(np.quantile(v, 0.75)), 5),
                             "share_neg": round(float((v < 0).mean()), 4), "n_coins": len(v),
                             "above_threshold": bool(np.median(v) > THRESHOLD),
                             "source": "live" if day == last_full_day else "backfill",
                             "computed_at": now.isoformat(timespec="seconds")})
            det.write(json.dumps({"date": str(day.date()), "apr": {k: round(x, 5) for k, x in apr.items()}}) + "\n")

    all_rows = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else hist
    # recompute streak for each new row so the column is meaningful in the log
    for r in new_rows:
        r["streak"] = streak_of(all_rows[pd.to_datetime(all_rows["date"]) <= pd.Timestamp(r["date"])])
    new_file = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in new_rows:
            w.writerow(r)

    streak = streak_of(all_rows)
    st.update({"last_run": now.isoformat(), "last_day": str(last_full_day.date()), "streak": streak,
               "last_median": new_rows[-1]["median_apr"] if new_rows else st.get("last_median"),
               "threshold": THRESHOLD, "streak_needed": STREAK_DAYS, "ban_until": None, "last_error": None})
    if streak >= STREAK_DAYS and not ALARM.exists():
        recent = all_rows.tail(STREAK_DAYS)[["date", "median_apr"]].to_string(index=False)
        ALARM.write_text(
            f"FUNDING ALARM  ({now.isoformat(timespec='minutes')})\n\n"
            f"Hacimde ilk {TOP_N} coin'in medyan yıllık funding'i {streak} gündür üst üste %{THRESHOLD*100:.0f}'un üzerinde.\n"
            f"Bu, carry stratejisinin backtest'te stablecoin getirisini geçtiği rejime (2021, 2024) benziyor.\n"
            f"Bu bir işlem sinyali değil: önce paper trading ile doğrula.\n\n{recent}\n")
        st["alarm_since"] = now.isoformat()
        notify.send("Funding alarmı")   # content-free push: no numbers
        log.warning("ALARM written: streak %d", streak)
    save_state(st)
    for r in new_rows:
        log.info("%s median %.2f%% p75 %.2f%% neg %.0f%% n=%d (%s) streak=%d", r["date"], 100 * r["median_apr"],
                 100 * r["p75_apr"], 100 * r["share_neg"], r["n_coins"], r["source"], r["streak"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
