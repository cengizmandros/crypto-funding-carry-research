"""Live PAPER trading of the leveraged cash-and-carry. Public market data only:
no API keys, no orders. Fills are simulated on live bid/ask with fees + slippage.

  every 5 min : mark-to-market, liquidation check, margin risk rule (rebalance)
  every 8h    : credit settled funding, then exits/entries from the funding signal
State: paper/paper.db (SQLite).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import requests

from .data import ROOT, STABLE
from .sim import MAJORS, Fees, Params, half_cost

log = logging.getLogger(__name__)
DB = ROOT / "paper" / "paper.db"
FAPI, SAPI = "https://fapi.binance.com/fapi/v1", "https://api.binance.com/api/v3"
CAPITAL = 10_000.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, q REAL, f_ref REAL, margin REAL, lev REAL,
  opened_at TEXT, funding REAL, costs REAL, entry_spot REAL, entry_fut REAL);
CREATE TABLE IF NOT EXISTS funding(ts TEXT, symbol TEXT, rate REAL, mark REAL, q REAL, amount REAL, lev REAL);
CREATE TABLE IF NOT EXISTS snapshots(ts TEXT PRIMARY KEY, equity REAL, cash REAL, n_pos INTEGER, gross REAL,
  min_margin_ratio REAL, btc_mark REAL);
CREATE TABLE IF NOT EXISTS pos_snap(ts TEXT, symbol TEXT, mark REAL, eff_lev REAL, margin_ratio REAL,
  liq_price REAL, dist_to_liq REAL, fut_equity REAL, spot_value REAL);
CREATE TABLE IF NOT EXISTS trades(ts TEXT, symbol TEXT, kind TEXT, q REAL, spot_px REAL, fut_px REAL, cost REAL);
CREATE TABLE IF NOT EXISTS signals(ts TEXT, symbol TEXT, apr REAL, pos_frac REAL, action TEXT);
CREATE TABLE IF NOT EXISTS events(ts TEXT, kind TEXT, detail TEXT);
"""


def db():
    con = sqlite3.connect(DB, timeout=30)
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def gs(con, k, d=None):
    r = con.execute("SELECT value FROM state WHERE key=?", (k,)).fetchone()
    return json.loads(r[0]) if r else d


def ss(con, k, v):
    con.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (k, json.dumps(v)))


def ev(con, kind, detail=""):
    con.execute("INSERT INTO events VALUES(?,?,?)", (now(), kind, str(detail)[:1000]))
    log.info("%s %s", kind, detail)


def params() -> Params:
    cfg = json.loads((ROOT / "reports" / "selected.json").read_text())
    p = cfg["params"]
    return Params(**{k: v for k, v in p.items() if k in Params.__dataclass_fields__ and k != "fees"})


def _get(url, **params):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def premium_index() -> pd.DataFrame:
    df = pd.DataFrame(_get(f"{FAPI}/premiumIndex"))
    df["markPrice"] = df["markPrice"].astype(float)
    return df.set_index("symbol")


def book(api, symbols):
    js = _get(f"{api}/ticker/bookTicker")
    df = pd.DataFrame(js).set_index("symbol")
    return df.loc[df.index.intersection(symbols), ["bidPrice", "askPrice"]].astype(float)


def universe(n=30) -> list[str]:
    fut = pd.DataFrame(_get(f"{FAPI}/ticker/24hr"))
    fut = fut[fut["symbol"].str.endswith("USDT")]
    spot_syms = {s["symbol"] for s in _get(f"{SAPI}/exchangeInfo", permissions="SPOT")["symbols"] if s["status"] == "TRADING"}
    fut["qv"] = fut["quoteVolume"].astype(float)
    fut = fut[fut["symbol"].isin(spot_syms) & ~fut["symbol"].str[:-4].isin(STABLE)]
    return list(fut.sort_values("qv", ascending=False)["symbol"].head(n))


def funding_signal(sym, lookback) -> tuple[float, float, pd.DataFrame]:
    rows = pd.DataFrame(_get(f"{FAPI}/fundingRate", symbol=sym, limit=100))
    rows["ts"] = pd.to_datetime(rows["fundingTime"].astype("int64"), unit="ms", utc=True).dt.round("min")
    rows["rate"] = rows["fundingRate"].astype(float)
    rows = rows.sort_values("ts")
    gap_h = rows["ts"].diff().dt.total_seconds().div(3600).fillna(8).clip(lower=1)
    rows["r8"] = rows["rate"] * 8 / gap_h
    w = rows[rows["ts"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=8 * lookback)]
    if not len(w):
        return np.nan, np.nan, rows
    return float(w["r8"].mean() * 1095), float((w["rate"] > 0).mean()), rows


def mmr(s):
    return 0.005 if s in MAJORS else 0.0125


def init():
    with closing(db()) as con:
        if gs(con, "cash") is None:
            ss(con, "cash", CAPITAL)
            ss(con, "started", now())
            ev(con, "init", f"capital {CAPITAL} params {params()}")
        con.commit()


def _cost(notional, stress=1.0):
    f = Fees()
    # live bid/ask already pays the spread; add fees + a slippage allowance
    return notional * (f.spot + f.fut + 2 * 0.0002 * stress)


def _positions(con):
    return pd.read_sql("SELECT * FROM positions", con).set_index("symbol")


def risk_check():
    """5-minute job: mark-to-market, liquidation, risk-rule rebalance, snapshots."""
    p = params()
    with closing(db()) as con:
        pos = _positions(con)
        pi = premium_index()
        ts = now()
        cash = gs(con, "cash")
        equity, gross, min_mr = cash, 0.0, np.inf
        for s, r in pos.iterrows():
            mark = float(pi.loc[s, "markPrice"]) if s in pi.index else r["f_ref"]
            fut_eq = r["margin"] + r["q"] * (r["f_ref"] - mark)
            m = mmr(s)
            liq_px = (r["margin"] + r["q"] * r["f_ref"]) / (r["q"] * (1 + m))
            if fut_eq <= m * r["q"] * mark:   # simulated liquidation: margin gone, sell spot
                bk = book(SAPI, [s])
                bid = float(bk.loc[s, "bidPrice"])
                c = r["q"] * bid * (Fees().spot + 0.0006 * p.stress_slip)
                cash += r["q"] * bid - c
                con.execute("DELETE FROM positions WHERE symbol=?", (s,))
                con.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?)", (ts, s, "LIQUIDATION", r["q"], bid, mark, c))
                ev(con, "LIQUIDATION", f"{s} mark={mark} liq={liq_px:.6g} lev={r['lev']}")
                continue
            eff = r["q"] * mark / max(fut_eq, 1e-9)
            if p.risk_rule and eff > p.hi * r["lev"]:
                bk = book(SAPI, [s])
                spx = float(bk.loc[s, "bidPrice"])
                total = fut_eq + r["q"] * spx
                q_new = total * r["lev"] / ((r["lev"] + 1) * spx)
                c = _cost(abs(r["q"] - q_new) * spx, p.stress_slip)
                con.execute("UPDATE positions SET q=?, f_ref=?, margin=?, costs=costs+? WHERE symbol=?",
                            (q_new, mark, total - q_new * spx - c, c, s))
                con.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?)", (ts, s, "risk_rebalance", r["q"] - q_new, spx, mark, c))
                ev(con, "risk_rebalance", f"{s} eff_lev {eff:.2f} > {p.hi}x{r['lev']:.1f}; q {r['q']:.6g}->{q_new:.6g}")
                fut_eq, r_q, f_ref, margin = total - q_new * spx - c, q_new, mark, total - q_new * spx - c
                liq_px = (margin + r_q * f_ref) / (r_q * (1 + m))
                eff = r_q * mark / fut_eq
            else:
                r_q = r["q"]
            spot_val = r_q * mark  # spot leg valued at mark (basis is small)
            mr = m * r_q * mark / max(fut_eq, 1e-9)
            min_mr = min(min_mr, (fut_eq - m * r_q * mark) / max(r_q * mark, 1e-9))
            equity += fut_eq + spot_val
            gross += r_q * mark
            con.execute("INSERT INTO pos_snap VALUES(?,?,?,?,?,?,?,?,?)",
                        (ts, s, mark, eff, mr, liq_px, liq_px / mark - 1, fut_eq, spot_val))
        ss(con, "cash", cash)
        btc = float(pi.loc["BTCUSDT", "markPrice"])
        con.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?,?,?)",
                    (ts, equity, cash, len(_positions(con)), gross / equity if equity else 0,
                     None if min_mr == np.inf else min_mr, btc))
        con.commit()


def funding_cycle():
    """8h job (a few minutes after settlement): credit funding, then exits/entries."""
    p = params()
    with closing(db()) as con:
        pos = _positions(con)
        for s, r in pos.iterrows():
            _, _, rows = funding_signal(s, p.lookback)
            opened = pd.Timestamp(r["opened_at"])
            last_done = pd.Timestamp(gs(con, f"lf_{s}", "1970-01-01T00:00:00+00:00"))
            due = rows[(rows["ts"] > max(last_done, opened))]
            newest = last_done
            for _, f in due.iterrows():
                mark = float(_get(f"{FAPI}/fundingRate", symbol=s, startTime=int(f["ts"].timestamp() * 1000) - 60000,
                                  limit=1)[0].get("markPrice") or r["f_ref"])
                amt = r["q"] * mark * f["rate"]
                con.execute("UPDATE positions SET margin=margin+?, funding=funding+? WHERE symbol=?", (amt, amt, s))
                con.execute("INSERT INTO funding VALUES(?,?,?,?,?,?,?)",
                            (f["ts"].isoformat(), s, f["rate"], mark, r["q"], amt, r["lev"]))
                newest = max(newest, f["ts"])
            if newest > last_done:
                ss(con, f"lf_{s}", newest.isoformat())
        con.commit()

        cfg = json.loads((ROOT / "reports" / "selected.json").read_text())
        uni = cfg.get("symbols") or universe()     # selected system may be restricted (e.g. BTC+ETH)
        pos = _positions(con)
        sig = {}
        for s in sorted(set(uni) | set(pos.index)):
            try:
                a, f, _ = funding_signal(s, p.lookback)
                sig[s] = (a, f)
            except Exception as e:
                log.warning("signal %s: %s", s, e)
        ts = now()
        cash = gs(con, "cash")
        pi = premium_index()
        sb, fb = book(SAPI, list(sig)), book(FAPI, list(sig))
        # exits
        for s, r in pos.iterrows():
            a = sig.get(s, (-1, 0))[0]
            if a < p.exit_apr or (s not in uni and a < p.enter_apr):
                spx, fpx = float(sb.loc[s, "bidPrice"]), float(fb.loc[s, "askPrice"])
                c = _cost(r["q"] * spx)
                val = r["margin"] + r["q"] * (r["f_ref"] - fpx) + r["q"] * spx - c
                cash += val
                con.execute("DELETE FROM positions WHERE symbol=?", (s,))
                con.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?)", (ts, s, "close", r["q"], spx, fpx, c))
                con.execute("INSERT INTO signals VALUES(?,?,?,?,?)", (ts, s, a, sig.get(s, (0, 0))[1], "exit"))
                ev(con, "close", f"{s} apr={a:.3f} funding_total={r['funding']:.2f} value={val:.2f}")
        pos = _positions(con)
        equity = cash + sum(r["margin"] + r["q"] * (r["f_ref"] - float(pi.loc[s, "markPrice"])) + r["q"] * float(pi.loc[s, "markPrice"])
                            for s, r in pos.iterrows() if s in pi.index)
        slot = equity / p.k
        cand = sorted([(a, s) for s, (a, f) in sig.items()
                       if s in uni and s not in pos.index and np.isfinite(a) and a >= p.enter_apr and f >= p.min_pos_frac],
                      reverse=True)
        for a, s in cand:
            if len(pos) >= p.k or cash < slot * 0.5:
                break
            A = min(slot, cash)
            L = p.lev
            N = A * L / (L + 1)
            spx, fpx = float(sb.loc[s, "askPrice"]), float(fb.loc[s, "bidPrice"])
            c = _cost(N)
            q = N / spx
            cash -= A
            con.execute("INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (s, q, fpx, A - N - c, L, ts, 0.0, c, spx, fpx))
            con.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?)", (ts, s, "open", q, spx, fpx, c))
            con.execute("INSERT INTO signals VALUES(?,?,?,?,?)", (ts, s, a, sig[s][1], "enter"))
            ev(con, "open", f"{s} apr={a:.3f} lev={L} notional={N:.2f} basis={fpx / spx - 1:+.4%}")
            pos = _positions(con)
        for s, (a, f) in sig.items():
            con.execute("INSERT INTO signals VALUES(?,?,?,?,?)", (ts, s, a, f, "scan"))
        ss(con, "cash", cash)
        ss(con, "last_cycle", ts)
        con.commit()
