"""Walk-forward research for leveraged cash-and-carry.

For every leverage setting:
  * simulate all signal-parameter combos over the full period (risk rule on, 15 min delay)
  * each quarter from WF_START: pick the combo with the best trailing-12m net return,
    use its out-of-sample returns for the next quarter (stitched)
  * holdout (HOLDOUT_START..end) is excluded from selection and evaluated once at the end
Outputs reports/research.json, reports/oos_daily.parquet
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from farb import data as D
from farb import sim
from farb.metrics import summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(D.ROOT / "logs" / "research.log"), logging.StreamHandler()])
log = logging.getLogger("research")

SIM_START, SIM_END = "2021-01-01", "2026-09-30"
FIRST_TRADE = "2021-02-01"
WF_START, HOLDOUT_START = "2022-01-01", "2025-10-01"
REPORTS = D.ROOT / "reports"

LEVS = [("1x", dict(lev=1)), ("2x", dict(lev=2)), ("3x", dict(lev=3)), ("5x", dict(lev=5)), ("8x", dict(lev=8)),
        ("10x", dict(lev=10)), ("15x", dict(lev=15)), ("20x", dict(lev=20)),
        ("dyn_c8", dict(dyn_c=8.0, lev_max=20)), ("dyn_c4", dict(dyn_c=4.0, lev_max=20))]
GRID = [dict(enter_apr=e, exit_apr=round(e * x, 4), lookback=lb, k=k)
        for e, x, lb, k in itertools.product([0.05, 0.10, 0.15, 0.25], [0.0, 0.5], [3, 9, 21], [3, 6])]

MK: sim.Market | None = None
UNI: pd.DataFrame | None = None


def daily(eq: pd.Series) -> pd.Series:
    d = eq.resample("D").last().dropna()
    return d.pct_change().dropna()


def run_one(args):
    lev_name, lev_kw, gi, extra = args
    p = sim.Params(**{**lev_kw, **GRID[gi], "delay": 3, **extra})
    t = time.time()
    r = sim.simulate(MK, UNI, p, start=FIRST_TRADE, end=SIM_END)
    ev = pd.DataFrame(r.events, columns=None) if r.events else pd.DataFrame()
    return {"lev": lev_name, "g": gi, "extra": extra, "ret": daily(r.equity["equity"]), "stats": r.stats,
            "gross": r.equity["gross"].resample("D").mean(), "npos": r.equity["n_pos"].resample("D").mean(),
            "liq_times": [e[0] for e in r.events if e[1] == "LIQUIDATION"],
            "liq_loss": [e[4] for e in r.events if e[1] == "LIQUIDATION"],
            "fund": r.fund_log.set_index("ts")["amount"].resample("D").sum() if len(r.fund_log) else pd.Series(dtype=float),
            "fund_neg": (r.fund_log.set_index("ts")["amount"] < 0).resample("D").sum() if len(r.fund_log) else pd.Series(dtype=float),
            "trades": r.trades, "sec": time.time() - t}


def walk_forward(runs: list[dict], wf_end: str) -> dict:
    """Stitch OOS quarters choosing, at each quarter start, the combo with best trailing-12m return."""
    by_g = {r["g"]: r for r in runs}
    qs = pd.date_range(WF_START, wf_end, freq="QS", tz="UTC")
    pieces, chosen = [], []
    for i, q in enumerate(qs):
        qe = qs[i + 1] if i + 1 < len(qs) else pd.Timestamp(wf_end, tz="UTC")
        tr0 = q - pd.DateOffset(months=12)
        best, best_v = None, -np.inf
        for g, r in by_g.items():
            x = r["ret"][(r["ret"].index >= tr0) & (r["ret"].index < q)]
            v = (1 + x).prod() - 1 if len(x) > 30 else -np.inf
            if v > best_v:
                best, best_v = g, v
        r = by_g[best]
        pieces.append(r["ret"][(r["ret"].index >= q) & (r["ret"].index < qe)])
        chosen.append((str(q.date()), best, GRID[best], round(best_v, 4),
                       sum(1 for t in r["liq_times"] if q <= t < qe)))
    return {"ret": pd.concat(pieces), "chosen": chosen}


def oos_extract(runs, chosen, key):
    by_g = {r["g"]: r for r in runs}
    out = []
    qs = [pd.Timestamp(c[0], tz="UTC") for c in chosen] + [pd.Timestamp(HOLDOUT_START, tz="UTC")]
    for (q0, g, *_), q1 in zip(chosen, qs[1:]):
        s = by_g[g][key]
        q0 = pd.Timestamp(q0, tz="UTC")
        out.append(s[(s.index >= q0) & (s.index < q1)])
    return pd.concat(out) if out else pd.Series(dtype=float)


def main():
    global MK, UNI
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse reports/runs_main.pkl")
    a = ap.parse_args()
    t0 = time.time()
    UNI = pd.read_parquet(D.DATA / "universe.parquet")
    syms = sorted(UNI["symbol"].unique())
    log.info("loading market for %d symbols", len(syms))
    MK = sim.Market(syms, SIM_START, SIM_END)
    log.info("market loaded: %d symbols with full data (%.0fs)", len(MK.sym), time.time() - t0)
    grid_idx = range(len(GRID)) if not a.quick else [5, 17]
    levs = LEVS if not a.quick else LEVS[:3]
    jobs = [(n, kw, g, {}) for n, kw in levs for g in grid_idx]
    if a.reuse and (REPORTS / "runs_main.pkl").exists():
        runs = pickle.load(open(REPORTS / "runs_main.pkl", "rb"))
        log.info("reusing %d saved main sims", len(runs))
    else:
        with mp.get_context("fork").Pool(a.workers) as pool:
            runs = []
            for k, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
                runs.append(r)
                if k % 20 == 0:
                    log.info("%d/%d sims done (last %.0fs)", k + 1, len(jobs), r["sec"])
        pickle.dump(runs, open(REPORTS / "runs_main.pkl", "wb"))
    hold_end = SIM_END
    wf = {}
    for n, _ in levs:
        rr = [r for r in runs if r["lev"] == n]
        w = walk_forward(rr, HOLDOUT_START)
        wf[n] = w
    # delay / risk-rule stress on the combos actually chosen
    stress_jobs = []
    for n, kw in levs:
        for g in sorted({c[1] for c in wf[n]["chosen"]}):
            for extra in ({"delay": 1}, {"delay": 12}, {"risk_rule": False}):
                stress_jobs.append((n, kw, g, extra))
    with mp.get_context("fork").Pool(a.workers) as pool:
        sruns = list(pool.imap_unordered(run_one, stress_jobs, chunksize=1))
    pickle.dump(sruns, open(REPORTS / "runs_stress.pkl", "wb"))
    out = {"wf_period": [WF_START, HOLDOUT_START], "grid": GRID, "levels": {}}
    oos = {}
    for n, kw in levs:
        rr = [r for r in runs if r["lev"] == n]
        w = wf[n]
        ret = w["ret"]
        oos[n] = ret
        fund = oos_extract(rr, w["chosen"], "fund")
        fneg = oos_extract(rr, w["chosen"], "fund_neg")
        gross = oos_extract(rr, w["chosen"], "gross")
        st = summary(ret)
        liq = sum(c[4] for c in w["chosen"])
        # costs in OOS quarters
        by_g = {r["g"]: r for r in rr}
        cost = 0.0
        for i, (q0, g, *_rest) in enumerate(w["chosen"]):
            q0 = pd.Timestamp(q0, tz="UTC")
            q1 = pd.Timestamp(w["chosen"][i + 1][0], tz="UTC") if i + 1 < len(w["chosen"]) else pd.Timestamp(HOLDOUT_START, tz="UTC")
            tr = by_g[g]["trades"]
            cost += tr[(tr["ts"] >= q0) & (tr["ts"] < q1)]["cost"].sum()
        st.update({"liquidations": liq, "funding_usdt": float(fund.sum()), "costs_usdt": float(cost),
                   "cost_to_funding": float(cost / fund.sum()) if fund.sum() > 0 else None,
                   "neg_funding_days_share": float((fneg > 0).mean()) if len(fneg) else None,
                   "avg_gross_notional": float(gross.mean()) if len(gross) else None,
                   "liq_distance": None, "chosen": w["chosen"]})
        L = kw.get("lev")
        if L:
            st["liq_distance"] = {"major": (1 / L - 0.005) / 1.005, "alt": (1 / L - 0.0125) / 1.0125}
        # stress variants: stitch with same chosen combos
        for tag, extra in (("delay_5m", {"delay": 1}), ("delay_60m", {"delay": 12}), ("no_risk_rule", {"risk_rule": False})):
            sr = [r for r in sruns if r["lev"] == n and r["extra"] == extra]
            pieces, liqs = [], 0
            sbg = {r["g"]: r for r in sr}
            for i, (q0, g, *_r) in enumerate(w["chosen"]):
                q0 = pd.Timestamp(q0, tz="UTC")
                q1 = pd.Timestamp(w["chosen"][i + 1][0], tz="UTC") if i + 1 < len(w["chosen"]) else pd.Timestamp(HOLDOUT_START, tz="UTC")
                x = sbg[g]["ret"]
                pieces.append(x[(x.index >= q0) & (x.index < q1)])
                liqs += sum(1 for t in sbg[g]["liq_times"] if q0 <= t < q1)
            sret = pd.concat(pieces)
            ss = summary(sret)
            st[tag] = {"cagr": ss["cagr"], "max_dd": ss["max_dd"], "worst_month": ss["worst_month"], "liquidations": liqs}
        out["levels"][n] = st
        log.info("%-7s CAGR %6.2f%% maxDD %6.2f%% worstM %6.2f%% liq %d | 60m-delay liq %d, no-rule liq %d",
                 n, 100 * st["cagr"], 100 * st["max_dd"], 100 * st["worst_month"], liq,
                 st["delay_60m"]["liquidations"], st["no_risk_rule"]["liquidations"])
    # ---- pre-committed recommendation rule (decided BEFORE the holdout is looked at) ----
    ok = {n: v for n, v in out["levels"].items()
          if v["liquidations"] == 0 and v["delay_60m"]["liquidations"] == 0 and v["max_dd"] > -0.10}
    rec = max(ok, key=lambda n: ok[n]["cagr"]) if ok else "1x"
    lev_kw = dict(LEVS)[rec]
    out["recommended"] = rec
    out["rule"] = "max WF CAGR s.t. 0 liquidations (base & 60-min delay) and maxDD > -10%; else 1x"
    (REPORTS / "selected.json").write_text(json.dumps({
        "recommended": rec, "lev_kw": lev_kw, "holdout_combo": wf[rec]["chosen"][-1][2],
        "committed_at": pd.Timestamp.now(tz="UTC").isoformat()}, indent=1))
    log.info("recommended leverage setting: %s", rec)
    pd.DataFrame(oos).to_parquet(REPORTS / "oos_daily.parquet")
    (REPORTS / "research.json").write_text(json.dumps(out, indent=1, default=str))
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    sys.exit(main())
