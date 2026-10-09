"""Simple BTC+ETH carry (added AFTER the alt-selection walk-forward results were seen; it has
NO tuned parameters - the rules are the ones used in the simulator sanity check):
  hold spot+short in BTC and ETH (half the capital each) whenever the last-day mean funding
  is >= 0; exit only if it falls below -20% APR.  Same leverage grid and stress tests.
Writes reports/majors.json and reports/majors_daily.parquet (WF period + holdout kept separate).
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from farb import data as D
from farb import sim
from farb.metrics import summary

sys.path.insert(0, str(D.ROOT / "scripts"))
from research import FIRST_TRADE, HOLDOUT_START, LEVS, REPORTS, SIM_END, SIM_START, WF_START, daily  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
RULE = dict(k=2, enter_apr=0.0, exit_apr=-0.2, lookback=3, min_pos_frac=0.0)


def main():
    mk = sim.Market(SYMS, SIM_START, SIM_END)
    months = pd.date_range(SIM_START, SIM_END, freq="MS", tz="UTC")
    uni = pd.DataFrame([(m, s, i + 1) for m in months for i, s in enumerate(SYMS)], columns=["month", "symbol", "rank"])
    out, daily_wf, daily_ho = {}, {}, {}
    for n, kw in LEVS:
        res = {}
        for tag, extra in (("base", {"delay": 3}), ("delay_5m", {"delay": 1}), ("delay_60m", {"delay": 12}),
                           ("no_risk_rule", {"delay": 3, "risk_rule": False}),
                           ("maker_fees", {"delay": 3, "fees": sim.Fees.maker()})):
            p = sim.Params(**{**kw, **RULE, **extra})
            r = sim.simulate(mk, uni, p, start=FIRST_TRADE, end=SIM_END)
            d = daily(r.equity["equity"])
            wf = d[(d.index >= WF_START) & (d.index < HOLDOUT_START)]
            s = summary(wf)
            liq = [e for e in r.events if e[1] == "LIQUIDATION"]
            s["liquidations"] = sum(1 for e in liq if pd.Timestamp(WF_START, tz="UTC") <= e[0] < pd.Timestamp(HOLDOUT_START, tz="UTC"))
            if tag == "base":
                fl = r.fund_log.set_index("ts")
                fl = fl[(fl.index >= WF_START) & (fl.index < HOLDOUT_START)]
                tr = r.trades[(r.trades["ts"] >= WF_START) & (r.trades["ts"] < HOLDOUT_START)]
                s["funding_usdt"] = float(fl["amount"].sum())
                s["costs_usdt"] = float(tr["cost"].sum())
                s["cost_to_funding"] = s["costs_usdt"] / s["funding_usdt"] if s["funding_usdt"] > 0 else None
                s["neg_funding_print_share"] = float((fl["amount"] < 0).mean())
                s["rebalances"] = r.stats["rebalances"]
                daily_wf[n] = wf
                daily_ho[n] = d[d.index >= HOLDOUT_START]
                s["_holdout_liq"] = sum(1 for e in liq if e[0] >= pd.Timestamp(HOLDOUT_START, tz="UTC"))
            res[tag] = s
        L = kw.get("lev")
        if L:
            res["liq_distance_btc"] = (1 / L - 0.005) / 1.005
        out[n] = res
        b = res["base"]
        print(f"{n:7s} CAGR {100*b['cagr']:6.2f}% maxDD {100*b['max_dd']:6.2f}% worstM {100*b['worst_month']:6.2f}% "
              f"liq {b['liquidations']} | 60m-delay liq {res['delay_60m']['liquidations']} no-rule liq {res['no_risk_rule']['liquidations']} "
              f"| maker CAGR {100*res['maker_fees']['cagr']:.2f}%")
    (REPORTS / "majors.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame(daily_wf).to_parquet(REPORTS / "majors_wf_daily.parquet")
    pd.DataFrame(daily_ho).to_parquet(REPORTS / "majors_holdout_daily.parquet")   # NOT read until final test


if __name__ == "__main__":
    main()
