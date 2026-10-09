import numpy as np
import pandas as pd

from farb import sim


def fake_market(price_path_5m: np.ndarray, rate=0.0001, start="2024-01-01", days=20):
    mk = object.__new__(sim.Market)
    mk.start = pd.Timestamp(start, tz="UTC")
    mk.end = mk.start + pd.Timedelta(days=days)
    mk.grid = pd.date_range(mk.start, mk.end, freq="5min", tz="UTC")
    n = len(mk.grid)
    px = np.resize(price_path_5m, n).astype(float)
    s = "AAAUSDT"
    mk.sym = [s]
    mk.mark_hi, mk.mark_cl = {s: px * 1.0}, {s: px}
    hours = pd.date_range(mk.start, mk.end, freq="1h", tz="UTC")
    hp = pd.Series(px[::12][: len(hours)], index=hours[: len(px[::12])])
    mk.spot_px, mk.fut_px = {s: hp}, {s: hp}
    mk.adv_s = {s: pd.Series(1e10, index=hours)}
    mk.adv_f = {s: pd.Series(1e10, index=hours)}
    mk.vol = {s: pd.Series(0.03, index=hours)}
    ft = pd.date_range(mk.start, mk.end, freq="8h", tz="UTC")
    mk.fund = {s: pd.DataFrame({"rate": rate, "interval_h": 8.0, "r8": rate}, index=ft)}
    mk.gidx = {}
    mk._sig = {}
    uni = pd.DataFrame({"month": [mk.start.to_period("M").to_timestamp().tz_localize("UTC")], "symbol": [s], "rank": [1]})
    return mk, uni


def test_flat_price_collects_funding_minus_costs():
    mk, uni = fake_market(np.full(10, 100.0), rate=0.0001)
    p = sim.Params(lev=3, k=1, enter_apr=0.05, exit_apr=0.0, lookback=3, min_pos_frac=0.5)
    r = sim.simulate(mk, uni, p, start="2024-01-03", end="2024-01-19")
    N = 10_000 * 3 / 4
    expected_funding = r.stats["funding"]
    assert abs(expected_funding - N * 0.0001 * r.stats["held_periods"]) < 1e-6 * N
    final = r.equity["equity"].iloc[-1]
    assert abs(final - (10_000 + expected_funding - r.stats["fees"] - r.stats["slippage"])) < 1.0


def test_spike_liquidates_high_leverage_without_risk_rule():
    path = np.full(5000, 100.0)
    path[3000:] = 112.0          # +12% jump in one 5m bar
    mk, uni = fake_market(path, rate=0.0003)
    p = sim.Params(lev=10, k=1, enter_apr=0.05, exit_apr=0.0, lookback=3, min_pos_frac=0.5, risk_rule=False)
    r = sim.simulate(mk, uni, p, start="2024-01-03", end="2024-01-19")
    assert r.stats["liquidations"] == 1


def test_risk_rule_saves_moderate_leverage_on_gradual_rise():
    path = np.full(5000, 100.0)
    path[3000:3200] = np.linspace(100, 125, 200)   # +25% over ~17 hours
    path[3200:] = 125.0
    mk, uni = fake_market(path, rate=0.0003)
    p = sim.Params(lev=5, k=1, enter_apr=0.05, exit_apr=0.0, lookback=3, min_pos_frac=0.5, risk_rule=True, delay=3)
    r = sim.simulate(mk, uni, p, start="2024-01-03", end="2024-01-19")
    assert r.stats["liquidations"] == 0 and r.stats["rebalances"] >= 1
    p2 = sim.Params(lev=5, k=1, enter_apr=0.05, exit_apr=0.0, lookback=3, min_pos_frac=0.5, risk_rule=False)
    r2 = sim.simulate(mk, uni, p2, start="2024-01-03", end="2024-01-19")
    assert r2.stats["liquidations"] == 1
