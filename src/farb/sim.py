"""Leveraged cash-and-carry simulator (spot long + USD-M perp short, isolated margin).

Clock
  decision times tau: every 8h (00/08/16 UTC), right after funding settles.
  signals use only funding settled at or before tau.
  trades fill at exec time e = tau + 1h, at the 1h close of spot and perp.
  between fills, each open position is walked on 5-minute MARK-price bars:
    * funding is credited/debited at its real settlement time, on q * mark
    * liquidation if futures equity at the bar's mark HIGH <= mmr * notional
    * risk rule: if effective leverage exceeds hi*L, rebalance back to L
      after `delay` bars (the fill uses the price `delay` bars later; liquidation
      can still happen during the delay)
  after a liquidation the margin is gone and the now-unhedged spot is sold
  `delay` bars later with stressed slippage.

Capital per position A:   notional N = A*L/(L+1)  (spot N, margin N/L)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import data as D

MAJORS = {"BTCUSDT", "ETHUSDT"}


@dataclass(frozen=True)
class Fees:
    spot: float = 0.0010      # Binance spot VIP0 (maker = taker = 0.10%)
    fut: float = 0.0005       # USD-M taker 0.05% (maker 0.02%)
    slip_mult: float = 1.0    # 1.0 = taker crossing the spread; 0.5 ~ passive/maker fills

    @staticmethod
    def maker():
        return Fees(spot=0.00075, fut=0.0002, slip_mult=0.5)   # BNB-discounted spot, futures maker


@dataclass(frozen=True)
class Params:
    lev: float = 3.0              # fixed leverage, ignored if dyn_c > 0
    dyn_c: float = 0.0            # dynamic: L = 1 / (mmr + dyn_c * daily_vol), capped at lev_max
    lev_max: float = 20.0
    enter_apr: float = 0.10       # enter if trailing mean funding (annualised) >= this
    exit_apr: float = 0.02        # exit if it falls below this (hysteresis)
    lookback: int = 9             # number of 8h periods for the signal
    min_pos_frac: float = 0.66    # persistence: share of positive prints in the window
    k: int = 5                    # max simultaneous positions, equal capital slots
    hi: float = 1.5               # risk rule: rebalance when eff. leverage > hi * L
    delay: int = 1                # risk-rule reaction delay in 5m bars (1=5min, 3=15min, 12=1h)
    risk_rule: bool = True
    stress_slip: float = 3.0      # slippage multiplier for forced/risk trades
    idle_yield: bool = False      # idle cash earns T-bill rate (off = conservative)
    fees: Fees = field(default_factory=Fees)


class Market:
    """All per-symbol arrays aligned on the 5m grid + 1h fills + funding events."""

    def __init__(self, symbols: list[str], start: str, end: str):
        self.start, self.end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        self.grid = pd.date_range(self.start, self.end, freq="5min", tz="UTC")
        self.sym = []
        self.mark_hi, self.mark_cl, self.spot_px, self.fut_px = {}, {}, {}, {}
        self.fund, self.adv_s, self.adv_f, self.vol = {}, {}, {}, {}
        for s in symbols:
            m = D.load("mark", s, "5m")
            sp = D.load("spot", s, "1h")
            fu = D.load("fut", s, "1h")
            fr = D.load("funding", s)
            if not (len(m) and len(sp) and len(fu) and len(fr)):
                continue
            m = m.set_index("ts").reindex(self.grid)
            self.mark_hi[s] = m["high"].to_numpy()
            self.mark_cl[s] = m["close"].to_numpy()
            # fill price at the END of the hour bar that starts at tau: index by bar open
            self.spot_px[s] = sp.set_index("ts")["close"]
            self.fut_px[s] = fu.set_index("ts")["close"]
            self.adv_s[s] = sp.set_index("ts")["quote_volume"].rolling(24 * 30, min_periods=24).mean() * 24
            self.adv_f[s] = fu.set_index("ts")["quote_volume"].rolling(24 * 30, min_periods=24).mean() * 24
            lr = np.log(fu.set_index("ts")["close"]).diff()
            self.vol[s] = lr.rolling(24 * 30, min_periods=24 * 7).std() * np.sqrt(24)   # daily vol
            fr = fr[(fr["ts"] >= self.start) & (fr["ts"] <= self.end)].copy()
            fr["r8"] = fr["rate"] * 8.0 / fr["interval_h"].clip(lower=1)    # 8h-equivalent
            self.fund[s] = fr.set_index("ts")
            self.sym.append(s)
        self.gidx = {t: i for i, t in enumerate(self.grid)}

        self._sig: dict = {}

    def signal(self, lookback: int, taus: pd.DatetimeIndex):
        """(annualised mean 8h-equivalent funding, share of positive prints) over the
        last `lookback` 8h periods, as known at each tau (settled prints only)."""
        key = (lookback, taus[0], taus[-1], len(taus))
        if key not in self._sig:
            apr, pos = {}, {}
            win = f"{8 * lookback}h"
            for s in self.sym:
                fr = self.fund[s]
                if not len(fr):
                    continue
                a = fr["r8"].rolling(win).mean() * 1095
                f = (fr["rate"] > 0).astype(float).rolling(win).mean()
                last = pd.Series(fr.index, index=fr.index)
                idx = fr.index.searchsorted(taus, side="right") - 1
                ok = idx >= 0
                stale = np.full(len(taus), True)
                stale[ok] = (taus[ok] - fr.index[idx[ok]]) > pd.Timedelta(hours=8 * lookback)
                av = np.where(ok & ~stale, a.to_numpy()[np.clip(idx, 0, None)], np.nan)
                fv = np.where(ok & ~stale, f.to_numpy()[np.clip(idx, 0, None)], np.nan)
                apr[s], pos[s] = av, fv
            self._sig[key] = (pd.DataFrame(apr, index=taus), pd.DataFrame(pos, index=taus))
        return self._sig[key]

    def mark(self, s, b, fallback):
        a = self.mark_cl[s]
        b = min(max(b, 0), len(a) - 1)
        v = a[b]
        return v if np.isfinite(v) else fallback

    def mmr(self, s):
        return 0.005 if s in MAJORS else 0.0125

    def bar(self, t) -> int:
        return int((t - self.start) / pd.Timedelta(minutes=5))


def half_cost(adv: float, slip_mult: float, stress: float = 1.0) -> float:
    """Half-spread + slippage for one leg, from 30d average daily volume."""
    adv = max(adv if np.isfinite(adv) else 1e6, 1e6)
    return min(0.0001 + 0.5 * 0.0010 * np.sqrt(1e8 / adv), 0.005) * slip_mult * stress


@dataclass
class Pos:
    sym: str
    q: float
    f_ref: float
    margin: float
    lev: float
    opened: pd.Timestamp
    funding: float = 0.0
    costs: float = 0.0


class Result:
    def __init__(self, equity, events, fund_log, trades, stats):
        self.equity, self.events, self.fund_log, self.trades, self.stats = equity, events, fund_log, trades, stats


def simulate(mk: Market, universe: pd.DataFrame, p: Params, start=None, end=None, capital=10_000.0) -> Result:
    start = pd.Timestamp(start, tz="UTC") if start else mk.start + pd.Timedelta(days=31)
    end = pd.Timestamp(end, tz="UTC") if end else mk.end - pd.Timedelta(hours=2)
    taus = pd.date_range(start.ceil("8h"), end, freq="8h", tz="UTC")
    mem = {m: set(g["symbol"]) for m, g in universe.groupby("month")}
    SIG_A, SIG_F = mk.signal(p.lookback, taus)
    tb = pd.read_parquet(D.DATA / "stable_yields.parquet")["tbill"].ffill() / 100
    cash = capital
    pos: dict[str, Pos] = {}
    eq_rows, events, fund_log, trades = [], [], [], []
    stats = dict(liquidations=0, rebalances=0, fees=0.0, slippage=0.0, funding=0.0, neg_funding=0.0,
                 neg_funding_periods=0, held_periods=0, liq_loss=0.0, spot_delist_exits=0)

    def fill_px(s, t_exec_bar_open):
        sp = mk.spot_px[s].get(t_exec_bar_open, np.nan)
        fu = mk.fut_px[s].get(t_exec_bar_open, np.nan)
        return sp, fu

    def trade_cost(s, notional, t, stress=1.0):
        a_s = mk.adv_s[s].asof(t) if len(mk.adv_s[s]) else np.nan
        a_f = mk.adv_f[s].asof(t) if len(mk.adv_f[s]) else np.nan
        fee = notional * (p.fees.spot + p.fees.fut)
        slip = notional * (half_cost(a_s, p.fees.slip_mult, stress) + half_cost(a_f, p.fees.slip_mult, stress))
        stats["fees"] += fee
        stats["slippage"] += slip
        return fee + slip

    def target_lev(s, t):
        if p.dyn_c <= 0:
            return p.lev
        v = mk.vol[s].asof(t)
        if not np.isfinite(v):
            return 1.0
        return float(np.clip(1.0 / (mk.mmr(s) + p.dyn_c * v), 1.0, p.lev_max))

    def walk(ps: Pos, b0: int, b1: int, t_end) -> bool:
        """Walk position over 5m bars [b0, b1). Returns False if it was liquidated (and closed)."""
        nonlocal cash
        s = ps.sym
        hi_a, cl_a = mk.mark_hi[s], mk.mark_cl[s]
        fr = mk.fund[s]
        lo_i = fr.index.searchsorted(mk.grid[b0], side="right")
        hi_i = fr.index.searchsorted(mk.grid[min(b1, len(mk.grid) - 1)], side="right")
        f_ev = fr.iloc[lo_i:hi_i]
        f_bars = {mk.bar(t): r for t, r in zip(f_ev.index, f_ev["rate"])}
        b = b0
        mmr = mk.mmr(s)
        def pay(x):
            mpx = mk.mark(s, x, ps.f_ref)
            amt = ps.q * mpx * f_bars.pop(x)
            ps.margin += amt
            ps.funding += amt
            stats["funding"] += amt
            stats["held_periods"] += 1
            if amt < 0:
                stats["neg_funding"] += amt
                stats["neg_funding_periods"] += 1
            fund_log.append((mk.grid[x], s, amt / max(ps.q * mpx, 1e-12), amt, ps.lev))

        while b < b1:
            for x in sorted(k for k in f_bars if k < b):   # funding that fell inside a delay window
                pay(x)
            # next funding bar or end
            nxt = min([x for x in f_bars if x >= b] + [b1])
            seg_hi = hi_a[b:nxt + 1] if nxt < b1 else hi_a[b:b1]
            seg_hi = np.where(np.isnan(seg_hi), -np.inf, seg_hi)
            eq_hi = ps.margin + ps.q * (ps.f_ref - seg_hi)            # futures equity at worst price
            liq = eq_hi <= mmr * ps.q * seg_hi
            if p.risk_rule:
                trig = (ps.q * seg_hi) > p.hi * ps.lev * np.maximum(eq_hi, 1e-9)
                trig |= eq_hi <= 0
            else:
                trig = np.zeros_like(liq)
            i_liq = int(np.argmax(liq)) if liq.any() else None
            i_trg = int(np.argmax(trig)) if trig.any() else None
            if i_trg is not None and (i_liq is None or i_trg + p.delay < i_liq):
                bt = b + i_trg + p.delay                               # risk action fills after delay
                if bt >= len(cl_a) or np.isnan(cl_a[bt]):
                    bt = b + i_trg
                # if liquidation strikes between trigger and fill, it wins
                win_hi = hi_a[b + i_trg: bt + 1]
                win_hi = np.where(np.isnan(win_hi), -np.inf, win_hi)
                if np.any(ps.margin + ps.q * (ps.f_ref - win_hi) <= mmr * ps.q * win_hi):
                    i_liq = i_trg + int(np.argmax(ps.margin + ps.q * (ps.f_ref - win_hi) <= mmr * ps.q * win_hi))
                else:
                    _rebalance(ps, bt)
                    b = bt + 1
                    continue
            if i_liq is not None:
                bl = b + i_liq
                _liquidate(ps, bl)
                return False
            # no event before next funding: apply funding and continue
            if nxt < b1 and nxt in f_bars:
                pay(nxt)
            b = nxt + 1 if nxt < b1 else b1
        return True

    def _rebalance(ps: Pos, bt: int):
        s = ps.sym
        t = mk.grid[bt]
        mpx = mk.mark(s, bt, ps.f_ref)
        ratio = _spot_ratio(s, t)
        spx = mpx * ratio
        fut_eq = ps.margin + ps.q * (ps.f_ref - mpx)
        total = fut_eq + ps.q * spx
        q_new = max(total * ps.lev / ((ps.lev + 1) * spx), 0.0)
        c = trade_cost(s, abs(ps.q - q_new) * spx, t, p.stress_slip)
        ps.costs += c
        ps.q, ps.f_ref = q_new, mpx
        ps.margin = total - q_new * spx - c
        stats["rebalances"] += 1
        events.append((t, "rebalance", s, round(ps.lev, 2)))

    def _liquidate(ps: Pos, bl: int):
        nonlocal cash
        s = ps.sym
        t = mk.grid[bl]
        mmr = mk.mmr(s)
        p_liq = (ps.margin + ps.q * ps.f_ref) / (ps.q * (1 + mmr))
        eq_before = ps.margin + ps.q * (ps.f_ref - p_liq) + ps.q * p_liq * _spot_ratio(s, t)
        # futures equity lost; sell the unhedged spot `delay` bars later, stressed
        bs = min(bl + p.delay, len(mk.grid) - 1)
        mpx = mk.mark(s, bs, p_liq)
        spx = mpx * _spot_ratio(s, mk.grid[bs])
        a_s = mk.adv_s[s].asof(t)
        c = ps.q * spx * (p.fees.spot + half_cost(a_s, p.fees.slip_mult, p.stress_slip))
        proceeds = ps.q * spx - c
        stats["fees"] += ps.q * spx * p.fees.spot
        cash += proceeds
        stats["liquidations"] += 1
        stats["liq_loss"] += eq_before - proceeds
        events.append((t, "LIQUIDATION", s, round(ps.lev, 2), round(eq_before - proceeds, 2)))
        trades.append((t, s, "liquidation", ps.q, spx, c))

    ratio_cache: dict = {}

    def _spot_ratio(s, t):
        """spot/perp ratio from the last COMPLETED hour before t (no look-ahead), clipped to
        +-5% so data glitches around listings/delistings cannot create fake basis profits."""
        h = t.floor("h") - pd.Timedelta(hours=1)
        key = (s, h)
        if key not in ratio_cache:
            sp, fu = mk.spot_px[s].get(h, np.nan), mk.fut_px[s].get(h, np.nan)
            r = sp / fu if np.isfinite(sp) and np.isfinite(fu) and fu > 0 else 1.0
            ratio_cache[key] = float(np.clip(r, 0.95, 1.05))
        return ratio_cache[key]

    def spot_stale(s, t) -> bool:
        """No spot print in the last 3 hours -> spot market halted/delisted."""
        sp = mk.spot_px[s]
        i = sp.index.searchsorted(t - pd.Timedelta(hours=1), side="right") - 1
        return i < 0 or (t - pd.Timedelta(hours=1) - sp.index[i]) > pd.Timedelta(hours=3)

    def force_close_delisted(ps: Pos, t):
        """Spot leg can no longer be sold on Binance: assume it is sold elsewhere at the last
        spot price minus 10%; the perp is bought back at mark."""
        nonlocal cash
        s = ps.sym
        sp = mk.spot_px[s]
        i = sp.index.searchsorted(t, side="right") - 1
        last_spot = float(sp.iloc[max(i, 0)])
        fu = mk.mark(s, mk.bar(t), ps.f_ref)
        c = trade_cost(s, ps.q * fu, t, p.stress_slip)
        val = ps.margin + ps.q * (ps.f_ref - fu) + ps.q * last_spot * 0.90 - c
        cash += val
        stats["spot_delist_exits"] = stats.get("spot_delist_exits", 0) + 1
        events.append((t, "spot_delisted_exit", s, round(ps.lev, 2)))
        trades.append((t, s, "close_spot_delisted", ps.q, last_spot * 0.9, c))

    def close(ps: Pos, t_bar_open, reason):
        nonlocal cash
        s = ps.sym
        sp, fu = fill_px(s, t_bar_open)
        t = t_bar_open + pd.Timedelta(hours=1)
        if not (np.isfinite(sp) and np.isfinite(fu)):
            b = mk.bar(t) - 1
            fu = mk.mark_cl[s][b] if b < len(mk.grid) and np.isfinite(mk.mark_cl[s][b]) else ps.f_ref
            sp = fu * _spot_ratio(s, t)
        c = trade_cost(s, ps.q * sp, t, 1.0 if reason == "signal" else p.stress_slip)
        # conservative: if the real prints diverge abnormally, take the worse of real vs clipped basis
        sp_clip = fu * float(np.clip(sp / fu, 0.95, 1.05))
        sp = min(sp, sp_clip)
        val = ps.margin + ps.q * (ps.f_ref - fu) + ps.q * sp - c
        cash += val
        trades.append((t, s, f"close_{reason}", ps.q, sp, c))

    for i, tau in enumerate(taus):
        t_exec_open = tau                       # 1h bar [tau, tau+1h) -> fill at its close
        t_exec = tau + pd.Timedelta(hours=1)
        b_exec = mk.bar(t_exec)
        # 1. walk open positions up to this fill
        if i > 0:
            b_prev = mk.bar(taus[i - 1] + pd.Timedelta(hours=1))
            for s in list(pos):
                if not walk(pos[s], b_prev, b_exec, t_exec):
                    del pos[s]
                elif spot_stale(s, t_exec):
                    force_close_delisted(pos.pop(s), t_exec)
        # 2. equity
        val = cash
        for s, ps in pos.items():
            b = min(b_exec, len(mk.grid) - 1)
            mpx = mk.mark(s, b, ps.f_ref)
            val += ps.margin + ps.q * (ps.f_ref - mpx) + ps.q * mpx * _spot_ratio(s, t_exec)
        if p.idle_yield and i > 0:
            r = tb.asof(tau.normalize()) if len(tb) else 0.0
            add = cash * (r if np.isfinite(r) else 0.0) / 1095
            cash += add
            val += add
        eq_rows.append((t_exec, val, len(pos), sum(ps.q * ps.f_ref for ps in pos.values()) / max(val, 1e-9)))
        if val <= 0:
            break
        # 3. signals at tau
        month = pd.Timestamp(tau.year, tau.month, 1, tz="UTC")
        members = mem.get(month, set())
        ra, rf = SIG_A.iloc[i], SIG_F.iloc[i]
        sig = {s: (ra[s], rf[s]) for s in ra.index if np.isfinite(ra[s])}
        # 4. exits
        for s in list(pos):
            apr = sig.get(s, (-1, 0))[0]
            if apr < p.exit_apr or s not in members and apr < p.enter_apr:
                close(pos.pop(s), t_exec_open, "signal")
        # 5. entries (best funding first), equal capital slots of current equity
        slot = val / p.k
        cand = sorted([(a, s) for s, (a, f) in sig.items()
                       if s in members and s not in pos and a >= p.enter_apr and f >= p.min_pos_frac],
                      reverse=True)
        for a, s in cand:
            if len(pos) >= p.k or cash < slot * 0.5:
                break
            sp, fu = fill_px(s, t_exec_open)
            if not (np.isfinite(sp) and np.isfinite(fu)) or not np.isfinite(mk.mark_cl[s][min(b_exec, len(mk.grid) - 1)]):
                continue
            if abs(sp / fu - 1) > 0.05:      # abnormal market (listing/delisting/squeeze): skip
                continue
            A = min(slot, cash)
            L = target_lev(s, tau)
            N = A * L / (L + 1)
            c = trade_cost(s, N, t_exec)
            q = N / sp
            margin = A - N - c
            cash -= A
            pos[s] = Pos(s, q, fu, margin, L, t_exec, costs=c)
            trades.append((t_exec, s, "open", q, sp, c))
    # close everything at the end for a clean final value
    eq = pd.DataFrame(eq_rows, columns=["ts", "equity", "n_pos", "gross"]).set_index("ts")
    return Result(eq, events, pd.DataFrame(fund_log, columns=["ts", "symbol", "rate", "amount", "lev"]),
                  pd.DataFrame(trades, columns=["ts", "symbol", "kind", "q", "price", "cost"]), stats)
