"""Small PNG figures for the README (docs/img/). Reads files under reports/ (+ local yield data if present)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from farb import data as D  # noqa: E402

REPORTS = D.ROOT / "reports"
OUT = D.ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})


def leverage():
    t = pd.read_csv(REPORTS / "leverage_table.csv")
    fixed = ["1x", "2x", "3x", "5x", "8x", "10x", "15x", "20x"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for strat, c in (("BTC+ETH", "#2f5fd0"), ("Altcoin seçimi", "#c0392b")):
        s = t[t.strateji == strat].set_index("kaldirac").loc[fixed]
        lab = "BTC+ETH carry" if strat == "BTC+ETH" else "Altcoin selection carry"
        axes[0].plot(fixed, s["WF_CAGR"], marker="o", color=c, label=lab)
        axes[1].plot(fixed, s["WF_tasfiye"], marker="o", color=c, label=f"{lab} (15-min delay)")
        axes[1].plot(fixed, s["gecikme60_tasfiye"], marker="x", ls="--", color=c, alpha=0.6, label=f"{lab} (60-min delay)")
    w = json.loads((REPORTS / "holdout.json").read_text())["wf_benchmarks"]
    axes[0].axhline(100 * w["tbill_3m"]["cagr"], color="#888888", ls="--", lw=1, label="3M T-bill")
    axes[0].set_ylabel("Net CAGR, % (walk-forward 2022-01 .. 2025-09)")
    axes[0].set_xlabel("Futures leverage")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_yscale("symlog")
    axes[1].set_ylabel("Liquidations")
    axes[1].set_xlabel("Futures leverage")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("More leverage stops paying after ~8x; liquidations explode")
    fig.tight_layout()
    fig.savefig(OUT / "leverage.png")
    plt.close(fig)


def regime():
    e = json.loads((REPORTS / "exchanges.json").read_text())["universe_monthly"]
    m = pd.DataFrame(e).T
    m.index = pd.to_datetime(m.index)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(m.index, 100 * m["median"], color="#2f5fd0", lw=1.6, label="Median annualised funding, top-30 perps (Binance)")
    try:
        y = pd.read_parquet(D.DATA / "stable_yields.parquet")["tbill"].resample("MS").mean()
        ax.plot(y.index.tz_localize(None) if y.index.tz else y.index, y.values, color="#888888", ls="--", lw=1.2, label="3M T-bill")
    except Exception:
        pass
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-20, 60)
    ax.annotate("early-2021 peaks of 60-130% off-scale", xy=(0.01, 0.95), xycoords="axes fraction", fontsize=7, color="#555")
    ax.set_ylabel("% per year")
    ax.set_title("Funding carry has been arbitraged away (crowding)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "funding_regime.png")
    plt.close(fig)


if __name__ == "__main__":
    leverage()
    regime()
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")
