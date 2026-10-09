from __future__ import annotations

import numpy as np
import pandas as pd


def summary(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("total", "cagr", "vol", "sharpe", "max_dd", "worst_month", "best_month", "pos_months")}
    eq = (1 + r).cumprod()
    m = (1 + r).resample("ME").prod() - 1
    yrs = len(r) / 365
    g = float(eq.iloc[-1])
    return {
        "start": str(r.index.min().date()), "end": str(r.index.max().date()),
        "total": g - 1,
        "cagr": g ** (1 / yrs) - 1 if g > 0 else -1.0,
        "vol": float(r.std() * np.sqrt(365)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        "worst_month": float(m.min()), "best_month": float(m.max()),
        "pos_months": float((m > 0).mean()),
        "monthly": {str(k.date())[:7]: round(float(v), 5) for k, v in m.items()},
        "yearly": {str(k.year): round(float(v), 5) for k, v in ((1 + r).resample("YE").prod() - 1).items()},
    }
