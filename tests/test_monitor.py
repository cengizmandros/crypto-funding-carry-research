import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("monitor", __file__.replace("tests/test_monitor.py", "scripts/monitor.py"))
mon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mon)


def df(vals, start="2026-01-01", gaps=()):
    dates = [d for d in pd.date_range(start, periods=len(vals) + len(gaps)) if str(d.date()) not in gaps]
    return pd.DataFrame({"date": [str(d.date()) for d in dates[: len(vals)]], "median_apr": vals})


def test_streak_counts_consecutive_days_above_threshold():
    assert mon.streak_of(df([0.05] + [0.12] * 14)) == 14
    assert mon.streak_of(df([0.12] * 5 + [0.08])) == 0
    assert mon.streak_of(df([0.10] * 3)) == 0          # must be strictly above 10%


def test_missing_day_breaks_streak():
    d = df([0.2] * 10, gaps=("2026-01-05",))
    assert mon.streak_of(d) == 6          # 01-06..01-11 after the 01-05 gap
