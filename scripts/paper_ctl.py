"""Funding-arb PAPER trading control:  risk | funding | status"""
from __future__ import annotations

import logging
import sys
from contextlib import closing

import pandas as pd

from farb import paper
from farb.data import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    handlers=[logging.FileHandler(ROOT / "logs" / "paper.log"), logging.StreamHandler()])
log = logging.getLogger("fa_ctl")


def main(cmd: str) -> int:
    paper.init()
    try:
        if cmd == "risk":
            paper.risk_check()
        elif cmd == "funding":
            paper.funding_cycle()
            paper.risk_check()
        elif cmd == "status":
            with closing(paper.db()) as con:
                print(pd.read_sql("SELECT * FROM snapshots ORDER BY ts DESC LIMIT 3", con).to_string())
                print(pd.read_sql("SELECT * FROM positions", con).to_string())
                print(pd.read_sql("SELECT * FROM funding ORDER BY ts DESC LIMIT 10", con).to_string())
                print(pd.read_sql("SELECT * FROM events ORDER BY ts DESC LIMIT 10", con).to_string())
        else:
            print(__doc__)
            return 2
    except Exception as e:
        log.exception("%s failed", cmd)
        with closing(paper.db()) as con:
            paper.ev(con, f"error_{cmd}", repr(e))
            con.commit()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "help"))
