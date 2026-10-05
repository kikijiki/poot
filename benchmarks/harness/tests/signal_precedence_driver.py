#!/usr/bin/env python3
"""Prove deferred SIGTERM wins over a concurrent observer cleanup failure."""

import os
from pathlib import Path
import signal
import sys


HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import observer  # noqa: E402

# A bound for a wait that must succeed; a passing run never reaches it.
UPPER_BOUND_S = 120.0


def main():
    fixture = Path(__file__).with_name("process_tree_fixture.py")
    finish_sampler = observer._finish_sampler

    def finish_then_interrupt(*args, **kwargs):
        finish_sampler(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        raise RuntimeError("concurrent cleanup failure must not replace SIGTERM")

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    observer._finish_sampler = finish_then_interrupt
    observer.run_observed(
        [sys.executable, str(fixture), "success"],
        timeout_s=UPPER_BOUND_S,
        interval_s=0.01,
        reader_join_timeout_s=UPPER_BOUND_S,
        sampler_join_timeout_s=UPPER_BOUND_S,
    )


if __name__ == "__main__":
    main()
