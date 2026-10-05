#!/usr/bin/env python3
"""Run an observed process tree so its parent test can interrupt the observer with SIGTERM."""

from pathlib import Path
import signal
import sys
import threading


HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import observer  # noqa: E402
from observer import run_observed  # noqa: E402

# The observer waits this long after signalling its process group before it escalates.
CLEANUP_GRACE_S = 0.2
UPPER_BOUND_S = 120.0


def main():
    pid_file = Path(sys.argv[1])
    cleanup_marker = Path(sys.argv[2])
    fixture = Path(__file__).with_name("process_tree_fixture.py")
    signal_process_group = observer._signal_process_group
    held = []

    def signal_group_and_hold_cleanup_for_second_sigterm(pgid, signum):
        """Once cleanup has signalled the group, tell the test, then stay in cleanup until a second SIGTERM
        has arrived, so the second signal always lands during cleanup and not after it."""
        signal_process_group(pgid, signum)
        if signum != signal.SIGTERM or held:
            return
        held.append(pgid)  # cleanup signals the runner's group, then the sampler's: hold at the first only
        observers_handler = signal.getsignal(signal.SIGTERM)
        second_sigterm = threading.Event()

        def note_second_sigterm(received, frame):
            observers_handler(received, frame)
            second_sigterm.set()

        signal.signal(signal.SIGTERM, note_second_sigterm)
        try:
            cleanup_marker.write_text("terminating", encoding="ascii")
            if not second_sigterm.wait(UPPER_BOUND_S):
                raise RuntimeError("the second SIGTERM never arrived")
        finally:
            signal.signal(signal.SIGTERM, observers_handler)

    observer._signal_process_group = signal_group_and_hold_cleanup_for_second_sigterm
    run_observed(
        [sys.executable, str(fixture), "fork-hold", "--pid-file", str(pid_file)],
        timeout_s=UPPER_BOUND_S,
        interval_s=0.01,
        termination_grace_s=CLEANUP_GRACE_S,
        reader_join_timeout_s=UPPER_BOUND_S,
        sampler_join_timeout_s=UPPER_BOUND_S,
    )


if __name__ == "__main__":
    main()
