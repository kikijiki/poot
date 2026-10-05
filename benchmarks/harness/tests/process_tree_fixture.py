#!/usr/bin/env python3
"""Model-free process fixtures for observer lifecycle tests."""

import argparse
import os
import signal
import sys
import time


def write_pids(path, *pids):
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as f:
        f.write(" ".join(str(pid) for pid in pids))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def fork_and_sleep(pid_file, ignore_term):
    # The PID record is the tests' readiness signal, so it is written only once the descendant is fully set
    # up: it tells the parent through this pipe after its signal disposition and its stdout line are in place.
    ready_read, ready_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("fixture descendant holds stdout open", flush=True)
        os.write(ready_write, b"1")
        os.close(ready_write)
        while True:
            time.sleep(60)

    os.close(ready_write)
    if os.read(ready_read, 1) != b"1":
        raise SystemExit("fixture descendant died before it was ready")
    os.close(ready_read)
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    write_pids(pid_file, os.getpid(), child, os.getpgrp(), os.getsid(0))
    print("fixture parent ready", file=sys.stderr, flush=True)
    while True:
        time.sleep(60)


def succeed_and_leave_child(pid_file):
    ready_read, ready_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("successful fixture descendant holds stdout open", flush=True)
        os.write(ready_write, b"1")
        os.close(ready_write)
        while True:
            time.sleep(60)

    os.close(ready_write)
    if os.read(ready_read, 1) != b"1":
        raise SystemExit("fixture descendant died before it was ready")
    os.close(ready_read)
    write_pids(pid_file, os.getpid(), child, os.getpgrp(), os.getsid(0))
    print("success progress", file=sys.stderr, flush=True)
    print('{"framework":"fixture","status":"ok"}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["fork-hold", "ignore-term", "success-child", "fail", "success"]
    )
    parser.add_argument("--pid-file")
    args = parser.parse_args()

    if args.mode in ("fork-hold", "ignore-term"):
        if not args.pid_file:
            parser.error("--pid-file is required for process-tree modes")
        fork_and_sleep(args.pid_file, ignore_term=args.mode == "ignore-term")
    elif args.mode == "success-child":
        if not args.pid_file:
            parser.error("--pid-file is required for success-child mode")
        succeed_and_leave_child(args.pid_file)
    elif args.mode == "fail":
        print("ordinary runner failure", file=sys.stderr, flush=True)
        raise SystemExit(7)
    else:
        print("success progress", file=sys.stderr, flush=True)
        print('{"framework":"fixture","status":"ok"}', flush=True)


if __name__ == "__main__":
    main()
