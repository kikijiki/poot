"""The host load a cell ran under, recorded for every engine and refused when it was not quiet (POOT-1523).

A short measurement on the host's launch path is a CPU-contention measurement as much as an engine one. Flash-Next on
llama.cpp ROCm took 400 ms for 128 tokens on a quiet Strix Halo, 432 ms with 12 of 32 CPUs busy elsewhere and 545 ms
with all 32 (Vulkan: 345, 378, 431 ms); where the busy work ran (same CCD, other CCD, pinned or not) changed nothing.
Three published repeats of that cell spread 376 to 630 ms because gates of another lane ran on the box meanwhile.

So the harness meters, for every cell and whichever engine it runs, the CPU time that user processes outside the
cell's process tree (the harness and every descendant) consume, as a share of the machine per 0.5 s window: the
"foreign" CPU. Kernel threads are not counted: the kworkers that read and decrypt a mapped weights file run because
the cell asked (a first version that counted them saw 7 % foreign on a box with nothing else running). The cell's row
records the windows' mean and maximum, and the load average (informational: it counts the cell's own threads). A
cell with a window over `FOREIGN_SHARE_LIMIT` is discarded and rerun, and the discards are counted in the row.

The limit follows from the dose-response above. TTFT@128 on the more sensitive backend moved about 0.35 % per
percentage point of foreign share (+35 % at 100 %, +9 % at 25 %); the contract's noise band is 10 %. A share that
moves the number by half the band is 5 % / 0.35 = 14 %, rounded down to 10 %. It is derived from the band, not tuned
to a cell, and the same for every framework and backend. An idle box measured a mean of 0.9 % and a maximum of
1.8 % per window.

Pure stdlib, GPU-free. Where `/proc` has no CPU list (not Linux) the row says so (`available: false`) and the
cell is not gated: that is a dev machine, not a benchmark host.
"""

import os
import threading
import time
from pathlib import Path

# A window with more foreign CPU than this share of the machine is not quiet.
FOREIGN_SHARE_LIMIT = 0.10
SAMPLE_INTERVAL_S = 0.5
# A cell is discarded and rerun at most this many times before it is reported as not measurable.
MAX_DISCARDS = 3

PF_KTHREAD = 0x00200000


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def cpu_count(proc_root="/proc"):
    text = _read(Path(proc_root) / "stat")
    if text is None:
        return None
    cpus = sum(1 for line in text.splitlines() if line.startswith("cpu") and line[3:4].isdigit())
    return cpus or None


def snapshot(proc_root="/proc"):
    """Every user process now, as {pid: (ppid, ticks, kernel thread?)}; ticks are user + system time of all its threads.

    Kernel threads are listed but never counted as foreign: the kworkers that read a mapped weights file or
    decrypt its blocks run because the cell asked, and they cannot be told from work done for somebody else."""
    out = {}
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return out
    for entry in entries:
        if not entry.name.isdigit():
            continue
        text = _read(entry / "stat")
        if not text or ")" not in text:
            continue
        rest = text[text.rindex(")") + 2:].split()
        try:
            out[int(entry.name)] = (int(rest[1]), int(rest[11]) + int(rest[12]), bool(int(rest[6]) & PF_KTHREAD))
        except (IndexError, ValueError):
            continue
    return out


def tree_pids(processes, root_pid):
    """`root_pid` and every process under it in a snapshot."""
    members, grew = {root_pid}, True
    while grew:
        grew = False
        for pid, (ppid, _, _) in processes.items():
            if ppid in members and pid not in members:
                members.add(pid)
                grew = True
    return members


def foreign_ticks(before, after, root_pid):
    """CPU ticks that user processes outside `root_pid`'s tree used between two snapshots. A process that ended in the
    window is counted up to its last sample, so a burst of very short processes is under-counted, never over-counted."""
    inside = tree_pids(after, root_pid) | tree_pids(before, root_pid)
    total = 0
    for pid, (_, ticks, kthread) in after.items():
        if kthread or pid in inside:
            continue
        total += max(0, ticks - before.get(pid, (0, 0, False))[1])
    return total


def loadavg(proc_root="/proc"):
    text = _read(Path(proc_root) / "loadavg")
    return float(text.split()[0]) if text else None


def foreign_share(ticks, cpus, seconds, clk_tck):
    """The share of the machine (0..1) that `ticks` of foreign CPU time are in a window."""
    capacity = cpus * seconds * clk_tck
    return ticks / capacity if capacity > 0 else 0.0


def verdict(windows, limit=FOREIGN_SHARE_LIMIT):
    """Summary of foreign-share windows; `quiet` iff none exceeded `limit`."""
    if not windows:
        return {"available": False}
    over = sum(1 for w in windows if w > limit)
    return {"available": True, "windows": len(windows), "foreign_share_mean": round(sum(windows) / len(windows), 4),
            "foreign_share_max": round(max(windows), 4), "windows_over_limit": over, "limit": limit,
            "quiet": over == 0}


class LoadMeter:
    """Meters the foreign CPU share of a cell on a sampling thread: `with LoadMeter() as m: ...; m.result()`.

    The cell's tree is this process and all its descendants, so the runner, the engine and the observer's sampler
    are the cell; anything else running is foreign.
    """

    def __init__(self, root_pid=None, proc_root="/proc", interval_s=SAMPLE_INTERVAL_S, limit=FOREIGN_SHARE_LIMIT):
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self.proc_root = proc_root
        self.interval_s = interval_s
        self.limit = limit
        self.windows = []
        self.loadavg_start = loadavg(proc_root)
        self._stop = threading.Event()
        self._thread = None
        self._clk_tck = os.sysconf("SC_CLK_TCK")

    def _point(self):
        return time.monotonic(), snapshot(self.proc_root)

    def _run(self, last):
        cpus = cpu_count(self.proc_root)
        while cpus and last[1] and not self._stop.wait(self.interval_s):
            now = self._point()
            self.windows.append(foreign_share(foreign_ticks(last[1], now[1], self.root_pid), cpus, now[0] - last[0],
                                              self._clk_tck))
            last = now

    def __enter__(self):
        # The baseline is taken before the cell can fork, so no process starts between its two reads.
        self._thread = threading.Thread(target=self._run, args=(self._point(),), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def result(self):
        out = verdict(self.windows, self.limit)
        out["loadavg_start"] = self.loadavg_start
        out["loadavg_end"] = loadavg(self.proc_root)
        return out


def probe(seconds, proc_root="/proc", limit=FOREIGN_SHARE_LIMIT, interval_s=SAMPLE_INTERVAL_S):
    """Is the box quiet right now? Meters `seconds` of idle: the verdict. The load average is recorded but does not
    gate: it counts the cell's own threads, so the previous repeat of a sweep would make the next one look loaded."""
    with LoadMeter(proc_root=proc_root, interval_s=interval_s, limit=limit) as meter:
        time.sleep(seconds)
    out = meter.result()
    out["quiet"] = out.get("quiet", True)
    return out


def wait_quiet(max_wait_s, probe_s=3.0, sleep=time.sleep, now=time.monotonic, probe_fn=probe):
    """Probe until the box is quiet or `max_wait_s` has passed; returns (quiet, waited seconds, last probe)."""
    start = now()
    while True:
        result = probe_fn(probe_s)
        waited = now() - start
        if result["quiet"]:
            return True, waited, result
        if waited >= max_wait_s:
            return False, waited, result
        sleep(2.0)
