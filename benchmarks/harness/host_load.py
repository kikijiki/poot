"""The host load a cell ran under, recorded for every engine and refused when it was not quiet (POOT-1523).

A short measurement on the host's launch path is a CPU-contention measurement as much as an engine one. Flash-Next on
llama.cpp ROCm took 400 ms for 128 tokens on a quiet Strix Halo, 432 ms with 12 of 32 CPUs busy elsewhere and 545 ms
with all 32 (Vulkan: 345, 378, 431 ms); where the busy work ran (same CCD, other CCD, pinned or not) changed nothing.
Three published repeats of that cell spread 376 to 630 ms because gates of another lane ran on the box meanwhile.

So the harness meters, for every cell and whichever engine it runs, the CPU time that user processes outside the
cell's process tree (the harness and every descendant) consume, as a share of the machine per 1 s window: the
"foreign" CPU. Kernel threads are not counted: the kworkers that read and decrypt a mapped weights file run because
the cell asked (a first version that counted them saw 7 % foreign on a box with nothing else running). The cell's row
records the windows' mean and maximum, and the load average (informational: it counts the cell's own threads). A
cell with a window over `FOREIGN_SHARE_LIMIT` is discarded and rerun, and the discards are counted in the row.

The limit follows from the dose-response above. TTFT@128 on the more sensitive backend moved about 0.35 % per
percentage point of foreign share (+35 % at 100 %, +9 % at 25 %); the contract's noise band is 10 %. A share that
moves the number by half the band is 5 % / 0.35 = 14 %, rounded down to 10 %. It is derived from the band, not tuned
to a cell, and the same for every framework and backend. An idle box measured a mean of 0.9 % and a maximum of
1.8 % per window.

The GPU is metered the same way (POOT-1508). The 2026-11-07 Strix Halo sweep had two unverified cells whose slow
repeats line up with desktop apps starting mid-cell on the shared iGPU, and `gpu_busy_percent` read once before a
cell sees none of that. The meter reads, once per window (1 Hz, a few sysfs reads), the device's `gpu_busy_percent`
and the engine time every foreign process holds in `/proc/<pid>/fdinfo` (`drm-engine-*`, the kernel's per-client
accounting); the cell's own tree is excluded as for the CPU. A foreign client that holds the engines for a share of
a window takes that share from a cell that saturates them (a slope of at most 1), so half the 10 % band, 5 %, is
the limit (`FOREIGN_GPU_SHARE_LIMIT`). A window over either limit makes the repeat `contended`. A CPU-contended
attempt is discarded and rerun as before; a GPU-contended one is kept, marked, and named by the comparison, because
the box cannot be asked to wait for a desktop app to stop. The row records the busy series, the top consumers by
process name in the first and last window, and every foreign process that appeared during the repeat.

Pure stdlib, GPU-free. Where `/proc` has no CPU list (not Linux) the row says so (`available: false`) and the
cell is not gated: that is a dev machine, not a benchmark host.
"""

import os
import threading
import time
from collections import namedtuple
from pathlib import Path

# A window with more foreign CPU than this share of the machine is not quiet.
FOREIGN_SHARE_LIMIT = 0.10
# The same dose-response argument for the GPU: a foreign client holding the engines for a share s of a window costs a
# saturating cell at most s of its throughput (slope 1); half the 10 % noise band is 5 %.
FOREIGN_GPU_SHARE_LIMIT = 0.05
SAMPLE_INTERVAL_S = 1.0
# Consumers listed in a row at the start and the end of a repeat.
TOP_CONSUMERS = 3
# Foreign processes listed as started during a repeat (the row stays small on a box that forks a lot).
MAX_STARTED = 10
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


Proc = namedtuple("Proc", "ppid ticks kthread comm started")


def snapshot(proc_root="/proc"):
    """Every user process now, as {pid: Proc}; ticks are user + system time of all its threads, `started` the
    process start time in ticks (a reused pid is a different process).

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
        comm = text[text.index("(") + 1:text.rindex(")")]
        try:
            started = int(rest[19]) if len(rest) > 19 else 0
            out[int(entry.name)] = Proc(int(rest[1]), int(rest[11]) + int(rest[12]), bool(int(rest[6]) & PF_KTHREAD),
                                        comm, started)
        except (IndexError, ValueError):
            continue
    return out


def tree_pids(processes, root_pid):
    """`root_pid` and every process under it in a snapshot."""
    members, grew = {root_pid}, True
    while grew:
        grew = False
        for pid, proc in processes.items():
            if proc.ppid in members and pid not in members:
                members.add(pid)
                grew = True
    return members


def foreign_ticks_by_pid(before, after, root_pid):
    """{pid: CPU ticks} that user processes outside `root_pid`'s tree used between two snapshots. A process that ended
    in the window is counted up to its last sample, so a burst of very short processes is under-counted, never
    over-counted."""
    inside = tree_pids(after, root_pid) | tree_pids(before, root_pid)
    out = {}
    for pid, proc in after.items():
        if proc.kthread or pid in inside:
            continue
        prior = before.get(pid)
        used = proc.ticks - (prior.ticks if prior and prior.started == proc.started else 0)
        if used > 0:
            out[pid] = used
    return out


def foreign_ticks(before, after, root_pid):
    """CPU ticks that user processes outside `root_pid`'s tree used between two snapshots."""
    return sum(foreign_ticks_by_pid(before, after, root_pid).values())


def gpu_engine_ns(proc_root="/proc"):
    """{pid: nanoseconds of GPU engine time the process's DRM clients have used so far}, from the kernel's per-client
    accounting in `/proc/<pid>/fdinfo` (`drm-engine-<name>: <n> ns`). A client shared by several descriptors counts
    once (`drm-client-id`); a process whose fdinfo cannot be read is absent, not zero."""
    out = {}
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return out
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            files = list((entry / "fdinfo").iterdir())
        except OSError:
            continue
        clients = {}
        for f in files:
            text = _read(f)
            if not text or "drm-engine-" not in text:
                continue
            client, total = None, 0
            for line in text.splitlines():
                key, _, value = line.partition(":")
                if key == "drm-client-id":
                    client = value.strip()
                elif key.startswith("drm-engine-") and value.strip().endswith(" ns"):
                    try:
                        total += int(value.split()[0])
                    except ValueError:
                        pass
            clients[client if client is not None else f.name] = total
        if clients:
            out[int(entry.name)] = sum(clients.values())
    return out


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


def default_gpu():
    """The first amdgpu card's sysfs reader (`util_pct()`), or None where there is none (not an AMD host)."""
    try:
        from observer import _amdgpu_device
    except ImportError:  # pragma: no cover - the harness directory is always on the path
        return None
    return _amdgpu_device(0)


class LoadMeter:
    """Meters the foreign load of a cell on a sampling thread: `with LoadMeter() as m: ...; m.result()`.

    The cell's tree is this process and all its descendants, so the runner, the engine and the observer's sampler
    are the cell; anything else running is foreign. Once per `interval_s` the meter takes one process snapshot and
    one read of every process's DRM engine time (and the device's `gpu_busy_percent`), which is all the contention
    record is made of: nothing is traced and the cell is not perturbed.
    """

    def __init__(self, root_pid=None, proc_root="/proc", interval_s=SAMPLE_INTERVAL_S, limit=FOREIGN_SHARE_LIMIT,
                 gpu_limit=FOREIGN_GPU_SHARE_LIMIT, gpu=None):
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self.proc_root = proc_root
        self.interval_s = interval_s
        self.limit = limit
        self.gpu_limit = gpu_limit
        self.gpu = default_gpu() if gpu is None else gpu
        self.windows = []
        self.gpu_windows = []
        self.busy = [] if self.gpu is None else [self.gpu.util_pct()]  # [0] is the reading before the repeat
        self.consumers = []  # per window: {comm: {"cpu": share, "gpu": share}}
        self.started = {}  # comm -> {"count": processes, "cpu_share_max": ..., "gpu_share_max": ...}
        self.contenders = {}  # comm -> the largest cpu / gpu share a window saw it hold above a limit
        self.loadavg_start = loadavg(proc_root)
        self._stop = threading.Event()
        self._thread = None
        self._clk_tck = os.sysconf("SC_CLK_TCK")

    def _point(self):
        return time.monotonic(), snapshot(self.proc_root), gpu_engine_ns(self.proc_root)

    def _window(self, last, now, cpus, seen):
        seconds = now[0] - last[0]
        inside = tree_pids(now[1], self.root_pid) | tree_pids(last[1], self.root_pid)
        by_pid = foreign_ticks_by_pid(last[1], now[1], self.root_pid)
        self.windows.append(foreign_share(sum(by_pid.values()), cpus, seconds, self._clk_tck))
        gpu_by_pid = {pid: max(0, ns - last[2].get(pid, 0)) / 1e9 / seconds
                      for pid, ns in now[2].items() if pid not in inside and seconds > 0}
        self.gpu_windows.append(sum(gpu_by_pid.values()))
        if self.gpu is not None:
            self.busy.append(self.gpu.util_pct())
        window = {}
        for pid in by_pid.keys() | gpu_by_pid.keys():
            proc = now[1].get(pid)
            if proc is None:
                continue
            entry = window.setdefault(proc.comm, {"cpu": 0.0, "gpu": 0.0})
            entry["cpu"] += foreign_share(by_pid.get(pid, 0), cpus, seconds, self._clk_tck)
            entry["gpu"] += gpu_by_pid.get(pid, 0.0)
        self.consumers.append(window)
        for comm, use in window.items():
            if use["cpu"] > self.limit or use["gpu"] > self.gpu_limit:
                held = self.contenders.setdefault(comm, {"cpu_share_max": 0.0, "gpu_share_max": 0.0})
                held["cpu_share_max"] = max(held["cpu_share_max"], use["cpu"])
                held["gpu_share_max"] = max(held["gpu_share_max"], use["gpu"])
        for pid, proc in now[1].items():
            if proc.kthread or pid in inside or (pid, proc.started) in seen:
                continue
            seen.add((pid, proc.started))
            born = self.started.setdefault(proc.comm, {"count": 0, "cpu_share_max": 0.0, "gpu_share_max": 0.0})
            born["count"] += 1
            use = window.get(proc.comm, {"cpu": 0.0, "gpu": 0.0})
            born["cpu_share_max"] = max(born["cpu_share_max"], use["cpu"])
            born["gpu_share_max"] = max(born["gpu_share_max"], use["gpu"])

    def _run(self, last):
        cpus = cpu_count(self.proc_root)
        seen = {(pid, proc.started) for pid, proc in last[1].items()}
        while cpus and last[1] and not self._stop.wait(self.interval_s):
            now = self._point()
            self._window(last, now, cpus, seen)
            last = now

    def __enter__(self):
        # The baseline is taken before the cell can fork, so no process starts between its two reads.
        self._thread = threading.Thread(target=self._run, args=(self._point(),), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _top(self, window):
        ranked = sorted(window.items(), key=lambda item: -(item[1]["cpu"] + item[1]["gpu"]))
        return [{"name": comm, "cpu_share": round(use["cpu"], 4), "gpu_share": round(use["gpu"], 4)}
                for comm, use in ranked[:TOP_CONSUMERS] if use["cpu"] + use["gpu"] > 0]

    def contention(self):
        """The repeat's contention record: GPU busy series, foreign GPU share, top consumers at both ends, the
        processes that started, and whether the repeat is `contended` (a window over either limit)."""
        if not self.windows:
            return {"available": False}
        gpu_over = sum(1 for w in self.gpu_windows if w > self.gpu_limit)
        busy = [b for b in self.busy[1:] if b is not None]
        return {
            "available": True,
            "interval_s": self.interval_s,
            "gpu_busy_percent": {"before": self.busy[0] if self.busy else None, "during": busy,
                                 "during_max": max(busy) if busy else None},
            "foreign_gpu_share_max": round(max(self.gpu_windows), 4),
            "foreign_gpu_share_mean": round(sum(self.gpu_windows) / len(self.gpu_windows), 4),
            "gpu_windows_over_limit": gpu_over,
            "gpu_limit": self.gpu_limit,
            "top_start": self._top(self.consumers[0]),
            "top_end": self._top(self.consumers[-1]),
            "started": [{"name": comm, **{k: round(v, 4) if isinstance(v, float) else v for k, v in born.items()}}
                        for comm, born in sorted(self.started.items(), key=lambda kv: -kv[1]["count"])[:MAX_STARTED]],
            "contenders": {comm: {k: round(v, 4) for k, v in held.items()} for comm, held in self.contenders.items()},
        }

    def result(self):
        out = verdict(self.windows, self.limit)
        out["loadavg_start"] = self.loadavg_start
        out["loadavg_end"] = loadavg(self.proc_root)
        if out["available"]:
            contention = self.contention()
            out["contention"] = contention
            out["contended"] = not out["quiet"] or contention["gpu_windows_over_limit"] > 0
        return out


def contenders(row):
    """The names a stored row's repeat was contended by, `[]` when it was not or recorded no contention."""
    load = row.get("host_load") or {}
    if not load.get("contended"):
        return []
    held = (load.get("contention") or {}).get("contenders") or {}
    return sorted(held, key=lambda comm: -max(held[comm].values())) or ["an unnamed process"]


def describe_contention(row):
    """`name (cpu 38%, gpu 12%)` for each contender of a contended row, else None."""
    names = contenders(row)
    if not names:
        return None
    held = ((row.get("host_load") or {}).get("contention") or {}).get("contenders") or {}
    parts = []
    for name in names:
        use = held.get(name)
        parts.append(name if use is None else
                     f"{name} (cpu {use['cpu_share_max']:.0%}, gpu {use['gpu_share_max']:.0%})")
    return ", ".join(parts)


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
