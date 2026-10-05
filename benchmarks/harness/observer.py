"""External measurement: wrap a runner subprocess and sample GPU memory + host RSS + GPU power.

Every engine's memory number comes from the same instrument (spec 112 FR-003), not each framework's own
accounting (torch's allocator, vLLM's KV pool, and llama.cpp's buffer report measure different things). The
runner's total device VRAM and process-tree RSS are sampled at a fixed interval and the peaks kept. GPU power
draw is sampled on the same loop and integrated over wall-clock time into joules and watt-hours for the cell.

Assumes one engine per pod with the GPU otherwise idle (see the README), so "peak total device memory.used
minus the pre-launch baseline" is the engine-agnostic footprint. The baseline is recorded raw so the report
can subtract it or not.

Prefers pynvml, falls back to `nvidia-smi`; prefers psutil for the process tree, falls back to /proc VmRSS
for the main pid. With no GPU (e.g. the dev Arc or a CPU box) VRAM is None and only RSS is reported.
"""

import json
import locale
import multiprocessing
import os
import selectors
import signal
import subprocess
import sys
import threading
import time


_PROCESS_TERM_GRACE_S = 2.0
_PROCESS_KILL_WAIT_S = 5.0
_READER_JOIN_TIMEOUT_S = 5.0
_READER_POLL_S = 0.05
_SAMPLER_JOIN_TIMEOUT_S = 5.5


class _SignalInterruption(BaseException):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


class _SamplerTermination(BaseException):
    pass


def _nvml_handle(index):
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        return ("pynvml", pynvml, pynvml.nvmlDeviceGetHandleByIndex(index))
    except Exception:
        return None


def _device_used_bytes(handle):
    """Total bytes in use on the GPU right now, or None if no GPU/tooling."""
    if handle is not None:
        kind, pynvml, h = handle
        try:
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).used)
        except Exception:
            return None
    # nvidia-smi fallback.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return int(out.decode().strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def _tree_rss_bytes(pid):
    """Resident bytes of pid + its descendants, or the main pid's peak (VmHWM) as a fallback."""
    try:
        import psutil  # type: ignore

        try:
            proc = psutil.Process(pid)
            procs = [proc] + proc.children(recursive=True)
            return sum(p.memory_info().rss for p in procs if p.is_running())
        except psutil.NoSuchProcess:
            return None
    except Exception:
        pass
    # /proc fallback: VmRSS of the main pid (children not counted without psutil).
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _device_util_pct(handle):
    """GPU compute utilization 0-100 right now (% of the last sample period the SMs were busy), or None."""
    if handle is not None:
        kind, pynvml, h = handle
        try:
            return int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def _device_power_w(handle):
    """Instantaneous GPU board power draw in watts, or None if no GPU/tooling.

    nvmlDeviceGetPowerUsage returns milliwatts; nvidia-smi's power.draw is already watts (e.g. "123.45").
    """
    if handle is not None:
        kind, pynvml, h = handle
        try:
            return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def _make_tree_cpu_sampler(pid):
    """Return a callable -> process-tree CPU% (100% == one full core), or None if psutil is missing.

    Holds persistent Process objects so psutil's since-last-call cpu_percent deltas stay correct.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    procs = {}  # pid -> psutil.Process (persistent for correct cpu_percent deltas)

    def sampler():
        try:
            root = psutil.Process(pid)
            cur = [root] + root.children(recursive=True)
        except Exception:
            return None
        total, seen = 0.0, set()
        for p in cur:
            seen.add(p.pid)
            obj = procs.get(p.pid)
            if obj is None:
                obj = procs[p.pid] = p
                try:
                    obj.cpu_percent(interval=None)  # prime a newly-seen process
                except Exception:
                    pass
            try:
                total += obj.cpu_percent(interval=None)
            except Exception:
                pass
        for dead in [k for k in procs if k not in seen]:
            procs.pop(dead, None)
        return total

    sampler()  # prime the root
    return sampler


def _empty_sample_result(baseline_vram):
    return {
        "peak_vram_bytes": baseline_vram,
        "vram_baseline_bytes": baseline_vram,
        "peak_rss_bytes": None,
        "gpu_util_mean_pct": None,
        "gpu_util_peak_pct": None,
        "cpu_util_mean_pct": None,
        "cpu_util_peak_pct": None,
        "peak_power_w": None,
        "energy_j": None,
        "energy_wh": None,
        "util_series": None,
        "samples": 0,
    }


def _sampler_worker(gpu_index, interval_s, stop, result_conn):
    """Collect samples in an owned session so stuck tool descendants remain cancellable."""
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    def interrupt_blocking_tool(_signum, _frame):
        # Raising through subprocess.run/check_output makes it kill and wait the direct tool child. The
        # parent still SIGKILLs this whole group for descendants or driver calls that do not return to
        # Python during the TERM grace period.
        raise _SamplerTermination()

    signal.signal(signal.SIGTERM, interrupt_blocking_tool)
    try:
        handle = _nvml_handle(gpu_index)
        baseline_vram = _device_used_bytes(handle)
        result_conn.send(("ready", baseline_vram))
        command = result_conn.recv()
        if command[0] == "abort":
            result_conn.send(("ok", _empty_sample_result(baseline_vram)))
            return
        if command[0] != "start":
            raise RuntimeError(f"unexpected sampler command: {command[0]}")
        pid = command[1]

        peak = {"vram": baseline_vram or 0, "rss": 0, "n": 0, "power_w": 0.0}
        gpu_acc = {"sum": 0.0, "n": 0, "peak": 0}
        cpu_acc = {"sum": 0.0, "n": 0, "peak": 0.0}
        energy_acc = {"joules": 0.0, "n": 0, "last_t": None}
        series = []
        t0 = time.time()
        last_series_t = 0.0
        cpu_meter = _make_tree_cpu_sampler(pid)
        while not stop.is_set():
            v = _device_used_bytes(handle)
            if v is not None and v > peak["vram"]:
                peak["vram"] = v
            if stop.is_set():
                break

            r = _tree_rss_bytes(pid)
            if r is not None and r > peak["rss"]:
                peak["rss"] = r
            if stop.is_set():
                break

            g = _device_util_pct(handle)
            if g is not None:
                gpu_acc["sum"] += g
                gpu_acc["n"] += 1
                if g > gpu_acc["peak"]:
                    gpu_acc["peak"] = g
            if stop.is_set():
                break

            c = cpu_meter() if cpu_meter else None
            if c is not None:
                cpu_acc["sum"] += c
                cpu_acc["n"] += 1
                if c > cpu_acc["peak"]:
                    cpu_acc["peak"] = c
            if stop.is_set():
                break

            w = _device_power_w(handle)
            now_p = time.time()
            if w is not None:
                if w > peak["power_w"]:
                    peak["power_w"] = w
                if energy_acc["last_t"] is not None:
                    energy_acc["joules"] += w * (now_p - energy_acc["last_t"])
                energy_acc["n"] += 1
            energy_acc["last_t"] = now_p
            now = time.time() - t0
            if now - last_series_t >= 1.0:
                series.append(
                    [
                        round(now, 2),
                        g,
                        round(c, 1) if c is not None else None,
                        v,
                        r,
                        round(w, 2) if w is not None else None,
                    ]
                )
                last_series_t = now
            peak["n"] += 1
            stop.wait(interval_s)

        result_conn.send(
            (
                "ok",
                {
                    "peak_vram_bytes": peak["vram"] if baseline_vram is not None else None,
                    "vram_baseline_bytes": baseline_vram,
                    "peak_rss_bytes": peak["rss"] or None,
                    "gpu_util_mean_pct": (
                        gpu_acc["sum"] / gpu_acc["n"] if gpu_acc["n"] else None
                    ),
                    "gpu_util_peak_pct": gpu_acc["peak"] if gpu_acc["n"] else None,
                    "cpu_util_mean_pct": (
                        cpu_acc["sum"] / cpu_acc["n"] if cpu_acc["n"] else None
                    ),
                    "cpu_util_peak_pct": cpu_acc["peak"] if cpu_acc["n"] else None,
                    "peak_power_w": peak["power_w"] if energy_acc["n"] else None,
                    "energy_j": energy_acc["joules"] if energy_acc["n"] else None,
                    "energy_wh": (
                        energy_acc["joules"] / 3600.0 if energy_acc["n"] else None
                    ),
                    "util_series": series or None,
                    "samples": peak["n"],
                },
            )
        )
    except BaseException as exc:
        try:
            result_conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        result_conn.close()


def _signal_process(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _terminate_sampler_process(sampler, wait_s, term_grace_s=0.5):
    """Terminate the sampler group, then reap its still-pinned direct worker."""
    pid = sampler.pid
    if pid is None:
        raise RuntimeError("benchmark sampler has no process id")

    # The worker calls setsid() before any metric or tool call. Signal both its group and its PID to cover
    # the pre-setsid startup window. Do not call is_alive() or join() before the final group signal: either
    # can reap the worker and release the PGID identity pin.
    _signal_process_group(pid, signal.SIGTERM)
    _signal_process(pid, signal.SIGTERM)
    deadline = time.monotonic() + term_grace_s
    while time.monotonic() < deadline:
        time.sleep(min(_READER_POLL_S, max(0.0, deadline - time.monotonic())))
    _signal_process_group(pid, signal.SIGKILL)
    _signal_process(pid, signal.SIGKILL)

    sampler.join(timeout=wait_s)
    if sampler.is_alive():
        raise RuntimeError(f"failed to terminate benchmark sampler pid {sampler.pid}")


def _finish_sampler(sampler, result_conn, stop, timeout_s):
    """Request the result, close the sampler group, and reap its direct worker."""
    stop.set()
    payload = None
    reaped = False
    try:
        if result_conn.poll(timeout_s):
            try:
                payload = result_conn.recv()
            except EOFError:
                payload = ("error", "sampler exited without a result")
        else:
            _terminate_sampler_process(sampler, _PROCESS_KILL_WAIT_S)
            reaped = True
            raise RuntimeError(
                f"benchmark sampler did not finish within {timeout_s}s; worker group was terminated"
            )

        # A result does not prove a tool descendant exited with the worker; close the group before join()
        # releases the worker PID.
        _terminate_sampler_process(sampler, _PROCESS_KILL_WAIT_S, term_grace_s=0)
        reaped = True
        if payload[0] != "ok":
            raise RuntimeError(f"benchmark sampler failed: {payload[1]}")
        return payload[1]
    finally:
        result_conn.close()
        if reaped:
            sampler.close()


def _signal_process_group(pgid, sig):
    """Signal only the process group created for this cell."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


class _OwnedProcessGroup:
    """Idempotent cleanup state for one runner-owned process group.

    The direct child is not polled or waited while a group signal may still be needed: its unreaped
    identity pins the PID and process-group number against reuse.
    """

    def __init__(self, proc):
        self.proc = proc
        self.pgid = proc.pid
        self.term_sent = False
        self.term_deadline = None
        self.final_signal_sent = False
        self.reaped = False

    def terminate_group(self, term_grace_s):
        if self.final_signal_sent:
            return
        if not self.term_sent:
            self.term_sent = True
            self.term_deadline = time.monotonic() + term_grace_s
            _signal_process_group(self.pgid, signal.SIGTERM)

        while time.monotonic() < self.term_deadline:
            time.sleep(min(_READER_POLL_S, max(0.0, self.term_deadline - time.monotonic())))

        # Always send the final group signal before wait() reaps the leader; the unreaped leader keeps the
        # PGID safe from reuse even if SIGTERM already stopped every member.
        self.final_signal_sent = True
        _signal_process_group(self.pgid, signal.SIGKILL)

    def reap(self, timeout_s):
        if self.reaped:
            return self.proc.returncode
        try:
            returncode = self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"failed to reap benchmark runner pid {self.proc.pid} after final group signaling"
            ) from None
        self.reaped = True
        return returncode


def _wait_without_reaping(proc, timeout_s, pending_signals):
    """Wait for the direct child to exit while retaining its waitable PID as the group identity pin."""
    deadline = time.monotonic() + timeout_s
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        if pending_signals:
            raise _SignalInterruption(pending_signals[0])
        info = os.waitid(os.P_PID, proc.pid, flags)
        if info is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout_s)
        time.sleep(min(_READER_POLL_S, remaining))


def _start_pipe_reader(stream, captured, relay, emit, stop):
    """Drain one binary pipe without an unbounded readline wait."""
    encoding = locale.getpreferredencoding(False)

    def drain():
        decoder = None
        pending = ""
        try:
            import codecs

            decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
            selector = selectors.DefaultSelector()
            selector.register(stream, selectors.EVENT_READ)
            try:
                while not stop.is_set():
                    if not selector.select(_READER_POLL_S):
                        continue
                    chunk = os.read(stream.fileno(), 65536)
                    if not chunk:
                        break
                    text = decoder.decode(chunk)
                    captured.append(text)
                    pending += text
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        line += "\n"
                        relay(line)
                        emit(line)
                tail = decoder.decode(b"", final=True)
                if tail:
                    captured.append(tail)
                    pending += tail
                if pending:
                    relay(pending)
                    emit(pending)
            finally:
                selector.close()
        except (OSError, ValueError):
            # The owner may close the pipe after the bounded drain window; data captured so far stays valid.
            pass

    thread = threading.Thread(target=drain, name="poot-observer-pipe-reader", daemon=True)
    thread.start()
    return thread


def _finish_pipe_readers(threads, streams, stop, timeout_s):
    """Allow EOF draining, then request a poll-bounded stop and close the parent pipe ends."""
    deadline = time.monotonic() + timeout_s
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    stop.set()
    stop_deadline = time.monotonic() + (_READER_POLL_S * 4)
    for thread in threads:
        thread.join(timeout=max(0.0, stop_deadline - time.monotonic()))
    for stream in streams:
        stream.close()

    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise RuntimeError(f"benchmark pipe readers did not finish: {', '.join(alive)}")


def run_observed(cmd, env=None, cwd=None, gpu_index=0, interval_s=0.05,
                 timeout_s=int(os.environ.get("POOT_BENCH_CELL_TIMEOUT_S", "5400")),
                 on_line=None, termination_grace_s=_PROCESS_TERM_GRACE_S,
                 reader_join_timeout_s=_READER_JOIN_TIMEOUT_S,
                 sampler_join_timeout_s=_SAMPLER_JOIN_TIMEOUT_S):
    """Run cmd, sampling device VRAM + host RSS + GPU/CPU utilization + GPU power until it exits.

    stdout and stderr are read line by line and each line is passed to the optional on_line callback, so
    per-ISL progress is parsed live whichever stream the runner uses (candle/transformers/vLLM print
    progress to stderr; poot to stdout). stderr is also forwarded to our stderr; stdout is captured whole
    but not forwarded (it carries the result JSON and engine banners); the result JSON is its last line.

    Energy is a left-Riemann sum of power_w * dt over the power samples, with dt the wall-clock gap since
    the previous sample, so it does not assume a uniform interval_s (sampling overhead and scheduling
    jitter are absorbed).

    Returns a dict: stdout, stderr, returncode, timed_out, peak_vram_bytes, vram_baseline_bytes, peak_rss_bytes,
    gpu_util_{mean,peak}_pct, cpu_util_{mean,peak}_pct, peak_power_w, energy_j, energy_wh,
    util_series (downsampled ~1 Hz list of [t_s, gpu_pct, cpu_pct, vram_bytes, rss_bytes, power_w]),
    sampler_interval_ms, samples (count). Fields may be None.
    """
    out_buf, err_buf = [], []

    def emit(line):
        if on_line:
            try:
                on_line(line)
            except Exception:
                pass

    def relay_stdout(_line):
        pass

    def relay_stderr(line):
        sys.stderr.write(line)
        sys.stderr.flush()

    proc = None
    owned_group = None
    reader_stop = threading.Event()
    reader_threads = []
    reader_streams = []
    sampler = None
    sampler_conn = None
    sampler_stop = None
    sampler_process_started = False
    sampler_sampling_started = False
    sample_result = _empty_sample_result(None)
    timed_out = False
    observed_exit = False
    interrupted = None
    cleanup_errors = []
    pending_signals = []
    previous_handlers = {}
    previous_sigchld = None
    sigchld_changed = False

    try:
        if threading.current_thread() is threading.main_thread():
            def defer_interruption(signum, _frame):
                if signum not in pending_signals:
                    pending_signals.append(signum)

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                if previous_handlers[signum] == signal.SIG_IGN:
                    continue
                signal.signal(signum, defer_interruption)

            # waitid(WNOWAIT), Popen.wait(), and multiprocessing.Process.join() require waitable children.
            # Inherited SIG_IGN and user handlers that reap SIGCHLD would otherwise release the leader PID
            # before the final process-group signal. Restore this process-wide disposition only after every
            # owned direct child has been reaped.
            previous_sigchld = signal.getsignal(signal.SIGCHLD)
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            sigchld_changed = True

        # The worker takes the pre-launch VRAM baseline before Popen. The readiness wait is bounded, so
        # driver/tool calls stay cancellable.
        mp = multiprocessing.get_context("spawn")
        sampler_stop = mp.Event()
        sampler_conn, worker_conn = mp.Pipe(duplex=True)
        sampler = mp.Process(
            target=_sampler_worker,
            args=(gpu_index, interval_s, sampler_stop, worker_conn),
            name="poot-observer-sampler",
            daemon=True,
        )
        sampler.start()
        sampler_process_started = True
        worker_conn.close()
        if not sampler_conn.poll(sampler_join_timeout_s):
            raise RuntimeError(
                f"benchmark sampler did not initialize within {sampler_join_timeout_s}s"
            )
        try:
            ready = sampler_conn.recv()
        except EOFError:
            raise RuntimeError("benchmark sampler exited before reporting readiness") from None
        if ready[0] != "ready":
            raise RuntimeError(f"benchmark sampler failed during initialization: {ready[1]}")
        sample_result = _empty_sample_result(ready[1])
        if pending_signals:
            raise _SignalInterruption(pending_signals[0])

        proc = subprocess.Popen(
            cmd,
            env={**os.environ, **(env or {})},
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        owned_group = _OwnedProcessGroup(proc)
        sampler_conn.send(("start", proc.pid))
        sampler_sampling_started = True

        # Pollable binary reads let cleanup stop these threads even if an inherited pipe never reaches EOF.
        reader_streams = [proc.stdout, proc.stderr]
        reader_threads.append(
            _start_pipe_reader(proc.stdout, out_buf, relay_stdout, emit, reader_stop)
        )
        reader_threads.append(
            _start_pipe_reader(proc.stderr, err_buf, relay_stderr, emit, reader_stop)
        )
        if pending_signals:
            raise _SignalInterruption(pending_signals[0])

        try:
            _wait_without_reaping(proc, timeout_s, pending_signals)
            observed_exit = True
        except subprocess.TimeoutExpired:
            timed_out = True
    except BaseException as exc:
        interrupted = exc
    finally:
        if proc is not None and owned_group is None:
            owned_group = _OwnedProcessGroup(proc)
        must_terminate_group = proc is not None and (timed_out or interrupted is not None)
        if must_terminate_group:
            try:
                owned_group.terminate_group(termination_grace_s)
            except BaseException as exc:
                cleanup_errors.append(exc)

        if sampler_process_started:
            if not sampler_sampling_started:
                try:
                    sampler_conn.send(("abort",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            try:
                sample_result = _finish_sampler(
                    sampler, sampler_conn, sampler_stop, sampler_join_timeout_s
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        elif sampler_conn is not None:
            sampler_conn.close()

        try:
            _finish_pipe_readers(reader_threads, reader_streams, reader_stop, reader_join_timeout_s)
        except BaseException as exc:
            cleanup_errors.append(exc)

        # A deferred signal or cleanup failure can first appear after a normal runner exit; the leader is
        # still unreaped, so final group signaling is still safe.
        if proc is not None and (must_terminate_group or pending_signals or cleanup_errors):
            try:
                owned_group.terminate_group(termination_grace_s)
            except BaseException as exc:
                cleanup_errors.append(exc)

        # Close the group even after a normal exit. With the leader still a waitable zombie, this
        # zero-grace TERM/KILL pair addresses only the owned group and leaves no live descendant if a
        # signal arrives during the final reap/handler restore.
        if proc is not None and not owned_group.final_signal_sent:
            try:
                owned_group.terminate_group(0)
            except BaseException as exc:
                cleanup_errors.append(exc)

        if proc is not None:
            try:
                owned_group.reap(_PROCESS_KILL_WAIT_S)
            except BaseException as exc:
                cleanup_errors.append(exc)

        handlers_to_restore = set(previous_handlers)
        if sigchld_changed:
            handlers_to_restore.add(signal.SIGCHLD)
        if handlers_to_restore:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handlers_to_restore)
            try:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                if sigchld_changed:
                    signal.signal(signal.SIGCHLD, previous_sigchld)
            except BaseException as exc:
                cleanup_errors.append(exc)
            finally:
                # A signal arriving during restoration is released only after every original handler is
                # back and all owned processes are terminated and reaped.
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    signal_interrupted = bool(pending_signals) or isinstance(interrupted, _SignalInterruption)
    deferred_signal = pending_signals[0] if pending_signals else None
    if deferred_signal is None and isinstance(interrupted, _SignalInterruption):
        deferred_signal = interrupted.signum
    if deferred_signal is not None:
        # Default handlers terminate with the original signal status; returning handlers run before any
        # concurrent observer failure is propagated.
        signal.raise_signal(deferred_signal)

    if interrupted is not None and not isinstance(interrupted, _SignalInterruption):
        raise interrupted
    if cleanup_errors:
        primary = cleanup_errors[0]
        for extra in cleanup_errors[1:]:
            primary.add_note(f"additional cleanup failure: {extra}")
        raise primary

    if not observed_exit and not timed_out and not signal_interrupted:
        raise RuntimeError("benchmark runner ended without an observed exit or timeout")

    out = "".join(out_buf)
    err = "".join(err_buf)
    if timed_out:
        err += f"\n[observer] killed after {timeout_s}s timeout"

    return {
        "stdout": out,
        "stderr": err,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "sampler_interval_ms": interval_s * 1000.0,
        **sample_result,
    }


NO_JSON_RESULT = object()
_JSON_CONTAINER_STARTS = frozenset("{[")
_JSON_SCALAR_STARTS = frozenset('\"-0123456789tfn')


def _scalar_has_token_boundary(line, end, value):
    if isinstance(value, str) or end == len(line):
        return True
    return not (line[end].isalnum() or line[end] in "._+-")


def parse_last_json_line(stdout):
    """Find the final parseable top-level JSON value on runner stdout.

    Containers may follow a same-line log prefix. Scalars must begin a line or immediately follow another
    decoded value, so shutdown text such as ``pid=42`` is not taken as a result. After a successful decode
    the scanner skips the whole value, so nested objects and braces inside strings are never promoted.
    Non-JSON text after the final value is ignored.

    ``NO_JSON_RESULT`` distinguishes absent JSON from a decoded top-level null. Every decoded value goes to
    the result contract.
    """
    decoder = json.JSONDecoder()
    final_value = NO_JSON_RESULT
    for line in stdout.splitlines():
        start = 0
        scalar_allowed = True
        while start < len(line):
            if line[start].isspace():
                start += 1
                continue
            is_container = line[start] in _JSON_CONTAINER_STARTS
            if not is_container and not (
                scalar_allowed and line[start] in _JSON_SCALAR_STARTS
            ):
                scalar_allowed = False
                start += 1
                continue
            try:
                value, length = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                scalar_allowed = False
                start += 1
                continue
            end = start + length
            if not isinstance(value, (dict, list, str)) and not _scalar_has_token_boundary(
                line, end, value
            ):
                scalar_allowed = False
                start += 1
                continue
            final_value = value
            start = end
            scalar_allowed = True
    return final_value


if __name__ == "__main__":
    # Smoke test: `python observer.py -- sleep 0.3` prints the observation as JSON.
    sep = sys.argv.index("--") if "--" in sys.argv else 0
    cmd = sys.argv[sep + 1 :] if sep else sys.argv[1:]
    res = run_observed(cmd)
    res.pop("stdout", None)
    res.pop("stderr", None)
    print(json.dumps(res, indent=2))
