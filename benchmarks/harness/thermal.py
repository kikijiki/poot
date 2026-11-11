"""The thermal start gate of every repeat, and the temperatures it ran at, recorded in the row (POOT-1523).

The 2026-11 Strix Halo second pass found both Flash-Next cells unverified from thermal drift: over a 35-minute
llama.cpp cell the first token rose while the shader clock fell from 2755 to 2470 MHz at 87 to 89 C (the CPU die read
89 C too, with the GPU idle and other lanes building). Starting every repeat after a cooldown made both cells verify.
So the gate is part of the harness and applies to every cell, every attempt and every framework the same way:

* before an attempt starts, the GPU and CPU temperatures must be at or below a start limit; the gate waits for that,
  at most `timeout_s`, and an attempt that never cools is an error row with the typed reason `NEVER_COOLED`;
* the wait holds no lock (`bench.run_quiet_cell` takes the device locks only after the gate passed, and re-checks the
  temperatures once it holds them, since a queue for the lock cools or heats the box);
* the row records the limits, the temperatures at the start and the end of the attempt and the peak between, the
  shader clock range, and `throttled` when the attempt reached the throttle point.

The limit is derived from the device, never from a model or a shape: `start limit = throttle point - MARGIN_C`.
The throttle point is what the driver reports (`temp1_crit` of the sensor's hwmon) and, where the driver reports
none (amdgpu on an APU and k10temp do not), `OBSERVED_THROTTLE_C`: the temperature at which the Strix Halo shader
clock was seen to fall, 87 C (GPU) and 89 C (CPU). The margin is the distance between that onset and the 65 C start
at which both cells verified: 22 C. That puts the start limits at 65 C (GPU) and 67 C (CPU); a cooler start can only
help, so the CPU limit is stricter than the 75 C the second pass used.

Pure stdlib. A host with none of the sensors (a pod) records `available: false` and the attempt is not gated: that
is a host without a thermal source, not a cool one.
"""

import threading
import time
from pathlib import Path

NEVER_COOLED = "thermal-never-cooled"
MARGIN_C = 22.0
OBSERVED_THROTTLE_C = {"gpu": 87.0, "cpu": 89.0}
# hwmon driver name -> the role of its temp1 sensor.
HWMON_ROLES = {"amdgpu": "gpu", "k10temp": "cpu", "coretemp": "cpu"}
POLL_S = 5.0
SAMPLE_INTERVAL_S = 1.0
# A start gate that has not passed in this long is a box that is not cooling (builds of other lanes, a fan fault).
DEFAULT_TIMEOUT_S = 900.0


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _milli(path):
    text = _read(path)
    try:
        return int(text) / 1000.0 if text is not None else None
    except ValueError:
        return None


class SysfsThermal:
    """The GPU and CPU temperatures, the driver's throttle points and the shader clock, from hwmon."""

    def __init__(self, hwmon_root="/sys/class/hwmon"):
        self.sensors = {}
        root = Path(hwmon_root)
        for hwmon in sorted(root.glob("hwmon*")) if root.is_dir() else ():
            role = HWMON_ROLES.get(_read(hwmon / "name"))
            if role and role not in self.sensors:
                self.sensors[role] = hwmon

    def read(self):
        """`{"gpu_c", "cpu_c", "sclk_mhz"}`; a value the host does not report is None."""
        gpu, cpu = self.sensors.get("gpu"), self.sensors.get("cpu")
        sclk = _read(gpu / "freq1_input") if gpu else None
        return {"gpu_c": _milli(gpu / "temp1_input") if gpu else None,
                "cpu_c": _milli(cpu / "temp1_input") if cpu else None,
                "sclk_mhz": int(sclk) / 1e6 if sclk and sclk.isdigit() else None}

    def driver_throttle_c(self, role):
        """The throttle point the driver reports for `role`'s sensor, or None."""
        hwmon = self.sensors.get(role)
        return _milli(hwmon / "temp1_crit") if hwmon else None


def limits(source):
    """`{role: {"throttle_c", "start_limit_c", "basis"}}` for every role the source has a sensor for."""
    out = {}
    for role in source.sensors:
        driver = source.driver_throttle_c(role)
        throttle = driver if driver is not None else OBSERVED_THROTTLE_C[role]
        out[role] = {"throttle_c": throttle, "start_limit_c": throttle - MARGIN_C,
                     "basis": "driver" if driver is not None else "observed"}
    return out


def hot_roles(reading, rules):
    """The roles whose temperature is above its start limit in `reading`."""
    return [role for role, rule in rules.items()
            if reading.get(f"{role}_c") is not None and reading[f"{role}_c"] > rule["start_limit_c"]]


class Gate:
    """The start gate of an attempt: `wait()` before the locks, `check()` once they are held, `meter()` around the run."""

    def __init__(self, source=None, timeout_s=DEFAULT_TIMEOUT_S, poll_s=POLL_S, sleep=time.sleep, now=time.monotonic):
        self.source = SysfsThermal() if source is None else source
        self.rules = limits(self.source)
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.sleep = sleep
        self.now = now

    def available(self):
        return bool(self.rules)

    def wait(self):
        """Poll until every sensor is at or below its start limit or `timeout_s` passed. Holds no lock and takes none.
        Returns `{"cooled", "waited_s", "polls", "reading", "hot"}`."""
        start, polls = self.now(), 0
        while True:
            reading = self.source.read()
            polls += 1
            hot = hot_roles(reading, self.rules)
            waited = self.now() - start
            if not hot or waited >= self.timeout_s:
                return {"cooled": not hot, "waited_s": round(waited, 1), "polls": polls, "reading": reading, "hot": hot}
            self.sleep(self.poll_s)

    def check(self):
        """One reading against the limits: `(still cool, reading)`."""
        reading = self.source.read()
        return not hot_roles(reading, self.rules), reading

    def meter(self):
        return ThermalMeter(self)

    def refusal(self, waited):
        """The text of the error row of an attempt that never cooled."""
        reading = waited["reading"]
        over = ", ".join(f"{role} {reading[f'{role}_c']:.0f} C > {self.rules[role]['start_limit_c']:.0f} C"
                         for role in waited["hot"])
        return f"{NEVER_COOLED}: after {waited['waited_s']:.0f} s the box was still above its start limit ({over})"


class ThermalMeter:
    """Records the temperatures and the shader clock of an attempt on a sampling thread, 1 Hz."""

    def __init__(self, gate, interval_s=SAMPLE_INTERVAL_S):
        self.gate = gate
        self.interval_s = interval_s
        self.start = None
        self.last = None
        self.peak = {}
        self.sclk = []
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        reading = self.gate.source.read()
        self.last = reading
        for role in self.gate.rules:
            value = reading.get(f"{role}_c")
            if value is not None:
                self.peak[role] = max(self.peak.get(role, value), value)
        if reading.get("sclk_mhz") is not None:
            self.sclk.append(reading["sclk_mhz"])

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        self.start = self.gate.source.read()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()

    def result(self, waited=None):
        """The row's `thermal` record."""
        throttled = [role for role, rule in self.gate.rules.items()
                     if self.peak.get(role, float("-inf")) >= rule["throttle_c"]]
        out = {
            "available": True,
            "margin_c": MARGIN_C,
            "limits": self.gate.rules,
            "start": {k: v for k, v in self.start.items() if k.endswith("_c")},
            "end": {k: v for k, v in (self.last or {}).items() if k.endswith("_c")},
            "peak": self.peak,
            "throttled": bool(throttled),
            "throttled_roles": throttled,
        }
        if self.sclk:
            out["sclk_mhz"] = {"min": min(self.sclk), "max": max(self.sclk)}
        if waited is not None:
            out["waited_s"] = waited["waited_s"]
        return out


def throttled_repeats(rows):
    """`[(repeat number, roles)]` for every row of a cell that records a throttled attempt."""
    return [(n, row["thermal"]["throttled_roles"]) for n, row in enumerate(rows, 1)
            if (row.get("thermal") or {}).get("throttled")]
