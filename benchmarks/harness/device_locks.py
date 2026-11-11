"""The shared-box device locks one measured attempt holds, GPU first and then heavy (POOT-1523).

Another lane's build disturbs a timed number and two GPU jobs disturb each other, so a measured attempt takes
`/tmp/poot-gpu.lock` and, inside it, `/tmp/poot-heavy.lock`. The harness takes them itself, per attempt and only after
every wait (quiet host, cooldown) is over: a sweep that waited inside a caller's `flock` blocked every other lane for
hours on 2026-11-07. A caller therefore does not wrap `bench run` in `flock`.
"""

import contextlib
import fcntl

GPU_LOCK = "/tmp/poot-gpu.lock"
HEAVY_LOCK = "/tmp/poot-heavy.lock"
ORDER = (GPU_LOCK, HEAVY_LOCK)


@contextlib.contextmanager
def held(paths=ORDER):
    """Hold the advisory locks `paths` in the order given (blocking), and release them in reverse."""
    with contextlib.ExitStack() as stack:
        for path in paths:
            handle = stack.enter_context(open(path, "a"))
            fcntl.flock(handle, fcntl.LOCK_EX)
            stack.callback(fcntl.flock, handle, fcntl.LOCK_UN)
        yield
