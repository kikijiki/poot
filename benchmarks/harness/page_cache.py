"""The page-cache state a cell starts from, made one explicit condition for every engine.

A model file is read from the page cache or major-faulted from disk, and an engine that maps it lazily (llama.cpp
keeps Flash-Next's 27 GiB per-layer embedding table as a mapped buffer) is 3.5x slower on a cold cache. Before this,
the cache state a cell met was whatever the engine that ran before it left, so the second engine's number depended on
the first (POOT-1507: llama.cpp's 8192-token first token was 3.2 s after the old poot runner and 12.1 s after the new
loader). The harness now reads the whole weights set to /dev/null before every cell, whichever engine it is, and
records the state and the major faults the cell's process tree took.

Pure stdlib, GPU-free.
"""

import os
import re
import resource
from pathlib import Path

WARM = "warm"

# A warm repeat still faults in the runner's own binary and shared libraries. The cold table that POOT-1507
# measured took 515,745 faults; this bound is 16 MiB of 4 KiB pages, far below any weights file.
UNEXPECTED_MAJOR_FAULTS = 4096

READ_CHUNK_BYTES = 8 << 20

# Refusal kinds: why a cell's weights cannot be put in the warm state.
UNREADABLE = "unreadable"
MISSING_SHARD = "missing-shard"
NO_WEIGHTS = "no-weights"
EXCEEDS_MEMORY = "exceeds-available-memory"

_SHARD = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")


class WarmRefusal(Exception):
    """A cell whose weights cannot be warmed; `kind` is one of the refusal kinds above."""

    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


def weight_files(path):
    """The files an engine reads for the model at `path`: a split GGUF set (named by any shard) in full, a single
    file, or every file under a checkpoint directory."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            raise WarmRefusal(NO_WEIGHTS, f"{path} holds no files")
        return files
    shard = _SHARD.match(path.name)
    if shard is None:
        return [path]
    total = int(shard["total"])
    files = [path.with_name(f"{shard['stem']}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise WarmRefusal(MISSING_SHARD, f"split set of {total} lacks {', '.join(missing)}")
    return files


def mem_available_bytes():
    """MemAvailable from /proc/meminfo: free memory plus the page cache the kernel would drop for a new reader."""
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise OSError("/proc/meminfo has no MemAvailable")


def warm(path, available_bytes=None):
    """Read every weights file of the model at `path` to the end, discarding the bytes, so the file set is in the page
    cache whichever engine ran before. Returns the state to record on the row (major faults are added after the cell).

    Raises `WarmRefusal` when the set is unreadable, incomplete or larger than the memory available to hold it.
    """
    files = weight_files(path)
    try:
        sizes = [f.stat().st_size for f in files]
    except OSError as error:
        raise WarmRefusal(UNREADABLE, str(error)) from error
    total = sum(sizes)
    if available_bytes is None:
        available_bytes = mem_available_bytes()
    if total > available_bytes:
        raise WarmRefusal(EXCEEDS_MEMORY, f"{total} bytes of weights, {available_bytes} bytes of memory available")
    buffer = bytearray(READ_CHUNK_BYTES)
    for file, size in zip(files, sizes):
        read = 0
        try:
            with open(file, "rb", buffering=0) as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                while n := stream.readinto(buffer):
                    read += n
        except OSError as error:
            raise WarmRefusal(UNREADABLE, f"{file}: {error}") from error
        if read != size:
            raise WarmRefusal(UNREADABLE, f"{file}: read {read} of {size} bytes")
    return {"state": WARM, "files": len(files), "bytes": total}


def children_major_faults():
    """Major page faults of every child this process has waited for: the difference across a cell is the faults
    its whole process tree took."""
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_majflt


def unexpected_faults(row):
    """The major faults of a row that state a warm cache and took more than the bound, else None. A row that records
    no cache state (a result committed before the harness set one) is not judged."""
    cache = row.get("page_cache")
    if not cache or cache.get("state") != WARM:
        return None
    faults = cache.get("major_faults")
    return faults if isinstance(faults, int) and faults > UNEXPECTED_MAJOR_FAULTS else None
