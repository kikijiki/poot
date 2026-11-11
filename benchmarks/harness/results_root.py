"""Where the benchmark result sets live: the one setting every harness tool and the orchestrator resolve.

The result sets are not part of this repository. `POOT_BENCH_RESULTS` names the directory holding them
(`<run-id>-r1` ... `-rN`, `INDEX.md`, `diagnostics/`):

* a run that records new sets writes to `$POOT_BENCH_RESULTS`, by default `benchmarks/results/` in this checkout
  (git-ignored), so a fresh sweep needs no setup;
* a tool that reads the committed baseline (the Performance page generator, `bench compare-committed`, the harness
  tests) reads `$POOT_BENCH_RESULTS`, by default the archive: the first sibling directory of this repository that
  holds `projects/poot/benchmark-results` (`<repo parent>/*/projects/poot/benchmark-results`).

The performance bank's measured output (`receipts/`, `receipts.json`, `ceilings/`) follows the same setting: it is the
`performance-bank` directory beside the results root (`bank_root`), so one archive checkout serves both. The case
definitions and runners stay in `benchmarks/performance-bank/`.

A baseline that cannot be found fails with every lookup tried, never an empty result.
"""

import os
from pathlib import Path

ENV = "POOT_BENCH_RESULTS"
BENCH_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_ROOT.parent
ARCHIVE_GLOB = "*/projects/poot/benchmark-results"
BANK_DIR = "performance-bank"


class ResultsRootMissing(Exception):
    """The committed baseline directory does not exist; the message lists the lookups tried."""


def recording_root():
    """The directory a run writes new result sets into."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV])
    return BENCH_ROOT / "results"


def committed_root():
    """The directory holding the committed baseline sets, which must exist."""
    if os.environ.get(ENV):
        root = Path(os.environ[ENV])
        if not root.is_dir():
            raise ResultsRootMissing(f"no committed results: ${ENV} names {root}, which is not a directory")
        return root
    found = sorted(REPO_ROOT.parent.glob(ARCHIVE_GLOB))
    if not found:
        raise ResultsRootMissing(
            f"no committed results: ${ENV} is not set and {REPO_ROOT.parent}/{ARCHIVE_GLOB} matches nothing; "
            f"set ${ENV} to the directory holding the result sets, or put an archive checkout beside {REPO_ROOT}"
        )
    return found[0]


def recording_bank_root():
    """The directory a bank run writes its receipts and registry into: `performance-bank` beside the recording root."""
    return recording_root().parent / BANK_DIR


def committed_bank_root():
    """The directory holding the committed bank receipts, registry and ceilings, which must exist."""
    root = committed_root().parent / BANK_DIR
    if not root.is_dir():
        raise ResultsRootMissing(f"no committed bank receipts: {root} is not a directory (beside the results root)")
    return root


if __name__ == "__main__":
    import sys

    try:
        print(committed_root())
    except ResultsRootMissing as error:
        sys.exit(str(error))
