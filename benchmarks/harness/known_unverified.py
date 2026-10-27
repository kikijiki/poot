"""The reviewed list of committed cells that stay unverified (stdlib only).

A committed set holds every framework of one model, so one cell whose repeats are too noisy to verify would keep
the whole model's set out of the repository, and the rows themselves are never edited. `benchmarks/known-unverified.toml`
is the one place a committed cell may be unverified: each entry names the cell, the pair that is too wide, the band
measured and the repeat count when it was listed, why, and the card that follows up.

`compare-committed` reads the list as an exact statement about the committed rows:

* an unverified cell that is listed, with the band, pair and repeat count the entry states, is `known unverified`;
  a cell with failed repeats is listed the same way, its band taken over the repeats that ran and `failed_repeats`
  stating how many did not;
* the Performance page draws a listed cell's measured medians muted and labelled unverified, never as a verified result
  and never in a ratio;
* an unverified cell that is not listed stays red, as before;
* a listed cell that now verifies, whose measurement no longer matches its entry, or that no longer exists is a
  `stale exception` and is red: delete or re-review the entry.

Only a cell unverified for a measured spread can be listed (failed repeats never make a cell verifiable). A cell with too few repeats, mixed binaries or identical
runs has no band to state and is never covered.
"""

import re
import tomllib
from pathlib import Path

try:
    from . import performance_receipt as receipts
except ImportError:  # direct `python harness/bench.py` execution
    import performance_receipt as receipts

FILE_NAME = "known-unverified.toml"

# The fields of a cell key (`bench.cell_key`), in key order. A host CPU or driver the run did not record is left out.
KEY_FIELDS = ("model", "scenario", "framework", "backend", "device", "host_cpu", "driver")
REQUIRED = ("model", "scenario", "framework", "device", "pair", "band", "repeats", "reason", "follow_up")
BAND_DIGITS = 4
FOLLOW_UP = re.compile(r"POOT-\d+")


def entry_key(entry):
    return tuple(entry.get(field) for field in KEY_FIELDS)


def beside(results_dir):
    """The list of the results directory: `known-unverified.toml` next to it (`benchmarks/known-unverified.toml`)."""
    return Path(results_dir).parent / FILE_NAME


def load(path):
    """The listed exceptions as `{cell key: entry}`, checked for completeness; an absent file lists nothing."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        entries = tomllib.load(f).get("cell", [])
    errors, found = [], {}
    for i, entry in enumerate(entries, 1):
        where = f"[[cell]] #{i}"
        missing = [field for field in REQUIRED if entry.get(field) in (None, "")]
        if missing:
            errors.append(f"{where} lacks {', '.join(missing)}")
            continue
        if not (isinstance(entry["band"], float) and 0 < entry["band"] < 1):
            errors.append(f"{where} band must be a fraction in (0, 1)")
        if not (isinstance(entry["repeats"], int) and entry["repeats"] >= receipts.POLICY["min_trials"]):
            errors.append(f"{where} repeats must be an integer of at least {receipts.POLICY['min_trials']}")
        if not (isinstance(entry.get("failed_repeats", 0), int) and entry.get("failed_repeats", 0) >= 0):
            errors.append(f"{where} failed_repeats must be a nonnegative integer")
        if not FOLLOW_UP.fullmatch(entry["follow_up"]):
            errors.append(f"{where} follow_up must be a card id like POOT-1234")
        key = entry_key(entry)
        if key in found:
            errors.append(f"{where} lists the cell {key} twice")
        found[key] = entry
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return found


def apply(cells, entries, blank_cell):
    """`cells` (from `bench.compare_cells`, a committed set compared with itself) judged against the exception list.

    `blank_cell(key)` builds the cell record of a listed key no committed row has.
    """
    judged = []
    for cell in cells:
        entry = entries.get(cell["key"])
        if entry is not None:
            cell = judge(cell, entry)
        judged.append(cell)
    present = {cell["key"] for cell in cells}
    for key, entry in entries.items():
        if key not in present:
            judged.append(_settle(blank_cell(key), receipts.STALE_EXCEPTION, "listed in known-unverified.toml, but no committed row has this cell: delete the entry"))
    return judged


def judge(cell, entry):
    """`cell` settled against its own entry: known unverified when the entry states what was measured, else stale."""
    if cell["verdict"] not in (receipts.UNVERIFIED, receipts.REPEAT_FAILED):
        return _settle(cell, receipts.STALE_EXCEPTION, f"listed as known unverified but it is now {cell['verdict']}: delete the entry")
    if cell["band"] is None:
        return _settle(cell, receipts.STALE_EXCEPTION, f"listed, but it is unverified for a reason a band cannot cover: {cell['note']}")
    measured = {"pair": cell["band_pair"], "band": round(cell["band"], BAND_DIGITS), "repeats": cell["n_a"],
                "failed_repeats": cell["failed_repeats"]}
    changed = [f"{field} {entry.get(field, 0)} -> {value}" for field, value in measured.items() if entry.get(field, 0) != value]
    if changed:
        return _settle(cell, receipts.STALE_EXCEPTION, f"the entry no longer matches the measurement ({'; '.join(changed)}): re-review it")
    cell["exception"] = entry
    return _settle(cell, receipts.KNOWN_UNVERIFIED, f"{entry['pair']} band {cell['band']:.2%}: {entry['reason']} ({entry['follow_up']})")


def _settle(cell, verdict, note):
    cell.update(verdict=verdict, note=note)
    return cell
