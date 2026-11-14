"""The fragment-round and grouping-scan figures of a device, fitted from its measured corpus (POOT-1547).

    python3 benchmarks/calibration/fit_fragment_cost.py [rows.tsv]        print the figures and how they serve
    python3 benchmarks/calibration/fit_fragment_cost.py --check [rows.tsv]  fail unless crates/poot-target's
                                                                          figures are this fit's result
    python3 benchmarks/calibration/fit_fragment_cost.py --heldout rows.tsv  score the shipped figures on rows of a
                                                                          format the corpus does not hold (POOT-1565)

`rows.tsv` (default `rocm-gfx1151.tsv`, recorded by `scripts/calibrate-launch rocm fragment`) holds one row per pinned
schedule: the cell it was measured in, the device microseconds of its contraction and of its `expert_group` dispatch,
and the cost-model terms of the schedule that ran. The model is `crates/poot-kernelgen/src/contraction/cost.rs`:

    contraction time = a * K * occupancy * lane_step
    lane_step, scalar tiling, and fragment tiling over dense rows
        = fma + lds_read * lds_reads + (staged_value + rounding) * staged_per_lane
    lane_step, fragment tiling over rows grouped by expert = fma + fragment_round / depth      (no staging term)
    lane_step, Gemv = fma + direct_value * direct_per_lane
    grouping = group_scan * scan_steps, scan_steps = 2 * ceil(buckets / 256) * rows     (once, not per lane)

The unit `a` (microseconds per FMA issue per `K` value) is the least-squares scale of the contractions the scalar
model prices (everything but the grouped fragment tilings) at the shipped ratios, by relative error. The corpus
identifies no staging cost on a grouped fragment tiling (a grouped tile is mostly padding rows, and the fit puts the
LDS read and the staged value at zero there), while the dense fragment cells are served by the scalar model. `fragment_round` is chosen by the standing objective of the calibration
design (minimise the worst per-cell regret, then the mean, a cell's regret being the device time of the schedule the
model chooses over the fastest measured schedule of the cell, the chosen schedule paying its grouping step), over a
grid of one octave in four; among the optimal cells the one that fits the corpus best and has every neighbour optimal
too (so a measurement shift of one grid step moves no decision). `group_scan` is the least-squares time of a scan step
over the packed rows, in units of `a`.

`--heldout` scores the figures `calibration.rs` ships, with the unit `a` they imply over `rocm-gfx1151.tsv`, on the cells
of another file, and the same cells under the scalar model before POOT-1547: worst and mean regret of each. Nothing in
that file is fitted.

Stdlib only; the grid and the least squares are small.
"""

import collections
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALIBRATION_RS = HERE.parents[1] / "crates" / "poot-target" / "src" / "calibration.rs"

# The ratios the scalar model keeps on this device (calibration.rs, STRIX_HALO_HSA_COST_RATIOS).
LDS_READ, STAGED_VALUE, DIRECT_VALUE = 9.88, 8.43, 13.4
# The expert-grouping kernel (crates/poot-kernelgen/src/data_movement/expert_group.rs): 256 lanes, buckets strided.
GROUP_LANES = 256
GROUP_BUCKETS = 513
# One octave in four.
ROUND_GRID = [25 * 2 ** (i / 4) for i in range(29)]


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("cell\t"):
            continue
        f = line.split("\t")
        row = dict(
            cell=f[0], k=float(f[1]), rows=float(f[2]), precision=f[3], step=float(f[4]), group=float(f[5]),
            contraction=float(f[6]), fma=float(f[7]), lds=float(f[8]), staged=float(f[9]), direct=float(f[10]),
            rounding=float(f[11]), occupancy=float(f[12]), depth=float(f[13]), lanes=float(f[14]), tile=f[15],
        )
        row["kind"] = "gemv" if row["tile"] == "gemv" else ("fragment" if row["precision"] == "narrow" else "scalar")
        row["grouped"] = row["cell"].startswith("packed") and row["kind"] != "gemv"
        rows.append(row)
    return rows


def lane_step(row, fragment_round):
    if row["kind"] == "gemv":
        return row["fma"] + DIRECT_VALUE * row["direct"]
    if row["kind"] == "fragment" and row["grouped"]:
        return row["fma"] + fragment_round / row["depth"]
    return row["fma"] + LDS_READ * row["lds"] + (STAGED_VALUE + row["rounding"]) * row["staged"]


def units(row, fragment_round):
    return row["occupancy"] * lane_step(row, fragment_round) * row["k"]


def scale(rows, fragment_round):
    """The microseconds per unit that minimises the relative error of the contraction times."""
    xs = [units(r, fragment_round) / r["contraction"] for r in rows if not (r["kind"] == "fragment" and r["grouped"])]
    return sum(xs) / sum(x * x for x in xs)


def group_scan(rows, a):
    """The grouping step in FMA issues: least squares of the grouping dispatch on the scan steps of the packed rows."""
    packed = [r for r in rows if r["grouped"]]
    steps = [2 * math.ceil(GROUP_BUCKETS / GROUP_LANES) * r["rows"] for r in packed]
    microseconds = sum(s * r["group"] for s, r in zip(steps, packed)) / sum(s * s for s in steps)
    return microseconds / a


def score(rows, fragment_round):
    a = scale(rows, fragment_round)
    scan = group_scan(rows, a)

    def predicted(r):
        total = a * units(r, fragment_round)
        if r["grouped"]:
            total += scan * a * 2 * math.ceil(GROUP_BUCKETS / GROUP_LANES) * r["rows"]
        return total

    cells = collections.defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)
    regrets = []
    for cell in cells.values():
        chosen = min(cell, key=predicted)
        regrets.append(chosen["step"] / min(r["step"] for r in cell))
    fitted = [r for r in rows if r["kind"] == "fragment" and r["grouped"]]
    rms = math.sqrt(sum((a * units(r, fragment_round) / r["contraction"] - 1) ** 2 for r in fitted) / len(fitted))
    return max(regrets), sum(regrets) / len(regrets), rms, a, scan


def fit(rows):
    scored = [(score(rows, g), g) for g in ROUND_GRID]
    best = min((s[0], s[1]) for s, _ in scored)
    optimal = {i for i, (s, _) in enumerate(scored) if (s[0], s[1]) == best}
    interior = [i for i in optimal if {i - 1, i + 1} <= optimal] or sorted(optimal)
    pick = min(interior, key=lambda i: scored[i][0][2])
    (worst, mean, rms, a, scan), fragment_round = scored[pick]
    return dict(fragment_round=fragment_round, group_scan=scan, worst=worst, mean=mean, rms=rms, a=a,
                optimal=(ROUND_GRID[min(optimal)], ROUND_GRID[max(optimal)]), rows=len(rows))


def regrets(rows, a, fragment_round, scan, grouped=True):
    """The regret of the model's choice in every cell: `{cell: (regret, chosen row, fastest row)}`. With `grouped`
    false no tiling is priced as grouped by expert, the scalar model before POOT-1547."""
    cells = collections.defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r if grouped else dict(r, grouped=False))

    def predicted(r):
        total = a * units(r, fragment_round)
        if r["grouped"]:
            total += scan * a * 2 * math.ceil(GROUP_BUCKETS / GROUP_LANES) * r["rows"]
        return total

    result = {}
    for name, cell in cells.items():
        chosen = min(cell, key=predicted)
        fastest = min(cell, key=lambda r: r["step"])
        result[name] = (chosen["step"] / fastest["step"], chosen, fastest)
    return result


def heldout(corpus, rows):
    """The shipped figures and the scalar model of before POOT-1547, each as `(worst, mean, per-cell regrets)` on the
    cells of `rows`; the unit of each is its own least-squares scale over `corpus`."""
    fragment_round, scan = constants()
    served = regrets(rows, scale(corpus, fragment_round), fragment_round, scan)
    scalar = regrets(rows, scale([dict(r, grouped=False) for r in corpus], 0.0), 0.0, 0.0, grouped=False)
    return {
        label: (max(v[0] for v in r.values()), sum(v[0] for v in r.values()) / len(r), r)
        for label, r in (("shipped", served), ("scalar", scalar))
    }


def shipped(rows):
    """The scalar model with no fragment round and no grouping, as it chose before POOT-1547: every fragment tiling
    keeps its LDS reads and staged values."""
    saved = [dict(r) for r in rows]
    for r in saved:
        r["grouped"] = False
    a = scale(saved, 0.0)
    cells = collections.defaultdict(list)
    for r in saved:
        cells[r["cell"]].append(r)
    regrets = [min(c, key=lambda r: a * units(r, 0.0))["step"] / min(r["step"] for r in c) for c in cells.values()]
    return max(regrets), sum(regrets) / len(regrets)


def constants():
    text = CALIBRATION_RS.read_text()
    round_ = re.search(r"fragment_round: Some\(([0-9.]+)\)", text)
    scan = re.search(r"group_scan: Some\(([0-9.]+)\)", text)
    return float(round_.group(1)), float(scan.group(1))


def main(argv):
    check = "--check" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if "--heldout" in argv:
        corpus = load(HERE / "rocm-gfx1151.tsv")
        for label, (worst, mean, cells) in heldout(corpus, load(paths[0])).items():
            print(f"{label}: worst regret {worst:.3f}, mean {mean:.4f}")
            for name, (regret, chosen, fastest) in sorted(cells.items()):
                print(f"  {name:22s} {regret:.3f}  chose {chosen['precision']} {chosen['tile']} {chosen['step']:.0f} us,"
                      f" fastest {fastest['precision']} {fastest['tile']} {fastest['step']:.0f} us")
        return 0
    rows = load(paths[0] if paths else HERE / "rocm-gfx1151.tsv")
    result = fit(rows)
    old = shipped(rows)
    print(f"{result['rows']} rows; fragment_round {result['fragment_round']:.1f} (optimal cells {result['optimal'][0]:.1f}"
          f" to {result['optimal'][1]:.1f}), group_scan {result['group_scan']:.1f}")
    print(f"worst regret {result['worst']:.3f}, mean {result['mean']:.4f} (without the terms: {old[0]:.3f}, {old[1]:.4f})")
    if check:
        have = constants()
        want = (round(result["fragment_round"], -1), round(result["group_scan"], 1))
        if have != want:
            print(f"FAIL: calibration.rs has fragment_round {have[0]} and group_scan {have[1]}, the fit gives {want}")
            return 1
        print("calibration.rs agrees with the fit")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
