"""The packed Gemv's device figures of a device, identified from its measured geometry sweep (POOT-1608, POOT-1611).

    python3 -I benchmarks/calibration/fit_packed_gemv.py [corpus.tsv]         print the fit, what it identifies, how the
                                                                              committed figures choose and how the fit
                                                                              does held out by cell, format and shape
    python3 -I benchmarks/calibration/fit_packed_gemv.py --check [corpus.tsv] fail unless the figures committed in
                                                                              crates/poot-target lie in the error region
                                                                              and are as good as its best, and the corpus
                                                                              files are the ones their sha256 file names
    python3 -I benchmarks/calibration/fit_packed_gemv.py --self-test          recover known figures from rows the model
                                                                              itself generates (no device)

`corpus.tsv` defaults to `ptx-sm120-packed-gemv.tsv`: one row per measured geometry,

    format  NxK  cols  rows  run  depth  device_us  warp_instructions  lines  units  warp_lanes  width  weight_bytes

The time is the corpus's. The body facts (`warp_instructions`, `lines`) are the generator's, and `lines` of the corpus is
the count of the retired model, which credited the column groups of a warp with sharing the activation read: this script
reads the facts of the model it restates from `ptx-sm120-packed-gemv-body.tsv`, which
`the_corpus_body_facts_are_the_generators` (crates/poot-graph-plan/tests/packed_gemv_geometry.rs) checks against the
generator. The model is `crates/poot-graph-plan/src/planner/gemv_geometry.rs`, restated here:

    lanes_per_col = width / cols          trips  = ceil(K / (lanes_per_col * run))
    workgroups    = ceil(N / (cols * rows))
    per_unit      = max(1, resident_lanes / width)         rounds = max(1, ceil(workgroups / (units * per_unit)))
    warps/unit    = ceil(ceil(workgroups / rounds) / units) * width / warp_lanes       (the busiest unit's whole workgroups)
    work          = warps/unit * max(warp_instructions / issue_per_cycle, lines / l1_lines_per_cycle)
    trip          = max(work, (dram_latency_cycles + warp_instructions * dependent_issue_cycles) / depth)
    cycles        = dispatch_cycles + rounds * trips * trip                time = cycles / clock

Of the figures, `resident_lanes`, `issue_per_cycle` and the clock are structural, `dispatch_cycles` is the measured empty
launch (a per-cell offset: it cannot move an argmin, so no sweep identifies it), and `dram_latency_cycles`,
`l1_lines_per_cycle` and `dependent_issue_cycles` are what the sweep can identify, each only where it moves the term it
scales. A figure the corpus leaves flat is reported as not identified and stays the prior.

The standing objective is the design's: a cell's regret is the measured time of the geometry the model chooses over the
measured time of the fastest geometry of the cell; minimise the worst regret over cells, then the mean. Because that
objective is piecewise constant, the figures are identified by the relative error of the modelled time (the sum of
squared log ratios, "log least squares"); the figures committed are then the point of the near-optimal error region
(within 5 % of the least error, the region the corpus cannot tell apart: 1608 used 5 %, and on the 15-cell corpus no point within 2 % reaches the bound) with the least worst-cell regret, then mean.
That choice is scored held out: refit on all cells but one (by cell), all shapes of all formats but one (by format) or
all formats of all shapes but one (by shape), and score the cells left out.

Stdlib only.
"""

import collections
import hashlib
import itertools
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "ptx-sm120-packed-gemv.tsv"
BODY = HERE / "ptx-sm120-packed-gemv-body.tsv"
CALIBRATION_RS = HERE.parents[1] / "crates" / "poot-target" / "src" / "calibration.rs"

# The rates the sweep can identify, in the order of `GemvDevice`.
RATES = ("dram_latency_cycles", "l1_lines_per_cycle", "dependent_issue_cycles")
# The held figures: the measured dispatch and the structural issue rate (four warp schedulers).
HELD = ("dispatch_cycles", "issue_per_cycle")
CONST = {"dram_latency_cycles": "DRAM_LATENCY_CYCLES", "l1_lines_per_cycle": "L1_LINES_PER_CYCLE",
         "dispatch_cycles": "DISPATCH_CYCLES", "dependent_issue_cycles": "DEPENDENT_ISSUE_CYCLES",
         "issue_per_cycle": "ISSUE_PER_CYCLE"}
# The sm_120 clock the figures are expressed in (calibration.rs: "the part's 2.43 GHz SM clock").
CLOCK_CYCLES_PER_US = 2430.0
RESIDENT_LANES = 1536
# The error region the corpus cannot tell apart: the figures within this factor of the least log error.
REGION = 1.05

Row = collections.namedtuple(
    "Row", "format cell cols rows run depth us instructions lines units warp width bytes n k "
           "rounds trips busy_warps order"
)


def read_rows(corpus, body):
    """The corpus rows with the generator's body facts of `body`, and what the model needs of each geometry
    precomputed (none of it depends on a figure)."""
    facts = {}
    for line in Path(body).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        facts[tuple(f[:6])] = (int(f[6]), float(f[7]))
    rows = []
    for number, line in enumerate(Path(corpus).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 13:
            raise SystemExit(f"{corpus}:{number}: expected 13 columns, got {len(f)}")
        key = tuple(f[:6])
        if key not in facts:
            raise SystemExit(f"{corpus}:{number}: no body facts for {key} in {body}")
        instructions, lines = facts[key]
        if instructions != int(f[7]):
            raise SystemExit(f"{corpus}:{number}: corpus warp_instructions {f[7]} != the generator's {instructions}")
        n, k = (int(v) for v in f[1].split("x"))
        cols, rows_, run, depth = (int(v) for v in f[2:6])
        units, warp, width = int(f[9]), int(f[10]), int(f[11])
        lanes_per_col = width // cols
        trips = -(-k // (lanes_per_col * run))
        workgroups = -(-n // (cols * rows_))
        per_unit = max(1, RESIDENT_LANES // width)
        rounds = max(1, -(-workgroups // (units * per_unit)))
        busiest = -(-(-(-workgroups // rounds)) // units)
        rows.append(Row(f[0], f[1], cols, rows_, run, depth, float(f[6]), instructions, lines, units, warp, width,
                        int(f[12]), n, k, rounds, trips, busiest * width / warp, None))
    if not rows:
        raise SystemExit(f"{corpus}: no rows")
    # The planner's tie order: rows, depth, run (longest first), cols.
    return [r._replace(order=(r.rows, r.depth, -r.run, r.cols)) for r in rows]


def modelled_us(row, fig):
    """The model's time of `row`'s geometry under `fig` (a dict of the figures)."""
    work = row.busy_warps * max(row.instructions / fig["issue_per_cycle"], row.lines / fig["l1_lines_per_cycle"])
    chain = (fig["dram_latency_cycles"] + row.instructions * fig["dependent_issue_cycles"]) / row.depth
    return (fig["dispatch_cycles"] + row.rounds * row.trips * max(work, chain)) / CLOCK_CYCLES_PER_US


def cells_of(rows):
    cells = collections.OrderedDict()
    for row in rows:
        cells.setdefault((row.format, row.cell), []).append(row)
    return cells


def chosen_of(cell, fig):
    """The geometry the planner takes: the cheapest, a tie (a relative 1e-9) to the one earliest in its search order."""
    times = [(modelled_us(r, fig), r) for r in cell]
    least = min(t for t, _ in times)
    return min((r for t, r in times if t <= least * (1 + 1e-9)), key=lambda r: r.order)


def regrets(rows, fig):
    """Per cell: (regret, chosen row, fastest row)."""
    out = {}
    for key, cell in cells_of(rows).items():
        best = min(cell, key=lambda r: r.us)
        chosen = chosen_of(cell, fig)
        out[key] = (chosen.us / best.us, chosen, best)
    return out


def objective(rows, fig):
    values = [r for r, _, _ in regrets(rows, fig).values()]
    return (max(values), sum(values) / len(values))


def log_error(rows, fig):
    return sum(math.log(modelled_us(r, fig) / r.us) ** 2 for r in rows)


def fit(rows, held, starts=10, seed=1611):
    """Log least squares of model/measured over `RATES`, the rest held, coordinate descent in log space from several
    starts. Returns the figures and their log error."""
    rng = random.Random(seed)
    bounds = {"dram_latency_cycles": (2.0, 4.5), "l1_lines_per_cycle": (-1.0, 1.5), "dependent_issue_cycles": (0.0, 1.5)}
    best = None
    for _ in range(starts):
        fig = dict(held)
        for name in RATES:
            fig[name] = 10 ** rng.uniform(*bounds[name])
        err = log_error(rows, fig)
        step = 1.5
        while step > 1.0005:
            moved = False
            for name in RATES:
                for factor in (step, 1 / step):
                    trial = dict(fig)
                    trial[name] *= factor
                    e = log_error(rows, trial)
                    if e < err * (1 - 1e-12):
                        fig, err, moved = trial, e, True
            if not moved:
                step = 1 + (step - 1) / 2
        if best is None or err < best[1]:
            best = (fig, err)
    return best


def region(rows, fit_fig, error, held):
    """Every figure on a log grid around the fit whose log error is within `REGION` of the least: what the corpus cannot
    tell from the fit. Returns (figures, log error) pairs."""
    grid = {name: [fit_fig[name] * 2 ** (i / 3) for i in range(-6, 7)] for name in RATES}
    points = []
    for latency, l1, cpi in itertools.product(*(grid[n] for n in RATES)):
        fig = dict(held, dram_latency_cycles=latency, l1_lines_per_cycle=l1, dependent_issue_cycles=cpi)
        e = log_error(rows, fig)
        if e <= error * REGION:
            points.append((fig, e))
    return points


def pick(rows, held):
    """The figures the model keeps: the least-regret point (worst cell, then mean) of the near-optimal error region of
    the log least squares fit. Returns (figures, fit figures, fit error, the region's points)."""
    fit_fig, error = fit(rows, held)
    points = region(rows, fit_fig, error, held)
    scored = sorted(((objective(rows, fig), fig) for fig, _ in points), key=lambda p: p[0])
    return scored[0][1], fit_fig, error, points


def held_out(rows, groups, held):
    """Refit on every row outside a group and score the group's cells with the figures picked there. `groups` maps a
    group name to the rows left out. Returns per group (worst, mean, figures)."""
    out = {}
    for name, left_out in groups.items():
        inside = [r for r in rows if r not in left_out]
        fig, *_ = pick(inside, held)
        worst, mean = objective(left_out, fig)
        out[name] = (worst, mean, fig)
    return out


def groups_by(rows, key):
    groups = collections.OrderedDict()
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {name: set(members) for name, members in groups.items()}


def committed_figures():
    text = CALIBRATION_RS.read_text()
    out = {}
    for field, const in CONST.items():
        match = re.search(r"name:\s*\"SM_120_GEMV_" + const + r"\",\s*value:\s*([0-9._]+)", text)
        out[field] = float(match.group(1).replace("_", "")) if match else None
    return out


def describe(rows, label, fig):
    worst, mean = objective(rows, fig)
    lines = [f"{label}: worst cell regret {worst:.3f}, mean {mean:.3f}, run-32 picks "
             f"{sum(1 for _, c, _ in regrets(rows, fig).values() if c.run == 32)}"]
    for key, (regret, chosen, best) in regrets(rows, fig).items():
        lines.append(
            f"    {key[0]:7s} {key[1]:12s} regret {regret:.3f}  chose (cols {chosen.cols}, rows {chosen.rows}, "
            f"run {chosen.run}, depth {chosen.depth}) {chosen.us:.1f} us; fastest (cols {best.cols}, rows {best.rows}, "
            f"run {best.run}, depth {best.depth}) {best.us:.1f} us")
    return lines


def report(rows, prior):
    held = {name: prior[name] for name in HELD}
    picked, fit_fig, error, points = pick(rows, held)
    out = [f"{len(rows)} rows, {len(cells_of(rows))} cells; log-error of the fit {error:.4f} "
           f"(rms {math.sqrt(error / len(rows)):.3f})"]
    out.append(f"{'figure':24s} {'prior':>10s} {'LS fit':>10s} {'picked':>10s}   region of the log error within {REGION - 1:.0%}")
    for name in RATES:
        values = [fig[name] for fig, _ in points]
        out.append(f"  {name:22s} {prior[name]:10.3f} {fit_fig[name]:10.3f} {picked[name]:10.3f}   "
                   f"{min(values):.3f} .. {max(values):.3f} (x{max(values) / min(values):.2f}, {len(points)} grid points)")
    for name in HELD:
        out.append(f"  {name:22s} {prior[name]:10.3f}  held ({'measured' if name == 'dispatch_cycles' else 'structural'})")
    for name in RATES:
        span = max(fig[name] for fig, _ in points) / min(fig[name] for fig, _ in points)
        out.append(f"  {name}: {'IDENTIFIED' if span < 4 else 'NOT IDENTIFIED (flat over a 4x range)'}")
    out += describe(rows, "committed figures", prior)
    out += describe(rows, "LS fit", dict(held, **{n: fit_fig[n] for n in RATES}))
    out += describe(rows, "picked (least regret in the region)", picked)
    regions = sorted(objective(rows, fig) for fig, _ in points)
    out.append(f"regret over the {len(points)} region points: worst {regions[0][0]:.3f} (best) .. {regions[-1][0]:.3f}, "
               f"median {regions[len(regions) // 2][0]:.3f}")
    for label, key in (("cell", lambda r: (r.format, r.cell)), ("format", lambda r: r.format),
                       ("shape", lambda r: r.cell)):
        result = held_out(rows, groups_by(rows, key), held)
        worst_group = max(result, key=lambda g: result[g][0])
        out.append(f"held out by {label} ({len(result)} groups, refit on the rest): worst regret "
                   f"{max(v[0] for v in result.values()):.3f} (group {worst_group}), mean over groups "
                   f"{sum(v[1] for v in result.values()) / len(result):.3f}")
        for name, (worst, mean, fig) in result.items():
            out.append(f"    out {str(name):24s} worst {worst:.3f} mean {mean:.3f}  latency {fig['dram_latency_cycles']:.0f} "
                       f"L1 {fig['l1_lines_per_cycle']:.2f} cpi {fig['dependent_issue_cycles']:.2f}")
    return "\n".join(out), picked


def check_files():
    """The corpus and register files are the ones their sha256 file names."""
    names = {}
    for line in (HERE / "ptx-sm120-packed-gemv.sha256").read_text().splitlines():
        digest, name = line.split()
        names[name] = digest
    for name, digest in names.items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if actual != digest:
            return f"{name}: sha256 {actual} is not the recorded {digest}"
    if "ptx-sm120-packed-gemv-body.tsv" not in names:
        return "ptx-sm120-packed-gemv-body.tsv has no sha256 row"
    return None


def self_test():
    """Rows generated from known figures and a geometry grid are recovered; a figure the rows leave flat is reported
    as the span of its region (the sweep design this tool ships is judged by this)."""
    truth = {"dram_latency_cycles": 1500.0, "l1_lines_per_cycle": 0.8, "dependent_issue_cycles": 5.0,
             "dispatch_cycles": 4450.0, "issue_per_cycle": 4.0}
    rng = random.Random(7)
    rows = []
    for n, k in ((17408, 5120), (5120, 17408), (2560, 9728), (9728, 2560)):
        for cols in (1, 2, 4, 8, 16, 32):
            for run in (4, 8, 16, 32):
                for rows_, depth in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3)):
                    instructions = int(200 * run / 16 * (1 + 0.8 * (rows_ - 1)) + 40 * depth)
                    lines = (2 + 0.06 * (256 / cols) * run / 16) * rows_ * 8 / (1 + math.log2(cols + 1)) * 8
                    units, warp, width = 188, 32, 256
                    trips = -(-k // ((width // cols) * run))
                    workgroups = -(-n // (cols * rows_))
                    rounds = max(1, -(-workgroups // (units * (RESIDENT_LANES // width))))
                    busiest = -(-(-(-workgroups // rounds)) // units)
                    base = Row("q4_k", f"{n}x{k}", cols, rows_, run, depth, 0.0, instructions, lines, units, warp, width,
                               0, n, k, rounds, trips, busiest * width / warp, (rows_, depth, -run, cols))
                    rows.append(base._replace(us=modelled_us(base, truth) * (1 + rng.gauss(0, 0.01))))
    held = {name: truth[name] for name in HELD}
    fig, _ = fit(rows, held)
    for name in RATES:
        ratio = fig[name] / truth[name]
        assert 0.7 < ratio < 1.4, f"{name}: fit {fig[name]:.2f} vs truth {truth[name]:.2f}"
    assert objective(rows, fig)[0] < 1.05, objective(rows, fig)
    print("self-test ok: " + ", ".join(f"{n} {fig[n]:.1f} (truth {truth[n]})" for n in RATES))


def main(argv):
    args = argv[1:]
    if args == ["--self-test"]:
        self_test()
        return 0
    check = "--check" in args
    paths = [a for a in args if not a.startswith("--")]
    corpus = Path(paths[0]) if paths else CORPUS
    if len(paths) > 1:
        print(__doc__)
        return 2
    prior = committed_figures()
    if any(v is None for v in prior.values()):
        raise SystemExit("could not read SM_120_GEMV_* from calibration.rs")
    rows = read_rows(corpus, BODY)
    text, picked = report(rows, prior)
    print(text)
    if check:
        problem = check_files()
        if problem:
            print(f"CHECK FAILED: {problem}")
            return 1
        held = {name: prior[name] for name in HELD}
        fit_fig, error = fit(rows, held)
        committed_error = log_error(rows, prior)
        if committed_error > error * REGION:
            print(f"CHECK FAILED: the committed figures' log error {committed_error:.3f} is outside the region "
                  f"({REGION - 1:.0%} of the least, {error:.3f})")
            return 1
        best = min(objective(rows, fig) for fig, _ in region(rows, fit_fig, error, held))
        achieved = objective(rows, prior)
        if achieved > best:
            print(f"CHECK FAILED: the committed figures' regret (worst, mean) {achieved} is worse than the best "
                  f"of the region, {best}")
            return 1
        print("check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
