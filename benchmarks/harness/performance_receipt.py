"""The one performance comparator and receipt validator (stdlib only).

Three questions, each answered here and nowhere else:

* May two records be compared at all? `identity_mismatches` names every workload-identity field two records
  do not share; a field a record does not state is a mismatch, never a wildcard.
* Is a request's measured window self-consistent? `request_window_errors` checks one request's boundaries
  (TTFT, ITL gaps, finish, E2E) and its step counter.
* Did the candidate hold up on everything the scenario requires? `compare_receipts` judges every required
  (cell, metric) pair, so a good headline cannot hide a slower long-context token, a missing cell or a failed
  repeat; `metric_verdict` is the bounded-uncertainty rule each pair is judged by.

`bench compare` reaches the same rule through `metric_verdict`; nothing else defines a verdict.
"""

import hashlib
import json
import math
import statistics
import tomllib
from pathlib import Path

REGRESSION = "regression"
IMPROVEMENT = "improvement"
WITHIN_NOISE = "within noise"
UNVERIFIED = "unverified"
INVALID = "invalid"
CANDIDATE_FAILED = "candidate failed"
REPEAT_FAILED = "repeat failed"
NO_BASELINE = "no baseline"
NEW = "new"
NOT_RUN = "not run"
REFUSED = "refused"
INCOMPARABLE = "incomparable"

COMPARED_VERDICTS = frozenset({REGRESSION, IMPROVEMENT, WITHIN_NOISE})
FAILING_VERDICTS = frozenset({REGRESSION, INVALID, CANDIDATE_FAILED, REPEAT_FAILED, INCOMPARABLE})

COMPLETE = "complete"
INCOMPLETE = "incomplete"

# Which way is worse. Latency is worse when higher, throughput worse when lower.
LOWER_IS_BETTER = "lower"
HIGHER_IS_BETTER = "higher"

MANIFEST = Path(__file__).resolve().parents[1] / "manifest.toml"

# The fields that decide whether two records measure the same work. Grouped by what a mutation of the
# group changes; a record that omits one cannot be compared on that axis.
IDENTITY_GROUPS = {
    "checkpoint": ("checkpoint_digest", "token_fixture"),
    "numerics": ("precision", "layout"),
    "shape": (
        "dimensions",
        "prompt_tokens",
        "live_context",
        "max_context",
        "requested_output",
        "actual_output",
    ),
    "sampling": ("sampler", "template"),
    "preparation": ("cache_state", "prepared_set"),
    "timing": ("timing_policy",),
    "hardware": ("backend", "device", "driver"),
}
IDENTITY_FIELDS = tuple(field for group in IDENTITY_GROUPS.values() for field in group)

_METRIC_FIELDS = ("direction", "practical_threshold", "max_uncertainty")


def load_policy(path=MANIFEST):
    """The performance policy a manifest declares (`[performance]`), checked for completeness.

    The bounded-trial rule, each metric's direction and tolerances, and which metrics a decode-curve cell is
    compared by are declared in the manifest and nowhere else; this module holds no threshold of its own.
    """
    with open(path, "rb") as f:
        policy = tomllib.load(f)["performance"]
    errors = []
    for field in ("min_trials", "max_trials"):
        if not (isinstance(policy.get(field), int) and policy[field] > 0):
            errors.append(f"[performance] {field} must be a positive integer")
    if isinstance(policy.get("min_trials"), int) and isinstance(policy.get("max_trials"), int):
        if policy["max_trials"] < policy["min_trials"]:
            errors.append("[performance] max_trials is below min_trials")
    metrics = policy.get("metrics", {})
    for name, entry in metrics.items():
        if entry.get("direction") not in (LOWER_IS_BETTER, HIGHER_IS_BETTER):
            errors.append(f"[performance.metrics.{name}] direction must be 'lower' or 'higher'")
        for field in _METRIC_FIELDS[1:]:
            if not (isinstance(entry.get(field), (int, float)) and 0 < entry[field] < 1):
                errors.append(f"[performance.metrics.{name}] {field} must be a fraction in (0, 1)")
    for field in ("headline_metric", "curve_point_metrics"):
        for name in [policy.get(field)] if field == "headline_metric" else policy.get(field, []):
            if name not in metrics:
                errors.append(f"[performance] {field} names {name!r}, which has no [performance.metrics] entry")
    if errors:
        raise ValueError("; ".join(errors))
    return policy


POLICY = load_policy()


def canonical(value):
    """The one byte form of a JSON value that identities and digests are taken over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity_digest(identity):
    """A digest of the identity fields only, so extra provenance never changes a fixture's identity."""
    subset = {field: identity.get(field) for field in IDENTITY_FIELDS}
    return hashlib.sha256(canonical(subset).encode()).hexdigest()


def identity_mismatches(baseline, candidate, fields=IDENTITY_FIELDS):
    """`[(field, reason)]` for every identity field the two records do not agree on, in field order."""
    found = []
    for field in fields:
        a, b = baseline.get(field), candidate.get(field)
        if a is None or b is None:
            side = "baseline" if a is None else "candidate"
            if a is None and b is None:
                side = "both"
            found.append((field, f"not recorded by the {side}"))
        elif canonical(a) != canonical(b):
            found.append((field, f"baseline {canonical(a)} != candidate {canonical(b)}"))
    return found


def is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric_verdict(samples_a, samples_b, policy, metric, *, failed_a=0, failed_b=0):
    """Judge one metric: baseline samples against candidate samples, each one repeat's value.

    `policy` is the manifest's `[performance]` table and `metric` names its entry. The rule: the band is twice
    the larger observed range of the two sides. A candidate further outside it than the band is a regression
    (or improvement). Inside it the result is within noise only when the band is narrow enough
    (`max_uncertainty`) to exclude a loss of `practical_threshold`; a band too wide, or too few repeats, or an
    exhausted repeat budget that never narrowed it, is unverified, never success. A failed repeat on either
    side prevents any pass. Returns a dict with `verdict`, `note` and the statistics.
    """
    tolerances = policy["metrics"][metric]
    direction = tolerances["direction"]
    result = {
        "verdict": None,
        "note": "",
        "n_a": len(samples_a),
        "n_b": len(samples_b),
        "median_a": None,
        "range_a": None,
        "median_b": None,
        "range_b": None,
        "budget_exhausted": False,
    }

    def settle(verdict, note=""):
        result.update(verdict=verdict, note=note)
        return result

    if not all(is_finite(x) for x in (*samples_a, *samples_b)):
        sides = [s for s, xs in (("baseline", samples_a), ("candidate", samples_b)) if not all(map(is_finite, xs))]
        return settle(INVALID, f"non-finite sample in the {' and '.join(sides)}")
    if not samples_a:
        return settle(NO_BASELINE, "the baseline has no successful run of this cell")
    if not samples_b:
        return settle(CANDIDATE_FAILED, "no successful candidate run")

    result.update(
        median_a=statistics.median(samples_a),
        range_a=max(samples_a) - min(samples_a),
        median_b=statistics.median(samples_b),
        range_b=max(samples_b) - min(samples_b),
    )
    if failed_a or failed_b:
        return settle(
            REPEAT_FAILED,
            f"{failed_a} baseline and {failed_b} candidate repeats failed; successes cannot stand in for them",
        )
    min_trials, max_trials = policy["min_trials"], policy["max_trials"]
    if len(samples_a) < min_trials or len(samples_b) < min_trials:
        return settle(UNVERIFIED, f"needs {min_trials} runs per side, have {len(samples_a)} and {len(samples_b)}")

    # Identical runs on both sides show no noise to size a band from (a real measurement is never that
    # steady): any difference would be judged against a zero-width band.
    noise = max(result["range_a"], result["range_b"])
    if not noise:
        return settle(UNVERIFIED, "the runs of both sides are identical, so no noise was observed")
    band = 2 * noise
    loss = result["median_a"] - result["median_b"]
    if direction == LOWER_IS_BETTER:
        loss = -loss
    if loss > band:
        return settle(REGRESSION)
    if -loss > band:
        return settle(IMPROVEMENT)

    scale = abs(result["median_a"])
    exhausted = min(len(samples_a), len(samples_b)) >= max_trials
    result["budget_exhausted"] = exhausted
    suffix = "; the repeat budget is exhausted" if exhausted else "; more repeats may narrow it"
    if scale == 0:
        return settle(UNVERIFIED, "a zero baseline median has no relative band")
    if band / scale > tolerances["max_uncertainty"]:
        return settle(
            UNVERIFIED,
            f"band {band / scale:.1%} exceeds the tolerated uncertainty {tolerances['max_uncertainty']:.1%}{suffix}",
        )
    if (max(loss, 0) + band) / scale >= tolerances["practical_threshold"]:
        return settle(UNVERIFIED, f"a loss of {tolerances['practical_threshold']:.1%} cannot be excluded{suffix}")
    return settle(WITHIN_NOISE)


def _repeat_samples(repeats, metric):
    """`(values, failed, perturbed)` of one cell's repeats for one metric.

    A repeat that is not `ok` is a failure the verdict must see. A repeat whose time was perturbed by
    profiling or instrumentation is a diagnostic, and a diagnostic cannot supply any speed metric.
    """
    values, failed, perturbed = [], 0, 0
    for repeat in repeats:
        if repeat.get("status") != "ok":
            failed += 1
        elif repeat.get("perturbed"):
            perturbed += 1
        elif metric in repeat.get("metrics", {}):
            values.append(repeat["metrics"][metric])
        else:
            failed += 1
    return values, failed, perturbed


def _cell_metric_verdict(cell, metric, base_cell, cand_cell, policy):
    entry = {"cell": cell, "metric": metric, "verdict": None, "note": ""}
    if base_cell is None or base_cell.get("outcome") in {"refused", "missing"}:
        entry.update(verdict=NO_BASELINE, note="the baseline has no record of this cell")
        return entry
    if cand_cell is None:
        entry.update(verdict=NOT_RUN, note="the candidate does not contain this required cell")
        return entry
    if cand_cell.get("outcome") == "refused":
        entry.update(verdict=REFUSED, note=cand_cell.get("reason", "refused without a reason"))
        return entry
    if cand_cell.get("outcome") == "missing":
        entry.update(verdict=NOT_RUN, note=cand_cell.get("reason", "missing"))
        return entry
    values_a, failed_a, perturbed_a = _repeat_samples(base_cell.get("repeats", []), metric)
    values_b, failed_b, perturbed_b = _repeat_samples(cand_cell.get("repeats", []), metric)
    if perturbed_a or perturbed_b:
        entry.update(
            verdict=INVALID,
            note=f"{perturbed_a + perturbed_b} perturbed diagnostic repeats cannot supply {metric}",
        )
        return entry
    entry.update(metric_verdict(values_a, values_b, policy, metric, failed_a=failed_a, failed_b=failed_b))
    entry.update(cell=cell, metric=metric)
    if entry["verdict"] == CANDIDATE_FAILED and failed_b:
        entry["note"] = f"all {failed_b} candidate repeats failed"
    return entry


def compare_receipts(baseline, candidate, spec, policy, fields=IDENTITY_FIELDS):
    """Compare two receipts over everything `spec` requires.

    A receipt is `{"identity": {...}, "cells": {cell_id: {"outcome"?, "reason"?, "repeats": [{"status",
    "perturbed"?, "metrics": {name: value}}]}}}`. `spec` is `{"cells": {cell_id: [metric, ...]}}` and `policy`
    the manifest's performance policy.
    The report is `complete` only when every required (cell, metric) has a compared verdict; otherwise it is
    `incomplete` and still lists every pair so the partial result stays inspectable. `passed` is complete with
    no regression, failure or invalid pair. Records that do not share a workload identity (over `fields`) are
    `incomparable` and compare nothing.
    """
    mismatches = identity_mismatches(baseline.get("identity", {}), candidate.get("identity", {}), fields)
    if mismatches:
        return {
            "status": INCOMPLETE,
            "passed": False,
            "verdict": INCOMPARABLE,
            "identity_mismatches": mismatches,
            "entries": [],
        }
    entries = [
        _cell_metric_verdict(
            cell, metric, baseline.get("cells", {}).get(cell), candidate.get("cells", {}).get(cell), policy
        )
        for cell, metrics in spec["cells"].items()
        for metric in metrics
    ]
    complete = bool(entries) and all(entry["verdict"] in COMPARED_VERDICTS for entry in entries)
    failing = any(entry["verdict"] in FAILING_VERDICTS for entry in entries)
    return {
        "status": COMPLETE if complete else INCOMPLETE,
        "passed": complete and not failing,
        "verdict": worst_verdict(entry["verdict"] for entry in entries),
        "identity_mismatches": [],
        "entries": entries,
    }


_SEVERITY = (
    INCOMPARABLE,
    INVALID,
    REGRESSION,
    CANDIDATE_FAILED,
    REPEAT_FAILED,
    NOT_RUN,
    REFUSED,
    NO_BASELINE,
    UNVERIFIED,
    NEW,
    IMPROVEMENT,
    WITHIN_NOISE,
)


def worst_verdict(verdicts):
    """The most serious verdict of a set; an empty set is unverified (nothing was compared)."""
    found = set(verdicts)
    return next((v for v in _SEVERITY if v in found), UNVERIFIED)


def coverage_matrix(spec, receipt):
    """Which required (cell, metric) pairs a receipt measured, left missing, was refused or left inconclusive."""
    matrix = {"measured": [], "missing": [], "refused": [], "inconclusive": []}
    for cell, metrics in spec["cells"].items():
        record = receipt.get("cells", {}).get(cell)
        for metric in metrics:
            pair = (cell, metric)
            if record is None or record.get("outcome") == "missing":
                matrix["missing"].append(pair)
            elif record.get("outcome") == "refused":
                matrix["refused"].append(pair)
            else:
                values, failed, perturbed = _repeat_samples(record.get("repeats", []), metric)
                clean = values and not failed and not perturbed
                matrix["measured" if clean else "inconclusive"].append(pair)
    return matrix


# --------------------------------------------------------------------------------------------------
# One request's measured window
# --------------------------------------------------------------------------------------------------
_WINDOW_TOLERANCE = 1e-9


def request_window_errors(request, requested_tokens):
    """Why one request's boundaries are not a consistent window; empty when they are.

    `request` carries the raw, unrounded boundaries a runner measured: `tokens`, `first_token_ms` and
    `last_token_ms` (null with no token), `ttft_ms`, `itl_ms`, `finish_ms`, `e2e_ms` and `steps` (executor
    steps the request cost). Every time is measured from the request start, so setup a reused request pays is
    inside TTFT. The window must satisfy E2E = TTFT + sum(ITL) + finish; token-to-token cost never includes
    finish; and the step counter must cover every delivered token (a token whose computation ran before the
    request began, in warmup, is not part of the request).

    Zero tokens have no TTFT or ITL: everything is finish. One token has TTFT and finish but no ITL. A
    request that stopped before `requested_tokens` is an early stop and cannot supply exact-work timing.
    """
    errors = []
    tokens = request.get("tokens")
    itls = request.get("itl_ms")
    if not (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0):
        return [f"tokens must be a nonnegative integer; got {tokens!r}"]
    if not isinstance(itls, list):
        return ["itl_ms must be an array"]
    for name in ("ttft_ms", "finish_ms", "e2e_ms"):
        if not (is_finite(request.get(name)) and request[name] >= 0):
            errors.append(f"{name} must be a finite nonnegative number; got {request.get(name)!r}")
    if any(not (is_finite(gap) and gap >= 0) for gap in itls):
        errors.append("itl_ms must hold finite nonnegative gaps")
    if errors:
        return errors

    if tokens < requested_tokens:
        errors.append(f"early stop: {tokens} of {requested_tokens} requested tokens delivered")
    expected_gaps = max(tokens - 1, 0)
    if len(itls) != expected_gaps:
        errors.append(f"{tokens} tokens have {expected_gaps} gaps; got {len(itls)} (the first gap may not be omitted)")
        return errors

    first, last = request.get("first_token_ms"), request.get("last_token_ms")
    if tokens == 0:
        if first is not None or last is not None:
            errors.append("a request with no token has no first or last token boundary")
        if request["ttft_ms"] != 0 or abs(request["finish_ms"] - request["e2e_ms"]) > _WINDOW_TOLERANCE:
            errors.append("with no token every millisecond of the request is finish")
        return errors
    if not (is_finite(first) and is_finite(last)):
        return errors + ["first_token_ms and last_token_ms are required once a token is delivered"]
    if request["ttft_ms"] != first:
        errors.append(f"ttft_ms {request['ttft_ms']!r} must be the first token boundary {first!r}")
    if abs(first + sum(itls) - last) > _WINDOW_TOLERANCE * max(1.0, abs(last)):
        errors.append(
            f"last token boundary {last!r} must be TTFT plus the gaps {first + sum(itls)!r}; "
            "completion is not last-token delivery"
        )
    total = request["ttft_ms"] + sum(itls) + request["finish_ms"]
    if abs(total - request["e2e_ms"]) > _WINDOW_TOLERANCE * max(1.0, abs(total)):
        errors.append(f"e2e_ms {request['e2e_ms']!r} must equal TTFT + sum(ITL) + finish {total!r}")
    steps = request.get("steps")
    if not (isinstance(steps, int) and not isinstance(steps, bool) and steps >= 0):
        errors.append(f"steps must be a nonnegative integer; got {steps!r}")
    elif steps < tokens:
        errors.append(
            f"{tokens} tokens were delivered by {steps} executor steps: "
            "computation hidden outside the request window (prefill yields token 1, each decode step one more)"
        )
    return errors


def request_tpot_ms(request):
    """Token-to-token cost: mean ITL. Undefined (None) below two tokens; finish never enters it."""
    itls = request["itl_ms"]
    return sum(itls) / len(itls) if itls else None


# --------------------------------------------------------------------------------------------------
# Preparation across layers
# --------------------------------------------------------------------------------------------------
COLD_EMPTY = "cold-process-empty-artifacts"
COLD_POPULATED = "cold-process-populated-artifacts"
REUSED_DRIVER = "reused-prepared-driver"
PREPARATION_STATES = (COLD_EMPTY, COLD_POPULATED, REUSED_DRIVER)

# Each layer's own producer counter (loading is timed separately as `open_ms`: it happens before any request).
# A layer no producer exposes is reported as null (Unknown), and a receipt with an Unknown layer is
# incomplete for that layer rather than silently warm.
PREPARATION_LAYERS = (
    "plan",
    "entries",
    "generated_bodies",
    "artifact_builds",
    "artifact_hits",
    "runtime_pipelines",
    "recordings",
    "graph_instantiations",
)


def preparation_shape_errors(preparation, *, require_state):
    """Why `preparation` is not the block a runner reports; empty when its fields are well formed.

    A layer a producer cannot count is null (Unknown) and well formed. `state` may be null only when the
    caller does not require a declared state (a generic harness row; a bank receipt always declares one).
    """
    if not isinstance(preparation, dict):
        return [f"preparation must be an object; got {preparation!r}"]
    errors = []
    state = preparation.get("state")
    if state is None and not require_state:
        pass
    elif state not in PREPARATION_STATES:
        errors.append(f"preparation state must be one of {PREPARATION_STATES}; got {state!r}")
    layers = preparation.get("layers")
    if not isinstance(layers, dict):
        return errors + [f"preparation layers must be an object; got {layers!r}"]
    for layer in PREPARATION_LAYERS:
        entry = layers.get(layer)
        if not isinstance(entry, dict):
            errors.append(f"preparation layer {layer!r} must be an object; got {entry!r}")
            continue
        for phase in ("prepare", "repeat"):
            value = entry.get(phase)
            if value is not None and not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
                errors.append(f"{layer}.{phase} must be a nonnegative integer or null; got {value!r}")
    return errors


def preparation_report(preparation):
    """`{"errors": [...], "unknown": [...], "complete": bool}` for one preparation receipt.

    `preparation` is `{"state", "artifact_entries_at_start", "layers": {layer: {"prepare": n|null,
    "repeat": n|null}}}`: `prepare` is the work the declared prepared set cost, `repeat` the work repeating
    it added. Every layer is held to its own counter; repeating the declared set must add no work to any
    layer, whatever the state. The states differ in what `prepare` must show: cold-empty did all the work
    including artifact builds, cold-populated served artifacts from the cache, a reused driver did none.
    """
    errors = preparation_shape_errors(preparation, require_state=True)
    if errors:
        return {"errors": errors, "unknown": [], "complete": False}
    state = preparation["state"]
    unknown = []
    count = {}
    for layer in PREPARATION_LAYERS:
        for phase in ("prepare", "repeat"):
            value = preparation["layers"][layer].get(phase)
            if value is None:
                unknown.append(f"{layer}.{phase}")
            else:
                count[(layer, phase)] = value
    errors = []
    for layer in PREPARATION_LAYERS:
        added = count.get((layer, "repeat"))
        if added:
            errors.append(f"repeating the prepared set added {added} {layer} work: the layer is not warm")

    entries = preparation.get("artifact_entries_at_start")
    prepared = {layer: count.get((layer, "prepare")) for layer in PREPARATION_LAYERS}
    if state == COLD_EMPTY:
        if entries != 0:
            errors.append(f"a cold-artifact claim started with {entries!r} cached artifacts")
        if prepared["artifact_builds"] == 0:
            errors.append("a cold-artifact claim built no artifact")
        if prepared["artifact_hits"]:
            errors.append(f"a cold-artifact claim served {prepared['artifact_hits']} artifacts from cache")
    elif state == COLD_POPULATED:
        if not entries:
            errors.append("a populated-artifact claim started with no cached artifacts")
        if prepared["artifact_builds"]:
            errors.append(f"a populated-artifact claim still built {prepared['artifact_builds']} artifacts")
        if prepared["artifact_hits"] == 0:
            errors.append("a populated-artifact claim served nothing from the cache")
    else:
        for layer, work in prepared.items():
            if work:
                errors.append(f"a reused prepared driver did {work} {layer} work to prepare")
    return {"errors": errors, "unknown": unknown, "complete": not errors and not unknown}


# --------------------------------------------------------------------------------------------------
# Harness result rows as receipts
# --------------------------------------------------------------------------------------------------
# What a results.jsonl row states about its workload. Backend and device are the cell key, not identity.
ROW_IDENTITY_FIELDS = ("precision", "prompt_tokens", "requested_output")
HEADLINE_CELL = "headline"


def _row_identity(rows):
    row = rows[0]
    return {
        "precision": row.get("precision"),
        "prompt_tokens": row.get("prompt_tokens"),
        "requested_output": row.get("osl", row.get("gen_tokens")),
    }


def rows_receipt(rows, policy):
    """One cell's result rows as a receipt: the headline plus one cell per curve point.

    Each row is one repeat. A row that is not `ok` is a failed repeat in every cell. A curve row also
    contributes, per context length, its TPOT and TTFT medians, so the long-context points are compared and
    not only the smallest-context headline.
    """
    ok = [row for row in rows if row.get("status") == "ok"]
    cells = {
        HEADLINE_CELL: {
            "repeats": [
                {"status": "ok", "metrics": {policy["headline_metric"]: row.get(policy["headline_metric"])}}
                for row in ok
            ]
            + [{"status": "failed"} for row in rows if row.get("status") != "ok"]
        }
    }
    for row in ok:
        for point in row.get("curve") or []:
            metrics = {name: point[name]["p50"] for name in policy["curve_point_metrics"]}
            cells.setdefault(f"isl={point['isl']}", {"repeats": []})["repeats"].append(
                {"status": "ok", "metrics": metrics}
            )
    for name, record in cells.items():
        if name != HEADLINE_CELL:
            record["repeats"] += [{"status": "failed"} for row in rows if row.get("status") != "ok"]
    return {"identity": _row_identity(ok or rows), "cells": cells}


def rows_spec(baseline_receipt, policy):
    """The coverage the baseline sets: the headline and every curve point it measured."""
    return {
        "cells": {
            name: [policy["headline_metric"]] if name == HEADLINE_CELL else list(policy["curve_point_metrics"])
            for name in baseline_receipt["cells"]
        }
    }
