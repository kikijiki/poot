"""Scenario-specific validation for untrusted benchmark runner JSON."""

import math
import re
from typing import TypeGuard

try:
    from .performance_receipt import (
        BUILD_IDENTITY_FRAMEWORKS,
        preparation_shape_errors,
        request_tpot_ms,
        request_window_errors,
    )
except ImportError:  # direct `python harness/bench.py` execution
    from performance_receipt import (
        BUILD_IDENTITY_FRAMEWORKS,
        preparation_shape_errors,
        request_tpot_ms,
        request_window_errors,
    )

CURVE_KIND = "decode-curve"

# Frameworks whose runner is a binary built from this repository (`BUILD_IDENTITY_FRAMEWORKS`) state which build
# produced a result and on which backend, so a row is attributable to one binary and one backend. Every other
# runner is an external tool: it names its own version (`framework_version`) and has no build of ours.
# The identity a repository-built runner states in every result, a refusal included: the commit, the cargo
# profile and features, and the compiler. Only a release build is a result.
BUILD_IDENTITY_FIELDS = ("build_sha", "build_profile", "build_features", "rustc_version")
# What a repository-built runner's measurement states beyond its build: the adapter and driver the executor
# reports, the digest of the weights it loaded and the dtype it computes in. The result's `precision` is the
# dtype the store holds, read by the runner after the load.
MEASUREMENT_IDENTITY_FIELDS = ("adapter", "driver_version", "checkpoint_digest", "compute_dtype")
RELEASE_PROFILE = "release"

# What timing a framework's runner claims. A runner that CLAIMS the request window reports every request's raw
# boundaries (tokens, first/last token, finish, steps) and its result is judged by `request_window_errors`,
# the one derivation of TTFT, ITL, finish, E2E and TPOT for that capability. Every other runner reports
# row-level timing only (ttft, e2e and per-token gaps): its rows carry `timing_claim: "row-level"`, compare on
# the row-level identity subset, and TPOT is (e2e - ttft) / (osl - 1). One derivation per capability, never both.
REQUEST_WINDOW = "request-window"
ROW_LEVEL = "row-level"
WINDOW_CLAIMING_FRAMEWORKS = frozenset({"poot"})


# Frameworks whose runner reports the device's own time for a profiled window (`device_time`). Every other
# runner has no device timestamps to report, and a `device_time` from one is refused rather than believed.
DEVICE_TIME_CLAIMING_FRAMEWORKS = frozenset({"poot"})


def timing_claim(framework):
    """The timing capability `framework`'s runner claims."""
    return REQUEST_WINDOW if framework in WINDOW_CLAIMING_FRAMEWORKS else ROW_LEVEL


_BUILD_SHA_PATTERN = re.compile(r"[0-9a-f]{40}(-dirty)?")
_CHECKPOINT_DIGEST_PATTERN = re.compile(r"blake3:[0-9a-f]{64}")
_IDENTITY_PATTERNS = {
    "build_sha": (_BUILD_SHA_PATTERN, "a 40-digit git sha, optionally ending in '-dirty'"),
    "checkpoint_digest": (_CHECKPOINT_DIGEST_PATTERN, "'blake3:' and 64 hex digits"),
}


class ResultValidationError(ValueError):
    """A runner payload that cannot be promoted to a trusted benchmark result."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_curve(scenario):
    return scenario.get("kind") == CURVE_KIND


def _check_fields(payload, rules, prefix=""):
    errors = []
    for field, predicate, description in rules:
        label = f"{prefix}{field}"
        if field not in payload:
            errors.append(f"missing required field {label!r}")
        elif not predicate(payload[field]):
            errors.append(f"{label} must be {description}; got {payload[field]!r}")
    return errors


_NONNEGATIVE_NUMBER_FIELDS = (
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "decode_tok_s",
)
_OPTIONAL_NONNEGATIVE_NUMBER_FIELDS = (
    "ttft_ms_stdev",
    "decode_tok_s_stdev",
)
_SINGLE_FIELD_RULES = (
    (
        "prompt_tokens",
        lambda value: _is_int(value) and value >= 0,
        "a nonnegative integer",
    ),
    ("gen_tokens", lambda value: _is_int(value) and value > 0, "a positive integer"),
    ("iterations", lambda value: _is_int(value) and value > 0, "a positive integer"),
    *(
        (
            field,
            lambda value: _is_finite_number(value) and value >= 0,
            "a finite nonnegative number",
        )
        for field in _NONNEGATIVE_NUMBER_FIELDS
    ),
)
# A timed request, curve iteration or single run alike, carries its raw boundaries; the window itself is
# judged by `request_window_errors`, the one definition of TTFT, ITL, finish and E2E.
_REQUEST_FIELDS = (
    "tokens",
    "first_token_ms",
    "last_token_ms",
    "ttft_ms",
    "itl_ms",
    "finish_ms",
    "e2e_ms",
    "steps",
)


def _row_level_errors(iteration, osl, prefix, framework):
    """A row-level runner's iteration: ttft, e2e and the osl - 1 inter-token gaps."""
    errors = _check_fields(iteration, _ROW_LEVEL_ITERATION_RULES, prefix)
    ttft, e2e = iteration.get("ttft_ms"), iteration.get("e2e_ms")
    if _is_finite_number(ttft) and _is_finite_number(e2e):
        if e2e <= ttft:
            errors.append(f"{prefix}e2e_ms must be greater than ttft_ms")
        elif not _is_finite_positive_number(_row_level_tpot_ms(e2e, ttft, osl)):
            errors.append(f"{prefix}derived TPOT must be a finite positive number")
    itls = iteration.get("itl_ms")
    if isinstance(itls, list):
        if len(itls) != osl - 1:
            errors.append(
                f"{prefix}itl_ms must contain {osl - 1} samples; got {len(itls)}: "
                f"{framework} generated {len(itls) + 1}, expected {osl} tokens"
            )
        for index, sample in enumerate(itls):
            if not (_is_finite_number(sample) and sample >= 0):
                errors.append(f"{prefix}itl_ms[{index}] must be a finite nonnegative number; got {sample!r}")
    return errors


def _row_level_tpot_ms(e2e, ttft, osl):
    try:
        return (e2e - ttft) / (osl - 1)
    except (OverflowError, TypeError, ZeroDivisionError):
        return None


_ROW_LEVEL_ITERATION_RULES = (
    ("ttft_ms", lambda value: _is_finite_number(value) and value >= 0, "a finite nonnegative number"),
    ("e2e_ms", lambda value: _is_finite_number(value) and value >= 0, "a finite nonnegative number"),
    ("itl_ms", lambda value: isinstance(value, list), "an array"),
)


def _request_errors(request, requested_tokens, prefix):
    missing = [f"missing required field {prefix}{field!r}" for field in _REQUEST_FIELDS if field not in request]
    if missing:
        return missing
    return [f"{prefix}{error}" for error in request_window_errors(request, requested_tokens)]


def runner_refusal(result, *, framework, backend=None):
    """The reason a runner refused the cell, or None when it did not refuse.

    A runner refuses a model by reporting `"status": "unsupported"` and, as `reason`, the refusal exactly as its
    tool reports it. The harness records that text; it does not decide support itself. A refusal from a runner
    in BUILD_IDENTITY_FRAMEWORKS still names the build and backend it came from.
    """
    if not isinstance(result, dict) or result.get("status") != "unsupported":
        return None
    errors = []
    reason = result.get("reason")
    if not (isinstance(reason, str) and reason.strip()):
        errors.append(f"an unsupported result needs a non-empty reason; got {reason!r}")
    if framework in BUILD_IDENTITY_FRAMEWORKS:
        errors.extend(_build_identity_errors(result, backend))
    if errors:
        raise ResultValidationError(errors)
    return reason


def validate_runner_result(result, *, framework, precision, scenario, gen_tokens, backend=None):
    """Validate and prepare one runner object before the harness trusts any measurement.

    `backend` is the backend the harness asked the runner to use. A runner in BUILD_IDENTITY_FRAMEWORKS must
    report it back unchanged, together with the git sha it was built from.
    """
    if not isinstance(result, dict):
        raise ResultValidationError(
            [f"runner result must be an object; got {type(result).__name__}"]
        )

    errors = []
    for field, expected in (("framework", framework), ("precision", precision)):
        if field not in result:
            errors.append(f"missing required identity field {field!r}")
        elif result[field] != expected:
            errors.append(
                f"{field} mismatch: requested {expected!r}, reported {result[field]!r}"
            )

    if framework in BUILD_IDENTITY_FRAMEWORKS:
        errors.extend(_build_identity_errors(result, backend))
        errors.extend(_measurement_identity_errors(result))
        errors.extend(preparation_shape_errors(result.get("preparation"), require_state=False))
    elif not (isinstance(result.get("framework_version"), str) and result["framework_version"].strip()):
        errors.append(f"an external runner names the version it measured; framework_version is {result.get('framework_version')!r}")

    errors.extend(_device_time_claim_errors(result, framework))

    if "error" in result:
        errors.append(f"runner reported error: {result['error']!r}")
    if "status" in result and result["status"] != "ok":
        errors.append(f"runner reported status {result['status']!r}")

    expected_mode = CURVE_KIND if _is_curve(scenario) else "single"
    if "mode" in result and result["mode"] != expected_mode:
        errors.append(
            f"mode mismatch: requested {expected_mode!r}, reported {result['mode']!r}"
        )

    caveats = result.get("caveats")
    if caveats is not None and (
        not isinstance(caveats, list)
        or any(not isinstance(caveat, str) for caveat in caveats)
    ):
        errors.append("caveats must be an array of strings")

    if "backend" in result and not isinstance(result["backend"], str):
        errors.append(f"backend must be a string; got {result['backend']!r}")
    version = result.get("framework_version")
    if version is not None and not isinstance(version, str):
        errors.append(f"framework_version must be a string; got {version!r}")
    if "profiled" in result and not isinstance(result["profiled"], bool):
        errors.append(f"profiled must be a boolean; got {result['profiled']!r}")
    for field in _OPTIONAL_NONNEGATIVE_NUMBER_FIELDS:
        if field in result and not (
            _is_finite_number(result[field]) and result[field] >= 0
        ):
            errors.append(
                f"{field} must be a finite nonnegative number; got {result[field]!r}"
            )
    if "self_reported_vram_bytes" in result and not (
        _is_int(result["self_reported_vram_bytes"])
        and result["self_reported_vram_bytes"] >= 0
    ):
        errors.append(
            "self_reported_vram_bytes must be a nonnegative integer; "
            f"got {result['self_reported_vram_bytes']!r}"
        )

    curve = None
    if _is_curve(scenario):
        curve_errors, curve = _validate_curve_result(result, scenario, timing_claim(framework), framework)
        errors.extend(curve_errors)
    else:
        errors.extend(_check_fields(result, _SINGLE_FIELD_RULES))
        if result.get("gen_tokens") != gen_tokens:
            errors.append(
                "gen_tokens mismatch: "
                f"requested {gen_tokens!r}, reported {result.get('gen_tokens')!r}"
            )
        if (
            _is_finite_number(result.get("e2e_ms"))
            and _is_finite_number(result.get("ttft_ms"))
            and result["e2e_ms"] < result["ttft_ms"]
        ):
            errors.append("e2e_ms must be greater than or equal to ttft_ms")
        if timing_claim(framework) == REQUEST_WINDOW:
            errors.extend(_single_run_errors(result, gen_tokens))

    if errors:
        raise ResultValidationError(errors)
    if curve is not None:
        prepared = dict(result)
        prepared["curve"] = curve
        return prepared
    return result


def _device_time_errors(value, prefix):
    """`device_time` is `null` (not profiled), `{"kind": "unknown"}` (the device reports none) or
    `{"kind": "measured", "sum_of_dispatch_ms", "device_span_ms"}` with at least one measure a finite
    nonnegative number and the other a number or null. A measured time with neither measure is the
    zero-for-unknown the type exists to prevent."""
    if value is None:
        return []
    if not isinstance(value, dict):
        return [f"{prefix}device_time must be null or an object; got {value!r}"]
    kind = value.get("kind")
    if kind == "unknown":
        extra = sorted(set(value) - {"kind"})
        return [f"{prefix}device_time unknown carries {extra!r}"] if extra else []
    if kind != "measured":
        return [f"{prefix}device_time kind must be 'measured' or 'unknown'; got {kind!r}"]
    errors = []
    measures = []
    for field in ("sum_of_dispatch_ms", "device_span_ms"):
        measure = value.get(field)
        if measure is None:
            continue
        if _is_finite_number(measure) and measure >= 0:
            measures.append(measure)
        else:
            errors.append(f"{prefix}device_time.{field} must be a finite nonnegative number or null; got {measure!r}")
    if not errors and not measures:
        errors.append(f"{prefix}device_time measured carries neither measure; report unknown instead")
    return errors


def _device_time_claim_errors(result, framework):
    """Every `device_time` of `result` (the row's, and each raw curve point's) is well formed and comes from a
    runner that claims device timing."""
    carriers = [("", result)]
    curve = result.get("curve")
    if isinstance(curve, list):
        carriers += [(f"curve[{i}].", p) for i, p in enumerate(curve) if isinstance(p, dict)]
    errors = []
    for prefix, carrier in carriers:
        if "device_time" not in carrier:
            continue
        if framework not in DEVICE_TIME_CLAIMING_FRAMEWORKS:
            errors.append(f"{prefix}device_time reported by {framework!r}, which does not claim device timing")
        else:
            errors.extend(_device_time_errors(carrier["device_time"], prefix))
    return errors


def _single_run_errors(result, gen_tokens):
    runs = result.get("runs")
    if not isinstance(runs, list) or not runs:
        return [f"runs must be a non-empty array of the timed requests; got {runs!r}"]
    errors = []
    if result.get("iterations") != len(runs):
        errors.append(f"iterations {result.get('iterations')!r} must equal the {len(runs)} timed runs")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"runs[{index}] must be an object; got {type(run).__name__}")
        else:
            errors.extend(_request_errors(run, gen_tokens, f"runs[{index}]."))
    return errors


def _identity_field_errors(result, fields):
    """Each of `fields` is present as a non-empty string, matching its pattern where it has one."""
    errors = []
    for field in fields:
        if field not in result:
            errors.append(f"missing required identity field {field!r}")
            continue
        value = result[field]
        pattern = _IDENTITY_PATTERNS.get(field)
        if pattern is not None:
            if not (isinstance(value, str) and pattern[0].fullmatch(value)):
                errors.append(f"{field} must be {pattern[1]}; got {value!r}")
        elif field == "build_features":
            # The runner's cargo features, comma-separated: empty when it has none.
            if not isinstance(value, str):
                errors.append(f"build_features must be a string; got {value!r}")
        elif not (isinstance(value, str) and value.strip()):
            errors.append(f"{field} must be a non-empty string; got {value!r}")
    return errors


def _build_identity_errors(result, backend):
    errors = _identity_field_errors(result, BUILD_IDENTITY_FIELDS)
    if result.get("build_profile", RELEASE_PROFILE) != RELEASE_PROFILE:
        errors.append(
            f"build_profile must be {RELEASE_PROFILE!r}: a {result['build_profile']!r} build is not a result"
        )
    if "backend" not in result:
        errors.append("missing required identity field 'backend'")
    elif backend is not None and result["backend"] != backend:
        errors.append(
            f"backend mismatch: requested {backend!r}, reported {result['backend']!r}"
        )
    return errors


def _measurement_identity_errors(result):
    errors = _identity_field_errors(result, MEASUREMENT_IDENTITY_FIELDS)
    if "profiled" not in result:
        errors.append("missing required identity field 'profiled'")
    chunk = result.get("prefill_chunk")
    if not _is_int(chunk) or chunk <= 0:
        errors.append(f"prefill_chunk must be a positive integer; got {chunk!r}")
    return errors


def committed_row_errors(row):
    """Why a stored row is not attributable, or an empty list. A row of a repository-built runner states its
    whole identity: a build from a clean commit under the release profile, and for a measurement the adapter,
    driver, checkpoint digest and dtypes. Every other row names the version of the tool that produced it.
    Rows that are errors say nothing about a measurement and are held to nothing here."""
    status = row.get("status")
    if status not in ("ok", "unsupported"):
        return []
    if row.get("framework") not in BUILD_IDENTITY_FRAMEWORKS:
        if status == "ok" and not row.get("framework_version"):
            return ["an ok row of an external tool names its framework_version"]
        return []
    fields = BUILD_IDENTITY_FIELDS + (MEASUREMENT_IDENTITY_FIELDS if status == "ok" else ())
    errors = _identity_field_errors(row, fields)
    errors += _identity_field_errors(row, ("backend",))
    if str(row.get("build_sha", "")).endswith("-dirty"):
        errors.append("build_sha ends in '-dirty': a committed result comes from a clean commit")
    if row.get("build_profile", RELEASE_PROFILE) != RELEASE_PROFILE:
        errors.append(f"build_profile is {row['build_profile']!r}: only a release build is a result")
    if row.get("profiled"):
        errors.append("profiled: a device-timed run is a diagnostic, stored under results/diagnostics/")
    return errors


def _validate_curve_result(result, scenario, claim, framework):
    errors = []
    requested_osl = scenario.get("osl", 128)
    requested_isls = list(scenario.get("isl", []))
    if not (_is_int(requested_osl) and requested_osl > 1):
        errors.append(
            f"requested OSL must be an integer greater than one; got {requested_osl!r}"
        )

    osl = result.get("osl")
    if not (_is_int(osl) and osl > 1):
        errors.append(f"osl must be an integer greater than one; got {osl!r}")
    elif osl != requested_osl:
        errors.append(f"osl mismatch: requested {requested_osl!r}, reported {osl!r}")

    curve = result.get("curve")
    if not isinstance(curve, list) or not curve:
        errors.append("curve must be a non-empty array")
        return errors, None

    reported_isls = [
        point.get("isl") if isinstance(point, dict) else None for point in curve
    ]
    coverage_matches = len(reported_isls) == len(requested_isls) and all(
        reported_isls.count(isl) == 1 for isl in requested_isls
    )
    if not coverage_matches:
        errors.append(
            "curve ISL coverage mismatch: "
            f"requested {requested_isls!r}, reported {reported_isls!r}"
        )

    # An OSL of one or less is already an error above; its windows are not judged on top of it.
    osl_valid = _is_int(requested_osl) and requested_osl > 1
    for point_index, point in enumerate(curve):
        prefix = f"curve[{point_index}]."
        if not isinstance(point, dict):
            errors.append(
                f"curve[{point_index}] must be an object; got {type(point).__name__}"
            )
            continue
        errors.extend(
            _check_fields(
                point,
                (
                    (
                        "isl",
                        lambda value: _is_int(value) and value > 0,
                        "a positive integer",
                    ),
                    (
                        "prompt_tokens",
                        lambda value: _is_int(value) and value > 0,
                        "a positive integer",
                    ),
                    (
                        "iters",
                        lambda value: isinstance(value, list) and bool(value),
                        "a non-empty array",
                    ),
                ),
                prefix,
            )
        )
        isl = point.get("isl")
        if point.get("prompt_tokens") != isl:
            errors.append(
                f"{prefix}prompt_tokens must equal its ISL {isl!r}; "
                f"got {point.get('prompt_tokens')!r}"
            )

        iterations = point.get("iters")
        if not isinstance(iterations, list):
            continue
        for iteration_index, iteration in enumerate(iterations):
            iter_prefix = f"{prefix}iters[{iteration_index}]."
            if not isinstance(iteration, dict):
                errors.append(
                    f"{prefix}iters[{iteration_index}] must be an object; "
                    f"got {type(iteration).__name__}"
                )
                continue
            if not osl_valid:
                continue
            if claim == REQUEST_WINDOW:
                window = _request_errors(iteration, requested_osl, iter_prefix)
                errors.extend(window)
                if not window and not _is_finite_positive_number(request_tpot_ms(iteration)):
                    errors.append(f"{iter_prefix}token-to-token TPOT must be a finite positive number")
            else:
                errors.extend(_row_level_errors(iteration, requested_osl, iter_prefix, framework))
    if errors:
        return errors, None

    aggregated = _aggregate_curve(curve, requested_osl, claim)
    errors.extend(_validate_curve_aggregates(aggregated))
    return errors, aggregated


def _is_finite_positive_number(value):
    return _is_finite_number(value) and value > 0


def _pct(values, percentile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percentile / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# A tail percentile of fewer samples than this is an interpolation between a handful of points, not an estimate
# of the tail: five iterations make "p99" the largest sample. Below it a summary is p50 with the observed range.
TAIL_PERCENTILE_MIN_SAMPLES = 20
_TAIL_PERCENTILES = (("p90", 90), ("p99", 99))


def _summary(values):
    """The median and observed range of `values`, plus p90 and p99 once there are enough samples to estimate a tail."""
    summary = {"p50": _pct(values, 50), "min": min(values), "max": max(values)}
    if len(values) >= TAIL_PERCENTILE_MIN_SAMPLES:
        summary.update({name: _pct(values, percentile) for name, percentile in _TAIL_PERCENTILES})
    return summary


_CURVE_AGGREGATE_FIELDS = (
    ("ttft_ms", lambda value: _is_finite_number(value) and value >= 0, "nonnegative"),
    ("tpot_ms", _is_finite_positive_number, "positive"),
    ("itl_ms", lambda value: _is_finite_number(value) and value >= 0, "nonnegative"),
)


def _aggregate_curve(raw_curve, osl, claim):
    points = []
    for point in raw_curve:
        iterations = point["iters"]
        ttfts = [iteration["ttft_ms"] for iteration in iterations]
        all_itls = [gap for iteration in iterations for gap in iteration["itl_ms"]]
        if claim == REQUEST_WINDOW:
            tpots = [request_tpot_ms(iteration) for iteration in iterations]
        else:
            tpots = [_row_level_tpot_ms(i["e2e_ms"], i["ttft_ms"], osl) for i in iterations]
        tpot_percentiles = _summary(tpots)
        tpot_median = tpot_percentiles["p50"]
        try:
            throughput = 1000.0 / tpot_median
        except (OverflowError, TypeError, ZeroDivisionError):
            throughput = None
        entry = {
                "isl": point["isl"],
                "osl": osl,
                "prompt_tokens": point["prompt_tokens"],
                "iters": len(iterations),
                "ttft_ms": _summary(ttfts),
                "tpot_ms": tpot_percentiles,
                "itl_ms": _summary(all_itls),
                "decode_tok_s": throughput,
        }
        if "device_time" in point:
            entry["device_time"] = point["device_time"]
        points.append(entry)
    points.sort(key=lambda point: point["isl"])
    return points


def _validate_curve_aggregates(curve):
    errors = []
    for point_index, point in enumerate(curve):
        prefix = f"curve[{point_index}]."
        for field, predicate, range_description in _CURVE_AGGREGATE_FIELDS:
            for percentile in point[field]:
                value = point[field][percentile]
                if not predicate(value):
                    errors.append(
                        f"derived {prefix}{field}.{percentile} must be a finite "
                        f"{range_description} number; got {value!r}"
                    )
        throughput = point["decode_tok_s"]
        if not _is_finite_positive_number(throughput):
            errors.append(
                f"derived {prefix}decode_tok_s must be a finite positive number; "
                f"got {throughput!r}"
            )
    return errors
