"""Scenario-specific validation for untrusted benchmark runner JSON."""

import math
import re
from typing import TypeGuard

CURVE_KIND = "decode-curve"

# Frameworks whose runner is a binary built from this repository. Their result must say which build
# produced it (`build_sha`) and on which backend, so a row is attributable to one binary and one backend.
BUILD_IDENTITY_FRAMEWORKS = frozenset({"poot"})

_BUILD_SHA_PATTERN = re.compile(r"[0-9a-f]{40}(-dirty)?")


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
_CURVE_ITERATION_RULES = (
    (
        "ttft_ms",
        lambda value: _is_finite_number(value) and value >= 0,
        "a finite nonnegative number",
    ),
    (
        "e2e_ms",
        lambda value: _is_finite_number(value) and value >= 0,
        "a finite nonnegative number",
    ),
    ("itl_ms", lambda value: isinstance(value, list), "an array"),
)


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
        curve_errors, curve = _validate_curve_result(result, scenario)
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

    if errors:
        raise ResultValidationError(errors)
    if curve is not None:
        prepared = dict(result)
        prepared["curve"] = curve
        return prepared
    return result


def _build_identity_errors(result, backend):
    errors = []
    build_sha = result.get("build_sha")
    if "build_sha" not in result:
        errors.append("missing required identity field 'build_sha'")
    elif not (isinstance(build_sha, str) and _BUILD_SHA_PATTERN.fullmatch(build_sha)):
        errors.append(
            f"build_sha must be a 40-digit git sha, optionally ending in '-dirty'; got {build_sha!r}"
        )
    if "backend" not in result:
        errors.append("missing required identity field 'backend'")
    elif backend is not None and result["backend"] != backend:
        errors.append(
            f"backend mismatch: requested {backend!r}, reported {result['backend']!r}"
        )
    return errors


def _validate_curve_result(result, scenario):
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

    itl_count = (
        requested_osl - 1 if _is_int(requested_osl) and requested_osl > 1 else None
    )
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
            errors.extend(_check_fields(iteration, _CURVE_ITERATION_RULES, iter_prefix))
            ttft = iteration.get("ttft_ms")
            e2e = iteration.get("e2e_ms")
            if _is_finite_number(ttft) and _is_finite_number(e2e):
                if e2e <= ttft:
                    errors.append(f"{iter_prefix}e2e_ms must be greater than ttft_ms")
                elif not _is_finite_positive_number(
                    _curve_tpot_ms(e2e, ttft, requested_osl)
                ):
                    errors.append(
                        f"{iter_prefix}derived TPOT must be a finite positive number"
                    )
            itls = iteration.get("itl_ms")
            if not isinstance(itls, list):
                continue
            if itl_count is not None and len(itls) != itl_count:
                errors.append(
                    f"{iter_prefix}itl_ms must contain {itl_count} samples; "
                    f"got {len(itls)}"
                )
            for sample_index, sample in enumerate(itls):
                if not (_is_finite_number(sample) and sample >= 0):
                    errors.append(
                        f"{iter_prefix}itl_ms[{sample_index}] must be a finite "
                        f"nonnegative number; got {sample!r}"
                    )
    if errors:
        return errors, None

    aggregated = _aggregate_curve(curve, requested_osl)
    errors.extend(_validate_curve_aggregates(aggregated))
    return errors, aggregated


def _is_finite_positive_number(value):
    return _is_finite_number(value) and value > 0


def _curve_tpot_ms(e2e, ttft, osl):
    try:
        return (e2e - ttft) / (osl - 1)
    except (OverflowError, TypeError, ZeroDivisionError):
        return None


def _pct(values, percentile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percentile / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _pcts(values):
    return {name: _pct(values, percentile) for name, percentile in _PERCENTILES}


_PERCENTILES = (("p50", 50), ("p90", 90), ("p99", 99))
_CURVE_AGGREGATE_FIELDS = (
    ("ttft_ms", lambda value: _is_finite_number(value) and value >= 0, "nonnegative"),
    ("tpot_ms", _is_finite_positive_number, "positive"),
    ("itl_ms", lambda value: _is_finite_number(value) and value >= 0, "nonnegative"),
)


def _aggregate_curve(raw_curve, osl):
    points = []
    for point in raw_curve:
        iterations = point["iters"]
        ttfts = [iteration["ttft_ms"] for iteration in iterations]
        all_itls = [gap for iteration in iterations for gap in iteration["itl_ms"]]
        tpots = [
            _curve_tpot_ms(iteration["e2e_ms"], iteration["ttft_ms"], osl)
            for iteration in iterations
        ]
        tpot_percentiles = _pcts(tpots)
        tpot_median = tpot_percentiles["p50"]
        try:
            throughput = 1000.0 / tpot_median
        except (OverflowError, TypeError, ZeroDivisionError):
            throughput = None
        points.append(
            {
                "isl": point["isl"],
                "osl": osl,
                "prompt_tokens": point["prompt_tokens"],
                "iters": len(iterations),
                "ttft_ms": _pcts(ttfts),
                "tpot_ms": tpot_percentiles,
                "itl_ms": _pcts(all_itls),
                "decode_tok_s": throughput,
            }
        )
    points.sort(key=lambda point: point["isl"])
    return points


def _validate_curve_aggregates(curve):
    errors = []
    for point_index, point in enumerate(curve):
        prefix = f"curve[{point_index}]."
        for field, predicate, range_description in _CURVE_AGGREGATE_FIELDS:
            for percentile, _ in _PERCENTILES:
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
