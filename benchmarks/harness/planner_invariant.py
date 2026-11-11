"""The planner invariant on the device (POOT-1520): the schedule the planner chooses for a narrowable case never
loses to the exact schedule of the same case.

The bank defines each narrowable case twice, an exact twin and a `-narrow` twin that differs only in its
`numerics.operand_precision` (the narrow twin runs under the `narrow` policy, so the planner chooses between its
exact and narrowed schedules by its cost model; the exact twin runs under the `exact` policy). A cost model
that is wrong about a device, a register spill the model has no term for, or a schedule the model never priced
shows up as the narrow twin measuring slower than the exact one on the same device (a pair that realized one plan
is `same plan`: the planner chose exact, and a time difference between two runs of one kernel set is drift): POOT-1503 (ROCm decode
vocabulary head), POOT-1505 (ROCm packed experts at short prompts) and POOT-1509 (PTX packed contraction at M of
about 128).

`pair_verdicts` judges every pair that both has a verified receipt on a backend with the one noise-aware comparator
(`performance_receipt.metric_verdict`, the exact twin as the baseline and the narrow twin as the candidate) on the
device-clock samples. The two receipts must come from one binary on one device: a pair measured in two sessions
says nothing about the planner. A pair with a missing or incomparable side is `unmeasured`, never a pass.
"""

import json
from pathlib import Path

try:
    from . import performance_bank as bank
    from . import performance_receipt as receipts
    from . import results_root
except ImportError:  # direct execution from harness/
    import performance_bank as bank
    import performance_receipt as receipts
    import results_root

# The verdict of a pair whose twins realized one plan: the invariant holds by identity.
SAME_PLAN = "same plan"

# The numerics fields that say how a case's operands are read; every other field must agree for two cases to be twins.
PRECISION_FIELDS = ("operand_precision", "operand_rounding")
# What a twin's definition may differ in besides its numerics and the prose that names it.
DESCRIPTIVE_FIELDS = ("id", "regime", "numerics", "identity_digest", "fixture_id", "schedule_id", "cache_id")


def _without_precision(numerics):
    return {k: v for k, v in numerics.items() if k not in PRECISION_FIELDS}


def _is_narrow(case):
    return case["numerics"].get("operand_precision") == "narrow"


def case_pairs(definitions):
    """`[(exact case, narrow case)]`: each `-narrow` case with the exact case that is the same work in every field
    but its operand precision. A narrow case with no exact twin is an error of the bank (`pair_errors`)."""
    exact = [c for c in definitions["cases"] if not _is_narrow(c)]
    pairs = []
    for narrow in (c for c in definitions["cases"] if _is_narrow(c)):
        same = [
            c
            for c in exact
            if _without_precision(c["numerics"]) == _without_precision(narrow["numerics"])
            and all(c.get(f) == narrow.get(f) for f in narrow if f not in DESCRIPTIVE_FIELDS)
            and set(c) == set(narrow)
        ]
        if len(same) == 1:
            pairs.append((same[0], narrow))
    return pairs


def pair_errors(definitions):
    """Why the bank's narrow cases are not all twins of one exact case: a narrow case with none (or several)."""
    paired = {narrow["id"] for _, narrow in case_pairs(definitions)}
    return [
        f"{c['id']}: a narrow case needs exactly one exact twin that differs only in operand_precision"
        for c in definitions["cases"]
        if _is_narrow(c) and c["id"] not in paired
    ]


def device_policy(policy=receipts.POLICY):
    """The comparator's policy with the device-clock metric judged by the tolerances of the step metric."""
    metrics = dict(policy["metrics"])
    metrics[bank.DEVICE_METRIC] = metrics[bank.KERNEL_METRIC]
    return {**policy, "metrics": metrics}


def _claim(manifest, case, backend):
    return next((e for e in manifest["entries"] if (e["case"], e["backend"]) == (case, backend)), None)


def _device_samples(root, entry):
    """`(receipt, samples)` of a verified claim's device cell; samples is `None` when the receipt has no device cell."""
    receipt = json.loads((Path(root) / entry["receipt"]).read_text())
    cell = receipt.get("cells", {}).get("device")
    if cell is None:
        return receipt, None
    values, failed, perturbed = receipts._repeat_samples(cell["repeats"], bank.DEVICE_METRIC)
    return receipt, (values, failed, perturbed)


def _same_session(exact, narrow):
    """Why two receipts are not one measurement of the planner's choice, or `None`: one binary on one device."""
    for field in ("binary_sha256", "parent_revision"):
        if exact.get(field) != narrow.get(field):
            return f"the twins were built differently ({field} differs)"
    for field in ("device", "driver"):
        if exact.get("identity", {}).get(field) != narrow.get("identity", {}).get(field):
            return f"the twins were measured on different devices ({field} differs)"
    return None


def pair_verdicts(definitions, manifest, root, policy=None):
    """`[{pair, backend, verdict, note, median_exact, median_narrow}]` for every (exact, narrow) pair and backend the
    narrow case claims. `verdict` is the comparator's (`regression`: the chosen schedule loses to exact beyond the
    band), or `unmeasured` with the reason when a side has no verified device cell from the same session."""
    policy = policy or device_policy()
    rows = []
    for exact_case, narrow_case in case_pairs(definitions):
        for backend in narrow_case["backends"]:
            row = {"pair": narrow_case["id"], "exact": exact_case["id"], "backend": backend}
            entries = [_claim(manifest, c["id"], backend) for c in (exact_case, narrow_case)]
            if any(e is None or e["status"] != "verified" for e in entries):
                rows.append({**row, "verdict": "unmeasured", "note": "a side has no verified receipt"})
                continue
            (exact_receipt, exact_samples), (narrow_receipt, narrow_samples) = (
                _device_samples(root, e) for e in entries
            )
            if exact_samples is None or narrow_samples is None:
                rows.append({**row, "verdict": "unmeasured", "note": "a side has no device-clock cell"})
                continue
            if reason := _same_session(exact_receipt, narrow_receipt):
                rows.append({**row, "verdict": "unmeasured", "note": reason})
                continue
            if exact_receipt.get("plan_id") == narrow_receipt.get("plan_id"):
                # The planner chose the exact schedule: the twins ran the same kernels, and any difference between
                # their times is the session's drift, not the planner's choice.
                rows.append({**row, "verdict": SAME_PLAN, "note": "the planner chose the exact schedule"})
                continue
            (a, failed_a, _), (b, failed_b, _) = exact_samples, narrow_samples
            result = receipts.metric_verdict(a, b, policy, bank.DEVICE_METRIC, failed_a=failed_a, failed_b=failed_b)
            rows.append(
                {
                    **row,
                    "verdict": result["verdict"],
                    "note": result["note"],
                    "median_exact": result["median_a"],
                    "median_narrow": result["median_b"],
                }
            )
    return rows


def invariant_errors_of(rows):
    """The violations among `pair_verdicts` rows: each pair whose chosen schedule measures slower than exact beyond
    the comparator's band."""
    return [
        f"{row['pair']} on {row['backend']}: the chosen schedule loses to exact "
        f"({row['median_narrow']:.1f} us against {row['median_exact']:.1f} us of {row['exact']})"
        for row in rows
        if row["verdict"] == receipts.REGRESSION
    ]


def invariant_errors(root=None):
    """Every violation of the invariant in the bank at `root` (default: the checkout's definitions and the results
    setting's receipts): a narrow case without an exact twin, and each pair whose chosen schedule loses."""
    definitions_root, root = bank._roots(root, results_root.committed_bank_root)
    definitions = bank.load_json(definitions_root / "cases.json")
    manifest = bank.load_json(root / "receipts.json")
    return pair_errors(definitions) + invariant_errors_of(pair_verdicts(definitions, manifest, root))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="judge every narrow case against its exact twin on the device")
    parser.add_argument("--root", help="a self-contained bank directory; default: the checkout's cases and the results setting's receipts")
    args = parser.parse_args(argv)
    definitions_root, root = bank._roots(args.root, results_root.committed_bank_root)
    definitions = bank.load_json(definitions_root / "cases.json")
    manifest = bank.load_json(root / "receipts.json")
    for row in pair_verdicts(definitions, manifest, root):
        medians = (
            f" {row['median_narrow']:.1f} vs {row['median_exact']:.1f} us"
            if row.get("median_narrow") is not None
            else ""
        )
        print(f"{row['verdict']:>10}  {row['pair']} on {row['backend']}{medians}{('  ' + row['note']) if row['note'] else ''}")
    errors = invariant_errors(args.root)
    for error in errors:
        print(f"error: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except results_root.ResultsRootMissing as error:
        raise SystemExit(f"planner_invariant.py: {error}")
