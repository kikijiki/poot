"""The checked-in performance bank: case definitions, their identities and the receipts claimed for them.

`performance-bank/cases.json` defines every case (shape, storage, numerics, token fixture, cache condition,
schedules) once, for both the Python validator and the Rust backend runners. `performance-bank/receipts.json`
lists, per case and backend, what is claimed: a verified baseline receipt, or the case as missing or refused
with its reason. `validate_bank` rebuilds every identity from the case definitions and refuses a claim the
receipt files do not back; it never decides a speed verdict (that is `performance_receipt`).

Hardware receipts live under `performance-bank/receipts/<backend>/`, apart from the model-free fixtures in
`harness/tests`: a receipt is a measurement, a fixture is a deterministic synthetic sample.
"""

import hashlib
import json
import os
import shlex
import statistics
import subprocess
from pathlib import Path

try:
    from . import performance_receipt as receipts
except ImportError:  # direct execution from harness/
    import performance_receipt as receipts

BANK_ROOT = Path(__file__).resolve().parents[1] / "performance-bank"
REPO_ROOT = Path(__file__).resolve().parents[2]

BACKENDS = ("wgpu", "rocm", "ptx")
BACKEND_PACKAGES = {"wgpu": "poot-gpu", "rocm": "poot-rocm-gpu", "ptx": "poot-ptx-gpu"}
RUNNABLE = "runnable"
PLANNED = "planned"
CASE_STATUSES = (RUNNABLE, PLANNED)
CLAIM_STATUSES = ("verified", "missing", "refused")

# What an independent reference is: computed without the engine under test. Internal parity (the engine
# against itself on another backend or path) is a different claim and does not back a speed receipt.
INDEPENDENT_REFERENCE = "independent-golden"
KERNEL_METRIC = "step_us"
BASELINE_ROLE = "baseline"

_SHA256_HEX = set("0123456789abcdef")


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def fixture_id(bank, case):
    """What the work is: token fixture (recipe included), shape, storage, layout and numerics."""
    return _sha(
        receipts.canonical(
            {
                "token_fixture": bank["token_fixtures"][case["token_fixture"]],
                "producer": case["producer"],
                "shape": case["shape"],
                "storage": case["storage"],
                "layout": case["layout"],
                "numerics": case["numerics"],
            }
        )
    )


def schedule_id(case):
    """The parent/default and candidate schedules the case names."""
    return _sha(receipts.canonical(case["schedules"]))


def cache_id(bank, case):
    """The cache-conditioning recipe itself, not its name, so editing the recipe changes the identity."""
    return _sha(receipts.canonical(bank["cache_conditions"][case["cache"]]))


def case_identity(bank, case, hardware=None):
    """The workload identity a receipt of this case states. `hardware` fills the backend/device/driver group."""
    hardware = hardware or {}
    rows = case["shape"].get("m", 1)
    fixture = bank["token_fixtures"][case["token_fixture"]]
    return {
        "checkpoint_digest": "synthetic-no-checkpoint",
        "token_fixture": f"{case['token_fixture']}:{_sha(receipts.canonical(fixture))[:16]}",
        "precision": case["storage"],
        "layout": case["layout"],
        "dimensions": case["shape"],
        "prompt_tokens": rows,
        "live_context": case["shape"].get("cap", "n/a"),
        "max_context": case["shape"].get("cap", "n/a"),
        "requested_output": "n/a",
        "actual_output": "n/a",
        "sampler": "n/a",
        "template": "n/a",
        "cache_state": {"condition": case["cache"], "id": cache_id(bank, case)},
        "prepared_set": [case["id"]],
        "timing_policy": bank["timing_policy"],
        "backend": hardware.get("backend"),
        "device": hardware.get("device"),
        "driver": hardware.get("driver"),
    }


def case_identity_digest(bank, case):
    """The backend-independent identity: the hardware group is excluded so every backend shares one digest."""
    identity = case_identity(bank, case)
    hardware = receipts.IDENTITY_GROUPS["hardware"]
    return _sha(receipts.canonical({k: v for k, v in identity.items() if k not in hardware}))


def stamps(bank, case):
    """The identities a case carries, recomputed from its definition."""
    return {
        "fixture_id": fixture_id(bank, case),
        "schedule_id": schedule_id(case),
        "cache_id": cache_id(bank, case),
        "identity_digest": case_identity_digest(bank, case),
    }


def stamp_bank(root=BANK_ROOT):
    """Rewrite `cases.json` with the identities its definitions give (run after editing a case on purpose)."""
    path = Path(root) / "cases.json"
    bank = load_json(path)
    for case in bank["cases"]:
        case.update(stamps(bank, case))
    path.write_text(json.dumps(bank, indent=2) + "\n")


def bank_command(bank, case, backend):
    """The one command that produces this case's baseline receipt on `backend`."""
    return [
        "cargo",
        "run",
        "--release",
        "-p",
        BACKEND_PACKAGES[backend],
        "--example",
        "performance_bank",
        "--",
        "--bank",
        "benchmarks/performance-bank/cases.json",
        "--case",
        case["id"],
        "--out",
        f"benchmarks/performance-bank/receipts/{backend}/{case['id']}.json",
    ]


def bank_errors(bank):
    """Why the case definitions are not a sound bank: stale identities, dangling references, bad fields."""
    errors = []
    seen = set()
    for case in bank["cases"]:
        name = case.get("id", "<unnamed>")
        if name in seen:
            errors.append(f"{name}: duplicate case id")
        seen.add(name)
        if case.get("status") not in CASE_STATUSES:
            errors.append(f"{name}: status must be one of {CASE_STATUSES}")
        resolved = True
        for ref, table in (("token_fixture", "token_fixtures"), ("cache", "cache_conditions")):
            if case.get(ref) not in bank[table]:
                resolved = False
                errors.append(f"{name}: {ref} {case.get(ref)!r} is not defined in {table}")
        if bad := [b for b in case.get("backends", []) if b not in BACKENDS]:
            errors.append(f"{name}: unknown backends {bad}")
        if case.get("status") == PLANNED and not case.get("planned_reason"):
            errors.append(f"{name}: a planned case states why it is not runnable")
        if not resolved:
            continue
        for label, expected in stamps(bank, case).items():
            if case.get(label) != expected:
                errors.append(f"{name}: {label} is {case.get(label)!r} but the definition gives {expected!r}")
    return errors


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _is_git_sha(value):
    return isinstance(value, str) and len(value) == 40 and set(value) <= _SHA256_HEX


def observed_errors(bank, case, observed):
    """Why the workload a runner says it executed is not the case's: shape, storage, layout, cache rotation."""
    errors = []
    for field in ("shape", "storage", "layout"):
        if observed.get(field) != case[field]:
            errors.append(f"observed {field} {observed.get(field)!r} is not the case's {case[field]!r}")
    condition = bank["cache_conditions"][case["cache"]]
    copies, per_copy = observed.get("weight_copies"), observed.get("weight_bytes_per_copy")
    if not (isinstance(copies, int) and isinstance(per_copy, int) and copies >= 1 and per_copy > 0):
        errors.append("observed weight_copies and weight_bytes_per_copy must be positive integers")
    elif condition["kind"] == "weight-rotation" and copies * per_copy < condition["min_rotation_bytes"]:
        errors.append(
            f"{copies} weight copies of {per_copy} bytes rotate {copies * per_copy} bytes, "
            f"less than the {condition['min_rotation_bytes']} the cold-weights condition needs"
        )
    elif condition["kind"] == "single-copy" and copies != 1:
        errors.append(f"a single-copy condition measured {copies} weight copies")
    return errors


def receipt_errors(bank, case, backend, receipt):
    """Why `receipt` is not a verified same-parent baseline for (case, backend); empty when it is."""
    errors = []
    if receipt.get("case") != case["id"] or receipt.get("backend") != backend:
        errors.append(f"receipt is for {receipt.get('case')!r} on {receipt.get('backend')!r}")
    if receipt.get("role") != BASELINE_ROLE:
        errors.append(f"role must be {BASELINE_ROLE!r}; got {receipt.get('role')!r}")
    if not _is_git_sha(receipt.get("parent_revision")):
        errors.append("parent_revision must be the 40-digit git sha the baseline was built from")
    if not _is_sha256(receipt.get("binary_sha256")):
        errors.append("binary_sha256 must be the 64-digit digest of the measuring binary")
    if receipt.get("command") != bank_command(bank, case, backend):
        errors.append("command is not the checked-in bank command for this case and backend")

    errors.extend(observed_errors(bank, case, receipt.get("observed", {})))
    identity = receipt.get("identity", {})
    hardware = {k: identity.get(k) for k in ("backend", "device", "driver")}
    expected = case_identity(bank, case, hardware)
    if mismatches := receipts.identity_mismatches(expected, identity):
        fields = ", ".join(field for field, _ in mismatches)
        errors.append(f"identity differs from the case definition in: {fields}")
    if receipt.get("identity_digest") != case["identity_digest"]:
        errors.append("identity_digest is not the case's checked-in identity digest")
    if identity.get("backend") != backend:
        errors.append(f"identity names backend {identity.get('backend')!r}, not {backend!r}")
    if receipt.get("schedule_id") != case["schedule_id"]:
        errors.append("schedule_id is not the case's checked-in schedule identity")
    if not receipt.get("plan_id"):
        errors.append("plan_id (the realized parent schedule) is required")

    reference = receipt.get("reference", {})
    if reference.get("provenance") != INDEPENDENT_REFERENCE:
        errors.append(
            f"reference provenance is {reference.get('provenance')!r}: only an {INDEPENDENT_REFERENCE!r} "
            "backs a speed receipt, internal parity does not"
        )
    if reference.get("agrees") is not True:
        errors.append("the measured output does not agree with its reference")

    repeats = receipt.get("cells", {}).get("kernel", {}).get("repeats", [])
    min_trials = receipts.POLICY["min_trials"]
    if len(repeats) < min_trials:
        errors.append(f"{len(repeats)} repeats recorded; the policy needs {min_trials}")
    for index, repeat in enumerate(repeats):
        value = repeat.get("metrics", {}).get(KERNEL_METRIC)
        if repeat.get("status") != "ok" or not (receipts.is_finite(value) and value > 0):
            errors.append(f"repeat {index} is not a finite successful {KERNEL_METRIC} sample: {repeat!r}")
    return errors


def manifest_errors(bank, manifest, root=BANK_ROOT):
    """Why the claims in `receipts.json` are not backed by `root`'s receipt files; empty when they are."""
    errors = []
    claims = {}
    for entry in manifest["entries"]:
        key = (entry.get("case"), entry.get("backend"))
        if key in claims:
            errors.append(f"{key}: claimed twice")
        claims[key] = entry

    cases = {case["id"]: case for case in bank["cases"]}
    for key in claims:
        if key[0] not in cases:
            errors.append(f"{key}: names a case the bank does not define")
        elif key[1] not in cases[key[0]].get("backends", []):
            errors.append(f"{key}: the case does not claim this backend")

    for case in bank["cases"]:
        for backend in case.get("backends", []):
            key = (case["id"], backend)
            entry = claims.get(key)
            if entry is None:
                errors.append(f"{key}: required but neither verified nor declared missing or refused")
                continue
            status = entry.get("status")
            if status not in CLAIM_STATUSES:
                errors.append(f"{key}: status must be one of {CLAIM_STATUSES}; got {status!r}")
            elif status != "verified":
                if not entry.get("reason"):
                    errors.append(f"{key}: {status} needs the reason it is not covered")
                if entry.get("receipt"):
                    errors.append(f"{key}: {status} cannot name a receipt")
            elif case["status"] != RUNNABLE:
                errors.append(f"{key}: a {case['status']} case cannot be verified")
            else:
                errors.extend(f"{key}: {e}" for e in _verified_errors(bank, case, backend, entry, root))
    return errors


def spread_of(receipt, policy=receipts.POLICY):
    """`{"spread", "exceeds_max_uncertainty"}` of a receipt's repeats: (max - min) / median against the metric's
    tolerated uncertainty. A baseline noisier than the tolerance makes any later comparison of the cell
    unverified until it is tightened."""
    samples = [r["metrics"][KERNEL_METRIC] for r in receipt["cells"]["kernel"]["repeats"]]
    median = statistics.median(samples)
    spread = round((max(samples) - min(samples)) / median, 4)
    return {"spread": spread, "exceeds_max_uncertainty": 2 * spread > policy["metrics"][KERNEL_METRIC]["max_uncertainty"]}


def _verified_errors(bank, case, backend, entry, root):
    path = Path(root) / entry.get("receipt", "")
    if not entry.get("receipt") or not path.is_file():
        return [f"verified but its receipt {entry.get('receipt')!r} is not a file under the bank"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("receipt_sha256"):
        return ["receipt_sha256 does not match the receipt file"]
    if entry.get("command") != bank_command(bank, case, backend):
        return ["the claimed command is not the checked-in bank command for this case and backend"]
    receipt = json.loads(data)
    errors = receipt_errors(bank, case, backend, receipt)
    for field, value in spread_of(receipt).items():
        if entry.get(field) != value:
            errors.append(f"{field} is {entry.get(field)!r} but the receipt's repeats give {value!r}")
    return errors


def _git(*args, repo=REPO_ROOT):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_identity():
    """`(device, driver)` of the machine, from the harness's own environment capture."""
    try:
        from . import bench
    except ImportError:
        import bench
    env = bench.capture_env(BANK_ROOT)
    return env["gpu_name"], env["gpu_driver"] or f"host kernel {env['host_kernel']}"


def record(case_id, backend, *, wrap=(), root=BANK_ROOT, repo=REPO_ROOT, device=None, run=subprocess.run):
    """Measure `case_id` on `backend` with the checked-in command and register the receipt as verified.

    The tree must be clean: the receipt names the commit the measuring binary was built from. The runner
    reports what it executed; this completes the receipt from the case definition and refuses it when the
    two disagree or any validation fails, so a receipt that is written is one `validate_bank` accepts.
    """
    root = Path(root)
    bank = load_json(root / "cases.json")
    case = next((c for c in bank["cases"] if c["id"] == case_id), None)
    if case is None or backend not in case.get("backends", []):
        raise SystemExit(f"{case_id} on {backend} is not a bank case")
    # Receipts recorded earlier in the session are not source: they cannot change the binary's behavior.
    bank_dir = Path(root).resolve().relative_to(Path(repo).resolve())
    if _git("status", "--porcelain", "--", ".", f":(exclude){bank_dir}/receipts", f":(exclude){bank_dir}/receipts.json", repo=repo):
        raise SystemExit("the tree has uncommitted source changes: a receipt names the commit it was built from")
    name, driver = device or device_identity()
    if not (name and driver):
        raise SystemExit("this host reports no device or driver to name in the receipt")

    command = bank_command(bank, case, backend)
    env = {**os.environ, "POOT_BANK_DEVICE": name, "POOT_BANK_DRIVER": driver}
    run([*wrap, *command], cwd=repo, env=env, check=True)

    path = Path(repo) / command[command.index("--out") + 1]
    receipt = json.loads(path.read_text())
    hardware = receipt.pop("hardware")
    binary = Path(receipt.pop("binary"))
    identity = case_identity(bank, case, hardware)
    receipt.update(
        command=command,
        parent_revision=_git("rev-parse", "HEAD", repo=repo),
        binary_sha256=_sha256_file(binary),
        identity=identity,
        identity_digest=case["identity_digest"],
        schedule_id=case["schedule_id"],
    )
    if errors := receipt_errors(bank, case, backend, receipt):
        raise SystemExit("the measured receipt is not a verified baseline:\n  " + "\n  ".join(errors))
    path.write_text(json.dumps(receipt, indent=2) + "\n")

    manifest_path = root / "receipts.json"
    manifest = load_json(manifest_path)
    previous = next((e for e in manifest["entries"] if (e["case"], e["backend"]) == (case_id, backend)), {})
    manifest["entries"] = [e for e in manifest["entries"] if (e["case"], e["backend"]) != (case_id, backend)]
    manifest["entries"].append(
        {
            "case": case_id,
            "backend": backend,
            "status": "verified",
            "receipt": str(path.relative_to(root)),
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "command": command,
            **spread_of(receipt),
            **({"note": previous["note"]} if "note" in previous else {}),
        }
    )
    manifest["entries"].sort(key=lambda e: (e["case"], e["backend"]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def validate_bank(root=BANK_ROOT):
    """Every error in the checked-in bank and its receipt claims, case definitions first."""
    root = Path(root)
    bank = load_json(root / "cases.json")
    errors = bank_errors(bank)
    if errors:
        return errors
    return manifest_errors(bank, load_json(root / "receipts.json"), root)


def coverage(root=BANK_ROOT):
    """`{status: [(case, backend)]}` over the claims, so a report separates verified from missing/refused."""
    manifest = load_json(Path(root) / "receipts.json")
    found = {status: [] for status in CLAIM_STATUSES}
    for entry in manifest["entries"]:
        found.setdefault(entry["status"], []).append((entry["case"], entry["backend"]))
    return found


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="validate the performance bank and its receipt claims")
    parser.add_argument("--root", default=str(BANK_ROOT))
    parser.add_argument("--stamp", action="store_true", help="rewrite the case identities after an intended edit")
    parser.add_argument("--record", nargs=2, metavar=("CASE", "BACKEND"), help="measure a case and register its receipt")
    parser.add_argument("--wrap", default="", help="command prefix for --record, one string, e.g. 'poot-cargo nix develop -c'")
    args = parser.parse_args(argv)
    if args.stamp:
        stamp_bank(args.root)
    if args.record:
        record(*args.record, wrap=shlex.split(args.wrap), root=args.root)
    errors = validate_bank(args.root)
    for error in errors:
        print(f"error: {error}")
    for status, pairs in coverage(args.root).items():
        print(f"{status}: {len(pairs)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
