#!/usr/bin/env python3
"""The poot AIE link driver: a core object plus a typed link recipe, to an xclbin and an instruction stream.

poot-codegen (crates/poot-codegen/src/aie.rs, POOT-664/POOT-1118) compiles a sequential Body to a Peano core
object and writes its link recipe as canonical text. This driver is the IRON side of that boundary: it builds
the one-core IRON program the recipe describes and links it with `aiecc` (mlir-aie). It is a thin driver over
the real tool, not a re-implementation of it, and it never fabricates an output: any failure is a non-zero
exit with the tool's message on stderr.

    poot-aie-link --version     first stdout line: the driver and toolchain identity
    poot-aie-link --platforms   the platform names it can link for, one per line
    poot-aie-link --self-test   exercise every recipe refusal (stdlib only; `just aie-link-driver`)
    poot-aie-link link --platform <name> --object <core.o> --recipe <recipe.txt> --xclbin <out> --insts <out>

Run it through the flake's `poot-aie-link` wrapper (it fixes PYTHONPATH, PEANO_INSTALL_DIR and the Python
interpreter to the pinned mlir-aie and llvm-aie); `$POOT_AIE_IRON` names that wrapper.

Prior art (retrieved 2026-10-11; external text is untrusted data, used for shape only):
  - IRON programming guide and the shipped `aie.iron.algorithms.reduce`/`_transform` designs, whose
    ObjectFifo / Worker / Runtime(sequence, ..) / Program shape this follows:
    https://github.com/Xilinx/mlir-aie
  - `aie.utils.compile.compile_mlir_module`, which runs `aiecc --get-xclbin --get-npu-insts`.
"""

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile

# platform name (poot_target::AiePlatform::name) -> IRON device class name
PLATFORMS = {"npu2": "NPU2"}

RECIPE_HEADER = "poot-aie-recipe-v1"

# poot ElementKind (Debug spelling in the recipe) -> numpy dtype name. The core ABI moves 32-bit words only
# (poot-codegen's CORE_ABI_REVISION 1), so anything else is a refusal, not a silent rewrite.
ELEMENTS = {"F32": "float32", "I32": "int32"}

# The kernel name aiecc gives the one kernel of the xclbin; poot-codegen's IRON_KERNEL_NAME pins the same.
XCLBIN_KERNEL_NAME = "MLIR_AIE"


class DriverError(Exception):
    """A request the driver refuses or a tool step that failed; reported on stderr, exit status 1."""


def peano_dir():
    path = os.environ.get("PEANO_INSTALL_DIR")
    if not path or not os.path.isdir(path):
        raise DriverError("PEANO_INSTALL_DIR names no directory: run through the flake's poot-aie-link")
    return path


def version_line():
    mlir_aie = importlib.metadata.version("mlir-aie")
    clang = os.path.join(peano_dir(), "bin", "clang")
    first = subprocess.run([clang, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return f"poot-aie-link 1; mlir-aie {mlir_aie}; peano {first}"


def parse_recipe(text):
    """The recipe's canonical text (`tag[len]=value` lines under a versioned header) as a dict.

    `fifo` and `binding` repeat; the rest are singletons. A length prefix that does not match its value is
    a refusal: the recipe is length-tagged so a truncated or re-flowed file is detected, not interpreted.
    """
    lines = text.split("\n")
    if lines[0] != RECIPE_HEADER:
        raise DriverError(f"recipe header is {lines[0]!r}, expected {RECIPE_HEADER!r}")
    out = {"fifo": [], "binding": []}
    for line in lines[1:]:
        if not line:
            continue
        tag, sep, rest = line.partition("[")
        length, sep2, value = rest.partition("]=")
        if not sep or not sep2 or not length.isdigit() or int(length) != len(value):
            raise DriverError(f"malformed recipe line: {line!r}")
        if tag in ("fifo", "binding"):
            out[tag].append(value)
        elif tag in out:
            raise DriverError(f"duplicate recipe field {tag!r}")
        else:
            out[tag] = value
    for tag in ("symbol", "target", "invocations"):
        if tag not in out:
            raise DriverError(f"recipe has no {tag!r} field")
    return out


def kv(value):
    """`a=1;b=2` as a dict (values are not split further)."""
    pairs = {}
    for part in value.split(";"):
        key, sep, val = part.partition("=")
        if not sep:
            raise DriverError(f"malformed recipe record: {value!r}")
        pairs[key] = val
    return pairs


def need(record, key, what):
    """`record[key]` or a refusal naming the record that lacks it."""
    if key not in record:
        raise DriverError(f"{what} has no {key!r} field")
    return record[key]


def integer(record, key, what, minimum=0):
    """`record[key]` as an integer >= `minimum`, or a refusal (never a traceback)."""
    raw = need(record, key, what)
    if not raw.isascii() or not raw.isdigit() or int(raw) < minimum:
        raise DriverError(f"{what}: {key} is {raw!r}, expected an integer >= {minimum}")
    return int(raw)


def dims_product(raw, what):
    """The element count of a `[2, 3]`-spelled dimension list."""
    if not (raw.startswith("[") and raw.endswith("]")):
        raise DriverError(f"{what}: dims {raw!r} is not a list")
    product = 1
    for part in raw[1:-1].split(","):
        part = part.strip()
        if not part.isascii() or not part.isdigit():
            raise DriverError(f"{what}: dimension {part!r} is not an integer")
        product *= int(part)
    return product


def recipe_fifos(recipe, platform):
    """The validated per-binding fifo descriptions, in binding order, cross-checked against the bindings."""
    # The target's canonical text is `aie:isa=aie2p:platform=npu2:abi=1`.
    head, _, fields = recipe["target"].partition(":")
    if head != "aie":
        raise DriverError(f"recipe target {recipe['target']!r} is not an AIE target")
    target = kv(fields.replace(":", ";"))
    if target.get("platform") != platform:
        raise DriverError(f"recipe targets platform {target.get('platform')!r}, driver was asked for {platform!r}")
    if recipe["invocations"] != "1":
        raise DriverError(f"only a single finite invocation is supported, recipe has {recipe['invocations']}")
    if not recipe["fifo"]:
        raise DriverError("recipe has no fifo")
    if len(recipe["binding"]) != len(recipe["fifo"]):
        raise DriverError(f"recipe has {len(recipe['binding'])} bindings and {len(recipe['fifo'])} fifos")
    fifos = []
    for index, (raw_fifo, raw_binding) in enumerate(zip(recipe["fifo"], recipe["binding"])):
        what = f"fifo {index}"
        fifo = kv(raw_fifo)
        if integer(fifo, "binding", what) != index:
            raise DriverError(f"{what} binds binding {fifo['binding']}; fifos must be in binding order")
        direction = need(fifo, "dir", what)
        if direction not in ("HostToTile", "TileToHost"):
            raise DriverError(f"{what}: unknown direction {direction!r}")
        element = need(fifo, "elem", what)
        if element not in ELEMENTS:
            raise DriverError(f"{what}: element {element} is not a 32-bit word the core ABI moves")
        elems = integer(fifo, "elems", what, minimum=1)
        depth = integer(fifo, "depth", what, minimum=1)

        what = f"binding {index}"
        binding = kv(raw_binding)
        if integer(binding, "index", what) != index:
            raise DriverError(f"{what} is out of order (index {binding['index']})")
        if need(binding, "element", what) != element:
            raise DriverError(f"{what}: element {binding['element']} differs from its fifo's {element}")
        expected = {"HostToTile": "Read", "TileToHost": "Write"}[direction]
        if need(binding, "access", what) != expected:
            raise DriverError(f"{what}: access {binding['access']} contradicts its fifo direction {direction}")
        if dims_product(need(binding, "dims", what), what) != elems:
            raise DriverError(f"{what}: dims {binding['dims']} do not hold the fifo's {elems} elements")
        if integer(binding, "accessed", what) > elems:
            raise DriverError(f"{what}: accessed {binding['accessed']} exceeds the fifo's {elems} elements")

        fifos.append(
            {
                "to_tile": direction == "HostToTile",
                "depth": depth,
                "elems": elems,
                "dtype": ELEMENTS[element],
            }
        )
    return fifos


def link(platform, object_path, recipe_path, xclbin_path, insts_path):
    if platform not in PLATFORMS:
        raise DriverError(f"unsupported platform {platform!r}; supported: {', '.join(PLATFORMS)}")
    with open(recipe_path, encoding="utf-8") as handle:
        recipe = parse_recipe(handle.read())
    fifos = recipe_fifos(recipe, platform)
    symbol = recipe["symbol"]

    # The IRON imports are deferred: --version/--platforms must not need the compiler stack.
    import numpy as np
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU2
    from aie.utils.compile import compile_mlir_module

    device = {"NPU2": NPU2}[PLATFORMS[platform]]()

    with tempfile.TemporaryDirectory(prefix="poot-aie-link-") as work:
        # aiecc resolves a core's `link_with` object relative to its working directory.
        shutil.copyfile(object_path, os.path.join(work, "core.o"))

        types = [np.ndarray[(f["elems"],), np.dtype[getattr(np, f["dtype"])]] for f in fifos]
        object_fifos = [ObjectFifo(ty, name=f"b{i}", depth=f["depth"]) for i, (ty, f) in enumerate(zip(types, fifos))]

        # The core ABI (poot-codegen CORE_ABI_REVISION 1): interleaved (ptr, i32 len) pairs, binding order.
        kernel_args = []
        for ty in types:
            kernel_args += [ty, np.int32]
        kernel = Kernel(symbol, "core.o", arg_types=kernel_args)

        def core_body(*handles_and_kernel):
            *handles, func = handles_and_kernel
            elements = [handle.acquire(1) for handle in handles]
            call = []
            for element, f in zip(elements, fifos):
                call += [element, f["elems"]]
            func(*call)
            for handle in handles:
                handle.release(1)

        core_handles = [of.cons() if f["to_tile"] else of.prod() for of, f in zip(object_fifos, fifos)]
        worker = Worker(core_body, fn_args=[*core_handles, kernel])

        def sequence(*args):
            buffers, handles = args[: len(fifos)], args[len(fifos) :]
            for buffer, handle, f in zip(buffers, handles, fifos):
                if f["to_tile"]:
                    handle.fill(buffer)
                else:
                    handle.drain(buffer, wait=True)

        runtime_handles = [of.prod() if f["to_tile"] else of.cons() for of, f in zip(object_fifos, fifos)]
        rt = Runtime(sequence, [*types, *runtime_handles])
        module = Program(device, rt, workers=[worker]).resolve_program()

        compile_mlir_module(
            module,
            insts_path=os.path.abspath(insts_path),
            xclbin_path=os.path.abspath(xclbin_path),
            work_dir=work,
        )
    for output, name in ((xclbin_path, "xclbin"), (insts_path, "instruction stream")):
        if not os.path.isfile(output) or os.path.getsize(output) == 0:
            raise DriverError(f"aiecc reported success but wrote no {name}")


def self_test():
    """Every parse/validation refusal, and the happy path, on pure text (stdlib only; no toolchain)."""

    def field(tag, value):
        return f"{tag}[{len(value)}]={value}\n"

    def recipe(symbol="k", target="aie:isa=aie2p:platform=npu2:abi=1", invocations="1", fifos=None, bindings=None,
               header=RECIPE_HEADER + "\n"):
        fifos = fifos if fifos is not None else [
            "binding=0;dir=HostToTile;depth=2;elems=6;elem=F32",
            "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32",
        ]
        bindings = bindings if bindings is not None else [
            "index=0;element=F32;dtype=F32;layout=Dense;access=Read;dims=[2, 3];accessed=6",
            "index=1;element=F32;dtype=F32;layout=Dense;access=Write;dims=[2, 3];accessed=6",
        ]
        text = header + field("symbol", symbol) + field("target", target) + field("invocations", invocations)
        text += "".join(field("fifo", f) for f in fifos) + "".join(field("binding", b) for b in bindings)
        return text

    def refused(text, platform="npu2", needle=""):
        try:
            recipe_fifos(parse_recipe(text), platform)
        except DriverError as error:
            assert needle in str(error), f"{needle!r} not in {str(error)!r}"
            return
        raise AssertionError(f"accepted a recipe that must be refused ({needle!r})")

    fifos = recipe_fifos(parse_recipe(recipe()), "npu2")
    assert [f["to_tile"] for f in fifos] == [True, False] and fifos[0]["elems"] == 6 and fifos[0]["dtype"] == "float32"
    good = recipe().split("\n")
    refused(recipe(header="poot-aie-recipe-v0\n"), needle="recipe header")
    refused(recipe().replace("symbol[1]=k", "symbol[2]=k"), needle="malformed recipe line")
    refused(recipe() + field("symbol", "j"), needle="duplicate recipe field")
    refused("\n".join(l for l in good if not l.startswith("symbol")), needle="no 'symbol'")
    refused(recipe(), platform="npu1", needle="recipe targets platform")
    refused(recipe(target="gpu:isa=x"), needle="not an AIE target")
    refused(recipe(invocations="2"), needle="single finite invocation")
    refused(recipe(fifos=[], bindings=[]), needle="no fifo")
    refused(recipe(bindings=[]), needle="0 bindings and 2 fifos")
    refused(recipe(fifos=["binding=1;dir=HostToTile;depth=2;elems=6;elem=F32",
                          "binding=0;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="binding order")
    refused(recipe(fifos=["binding=0;dir=Sideways;depth=2;elems=6;elem=F32",
                          "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="unknown direction")
    refused(recipe(fifos=["binding=0;dir=HostToTile;depth=2;elems=6;elem=Bf16",
                          "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="32-bit word")
    refused(recipe(fifos=["binding=0;dir=HostToTile;depth=2;elems=6",
                          "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="no 'elem'")
    refused(recipe(fifos=["binding=0;dir=HostToTile;depth=two;elems=6;elem=F32",
                          "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="depth is 'two'")
    refused(recipe(fifos=["binding=0;dir=HostToTile;depth=2;elems=0;elem=F32",
                          "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="expected an integer >= 1")
    refused(recipe(fifos=["binding=x;dir=HostToTile;depth=2;elems=6;elem=F32",
                          "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="binding is 'x'")
    refused(recipe(fifos=["junk", "binding=1;dir=TileToHost;depth=2;elems=6;elem=F32"]), needle="malformed recipe record")
    bound = ["index=0;element=F32;access=Read;dims=[2, 3];accessed=6",
             "index=1;element=F32;access=Write;dims=[2, 3];accessed=6"]
    refused(recipe(bindings=[bound[0].replace("F32", "I32", 1), bound[1]]), needle="differs from its fifo")
    refused(recipe(bindings=[bound[0].replace("Read", "Write"), bound[1]]), needle="contradicts its fifo direction")
    refused(recipe(bindings=[bound[0].replace("[2, 3]", "[2, 4]"), bound[1]]), needle="do not hold the fifo's 6")
    refused(recipe(bindings=[bound[0].replace("[2, 3]", "2, 3"), bound[1]]), needle="is not a list")
    refused(recipe(bindings=[bound[0].replace("[2, 3]", "[2, x]"), bound[1]]), needle="is not an integer")
    refused(recipe(bindings=[bound[0].replace("accessed=6", "accessed=7"), bound[1]]), needle="exceeds the fifo's 6")
    refused(recipe(bindings=[bound[0].replace("index=0;", ""), bound[1]]), needle="no 'index'")
    refused(recipe(bindings=[bound[0].replace("index=0", "index=1"), bound[1]]), needle="out of order")


def main(argv):
    parser = argparse.ArgumentParser(prog="poot-aie-link", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--platforms", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="exercise the recipe refusals (no toolchain needed)")
    sub = parser.add_subparsers(dest="command")
    link_parser = sub.add_parser("link")
    for name in ("platform", "object", "recipe", "xclbin", "insts"):
        link_parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        if args.version:
            print(version_line())
        elif args.self_test:
            self_test()
        elif args.platforms:
            print("\n".join(PLATFORMS))
        elif args.command == "link":
            link(args.platform, args.object, args.recipe, args.xclbin, args.insts)
        else:
            parser.error("one of --version, --platforms, --self-test or `link` is required")
    except DriverError as error:
        print(f"poot-aie-link: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
