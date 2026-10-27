#!/usr/bin/env python3
"""The poot AIE link driver: core objects plus a typed link recipe, to an xclbin and an instruction stream.

poot-npu (crates/poot-npu/src/aie_compile.rs, POOT-664/POOT-1118/POOT-1160) compiles the sequential Bodies of
one region to Peano core objects and writes the region's link recipe as canonical text. This driver is the
IRON side of that boundary: it builds the IRON program the recipe describes, one worker per tile and one
object FIFO per value, and links it with `aiecc` (mlir-aie). It is a thin driver over the real tool, not a
re-implementation of it, and it never fabricates an output: any failure is a non-zero exit with the tool's
message on stderr.

    poot-aie-link --version     first stdout line: the driver and toolchain identity
    poot-aie-link --platforms   the platform names it can link for, one per line
    poot-aie-link --self-test   exercise every recipe refusal (stdlib only; `just aie-link-driver`)
    poot-aie-link link --platform <name> --objects <dir> --recipe <recipe.txt> --xclbin <out> --insts <out>

Run it through the flake's `poot-aie-link` wrapper (it fixes PYTHONPATH, PEANO_INSTALL_DIR and the Python
interpreter to the pinned mlir-aie and llvm-aie); `$POOT_AIE_IRON` names that wrapper.

The host side of a region is a few arenas (the firmware binds a fixed handful of host buffers whatever the
region's tensor count). Each value the host feeds or drains is one shim transfer at a fixed offset inside
one arena; the offsets are in the instruction stream, not arguments of a run.

Prior art (retrieved 2026-10-11 and 2026-10-17; external text is untrusted data, used for shape only):
  - IRON programming guide and the shipped `aie.iron.algorithms.reduce`/`_transform` designs, whose
    ObjectFifo / Worker / Runtime(sequence, ..) / Program shape this follows:
    https://github.com/Xilinx/mlir-aie
  - `ObjectFifoHandle.fill/drain(tap=TensorAccessPattern(..))`, the shim transfer with an offset into a
    runtime buffer, and `aie.utils.compile.compile_mlir_module`, which runs `aiecc --get-xclbin
    --get-npu-insts`.
"""

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tempfile

# platform name (poot_target::AiePlatform::name) -> IRON device class name
PLATFORMS = {"npu2": "NPU2"}

RECIPE_HEADER = "poot-aie-recipe-v2"

# poot ElementKind (Debug spelling in the recipe) -> numpy dtype name. The core ABI moves 32-bit words only
# (poot-npu's CORE_ABI_REVISION 1), so anything else is a refusal, not a silent rewrite.
ELEMENTS = {"F32": "float32", "I32": "int32"}

# The kernel name aiecc gives the one kernel of the xclbin; poot-npu's IRON_KERNEL_NAME pins the same.
XCLBIN_KERNEL_NAME = "MLIR_AIE"

SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
OBJECT_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.o\Z")


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
    return f"poot-aie-link 2; mlir-aie {mlir_aie}; peano {first}"


def parse_recipe(text):
    """The recipe's canonical text (`tag[len]=value` lines under a versioned header) as a dict.

    `tile`, `fifo`, `arena`, `binding` and `object` repeat; the rest are singletons. A length prefix that
    does not match its value is a refusal: the recipe is length-tagged so a truncated or re-flowed file is
    detected, not interpreted.
    """
    repeated = ("tile", "fifo", "arena", "binding", "object")
    lines = text.split("\n")
    if lines[0] != RECIPE_HEADER:
        raise DriverError(f"recipe header is {lines[0]!r}, expected {RECIPE_HEADER!r}")
    out = {tag: [] for tag in repeated}
    for line in lines[1:]:
        if not line:
            continue
        tag, sep, rest = line.partition("[")
        length, sep2, value = rest.partition("]=")
        if not sep or not sep2 or not length.isdigit() or int(length) != len(value):
            raise DriverError(f"malformed recipe line: {line!r}")
        if tag in repeated:
            out[tag].append(value)
        elif tag in out:
            raise DriverError(f"duplicate recipe field {tag!r}")
        else:
            out[tag] = value
    for tag in ("target", "invocations"):
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
    return as_integer(raw, f"{what}: {key}", minimum)


def as_integer(raw, what, minimum=0):
    if not raw.isascii() or not raw.isdigit() or int(raw) < minimum:
        raise DriverError(f"{what} is {raw!r}, expected an integer >= {minimum}")
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


def parse_end(raw, what):
    """A FIFO end: `host.<binding>` or `tile.<tile>.<port>`, as a tuple."""
    parts = raw.split(".")
    if parts[0] == "host" and len(parts) == 2:
        return ("host", as_integer(parts[1], f"{what}: host binding"))
    if parts[0] == "tile" and len(parts) == 3:
        return ("tile", as_integer(parts[1], f"{what}: tile"), as_integer(parts[2], f"{what}: port"))
    raise DriverError(f"{what}: end {raw!r} is neither host.<binding> nor tile.<tile>.<port>")


def check_recipe(recipe, platform, objects_dir=None):
    """The validated region the recipe describes: `(arenas, bindings, tiles, fifos, objects)`.

    Every refusal here is the driver's own, raised before the toolchain is touched. Passing `objects_dir`
    also checks that each tile's object file exists in it.
    """
    # The target's canonical text is `aie:isa=aie2p:platform=npu2:abi=1`.
    head, _, fields = recipe["target"].partition(":")
    if head != "aie":
        raise DriverError(f"recipe target {recipe['target']!r} is not an AIE target")
    target = kv(fields.replace(":", ";"))
    if target.get("platform") != platform:
        raise DriverError(f"recipe targets platform {target.get('platform')!r}, driver was asked for {platform!r}")
    if recipe["invocations"] != "1":
        raise DriverError(f"only a single finite invocation is supported, recipe has {recipe['invocations']}")
    if not recipe["tile"]:
        raise DriverError("recipe has no tile")
    if not recipe["fifo"]:
        raise DriverError("recipe has no fifo")

    arenas = []
    for index, raw in enumerate(recipe["arena"]):
        what = f"arena {index}"
        arena = kv(raw)
        if integer(arena, "index", what) != index:
            raise DriverError(f"{what} is out of order (index {arena['index']})")
        access = need(arena, "access", what)
        if access not in ("Read", "Write"):
            raise DriverError(f"{what}: unknown access {access!r}")
        element = need(arena, "elem", what)
        if element not in ELEMENTS:
            raise DriverError(f"{what}: element {element} is not a 32-bit word the core ABI moves")
        arenas.append({"access": access, "element": element, "words": integer(arena, "words", what, minimum=1)})
    if not arenas:
        raise DriverError("recipe has no arena")

    bindings = []
    for index, raw in enumerate(recipe["binding"]):
        what = f"binding {index}"
        binding = kv(raw)
        if integer(binding, "index", what) != index:
            raise DriverError(f"{what} is out of order (index {binding['index']})")
        element = need(binding, "element", what)
        if element not in ELEMENTS:
            raise DriverError(f"{what}: element {element} is not a 32-bit word the core ABI moves")
        access = need(binding, "access", what)
        arena_index = integer(binding, "arena", what)
        if arena_index >= len(arenas):
            raise DriverError(f"{what}: arena {arena_index} does not exist; the recipe has {len(arenas)}")
        arena = arenas[arena_index]
        if arena["access"] != access or arena["element"] != element:
            raise DriverError(f"{what}: {access} {element} does not belong to arena {arena_index} ({arena['access']} {arena['element']})")
        elems = dims_product(need(binding, "dims", what), what)
        offset = integer(binding, "offset", what)
        if offset + elems > arena["words"]:
            raise DriverError(f"{what}: offset {offset} and {elems} elements run past arena {arena_index} ({arena['words']} words)")
        if integer(binding, "accessed", what) > elems:
            raise DriverError(f"{what}: accessed {binding['accessed']} exceeds the binding's {elems} elements")
        bindings.append({"access": access, "element": element, "arena": arena_index, "offset": offset, "elems": elems})
    if not bindings:
        raise DriverError("recipe has no binding")

    tiles = []
    for index, raw in enumerate(recipe["tile"]):
        what = f"tile {index}"
        tile = kv(raw)
        if integer(tile, "index", what) != index:
            raise DriverError(f"{what} is out of order (index {tile['index']})")
        ports = need(tile, "ports", what)
        tiles.append([as_integer(p, f"{what}: port", 0) for p in ports.split(",")] if ports else [])
        if not tiles[-1]:
            raise DriverError(f"{what} has no port")

    fifos = []
    host_ends = [0] * len(bindings)
    claimed = [[0] * len(ports) for ports in tiles]
    for index, raw in enumerate(recipe["fifo"]):
        what = f"fifo {index}"
        fifo = kv(raw)
        if integer(fifo, "index", what) != index:
            raise DriverError(f"{what} is out of order (index {fifo['index']})")
        element = need(fifo, "elem", what)
        if element not in ELEMENTS:
            raise DriverError(f"{what}: element {element} is not a 32-bit word the core ABI moves")
        elems = integer(fifo, "elems", what, minimum=1)
        depth = integer(fifo, "depth", what, minimum=1)
        producer = parse_end(need(fifo, "producer", what), what)
        consumers = [parse_end(c, what) for c in need(fifo, "consumers", what).split(",") if c]
        if not consumers:
            raise DriverError(f"{what} has no consumer")
        if producer[0] == "host" and any(c[0] == "host" for c in consumers):
            raise DriverError(f"{what} runs from the host to the host")
        for end in (producer, *consumers):
            if end[0] == "host":
                if end[1] >= len(bindings):
                    raise DriverError(f"{what}: host binding {end[1]} does not exist; the recipe has {len(bindings)}")
                binding = bindings[end[1]]
                fills = end is producer
                if fills != (binding["access"] == "Read"):
                    raise DriverError(f"{what}: host binding {end[1]} is {binding['access']}, contradicting the FIFO's direction")
                if binding["elems"] != elems or binding["element"] != element:
                    raise DriverError(f"{what}: host binding {end[1]} does not hold the FIFO's {elems} {element} elements")
                host_ends[end[1]] += 1
            else:
                _, tile, port = end
                if tile >= len(tiles) or port >= len(tiles[tile]):
                    raise DriverError(f"{what}: tile {tile} port {port} does not exist")
                if tiles[tile][port] == index:
                    claimed[tile][port] += 1
        fifos.append({"element": element, "elems": elems, "depth": depth, "producer": producer, "consumers": consumers})
    for binding, count in enumerate(host_ends):
        if count != 1:
            raise DriverError(f"binding {binding} is the host end of {count} FIFOs, not one")
    for tile, ports in enumerate(tiles):
        for port, fifo in enumerate(ports):
            if fifo >= len(fifos):
                raise DriverError(f"tile {tile} port {port} lists fifo {fifo}, which the recipe does not have")
            if claimed[tile][port] != 1:
                raise DriverError(f"tile {tile} port {port} is wired by {claimed[tile][port]} FIFO ends, not one")

    objects = []
    for index, raw in enumerate(recipe["object"]):
        what = f"object {index}"
        obj = kv(raw)
        if integer(obj, "tile", what) != index:
            raise DriverError(f"{what} is out of order (tile {obj['tile']})")
        file = need(obj, "file", what)
        if not OBJECT_FILE.match(file):
            raise DriverError(f"{what}: file {file!r} is not a plain object file name")
        symbol = need(obj, "symbol", what)
        if not SYMBOL.match(symbol):
            raise DriverError(f"{what}: symbol {symbol!r} is not a C identifier")
        if objects_dir is not None and not os.path.isfile(os.path.join(objects_dir, file)):
            raise DriverError(f"{what}: {file!r} is not in the objects directory")
        objects.append({"file": file, "symbol": symbol})
    if len(objects) != len(tiles):
        raise DriverError(f"recipe has {len(objects)} objects for {len(tiles)} tiles")
    return arenas, bindings, tiles, fifos, objects


def link(platform, objects_dir, recipe_path, xclbin_path, insts_path):
    if platform not in PLATFORMS:
        raise DriverError(f"unsupported platform {platform!r}; supported: {', '.join(PLATFORMS)}")
    with open(recipe_path, encoding="utf-8") as handle:
        recipe = parse_recipe(handle.read())
    arenas, bindings, tiles, fifos, objects = check_recipe(recipe, platform, objects_dir)

    # The IRON imports are deferred: --version/--platforms must not need the compiler stack.
    import numpy as np
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU2
    from aie.utils.compile import compile_mlir_module

    device = {"NPU2": NPU2}[PLATFORMS[platform]]()

    with tempfile.TemporaryDirectory(prefix="poot-aie-link-") as work:
        # aiecc resolves a core's `link_with` object relative to its working directory.
        for obj in objects:
            shutil.copyfile(os.path.join(objects_dir, obj["file"]), os.path.join(work, obj["file"]))

        def fifo_type(fifo):
            return np.ndarray[(fifo["elems"],), np.dtype[getattr(np, ELEMENTS[fifo["element"]])]]

        object_fifos = [ObjectFifo(fifo_type(f), name=f"v{i}", depth=f["depth"]) for i, f in enumerate(fifos)]

        # One worker per tile. The core ABI (poot-npu CORE_ABI_REVISION 1): interleaved (ptr, i32 len)
        # pairs in body-parameter order, one pair per port.
        workers = []
        kernels = {}
        for index, ports in enumerate(tiles):
            kernel_args = []
            for fifo_index in ports:
                kernel_args += [fifo_type(fifos[fifo_index]), np.int32]
            # One kernel per core object: the symbol is global to the IRON module, so tiles that run the
            # same body share it. (Same object, same body, same ports: the types agree.)
            obj = objects[index]
            if obj["file"] not in kernels:
                kernels[obj["file"]] = Kernel(obj["symbol"], obj["file"], arg_types=kernel_args)
            kernel = kernels[obj["file"]]
            handles = []
            for port, fifo_index in enumerate(ports):
                if fifos[fifo_index]["producer"] == ("tile", index, port):
                    handles.append(object_fifos[fifo_index].prod())
                else:
                    handles.append(object_fifos[fifo_index].cons())
            lengths = [fifos[fifo_index]["elems"] for fifo_index in ports]

            def core_body(*handles_and_kernel, lengths=lengths):
                *core_handles, func = handles_and_kernel
                elements = [handle.acquire(1) for handle in core_handles]
                call = []
                for element, length in zip(elements, lengths):
                    call += [element, length]
                func(*call)
                for handle in core_handles:
                    handle.release(1)

            workers.append(Worker(core_body, fn_args=[*handles, kernel]))

        # The host side: one shim transfer per host binding, at its arena offset.
        arena_types = [
            np.ndarray[(a["words"],), np.dtype[getattr(np, ELEMENTS[a["element"]])]] for a in arenas
        ]
        transfers = []
        runtime_handles = []
        for fifo_index, fifo in enumerate(fifos):
            of = object_fifos[fifo_index]
            if fifo["producer"][0] == "host":
                transfers.append((True, bindings[fifo["producer"][1]]))
                runtime_handles.append(of.prod())
            for end in fifo["consumers"]:
                if end[0] == "host":
                    transfers.append((False, bindings[end[1]]))
                    runtime_handles.append(of.cons())

        def sequence(*args):
            buffers, handles = args[: len(arenas)], args[len(arenas) :]
            for (fills, binding), handle in zip(transfers, handles):
                tap = TensorAccessPattern(
                    (1, arenas[binding["arena"]]["words"]),
                    binding["offset"],
                    [1, 1, 1, binding["elems"]],
                    [0, 0, 0, 1],
                )
                if fills:
                    handle.fill(buffers[binding["arena"]], tap=tap)
                else:
                    handle.drain(buffers[binding["arena"]], tap=tap, wait=True)

        rt = Runtime(sequence, [*arena_types, *runtime_handles])
        module = Program(device, rt, workers=workers).resolve_program()

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

    def recipe(
        target="aie:isa=aie2p:platform=npu2:abi=1",
        invocations="1",
        tiles=None,
        fifos=None,
        arenas=None,
        bindings=None,
        objects=None,
        header=RECIPE_HEADER + "\n",
    ):
        # Two tiles: tile 0 adds host bindings 0 and 1 into the link fifo 2 (never host-visible); tile 1
        # adds that link to host binding 2's fifo and writes host binding 3.
        tiles = tiles if tiles is not None else ["index=0;ports=0,1,2", "index=1;ports=2,3,4"]
        fifos = fifos if fifos is not None else [
            "index=0;producer=host.0;consumers=tile.0.0;depth=2;elems=6;elem=F32",
            "index=1;producer=host.1;consumers=tile.0.1;depth=2;elems=6;elem=F32",
            "index=2;producer=tile.0.2;consumers=tile.1.0;depth=2;elems=6;elem=F32",
            "index=3;producer=host.2;consumers=tile.1.1;depth=2;elems=6;elem=F32",
            "index=4;producer=tile.1.2;consumers=host.3;depth=2;elems=6;elem=F32",
        ]
        arenas = arenas if arenas is not None else [
            "index=0;access=Read;elem=F32;words=18",
            "index=1;access=Write;elem=F32;words=6",
        ]
        bindings = bindings if bindings is not None else [
            "index=0;element=F32;dtype=F32;layout=Dense;access=Read;dims=[2, 3];accessed=6;arena=0;offset=0",
            "index=1;element=F32;dtype=F32;layout=Dense;access=Read;dims=[2, 3];accessed=6;arena=0;offset=6",
            "index=2;element=F32;dtype=F32;layout=Dense;access=Read;dims=[2, 3];accessed=6;arena=0;offset=12",
            "index=3;element=F32;dtype=F32;layout=Dense;access=Write;dims=[2, 3];accessed=6;arena=1;offset=0",
        ]
        objects = objects if objects is not None else [
            "tile=0;file=core-a.o;symbol=poot_seq_binary",
            "tile=1;file=core-b.o;symbol=poot_seq_binary",
        ]
        text = header + field("target", target) + field("invocations", invocations)
        text += "".join(field("tile", t) for t in tiles) + "".join(field("fifo", f) for f in fifos)
        text += "".join(field("arena", a) for a in arenas) + "".join(field("binding", b) for b in bindings)
        text += "".join(field("object", o) for o in objects)
        return text

    def refused(text, platform="npu2", needle=""):
        try:
            check_recipe(parse_recipe(text), platform)
        except DriverError as error:
            assert needle in str(error), f"{needle!r} not in {str(error)!r}"
            return
        raise AssertionError(f"accepted a recipe that must be refused ({needle!r})")

    arenas, bindings, tiles, fifos, objects = check_recipe(parse_recipe(recipe()), "npu2")
    assert len(arenas) == 2 and arenas[0]["words"] == 18 and arenas[1]["access"] == "Write"
    assert [b["offset"] for b in bindings] == [0, 6, 12, 0]
    assert tiles == [[0, 1, 2], [2, 3, 4]]
    assert fifos[2]["producer"] == ("tile", 0, 2) and fifos[2]["consumers"] == [("tile", 1, 0)]
    assert fifos[4]["consumers"] == [("host", 3)] and objects[1]["file"] == "core-b.o"

    good = recipe().split("\n")
    refused(recipe(header="poot-aie-recipe-v1\n"), needle="recipe header")
    refused(recipe().replace("target[33]", "target[34]"), needle="malformed recipe line")
    refused(recipe() + field("target", "aie:x"), needle="duplicate recipe field")
    refused("\n".join(l for l in good if not l.startswith("invocations")), needle="no 'invocations'")
    refused(recipe(), platform="npu1", needle="recipe targets platform")
    refused(recipe(target="gpu:isa=x"), needle="not an AIE target")
    refused(recipe(invocations="2"), needle="single finite invocation")
    refused(recipe(tiles=[]), needle="no tile")
    refused(recipe(fifos=[]), needle="no fifo")
    refused(recipe(arenas=[]), needle="no arena")
    refused(recipe(bindings=[]), needle="no binding")
    # tiles
    refused(recipe(tiles=["index=1;ports=0,1,2", "index=1;ports=2,3,4"]), needle="out of order")
    refused(recipe(tiles=["index=0;ports=", "index=1;ports=2,3,4"]), needle="has no port")
    refused(recipe(tiles=["index=0;ports=0,1,x", "index=1;ports=2,3,4"]), needle="port is 'x'")
    refused(recipe(tiles=["index=0;ports=0,1,9", "index=1;ports=2,3,4"]), needle="lists fifo 9")
    refused(recipe(tiles=["index=0;ports=0,1", "index=1;ports=2,3,4"]), needle="tile 0 port 2 does not exist")
    # fifos
    one = "index=0;producer=host.0;consumers=tile.0.0;depth=2;elems=6;elem=F32"
    rest = good_fifos = [
        "index=1;producer=host.1;consumers=tile.0.1;depth=2;elems=6;elem=F32",
        "index=2;producer=tile.0.2;consumers=tile.1.0;depth=2;elems=6;elem=F32",
        "index=3;producer=host.2;consumers=tile.1.1;depth=2;elems=6;elem=F32",
        "index=4;producer=tile.1.2;consumers=host.3;depth=2;elems=6;elem=F32",
    ]
    refused(recipe(fifos=[one.replace("elem=F32", "elem=Bf16"), *rest]), needle="32-bit word")
    refused(recipe(fifos=[one.replace("elems=6", "elems=0"), *rest]), needle="expected an integer >= 1")
    refused(recipe(fifos=[one.replace("depth=2", "depth=two"), *rest]), needle="depth is 'two'")
    refused(recipe(fifos=[one.replace("consumers=tile.0.0", "consumers="), *rest]), needle="has no consumer")
    refused(recipe(fifos=[one.replace("consumers=tile.0.0", "consumers=host.3"), *rest]), needle="host to the host")
    refused(recipe(fifos=[one.replace("host.0", "disk.0"), *rest]), needle="neither host")
    refused(recipe(fifos=[one.replace("host.0", "host.9"), *rest]), needle="host binding 9 does not exist")
    refused(recipe(fifos=[one.replace("host.0", "host.3"), *rest]), needle="contradicting the FIFO's direction")
    refused(recipe(fifos=[one.replace("elems=6", "elems=7"), *rest]), needle="does not hold the FIFO's 7")
    refused(recipe(fifos=[one.replace("tile.0.0", "tile.5.0"), *rest]), needle="tile 5 port 0 does not exist")
    refused(recipe(fifos=[one.replace("host.0", "host.1"), *rest]), needle="binding 0 is the host end of 0 FIFOs")
    refused(recipe(fifos=[one.replace("tile.0.0", "tile.0.1"), *rest]), needle="wired by 0 FIFO ends")
    refused(recipe(fifos=[one.replace("consumers=tile.0.0", "consumers=tile.0.0,tile.0.0"), *rest]), needle="wired by 2 FIFO ends")
    # arenas and bindings
    arena = ["index=0;access=Read;elem=F32;words=18", "index=1;access=Write;elem=F32;words=6"]
    refused(recipe(arenas=["index=1;access=Read;elem=F32;words=18", arena[1]]), needle="arena 0 is out of order")
    refused(recipe(arenas=["index=0;access=Both;elem=F32;words=18", arena[1]]), needle="unknown access")
    refused(recipe(arenas=["index=0;access=Read;elem=Bf16;words=18", arena[1]]), needle="32-bit word")
    refused(recipe(arenas=["index=0;access=Read;elem=F32;words=0", arena[1]]), needle="expected an integer >= 1")
    bound = [
        "index=0;element=F32;access=Read;dims=[2, 3];accessed=6;arena=0;offset=0",
        "index=1;element=F32;access=Read;dims=[2, 3];accessed=6;arena=0;offset=6",
        "index=2;element=F32;access=Read;dims=[2, 3];accessed=6;arena=0;offset=12",
        "index=3;element=F32;access=Write;dims=[2, 3];accessed=6;arena=1;offset=0",
    ]

    def bind(i, text):
        return [text if j == i else b for j, b in enumerate(bound)]

    refused(recipe(bindings=bind(0, bound[0].replace("index=0", "index=1"))), needle="binding 0 is out of order")
    refused(recipe(bindings=bind(0, bound[0].replace("F32", "I32", 1))), needle="does not belong to arena 0")
    refused(recipe(bindings=bind(0, bound[0].replace("Read", "Write"))), needle="does not belong to arena 0")
    refused(recipe(bindings=bind(0, bound[0].replace("arena=0", "arena=7"))), needle="arena 7 does not exist")
    refused(recipe(bindings=bind(2, bound[2].replace("offset=12", "offset=13"))), needle="run past arena 0")
    refused(recipe(bindings=bind(0, bound[0].replace("[2, 3]", "2, 3"))), needle="is not a list")
    refused(recipe(bindings=bind(0, bound[0].replace("[2, 3]", "[2, x]"))), needle="is not an integer")
    refused(recipe(bindings=bind(0, bound[0].replace("accessed=6", "accessed=7"))), needle="exceeds the binding's 6")
    refused(recipe(bindings=bind(0, bound[0].replace("index=0;", ""))), needle="no 'index'")
    # objects
    refused(recipe(objects=["tile=0;file=core-a.o;symbol=poot_seq_binary"]), needle="1 objects for 2 tiles")
    refused(recipe(objects=["tile=1;file=core-a.o;symbol=s", "tile=1;file=core-b.o;symbol=s"]), needle="object 0 is out of order")
    refused(recipe(objects=["tile=0;file=../core-a.o;symbol=s", "tile=1;file=core-b.o;symbol=s"]), needle="not a plain object file name")
    refused(recipe(objects=["tile=0;file=core-a.o;symbol=1bad", "tile=1;file=core-b.o;symbol=s"]), needle="not a C identifier")
    with tempfile.TemporaryDirectory() as empty:
        try:
            check_recipe(parse_recipe(recipe()), "npu2", empty)
        except DriverError as error:
            assert "not in the objects directory" in str(error), str(error)
        else:
            raise AssertionError("accepted a recipe whose object file is missing")


def main(argv):
    parser = argparse.ArgumentParser(prog="poot-aie-link", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--platforms", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="exercise the recipe refusals (no toolchain needed)")
    sub = parser.add_subparsers(dest="command")
    link_parser = sub.add_parser("link")
    for name in ("platform", "objects", "recipe", "xclbin", "insts"):
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
            link(args.platform, args.objects, args.recipe, args.xclbin, args.insts)
        else:
            parser.error("one of --version, --platforms, --self-test or `link` is required")
    except DriverError as error:
        print(f"poot-aie-link: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
