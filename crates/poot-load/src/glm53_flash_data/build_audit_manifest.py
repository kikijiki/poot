#!/usr/bin/env python3
"""Rebuild the GLM-5.3-Flash descriptor manifest from pinned metadata.

Inputs are the raw checkpoint index, the revision API response with `blobs=true`,
the prior manifest (for exclusion evidence), and 62 prefix-plus-header range
responses. The script reads no tensor payload.
"""

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalize(name):
    parts = name.split(".")
    replacements = {"layers": "{layer}", "experts": "{expert}", "blocks": "{block}"}
    for index, part in enumerate(parts[:-1]):
        if part in replacements and parts[index + 1].isdigit():
            parts[index + 1] = replacements[part]
    return ".".join(parts)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index_bytes = args.index.read_bytes()
    if sha256(index_bytes) != INDEX_SHA256:
        raise ValueError("raw index digest mismatch")
    index = json.loads(index_bytes)
    api = json.loads(args.api.read_bytes())
    previous = json.loads(args.previous.read_bytes())
    if api["sha"] != REVISION:
        raise ValueError("API revision mismatch")

    shard_api = {
        item["rfilename"]: item
        for item in api["siblings"]
        if item["rfilename"].startswith("model-")
        and item["rfilename"].endswith("-of-00062.safetensors")
    }
    if len(shard_api) != 62:
        raise ValueError("expected 62 shard API rows")

    tensor_meta = {}
    shard_rows = []
    for shard_name in sorted(shard_api):
        raw = (args.headers / f"{shard_name}.header").read_bytes()
        if len(raw) < 8:
            raise ValueError(f"short prefix for {shard_name}")
        header_length = struct.unpack("<Q", raw[:8])[0]
        header_bytes = raw[8:]
        if len(header_bytes) != header_length:
            raise ValueError(f"header length mismatch for {shard_name}")
        header = json.loads(header_bytes)
        header.pop("__metadata__", None)
        offsets_end = 0
        for name, meta in header.items():
            if name in tensor_meta:
                raise ValueError(f"duplicate tensor {name}")
            if index["weight_map"].get(name) != shard_name:
                raise ValueError(f"index/header shard mismatch for {name}")
            tensor_meta[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "shard": shard_name,
            }
            offsets_end = max(offsets_end, meta["data_offsets"][1])
        api_row = shard_api[shard_name]
        size = api_row["size"]
        if offsets_end + 8 + header_length != size:
            raise ValueError(f"payload/physical size mismatch for {shard_name}")
        shard_rows.append(
            {
                "name": shard_name,
                "size": size,
                "lfs_sha256": api_row["lfs"]["sha256"],
                "header_length": header_length,
                "header_sha256": sha256(header_bytes),
                "header_entry_count": len(header),
                "payload_bytes": offsets_end,
            }
        )

    if tensor_meta.keys() != index["weight_map"].keys():
        raise ValueError("raw index and complete header inventory differ")

    families = defaultdict(list)
    for name, meta in tensor_meta.items():
        scale_name = f"{name[:-7]}.weight_scale_inv" if name.endswith(".weight") else None
        if name.endswith(".weight_scale_inv"):
            weight_name = f"{name[:-17]}.weight"
            weight = tensor_meta.get(weight_name)
            if weight is None:
                raise ValueError(f"missing weight partner for {name}")
            role = "f32_block_scale"
            pair_pattern = normalize(weight_name)
            logical_shape = weight["shape"]
            block_shape = [128, 128]
        elif scale_name in tensor_meta:
            scale = tensor_meta[scale_name]
            if meta["dtype"] != "F8_E4M3" or scale["dtype"] != "F32":
                raise ValueError(f"unsupported quantized pair {name}")
            if meta["shard"] != scale["shard"]:
                raise ValueError(f"split quantized pair {name}")
            role = "e4m3_weight"
            pair_pattern = normalize(scale_name)
            logical_shape = meta["shape"]
            block_shape = [128, 128]
        else:
            role = "unquantized"
            pair_pattern = None
            logical_shape = meta["shape"]
            block_shape = None
        families[normalize(name)].append(
            {
                "dtype": meta["dtype"],
                "physical_shape": meta["shape"],
                "logical_shape": logical_shape,
                "quant_role": role,
                "pair_pattern": pair_pattern,
                "block_shape": block_shape,
                "shard": meta["shard"],
            }
        )

    family_rows = []
    for pattern in sorted(families):
        entries = families[pattern]
        variant_groups = defaultdict(list)
        for entry in entries:
            key = (
                entry["dtype"],
                tuple(entry["physical_shape"]),
                tuple(entry["logical_shape"]),
                entry["quant_role"],
                entry["pair_pattern"],
                tuple(entry["block_shape"]) if entry["block_shape"] else None,
            )
            variant_groups[key].append(entry["shard"])
        variants = []
        for key in sorted(variant_groups, key=lambda item: repr(item)):
            dtype, physical, logical, role, pair_pattern, block_shape = key
            shard_counts = Counter(variant_groups[key])
            variants.append(
                {
                    "dtype": dtype,
                    "physical_shape": list(physical),
                    "logical_shape": list(logical),
                    "quant_role": role,
                    "pair_pattern": pair_pattern,
                    "block_shape": list(block_shape) if block_shape else None,
                    "count": sum(shard_counts.values()),
                    "shards": [
                        {"name": shard, "count": count}
                        for shard, count in sorted(shard_counts.items())
                    ],
                }
            )
        family_rows.append({"pattern": pattern, "count": len(entries), "variants": variants})

    family_count_rows = "".join(f'{row["pattern"]}\t{row["count"]}\n' for row in family_rows)
    descriptors = {"tensor_families": family_rows, "shards": shard_rows}
    descriptor_sha256 = sha256(compact_json(descriptors))
    output = {
        "schema": "poot.glm53_flash.audit_manifest.v2",
        "source": previous["source"],
        "normalization": previous["normalization"],
        "total_size": index["metadata"]["total_size"],
        "tensor_count": len(index["weight_map"]),
        "shard_count": len(shard_rows),
        "tensor_family_sha256": sha256(family_count_rows.encode()),
        "descriptor_sha256": descriptor_sha256,
        "tensor_families": family_rows,
        "shards": shard_rows,
        "exclusion_source": previous["exclusion_source"],
        "exclusion_count": previous["exclusion_count"],
        "exclusion_canonical_sha256": previous["exclusion_canonical_sha256"],
        "exclusion_family_sha256": previous["exclusion_family_sha256"],
        "exclusion_families": previous["exclusion_families"],
        "selected_shard": previous["selected_shard"],
        "selected_tensors": previous["selected_tensors"],
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
