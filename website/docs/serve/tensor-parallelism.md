---
id: tensor-parallelism
title: Tensor parallelism
sidebar_position: 12
---

# Tensor parallelism

`--placement tensor:ORD,ORD,...` asks for one tensor-parallel rank per GPU ordinal, on the `ptx` backend only. The weights are divided by role (Q, K, V, gate and up by output, O and down by input) and every rank loads only its share. The memory fit checks each rank against its own GPU, and a rank that is short stops startup naming it. The shard pass divides the server's paged KV pool by KV heads, so each rank holds its own heads of every pool, and the per-rank pool and state bytes are part of that rank's fit. A head count that does not divide over the ranks is refused at startup by name. Tensor groups and layer pipelines have served requests on two NVIDIA GPUs of one node joined by NVLink: greedy requests matched the single-GPU result to rounding, and sampled requests matched the single-GPU sampler's tokens. A tensor group of four NVIDIA GPUs joined by NVLink decoded a tiny model to the single-GPU logits, every rank summing the partials of all four. Not yet covered: serving a group of four, a real checkpoint, and GPUs that reach each other only across CPU sockets (on one such host, peer copies delivered no data, so check peer copies on your hardware first). `--placement pipeline:ORD,ORD,...` splits the layers across GPUs and serves. No environment variable selects either.

Tensor parallelism splits one model's weights across several GPUs. Design notes live under
[Architecture: Serving design](../architecture/serving-design.mdx).
