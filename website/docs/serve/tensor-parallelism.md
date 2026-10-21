---
id: tensor-parallelism
title: Tensor parallelism
sidebar_position: 12
---

# Tensor parallelism

`--placement tensor:ORD,ORD,...` asks for one tensor-parallel rank per GPU ordinal, on the `ptx` backend only: the
other backends cannot copy between their devices, so a group on them is refused at startup.
`--placement pipeline:ORD,ORD,...` splits the layers across GPUs instead and serves too. No environment variable
selects either.

## What a rank holds

- The weights are divided by role (Q, K, V, gate and up by output, O and down by input), and every rank loads only
  its share.
- The server's paged KV pool is divided by KV heads, so each rank holds its own heads of every pool.
- The memory fit checks each rank against its own GPU, with the per-rank pool and state bytes included. A rank that
  is short stops startup naming it.
- A head count that does not divide over the ranks is refused at startup by name.

## What is verified

Tensor groups and layer pipelines have served requests on two NVIDIA GPUs of one node joined by NVLink: greedy
requests matched the single-GPU result to rounding, and sampled requests matched the single-GPU sampler's tokens. A
tensor group of four NVLink GPUs decoded a tiny model to the single-GPU logits, every rank summing the partials of
all four. Not covered: serving a group of four, a real checkpoint, and GPUs that reach each other only across CPU
sockets (on one such host, peer copies delivered no data, so check peer copies on your hardware first).

Design: [Architecture: Serving design](../architecture/serving-design.mdx).
