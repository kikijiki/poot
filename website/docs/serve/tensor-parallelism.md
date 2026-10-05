---
id: tensor-parallelism
title: Tensor parallelism
sidebar_position: 12
---

# Tensor parallelism

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today the server refuses `--tensor-parallel` as an unknown option, and `POOT_TP_GPUS` is not read.

Tensor parallelism splits one model's weights across several GPUs. Design notes live under
[Architecture: Serving design](../architecture/serving-design.mdx).
