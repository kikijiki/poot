---
id: speculative-decoding
title: Speculative decoding
sidebar_position: 10
---

# Speculative decoding

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today `/v1/completions` and `/v1/chat/completions` return `503` naming the missing engine, and the `POOT_SPEC_*` settings are not read.

Design notes on the method live under [Architecture: Serving design](../architecture/serving-design.mdx).
