# poot benchmark index

Every committed run, grouped by model and scenario. The point is to watch poot's column move as we optimize. Regenerate with `bench index`. Full tables: `bench report results/<run>`; to test a change, `bench compare --baseline <runs> --candidate <runs>` (5 repeats per side). `behind` is how many days a run is older than the newest run here.

## olmo2-1b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-19-geforce-rtx-3090-olmo2-1b-91a511b2 | 2026-08-19 | 36 | - | 91a511b2 | GeForce RTX 3090 | 39.2 | 290.6 | 11.08 | llamacpp 279.4 tok/s | 0.140x |
| 2026-09-23-strix-halo-olmo2-1b-rocm-7e71ad05 | 2026-09-23 | 1 | - | 7e71ad05 | AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) | 0.4 | 2855.6 | - | - | - |
| 2026-09-25-strix-halo-olmo2-1b-rocm-6b64abcb2 | 2026-09-24 | 0 | - | 6b64abcb | AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) | 15.9 | 1929.4 | - | - | - |
| 2026-09-25-strix-halo-olmo2-1b-wgpu-6b64abcb2 | 2026-09-24 | 0 | - | 6b64abcb | AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) | 12.4 | 404.1 | - | - | - |

## phi-3.5-mini - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-19-geforce-rtx-3090-phi-3-5-mini-1ece8859 | 2026-08-19 | 36 | - | 1ece8859 | GeForce RTX 3090 | 25.8 | 513.3 | 21.99 | llamacpp 100.4 tok/s | 0.257x |

## phi-4 - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-20-rtx-a6000-phi-4-b9b21fef | 2026-08-20 | 35 | - | - | RTX A6000 | - | - | - | llamacpp 24.0 tok/s | - |

## qwen2.5-0.5b - decode

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-01-strix-halo-qwen2-5-0-5b-2cf7758 | 2026-07-01 | 85 | ptx | 2cf7758 | AMD Radeon 8060S (Strix Halo, gfx1151, RDNA3.5) | 94.6 | 82.0 | - | - | - |

## qwen2.5-0.5b - decode-128

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen2-5-0-5b-7507b10 | 2026-06-17 | 99 | - | 7507b10 | GeForce RTX 3090 | 88.6 | 1031.9 | 3.25 | llamacpp 454.6 tok/s | 0.195x |
| 2026-07-03-geforce-rtx-3090-qwen2-5-0-5b-8c9192e7 | 2026-07-03 | 83 | - | - | GeForce RTX 3090 | - | - | - | llamacpp 462.6 tok/s | - |
| 2026-07-03-geforce-rtx-3090-qwen2-5-0-5b-a39b146b | 2026-07-03 | 83 | - | a39b146b | GeForce RTX 3090 | 57.5 | 123.2 | 3.22 | llamacpp 461.5 tok/s | 0.125x |
| 2026-07-03-geforce-rtx-3090-qwen2-5-0-5b-e2a3070c | 2026-07-03 | 83 | - | e2a3070c | GeForce RTX 3090 | 44.1 | 159.1 | 3.22 | llamacpp 440.7 tok/s | 0.100x |
| 2026-07-28-geforce-rtx-3090-qwen2-5-0-5b-78ca82f2 | 2026-07-28 | 58 | - | 78ca82f2 | GeForce RTX 3090 | 100.8 | 73.1 | 3.22 | llamacpp 423.8 tok/s | 0.238x |

## qwen2.5-0.5b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-13-geforce-rtx-3090-4b8e0b3 | 2026-06-13 | 103 | - | 4b8e0b3 | GeForce RTX 3090 | 38.7 | 3170.5 | 2.20 | vllm 484.2 tok/s | 0.080x |
| 2026-06-14-geforce-rtx-3090-qwen2-5-0-5b-nosha | 2026-06-14 | 102 | - | - | GeForce RTX 3090 | - | - | - | candle 135.5 tok/s | - |
| 2026-06-17-geforce-rtx-3090-qwen2-5-0-5b-9136d22 | 2026-06-17 | 99 | - | - | GeForce RTX 3090 | - | - | - | llamacpp 452.0 tok/s | - |
| 2026-06-18-a40-qwen2-5-0-5b-4f54175 | 2026-06-18 | 98 | - | 4f54175 | A40 | 45.4 | 531.5 | 11.20 | llamacpp 378.9 tok/s | 0.120x |
| 2026-06-19-l40s-qwen2-5-0-5b-f284ad7 | 2026-06-19 | 97 | - | f284ad7 | L40S | 92.7 | 121.5 | 11.39 | llamacpp 487.1 tok/s | 0.190x |
| 2026-06-24-geforce-rtx-3090-qwen2-5-0-5b-8120062 | 2026-06-24 | 92 | - | 8120062 | GeForce RTX 3090 | 71.1 | 165.5 | 11.08 | llamacpp 449.4 tok/s | 0.158x |
| 2026-07-03-geforce-rtx-3090-qwen2-5-0-5b-53b9dfb8 | 2026-07-03 | 83 | - | 53b9dfb8 | GeForce RTX 3090 | 31.5 | 119.4 | 11.19 | llamacpp 425.0 tok/s | 0.074x |
| 2026-07-12-strix-halo-qwen2-5-0-5b-decode-curve-c4b4a90a | 2026-07-12 | 74 | - | c4b4a90a | AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) | 25.7 | 144.1 | - | llamacpp 357.0 tok/s | 0.072x |
| 2026-07-12-strix-halo-qwen2-5-0-5b-decode-curve-rocm-4512e82b | 2026-07-12 | 74 | - | 4512e82b | AMD Radeon 8060S Graphics (RADV/ROCm STRIX_HALO, gfx1151, RDNA3.5) | 25.9 | 235.6 | - | llamacpp 357.0 tok/s | 0.073x |
| 2026-08-20-geforce-rtx-3090-qwen2-5-0-5b-319c6ca3 | 2026-08-20 | 35 | - | 319c6ca3 | GeForce RTX 3090 | 74.0 | 140.2 | 11.08 | llamacpp 464.6 tok/s | 0.159x |
| 2026-09-10-geforce-rtx-3090-qwen2-5-0-5b-d28d5fbe | 2026-09-10 | 14 | - | d28d5fbe | GeForce RTX 3090 | 72.6 | 165.8 | 10.40 | vllm 476.1 tok/s | 0.153x |

## qwen2.5-0.5b - prefill-1k

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen2-5-0-5b-f6c982a | 2026-06-17 | 99 | - | f6c982a | GeForce RTX 3090 | 60.5 | 13875.5 | 3.53 | llamacpp 416.2 tok/s | 0.145x |
| 2026-07-03-geforce-rtx-3090-qwen2-5-0-5b-67e70970 | 2026-07-03 | 83 | - | 67e70970 | GeForce RTX 3090 | 12.0 | 86572.5 | 3.50 | llamacpp 458.9 tok/s | 0.026x |
| 2026-08-20-geforce-rtx-3090-qwen2-5-0-5b-151fd498 | 2026-08-20 | 35 | - | 151fd498 | GeForce RTX 3090 | 53.4 | 17723.5 | 4.72 | llamacpp 452.9 tok/s | 0.118x |
| 2026-08-20-geforce-rtx-3090-qwen2-5-0-5b-1f63ed0a | 2026-08-20 | 35 | - | 1f63ed0a | GeForce RTX 3090 | 49.1 | 1677.2 | 4.73 | llamacpp 462.0 tok/s | 0.106x |

## qwen2.5-1.5b - decode-128

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-12-strix-halo-qwen2-5-1-5b-q4km-edca5d2f | 2026-07-12 | 74 | wgpu | edca5d2f | AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) | 42.8 | 133.9 | - | - | - |

## qwen2.5-1.5b - pp512+tg128

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-12-strix-halo-qwen2-5-1-5b-q4km-edca5d2f | 2026-07-12 | 74 | - | - | AMD Radeon 8060S Graphics (RADV STRIX_HALO, gfx1151, RDNA3.5) | - | - | - | - | - |

## qwen2.5-3b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-14-geforce-rtx-3090-qwen2-5-3b-nosha | 2026-06-14 | 102 | - | - | GeForce RTX 3090 | - | - | - | - | - |

## qwen3-0.6b - decode-128

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen3-0-6b-7507b10 | 2026-06-17 | 99 | - | 7507b10 | GeForce RTX 3090 | 78.5 | 1268.5 | 3.75 | vllm 395.4 tok/s | 0.198x |
| 2026-07-03-geforce-rtx-3090-qwen3-0-6b-a39b146b | 2026-07-03 | 83 | - | a39b146b | GeForce RTX 3090 | 45.4 | 168.4 | 3.58 | vllm 397.5 tok/s | 0.114x |

## qwen3-0.6b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-13-geforce-rtx-3090-ca0b7b2 | 2026-06-13 | 103 | - | ca0b7b2 | GeForce RTX 3090 | 26.4 | 4224.6 | 2.69 | vllm 403.6 tok/s | 0.065x |
| 2026-06-14-geforce-rtx-3090-qwen3-0-6b-nosha | 2026-06-14 | 102 | - | - | GeForce RTX 3090 | - | - | - | candle 139.9 tok/s | - |
| 2026-06-17-geforce-rtx-3090-qwen3-0-6b-9136d22 | 2026-06-17 | 99 | - | - | GeForce RTX 3090 | - | - | - | vllm 399.2 tok/s | - |
| 2026-06-18-a40-qwen3-0-6b-4f54175 | 2026-06-18 | 98 | - | 4f54175 | A40 | 38.1 | 780.5 | 15.73 | llamacpp 316.2 tok/s | 0.120x |
| 2026-06-19-l40s-qwen3-0-6b-f284ad7 | 2026-06-19 | 97 | - | f284ad7 | L40S | 77.4 | 170.2 | 15.92 | llamacpp 407.1 tok/s | 0.190x |
| 2026-06-24-geforce-rtx-3090-qwen3-0-6b-8120062 | 2026-06-24 | 92 | - | 8120062 | GeForce RTX 3090 | 56.7 | 189.5 | 15.61 | vllm 404.7 tok/s | 0.140x |
| 2026-07-03-geforce-rtx-3090-qwen3-0-6b-53b9dfb8 | 2026-07-03 | 83 | - | 53b9dfb8 | GeForce RTX 3090 | 27.5 | 169.7 | 15.61 | vllm 396.3 tok/s | 0.069x |

## qwen3-0.6b - prefill-1k

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen3-0-6b-f6c982a | 2026-06-17 | 99 | - | f6c982a | GeForce RTX 3090 | 49.4 | 17327.6 | 4.71 | llamacpp 362.3 tok/s | 0.136x |
| 2026-07-03-geforce-rtx-3090-qwen3-0-6b-67e70970 | 2026-07-03 | 83 | - | 67e70970 | GeForce RTX 3090 | 6.8 | 151882.3 | 4.14 | llamacpp 391.7 tok/s | 0.017x |

## qwen3-14b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-20-l40s-qwen3-14b-474f03b2 | 2026-08-20 | 35 | - | - | L40S | - | - | - | llamacpp 25.8 tok/s | - |

## qwen3-4b - decode-128

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen3-4b-7507b10 | 2026-06-17 | 99 | - | 7507b10 | GeForce RTX 3090 | 17.7 | 13649.0 | 18.02 | llamacpp 95.1 tok/s | 0.186x |
| 2026-07-28-geforce-rtx-3090-qwen3-4b-ded22c11 | 2026-07-28 | 58 | - | ded22c11 | GeForce RTX 3090 | 23.7 | 293.6 | 17.33 | llamacpp 94.8 tok/s | 0.250x |

## qwen3-4b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen3-4b-9136d22 | 2026-06-17 | 99 | - | - | GeForce RTX 3090 | - | - | - | llamacpp 94.0 tok/s | - |
| 2026-06-18-a40-qwen3-4b-4f54175 | 2026-06-18 | 98 | - | - | A40 | - | - | - | llamacpp 67.5 tok/s | - |
| 2026-06-19-l40s-qwen3-4b-f284ad7 | 2026-06-19 | 97 | - | f284ad7 | L40S | 23.1 | 955.2 | 38.83 | llamacpp 83.7 tok/s | 0.276x |

## qwen3-4b - prefill-1k

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-17-geforce-rtx-3090-qwen3-4b-f6c982a | 2026-06-17 | 99 | - | f6c982a | GeForce RTX 3090 | 9.7 | 66945.5 | 19.64 | llamacpp 92.2 tok/s | 0.105x |

## smollm3-3b - decode-curve

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-20-geforce-rtx-3090-smollm3-3b-d4461ffc | 2026-08-20 | 35 | - | d4461ffc | GeForce RTX 3090 | 16.5 | 693.3 | 23.96 | llamacpp 121.0 tok/s | 0.136x |
