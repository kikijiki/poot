# poot benchmark index

Every committed run, grouped by model and scenario. The point is to watch poot's column move as we optimize. Regenerate with `bench index`. Full tables: `bench report results/<run>`; to test a change, `bench compare --baseline <runs> --candidate <runs>` (5 repeats per side). `behind` is how many days a run is older than the newest run here.

## qwen3-0.6b - decode-curve

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-091516-r1 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9335 32-Core Processor | - | - | - | transformers 112.0 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-091516-r2 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9335 32-Core Processor | - | - | - | transformers 111.5 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-091516-r3 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9335 32-Core Processor | - | - | - | transformers 112.5 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-091516-r4 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9335 32-Core Processor | - | - | - | transformers 113.2 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-091516-r5 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9335 32-Core Processor | - | - | - | transformers 113.4 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-0-6b-073455-r1 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 161.3 | 54.5 | 4.46 (load 0.85) | llamacpp 700.8 tok/s | 0.230x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-0-6b-073455-r2 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 160.2 | 54.5 | 4.46 (load 2.30) | llamacpp 702.2 tok/s | 0.228x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-0-6b-073455-r3 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 159.9 | 54.6 | 4.46 (load 2.57) | llamacpp 703.5 tok/s | 0.227x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-0-6b-073455-r4 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 159.8 | 54.6 | 4.46 (load 2.40) | llamacpp 697.5 tok/s | 0.229x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-0-6b-073455-r5 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 159.8 | 54.6 | 4.46 (load 2.58) | llamacpp 700.5 tok/s | 0.228x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-143238-r1 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 295.0 | 40.4 | 3.59 (load 2.95) | llamacpp 599.1 tok/s | 0.492x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-143238-r2 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 292.0 | 40.5 | 3.59 (load 3.08) | llamacpp 615.7 tok/s | 0.474x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-143238-r3 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 294.1 | 40.3 | 3.59 (load 3.08) | llamacpp 611.7 tok/s | 0.481x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-143238-r4 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 292.9 | 43.4 | 3.59 (load 3.08) | llamacpp 613.4 tok/s | 0.478x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-143238-r5 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 293.8 | 41.8 | 3.59 (load 3.08) | llamacpp 615.4 tok/s | 0.477x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-153650-r1 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 289.6 | 42.1 | 3.59 (load 2.21) | llamacpp 606.2 tok/s | 0.478x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-153650-r2 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 288.3 | 43.8 | 3.59 (load 3.08) | llamacpp 622.3 tok/s | 0.463x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-153650-r3 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 298.2 | 40.7 | 3.59 (load 3.57) | llamacpp 621.7 tok/s | 0.480x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-153650-r4 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 298.0 | 41.9 | 3.59 (load 3.08) | llamacpp 618.1 tok/s | 0.482x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-0-6b-153650-r5 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 295.7 | 41.7 | 3.59 (load 3.08) | llamacpp 621.9 tok/s | 0.475x |
| 2026-10-26-strix-06b-llamacpp-r1 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 153.6 tok/s | - |
| 2026-10-26-strix-06b-llamacpp-r2 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 153.7 tok/s | - |
| 2026-10-26-strix-06b-llamacpp-r3 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 152.8 tok/s | - |
| 2026-10-26-strix-06b-llamacpp-r4 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 153.9 tok/s | - |
| 2026-10-26-strix-06b-llamacpp-r5 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 152.2 tok/s | - |
| 2026-10-26-strix-06b-rocm-r1 | 2026-10-26 | 1 | rocm | 80bee780 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 101.9 | 70.6 | 3.21 (load 3.17) | - | - |
| 2026-10-26-strix-06b-rocm-r2 | 2026-10-26 | 1 | rocm | 80bee780 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 101.9 | 69.8 | 3.21 (load 2.68) | - | - |
| 2026-10-26-strix-06b-rocm-r3 | 2026-10-26 | 1 | rocm | 80bee780 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 102.4 | 70.2 | 3.21 (load 2.68) | - | - |
| 2026-10-26-strix-06b-rocm-r4 | 2026-10-26 | 1 | rocm | 80bee780 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 102.0 | 70.2 | 3.21 (load 2.68) | - | - |
| 2026-10-26-strix-06b-rocm-r5 | 2026-10-26 | 1 | rocm | 80bee780 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 101.8 | 69.8 | 3.21 (load 2.68) | - | - |
| 2026-10-26-strix-06b-vulkan-r1 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 92.0 | 71.7 | 3.16 (load 2.92) | - | - |
| 2026-10-26-strix-06b-vulkan-r2 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 92.0 | 67.0 | 3.16 (load 2.92) | - | - |
| 2026-10-26-strix-06b-vulkan-r3 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 92.0 | 67.3 | 3.16 (load 2.92) | - | - |
| 2026-10-26-strix-06b-vulkan-r4 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 91.9 | 65.3 | 3.16 (load 2.92) | - | - |
| 2026-10-26-strix-06b-vulkan-r5 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 92.3 | 66.8 | 3.16 (load 2.92) | - | - |
| 2026-10-26-strix-06b-wgpu-r1 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 83.1 | 149.6 | 3.40 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r2 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 83.7 | 369.2 | 4.28 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r3 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 83.1 | 347.4 | 4.15 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r4 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 82.7 | 156.8 | 4.28 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r5 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 83.8 | 142.9 | 4.28 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r6 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 85.8 | 155.8 | 4.03 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r7 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 84.4 | 152.4 | 4.15 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r8 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 82.7 | 156.0 | 4.15 (load 4.52) | - | - |
| 2026-10-26-strix-06b-wgpu-r9 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 85.3 | 157.2 | 3.40 (load 4.52) | - | - |

## qwen3-4b - decode-curve

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-4b-090426-r1 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | - | - | - | transformers 84.7 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-4b-090426-r2 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | - | - | - | transformers 84.5 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-4b-090426-r3 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | - | - | - | transformers 86.0 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-4b-090426-r4 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | - | - | - | transformers 85.5 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-4b-090426-r5 | 2026-10-24 | 3 | - | - | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | - | - | - | transformers 83.5 tok/s | - |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-073455-r1 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 78.9 | 151.3 | 12.32 (load 7.26) | llamacpp 169.2 tok/s | 0.466x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-073455-r2 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 78.7 | 152.1 | 12.86 (load 10.94) | llamacpp 169.1 tok/s | 0.465x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-073455-r3 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 78.7 | 152.1 | 12.32 (load 1.94) | llamacpp 169.0 tok/s | 0.465x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-073455-r4 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 78.6 | 152.3 | 12.32 (load 3.54) | llamacpp 169.3 tok/s | 0.464x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-073455-r5 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 78.6 | 152.1 | 12.32 (load 12.31) | llamacpp 169.3 tok/s | 0.464x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-143238-r1 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 113.1 | 110.4 | 11.20 (load 11.17) | llamacpp 155.8 tok/s | 0.726x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-143238-r2 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 114.3 | 106.1 | 11.20 (load 9.81) | llamacpp 155.6 tok/s | 0.735x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-143238-r3 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 114.5 | 106.6 | 11.20 (load 11.17) | llamacpp 155.5 tok/s | 0.736x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-143238-r4 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 114.4 | 107.4 | 11.20 (load 11.17) | llamacpp 155.7 tok/s | 0.735x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-143238-r5 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 114.4 | 107.2 | 11.20 (load 9.81) | llamacpp 155.6 tok/s | 0.736x |
| 2026-10-26-strix-4b-llamacpp-r1 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 25.7 tok/s | - |
| 2026-10-26-strix-4b-llamacpp-r2 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 25.7 tok/s | - |
| 2026-10-26-strix-4b-llamacpp-r3 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 25.7 tok/s | - |
| 2026-10-26-strix-4b-llamacpp-r4 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 25.6 tok/s | - |
| 2026-10-26-strix-4b-llamacpp-r5 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 25.7 tok/s | - |
| 2026-10-26-strix-4b-rocm-r1 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 25.2 | 249.1 | 10.81 (load 9.42) | - | - |
| 2026-10-26-strix-4b-rocm-r2 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 25.2 | 261.5 | 10.81 (load 9.07) | - | - |
| 2026-10-26-strix-4b-rocm-r3 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 25.2 | 267.1 | 10.81 (load 10.78) | - | - |
| 2026-10-26-strix-4b-rocm-r4 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 25.2 | 261.8 | 10.81 (load 10.79) | - | - |
| 2026-10-26-strix-4b-rocm-r5 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 25.2 | 264.8 | 10.81 (load 10.78) | - | - |
| 2026-10-26-strix-4b-vulkan-r1 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 24.4 | 253.5 | 11.07 (load 10.39) | - | - |
| 2026-10-26-strix-4b-vulkan-r2 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 24.4 | 252.1 | 11.07 (load 10.39) | - | - |
| 2026-10-26-strix-4b-vulkan-r3 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 24.4 | 265.2 | 11.07 (load 10.39) | - | - |
| 2026-10-26-strix-4b-vulkan-r4 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 24.4 | 251.4 | 11.07 (load 10.39) | - | - |
| 2026-10-26-strix-4b-vulkan-r5 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 24.4 | 257.2 | 11.07 (load 10.39) | - | - |
| 2026-10-26-strix-4b-wgpu-r1 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 23.4 | 422.5 | 12.36 (load 21.04) | - | - |
| 2026-10-26-strix-4b-wgpu-r2 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 23.4 | 418.3 | 11.73 (load 21.04) | - | - |
| 2026-10-26-strix-4b-wgpu-r3 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 23.4 | 417.4 | 11.86 (load 21.04) | - | - |
| 2026-10-26-strix-4b-wgpu-r4 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 23.4 | 419.1 | 12.99 (load 21.04) | - | - |
| 2026-10-26-strix-4b-wgpu-r5 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 23.4 | 670.5 | 12.24 (load 21.04) | - | - |

## qwen3-4b-q4km - decode-curve

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-q4km-073455-r1 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 85.8 | 187.5 | 7.73 (load 5.80) | llamacpp 357.8 tok/s | 0.240x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-q4km-073455-r2 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 85.6 | 188.1 | 7.18 (load 4.84) | llamacpp 357.5 tok/s | 0.240x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-q4km-073455-r3 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 85.5 | 188.3 | 7.18 (load 5.45) | llamacpp 357.2 tok/s | 0.239x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-q4km-073455-r4 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 85.5 | 188.4 | 7.18 (load 5.80) | llamacpp 356.7 tok/s | 0.240x |
| 2026-10-24-rtx-pro-6000-blackwell-workstation-edition-qwen3-4b-q4km-073455-r5 | 2026-10-24 | 3 | ptx | d4653ef3 | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9575F 64-Core Processor | 85.5 | 188.5 | 7.18 (load 5.45) | llamacpp 357.0 tok/s | 0.239x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-q4km-143238-r1 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 158.5 | 141.1 | 6.06 (load 4.33) | llamacpp 314.7 tok/s | 0.504x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-q4km-143238-r2 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 163.2 | 138.0 | 6.06 (load 6.04) | llamacpp 319.8 tok/s | 0.510x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-q4km-143238-r3 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 163.2 | 136.1 | 6.06 (load 6.04) | llamacpp 319.7 tok/s | 0.511x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-q4km-143238-r4 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 163.2 | 135.8 | 6.06 (load 6.05) | llamacpp 319.7 tok/s | 0.510x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-4b-q4km-143238-r5 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 163.1 | 137.0 | 6.60 (load 6.04) | llamacpp 320.4 tok/s | 0.509x |
| 2026-10-26-strix-4bq4-llamacpp-r1 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 77.2 tok/s | - |
| 2026-10-26-strix-4bq4-llamacpp-r2 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 77.8 tok/s | - |
| 2026-10-26-strix-4bq4-llamacpp-r3 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 77.4 tok/s | - |
| 2026-10-26-strix-4bq4-llamacpp-r4 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 77.4 tok/s | - |
| 2026-10-26-strix-4bq4-llamacpp-r5 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 77.5 tok/s | - |
| 2026-10-26-strix-4bq4-rocm-r1 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 59.3 | 226.8 | 5.67 (load 4.28) | - | - |
| 2026-10-26-strix-4bq4-rocm-r2 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 58.9 | 236.2 | 5.67 (load 4.28) | - | - |
| 2026-10-26-strix-4bq4-rocm-r3 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 58.8 | 232.8 | 5.67 (load 4.28) | - | - |
| 2026-10-26-strix-4bq4-rocm-r4 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 59.0 | 238.5 | 5.67 (load 4.28) | - | - |
| 2026-10-26-strix-4bq4-rocm-r5 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 58.6 | 235.3 | 5.67 (load 4.28) | - | - |
| 2026-10-26-strix-4bq4-vulkan-r1 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 60.6 | 208.0 | 5.92 (load 5.22) | - | - |
| 2026-10-26-strix-4bq4-vulkan-r2 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 60.7 | 213.9 | 5.92 (load 5.22) | - | - |
| 2026-10-26-strix-4bq4-vulkan-r3 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 60.5 | 207.7 | 5.92 (load 5.22) | - | - |
| 2026-10-26-strix-4bq4-vulkan-r4 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 60.9 | 209.5 | 5.92 (load 5.22) | - | - |
| 2026-10-26-strix-4bq4-vulkan-r5 | 2026-10-26 | 1 | vulkan | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 60.7 | 210.5 | 5.92 (load 5.22) | - | - |
| 2026-10-26-strix-4bq4-wgpu-r1 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 50.4 | 320.6 | 6.81 (load 7.80) | - | - |
| 2026-10-26-strix-4bq4-wgpu-r2 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 51.8 | 330.6 | 5.56 (load 7.80) | - | - |
| 2026-10-26-strix-4bq4-wgpu-r3 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 50.5 | 338.8 | 6.81 (load 7.80) | - | - |
| 2026-10-26-strix-4bq4-wgpu-r4 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 49.6 | 335.3 | 6.31 (load 7.80) | - | - |
| 2026-10-26-strix-4bq4-wgpu-r5 | 2026-10-26 | 1 | wgpu | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 49.9 | 332.6 | 5.56 (load 7.80) | - | - |

## qwen3.8-27b-q4 - decode-curve

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-143238-r1 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 38.0 | 537.8 | 25.76 (load 25.49) | llamacpp 67.5 tok/s | 0.563x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-143238-r2 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 38.0 | 533.3 | 25.76 (load 25.49) | llamacpp 67.5 tok/s | 0.564x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-143238-r3 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 38.0 | 533.3 | 25.76 (load 25.49) | llamacpp 67.4 tok/s | 0.564x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-143238-r4 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 38.0 | 533.3 | 25.76 (load 25.72) | llamacpp 67.5 tok/s | 0.564x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-143238-r5 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 38.1 | 532.9 | 26.30 (load 25.72) | llamacpp 67.5 tok/s | 0.564x |

## qwen3.8-27b-q4 - decode-curve-large

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-075810-r1 | 2026-10-24 | 3 | ptx | 96ddd6f9 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9554 64-Core Processor | 20.8 | 735.5 | 24.64 (load 24.62) | llamacpp 66.2 tok/s | 0.315x |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-075810-r2 | 2026-10-24 | 3 | ptx | 96ddd6f9 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9554 64-Core Processor | 22.0 | 703.7 | 24.64 (load 24.62) | llamacpp 67.3 tok/s | 0.326x |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-075810-r3 | 2026-10-24 | 3 | ptx | 96ddd6f9 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9554 64-Core Processor | 22.0 | 702.6 | 24.64 (load 24.39) | llamacpp 67.2 tok/s | 0.327x |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-075810-r4 | 2026-10-24 | 3 | ptx | 96ddd6f9 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9554 64-Core Processor | 22.0 | 703.5 | 24.64 (load 24.62) | llamacpp 67.2 tok/s | 0.327x |
| 2026-10-24-rtx-pro-6000-blackwell-server-edition-qwen3-8-27b-q4-075810-r5 | 2026-10-24 | 3 | ptx | 96ddd6f9 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9554 64-Core Processor | 22.0 | 701.6 | 25.19 (load 24.39) | llamacpp 67.2 tok/s | 0.327x |
| 2026-10-26-strix-27bq4-llamacpp-r1 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 12.3 tok/s | - |
| 2026-10-26-strix-27bq4-llamacpp-r2 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 12.2 tok/s | - |
| 2026-10-26-strix-27bq4-llamacpp-r3 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 12.2 tok/s | - |
| 2026-10-26-strix-27bq4-llamacpp-r4 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 12.2 tok/s | - |
| 2026-10-26-strix-27bq4-llamacpp-r5 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 12.2 tok/s | - |
| 2026-10-26-strix-27bq4-rocm-r1 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 9.8 | 1285.7 | 23.75 (load 23.49) | - | - |
| 2026-10-26-strix-27bq4-rocm-r2 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 9.8 | 1295.6 | 23.75 (load 23.49) | - | - |
| 2026-10-26-strix-27bq4-rocm-r3 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 9.8 | 1287.0 | 23.75 (load 23.49) | - | - |
| 2026-10-26-strix-27bq4-rocm-r4 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 9.8 | 1295.0 | 23.75 (load 23.49) | - | - |
| 2026-10-26-strix-27bq4-rocm-r5 | 2026-10-26 | 1 | rocm | f2936210 | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 9.8 | 1287.7 | 23.75 (load 23.49) | - | - |

## qwen3.8-flash-next-q2 - decode-curve

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-135322-r1 | 2026-10-23 | 4 | ptx | 3ba1c127 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9334 32-Core Processor | 39.9 | 823.2 | 90.06 (load 90.01) | llamacpp 98.6 tok/s | 0.405x |
| 2026-10-23-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-135322-r2 | 2026-10-23 | 4 | ptx | 3ba1c127 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9334 32-Core Processor | 39.6 | 825.1 | 90.06 (load 90.01) | llamacpp 98.8 tok/s | 0.401x |
| 2026-10-23-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-135322-r3 | 2026-10-23 | 4 | ptx | 3ba1c127 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9334 32-Core Processor | 39.6 | 824.6 | 90.06 (load 89.82) | llamacpp 98.2 tok/s | 0.403x |
| 2026-10-23-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-135322-r4 | 2026-10-23 | 4 | ptx | 3ba1c127 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9334 32-Core Processor | 39.6 | 824.2 | 90.06 (load 90.01) | llamacpp 98.4 tok/s | 0.403x |
| 2026-10-23-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-135322-r5 | 2026-10-23 | 4 | ptx | 3ba1c127 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9334 32-Core Processor | 39.6 | 824.6 | 90.06 (load 89.82) | llamacpp 98.4 tok/s | 0.403x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-143238-r1 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 50.9 | 867.4 | 94.03 (load 93.99) | llamacpp 97.2 tok/s | 0.523x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-143238-r2 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 51.1 | 866.8 | 94.03 (load 93.99) | llamacpp 99.9 tok/s | 0.512x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-143238-r3 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 51.0 | 867.1 | 94.03 (load 93.80) | llamacpp 98.3 tok/s | 0.519x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-143238-r4 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 51.1 | 866.8 | 94.03 (load 93.80) | llamacpp 99.9 tok/s | 0.512x |
| 2026-10-26-rtx-pro-6000-blackwell-server-edition-qwen3-8-flash-next-q2-143238-r5 | 2026-10-26 | 1 | ptx | 9b55b888 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 50.8 | 867.5 | 94.03 (load 93.80) | llamacpp 99.8 tok/s | 0.509x |

## qwen3.8-flash-next-q2 - decode-curve-large

| run | date | behind | backend | poot commit | GPU | host CPU | poot tok/s | poot TTFT ms | poot GPU GiB at inference | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-26-strix-flashq2-llamacpp-r1 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 32.8 tok/s | - |
| 2026-10-26-strix-flashq2-llamacpp-r2 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 32.8 tok/s | - |
| 2026-10-26-strix-flashq2-llamacpp-r3 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 32.8 tok/s | - |
| 2026-10-26-strix-flashq2-llamacpp-r4 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 32.9 tok/s | - |
| 2026-10-26-strix-flashq2-llamacpp-r5 | 2026-10-26 | 1 | - | - | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | - | - | - | llamacpp 32.8 tok/s | - |
| 2026-10-27-strix-flashq2-rocm-r1 | 2026-10-27 | 0 | rocm | 310148fc | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 19.7 | 1308.3 | 93.67 (load 93.61) | - | - |
| 2026-10-27-strix-flashq2-rocm-r2 | 2026-10-27 | 0 | rocm | 310148fc | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 19.6 | 1312.9 | 93.66 (load 93.60) | - | - |
| 2026-10-27-strix-flashq2-rocm-r3 | 2026-10-27 | 0 | rocm | 310148fc | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 19.6 | 1310.5 | 93.66 (load 93.61) | - | - |
| 2026-10-27-strix-flashq2-rocm-r4 | 2026-10-27 | 0 | rocm | 310148fc | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 19.6 | 1309.1 | 93.66 (load 93.60) | - | - |
| 2026-10-27-strix-flashq2-rocm-r5 | 2026-10-27 | 0 | rocm | 310148fc | AMD Radeon 8060S Graphics | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | 19.7 | 1310.7 | 93.66 (load 93.56) | - | - |

## Known-unverified cells

Committed cells whose repeats do not verify (`benchmarks/known-unverified.toml`). The page shows their measured median muted and labelled unverified, never ranked and never in a ratio; `bench compare-committed` fails when an entry goes stale.

| model | scenario | framework | backend | device | host CPU | driver | pair | band | repeats | follow-up | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | decode-curve | candle | - | NVIDIA RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 595.91.07 | isl=128 tpot_ms | 19.14% | 5 | POOT-1412 | two modes in the five repeats of a 0.6b bf16 decode (about 3 ms per token): the spread, not the repeat count, is the limit |
| qwen3-0.6b | decode-curve | poot | ptx | NVIDIA RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 595.91.07 | isl=2048 tpot_ms | 11.80% | 5 | POOT-1412 | one slow repeat of a 0.6b decode at 2048 tokens (about 3 ms per token); the first sweep of the same binary on this host was unverified at isl=128 ttft_ms instead |
| qwen3-0.6b | decode-curve | transformers | cuda | NVIDIA RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9355 32-Core Processor | 595.91.07 | isl=128 ttft_ms | 15.25% | 5 | POOT-1412 | host-launch jitter on a first-token latency of a few tens of milliseconds |
