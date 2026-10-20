# poot benchmark index

Every committed run, grouped by model and scenario. The point is to watch poot's column move as we optimize. Regenerate with `bench index`. Full tables: `bench report results/<run>`; to test a change, `bench compare --baseline <runs> --candidate <runs>` (5 repeats per side). `behind` is how many days a run is older than the newest run here.

## qwen2.5-0.5b - decode-128

| run | date | behind | backend | poot commit | GPU | poot tok/s | poot TTFT ms | poot VRAM GiB | fastest baseline | poot vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-rocm-195531-r1 | 2026-10-19 | 0 | rocm | 2f639c88 | AMD Radeon 8060S Graphics | 158.2 | 9.6 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-rocm-195531-r2 | 2026-10-19 | 0 | rocm | 2f639c88 | AMD Radeon 8060S Graphics | 163.6 | 9.4 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-rocm-195531-r3 | 2026-10-19 | 0 | rocm | 2f639c88 | AMD Radeon 8060S Graphics | 158.1 | 9.5 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-rocm-195531-r4 | 2026-10-19 | 0 | rocm | 2f639c88 | AMD Radeon 8060S Graphics | 163.3 | 9.5 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-rocm-195531-r5 | 2026-10-19 | 0 | rocm | 2f639c88 | AMD Radeon 8060S Graphics | 163.9 | 9.4 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-wgpu-193702-r1 | 2026-10-19 | 0 | wgpu | 2f639c88 | AMD Radeon 8060S Graphics | 171.8 | 10.0 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-wgpu-193702-r2 | 2026-10-19 | 0 | wgpu | 2f639c88 | AMD Radeon 8060S Graphics | 171.3 | 9.8 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-wgpu-193702-r3 | 2026-10-19 | 0 | wgpu | 2f639c88 | AMD Radeon 8060S Graphics | 172.8 | 9.9 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-wgpu-193702-r4 | 2026-10-19 | 0 | wgpu | 2f639c88 | AMD Radeon 8060S Graphics | 170.7 | 9.8 | - | - | - |
| 2026-10-19-amd-radeon-8060s-graphics-qwen2-5-0-5b-wgpu-193702-r5 | 2026-10-19 | 0 | wgpu | 2f639c88 | AMD Radeon 8060S Graphics | 172.6 | 9.8 | - | - | - |
