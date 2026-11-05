# wgpu-hal 29.0.4, patched for POOT-1464

`0001-pass-begin-timestamp-at-top-of-pipe.patch` is one hunk against upstream `wgpu-hal` 29.0.4, in
`src/vulkan/command.rs` (`begin_compute_pass`): the `beginning_of_pass_write_index` timestamp is written at
`TOP_OF_PIPE` instead of `BOTTOM_OF_PIPE`.

Why: a `BOTTOM_OF_PIPE` stamp is an event that later commands do not wait for. On RADV (Strix Halo, Mesa 26.1.3)
it can complete after the pass's dispatch has already run, so the pass reads a few ticks and its time is booked to
the next pass. Measured with three probe processes sharing the GPU: 0.8% to 9% of steps affected before, 0 of
about 81000 after (POOT-1464). The end-of-pass stamp stays `BOTTOM_OF_PIPE`
(it must wait for the pass's work). Only the Vulkan backend is changed; poot's wgpu device is Vulkan only.

## How the patched crate gets built

The repository holds the patch, not the crate. `scripts/wgpu-hal-patched.sh` produces `.patched/wgpu-hal`
(gitignored), and the workspace `Cargo.toml` points `[patch.crates-io] wgpu-hal` at it. The script:

1. takes the pinned crates.io archive (`wgpu-hal-29.0.4.crate`, from the cargo registry cache or
   `static.crates.io`) and refuses it unless its SHA-256 equals the pin in the script;
2. unpacks it and applies the patch with `git apply`, which has no fuzz: every context line must match
   exactly (a hunk may move to a different line number) or it fails;
3. writes a stamp (crate checksum + patch checksum) and is a no-op while the stamp matches.

The devshell runs it on entry (`flake.nix` shellHook), so `nix develop -c cargo ...` (and `poot-cargo`, the
orchestrator, `just`) always have the directory. `scripts/ptx-pod-lane` runs it for the pod build. A build that
skips it fails in cargo with `failed to read .patched/wgpu-hal/Cargo.toml`; it never silently builds the
unpatched crate. `scripts/wgpu-hal-patched.sh check` (part of `just test`) asserts the directory is current, the
pin matches `Cargo.lock`'s `wgpu-hal` version, and cargo resolves `wgpu-hal` to the patched path.

## Removing it

Delete `patches/wgpu-hal`, `scripts/wgpu-hal-patched.sh`, the `[patch.crates-io]` entry in `Cargo.toml`, the
`.gitignore` line, the shellHook and `ptx-pod-lane` calls and the `just` check once an upstream `wgpu-hal` writes
pass-begin stamps at `TOP_OF_PIPE` (card 1480 prepares the upstream PR). To move to a new upstream version,
update the pin in the script and regenerate the patch if its context moved.
