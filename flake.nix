{
  description = "poot - devshell: pinned Rust nightly (+rustc-dev) + system LLVM 22 (NVPTX+SPIR-V) + Vulkan + ROCm";

  inputs = {
    # Match the host channel (NixOS 26.05) so LLVM/Mesa come from the binary cache, not a source build.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Pinned nightly with rustc internals (single source of truth: ./rust-toolchain.toml). The kernel
        # codegen path (#[kernel] -> LLVM -> SPIR-V/PTX) links rustc_private, so the toolchain is pinned
        # exactly and bumped deliberately. The graph-IR / tracer / executor crates are plain host Rust.
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # System LLVM 22 (NVPTX + SPIR-V targets are built by default in nixpkgs). Only the `llc` binary is
        # used; LLVM is not built here.
        llvm = pkgs.llvmPackages_22;

        # TheRock ROCm SDK (nightly): amdclang with GFX11.7 WMMA support. nixpkgs's rocm-toolchain
        # (rocm-7.2.3, LLVM 22.0.0) cannot select WMMA intrinsics for gfx1151; TheRock's LLVM (22.1+) can.
        # The tarball is the gfx1151 family (nix/therock-rocm.nix `target`); another AMD family needs its own
        # `target` and `hash`. TheRock ships no `llc`: the AMDGPU compile goes through its clang.
        # See https://github.com/ROCm/TheRock, https://github.com/hellas-ai/nix-strix-halo
        therockRocm = pkgs.callPackage ./nix/therock-rocm.nix {};

        # ROCm / HSA runtime for the AMD iGPU path (card 106), default devShell only. ROCm 7.2.x in
        # nixos-26.05 covers gfx1151 (Strix Halo). rocm-runtime ships libhsa-runtime64.so.1 (HSA runtime +
        # KFD ioctl; hsakmt merged there in 2025-03). The compiler and the device-libs bitcode come from
        # TheRock above, not from nixpkgs.
        rocm = pkgs.rocmPackages;

        # Environment shared by every devShell that builds the workspace.
        #
        # Card 395: one Cargo target dir shared across every worktree, so cargo's own file lock serializes
        # concurrent builds. Card 432: an existing CARGO_TARGET_DIR is respected, so a caller can set one to
        # isolate a build. The default is not a `poot-cargo` slot dir (see below), so a plain `nix develop`
        # build is never wiped from under itself.
        #
        # Invariant: a shared target dir is only safe with checksum freshness on. Cargo's unit hash for path
        # packages omits the worktree path, so default mtime freshness can reuse an artifact built from
        # another worktree's different source (observed: E0432 flakiness). That is enabled by the committed
        # .cargo/config.toml [unstable] checksum-freshness = true, not by this shell: config travels with
        # the worktree and covers cargo invocations that never enter it.
        #
        # Optional dev-box accelerators, none of which the committed build depends on:
        #  - kache (https://github.com/kunobi-ninja/kache), a content-addressed rustc output cache. It is
        #    used as the rustc wrapper only when it is on PATH and the caller has not set RUSTC_WRAPPER
        #    (`RUSTC_WRAPPER= nix develop` opts out). `kache doctor` checks the setup.
        #  - poot-cargo, a per-machine slot wrapper that gives each concurrent build its own target dir
        #    (~/.cache/poot-cargo-target{,-2,-3}) and wipes a slot dir when disk runs low. It sets
        #    CARGO_TARGET_DIR itself; it is not in this repository.
        #  - /tmp/poot-gpu.lock, a convention for one device job at a time across cargo invocations:
        #    `flock /tmp/poot-gpu.lock cargo nextest run --test-threads=1 ...`. Nothing in the repository
        #    takes it; .config/nextest.toml serializes device tests within one nextest run only.
        cargoEnv = ''
          export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-$HOME/.cache/poot-shell-target}"
          if [ -z "''${RUSTC_WRAPPER+x}" ] && command -v kache >/dev/null 2>&1; then
            export RUSTC_WRAPPER=kache
          fi
        '';
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain

            llvm.llvm            # llc / opt / llvm-as  (the SPIR-V + NVPTX emitter)
            llvm.clang           # wrapped clang for host C/C++
            llvm.clang-unwrapped # raw clang for cross-target HLSL->SPIR-V (no nix cc-wrapper hardening flags)
            llvm.libclang        # libclang for kernelgen bindgen (rustc_public host bindings)

            pkgs.spirv-tools # spirv-dis / spirv-val / spirv-as

            pkgs.vulkan-loader
            pkgs.vulkan-headers
            pkgs.vulkan-tools             # vulkaninfo
            pkgs.vulkan-validation-layers
            pkgs.mesa                     # RADV/ANV Vulkan ICD, from this flake's own pinned glibc (see shellHook)

            pkgs.pkg-config

            # ROCm / HSA stack for the AMD iGPU path (card 106). clr (HIP runtime) is not linked: raw HSA via
            # libhsa-runtime64.so.1 from rocm-runtime. The AMDGPU compiler (amdclang, with the device
            # libs) is TheRock's; see POOT_AMD_CLANG below.
            rocm.rocm-runtime        # libhsa-runtime64.so.1: HSA runtime + KFD
            rocm.rocminfo            # rocminfo CLI, confirms gfx1151 iGPU visibility

            pkgs.just          # task runner, recipes in ./justfile
            pkgs.jq            # JSON queries in scripts/test-model-free.sh
            pkgs.cargo-nextest # test runner
            pkgs.cargo-machete # unused-dependency check (`just machete`)
            pkgs.cargo-mutants # mutation-sampling recipe (`just mutation-sample`, card 538)

            pkgs.prettier # markdown autoformatter (`just docs-fix`)
            pkgs.lychee   # broken-link checker for the docs (`just docs-check`)

            pkgs.bun    # JS runtime + package manager for the Docusaurus site in ./website (`just site-dev` / `site-build`)
            pkgs.nodejs # Node runtime the Docusaurus toolchain shells out to

            pkgs.python3Packages.huggingface-hub # `hf` CLI, for downloading model weights
          ];

          shellHook = ''
            ${cargoEnv}
            # Put system LLVM's llc ahead of anything the rust toolchain exposes.
            export PATH="${llvm.llvm}/bin:$PATH"
            export LIBCLANG_PATH="${llvm.libclang.lib}/lib"
            export CLANG_HLSL="${llvm.clang-unwrapped}/bin/clang"   # unwrapped clang for HLSL->SPIR-V

            # ---- Real-GPU Vulkan ICD discovery from this flake's own pinned Mesa ----
            # Pin a hardware ICD (avoiding the lavapipe/llvmpipe software fallback). Prefer AMD RADV (this
            # dev box is a Ryzen AI Max+ 395 / Radeon 8060S, Strix Halo); fall back to Intel ANV (the
            # Lunar Lake Arc box) when RADV is absent.
            #
            # Deliberately NOT /run/opengl-driver (the host's NixOS `hardware.graphics` Mesa): that one
            # tracks the host's own nixpkgs channel, independent of this flake's pin. A host rebuild moved
            # it to a newer glibc than this shell's (2.44 vs the 2.42 this flake pins), and the loader
            # (built against this shell's glibc) then failed to load the host driver's libLLVM with a
            # GLIBC_2.44 symbol-version error - "Found no drivers!". Sourcing the ICD json from
            # `pkgs.mesa` instead keeps loader, driver and glibc all from the same pinned closure; each ICD
            # json's `library_path` is an absolute store path, and the driver .so resolves its own deps
            # (libLLVM, libdrm, ...) via its own RUNPATH, so no LD_LIBRARY_PATH entry is needed for it.
            hw_icd="$(ls ${pkgs.mesa}/share/vulkan/icd.d/radeon_icd*.json 2>/dev/null | head -1)"
            [ -z "$hw_icd" ] && hw_icd="$(ls ${pkgs.mesa}/share/vulkan/icd.d/intel_icd*.json 2>/dev/null | head -1)"
            if [ -n "$hw_icd" ]; then
              export VK_ICD_FILENAMES="$hw_icd"
              export VK_DRIVER_FILES="$hw_icd"
            else
              echo "poot: WARN no RADV/ANV Vulkan ICD found under ${pkgs.mesa}/share/vulkan/icd.d"
            fi
            # libvulkan.so.1 itself is usually dlopen'd by name (wgpu/ash), which does not consult a
            # binary's RUNPATH, so the loader still needs to be on LD_LIBRARY_PATH explicitly.
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
            export WGPU_BACKEND=vulkan

            # ---- ROCm / HSA (AMD iGPU, card 106) ----
            # Detected AMD GPU ISA targets, space separated (e.g. " gfx1151"), read from the KFD topology
            # (gfx_target_version is decimal: major*10000 + minor*100 + step). POOT_GPU_TARGETS, when set
            # (even to the empty string), replaces the probe: that is how a box without the hardware
            # exercises the gates below.
            if [ -z "''${POOT_GPU_TARGETS+x}" ]; then
              POOT_GPU_TARGETS=""
              for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
                [ -r "$props" ] || continue
                ver="$(sed -n 's/^gfx_target_version //p' "$props")"
                [ "''${ver:-0}" -gt 0 ] || continue
                POOT_GPU_TARGETS="$POOT_GPU_TARGETS $(printf 'gfx%d%x%x' $((ver / 10000)) $((ver / 100 % 100)) $((ver % 100)))"
              done
            fi

            # Strix Halo workaround: kernel 7.1.1 reported gfx1100 for the Radeon 8060S, and without
            # HSA_OVERRIDE_GFX_VERSION ROCm dispatch failed opaquely (spec 120). Forcing it on other AMD
            # hardware would misdirect dispatch, so it is set only when a gfx1151 agent is detected. The
            # caller can override (''${VAR-default} applies only when VAR is unset, so
            # `HSA_OVERRIDE_GFX_VERSION= nix develop` disables it). Set early so hsa_init sees it.
            case " $POOT_GPU_TARGETS " in
              *" gfx1151 "*) export HSA_OVERRIDE_GFX_VERSION="''${HSA_OVERRIDE_GFX_VERSION-11.5.1}" ;;
            esac

            # ROCm runtime libs on LD_LIBRARY_PATH.
            export LD_LIBRARY_PATH="${rocm.rocm-runtime}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # poot-codegen's Target::AmdGcn invokes $POOT_AMD_CLANG (amdclang) for the final HSACO compile
            # step; it adds the PT_AMDGPU_HSA_LOAD_* program headers that
            # `hsa_executable_load_agent_code_object` requires (llc alone yields a relocatable that fails
            # to load with HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0x1010). See target AmdGcn in
            # crates/poot-codegen/src/lib.rs. This is TheRock's amdclang (GFX11.7 WMMA-capable, and the
            # tarball has no llc, and a bare llc object cannot be loaded anyway).
            export POOT_AMD_CLANG="${therockRocm}/llvm/bin/clang"

            # poot-codegen's Target::AmdGcn passes --rocm-device-lib-path to the clang above (ockl +
            # oclc_isa_version_1151 + oclc_wavefrontsize64 + oclc_finite_only_off + oclc_unsafe_math_off);
            # this points it at TheRock's bitcode dir.
            export POOT_ROCM_DEVICE_LIBS="${therockRocm}/lib/llvm/amdgcn/bitcode"

            # ---- NVIDIA PTX lane: libcuda.so comes from the host driver, not nixpkgs ----
            # cudarc (poot-ptx-runtime) uses the `dynamic-loading` feature: it dlopens libcuda.so.1 at
            # runtime rather than linking it, so the crate builds with no CUDA toolkit in this closure and
            # the lookup happens only when a PTX executor is actually constructed. On an NVIDIA host (e.g. a
            # RunPod pod) that library lives under /run/opengl-driver/lib (NixOS `hardware.nvidia`), the same
            # directory the old Vulkan ICD discovery used to search. Appended, not prepended: it must never
            # come before this shell's own glibc/libdrm/libLLVM (that ordering is exactly what broke Vulkan
            # above), and it is only added when an NVIDIA driver is actually present so an AMD/Intel box
            # never routes any lookup through a host path.
            if [ -e /run/opengl-driver/lib/libcuda.so.1 ] || [ -e /run/opengl-driver/lib/libcuda.so ]; then
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
            fi

            echo "poot devshell"
            echo "  rustc: $(rustc --version 2>/dev/null)"
            echo "  llc:   $(llc --version 2>/dev/null | sed -n 's/.*LLVM version/LLVM/p' | head -1)"
            echo "  llc targets: $(llc --version 2>/dev/null | grep -ioE 'spirv(32|64)?|nvptx(64)?' | sort -u | paste -sd',' -)"
            echo "  amd-clang: $($POOT_AMD_CLANG --version 2>/dev/null | head -1)"
            echo "  therock-rocm: ${therockRocm.version}"
            echo "  gpu targets: ''${POOT_GPU_TARGETS:-<none>}"
            echo "  HSA_OVERRIDE_GFX_VERSION=''${HSA_OVERRIDE_GFX_VERSION:-<unset>}"
            echo "  RUSTC_WRAPPER=''${RUSTC_WRAPPER:-<unset>}"
            echo "  VK_ICD: ''${VK_ICD_FILENAMES:-<none>}"
          '';
        };
      });
}
