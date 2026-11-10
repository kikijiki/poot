# llama.cpp with the HIP (ROCm) backend for the Strix Halo iGPU (gfx1151), at the release the recorded llama.cpp
# Vulkan cells were measured with (v0.6.0, commit d812350), so the poot ROCm <-> llama.cpp ROCm pair differs from
# the Vulkan pair by the backend only. Build: `nix build --impure --file benchmarks/nix/llamacpp-hip.nix`.
{ pkgs ? import (builtins.getFlake "nixpkgs") { system = "x86_64-linux"; } }:
(pkgs.llama-cpp.override {
  rocmSupport = true;
  vulkanSupport = false;
  rocmGpuTargets = [ "gfx1151" ];
}).overrideAttrs (old: rec {
  version = "0.6.0";
  # nixpkgs hard-codes the build number and commit of its own pin (v0.7.0) into llama-bench's output.
  cmakeFlags = builtins.filter (f: builtins.match ".*LLAMA_BUILD_(NUMBER|COMMIT).*" f == null) old.cmakeFlags ++ [
    "-DLLAMA_BUILD_NUMBER:STRING=11429"
    "-DLLAMA_BUILD_COMMIT:STRING=d812350"
  ];
  doCheck = false; # the upstream tests download models
  npmDepsHash = "sha256-a17M+L3nLdRnN6WMB6imPFmwqG2g8uv+gwN0XTAUrf8=";
  src = pkgs.fetchFromGitHub {
    owner = "ggml-org";
    repo = "llama.cpp";
    tag = "v${version}";
    hash = "sha256-l6l6JIlIVTaVC6xh5M4fRHFtXsweQuugtkNTWHcZZF4=";
  };
})
