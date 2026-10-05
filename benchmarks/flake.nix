{
  description = "poot benchmark image - build/push tooling (podman + skopeo) for the pre-baked GHCR bench image";

  # Separate from the root poot devshell: carries only the tools to build/push the benchmark Docker image
  # and drive the harness. Nix does not build the image (vLLM/torch/candle are not packaged at our pins);
  # podman/skopeo build and push docker/Dockerfile.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.podman   # daemonless, rootless OCI build of docker/Dockerfile
            pkgs.buildah  # podman's builder backend
            pkgs.skopeo   # push / copy / inspect images without a daemon
            pkgs.dive     # inspect image layers + size

            pkgs.just     # bench task runner (./justfile)
            pkgs.gh       # manage the GHCR package
            pkgs.uv       # run the harness locally (`just harness ...`)
            pkgs.jq
            pkgs.awscli2  # upload model weights to a RunPod network volume via its S3 API (just volume-sync)

            pkgs.python3Packages.huggingface-hub # `hf` CLI - fetch eval models onto a pod
          ];

          shellHook = ''
            echo "poot-bench devshell (image build/push tooling)"
            echo "  podman: $(podman --version 2>/dev/null)"
            echo "  skopeo: $(skopeo --version 2>/dev/null | head -1)"
            echo "  recipes: just --list   (build: just image-build | build+smoke+push: just image-build-push)"
            echo "  image:   ghcr.io/\''${GHCR_OWNER:-kikijiki}/poot-bench"
          '';
        };
      });
}
