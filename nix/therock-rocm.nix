# TheRock ROCm SDK: pre-built nightly from AMD. Provides amdclang (GFX11.7 WMMA-capable), device libs, and
# the HSA runtime. Used alongside nixpkgs ROCm for the runtime (libhsa-runtime64.so.1).
#
# `target` selects the AMDGPU family tarball (spec 120). The default is the dev box (gfx1151, Strix Halo).
# Another family needs both `target` and its matching `hash` overridden (each has its own tarball and
# checksum).
{ lib, stdenvNoCC, fetchurl, patchelf, libdrm, numactl, rdma-core
, target ? "gfx1151"
, hash ? "sha256-hAMl7gvHGJcWIJb0cinJhBoG01YY9ZlpYwBYsMcj6Xo="
}:

let
  version = "7.13.0a20260515";
in
stdenvNoCC.mkDerivation {
  pname = "therock-rocm-sdk-${target}";
  inherit version;

  src = fetchurl {
    url = "https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-${target}-${version}.tar.gz";
    inherit hash;
  };

  # Expose the family so the flake can scope target-specific workarounds (e.g. the gfx1151
  # HSA_OVERRIDE_GFX_VERSION) to the build they apply to.
  passthru = { inherit target; };

  nativeBuildInputs = [ patchelf ];
  propagatedBuildInputs = [ libdrm numactl rdma-core ];

  dontConfigure = true;
  dontBuild = true;
  dontPatchELF = true;
  dontStrip = true;

  unpackPhase = ''
    runHook preUnpack
    tar -xzf "$src"
    runHook postUnpack
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    shopt -s dotglob nullglob
    if [ -d install ]; then
      cp -R install/* "$out/"
    else
      entries=(*/)
      if [ "''${#entries[@]}" -eq 1 ] && [ -d "''${entries[0]}install" ]; then
        cp -R "''${entries[0]}install/"* "$out/"
      else
        cp -R ./* "$out/"
      fi
    fi
    runHook postInstall
  '';

  meta = with lib; {
    description = "TheRock ROCm SDK for ${target} (nightly ${version})";
    homepage = "https://github.com/ROCm/TheRock";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
