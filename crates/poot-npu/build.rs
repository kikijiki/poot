//! Compiles the XRT C++ shim and links `libxrt_coreutil` when the `xdna` feature is on. The prefix is
//! `$XILINX_XRT` and the `uuid/uuid.h` that XRT's `xclbin.h` includes is in `$POOT_LIBUUID_INCLUDE` (both
//! exported by `nix develop .#npu`); without the feature this build script does nothing.

fn main() {
    #[cfg(feature = "xdna")]
    link_xrt();
}

#[cfg(feature = "xdna")]
fn link_xrt() {
    let xrt = std::env::var("XILINX_XRT").expect(
        "XILINX_XRT is not set: build poot-npu with the `xdna` feature inside `nix develop .#npu`",
    );
    let uuid = std::env::var("POOT_LIBUUID_INCLUDE").expect(
        "POOT_LIBUUID_INCLUDE is not set: build poot-npu with the `xdna` feature inside `nix develop .#npu`",
    );
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(format!("{xrt}/include"))
        .include(uuid)
        .file("src/xrt_shim.cpp")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        // XRT's own headers use deprecated constructs.
        .flag_if_supported("-Wno-deprecated-declarations")
        // The toolchain wrapper hardens with `_FORTIFY_SOURCE`, which glibc's headers warn about below -O1;
        // this is FFI glue, so it is built optimized in every profile.
        .opt_level(1)
        .compile("poot_xrt_shim");
    println!("cargo:rustc-link-search=native={xrt}/lib");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{xrt}/lib");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-env-changed=XILINX_XRT");
    println!("cargo:rerun-if-env-changed=POOT_LIBUUID_INCLUDE");
    println!("cargo:rerun-if-changed=src/xrt_shim.cpp");
}
