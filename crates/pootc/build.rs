//! Embed the toolchain's `lib/` as an rpath so the driver finds `librustc_driver-*.so` (and the std
//! dylib) at runtime. A custom rustc driver links `librustc_driver` dynamically and cannot start without it.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("run rustc --print sysroot");
    let sysroot = String::from_utf8(out.stdout).expect("sysroot not utf-8");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", sysroot.trim());
    println!("cargo:rerun-if-changed=build.rs");
}
