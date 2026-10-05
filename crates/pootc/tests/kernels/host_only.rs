//! Ordinary host-only crate used to prove that pootc still accepts crates with no kernels.
#![crate_type = "lib"]

pub fn host_add(a: u32, b: u32) -> u32 {
    a + b
}
