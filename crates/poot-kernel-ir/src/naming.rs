//! The kernel naming convention shared by the `#[kernel]` macro (producer) and the `pootc` driver
//! (consumer). A proc-macro attribute is expanded away before MIR, so the macro renames the device fn into
//! a reserved namespace and the driver discovers kernels by scanning monomorphized item names for the
//! prefix. Dependency-free.

/// Reserved prefix the `#[kernel]` macro mangles a device fn into, so it survives to MIR for discovery.
pub const KERNEL_PREFIX: &str = "__poot_kernel_";

/// The mangled symbol for a kernel named `name`.
pub fn mangle(name: &str) -> String {
    format!("{KERNEL_PREFIX}{name}")
}

/// Is this (mangled) symbol a poot kernel?
pub fn is_kernel(symbol: &str) -> bool {
    last_segment(symbol).starts_with(KERNEL_PREFIX)
}

/// Recover the source kernel name from a (possibly path-qualified) mangled symbol, or `None` if it is
/// not a kernel symbol. `foo::bar::__poot_kernel_add` -> `Some("add")`.
pub fn source_name_of_path(symbol: &str) -> Option<String> {
    let seg = last_segment(symbol);
    seg.strip_prefix(KERNEL_PREFIX).map(|s| s.to_string())
}

fn last_segment(symbol: &str) -> &str {
    symbol.rsplit("::").next().unwrap_or(symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let m = mangle("add");
        assert_eq!(m, "__poot_kernel_add");
        assert!(is_kernel(&m));
        assert_eq!(source_name_of_path(&m).as_deref(), Some("add"));
    }

    #[test]
    fn path_qualified() {
        let sym = "my_crate::kernels::__poot_kernel_gemv";
        assert!(is_kernel(sym));
        assert_eq!(source_name_of_path(sym).as_deref(), Some("gemv"));
    }

    #[test]
    fn non_kernel() {
        assert!(!is_kernel("my_crate::main"));
        assert_eq!(source_name_of_path("std::vec::Vec"), None);
    }
}
