//! A plain host call of every declared intrinsic. `pootc` never imports these bodies (it recognizes the
//! *call* and replaces it with a device terminator/statement), so this only proves the crate's API surface
//! links and type-checks as kernel sources use it; it is not a kernel-lowering test (see `crates/pootc/tests`
//! for that).

#[test]
fn every_intrinsic_is_callable() {
    assert_eq!(poot_kernel_intrinsics::thread_index(), 0);
    assert_eq!(poot_kernel_intrinsics::local_index(), 0);
    assert_eq!(poot_kernel_intrinsics::group_index(), 0);
    assert_eq!(poot_kernel_intrinsics::group_index_y(), 0);
    poot_kernel_intrinsics::workgroup_barrier();
    poot_kernel_intrinsics::wg_write(0, 0, 1.0);
    assert_eq!(poot_kernel_intrinsics::wg_read(0, 0), 0.0);
    let mut buf = [0u32];
    assert_eq!(poot_kernel_intrinsics::atomic_add(&mut buf, 0, 1), 0);
}
