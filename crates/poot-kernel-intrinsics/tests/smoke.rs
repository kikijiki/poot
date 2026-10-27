//! A plain host call of every declared intrinsic. `pootc` never imports these bodies (it recognizes the
//! *call* and replaces it with a device terminator/statement), so this only proves the crate's API surface
//! links and type-checks as kernel sources use it; it is not a kernel-lowering test (see `crates/pootc/tests`
//! for that).

#[test]
fn every_intrinsic_is_callable() {
    use poot_kernel_intrinsics as k;
    for axis in [
        k::thread_index_x(),
        k::thread_index_y(),
        k::thread_index_z(),
        k::local_index_x(),
        k::local_index_y(),
        k::local_index_z(),
        k::group_index_x(),
        k::group_index_y(),
        k::group_index_z(),
    ] {
        assert_eq!(axis, 0);
    }
    k::workgroup_barrier();
    k::declare_workgroup(64, 1, 1);
    k::declare_lds::<f32>(64);
    k::declare_lds::<u32>(64);
    k::wg_write(0, 0, 1.0f32);
    assert_eq!(k::wg_read::<f32>(0, 0), 0.0);
    assert_eq!(k::wg_read::<u32>(1, 0), 0);
    let mut buf = [0u32];
    assert_eq!(k::atomic_add(&mut buf, 0, 1), 0);
}
