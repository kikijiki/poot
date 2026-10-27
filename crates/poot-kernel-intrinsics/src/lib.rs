//! The kernel-authoring intrinsics: the nine dispatch-axis indices, the workgroup barrier, the launch declarations
//! (the workgroup shape and the workgroup-local array types `#[kernel(...)]` records), workgroup-local (LDS)
//! array access, the integer atomic add (spec 054, card 148 phase 2b), and the no-contraction marked add
//! `no_contract_add` (card 675, ADR 0114 tier 1).
//!
//! A `#[kernel]`-authored kernel source depends on this crate (`--extern poot_kernel_intrinsics=...`) and
//! calls these functions directly; it never redeclares its own stubs. `pootc`'s importer
//! ([`import_body`](https://docs.rs/pootc) - see `crates/pootc/src/import.rs`) recognizes a call by its
//! *resolved declaration*: the callee's full path must be `poot_kernel_intrinsics::<name>`, not merely end
//! in a matching bare name. A kernel-authored helper that happens to share a name with one of these (for
//! example a local `fn workgroup_barrier()`) is therefore never mistaken for the intrinsic (R468-009).
//!
//! On a normal host build these bodies are placeholders: `pootc` never imports them (it recognizes the
//! *call*, replaces it with the device terminator or statement the call lowers to, and never looks at the
//! callee's MIR body), so the values below only matter for a plain `cargo check`/`cargo test` of this
//! crate or of a kernel source built outside `pootc`.

/// One dispatch-axis intrinsic per name: a placeholder `0` on the host, replaced by `pootc` with the device's index
/// register. A kernel source calls these by name, so the nine are declared once, here, from one list.
macro_rules! axis_intrinsics {
    ($($(#[$doc:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$doc])*
            #[inline(never)]
            pub fn $name() -> usize {
                0
            }
        )+
    };
}

axis_intrinsics! {
    /// The global dispatch index along X (the thread's position in the whole grid).
    thread_index_x,
    /// The global dispatch index along Y.
    thread_index_y,
    /// The global dispatch index along Z.
    thread_index_z,
    /// The lane's position within its workgroup along X (`0..workgroup_size[0]`), used to address per-lane
    /// workgroup-local (LDS) slots (spec 054).
    local_index_x,
    /// The lane's position within its workgroup along Y.
    local_index_y,
    /// The lane's position within its workgroup along Z.
    local_index_z,
    /// The workgroup's index along X (the row a per-row workgroup reduction owns).
    group_index_x,
    /// The workgroup's index along Y.
    group_index_y,
    /// The workgroup's index along Z.
    group_index_z,
}

/// A workgroup-scope control and memory barrier: every lane in the workgroup waits here, and every
/// workgroup-local (LDS) write issued before the barrier is visible to every lane after it.
#[inline(never)]
pub fn workgroup_barrier() {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

/// An element type a workgroup-local (LDS) array holds: `f32`, `u32` or `i32`.
pub trait LdsElement: sealed::Sealed + Copy + Default {}
impl LdsElement for f32 {}
impl LdsElement for u32 {}
impl LdsElement for i32 {}

/// Declare the kernel's workgroup shape in lanes along `x`, `y` and `z`. A kernel that reads a lane or workgroup
/// index, calls the barrier or touches an LDS array must declare it. `#[kernel(workgroup(x, y, z))]` expands to
/// this call at the head of the kernel; `pootc` reads the three constants and drops the call.
#[inline(never)]
pub fn declare_workgroup(_x: usize, _y: usize, _z: usize) {}

/// Declare the next workgroup-local (LDS) array: `len` elements of type `T`. The first declaration is array 0, the
/// second array 1, and so on, in call order. `#[kernel(lds(T, len))]` expands to this call at the head of the
/// kernel; `pootc` reads `T` and `len` and drops the call.
#[inline(never)]
pub fn declare_lds<T: LdsElement>(_len: usize) {}

/// Write `val` to workgroup-local (LDS) array `array` (a declared array id, a constant) at `idx`. `T` is the
/// array's declared element type.
#[inline(never)]
pub fn wg_write<T: LdsElement>(_array: usize, _idx: usize, _val: T) {}

/// Read workgroup-local (LDS) array `array` (a declared array id, a constant) at `idx`. `T` is the array's
/// declared element type.
#[inline(never)]
pub fn wg_read<T: LdsElement>(_array: usize, _idx: usize) -> T {
    T::default()
}

/// Atomically add `value` to `buffer[index]`, returning the old value: an order-independent exact integer
/// reduction (float atomic add is order-dependent, so this is `u32`-only).
#[inline(never)]
pub fn atomic_add(_buffer: &mut [u32], _index: usize, _value: u32) -> u32 {
    0
}

/// `a + b` as two separate roundings that codegen must never contract into a fused multiply-add: `pootc`
/// lowers this call to `poot_kernel_ir::Rvalue::BinaryOpNoContract(BinOp::Add, a, b)` (card 675), the same
/// per-op marker `poot-kernelgen`'s Body API already uses for the packed-dequant decode formula (card 628).
/// Write the preceding multiply as its own statement (`let prod = x * y;`) and pass it as `b`: a tier-1
/// (bit-exact) sampling primitive whose source has two roundings (e.g. the Gumbel select's
/// `v*inv_temp + noise_scale*noise`, ADR 0114) must keep both on every backend, never let one backend's
/// compiler fuse the trailing multiply into this add.
#[inline(never)]
pub fn no_contract_add(a: f32, b: f32) -> f32 {
    a + b
}
