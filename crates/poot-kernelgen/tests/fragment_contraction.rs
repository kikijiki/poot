//! POOT-1436: the fragment contraction. The register-tiled GEMM with a `MicroTile::Fragment` runs a dense `[N, K]`
//! (or `[K, N]`) contraction on the matrix unit of a real device (ROCm gfx11 WMMA with a declared lane map, wgpu and
//! raw Vulkan cooperative matrices with an opaque one) and is held to two references:
//!
//! - **The declared oracle** (parity rows). Operands rounded as `OperandPrecision` says (`Narrow`: round to nearest
//!   even to the fragment dtype; `Split`: `hi + lo`), products accumulated exactly (f64). The device differs from it
//!   by accumulation order only, so the bound is an f32 accumulation bound, far below the narrowing error.
//! - **The exact product** (contract rows, design section 6.2). Componentwise
//!   `|y - y_exact| <= c * u * sum |a||w|` with the coefficient `c` the dtype's recorded `OperandNarrowing`, on the bank
//!   shapes (`prefill-dense-bf16-nk`'s `512x3072x1024`, a K 9728 down shape) with a Gaussian and a heavy-tailed
//!   (Student t, 3 degrees of freedom, four x50 outlier channels) activation, bf16 weights N(0, 0.02).
//!
//! The rows skip (reported) without the device. PTX rows run on a pod. Device time is the performance bank's
//! (`*-narrow` cells beside their f32 cells). Mutation: in `crate::precision`'s
//! `round_bf16_magnitude` drop the `0x7fff` rounding bias (truncate); the bf16 heavy contract row and every bf16 parity
//! row go red.

use poot_codegen::{Target, artifact_path, compile, compile_kernel, kernel_handle};
use poot_eval::round_to_fragment;
use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_kernel_ir::Body;
use poot_kernelgen as kg;
use poot_target::{Backend, BufferStorage, DeviceCaps, FragmentLayout};

/// Runs a body over its launch with `[activation, weight, metadata]` inputs and returns the f32 outputs of the
/// given length.
type DispatchFn = Box<dyn Fn(&Body, &kg::Launch, &[f32], &[u32], usize) -> Vec<f32>>;

/// The probed device and what it offers to a fragment tiling.
struct Device {
    name: &'static str,
    caps: DeviceCaps,
    /// The fragment dtypes it multiplies in.
    dtypes: Vec<FragmentDtype>,
    /// Runs `body` over `launch` with `[activation, weight, metadata]` inputs, returning the `out_len` f32 outputs.
    run: DispatchFn,
}

fn work_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("poot-kernelgen-test").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn words_as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn f32_as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn wgpu_device() -> Option<Device> {
    use poot_wgpu_runtime::Context;
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            poot_test_util::skip_report("poot-kernelgen", &format_args!("no wgpu context: {e}"));
            return None;
        }
    };
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let dir = work_dir("fragment_contraction_wgpu");
    Some(Device {
        name: "wgpu (RADV cooperative matrix)",
        dtypes: vec![FragmentDtype::F16],
        caps,
        run: Box::new(move |body, launch, activation, weight, out_len| {
            let out = artifact_path(&dir, &body.name, Target::SpirvVulkan);
            let artifact = compile_kernel(body, Backend::SpirvVulkan, &out)
                .unwrap_or_else(|e| panic!("{}: {e}", body.name));
            let kernel = kernel_handle(body, &artifact).unwrap();
            let mut buffers = [
                poot_wgpu_runtime::KernelBuffer::read_only_f32(activation),
                poot_wgpu_runtime::KernelBuffer::read_only_u32(weight),
                poot_wgpu_runtime::KernelBuffer::read_only_u32(
                    launch.meta.as_deref().expect("metadata"),
                ),
                poot_wgpu_runtime::KernelBuffer::write_f32(out_len),
            ];
            ctx.dispatch(
                &body.name,
                &kernel,
                body.workgroup_size,
                launch.grid,
                &mut buffers,
            )
            .unwrap_or_else(|e| panic!("{}: dispatch: {e}", body.name));
            buffers[3].as_f32().to_vec()
        }),
    })
}

fn vulkan_device() -> Option<Device> {
    use poot_runtime_common::BufferRole;
    use poot_vulkan_runtime::Context;
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            poot_test_util::skip_report("poot-kernelgen", &format_args!("no Vulkan device: {e}"));
            return None;
        }
    };
    let caps = poot_test_util::device_caps::vulkan_rdna3_igpu();
    let dir = work_dir("fragment_contraction_vulkan");
    Some(Device {
        name: "raw Vulkan (cooperative matrix)",
        dtypes: vec![FragmentDtype::F16],
        caps,
        run: Box::new(move |body, launch, activation, weight, out_len| {
            let out = artifact_path(&dir, &body.name, Target::SpirvVulkan);
            let artifact = compile_kernel(body, Backend::SpirvVulkan, &out)
                .unwrap_or_else(|e| panic!("{}: {e}", body.name));
            let kernel = kernel_handle(body, &artifact).unwrap();
            let upload = |bytes: &[u8]| {
                let buf = ctx
                    .alloc_storage(BufferRole::Input, BufferStorage::f32(), bytes.len() / 4)
                    .expect("alloc");
                buf.write_with(0, bytes.len(), &mut |range, out| {
                    out.copy_from_slice(&bytes[range]);
                })
                .expect("write");
                buf
            };
            let a = upload(&f32_as_bytes(activation));
            let w = upload(&words_as_bytes(weight));
            let meta = upload(&words_as_bytes(launch.meta.as_deref().expect("metadata")));
            let result = ctx
                .alloc_storage(BufferRole::Output, BufferStorage::f32(), out_len)
                .expect("alloc");
            ctx.dispatch(
                &kernel,
                body.workgroup_size,
                launch.grid,
                &[&a, &w, &meta, &result],
            )
            .unwrap_or_else(|e| panic!("{}: dispatch: {e}", body.name));
            let mut bytes = vec![0u8; out_len * 4];
            result.read_bytes(&mut bytes).unwrap();
            f32s(&bytes)
        }),
    })
}

fn rocm_device() -> Option<Device> {
    use poot_rocm_runtime::{RocmContext, RocmError};
    use poot_runtime_common::BufferRole;
    use poot_target::ElementKind;
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(_) | RocmError::NoGpuAgent) => {
            poot_test_util::skip_report("poot-kernelgen", &"no ROCm device");
            return None;
        }
        Err(error) => panic!("{error}"),
    };
    let arch = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
        .expect("classify the device arch");
    let caps = poot_test_util::device_caps::default_caps_for(Backend::AmdGcn(arch));
    let dtypes = caps
        .matrix_units
        .known()
        .map(|units| units.iter().map(|config| config.dtype).collect())
        .unwrap_or_default();
    let dir = work_dir("fragment_contraction_rocm");
    Some(Device {
        name: "ROCm (gfx11 WMMA)",
        dtypes,
        caps,
        run: Box::new(move |body, launch, activation, weight, out_len| {
            let target = Target::AmdGcn(arch);
            let out = artifact_path(&dir, &body.name, target);
            let artifact = compile_kernel(body, Backend::AmdGcn(arch), &out)
                .unwrap_or_else(|e| panic!("{}: {e}", body.name));
            let compiled = kernel_handle(body, &artifact).unwrap();
            let module = ctx.load_hsaco(&compiled).unwrap();
            let kernel = ctx.lookup_kernel(&module, compiled.entry_point()).unwrap();
            // Every input travels as 4-byte words; the f32 upload is a lossless bit copy.
            let bits =
                |words: &[u32]| -> Vec<f32> { words.iter().map(|&w| f32::from_bits(w)).collect() };
            let a = ctx.upload_f32(activation, BufferRole::Input).unwrap();
            let w = ctx.upload_f32(&bits(weight), BufferRole::Input).unwrap();
            let meta = ctx
                .upload_f32(
                    &bits(launch.meta.as_deref().expect("metadata")),
                    BufferRole::Input,
                )
                .unwrap();
            let result = ctx.allocate_f32(out_len, BufferRole::Output).unwrap();
            let buffers = [&a, &w, &meta, &result];
            let slots: Vec<_> = buffers
                .iter()
                .map(|buffer| (*buffer, buffer.elem_count() as u64))
                .collect();
            let elements = [
                ElementKind::F32,
                ElementKind::I32,
                ElementKind::I32,
                ElementKind::F32,
            ];
            let kernarg = ctx.bind_kernarg(&kernel, &slots).unwrap();
            let wg = body.workgroup_size[0];
            ctx.dispatch(
                &kernel,
                &kernarg,
                &elements,
                (launch.grid[0], launch.grid[1], launch.grid[2]),
                (wg, 1, 1),
            )
            .unwrap_or_else(|e| panic!("{}: dispatch: {e}", body.name));
            let mut got = vec![0.0f32; out_len];
            ctx.download_f32(&result, &mut got).unwrap();
            got
        }),
    })
}

fn devices() -> Vec<Device> {
    [rocm_device(), wgpu_device(), vulkan_device()]
        .into_iter()
        .flatten()
        .collect()
}

// ---- data ----------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }

    /// Student t with 3 degrees of freedom.
    fn student3(&mut self) -> f64 {
        let chi = self.gauss().powi(2) + self.gauss().powi(2) + self.gauss().powi(2);
        self.gauss() / (chi / 3.0).sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
enum Dist {
    Gauss,
    Heavy,
}

/// `[m, k]` activations: N(0, 1), or Student t(3) with four outlier channels scaled by 50 (a massive-activation
/// shape).
fn activations(dist: Dist, m: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..m * k)
        .map(|i| match dist {
            Dist::Gauss => rng.gauss() as f32,
            Dist::Heavy => {
                let outlier = [3usize, k / 3, k / 2, k - 2].contains(&(i % k));
                let scale = if outlier { 50.0 } else { 1.0 };
                (rng.student3() * scale) as f32
            }
        })
        .collect()
}

/// `[n, k]` weights N(0, 0.02) rounded to bf16 (what a bf16 checkpoint stores).
fn weights(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n * k)
        .map(|_| {
            let bits = poot_quant::scalar::f32_to_bf16((rng.gauss() * 0.02) as f32);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

/// A bf16 weight as the packed words a device buffer holds: element `i` in word `i / 2` at bit `16 * (i % 2)`.
fn bf16_words(values: &[f32]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|pair| {
            let half = |v: f32| u32::from(poot_quant::scalar::f32_to_bf16(v));
            half(pair[0]) | (pair.get(1).map_or(0, |&v| half(v)) << 16)
        })
        .collect()
}

/// `value` as `precision` enters a product: rounded (`Narrow`), or `hi + lo` (`Split`), exactly, in f64.
fn operand(precision: OperandPrecision, value: f32) -> f64 {
    match precision {
        OperandPrecision::Exact => f64::from(value),
        OperandPrecision::Narrow(dtype) => f64::from(round_to_fragment(dtype, value)),
        OperandPrecision::Split(dtype) => {
            let hi = round_to_fragment(dtype, value);
            let lo = round_to_fragment(dtype, value - hi);
            f64::from(hi) + f64::from(lo)
        }
    }
}

// ---- the contraction under test ---------------------------------------------------------------------

/// `[m, k]` against an `[n, k]` weight (`Nk`) or a `[k, n]` one (`Kn`).
#[derive(Clone, Copy, Debug)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightKind {
    /// f32 `[N, K]`.
    NkF32,
    /// bf16 `[N, K]`, packed two per word.
    NkBf16,
    /// f32 `[K, N]`.
    KnF32,
}

impl WeightKind {
    fn tiled(self) -> kg::DenseTiledWeight {
        match self {
            Self::NkF32 => kg::DenseTiledWeight::Nk(BufferStorage::f32()),
            Self::NkBf16 => kg::DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
            Self::KnF32 => kg::DenseTiledWeight::KnF32,
        }
    }
}

/// Generate the fragment contraction of `shape` under `schedule`.
fn generate(
    caps: &DeviceCaps,
    name: &str,
    shape: Shape,
    kind: WeightKind,
    schedule: kg::Schedule,
    precision: OperandPrecision,
) -> Result<(Body, kg::Launch), kg::KernelGenError> {
    let tiling = kg::RegisterTiling::of(schedule).expect("a tiled schedule");
    let (tile_rows, tile_cols) = tiling.output_tile();
    let tiles = shape.m.div_ceil(tile_rows) * shape.n.div_ceil(tile_cols);
    let request = kg::KernelRequest::Contraction(kg::ContractionSpec::DenseTiled(
        kg::DenseTiledContraction {
            weight: kind.tiled(),
            schedule,
            epilogue: None,
            grid: [(tiles * tiling.workgroup_lanes()) as u32, 1, 1],
            meta: vec![
                shape.k as u32,
                shape.n as u32,
                shape.m as u32,
                tiles as u32,
                0,
            ],
            precision,
        },
    ));
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 24).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 20).unwrap(),
    };
    let generated = kg::generate(&request, caps, &limits)?;
    let mut body = generated.body;
    body.name = name.to_string();
    Ok((body, generated.launch))
}

/// The weight in the order and storage `kind` binds it.
fn bound_weight(kind: WeightKind, shape: Shape, w: &[f32]) -> Vec<u32> {
    match kind {
        WeightKind::NkF32 => w.iter().map(|v| v.to_bits()).collect(),
        WeightKind::NkBf16 => bf16_words(w),
        WeightKind::KnF32 => {
            let mut transposed = vec![0u32; w.len()];
            for n in 0..shape.n {
                for k in 0..shape.k {
                    transposed[k * shape.n + n] = w[n * shape.k + k].to_bits();
                }
            }
            transposed
        }
    }
}

/// The fragment schedules a device can run, as `(wave grid, fragments per wave)`: the smallest, a square one, and the
/// large tile the planner picks for a long prefill. Each is offered with the layout and dtype the device reports.
fn schedules(device: &Device, dtype: FragmentDtype, shape: Shape) -> Vec<kg::Schedule> {
    let k = shape.k;
    let units = device
        .caps
        .matrix_units
        .known()
        .expect("known matrix units");
    let config = *units.config(dtype).expect("the device reports this dtype");
    let run = kg::RunLength::dividing(k);
    [
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 1), (1, 2)),
        ((2, 2), (2, 2)),
        ((4, 2), (2, 4)),
    ]
    .into_iter()
    .map(
        |((waves_m, waves_n), (rows, cols))| kg::Schedule::FragmentTiled {
            waves_m: kg::TileSize::new(waves_m).unwrap(),
            waves_n: kg::TileSize::new(waves_n).unwrap(),
            tile: kg::FragmentTile {
                dtype,
                layout: config.layout,
                rows,
                cols,
            },
            run,
        },
    )
    .filter(|schedule| {
        let lanes = kg::RegisterTiling::of(*schedule).unwrap().workgroup_lanes();
        lanes <= device.caps.max_workgroup_invocations as usize
            && waves_of(*schedule) <= config.waves
    })
    .collect()
}

fn waves_of(schedule: kg::Schedule) -> u32 {
    match schedule {
        kg::Schedule::FragmentTiled {
            waves_m, waves_n, ..
        } => waves_m.get() * waves_n.get(),
        other => panic!("not a fragment schedule: {other:?}"),
    }
}

/// The bytes of workgroup memory a body declares.
fn lds_bytes(body: &Body) -> u64 {
    body.workgroup_locals
        .iter()
        .map(|decl| {
            let width = match decl.elem_ty {
                poot_kernel_ir::Ty::F32 => 4,
                poot_kernel_ir::Ty::F16 | poot_kernel_ir::Ty::BF16 => 2,
                ref other => panic!("an LDS array of {other:?}"),
            };
            width * u64::from(decl.len)
        })
        .sum()
}

/// The largest `|got - want| / bound` over the sampled outputs, with `bound` the output's `sum |a||w|` in the
/// operands' own (declared) values. Every sampled output must be finite.
struct Residual {
    worst: f64,
    at: (usize, usize),
}

/// Sample positions of an `m x n` output: both edges (ragged tiles) and a spread of interior points.
fn sample(m: usize, n: usize, count: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Rng(seed);
    let mut points = vec![(0, 0), (m - 1, n - 1), (0, n - 1), (m - 1, 0)];
    points.extend((0..count).map(|_| (rng.next() as usize % m, rng.next() as usize % n)));
    points
}

/// Compare `got` with the dot products of the operands under `reference` over the sampled outputs.
fn residual(
    got: &[f32],
    (a, w): (&[f32], &[f32]),
    shape: Shape,
    points: &[(usize, usize)],
    reference: impl Fn(f32) -> f64,
    weight_reference: impl Fn(f32) -> f64,
) -> Residual {
    let mut worst = Residual {
        worst: 0.0,
        at: (0, 0),
    };
    for &(row, col) in points {
        let (mut sum, mut scale) = (0.0f64, 0.0f64);
        for k in 0..shape.k {
            let x = reference(a[row * shape.k + k]);
            let y = weight_reference(w[col * shape.k + k]);
            sum += x * y;
            scale += (x * y).abs();
        }
        let value = f64::from(got[row * shape.n + col]);
        assert!(value.is_finite(), "({row}, {col}) is not finite");
        let error = (value - sum).abs() / scale.max(f64::MIN_POSITIVE);
        if error > worst.worst {
            worst = Residual {
                worst: error,
                at: (row, col),
            };
        }
    }
    worst
}

/// The accumulation order of a matrix unit differs from the oracle's, by f32 rounding of the partial sums: well
/// under `2^-17` of the output's `sum |a||w|` at these depths, and a thousand times below the narrowing error the
/// contract rows bound.
const ORACLE_BOUND: f64 = 1.0 / 131072.0;

/// Every schedule, precision, weight kind and ragged edge of a small contraction matches the declared oracle.
/// `K = 45` stages in single values with a tail past the last 32-value round; `M = 37`, `N = 41` are ragged in
/// every tile.
#[test]
fn fragment_contractions_match_the_declared_oracle_on_every_device() {
    let shape = Shape {
        m: 37,
        n: 41,
        k: 45,
    };
    let a = activations(Dist::Gauss, shape.m, shape.k, 1);
    let w = weights(shape.n, shape.k, 2);
    let points = sample(shape.m, shape.n, 200, 3);
    let mut failures = Vec::new();
    for device in devices() {
        for &dtype in &device.dtypes {
            for precision in [
                OperandPrecision::Narrow(dtype),
                OperandPrecision::Split(dtype),
            ] {
                for kind in [WeightKind::NkF32, WeightKind::NkBf16, WeightKind::KnF32] {
                    let weight = bound_weight(kind, shape, &w);
                    for (i, schedule) in schedules(&device, dtype, shape).into_iter().enumerate() {
                        let name = format!("frag_parity_{dtype:?}_{i}_{kind:?}");
                        let (body, launch) =
                            generate(&device.caps, &name, shape, kind, schedule, precision)
                                .unwrap_or_else(|e| panic!("{name}: {e}"));
                        // A tiling the device cannot hold in workgroup memory is not one the planner offers.
                        if lds_bytes(&body) > u64::from(device.caps.lds_bytes) {
                            continue;
                        }
                        let got = (device.run)(&body, &launch, &a, &weight, shape.m * shape.n);
                        let r = residual(
                            &got,
                            (&a, &w),
                            shape,
                            &points,
                            |x| operand(precision, x),
                            |x| operand(precision, x),
                        );
                        eprintln!(
                            "{}: {precision} {kind:?} {schedule:?}: oracle residual {:.2e} at {:?}",
                            device.name, r.worst, r.at
                        );
                        if r.worst > ORACLE_BOUND {
                            failures.push(format!(
                                "{}: {name} ({precision}, {schedule:?}) differs from the declared \
                                 oracle by {:.3e} of sum|a||w| at {:?}",
                                device.name, r.worst, r.at
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A fragment schedule needs operands narrowed or split to its dtype, and a device that does not offer a dtype
/// is never handed one: an `Exact` precision, or the other dtype, is a typed refusal at generation.
#[test]
fn a_fragment_schedule_refuses_operands_that_are_not_narrowed_to_its_dtype() {
    let Some(device) = devices().into_iter().next() else {
        return;
    };
    let dtype = device.dtypes[0];
    let other = if dtype == FragmentDtype::F16 {
        FragmentDtype::Bf16
    } else {
        FragmentDtype::F16
    };
    let shape = Shape {
        m: 33,
        n: 33,
        k: 32,
    };
    let schedule = schedules(&device, dtype, shape)[0];
    for precision in [
        OperandPrecision::Exact,
        OperandPrecision::Narrow(other),
        OperandPrecision::Split(other),
    ] {
        let error = generate(
            &device.caps,
            "refused",
            shape,
            WeightKind::NkF32,
            schedule,
            precision,
        )
        .expect_err("a fragment schedule over operands that are not narrowed to its dtype");
        assert!(
            error.to_string().contains("narrowed or split to its dtype"),
            "{precision}: {error}"
        );
    }
}

/// The contract rows (design 6.2): the device result against the exact f64 product, componentwise, on the bank
/// shapes with a Gaussian and a heavy-tailed activation, at most the recorded `OperandNarrowing` bound
/// `c * u * sum |a||w|` for a `Narrow` precision.
#[test]
fn narrow_contractions_stay_within_the_recorded_contract_on_the_bank_shapes() {
    let shapes = [
        (
            "prefill-dense-bf16-nk",
            Shape {
                m: 512,
                n: 3072,
                k: 1024,
            },
        ),
        (
            "down K 9728",
            Shape {
                m: 256,
                n: 2560,
                k: 9728,
            },
        ),
        (
            "bank down K 9728",
            Shape {
                m: 2048,
                n: 2560,
                k: 9728,
            },
        ),
    ];
    for device in devices() {
        for &(label, shape) in &shapes {
            let w = weights(shape.n, shape.k, 7);
            let weight = bound_weight(WeightKind::NkBf16, shape, &w);
            let points = sample(shape.m, shape.n, 96, 5);
            for dist in [Dist::Gauss, Dist::Heavy] {
                let a = activations(dist, shape.m, shape.k, 11);
                for &dtype in &device.dtypes {
                    let precision = OperandPrecision::Narrow(dtype);
                    let tiling = kg::RegisterTiling::choose(
                        kg::TiledShape {
                            blocks: 1,
                            m_per_block: shape.m,
                            out_per_block: shape.n,
                            k: shape.k,
                            layout: kg::WeightLayout::Nk,
                            segments: 1,
                            terms: kg::StagedTerms::of(precision, None, Some(FragmentDtype::Bf16)),
                        },
                        &device.caps,
                    );
                    let schedule = tiling.schedule();
                    assert!(
                        matches!(schedule, kg::Schedule::FragmentTiled { .. }),
                        "{}: the planner takes a fragment tiling for {label}: {schedule:?}",
                        device.name
                    );
                    let name =
                        format!("frag_contract_{dtype:?}_{}", label.replace([' ', '-'], "_"));
                    let (body, launch) = generate(
                        &device.caps,
                        &name,
                        shape,
                        WeightKind::NkBf16,
                        schedule,
                        precision,
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                    let got = (device.run)(&body, &launch, &a, &weight, shape.m * shape.n);
                    let r = residual(&got, (&a, &w), shape, &points, f64::from, f64::from);
                    let oracle = residual(
                        &got,
                        (&a, &w),
                        shape,
                        &points,
                        |x| operand(precision, x),
                        |x| operand(precision, x),
                    );
                    eprintln!(
                        "{}: {label} {dist:?} {dtype:?}: declared-oracle residual {:.2e}",
                        device.name, oracle.worst
                    );
                    assert!(
                        oracle.worst <= ORACLE_BOUND,
                        "{}: {label} {dist:?} {dtype:?}: differs from the declared oracle by {:.3e} at {:?}",
                        device.name,
                        oracle.worst,
                        oracle.at
                    );
                    let contract = poot_target::OperandNarrowing::of(dtype);
                    let unit_roundoff = match dtype {
                        FragmentDtype::Bf16 => 2f64.powi(-8),
                        FragmentDtype::F16 => 2f64.powi(-11),
                    };
                    let bound = f64::from(contract.componentwise_coefficient_milli) / 1000.0
                        * unit_roundoff;
                    eprintln!(
                        "{}: {label} {dist:?} {dtype:?} {schedule:?}: componentwise {:.3e} of u = {:.3} (bound {:.3e})",
                        device.name,
                        r.worst,
                        r.worst / unit_roundoff,
                        bound
                    );
                    assert!(
                        r.worst <= bound,
                        "{}: {label} {dist:?} {dtype:?}: componentwise error {:.3e} at {:?} exceeds the recorded \
                         contract {:.3e} (c = {})",
                        device.name,
                        r.worst,
                        r.at,
                        bound,
                        contract.componentwise_coefficient_milli
                    );
                }
            }
        }
    }
}

/// The layouts a device reports decide the body: a `Declared` accumulator is read in place, an `Opaque` one spills
/// through a scratch array. Both are exercised above; this row names which each device took, so a device whose
/// layout changes shows up here.
#[test]
fn the_devices_report_the_layouts_the_bodies_are_generated_for() {
    for device in devices() {
        let units = device.caps.matrix_units.known().expect("known units");
        for config in units.iter() {
            let want_declared = device.name.starts_with("ROCm");
            assert_eq!(
                matches!(config.layout, FragmentLayout::Declared(_)),
                want_declared,
                "{}: {config:?}",
                device.name
            );
        }
    }
}

/// The SPIR-V of an `Opaque` fragment contraction is valid for the Vulkan versions the runtimes create devices at:
/// the Khronos validation layer (`POOT_VULKAN_VALIDATION`, the raw Vulkan lane) rejects a module that `spirv-val`
/// accepts for another environment, so the rows validate for each. Needs `llc` and `spirv-val` (skips without).
#[test]
fn fragment_contraction_spirv_validates_for_every_vulkan_environment() {
    let have = |tool: &str| {
        std::process::Command::new(tool)
            .arg("--version")
            .output()
            .is_ok()
    };
    if !have("llc") || !have("spirv-val") {
        poot_test_util::skip_report("poot-kernelgen", &"llc or spirv-val not on PATH");
        return;
    }
    let caps = poot_test_util::device_caps::vulkan_rdna3_igpu();
    let device = Device {
        name: "spirv",
        caps,
        dtypes: vec![FragmentDtype::F16],
        run: Box::new(|_, _, _, _, _| unreachable!("compile only")),
    };
    let dir = work_dir("fragment_contraction_spirv");
    let shape = Shape {
        m: 100,
        n: 90,
        k: 64,
    };
    for kind in [WeightKind::NkF32, WeightKind::NkBf16, WeightKind::KnF32] {
        for (i, schedule) in schedules(&device, FragmentDtype::F16, shape)
            .into_iter()
            .enumerate()
        {
            for precision in [
                OperandPrecision::Narrow(FragmentDtype::F16),
                OperandPrecision::Split(FragmentDtype::F16),
            ] {
                let name = format!("spv_{kind:?}_{i}");
                let (body, _) = generate(&device.caps, &name, shape, kind, schedule, precision)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let out = artifact_path(&dir, &name, Target::SpirvVulkan);
                compile(&body, Target::SpirvVulkan, &out)
                    .unwrap_or_else(|e| panic!("{name} {schedule:?}: {e}"));
                for env in ["vulkan1.2", "vulkan1.3"] {
                    let v = std::process::Command::new("spirv-val")
                        .args(["--target-env", env, "--relax-block-layout"])
                        .arg(&out)
                        .output()
                        .unwrap();
                    assert!(
                        v.status.success(),
                        "{name} {precision} {schedule:?} for {env}:\n{}",
                        String::from_utf8_lossy(&v.stderr)
                    );
                }
            }
        }
    }
}
