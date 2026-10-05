use std::mem::MaybeUninit;
use std::sync::Arc;

use poot_tensor::{HostData, HostTensor};

/// The one layout transform of a checkpoint tensor: a tensor of `shape` whose element `d` is the
/// source's element `order[d]`, in the source's own storage class and dtype (a BF16/F16 tensor
/// stays its 16-bit words; nothing widens). `order` yields exactly `shape`'s element count of
/// source element indices. Every transpose and slice in the loaders is an `order`.
///
/// The destination is written once, straight into its `Arc` (an `Arc<[T]>` collocates its
/// refcount header with the data, so a `Vec` built first and converted would memcpy a second
/// time; for a checkpoint's largest tensor, the tied embedding/lm_head pair, that second copy was
/// a measured host-RSS peak, card 540b SC-001).
pub(crate) fn gather(
    src: &HostTensor,
    shape: Vec<usize>,
    order: impl Iterator<Item = usize>,
) -> HostTensor {
    gather_from(&[src], shape, order.map(|k| (0, k)))
}

/// [`gather`] over several same-dtype sources: element `d` is element `order[d].1` of source
/// `order[d].0`.
pub(crate) fn gather_from(
    srcs: &[&HostTensor],
    shape: Vec<usize>,
    order: impl Iterator<Item = (usize, usize)>,
) -> HostTensor {
    let dtype = srcs[0].dtype();
    assert!(
        srcs.iter().all(|src| src.dtype() == dtype),
        "gather: every source must share one dtype"
    );
    let count: usize = shape.iter().product();
    let data = match srcs[0].data() {
        HostData::F32(_) => {
            HostData::F32(gather_arc(&slices(srcs, HostTensor::as_f32), count, order))
        }
        HostData::I32(_) => {
            HostData::I32(gather_arc(&slices(srcs, HostTensor::as_i32), count, order))
        }
        HostData::Half(_) => {
            HostData::Half(gather_arc(&slices(srcs, HostTensor::as_half), count, order))
        }
        HostData::Bytes(_) => {
            // An element is `byte_size` consecutive bytes, so its byte order is the element
            // order expanded: one generic loop covers every width (E4M3FN, I8, I64, ...).
            let width = dtype.byte_size();
            let byte_order =
                order.flat_map(move |(s, k)| (0..width).map(move |b| (s, k * width + b)));
            let bytes: Vec<&[u8]> = srcs.iter().map(|src| src.view().bytes()).collect();
            HostData::Bytes(gather_arc(&bytes, count * width, byte_order))
        }
    };
    HostTensor::new(dtype, shape, data)
        .expect("a gather writes exactly the destination shape's elements in the source's class")
}

fn slices<'a, T>(
    srcs: &[&'a HostTensor],
    words: impl Fn(&'a HostTensor) -> Option<&'a [T]>,
) -> Vec<&'a [T]> {
    srcs.iter()
        .map(|src| words(src).expect("one dtype means one storage class"))
        .collect()
}

fn gather_arc<T: Copy>(
    srcs: &[&[T]],
    count: usize,
    order: impl Iterator<Item = (usize, usize)>,
) -> Arc<[T]> {
    let mut out: Arc<[MaybeUninit<T>]> = Arc::new_uninit_slice(count);
    let buf = Arc::get_mut(&mut out).expect("freshly allocated Arc has exactly one owner");
    let mut written = 0;
    for (slot, (s, k)) in buf.iter_mut().zip(order) {
        slot.write(srcs[s][k]);
        written += 1;
    }
    assert_eq!(
        written, count,
        "gather: `order` must yield every destination element"
    );
    // SAFETY: `buf` has `count` slots and the loop wrote `written == count` of them (the zip stops
    // at the shorter side, and the assert rules out a short `order`), each exactly once.
    unsafe { out.assume_init() }
}

/// Transpose a 2D `[r, c]` row-major tensor to `[c, r]`, whatever its storage class.
pub(crate) fn transpose2d(rt: &HostTensor) -> HostTensor {
    let (r, c) = (rt.shape()[0], rt.shape()[1]);
    gather(
        rt,
        vec![c, r],
        (0..c).flat_map(move |j| (0..r).map(move |i| i * c + j)),
    )
}

/// Per-expert transpose of a stacked `[E, A, B]` row-major tensor to `[E, B, A]` (the last two
/// axes), for MoE expert weights stored `[out, in]` per expert -> the `[in, out]` matmul layout.
pub(crate) fn transpose_experts(rt: &HostTensor) -> HostTensor {
    let (e, a, b) = (rt.shape()[0], rt.shape()[1], rt.shape()[2]);
    gather(
        rt,
        vec![e, b, a],
        (0..e).flat_map(move |x| {
            (0..b).flat_map(move |j| (0..a).map(move |i| x * a * b + i * b + j))
        }),
    )
}

/// Take rows `[lo, hi)` of a row-major `[rows, cols]` tensor as a new `[hi-lo, cols]` tensor.
/// Used to split fused qkv/gate_up projections (the slices are contiguous in row-major order).
pub(crate) fn row_slice(rt: &HostTensor, lo: usize, hi: usize) -> HostTensor {
    let cols = rt.shape()[1];
    gather(rt, vec![hi - lo, cols], lo * cols..hi * cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_tensor::DType;

    /// `[2, 3]` row-major `0..6`: transposes to `[[0, 3], [1, 4], [2, 5]]`.
    const TRANSPOSED: [usize; 6] = [0, 3, 1, 4, 2, 5];

    #[test]
    fn transpose2d_moves_every_storage_class_without_widening() {
        let f32s = HostTensor::f32(vec![2, 3], (0..6).map(|v| v as f32).collect());
        let t = transpose2d(&f32s);
        assert_eq!((t.dtype(), t.shape()), (DType::F32, &[3, 2][..]));
        let want: Vec<f32> = TRANSPOSED.iter().map(|&v| v as f32).collect();
        assert_eq!(t.as_f32().unwrap(), want);

        let words: Vec<u16> = (0..6).map(|v| 0x4000 + v).collect();
        let want_words: Vec<u16> = TRANSPOSED.iter().map(|&v| 0x4000 + v as u16).collect();
        for tensor in [
            HostTensor::bf16(vec![2, 3], words.clone()),
            HostTensor::f16(vec![2, 3], words.clone()),
        ] {
            let dtype = tensor.dtype();
            let t = transpose2d(&tensor);
            assert_eq!((t.dtype(), t.shape()), (dtype, &[3, 2][..]));
            assert_eq!(
                t.as_half().unwrap(),
                want_words,
                "{dtype} words move, not widen"
            );
        }

        let ints = HostTensor::i32(vec![2, 3], (0..6).collect());
        assert_eq!(
            transpose2d(&ints).as_i32().unwrap(),
            TRANSPOSED.map(|v| v as i32)
        );
    }

    #[test]
    fn transpose2d_moves_byte_class_tensors_of_every_element_width() {
        // E4M3FN and I8 are 1-byte elements, I64 is 8: the byte expansion keeps each element whole.
        for (dtype, width) in [(DType::E4M3FN, 1), (DType::I8, 1), (DType::I64, 8)] {
            let bytes: Vec<u8> = (0..6 * width).map(|b| b as u8).collect();
            let tensor = HostTensor::from_le_bytes(dtype, vec![2, 3], &bytes).unwrap();
            let t = transpose2d(&tensor);
            assert_eq!((t.dtype(), t.shape()), (dtype, &[3, 2][..]));
            let want: Vec<u8> = TRANSPOSED
                .iter()
                .flat_map(|&e| bytes[e * width..(e + 1) * width].to_vec())
                .collect();
            assert_eq!(t.view().bytes(), want, "{dtype}");
        }
    }

    #[test]
    fn transpose_experts_transposes_each_expert_in_its_own_dtype() {
        // [2, 2, 3]: two experts, each the `0..6` matrix offset by 6e.
        let words: Vec<u16> = (0..12).collect();
        let t = transpose_experts(&HostTensor::bf16(vec![2, 2, 3], words));
        assert_eq!((t.dtype(), t.shape()), (DType::BF16, &[2, 3, 2][..]));
        let want: Vec<u16> = (0..2u16)
            .flat_map(|e| TRANSPOSED.iter().map(move |&v| 6 * e + v as u16))
            .collect();
        assert_eq!(t.as_half().unwrap(), want);
    }

    #[test]
    fn row_slice_takes_contiguous_rows_in_the_source_dtype() {
        let words: Vec<u16> = (0..12).collect();
        let t = row_slice(&HostTensor::f16(vec![4, 3], words), 1, 3);
        assert_eq!((t.dtype(), t.shape()), (DType::F16, &[2, 3][..]));
        assert_eq!(t.as_half().unwrap(), (3..9).collect::<Vec<u16>>());
    }

    #[test]
    fn gather_from_reads_across_sources_and_rejects_mixed_dtypes() {
        let a = HostTensor::bf16(vec![2], vec![10, 11]);
        let b = HostTensor::bf16(vec![2], vec![20, 21]);
        let t = gather_from(
            &[&a, &b],
            vec![4],
            [(1, 1), (0, 0), (1, 0), (0, 1)].into_iter(),
        );
        assert_eq!(t.as_half().unwrap(), [21, 10, 20, 11]);

        let mixed = HostTensor::f16(vec![2], vec![0, 0]);
        let refused = std::panic::catch_unwind(|| {
            gather_from(&[&a, &mixed], vec![1], std::iter::once((0, 0)))
        });
        assert!(refused.is_err(), "one gather never mixes dtypes");
    }
}
