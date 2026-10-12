//! POOT-1139: the strided-read ceiling on ROCm. A read of one 256 B granule per row at a row stride that
//! reaches `c` of the APU's sixteen DRAM channels reads faster the more channels it reaches. The probe and its
//! order check are `poot_executor_parity::strided_read`; this file runs them on the device. Only the order is
//! asserted (a rate ratio floor per step); the fit to the gcd model is a printed report.
//!
//! Recorded on the Strix Halo iGPU (gfx1151), 2026-10-12, 131072 rows (32 MiB read per step), GB/s of granule
//! bytes per second of the read dispatch's device time; median over five holds of nine-step medians:
//!
//! ```text
//! stride B  channels  median GB/s  (range of the five medians)
//!    4096         1        13.4   12.9..14.0
//!    8192         1        13.4   13.3..13.5   (control)
//!   12288         1        12.9   11.3..13.5   (a multiple of 4096 B)
//!    2048         2        25.1   24.2..26.0   (not a multiple of 4096 B)
//!    6144         2        23.7   21.8..24.8   (not a multiple of 4096 B)
//!   10240         2        24.3   23.5..24.8
//!    1024         4        45.4   44.2..46.1   (not a multiple of 4096 B)
//!    3072         4        38.9   38.6..41.7   (not a multiple of 4096 B)
//!    5120         4        35.0   34.8..36.4
//!    2560         8        48.9   48.7..49.0
//!   10752         8        38.3   38.1..39.4
//!   10496        16        41.6   41.6..42.1
//!     256        16        64.0   63.9..64.1   (contiguous)
//! ```
//!
//! Reading the table: rates follow the channels a stride reaches, not its size or its alignment to 4096 B. The
//! 12288 B stride (a multiple of 4096 B, one channel) matches 4096 B and 8192 B; 6144 B and 2048 B (not
//! multiples of 4096 B, two channels) match 10240 B; 1024 B and 3072 B (four channels) read 2.9x to 3.4x the
//! one-channel rate. Against these medians the gcd model (one-channel rate times channels, capped at the
//! unaliased rate) is off by at most 19%, a two-constant "multiple of 4096 B" model by up to 62% and a
//! two-constant page model (rows share a 4 KiB page below 4096 B) by up to 86%. The unaliased stride is not the
//! contiguous ceiling: it reads about 65% of it. The research note reports HIP does not pay this tax; on this
//! box it does.

use poot_executor::{Device, Engine};
use poot_executor_parity::strided_read::{check_order, measure_strides, report};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

/// Row count of the probe: 131072 rows x 256 B = 32 MiB read per step.
const ROWS: usize = 131072;
/// The strides: the one-channel stride and its control (twice the stride, the same alias factor), then 2, 4
/// and 8 channels, then the unaliased stride.
const STRIDES: [usize; 6] = [4096, 8192, 10240, 5120, 10752, 10496];
/// Reported only, to tell channel aliasing from a page effect: strides that are not multiples of 4096 B and
/// still alias (2048 B and 2560 B reach 2 and 8 channels, 1024 B and 3072 B 4 channels, 6144 B 2) and
/// 12288 B, a multiple of 4096 B with 1 channel.
const EXTRA_STRIDES: [usize; 6] = [1024, 2048, 2560, 3072, 6144, 12288];
/// Rows 256 B apart: the same bytes read as one contiguous run. Reported, not asserted (a contiguous read may
/// use a wider path than the strided kernel).
const CONTIGUOUS: usize = 256;
/// Steps timed per stride; the spread in the table above comes from this many.
const RUNS: usize = 9;

/// Asserts the order only: 2 channels read at least 1.3x the rate of 1, 4 at least 1.2x of 2, and the
/// unaliased stride at least 2.0x of 1 (floors below the smallest ratio of five holds); the fitted table is
/// printed, not asserted. Mutation (measured side): have `measure_strides` allocate every slot at one granule
/// per row; every stride then reads at one rate and this goes red on the 2-over-1 ratio. Mutation (model
/// side): have `alias_factor` use a period of eight channels; no stride reaches one channel and the lookup
/// panics.
#[test]
fn read_rate_rises_with_the_dram_channels_a_stride_reaches() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new_with_device_timing())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let rates = measure_strides(&mut engine, target, ROWS, &STRIDES, RUNS);
    let contiguous = measure_strides(&mut engine, target, ROWS, &[CONTIGUOUS], RUNS);
    let extra = measure_strides(&mut engine, target, ROWS, &EXTRA_STRIDES, RUNS);
    eprintln!("{}", report(&[rates.clone(), extra, contiguous].concat()));
    if let Err(broken) = check_order(&rates) {
        panic!("a stride that reaches more channels does not read faster:\n{broken}");
    }
}
