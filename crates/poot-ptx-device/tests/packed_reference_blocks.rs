//! POOT-1337: the PTX decode GEMV decodes the real Flash-Next blocks to llama.cpp's values, bit for bit
//! ([`poot_executor_parity::reference_blocks`]). Mutation: swap two entries of the IQ2_XS grid in
//! `poot_quant::grids`; the `reference_iq2_xs` rows go red with the tensor, byte offset, block, column, the
//! device value and gguf-py's.

use poot_executor::{Device, Engine};
use poot_ptx_device::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

#[test]
fn ptx_decode_gemv_matches_llama_cpp_dequantisation_on_real_blocks() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec = Engine::new(device);
    poot_executor_parity::reference_blocks::assert_reference_blocks(&mut exec, target);
}
