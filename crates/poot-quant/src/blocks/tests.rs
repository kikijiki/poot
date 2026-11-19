use super::lattice::{
    IQ1S_LATTICE, IQ2S_LATTICE, IQ2XS_LATTICE, IQ2XXS_LATTICE, IQ3S_LATTICE, IQ3XXS_LATTICE,
};
use crate::families::lattice::{Lattice, SignRule};
use crate::format::{Field, FieldEncoding, OperandRole, Storage, ValueMap, WeightFormat};
use crate::grids::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};
use crate::test_support::formats::GGUF_FORMATS;

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn assert_block(format: WeightFormat, block: &[u8], expected: &[(usize, f32)]) {
    let descriptor = format.descriptor();
    for &(index, want) in expected {
        let got = descriptor.decode_block_value(block, index).unwrap();
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{format:?} element {index}: {got:?} vs reference {want:?}"
        );
    }
}

/// One literal block per GGUF format with values from a transcription of its
/// `dequantize_row_*` in `ggml/src/ggml-quants.c` (llama.cpp 03a667aa3), rounding to f32 after
/// every operation as the C code does. MXFP4 code 8 is `-0.0` where ggml's `int8_t` table
/// gives `+0.0` (the one deliberate difference; see [`crate::format`]).
macro_rules! reference {
    ($test:ident, $format:ident, $block:expr, [$(($index:literal, $value:expr)),+ $(,)?]) => {
        #[test]
        fn $test() {
            assert_block(WeightFormat::$format, &hex($block), &[$(($index, $value)),+]);
        }
    };
}

reference!(
    q4_0_block_decodes_to_the_ggml_reference,
    Q4_0,
    "042397621649bd7821226d9d3dd30a06d54a",
    [
        (0, -0.013702393),
        (3, 0.013702393),
        (6, -0.09591675),
        (9, 0.06851196),
        (12, 0.027404785),
        (15, 0.027404785),
        (18, -0.09591675),
        (21, -0.013702393),
        (24, -0.027404785),
        (27, 0.06851196),
        (30, 0.06851196),
        (31, -0.05480957),
    ]
);

reference!(
    q4_1_block_decodes_to_the_ggml_reference,
    Q4_1,
    "0423b8b21d145901fadc73a8ffbf47d8a2a2457b",
    [
        (0, -0.031829834),
        (3, -0.19625854),
        (6, -0.16885376),
        (9, -0.004425049),
        (12, -0.18255615),
        (15, -0.05923462),
        (18, -0.14144897),
        (21, -0.031829834),
        (24, -0.004425049),
        (27, -0.031829834),
        (30, -0.15515137),
        (31, -0.11404419),
    ]
);

reference!(
    q5_0_block_decodes_to_the_ggml_reference,
    Q5_0,
    "0423ee8e46327f2c16fa536988e9d93c837043d73aeb",
    [
        (0, -0.013702393),
        (3, 0.13702393),
        (6, 0.10961914),
        (9, 0.16442871),
        (12, -0.1781311),
        (15, 0.15072632),
        (18, 0.013702393),
        (21, -0.13702393),
        (24, -0.041107178),
        (27, -0.12332153),
        (30, -0.1781311),
        (31, -0.027404785),
    ]
);

reference!(
    q5_1_block_decodes_to_the_ggml_reference,
    Q5_1,
    "0423b8b2490a0e50da8a62ee274c95b68d565f37472ea84e",
    [
        (0, 0.14630127),
        (3, 0.20111084),
        (6, 0.07778931),
        (9, 0.0914917),
        (12, -0.11404419),
        (15, -0.018127441),
        (18, 0.0914917),
        (21, -0.15515137),
        (24, -0.1003418),
        (27, -0.16885376),
        (30, 0.14630127),
        (31, -0.15515137),
    ]
);

reference!(
    q8_0_block_decodes_to_the_ggml_reference,
    Q8_0,
    concat!(
        "04230f9e167c0081016eefe8f408aad58096763bee663bf085c2c81e77320764",
        "6c57",
    ),
    [
        (0, 0.20553589),
        (3, 1.6990967),
        (6, 0.013702393),
        (9, -0.32885742),
        (12, -1.1784058),
        (15, -1.4524536),
        (18, -0.24664307),
        (21, -0.21923828),
        (24, -0.767334),
        (27, 0.6851196),
        (30, 1.4798584),
        (31, 1.1921082),
    ]
);

reference!(
    q2_k_block_decodes_to_the_ggml_reference,
    Q2_K,
    concat!(
        "12ff11bdd71c566ee39a03f9ebde5f9e3eeafd5ab4f4aa3511799ed220538bc8",
        "80deddf9d3d7baf93f42bdc17e7977f42729f4435ff7baca9e8328d9fa4e3184",
        "0698d2fd7862a239e04cf9ef504b2044451fc921",
    ),
    [
        (0, 0.017097473),
        (11, 0.017097473),
        (22, 0.04348755),
        (33, 0.00289917),
        (44, -0.011299133),
        (55, 0.060287476),
        (66, 0.002193451),
        (77, -0.09719467),
        (88, 0.24427032),
        (99, -0.013900757),
        (110, 0.028694153),
        (121, 0.031593323),
        (132, -0.0942955),
        (143, -0.15818787),
        (154, -0.030700684),
        (165, 0.021297455),
        (176, -0.105594635),
        (187, 0.022190094),
        (198, 0.07608414),
        (209, -0.04750061),
        (220, -0.04750061),
        (231, 0.26296616),
        (242, 0.19647217),
        (253, -0.0023040771),
        (255, -0.0023040771),
    ]
);

reference!(
    q3_k_block_decodes_to_the_ggml_reference,
    Q3_K,
    concat!(
        "513cd71855113ac60e98a370705c0c6cc5a6d6b9e91d0b38a023e0ae014fbda0",
        "4709e8fffcf2cff8d6c7c057de7e9ff123903238bbe563f56ec7b266bff9c474",
        "76287bad196dbc8999aa926c0937bcfeaf0036aa9d9d1ce704c9815872eabf63",
        "c17e9754da3ea7405f12bf6a451f",
    ),
    [
        (0, 0.36205673),
        (11, -0.12068558),
        (22, 0.29816437),
        (33, -0.32656097),
        (44, -0.16328049),
        (55, -0.08518982),
        (66, 0.36915588),
        (77, 0.5537338),
        (88, 0.25556946),
        (99, 0.48984146),
        (110, 0.32656097),
        (121, -0.0),
        (132, -0.028396606),
        (143, 0.056793213),
        (154, 0.1916771),
        (165, -0.17747879),
        (176, -0.035495758),
        (187, 0.070991516),
        (198, 0.021297455),
        (209, 0.8235016),
        (220, 0.2058754),
        (231, 0.14198303),
        (242, -0.0),
        (253, 0.08518982),
        (255, -0.08518982),
    ]
);

reference!(
    q4_k_block_decodes_to_the_ggml_reference,
    Q4_K,
    concat!(
        "451fc9219d1b645efb1d0f891f7c9a3575baad5711b8b0937499e433c92297b3",
        "75d815440095827c5450bb1709de3034def600c254b5c9d329f373a8c85aba7e",
        "e354b9a43dae4cf652843c4f9a82d24f47705a7fd09054d8eb19d3fc8eee07ed",
        "1bce13c0b61a602e2962b5d55aa5f7a312de08a01e726c42af359e0fc5d702ae",
        "dbd3a000d371cb5ca84a616b0201a2ac",
    ),
    [
        (0, 0.36272812),
        (11, -0.049022675),
        (22, -0.25489807),
        (33, 1.7807732),
        (44, 1.9724503),
        (55, 1.0140648),
        (66, -0.169487),
        (77, 2.3862076),
        (88, 0.34165192),
        (99, 2.4540024),
        (110, 2.2410278),
        (121, 1.6021042),
        (132, -0.55365753),
        (143, 3.783924),
        (154, 1.1146431),
        (165, 0.68761444),
        (176, 0.006095886),
        (187, 1.0283737),
        (198, 2.113243),
        (209, 0.45204163),
        (220, 0.26746368),
        (231, 0.20085907),
        (242, 1.0953522),
        (253, -0.39546967),
        (255, 1.0953522),
    ]
);

reference!(
    q5_k_block_decodes_to_the_ggml_reference,
    Q5_K,
    concat!(
        "451fc921c0478de684332890a3cc170b9b6de18c07870e9fad5242c340f1d5f3",
        "571a647f5d2580560c84c9a3c6128d5cb3f309e886381fda2c4f2e5f419e6899",
        "988df40b2002fc09c4d8045f8a98e7cbaf877ca5af50ed49c394b683b8add5de",
        "aaa0e4a1690aef046454c8380c066eb53f0e94b260965b82c75beb9f01c732f2",
        "63394909d8fb84c14094526f8fed8465b7b9bad61847131139407708ebbfa851",
        "469289e2813e364cc0e13624b2ea49ec",
    ),
    [
        (0, -0.045196533),
        (11, -0.045196533),
        (22, -0.045196533),
        (33, 0.16915512),
        (44, -0.37747955),
        (55, 0.21884918),
        (66, 0.6555023),
        (77, 0.7477913),
        (88, 1.3938141),
        (99, 6.8331757),
        (110, 3.3261948),
        (121, 1.1680527),
        (132, -0.4745636),
        (143, 6.0424576),
        (154, 0.24954987),
        (165, 1.6533966),
        (176, 1.0570679),
        (187, 4.237488),
        (198, 0.8193016),
        (209, 0.5424347),
        (220, 4.9723053),
        (231, 6.758877),
        (242, 2.9892273),
        (253, 5.502327),
        (255, 5.502327),
    ]
);

reference!(
    q6_k_block_decodes_to_the_ggml_reference,
    Q6_K,
    concat!(
        "4293ba0b32a291d6db143d29f323d53c314f8968cf76ea446fa012c169ad04a3",
        "b59907cf091577286170d493bec0a383ebeac662b4b4425ea26c6c3e7dac9d55",
        "137c0107d87f14ec47749c4ed8f12633cb5508cd1f648aacc09e3857cb28a01e",
        "8a4758942e6d4a7412b3d5e0d295debababe2a918bf1468a85ea8249d7e8ff23",
        "54dd2c0b4c4883524b6155bdd59bad0b037011bec8b1dca798afd5061caae997",
        "29cb881d7ff762bd76afc455feca077095647b733ccbb555ec8a3f9a98d0ef75",
        "6013767cf76acf8d4403a28f52346e83451f",
    ),
    [
        (0, -20.445557),
        (11, -4.77063),
        (22, -2.9674454),
        (33, 20.942497),
        (44, -1.6753998),
        (55, -1.7605896),
        (66, -0.702816),
        (77, 0.8944931),
        (88, -7.5251007),
        (99, 6.9571686),
        (110, -3.4785843),
        (121, -4.8984146),
        (132, 11.585815),
        (143, -13.999527),
        (154, 0.5111389),
        (165, 2.0019608),
        (176, 4.813225),
        (187, -7.219837),
        (198, 0.58213043),
        (209, 1.8457794),
        (220, -1.4766235),
        (231, 5.4663467),
        (242, 12.423515),
        (253, -26.621819),
        (255, 12.423515),
    ]
);

reference!(
    iq4_nl_block_decodes_to_the_ggml_reference,
    Iq4_Nl,
    "04238760260d2c74c00575dfcf611de25bd8",
    [
        (0, -0.13702393),
        (3, 0.9454651),
        (6, -1.7402039),
        (9, 1.5483704),
        (12, 0.9454651),
        (15, 0.013702393),
        (18, -1.1372986),
        (21, -0.13702393),
        (24, -0.13702393),
        (27, -0.30145264),
        (30, -0.47958374),
        (31, 0.9454651),
    ]
);

// Rows from gguf-py's `dequantize` (`gguf/quants.py`, the numpy port of ggml's
// `dequantize_row_iq2_xs` / `dequantize_row_iq3_xxs`) over seeded random blocks.
reference!(
    iq2_xs_block_decodes_to_the_ggml_reference,
    Iq2_Xs,
    concat!(
        "66324850543a2478bb5f4ae181b7c6af9a24d0cb74fab4c5c2ba3c17da9e347f",
        "0b19634819fdd757c7c9addddb33c1cea60186d12c4cac91ae62443db02a979a",
        "3349fddde3cbd37af7d3",
    ),
    [
        (0, 16.87088),
        (11, -29.017914),
        (22, 6.1984863),
        (33, 16.87088),
        (44, -29.017914),
        (55, -29.017914),
        (66, -1.3996582),
        (77, -7.523163),
        (88, -5.798584),
        (99, -14.3714905),
        (110, -24.718964),
        (121, 26.868439),
        (132, 7.523163),
        (143, 4.373932),
        (154, -5.3986816),
        (165, -13.121796),
        (176, -9.372711),
        (187, 2.9992676),
        (198, -16.121063),
        (209, 6.1984863),
        (220, -6.1984863),
        (231, 4.373932),
        (242, -5.3986816),
        (253, -16.87088),
        (255, 16.87088),
    ]
);

reference!(
    iq3_xxs_block_decodes_to_the_ggml_reference,
    Iq3_Xxs,
    concat!(
        "1c2e5d3697b99b819a8fdb4712e7ef54c41d7132946c8f368af080fdad616634",
        "bbd1b28d866741f17d79ad17fff61b885cb3311872ad6faabba09eb4ff6f4e99",
        "4ea5d22acf104e0f70fac5986f866d22552a0a9e77f341b58ea82c2798810d0a",
        "7731",
    ),
    [
        (0, 0.85913086),
        (11, 2.0046387),
        (22, 2.0046387),
        (33, -14.796143),
        (44, -14.796143),
        (55, -8.877686),
        (66, -4.868408),
        (77, -1.6228027),
        (88, -21.096436),
        (99, -2.3864746),
        (110, -0.47729492),
        (121, -4.2956543),
        (132, 2.9592285),
        (143, 14.796143),
        (154, 20.7146),
        (165, 2.0046387),
        (176, 2.0046387),
        (187, 6.013916),
        (198, 8.114014),
        (209, 8.114014),
        (220, 4.868408),
        (231, -6.013916),
        (242, -3.3410645),
        (253, 0.6682129),
        (255, -4.67749),
    ]
);

reference!(
    iq3_s_block_decodes_to_the_ggml_reference,
    Iq3_S,
    concat!(
        "552f4850543a2478bb5f4ae181b7c6af9a24d0cb74fab4c5c2ba3c17da9e347f",
        "0b19634819fdd757c7c9addddb33c1cea60186d12c4cac91ae62443db02a979a",
        "3349fddde3cbd37af7d38d895d3697b99b819a8fdb4712e7ef54c41d7132946c",
        "8f368af080fdad616634bbd1b28d",
    ),
    [
        (0, -7.904846),
        (11, -18.444641),
        (22, -13.174744),
        (33, -34.254333),
        (44, -34.254333),
        (55, -7.904846),
        (66, 2.4058228),
        (77, 3.0932007),
        (88, -2.4058228),
        (99, 21.652405),
        (110, -9.279602),
        (121, 34.025208),
        (132, 1.7184448),
        (143, 2.8640747),
        (154, 1.7184448),
        (165, 2.6349487),
        (176, -7.904846),
        (187, 28.984436),
        (198, 21.652405),
        (209, 15.466003),
        (220, -9.279602),
        (231, -13.632996),
        (242, -1.9475708),
        (253, -21.423279),
        (255, 13.632996),
    ]
);

reference!(
    iq4_xs_block_decodes_to_the_ggml_reference,
    Iq4_Xs,
    concat!(
        "451f40ea9cb8b3a47f4a630b23f32b3622e927be470ce34ff63d9ec4db621023",
        "dfce18a8378c574185433b3128c79dbbefe90a0690397faf39fb09b28497e655",
        "6cf7c412c93121c3192ba819812b1786651383708c1e4f244e9ece2edc660fb4",
        "a25109ca0cf962c13e0018aead9babf9d1e4f9ac07d10b342fa353897ce27c29",
        "1357df59f780d916",
    ),
    [
        (0, -16.044083),
        (11, -12.63649),
        (22, 11.784592),
        (33, -11.266354),
        (44, 1.6328049),
        (55, 13.55228),
        (66, -6.4744263),
        (77, -2.2149353),
        (88, -15.163788),
        (99, 2.946148),
        (110, 3.6915588),
        (121, -4.0110207),
        (132, -2.2149353),
        (143, -1.0435753),
        (154, 0.021297455),
        (165, -1.7179947),
        (176, -3.8264427),
        (187, 4.1388054),
        (198, 1.079071),
        (209, -3.606369),
        (220, -3.606369),
        (231, 2.3995132),
        (242, -6.460228),
        (253, 0.18457794),
        (255, -19.196106),
    ]
);

reference!(
    mxfp4_block_decodes_to_the_ggml_reference,
    Mxfp4,
    "7c5c44b968cff4eb51b989d24e70cf4430",
    [
        (0, -0.25),
        (3, -0.0),
        (6, -0.1875),
        (9, -0.0625),
        (12, 0.0),
        (15, 0.0),
        (18, -0.1875),
        (21, -0.75),
        (24, -0.1875),
        (27, 0.25),
        (30, 0.25),
        (31, 0.1875),
    ]
);

/// A hand-built `Q4_K` block: `d = 0.25`, `dmin = 0.5`, scales chosen so that every sub-block
/// has a distinct 6-bit scale and min and sub-blocks 4..8 take their high bits from bytes 0..8
/// (`get_scale_min_k4`'s `j >= 4` branch), and `qs[b] = (7b mod 16) | ((3b + 5) mod 16) << 4`.
/// Sub-block `(sc, m)` pairs are (1, 5), (2, 6), (3, 7), (4, 8), (17, 18), (35, 4), (53, 38),
/// (7, 56); values from `dequantize_row_q4_K`.
#[test]
fn q4_k_split_six_bit_scales_decode_to_the_ggml_reference() {
    let mut block = vec![0x00, 0x34, 0x00, 0x38];
    block.extend_from_slice(&[
        0x41, 0x82, 0xc3, 0x04, 0x45, 0x06, 0x87, 0xc8, 0x21, 0x43, 0x65, 0x87,
    ]);
    block.extend((0..128u32).map(|b| ((b * 7) & 0xf) as u8 | ((((b * 3 + 5) & 0xf) as u8) << 4)));
    assert_block(
        WeightFormat::Q4_K,
        &block,
        &[
            (0, -2.5),
            (1, -0.75),
            (31, -0.25),
            (32, -0.5),
            (33, 1.0),
            (63, -2.0),
            (64, -3.5),
            (95, 3.25),
            (96, 1.0),
            (127, -2.0),
            (128, -9.0),
            (159, 29.25),
            (160, 41.75),
            (191, 15.5),
            (192, -19.0),
            (223, 100.25),
            (224, -19.25),
            (255, -24.5),
        ],
    );
}

/// IEEE-754 binary32 and binary16 and bfloat16 (the high half of a binary32), little-endian,
/// as GGUF and safetensors store them.
#[test]
fn dense_floats_decode_to_their_ieee_values() {
    let cases: [(WeightFormat, &[u8], f32); 9] = [
        (WeightFormat::F32, &[0x00, 0x00, 0x80, 0x3f], 1.0),
        (
            WeightFormat::F32,
            &[0xdb, 0x0f, 0x49, 0xc0],
            -std::f32::consts::PI,
        ),
        (
            WeightFormat::F32,
            &[0x01, 0x00, 0x00, 0x00],
            f32::from_bits(1),
        ),
        (WeightFormat::F16, &[0x55, 0x35], 0.333_251_95),
        (WeightFormat::F16, &[0x01, 0x80], -5.960_464_5e-8),
        (WeightFormat::F16, &[0xff, 0x7b], 65504.0),
        (WeightFormat::Bf16, &[0x80, 0x3f], 1.0),
        (WeightFormat::Bf16, &[0x49, 0xc0], -3.140_625),
        (WeightFormat::Bf16, &[0x00, 0x80], -0.0),
    ];
    for (format, bytes, want) in cases {
        assert_block(format, bytes, &[(0, want)]);
    }
}

#[test]
fn a_stored_nan_is_a_decode_error() {
    let error = WeightFormat::Mxfp4
        .descriptor()
        .decode_block_value(&[0xff; 17], 0)
        .unwrap_err();
    assert_eq!(
        error,
        crate::decode::DecodeError::NotANumber {
            format: WeightFormat::Mxfp4,
            role: OperandRole::Scale,
        }
    );
}

/// FNV-1a over a table's bytes.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The lattice tables are ggml's, entry for entry: their digests equal those of the grids
/// `gguf-py` (`gguf/quants.py`, `grid` of `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS` and `IQ3_S`, unpacked
/// from `grid_hex`) builds, an independent copy of `ggml-common.h`'s `iq2xxs_grid`, `iq2xs_grid`,
/// `iq2s_grid`, `iq3xxs_grid` and `iq3s_grid`. A single swapped or edited grid point changes the digest.
/// Mutation: swap two entries of `IQ2XS_GRID`, `IQ2XXS_GRID` or `IQ2S_GRID`; the matching assertion goes red.
#[test]
fn the_lattice_tables_are_the_gguf_py_grids() {
    /// The table's cells as the little-endian byte run ggml's grids are.
    fn cell_bytes(lattice: &Lattice) -> Vec<u8> {
        lattice
            .cells
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
    assert_eq!(fnv1a(&cell_bytes(&IQ2XXS_LATTICE)), 0xbb4e_e025_b5ac_6e8e);
    assert_eq!(fnv1a(&cell_bytes(&IQ2XS_LATTICE)), 0xc9b1_ee61_e799_09bd);
    assert_eq!(fnv1a(&cell_bytes(&IQ2S_LATTICE)), 0x123e_dd38_a3b6_2b90);
    assert_eq!(fnv1a(&cell_bytes(&IQ3XXS_LATTICE)), 0xdfa5_dc83_d6a1_55d5);
    assert_eq!(IQ2XXS_LATTICE.points(), 256);
    assert_eq!(IQ2XS_LATTICE.points(), 512);
    assert_eq!(IQ2S_LATTICE.points(), 1024);
    assert_eq!(fnv1a(&cell_bytes(&IQ3S_LATTICE)), 0xfa37_020c_25b4_4829);
    // `IQ1_S`'s points as `gguf-py`'s `IQ1_S.grid` (`-1, 0, 1` per cell), each cell plus one in a nibble.
    assert_eq!(fnv1a(&cell_bytes(&IQ1S_LATTICE)), 0xdf33_d6de_af4e_7cbe);
    assert_eq!(IQ1S_LATTICE.points(), 2048);
    assert_eq!(IQ3XXS_LATTICE.points(), 256);
    assert_eq!(IQ3S_LATTICE.points(), 512);
    // A grid point is a little-endian run of magnitude bytes, as ggml's `uint64_t` entries.
    let cell = |lattice: &Lattice, point: usize, lane: usize| {
        lattice.cells.cell(point * lattice.lanes + lane)
    };
    assert_eq!(cell(&IQ2XS_LATTICE, 1, 0), 0x2b);
    assert_eq!(cell(&IQ2XS_LATTICE, 1, 7), 0x08);
    assert_eq!(cell(&IQ3XXS_LATTICE, 3, 0), 0x0c);
    assert_eq!(cell(&IQ3XXS_LATTICE, 3, 3), 0x04);
}

/// The sign rule is ggml's `ksigns_iq2xs`: the stored seven bits plus their parity in bit 7.
#[test]
fn the_eighth_sign_is_the_parity_of_the_stored_seven() {
    // `ksigns_iq2xs` begins 0, 129, 130, 3, 132: index 1 sets bit 7, index 3 does not.
    let ksigns = |index: u32| -> u32 {
        (0..8).fold(0, |byte, lane| {
            byte | (u32::from(SignRule::ParityOfSeven.negative(index, lane)) << lane)
        })
    };
    assert_eq!(
        [0, 1, 2, 3, 4].map(ksigns),
        [0, 129, 130, 3, 132],
        "ggml-common.h ksigns_iq2xs[0..5]"
    );
    assert_eq!(ksigns(127), 255);
    assert_eq!(ksigns(126), 126);
    for index in 0..128 {
        assert_eq!(ksigns(index).count_ones() % 2, 0, "index {index}");
    }
}

/// `dequantize_row_iq2_xxs`, `_iq2_xs`, `_iq2_s`, `_iq3_xxs` and `_iq3_s` of `ggml/src/ggml-quants.c`, written
/// as ggml loops over the raw block with ggml's `ksigns_iq2xs` / `kmask_iq2xs` semantics, one
/// block at a time. The descriptor decodes the same random blocks bit for bit, so the bit
/// layouts (which `qs` word and which scale nibble feeds which value) match ggml's loops.
fn ggml_dequantize(format: WeightFormat, block: &[u8]) -> Vec<f32> {
    let d = poot_tensor::float::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let ksigns = |index: u32| index | ((index.count_ones() & 1) << 7);
    let mut y = Vec::new();
    match format {
        WeightFormat::Iq2_Xs => {
            let qs: Vec<u16> = block[2..66]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| u16::from_le_bytes(b))
                .collect();
            let scales = &block[66..74];
            for ib32 in 0..8 {
                let db = [
                    d * (0.5 + f32::from(scales[ib32] & 0xf)) * 0.25,
                    d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
                ];
                for l in 0..4 {
                    let word = qs[4 * ib32 + l];
                    let grid = IQ2XS_GRID[usize::from(word & 511)].to_le_bytes();
                    let signs = ksigns(u32::from(word >> 9));
                    for (j, &magnitude) in grid.iter().enumerate() {
                        let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                        y.push(db[l / 2] * f32::from(magnitude) * sign);
                    }
                }
            }
        }
        WeightFormat::Iq2_Xxs => {
            for ib32 in 0..8 {
                let run = &block[2 + 8 * ib32..2 + 8 * ib32 + 8];
                let aux32 = u32::from_le_bytes(run[4..8].try_into().unwrap());
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[usize::from(run[l])].to_le_bytes();
                    let signs = ksigns((aux32 >> (7 * l)) & 127);
                    for (j, &magnitude) in grid.iter().enumerate() {
                        let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                        y.push(db * f32::from(magnitude) * sign);
                    }
                }
            }
        }
        WeightFormat::Iq2_S => {
            let (qs, rest) = block[2..].split_at(64);
            let (qh, scales) = rest.split_at(8);
            for ib32 in 0..8 {
                let db = [
                    d * (0.5 + f32::from(scales[ib32] & 0xf)) * 0.25,
                    d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
                ];
                for l in 0..4 {
                    let point = usize::from(qs[4 * ib32 + l])
                        | ((usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300);
                    let grid = IQ2S_GRID[point].to_le_bytes();
                    let signs = qs[32 + 4 * ib32 + l];
                    for (j, &magnitude) in grid.iter().enumerate() {
                        let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                        y.push(db[l / 2] * f32::from(magnitude) * sign);
                    }
                }
            }
        }
        WeightFormat::Iq3_Xxs => {
            let (qs, scales_and_signs) = block[2..].split_at(64);
            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes(
                    scales_and_signs[4 * ib32..4 * ib32 + 4].try_into().unwrap(),
                );
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = ksigns((aux32 >> (7 * l)) & 127);
                    let grid1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])].to_le_bytes();
                    let grid2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])].to_le_bytes();
                    for (half, grid) in [grid1, grid2].into_iter().enumerate() {
                        for (j, &magnitude) in grid.iter().enumerate() {
                            let lane = j + 4 * half;
                            let sign = if signs & (1 << lane) != 0 { -1.0 } else { 1.0 };
                            y.push(db * f32::from(magnitude) * sign);
                        }
                    }
                }
            }
        }
        WeightFormat::Iq3_S => {
            let (qs, rest) = block[2..].split_at(64);
            let (qh, rest) = rest.split_at(8);
            let (signs, scales) = rest.split_at(32);
            for (pair, &scale) in scales.iter().enumerate() {
                let db = [
                    d * (1.0 + 2.0 * f32::from(scale & 0xf)),
                    d * (1.0 + 2.0 * f32::from(scale >> 4)),
                ];
                for (half, db) in db.into_iter().enumerate() {
                    let run = 2 * pair + half;
                    for l in 0..4 {
                        let low = |k: usize| usize::from(qs[8 * run + 2 * l + k]);
                        let high = |shift: usize| (usize::from(qh[run]) << shift) & 256;
                        let grid1 = IQ3S_GRID[low(0) | high(8 - 2 * l)].to_le_bytes();
                        let grid2 = IQ3S_GRID[low(1) | high(7 - 2 * l)].to_le_bytes();
                        let sign_byte = signs[4 * run + l];
                        for (lane, &magnitude) in grid1.iter().chain(&grid2).enumerate() {
                            let sign = if sign_byte & (1 << lane) != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            y.push(db * f32::from(magnitude) * sign);
                        }
                    }
                }
            }
        }
        _ => unreachable!("only the lattice formats are transcribed"),
    }
    y
}

/// Random lattice-format blocks (every sign pattern, grid point and scale nibble
/// occurs) decode, through the descriptor interpreter, to exactly what ggml's loops give.
/// Mutation: read the `Signs` field of `IQ3_XXS` at bit 529 instead of 528; red.
#[test]
fn lattice_blocks_decode_as_ggml_loops_do() {
    let mut state = 0x1192_u64;
    for format in GGUF_FORMATS.into_iter().filter(|format| {
        matches!(format.descriptor().value, ValueMap::Lattice(lattice) if lattice.signs.is_some())
    }) {
        let descriptor = format.descriptor();
        let Storage::Blocks(layout) = descriptor.storage else {
            unreachable!()
        };
        for _ in 0..256 {
            let mut block: Vec<u8> = (0..layout.bytes)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 24) as u8
                })
                .collect();
            // A finite scale: keep the f16 exponent below all-ones.
            block[1] &= 0x7b;
            let want = ggml_dequantize(format, &block);
            for (index, want) in want.into_iter().enumerate() {
                let got = descriptor.decode_block_value(&block, index).unwrap();
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{format:?} element {index}: {got:?} vs ggml {want:?}"
                );
            }
        }
    }
}

/// `dequantize_row_tq1_0` of `ggml/src/ggml-quants.c`, loop for loop over the raw block: five trits to a byte
/// along 32 bytes of `qs`, five along the next 16, four along the 4 bytes of `qh`, each `(((uint8_t)(q *
/// 3^n)) * 3) >> 8`.
fn ggml_dequantize_tq1_0(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), 54);
    let (qs, rest) = block.split_at(48);
    let (qh, scale) = rest.split_at(4);
    let d = poot_tensor::float::f16_to_f32(u16::from_le_bytes([scale[0], scale[1]]));
    let pow3: [u8; 5] = [1, 3, 9, 27, 81];
    let trit = |byte: u8, n: usize| {
        let q = byte.wrapping_mul(pow3[n]);
        (f32::from((u16::from(q) * 3) >> 8) - 1.0) * d
    };
    let mut y = Vec::new();
    for n in 0..5 {
        y.extend(qs[..32].iter().map(|&byte| trit(byte, n)));
    }
    for n in 0..5 {
        y.extend(qs[32..].iter().map(|&byte| trit(byte, n)));
    }
    for n in 0..4 {
        y.extend(qh.iter().map(|&byte| trit(byte, n)));
    }
    y
}

/// Random `TQ1_0` blocks (every byte value in `qs` and `qh`, every scale) decode, through the descriptor
/// interpreter, to exactly what ggml's loop gives: the three stretches of the layout (160 values from
/// `qs[0..32]`, 80 from `qs[32..48]`, 16 from `qh`) and the lane each reads are ggml's. Mutation: start the
/// second segment of `TQ1_0`'s codes at element 128 instead of 160; red at element 128.
#[test]
fn tq1_0_blocks_decode_as_ggmls_loop_does() {
    let descriptor = WeightFormat::Tq1_0.descriptor();
    let mut state = 0x7101_u64;
    for _ in 0..256 {
        let mut block: Vec<u8> = (0..54)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect();
        // A finite scale: keep the f16 exponent below all-ones.
        block[53] &= 0x7b;
        for (index, want) in ggml_dequantize_tq1_0(&block).into_iter().enumerate() {
            let got = descriptor.decode_block_value(&block, index).unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "Tq1_0 element {index}: {got:?} vs ggml {want:?}"
            );
        }
    }
}

/// The `TQ1_0` table is ggml's trit arithmetic for every byte and digit, and it inverts ggml's encoder: the
/// byte `quantize_row_tq1_0_ref` writes for five trits (`q = (q * 256 + 242) / 243` over their base-3 value;
/// four trits over 81 for `qh`) reads back as those trits. Mutation: change one cell of `trit_words`; red
/// naming the byte and digit.
#[test]
fn the_tq1_0_table_is_ggmls_trit_arithmetic_and_inverts_its_encoder() {
    let lattice = match WeightFormat::Tq1_0.descriptor().value {
        ValueMap::Lattice(lattice) => lattice,
        _ => unreachable!("TQ1_0 is a lattice format"),
    };
    let trit = |byte: usize, lane: usize| lattice.cells.cell(byte * lattice.lanes + lane);
    for byte in 0..256usize {
        let mut power = 1u8;
        for lane in 0..5 {
            let q = (byte as u8).wrapping_mul(power);
            assert_eq!(
                trit(byte, lane),
                (u32::from(q) * 3) >> 8,
                "byte {byte} digit {lane}"
            );
            power = power.wrapping_mul(3);
        }
    }
    for (digits, modulus) in [(5, 243usize), (4, 81)] {
        for value in 0..modulus {
            let trits: Vec<u32> = (0..digits)
                .map(|n| (value / 3usize.pow((digits - 1 - n) as u32) % 3) as u32)
                .collect();
            // ggml's ceiling division: `(q * 256 + (modulus - 1)) / modulus`.
            let byte = (value * 256).div_ceil(modulus);
            for (n, &want) in trits.iter().enumerate() {
                assert_eq!(
                    trit(byte, n),
                    want,
                    "the byte {byte} ggml writes for trits {trits:?}, digit {n}"
                );
            }
        }
    }
}

/// The segments of a layout or lane map start at element 0, each begins where the one before ends, and the last
/// one ends at the block's last element (the layouts' extents are checked against the block separately).
fn assert_segments_tile_the_block(
    format: WeightFormat,
    what: &str,
    segments: &[crate::format::Segment],
) {
    assert_eq!(
        segments[0].first, 0,
        "{format:?} {what}: starts at element 0"
    );
    for pair in segments.windows(2) {
        assert_eq!(
            pair[0].first + pair[0].extent(),
            pair[1].first,
            "{format:?} {what}: a segment ends where the next begins"
        );
    }
}

/// Every block descriptor is well formed: each piece's layout covers the block's values, stays
/// inside the block, an integer value map has an integer `Codes` field, a lattice code indexes exactly
/// its points, and a min sign exactly when there is a `Min` operand.
#[test]
fn every_block_descriptor_is_well_formed() {
    for format in GGUF_FORMATS {
        let descriptor = format.descriptor();
        assert_eq!(descriptor.format, format);
        let Storage::Blocks(layout) = descriptor.storage else {
            panic!("{format:?} is a block format");
        };
        for field in layout.fields {
            let Field { pieces, .. } = field.field;
            assert!(field.field.width() <= 32, "{format:?} {}", field.name);
            for piece in pieces {
                assert_eq!(
                    piece.layout.extent() as usize,
                    layout.values,
                    "{format:?} {} layout extent",
                    field.name
                );
                assert_segments_tile_the_block(format, field.name, piece.layout.segments);
                let last = (0..layout.values as u32)
                    .map(|i| piece.layout.bit(i) + piece.width)
                    .max()
                    .unwrap();
                assert!(
                    last as usize <= layout.bytes * 8,
                    "{format:?} {} reads past the block",
                    field.name
                );
            }
        }
        let codes = layout.field(OperandRole::Codes).expect("a Codes field");
        match descriptor.value {
            ValueMap::Integer => {
                assert!(!matches!(codes.field.encoding, FieldEncoding::Float(_)))
            }
            ValueMap::Lattice(lattice) => {
                let cells = lattice.cells;
                assert!(
                    cells.width.is_power_of_two() && cells.width <= 32,
                    "{format:?} cells fill whole words"
                );
                assert!(
                    matches!(
                        cells.encoding,
                        FieldEncoding::Unsigned { .. } | FieldEncoding::Float(_)
                    ),
                    "{format:?} a cell is an unsigned integer or a float"
                );
                assert_eq!(
                    lattice.lanes * cells.width as usize % 32,
                    0,
                    "{format:?} a point fills whole words"
                );
                assert_eq!(
                    cells.cells() % lattice.lanes,
                    0,
                    "{format:?} the table is whole points"
                );
                assert_segments_tile_the_block(format, "lane map", lattice.lane.segments);
                assert!(
                    (0..layout.values).all(|i| lattice.lane.lane(i) < lattice.lanes),
                    "{format:?} an element reads a lane its points have"
                );
                assert_eq!(
                    1usize << codes.field.width(),
                    lattice.points(),
                    "{format:?}: the code indexes exactly the lattice points"
                );
                let signs = layout.field(OperandRole::Signs);
                assert_eq!(
                    signs.map(|signs| signs.field.width()),
                    lattice.signs.map(|rule| match rule {
                        SignRule::ParityOfSeven => 7,
                        SignRule::Direct => 8,
                    }),
                    "{format:?} stored sign bits"
                );
            }
            ValueMap::Float => {
                assert!(matches!(codes.field.encoding, FieldEncoding::Float(_)))
            }
        }
        assert_eq!(
            descriptor.min_sign.is_some(),
            layout.field(OperandRole::Min).is_some(),
            "{format:?} min sign"
        );
    }
}
