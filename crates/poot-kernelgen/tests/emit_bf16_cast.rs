//! `emit::value`'s AMDGCN f32 -> bf16 cast lowering, moved from `src/emit/value.rs`'s own `#[cfg(test)]`
//! module (card 671; see `tests/emit_golden_ir.rs`'s header for why this needed to become an integration
//! test instead of staying in `src/`).

use std::collections::HashMap;

use poot_codegen::Target;
use poot_target::AmdArch;
use poot_test_util::kernel_fixtures::emit_llvm_ir;

/// Run the straight-line f32 -> bf16 cast sequence of the emitted AmdGcn IR on `x` and return the bf16
/// bits. The interpreter knows only the handful of integer ops the lowering uses and panics on any other,
/// so a changed lowering fails loudly instead of being skipped.
fn amdgcn_f32_to_bf16(x: f32) -> u16 {
    let ir = emit_llvm_ir(
        &poot_kernelgen::cast_f32_to_bf16("cast_f32_bf16_nan"),
        Target::AmdGcn(AmdArch::gfx1151()),
    )
    .unwrap();
    let lines: Vec<&str> = ir.lines().map(str::trim).collect();
    let start = lines
        .iter()
        .position(|l| l.contains("= bitcast float") && l.ends_with("to i32"))
        .unwrap_or_else(|| panic!("no f32 -> bf16 lowering in the emitted IR:\n{ir}"));
    let end = lines
        .iter()
        .position(|l| l.contains("= bitcast i16") && l.ends_with("to bfloat"))
        .unwrap_or_else(|| panic!("f32 -> bf16 lowering never yields a bfloat:\n{ir}"));
    let mut regs: HashMap<String, u32> = HashMap::new();
    let first: Vec<&str> = lines[start].split_whitespace().collect();
    regs.insert(first[4].to_string(), x.to_bits()); // `%d = bitcast float %src to i32`
    let mut last = 0u32;
    for line in &lines[start..=end] {
        let (dst, rhs) = line.split_once(" = ").expect("an SSA definition");
        let words: Vec<&str> = rhs
            .split_whitespace()
            .map(|w| w.trim_end_matches(','))
            .collect();
        let val = |tok: &str| -> u32 {
            match tok.strip_prefix('%') {
                Some(_) => *regs
                    .get(tok)
                    .unwrap_or_else(|| panic!("`{line}` reads undefined {tok}")),
                None if tok == "true" => 1,
                None if tok == "false" => 0,
                None => tok
                    .parse::<i64>()
                    .unwrap_or_else(|_| panic!("`{line}`: literal {tok}"))
                    as u32,
            }
        };
        let out = match words[0] {
            "bitcast" if words[1] == "float" => x.to_bits(), // the seed, already registered
            "bitcast" => val(words[2]),                      // i16 -> bfloat: same bits
            "trunc" => val(words[2]) & 0xffff,
            "lshr" => val(words[2]) >> val(words[3]),
            "and" => val(words[2]) & val(words[3]),
            "or" => val(words[2]) | val(words[3]),
            "add" => val(words[2]).wrapping_add(val(words[3])),
            "fcmp" => {
                assert_eq!(words[1], "uno", "`{line}`");
                (f32::from_bits(val(words[3])).is_nan() || f32::from_bits(val(words[4])).is_nan())
                    as u32
            }
            "select" => {
                if val(words[2]) != 0 {
                    val(words[4])
                } else {
                    val(words[6])
                }
            }
            op => panic!("unsupported op `{op}` in the f32 -> bf16 lowering: `{line}`"),
        };
        regs.insert(dst.to_string(), out);
        last = out;
    }
    last as u16
}

fn is_bf16_nan(bits: u16) -> bool {
    bits & 0x7f80 == 0x7f80 && bits & 0x007f != 0
}

/// ADR-0101 decision 4 (R482-012): a NaN in is a NaN out. The plain round-to-nearest bias add turns a NaN
/// with a low payload into Inf (0x7f800001 -> 0x7f80) and an all-ones NaN into -0.0 (0xffffffff -> 0x0000).
#[test]
fn amdgcn_f32_to_bf16_keeps_nan() {
    for bits in [
        0x7fc0_0000u32,
        0x7f80_0001,
        0x7fff_ffff,
        0xffff_ffff,
        0xff80_0001,
    ] {
        let got = amdgcn_f32_to_bf16(f32::from_bits(bits));
        assert!(
            is_bf16_nan(got),
            "f32 {bits:#010x} (NaN) lowered to bf16 {got:#06x}, want a NaN"
        );
        assert_eq!(
            got >> 15,
            (bits >> 31) as u16,
            "NaN sign of {bits:#010x} carries through"
        );
    }
}

/// The NaN guard leaves every non-NaN result unchanged: round to nearest even, Inf stays Inf.
#[test]
fn amdgcn_f32_to_bf16_still_rounds_to_nearest_even() {
    let cases: [(u32, u16, &str); 7] = [
        (0x3f80_0000, 0x3f80, "1.0 is exact"),
        (0x3f80_8000, 0x3f80, "tie with an even mantissa stays"),
        (0x3f81_8000, 0x3f82, "tie with an odd mantissa rounds up"),
        (0x3f80_8001, 0x3f81, "above the tie rounds up"),
        (0x7f80_0000, 0x7f80, "+Inf stays Inf"),
        (0xff80_0000, 0xff80, "-Inf stays -Inf"),
        (0x8000_0000, 0x8000, "-0.0 stays -0.0"),
    ];
    for (bits, want, why) in cases {
        let got = amdgcn_f32_to_bf16(f32::from_bits(bits));
        assert_eq!(
            got, want,
            "f32 {bits:#010x} -> {got:#06x}, want {want:#06x}: {why}"
        );
    }
}
