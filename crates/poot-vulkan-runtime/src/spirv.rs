//! The two facts about a SPIR-V module a pipeline's creation depends on, read from its words: whether it
//! declares cooperative matrices, and its `LocalSize` X. A module is a header of five words followed by
//! instructions, each one word of `(word count << 16) | opcode` and its operands; capabilities come first
//! and execution modes follow the entry points.

const HEADER_WORDS: usize = 5;
const OP_CAPABILITY: u32 = 17;
const OP_EXECUTION_MODE: u32 = 16;
const CAPABILITY_COOPERATIVE_MATRIX_KHR: u32 = 6022;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

/// What [`scan`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModuleFacts {
    /// `OpCapability CooperativeMatrixKHR` is declared.
    pub(crate) cooperative_matrix: bool,
    /// `OpExecutionMode ... LocalSize x y z`'s `x`, if the module declares a literal one.
    pub(crate) local_size_x: Option<u32>,
}

/// Read `words` once. A malformed instruction (a zero word count, or one past the end) ends the scan: what
/// was read stays, so a truncated module reports what it declared before the cut.
pub(crate) fn scan(words: &[u32]) -> ModuleFacts {
    let mut facts = ModuleFacts {
        cooperative_matrix: false,
        local_size_x: None,
    };
    let mut at = HEADER_WORDS;
    while at < words.len() {
        let head = words[at];
        let (count, opcode) = ((head >> 16) as usize, head & 0xffff);
        if count == 0 || at + count > words.len() {
            break;
        }
        let operands = &words[at + 1..at + count];
        match opcode {
            OP_CAPABILITY if operands.first() == Some(&CAPABILITY_COOPERATIVE_MATRIX_KHR) => {
                facts.cooperative_matrix = true;
            }
            // `OpExecutionMode %entry mode ...`: `LocalSize` carries three literal sizes.
            OP_EXECUTION_MODE
                if operands.get(1) == Some(&EXECUTION_MODE_LOCAL_SIZE) && operands.len() >= 5 =>
            {
                facts.local_size_x = Some(operands[2]);
            }
            _ => {}
        }
        at += count;
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![((operands.len() as u32 + 1) << 16) | opcode];
        words.extend_from_slice(operands);
        words
    }

    fn module(instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0500, 0, 100, 0];
        words.extend(instructions.iter().flatten());
        words
    }

    /// A module that declares cooperative matrices and a 32-wide workgroup is read as such; one that
    /// declares only shader capability and another size is not. Mutation: match capability 6021 (or a
    /// different `LocalSize` operand); the first assertion fails with the wrong fact.
    #[test]
    fn capabilities_and_local_size_are_read_from_the_instruction_stream() {
        let coopmat = module(&[
            instruction(OP_CAPABILITY, &[1]),
            instruction(OP_CAPABILITY, &[CAPABILITY_COOPERATIVE_MATRIX_KHR]),
            instruction(OP_EXECUTION_MODE, &[7, EXECUTION_MODE_LOCAL_SIZE, 32, 1, 1]),
        ]);
        assert_eq!(
            scan(&coopmat),
            ModuleFacts {
                cooperative_matrix: true,
                local_size_x: Some(32)
            }
        );
        let plain = module(&[
            instruction(OP_CAPABILITY, &[1]),
            instruction(OP_EXECUTION_MODE, &[7, EXECUTION_MODE_LOCAL_SIZE, 64, 1, 1]),
        ]);
        assert_eq!(
            scan(&plain),
            ModuleFacts {
                cooperative_matrix: false,
                local_size_x: Some(64)
            }
        );
    }

    /// A truncated or malformed stream never reads past its end and keeps what came before the damage.
    #[test]
    fn a_malformed_instruction_ends_the_scan_without_reading_past_it() {
        let mut words = module(&[instruction(
            OP_CAPABILITY,
            &[CAPABILITY_COOPERATIVE_MATRIX_KHR],
        )]);
        words.push(5 << 16 | OP_EXECUTION_MODE); // claims five words, has none
        assert_eq!(
            scan(&words),
            ModuleFacts {
                cooperative_matrix: true,
                local_size_x: None
            }
        );
        assert_eq!(scan(&[]).local_size_x, None);
        assert_eq!(
            scan(&[0, 0, 0, 0, 0, 0]).local_size_x,
            None,
            "a zero word count ends the scan"
        );
    }
}
