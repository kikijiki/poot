//! A `Gather` over an I32 table whose words need all 32 bits: the rows an I32 const table (a hash
//! table of ids) hands out must come back word for word. Each backend's device row compiles
//! [`graph`], binds [`table`] as the `"table"` const and [`tokens`] as the `Slot::Token` input,
//! runs it, and hands the output words to [`check`].
//!
//! Every table word is odd and above 2^24 (or an I32 extreme), so a read through an F32 lane rounds
//! it to a different integer.

use std::collections::HashMap;

use crate::{EvalBudget, EvalOptions, Value};
use poot_graph_ir::{Builder, Graph, Slot, Storage, TensorType, ValueId};
use poot_tensor::{DType, HostTensor};

const ROWS: usize = 6;
const WIDTH: usize = 4;
const TOKENS: [i32; 5] = [4, 0, 5, 2, 4];

/// `gather(table[ROWS, WIDTH] : I32, token[N] : I32)`, an I32 `[N, WIDTH]` output.
pub fn graph() -> Graph {
    let b = Builder::for_tests();
    let tokens = b
        .slot(Slot::Token, TensorType::new(vec![TOKENS.len()], DType::I32))
        .unwrap();
    let table = b
        .constant("table", TensorType::new(vec![ROWS, WIDTH], DType::I32))
        .unwrap();
    let rows = b.gather(table, 0, tokens).unwrap();
    b.finish(rows).unwrap()
}

/// The table's words, row-major: odd integers from 2^24 + 1 up, then the I32 extremes at the corners.
pub fn table() -> Vec<i32> {
    let mut words: Vec<i32> = (0..ROWS * WIDTH)
        .map(|i| (1 << 24) + 1 + 2 * i as i32)
        .collect();
    words[0] = i32::MIN;
    words[ROWS * WIDTH - 1] = i32::MAX;
    words
}

/// The step's token ids, the gather's rows (repeats included).
pub fn tokens() -> Vec<i32> {
    TOKENS.to_vec()
}

/// The output shape.
pub fn out_shape() -> Vec<usize> {
    vec![TOKENS.len(), WIDTH]
}

/// `graph`'s inputs bound by what they are: the token slot and the `"table"` const.
pub fn inputs(graph: &Graph) -> HashMap<ValueId, Value> {
    graph
        .inputs()
        .iter()
        .map(|&id| {
            let tensor = match &graph.meta(id).storage {
                Storage::Slot(Slot::Token) => HostTensor::i32(vec![TOKENS.len()], tokens()),
                Storage::Const => HostTensor::i32(vec![ROWS, WIDTH], table()),
                other => panic!("unexpected input v{id}: {other:?}"),
            };
            (id, Value::from(tensor))
        })
        .collect()
}

/// Compare `got` (a `backend` run of [`graph`]) with the rows of the table the tokens name, and the CPU
/// oracle with the same; the first differing word is the error.
pub fn check(backend: &str, got: &[i32]) -> Result<(), String> {
    let table = table();
    let truth: Vec<i32> = TOKENS
        .iter()
        .flat_map(|&t| {
            table[t as usize * WIDTH..(t as usize + 1) * WIDTH]
                .iter()
                .copied()
        })
        .collect();
    let g = graph();
    let inputs = inputs(&g);
    let oracle = crate::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map_err(|e| format!("CPU oracle: {e}"))?
        .output
        .into_host()
        .map_err(|e| format!("CPU oracle: {e}"))?;
    let first_difference = |label: &str, words: &[i32]| -> Result<(), String> {
        if words.len() != truth.len() {
            return Err(format!(
                "{label} has {} words, want {}",
                words.len(),
                truth.len()
            ));
        }
        match words
            .iter()
            .zip(&truth)
            .enumerate()
            .find(|(_, (g, t))| g != t)
        {
            Some((i, (g, t))) => Err(format!("{label} word {i}: {g} vs table {t}")),
            None => Ok(()),
        }
    };
    first_difference("CPU oracle", oracle.as_i32().expect("an I32 gather output"))?;
    first_difference(backend, got)
}
