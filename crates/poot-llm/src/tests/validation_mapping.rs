// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

use std::error::Error;

use super::*;
use poot_graph_ir::{ExecutionValidationFailure, ValidationId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckContext {
    Attention { layer: usize },
    Router { layer: usize },
}

#[derive(Debug, thiserror::Error)]
enum AdapterError {
    #[error("validation {context:?} failed")]
    Failed {
        context: CheckContext,
        #[source]
        source: ExecutionValidationFailure,
    },
    #[error("validation id {id:?} is absent from the model contract")]
    UnknownId {
        id: ValidationId,
        #[source]
        source: ExecutionValidationFailure,
    },
}

fn map_failure(source: ExecutionValidationFailure) -> AdapterError {
    let context = match source.id {
        ValidationId(3) => CheckContext::Attention { layer: 2 },
        ValidationId(8) => CheckContext::Router { layer: 5 },
        id => return AdapterError::UnknownId { id, source },
    };
    AdapterError::Failed { context, source }
}

#[test]
fn typed_adapter_maps_id_not_name() {
    let failure = ExecutionValidationFailure {
        id: ValidationId(8),
        name: "attention-looking-name".into(),
        lane: 1,
        observed_bits: 1.0f32.to_bits(),
    };
    let runner_error = RunnerError::model_validation(map_failure(failure));

    let source = runner_error
        .source()
        .and_then(|source| source.downcast_ref::<AdapterError>())
        .expect("RunnerError must retain the typed adapter source");
    assert!(matches!(
        source,
        AdapterError::Failed {
            context: CheckContext::Router { layer: 5 },
            source: ExecutionValidationFailure {
                id: ValidationId(8),
                lane: 1,
                ..
            },
        }
    ));

    let unknown = map_failure(ExecutionValidationFailure {
        id: ValidationId(99),
        name: "router".into(),
        lane: 0,
        observed_bits: f32::NAN.to_bits(),
    });
    assert!(matches!(
        unknown,
        AdapterError::UnknownId {
            id: ValidationId(99),
            ..
        }
    ));
}
