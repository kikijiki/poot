//! One text prompt generated to the end of its context window: the shape of a command-line tool, as
//! opposed to a server's many concurrent requests. [`generate_text`] frames the prompt for the model
//! (its own chat template, or the raw text), sizes a driver to the model's window, and streams the text
//! through a sink.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_executor::Executor;
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};

use crate::GenerationControl;
use crate::core::sampler::Sampler;
use crate::driver::error::{DriverError, InvalidRequest};
use crate::driver::stops::StopScanner;
use crate::driver::{
    DraftPolicy, Driver, DriverOptions, GenerateRequest, Generation, ModelHandle,
    PreparedSetLimits, program_retention,
};
use crate::error::TextError;
use crate::text::tokenize::TextCodec;

/// Tokens per prefill step, rounded up to the model's prefill granule.
const PREFILL_CHUNK: usize = 128;

/// How the next token is chosen: the sampling flags of a command line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// 0 is greedy, the exact-token convention.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// The multiplicative repetition penalty over the prompt and the output; 1 is off.
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Sampling {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        seed: 0,
    };

    fn sampler(&self) -> Sampler {
        Sampler::new(self.temperature, self.top_k, self.top_p, self.seed)
            .with_min_p(self.min_p)
            .with_penalties(self.repeat_penalty, 0.0, 0.0)
    }
}

/// What to generate from.
#[derive(Clone, Debug)]
pub struct TextRequest {
    pub prompt: String,
    /// A system turn ahead of the prompt; needs the chat template, so it is refused when the prompt is
    /// used as written.
    pub system: Option<String>,
    /// Use `prompt` as written even when the model ships a chat template.
    pub raw: bool,
    /// Stop strings beyond the chat template's own turn end.
    pub stops: Vec<String>,
    /// At most this many new tokens; `None` runs to EOS, a stop string or the end of the window.
    pub max_new: Option<usize>,
    /// The window, in positions, that the prompt and the new tokens must fit, and the capacity the
    /// driver is sized to; `None` is the model's `max_positions`.
    pub ctx: Option<usize>,
    pub sampling: Sampling,
    /// Keep going through the model's end-of-sequence token.
    pub ignore_eos: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum OneShotError {
    #[error("encoding the prompt: {0}")]
    Encode(#[source] TextError),
    #[error(
        "a system prompt needs the model's chat template, and this prompt is used as written (the model ships no template, or the template was switched off)"
    )]
    SystemWithoutTemplate,
    #[error(transparent)]
    Driver(DriverError),
}

impl From<DriverError> for OneShotError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl From<InvalidRequest> for OneShotError {
    fn from(error: InvalidRequest) -> Self {
        Self::Driver(error.into())
    }
}

/// A request as the model reads it.
#[derive(Debug, PartialEq, Eq)]
struct Framed {
    /// The text that is encoded.
    prompt: String,
    /// Whether `prompt` is the caller's own text, which the output continues, rather than a chat turn.
    continued: bool,
    /// The chat template's turn-end markers: they end the run and never reach the sink.
    turn_ends: Vec<String>,
    /// The turn ends, then the caller's stops.
    stops: Vec<String>,
}

/// A model that ships a chat template answers `request.prompt` as a user turn of it; a base model, or
/// `request.raw`, continues the text.
fn frame(text: &TextCodec, request: &TextRequest) -> Result<Framed, OneShotError> {
    let chat = if request.raw {
        None
    } else {
        text.render_user_turn(request.system.as_deref(), &request.prompt)
    };
    match (chat, &request.system) {
        (Some(rendered), _) => Ok(Framed {
            prompt: rendered.prompt,
            continued: false,
            turn_ends: rendered.stops.clone(),
            stops: [rendered.stops, request.stops.clone()].concat(),
        }),
        (None, Some(_)) => Err(OneShotError::SystemWithoutTemplate),
        (None, None) => Ok(Framed {
            prompt: request.prompt.clone(),
            continued: true,
            turn_ends: Vec::new(),
            stops: request.stops.clone(),
        }),
    }
}

/// A request sized to its model: the prompt framed and encoded, the tokens it may generate and the
/// capacity the driver needs.
struct Plan {
    framed: Framed,
    prompt: Vec<u32>,
    max_new: usize,
    capacity: NonZeroUsize,
}

fn plan(handle: &ModelHandle, request: &TextRequest) -> Result<Plan, OneShotError> {
    let framed = frame(handle.text(), request)?;
    let prompt = handle
        .text()
        .encode(&framed.prompt)
        .map_err(OneShotError::Encode)?;
    let max_positions = handle.config().max_positions;
    let window = request.ctx.unwrap_or(max_positions);
    if window > max_positions {
        return Err(InvalidRequest::Capacity {
            prompt: prompt.len(),
            max_new: request.max_new.unwrap_or(1),
            max: max_positions,
        }
        .into());
    }
    let max_new = GenerateRequest::max_new_within(prompt.len(), window, request.max_new)?;
    // An explicit window is the capacity asked for; otherwise only what the run can use.
    let capacity = request.ctx.unwrap_or(prompt.len() + max_new);
    let capacity = NonZeroUsize::new(capacity).expect("the prompt is nonempty");
    Ok(Plan {
        framed,
        prompt,
        max_new,
        capacity,
    })
}

fn options(capacity: NonZeroUsize, granule: NonZeroUsize) -> DriverOptions {
    let chunk = NonZeroUsize::new(PREFILL_CHUNK.div_ceil(granule.get()) * granule.get())
        .expect("a granule multiple is nonzero");
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity,
        prefill_chunk: chunk,
        max_trace_tokens: chunk,
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
    }
}

/// `request` generated on `executor`, each piece of text passed to `sink` as it is decoded. A run that
/// continues the caller's text begins by passing that text to `sink`, so the sink sees the whole
/// continued text; a chat answer is the model's reply alone. The prompt and the new tokens must fit the
/// window; one that does not is [`InvalidRequest::Capacity`], refused before the executor does any work.
pub fn generate_text(
    handle: &Arc<ModelHandle>,
    executor: Box<dyn Executor>,
    request: &TextRequest,
    sink: &mut dyn FnMut(&str),
) -> Result<Generation, OneShotError> {
    let Plan {
        framed,
        prompt,
        max_new,
        capacity,
    } = plan(handle, request)?;
    let mut driver = Driver::new(
        Arc::clone(handle),
        executor,
        options(capacity, handle.config().prefill_granule),
    )?
    .with_phase_timing();
    if framed.continued {
        sink(&framed.prompt);
    }
    // The driver hands the sink the piece that completes a stop string. A chat template's turn end is
    // markup, not reply, so it is cut out; a caller's own `-r` string stays in the text, as in llama-cli.
    let mut turn_ends = StopScanner::new(framed.turn_ends);
    let generation = driver.generate(
        GenerateRequest {
            prompt,
            max_new,
            sampler: request.sampling.sampler(),
            stops: framed.stops,
            ignore_eos: request.ignore_eos,
        },
        &mut |_: u32, piece: &str| {
            sink(&turn_ends.push(piece));
            GenerationControl::Continue(())
        },
    )?;
    sink(&turn_ends.finish());
    Ok(generation)
}

#[cfg(test)]
mod tests {
    use poot_models::chat::ChatFormat;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    use super::*;
    use crate::text::tokenize::ChatTemplate;

    const TEMPLATE: &str = "{% for m in messages %}[{{ m['role'] }}:{{ m['content'] }}]{% endfor %}{% if add_generation_prompt %}[assistant:{% endif %}";

    fn codec(template: Option<&str>) -> TextCodec {
        let model = WordLevel::builder()
            .vocab([("<unk>".to_string(), 0u32)].into_iter().collect())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel model");
        TextCodec::new(
            Tokenizer::new(model),
            None,
            None,
            0,
            ChatFormat::ChatML,
            ChatTemplate {
                template: template.map(str::to_string),
                bos_token: None,
                eos_token: None,
            },
        )
    }

    fn request(raw: bool) -> TextRequest {
        TextRequest {
            prompt: "Hi".into(),
            system: None,
            raw,
            stops: vec!["END".into()],
            max_new: None,
            ctx: None,
            sampling: Sampling::GREEDY,
            ignore_eos: false,
        }
    }

    /// A model with a template is asked in its chat format, stopping at the template's turn end and
    /// then the caller's stops. Mutation: frame a templated model as raw text.
    #[test]
    fn a_templated_model_is_asked_as_a_chat_turn_with_the_callers_stops_after_its_own() {
        let framed = frame(&codec(Some(TEMPLATE)), &request(false)).unwrap();
        assert_eq!(framed.prompt, "[user:Hi][assistant:");
        assert!(!framed.continued);
        assert_eq!(framed.stops.last().map(String::as_str), Some("END"));
        assert!(framed.stops.len() > 1, "{:?}", framed.stops);
    }

    /// `raw` forces the caller's text even for a templated model, and the template's turn end no longer
    /// applies. Mutation: ignore `raw`.
    #[test]
    fn raw_continues_the_text_of_a_templated_model_with_only_the_callers_stops() {
        let framed = frame(&codec(Some(TEMPLATE)), &request(true)).unwrap();
        assert_eq!(
            framed,
            Framed {
                prompt: "Hi".into(),
                continued: true,
                turn_ends: Vec::new(),
                stops: vec!["END".into()],
            }
        );
    }

    /// A base model (no template) continues the text without `raw`. Mutation: fall back to the family's
    /// hardcoded chat format for a model with no template.
    #[test]
    fn a_base_model_continues_the_text() {
        let framed = frame(&codec(None), &request(false)).unwrap();
        assert_eq!(
            framed,
            Framed {
                prompt: "Hi".into(),
                continued: true,
                turn_ends: Vec::new(),
                stops: vec!["END".into()],
            }
        );
    }

    /// A system prompt reaches the template, and is refused where there is no template to carry it
    /// (a base model, or `raw`) rather than dropped. Mutation: drop the refusal arm (the prompt runs
    /// without its system turn).
    #[test]
    fn a_system_prompt_is_framed_by_the_template_and_refused_without_one() {
        let with_system = |raw| TextRequest {
            system: Some("Be terse.".into()),
            ..request(raw)
        };
        let framed = frame(&codec(Some(TEMPLATE)), &with_system(false)).unwrap();
        assert_eq!(framed.prompt, "[system:Be terse.][user:Hi][assistant:");
        assert!(matches!(
            frame(&codec(None), &with_system(false)),
            Err(OneShotError::SystemWithoutTemplate)
        ));
        assert!(matches!(
            frame(&codec(Some(TEMPLATE)), &with_system(true)),
            Err(OneShotError::SystemWithoutTemplate)
        ));
    }

    /// The sampling flags reach the sampler: 0 is greedy (the exact-token convention), a positive
    /// temperature is not, and each knob lands in its own field. Mutations: map `temperature` to a
    /// constant 0; swap `top_k` and `top_p`; drop `min_p`; drop the repeat penalty.
    #[test]
    fn sampling_flags_reach_the_sampler() {
        assert!(Sampling::GREEDY.sampler().is_greedy());
        let flags = Sampling {
            temperature: 0.7,
            top_k: 11,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.3,
            seed: 5,
        };
        let sampler = flags.sampler();
        assert!(!sampler.is_greedy());
        assert_eq!(
            (
                sampler.temperature,
                sampler.top_k,
                sampler.top_p,
                sampler.min_p
            ),
            (0.7, 11, 0.9, 0.05)
        );
        assert_eq!(sampler.repetition_penalty, 1.3);
        // Without a penalty the same flags are a plain temperature draw the device can take.
        let plain = Sampling {
            repeat_penalty: 1.0,
            ..flags
        };
        assert!(plain.sampler().is_simple_temperature());
        assert!(!sampler.is_simple_temperature());
    }

    /// The seed is the draw: one seed repeats its tokens and a different seed is another stream.
    /// Mutation: ignore `seed`.
    #[test]
    fn the_seed_decides_the_draw() {
        let draw = |seed| {
            let mut sampler = Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                seed,
            }
            .sampler();
            let flat = vec![0.0f32; 64];
            (0..16)
                .map(|_| sampler.pick(&flat).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::driver::device_tests::{NO_EOS, Weights, handle};

    /// Seven one-character words of the fixture's vocabulary.
    const PROMPT: &str = "b c d e f g h";

    fn request(max_new: Option<usize>, ctx: Option<usize>) -> TextRequest {
        TextRequest {
            prompt: PROMPT.into(),
            system: None,
            raw: false,
            stops: Vec::new(),
            max_new,
            ctx,
            sampling: Sampling::GREEDY,
            ignore_eos: false,
        }
    }

    /// With no `-n` and no `-c` a run is sized to the rest of the model's own window, and the driver
    /// needs only the positions the run can use. Mutation: size the run to the whole window, or leave
    /// the prompt out of the subtraction.
    #[test]
    fn the_default_run_takes_the_rest_of_the_models_window() {
        let handle = handle(Weights::Random, NO_EOS);
        let plan = plan(&handle, &request(None, None)).unwrap();
        assert_eq!(plan.prompt.len(), 7);
        assert_eq!(plan.max_new, handle.config().max_positions - 7);
        assert_eq!(plan.capacity.get(), handle.config().max_positions);
    }

    /// `-c` bounds the window and sizes the capacity; `-n` within it caps the run, and `-n` past it is
    /// the typed `Capacity` refusal before any device work. Mutations: ignore `ctx`; compare `<` where
    /// the bound is `<=`.
    #[test]
    fn a_context_size_bounds_the_run_and_an_overrun_is_a_typed_refusal() {
        let handle = handle(Weights::Random, NO_EOS);
        let sized = plan(&handle, &request(None, Some(16))).unwrap();
        assert_eq!((sized.max_new, sized.capacity.get()), (9, 16));
        let capped = plan(&handle, &request(Some(4), Some(16))).unwrap();
        assert_eq!((capped.max_new, capped.capacity.get()), (4, 16));
        let unsized_ = plan(&handle, &request(Some(4), None)).unwrap();
        assert_eq!((unsized_.max_new, unsized_.capacity.get()), (4, 11));
        let Err(OneShotError::Driver(DriverError::InvalidRequest(refused))) =
            plan(&handle, &request(Some(10), Some(16)))
        else {
            panic!("an overrun was not refused as a typed Capacity");
        };
        assert_eq!(
            refused,
            InvalidRequest::Capacity {
                prompt: 7,
                max_new: 10,
                max: 16
            }
        );
    }

    /// `-n` equal to the room left is the exact fit and runs; an empty room (the window is the prompt)
    /// with the default `-n` is the typed refusal. Mutations: `<` for `<=` in `max_new_within`'s
    /// explicit bound (the exact fit is refused); `room >= 0` for `room > 0` in its default arm (the
    /// empty room plans zero tokens).
    #[test]
    fn an_exact_fit_runs_and_an_empty_room_is_refused() {
        let handle = handle(Weights::Random, NO_EOS);
        let exact = plan(&handle, &request(Some(9), Some(16))).unwrap();
        assert_eq!((exact.max_new, exact.capacity.get()), (9, 16));
        let Err(OneShotError::Driver(DriverError::InvalidRequest(refused))) =
            plan(&handle, &request(None, Some(7)))
        else {
            panic!("an empty room was not refused as a typed Capacity");
        };
        assert_eq!(
            refused,
            InvalidRequest::Capacity {
                prompt: 7,
                max_new: 1,
                max: 7
            }
        );
    }

    /// A `-c` above the model's own window is the typed refusal, from the plan, before an executor
    /// opens. Mutation: drop the `window > max_positions` check (the plan succeeds).
    #[test]
    fn a_context_above_the_models_window_is_refused_in_the_plan() {
        let handle = handle(Weights::Random, NO_EOS);
        let max = handle.config().max_positions;
        let Err(OneShotError::Driver(DriverError::InvalidRequest(refused))) =
            plan(&handle, &request(None, Some(max + 1)))
        else {
            panic!("a context above max_positions was not refused");
        };
        assert_eq!(
            refused,
            InvalidRequest::Capacity {
                prompt: 7,
                max_new: 1,
                max
            }
        );
    }
}
