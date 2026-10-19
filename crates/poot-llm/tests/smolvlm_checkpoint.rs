//! The real `smolvlm-256m` checkpoint through the front end on a device: rows of the card's
//! acceptance (SC-001, SC-002). They need the local checkpoint and a device, so they are ignored
//! (`just test` hides every checkpoint test) and run by exact name under `/tmp/poot-gpu.lock`.

use std::sync::Arc;

use poot_models::multimodal::front_end::{ChatTurn, FrontEndError, Image};
use poot_models::registry::Registry;
use poot_runtime_common::DeviceBackend;

use poot_llm::driver::error::DriverError;
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions, ModelHandle,
    Serving, open_executor,
};
use poot_llm::{FrontEndSession, GenerationControl, Sampler, tower_options};

/// A loaded SmolVLM: its front end on a vision driver and the text decoder on another.
struct Smolvlm {
    session: FrontEndSession,
    text: Driver<Serving>,
    /// The vision tower's name, for its driver's stats.
    tower: poot_models::multimodal::front_end::TowerId,
}

fn text_options(capacity: usize, chunk: usize) -> DriverOptions {
    use std::num::{NonZeroU64, NonZeroUsize};

    use poot_graph_plan::{CompileLimits, CompileOptions};

    use poot_llm::driver::{DraftPolicy, PreparedSetLimits, program_retention};

    let compile = CompileOptions::new(CompileLimits::STANDARD);
    let nz = |n: usize| NonZeroUsize::new(n).unwrap();
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(chunk),
        trace: poot_llm::driver::serving_trace_limits(nz(chunk)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

/// The checkpoint on `choice`'s device, or `None` when the checkpoint or the device is absent.
fn load(choice: BackendChoice) -> Option<Smolvlm> {
    let dir = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))?;
    let backend = match choice {
        BackendChoice::Wgpu => DeviceBackend::Wgpu,
        BackendChoice::Rocm => DeviceBackend::Rocm,
        BackendChoice::Ptx => DeviceBackend::Ptx,
        other => unreachable!("{other:?} has no smolvlm row"),
    };
    let open = || poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice));
    let (vision, text) = (open()?, open()?);
    let handle = Arc::new(
        ModelHandle::load(&dir, &Registry::builtin().unwrap(), &LoadOptions::default())
            .expect("load smolvlm-256m"),
    );
    let mut vision = Some(vision);
    let session = FrontEndSession::open(Arc::clone(&handle), |_, positions| {
        Ok::<_, DriverError>((vision.take().expect("one tower"), tower_options(positions)))
    })
    .expect("open the vision tower")
    .expect("smolvlm has a front end");
    let tower = handle.towers()[0].0;
    Some(Smolvlm {
        session,
        text: Driver::new(handle, text, text_options(4096, 256))
            .expect("a text driver")
            .serve(Layout::Contiguous),
        tower,
    })
}

impl Smolvlm {
    /// The greedy `max_new` tokens for `images` and the question of one user turn.
    fn caption(
        &mut self,
        images: &[&Image],
        question: &str,
        max_new: usize,
    ) -> Result<Vec<u32>, FrontEndError> {
        use poot_models::multimodal::front_end::Media;

        let mut media: Vec<Media<'_>> = images.iter().map(|image| Media::Image(image)).collect();
        let prepared = self
            .session
            .prepare(&[ChatTurn::user(images.len(), question)], &mut media)?;
        Ok(self
            .text
            .generate_with_inputs(
                GenerateRequest {
                    prompt: prepared.prompt,
                    max_new,
                    sampler: Sampler::greedy(),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                prepared.inputs,
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .expect("caption")
            .tokens)
    }
}

fn pattern(h: usize, w: usize) -> Image {
    Image::from_rgb8(
        h,
        w,
        (0..h * w * 3).map(|i| ((i * 7) % 256) as u8).collect(),
    )
    .unwrap()
}

fn solid(h: usize, w: usize) -> Image {
    Image::from_rgb8(h, w, vec![128; h * w * 3]).unwrap()
}

fn stripes(h: usize, w: usize) -> Image {
    let rgb = (0..h * w)
        .flat_map(|i| {
            let v = if (i / w / 8).is_multiple_of(2) {
                0
            } else {
                255
            };
            [v, v, v]
        })
        .collect();
    Image::from_rgb8(h, w, rgb).unwrap()
}

/// The tokens the pre-driver engine (base commit 01aa1d40f) generated for the 96x96 pattern image and this
/// question on wgpu (12 greedy tokens: " The image displays a series of parallel, horizontal
/// stripes that appear").
const PATTERN_TOKENS: [u32; 12] = [
    378, 2443, 10637, 253, 3086, 282, 7623, 28, 10691, 23158, 338, 1972,
];

/// SC-001: the SmolVLM fixture (an image and a prompt) generates through `Driver` with the front
/// end producing the step inputs; the tokens equal the literals recorded from the pre-driver engine at the
/// base commit (tier 3). Mutation: feed zeroed image embeddings; the tokens change and the row
/// goes red.
fn caption_matches_the_recorded_vlm_runner_tokens(choice: BackendChoice) {
    let Some(mut vlm) = load(choice) else {
        return;
    };
    let tokens = vlm
        .caption(&[&pattern(96, 96)], "Can you describe this image?", 12)
        .expect("caption");
    assert_eq!(tokens, PATTERN_TOKENS);
}

#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + wgpu (17 vision frames, ~1200-token prefill)"]
fn caption_matches_the_recorded_vlm_runner_tokens_wgpu() {
    caption_matches_the_recorded_vlm_runner_tokens(BackendChoice::Wgpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + ROCm (17 vision frames, ~1200-token prefill)"]
fn caption_matches_the_recorded_vlm_runner_tokens_rocm() {
    caption_matches_the_recorded_vlm_runner_tokens(BackendChoice::Rocm);
}

#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + a PTX device (17 vision frames, ~1200-token prefill)"]
fn caption_matches_the_recorded_vlm_runner_tokens_ptx() {
    caption_matches_the_recorded_vlm_runner_tokens(BackendChoice::Ptx);
}

/// SC-002: two frames of equal numel and different content in one front end give different
/// embedding rows and different captions, each the recorded pre-driver tokens, with no eviction
/// between them: the frames are bound by value. Mutation: bind `pixels` as a const again; the
/// second frame cannot load (or reuses the first) and the row goes red.
fn two_captions_on_one_front_end_are_not_stale(choice: BackendChoice) {
    let Some(mut vlm) = load(choice) else {
        return;
    };
    let question = "Can you describe this image?";
    let first = vlm.caption(&[&solid(96, 96)], question, 12).unwrap();
    let second = vlm.caption(&[&stripes(96, 96)], question, 12).unwrap();
    assert_eq!(
        first,
        [
            378, 2443, 21559, 253, 2232, 28, 2408, 18029, 28, 15932, 11955, 282
        ]
    );
    assert_eq!(
        second,
        [
            378, 2443, 21559, 253, 2632, 284, 2537, 2443, 282, 253, 2244, 28
        ]
    );
    assert_ne!(first, second, "the second caption reused the first image");
}

#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + wgpu (two captions)"]
fn two_captions_on_one_front_end_are_not_stale_wgpu() {
    two_captions_on_one_front_end_are_not_stale(BackendChoice::Wgpu);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + ROCm (two captions)"]
fn two_captions_on_one_front_end_are_not_stale_rocm() {
    two_captions_on_one_front_end_are_not_stale(BackendChoice::Rocm);
}

#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + a PTX device (two captions)"]
fn two_captions_on_one_front_end_are_not_stale_ptx() {
    two_captions_on_one_front_end_are_not_stale(BackendChoice::Ptx);
}

/// Two images in one turn caption together: each splits into 17 frames, so 34 vision steps and a
/// ~2200-token prompt (2 x 1088 image tokens) go through the one scatter.
#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + wgpu (34 vision frames, ~2200-token prefill)"]
fn two_images_in_one_turn_caption_together() {
    let Some(mut vlm) = load(BackendChoice::Wgpu) else {
        return;
    };
    let tokens = vlm
        .caption(
            &[&solid(96, 96), &stripes(96, 96)],
            "Describe both images.",
            16,
        )
        .expect("caption");
    assert!(!tokens.is_empty(), "produced an empty caption");
}

/// DoS guard: each image tiles into about 17 in-memory frames and the image count is
/// client-controlled, so the total is capped and refused in preprocessing, before the tower runs.
#[test]
#[ignore = "needs the local smolvlm-256m checkpoint + wgpu (bails before any dispatch)"]
fn too_many_images_are_refused_before_the_tower_runs() {
    let Some(mut vlm) = load(BackendChoice::Wgpu) else {
        return;
    };
    // 40 tiny images tile to about 17 frames each: over the 512-frame cap.
    let one = Image::from_rgb8(4, 4, (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect()).unwrap();
    let images: Vec<&Image> = vec![&one; 40];
    let error = vlm
        .caption(&images, "describe", 8)
        .expect_err("an oversized batch is refused");
    assert!(
        matches!(&error, FrontEndError::Refused(reason) if reason.contains("too many image frames")),
        "{error}"
    );
    let tower = vlm.session.towers().driver(vlm.tower);
    assert_eq!(
        tower.expect("the vision tower").stats().steps,
        0,
        "no frame reached the tower"
    );
}
