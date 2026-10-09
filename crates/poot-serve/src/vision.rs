//! The image side of a vision-language checkpoint: the vision tower's driver on a thread of its own,
//! beside the engine thread that serves the text tower.
//!
//! A chat request with images is lowered here to what the engine already takes: the prompt's token ids
//! and the embedding rows that replace its `<image>` placeholders ([`PromptInput::Multimodal`]). The
//! tower runs one `Encode` step per frame, so a request's images are encoded off the engine thread and
//! the engine's decode loop never waits for them. An executor is not `Send`, so the driver is built
//! on the worker thread and only the requests and their results cross it.
//!
//! [`PromptInput::Multimodal`]: crate::types::PromptInput::Multimodal

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::mpsc::{self, Sender};

use anyhow::{Context, Result};
use base64::Engine as _;
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, DriverOptions, PreparedSetLimits, open_executor, program_retention,
};
use poot_llm::front_end::{
    CaptionPrompt, DecodedImage, FrontEndError, Idefics3Checkpoint, ImageFrontEnd, decode_image,
};

use crate::api::RefusedRequest;
use crate::lifecycle::WorkerRegistry;

/// What the engine's driver is asked to take of a request's images: they are decoded already, and the
/// question is the text of the one user message that carries them.
struct VisionJob {
    images: Vec<DecodedImage>,
    question: String,
    reply: Sender<Result<CaptionPrompt, FrontEndError>>,
}

/// What a connection thread holds of the image front end: the queue to the vision thread.
pub(crate) struct ImageClient {
    jobs: Sender<VisionJob>,
}

/// Why a request's images did not become prompt rows.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ImageError {
    /// The images or the question are the client's to fix.
    #[error(transparent)]
    Refused(#[from] RefusedRequest),
    /// The vision thread is gone or the tower failed.
    #[error("the image front end failed: {0}")]
    Failed(String),
}

/// The driver options of the vision tower: a frame is one `Encode` step, so the capacity and the
/// prefill piece are both the frame's patches.
fn vision_options(patches: usize) -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    let patches = NonZeroUsize::new(patches).expect("a frame has patches");
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: patches,
        prefill_chunk: patches,
        max_trace_tokens: patches,
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
    }
}

/// Start the vision thread for `checkpoint` on `backend` and wait for its driver to open, so a device
/// that refuses the tower is a startup error before the listener binds.
pub(crate) fn spawn_image_front_end(
    workers: &mut WorkerRegistry,
    checkpoint: std::sync::Arc<Idefics3Checkpoint>,
    backend: BackendChoice,
) -> Result<ImageClient> {
    let (jobs_tx, jobs_rx) = mpsc::channel::<VisionJob>();
    let (started_tx, started_rx) = mpsc::channel::<Result<()>>();
    workers
        .spawn("poot-vision", move || {
            let opened = open_executor(backend)
                .map_err(anyhow::Error::from)
                .and_then(|executor| {
                    checkpoint
                        .front_end((executor, vision_options(checkpoint.patches())))
                        .map_err(anyhow::Error::from)
                });
            let mut front_end = match opened {
                Ok(front_end) => front_end,
                Err(error) => {
                    let _ = started_tx.send(Err(error));
                    return;
                }
            };
            if started_tx.send(Ok(())).is_err() {
                return;
            }
            run_vision_thread(&mut front_end, &jobs_rx);
        })
        .context("spawn the vision thread")?;
    started_rx
        .recv()
        .context("the vision thread ended during startup")?
        .context("open the vision tower")?;
    Ok(ImageClient { jobs: jobs_tx })
}

/// Lower jobs until every sender is gone.
fn run_vision_thread(front_end: &mut ImageFrontEnd, jobs: &mpsc::Receiver<VisionJob>) {
    while let Ok(job) = jobs.recv() {
        let images: Vec<_> = job.images.iter().map(DecodedImage::as_image).collect();
        let prepared = front_end.prepare(&images, &job.question);
        let _ = job.reply.send(prepared);
    }
}

/// The image bytes of an OpenAI `image_url`: a `data:` URI with base64 content. A URL to fetch is
/// refused; this server makes no outbound requests.
fn image_bytes(url: &str) -> Result<Vec<u8>, RefusedRequest> {
    let refuse = |reason: &str| RefusedRequest(format!("image_url: {reason}"));
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(refuse(
            "only data: URIs are accepted; this server does not fetch URLs",
        ));
    };
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| refuse("the data: URI has no payload"))?;
    if !meta.ends_with(";base64") {
        return Err(refuse("the data: URI is not base64"));
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| refuse(&format!("the base64 payload does not decode: {e}")))
}

impl ImageClient {
    /// Decode `urls` and have the vision thread lower them with `question` to a prompt and its rows.
    pub(crate) fn prepare(
        &self,
        urls: &[String],
        question: &str,
    ) -> Result<CaptionPrompt, ImageError> {
        let images = urls
            .iter()
            .map(|url| {
                let bytes = image_bytes(url)?;
                decode_image(&bytes).map_err(|e| ImageError::Refused(RefusedRequest(e.to_string())))
            })
            .collect::<Result<Vec<_>, ImageError>>()?;
        let (reply, answer) = mpsc::channel();
        self.jobs
            .send(VisionJob {
                images,
                question: question.to_string(),
                reply,
            })
            .map_err(|_| ImageError::Failed("the vision thread is gone".into()))?;
        let prepared = answer
            .recv()
            .map_err(|_| ImageError::Failed("the vision thread ended mid-request".into()))?;
        prepared.map_err(|error| match error {
            FrontEndError::Image(_)
            | FrontEndError::ImageTooLarge { .. }
            | FrontEndError::NoImage
            | FrontEndError::TooManyFrames { .. }
            | FrontEndError::Placeholders { .. } => RefusedRequest(error.to_string()).into(),
            other => ImageError::Failed(other.to_string()),
        })
    }
}
