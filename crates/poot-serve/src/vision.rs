//! The image side of a vision-language checkpoint: the vision tower's driver on a thread of its own,
//! beside the engine thread that serves the text tower.
//!
//! A chat request with images is lowered here to what the engine already takes: the prompt's token ids
//! and the request inputs the text model declares, the embedding rows that replace its `<image>`
//! placeholders among them ([`PromptInput`]). The
//! tower runs one `Encode` step per frame, so a request's images are encoded off the engine thread and
//! the engine's decode loop never waits for them. An executor is not `Send`, so the driver is built
//! on the worker thread and only the requests and their results cross it.
//!
//! [`PromptInput`]: crate::types::PromptInput

use std::io::Cursor;
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use base64::Engine as _;
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, DriverOptions, PreparedSetLimits, open_executor, program_retention,
};
use poot_llm::{ChatTurn, ImageCheckpoint, ImageFrontEnd, ImageRequestError, MultimodalPrompt};

use crate::api::RefusedRequest;
use crate::batch::engine::panic_reason;
use crate::engine_plan::{Death, EngineHealth};
use crate::lifecycle::{ClientGone, WorkerRegistry, drain_cancellation_requested};
use crate::types::GenerationError;

/// Most images one request may carry; more is a 400 before any base64 is decoded.
pub(crate) const MAX_IMAGES_PER_REQUEST: usize = 16;
/// Most pixels the images of one request may hold together, summed from their headers; more is a 400
/// before any pixel is decoded. A decoded frame is widened to f32 (4x for idefics3), so this bounds the
/// memory a request can ask of the front end.
pub(crate) const MAX_IMAGE_PIXELS_PER_REQUEST: u64 = 64 * 1024 * 1024;
/// How long a request waits for the vision thread to answer before it fails.
pub(crate) const IMAGE_PREPARE_DEADLINE: Duration = Duration::from_secs(300);
/// How often the wait for the vision thread asks whether the client has hung up.
const REPLY_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// One message of a request that holds images: its role, how many of the request's images it carries
/// and its text.
pub(crate) struct ImageTurn {
    pub(crate) role: String,
    pub(crate) images: usize,
    pub(crate) text: String,
}

/// What the vision thread is asked to take of a request: its images, as encoded bytes in the order of
/// the turns, and the conversation's turns that carry them.
struct VisionJob {
    images: Vec<Vec<u8>>,
    turns: Vec<ImageTurn>,
    reply: Sender<Result<MultimodalPrompt, ImageRequestError>>,
}

/// What a connection thread holds of the image front end: the queue to the vision thread.
pub(crate) struct ImageClient {
    jobs: Sender<VisionJob>,
    /// How long a request waits for the vision thread's answer.
    deadline: Duration,
}

/// Why a request's images did not become prompt rows.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ImageError {
    /// The images or the question are the client's to fix.
    #[error(transparent)]
    Refused(#[from] RefusedRequest),
    /// The vision thread is gone, hung past the deadline, or the tower failed.
    #[error("the image front end failed: {0}")]
    Failed(String),
    /// The wait ended because the client hung up or the server is draining.
    #[error(transparent)]
    Interrupted(GenerationError),
}

/// The driver options of the vision tower: a frame is one `Encode` step, so the capacity and the
/// prefill piece are both the frame's patches.
fn vision_options(patches: usize) -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    let patches = NonZeroUsize::new(patches).expect("a frame has patches");
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: patches,
        prefill_chunk: patches,
        trace: poot_llm::driver::serving_trace_limits(patches),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

/// Start the vision thread for `checkpoint` on `backend` and wait for its driver to open, so a device
/// that refuses the tower is a startup error before the listener binds.
pub(crate) fn spawn_image_front_end(
    workers: &mut WorkerRegistry,
    health: &EngineHealth,
    checkpoint: std::sync::Arc<dyn ImageCheckpoint>,
    backend: BackendChoice,
) -> Result<ImageClient> {
    spawn_vision_thread(workers, health, IMAGE_PREPARE_DEADLINE, move || {
        let executor = open_executor(backend).map_err(anyhow::Error::from)?;
        checkpoint
            .front_end((executor, vision_options(checkpoint.patches())))
            .map_err(anyhow::Error::from)
    })
}

/// Run a vision thread whose front end `open` builds on that thread (an executor is not `Send`). The
/// thread has the engine's death contract: a panic in the front end publishes `Dead` to `/health`, and
/// every request waiting on the thread fails.
pub(crate) fn spawn_vision_thread(
    workers: &mut WorkerRegistry,
    health: &EngineHealth,
    deadline: Duration,
    open: impl FnOnce() -> Result<Box<dyn ImageFrontEnd>> + Send + 'static,
) -> Result<ImageClient> {
    let (jobs_tx, jobs_rx) = mpsc::channel::<VisionJob>();
    let (started_tx, started_rx) = mpsc::channel::<Result<()>>();
    let health = health.clone();
    workers
        .spawn("poot-vision", move || {
            let mut front_end = match open() {
                Ok(front_end) => front_end,
                Err(error) => {
                    let _ = started_tx.send(Err(error));
                    return;
                }
            };
            if started_tx.send(Ok(())).is_err() {
                return;
            }
            let run = catch_unwind(AssertUnwindSafe(|| {
                run_vision_thread(front_end.as_mut(), &jobs_rx);
            }));
            if let Err(payload) = run {
                let reason = panic_reason(payload.as_ref());
                tracing::error!(reason, "vision thread panicked");
                health.die(Death::Panicked(reason));
            }
        })
        .context("spawn the vision thread")?;
    started_rx
        .recv()
        .context("the vision thread ended during startup")?
        .context("open the vision tower")?;
    Ok(ImageClient {
        jobs: jobs_tx,
        deadline,
    })
}

/// Lower jobs until every sender is gone.
fn run_vision_thread(front_end: &mut dyn ImageFrontEnd, jobs: &mpsc::Receiver<VisionJob>) {
    while let Ok(job) = jobs.recv() {
        let images: Vec<&[u8]> = job.images.iter().map(Vec::as_slice).collect();
        let turns: Vec<_> = job
            .turns
            .iter()
            .map(|turn| ChatTurn {
                role: &turn.role,
                images: turn.images,
                text: &turn.text,
            })
            .collect();
        let prepared = front_end.prepare(&images, &turns);
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

/// The bytes of a request's images, admitted against the per-request caps one image at a time: the
/// count is checked before any payload is decoded, and each image's header size is added to the running
/// pixel total before the next payload is decoded, so no pixel is decoded for a request that is refused.
/// `decode` turns one `image_url` into its bytes.
pub(crate) fn admit_images(
    urls: &[String],
    mut decode: impl FnMut(&str) -> Result<Vec<u8>, RefusedRequest>,
) -> Result<Vec<Vec<u8>>, RefusedRequest> {
    if urls.len() > MAX_IMAGES_PER_REQUEST {
        return Err(RefusedRequest(format!(
            "the request carries {} images: the limit is {MAX_IMAGES_PER_REQUEST}",
            urls.len()
        )));
    }
    let mut pixels = 0u64;
    urls.iter()
        .map(|url| {
            let bytes = decode(url)?;
            pixels = pixels.saturating_add(header_pixels(&bytes)?);
            if pixels > MAX_IMAGE_PIXELS_PER_REQUEST {
                return Err(RefusedRequest(format!(
                    "the request's images hold at least {pixels} pixels: the limit is {MAX_IMAGE_PIXELS_PER_REQUEST}"
                )));
            }
            Ok(bytes)
        })
        .collect()
}

/// The pixel count an image's header declares, read without decoding a pixel.
fn header_pixels(bytes: &[u8]) -> Result<u64, RefusedRequest> {
    let refuse = |reason: String| RefusedRequest(format!("image_url: {reason}"));
    let (width, height) = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| refuse(format!("the image cannot be read: {e}")))?
        .into_dimensions()
        .map_err(|e| refuse(format!("the image header does not decode: {e}")))?;
    Ok(u64::from(width) * u64::from(height))
}

impl ImageClient {
    /// Read the bytes of `urls` (in the order of the turns) under the per-request caps and have the
    /// vision thread lower the conversation `turns` and its images to a prompt and its rows. The wait
    /// for the thread ends when it answers, dies, exceeds the deadline, the client hangs up or the
    /// server's drain expires.
    pub(crate) fn prepare(
        &self,
        urls: &[String],
        turns: Vec<ImageTurn>,
        client_gone: ClientGone<'_>,
    ) -> Result<MultimodalPrompt, ImageError> {
        let images = admit_images(urls, image_bytes)?;
        let (reply, answer) = mpsc::channel();
        self.jobs
            .send(VisionJob {
                images,
                turns,
                reply,
            })
            .map_err(|_| ImageError::Failed("the vision thread is gone".into()))?;
        let prepared = self.await_answer(&answer, client_gone)?;
        prepared.map_err(|error| match error {
            ImageRequestError::Refused(reason) => RefusedRequest(reason).into(),
            failed @ ImageRequestError::Failed(_) => ImageError::Failed(failed.to_string()),
        })
    }

    fn await_answer(
        &self,
        answer: &mpsc::Receiver<Result<MultimodalPrompt, ImageRequestError>>,
        client_gone: ClientGone<'_>,
    ) -> Result<Result<MultimodalPrompt, ImageRequestError>, ImageError> {
        let deadline = Instant::now() + self.deadline;
        loop {
            if drain_cancellation_requested() {
                return Err(ImageError::Interrupted(GenerationError::ServerDrainExpired));
            }
            match answer.recv_timeout(REPLY_POLL_INTERVAL) {
                Ok(prepared) => return Ok(prepared),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ImageError::Failed(
                        "the vision thread ended mid-request".into(),
                    ));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if client_gone() {
                        return Err(ImageError::Interrupted(GenerationError::ClientCancelled));
                    }
                    if Instant::now() >= deadline {
                        return Err(ImageError::Failed(format!(
                            "the vision thread did not answer within {} s",
                            self.deadline.as_secs()
                        )));
                    }
                }
            }
        }
    }
}
