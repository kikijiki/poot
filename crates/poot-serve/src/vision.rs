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
use poot_llm::{ChatTurn, ImageCheckpoint, ImageFrontEnd, ImageRequestError, MultimodalPrompt};

use crate::api::RefusedRequest;
use crate::lifecycle::WorkerRegistry;

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
    checkpoint: std::sync::Arc<dyn ImageCheckpoint>,
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
            run_vision_thread(front_end.as_mut(), &jobs_rx);
        })
        .context("spawn the vision thread")?;
    started_rx
        .recv()
        .context("the vision thread ended during startup")?
        .context("open the vision tower")?;
    Ok(ImageClient { jobs: jobs_tx })
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

impl ImageClient {
    /// Read the bytes of `urls` (in the order of the turns) and have the vision thread lower the conversation
    /// `turns` and its images to a prompt and its rows.
    pub(crate) fn prepare(
        &self,
        urls: &[String],
        turns: Vec<ImageTurn>,
    ) -> Result<MultimodalPrompt, ImageError> {
        let images = urls
            .iter()
            .map(|url| image_bytes(url))
            .collect::<Result<Vec<_>, _>>()?;
        let (reply, answer) = mpsc::channel();
        self.jobs
            .send(VisionJob {
                images,
                turns,
                reply,
            })
            .map_err(|_| ImageError::Failed("the vision thread is gone".into()))?;
        let prepared = answer
            .recv()
            .map_err(|_| ImageError::Failed("the vision thread ended mid-request".into()))?;
        prepared.map_err(|error| match error {
            ImageRequestError::Refused(reason) => RefusedRequest(reason).into(),
            failed @ ImageRequestError::Failed(_) => ImageError::Failed(failed.to_string()),
        })
    }
}
