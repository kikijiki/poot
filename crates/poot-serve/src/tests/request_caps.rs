//! Per-request content totals (POOT-1202): the image count and pixel caps, the `logit_bias` and `stop`
//! caps, the supervised vision thread and the random default seed. Every row drives the production
//! handlers (`handle_chat`, `handle_completion`) over a fake behind the real queue or the real vision
//! thread, so a refusal is told from an admission by what reached the fake.

use std::cell::Cell;
use std::io::Cursor;

use poot_llm::{ChatTurn, ImageFrontEnd, ImageRequestError, MultimodalPrompt};

use super::*;
use crate::api::{MAX_LOGIT_BIAS_ENTRIES, MAX_STOP_BYTES, MAX_TOTAL_STOP_BYTES, RefusedRequest};
use crate::engine_plan::{Death, EngineState};
use crate::handlers::handle_completion;
use crate::vision::{
    ImageError, MAX_IMAGE_PIXELS_PER_REQUEST, MAX_IMAGES_PER_REQUEST, admit_images,
    spawn_vision_thread,
};

/// What the fake front end does with a request that reaches it.
enum Behaviour {
    /// Refuse it, as a front end does a request it cannot read.
    Refuse,
    /// Panic, as a front end with a bug does.
    Panic,
    /// Never answer until the sender is dropped.
    Hang(mpsc::Receiver<()>),
}

/// A front end that counts the requests it is given and never builds a prompt.
struct FakeFrontEnd {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
}

impl ImageFrontEnd for FakeFrontEnd {
    fn prepare(
        &mut self,
        _images: &[&[u8]],
        _turns: &[ChatTurn<'_>],
    ) -> Result<MultimodalPrompt, ImageRequestError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match &self.behaviour {
            Behaviour::Refuse => Err(ImageRequestError::Refused("the fake refuses".into())),
            Behaviour::Panic => panic!("the fake vision tower broke"),
            Behaviour::Hang(gate) => {
                let _ = gate.recv();
                Err(ImageRequestError::Refused("released".into()))
            }
        }
    }
}

/// A served model whose image front end is a [`FakeFrontEnd`] on a real vision thread, with the health
/// the thread publishes to and the count of requests the front end was given.
struct VisionRig {
    served: Arc<Served>,
    _queue: Receiver<EngineMsg>,
    health: EngineHealth,
    calls: Arc<AtomicUsize>,
    _workers: WorkerRegistry,
}

fn vision_rig(behaviour: Behaviour, deadline: Duration) -> VisionRig {
    let (mut served, queue) = served_over(tiny_model(), 1);
    let health = EngineHealth::serving_for_test("fake", 1, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut workers = WorkerRegistry::new();
    let front_end_calls = Arc::clone(&calls);
    let client = spawn_vision_thread(&mut workers, &health, deadline, move || {
        Ok(Box::new(FakeFrontEnd {
            behaviour,
            calls: front_end_calls,
        }))
    })
    .expect("the fake front end opens");
    Arc::get_mut(&mut served).expect("the only handle").images = Some(client);
    VisionRig {
        served,
        _queue: queue,
        health,
        calls,
        _workers: workers,
    }
}

/// A 1x1 PNG, the smallest real image.
fn tiny_png() -> Vec<u8> {
    let mut png = Vec::new();
    image::RgbImage::from_pixel(1, 1, image::Rgb([9, 9, 9]))
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode a PNG");
    png
}

/// One PNG chunk: length, type and data, and the CRC over type and data.
fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut body = kind.to_vec();
    body.extend(data);
    let mut crc = 0xFFFF_FFFFu32;
    for byte in &body {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
    chunk.extend(&body);
    chunk.extend((!crc).to_be_bytes());
    chunk
}

/// A PNG that declares `width` x `height` and holds no pixel data: the signature, a valid IHDR and an
/// empty IDAT. Only a header read can size it; a pixel decode of it fails.
fn png_header(width: u32, height: u32) -> Vec<u8> {
    let mut ihdr = width.to_be_bytes().to_vec();
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend(png_chunk(b"IHDR", &ihdr));
    png.extend(png_chunk(b"IDAT", &[]));
    png
}

fn data_url(bytes: &[u8]) -> String {
    use base64::Engine as _;
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// A chat request with one user message holding `urls` as image parts and a question.
fn chat_with_images(urls: &[String]) -> ChatReq {
    let mut content = vec![serde_json::json!({ "type": "text", "text": "what is this?" })];
    content.extend(
        urls.iter()
            .map(|url| serde_json::json!({ "type": "image_url", "image_url": { "url": url } })),
    );
    serde_json::from_value(serde_json::json!({
        "messages": [{ "role": "user", "content": content }]
    }))
    .expect("a chat request")
}

fn refusal(error: &anyhow::Error) -> &str {
    &error
        .downcast_ref::<RefusedRequest>()
        .unwrap_or_else(|| panic!("a refusal, got: {error:#}"))
        .0
}

/// SC-001. Three hundred 1x1 PNGs with the cap at 16 are refused before any payload is base64-decoded:
/// the counting decoder is called 0 times. Mutation: check the count after decoding (move the cap check
/// below the `decode` call in `admit_images`); the decoder is called and the row goes red.
#[test]
fn the_image_count_cap_is_checked_before_any_base64_decode() {
    let url = data_url(&tiny_png());
    let urls = vec![url; 300];
    let decoded = Cell::new(0usize);
    let error = admit_images(&urls, |_| {
        decoded.set(decoded.get() + 1);
        Ok(tiny_png())
    })
    .expect_err("300 images are over the cap of 16");
    assert!(error.0.contains("limit is 16"), "{}", error.0);
    assert_eq!(MAX_IMAGES_PER_REQUEST, 16);
    assert_eq!(
        decoded.get(),
        0,
        "no payload is decoded for a refused count"
    );
}

/// SC-001 on the served path: a chat with 300 images over a real vision thread is a 400 that names the
/// cap, and nothing reaches the front end. Mutation: as above; the payloads (`AAAA`, which is not an
/// image) are decoded first and the refusal is about the image header instead.
#[test]
fn a_chat_with_too_many_images_is_refused_before_the_vision_thread() {
    let rig = vision_rig(Behaviour::Refuse, Duration::from_secs(5));
    let urls = vec!["data:image/png;base64,AAAA".to_string(); 300];
    let error = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &|| false,
    )
    .expect_err("300 images are refused");
    assert!(refusal(&error).contains("limit is 16"), "{error:#}");
    assert_eq!(classify_handler_error(&error).0, 400);
    assert_eq!(rig.calls.load(Ordering::SeqCst), 0);
}

/// SC-002. Three images whose headers sum past the pixel cap are refused from their headers alone: the
/// bytes are bare PNG headers a pixel decode could not read, and the front end is never asked.
/// Mutation: drop the `pixels > MAX_IMAGE_PIXELS_PER_REQUEST` comparison; the request reaches the front
/// end and the row goes red.
#[test]
fn images_whose_headers_sum_past_the_pixel_cap_are_refused_from_headers_alone() {
    let rig = vision_rig(Behaviour::Refuse, Duration::from_secs(5));
    // Each image is under the cap alone; together they are over it.
    let side = ((MAX_IMAGE_PIXELS_PER_REQUEST / 2) as f64).sqrt() as u32;
    let urls = vec![data_url(&png_header(side, side)); 3];
    let error = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &|| false,
    )
    .expect_err("the pixels are over the cap");
    assert!(refusal(&error).contains("pixels"), "{error:#}");
    assert_eq!(classify_handler_error(&error).0, 400);
    assert_eq!(rig.calls.load(Ordering::SeqCst), 0);
}

/// Images within both caps reach the front end: the caps are not a blanket refusal.
#[test]
fn images_within_the_caps_reach_the_front_end() {
    let rig = vision_rig(Behaviour::Refuse, Duration::from_secs(5));
    let urls = vec![data_url(&tiny_png()); MAX_IMAGES_PER_REQUEST];
    let error = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &|| false,
    )
    .expect_err("the fake front end refuses");
    assert_eq!(refusal(&error), "the fake refuses");
    assert_eq!(rig.calls.load(Ordering::SeqCst), 1);
}

fn completion(body: serde_json::Value) -> CompletionReq {
    serde_json::from_value(body).expect("a completion request")
}

/// SC-003. 10,001 `logit_bias` entries are a 400. Mutation: delete the entries check in
/// `SamplingParams::check`; the request is admitted and the row goes red.
#[test]
fn a_logit_bias_over_the_entry_cap_is_a_400() {
    let (served, queue) = served_over(tiny_model(), 1);
    let bias: serde_json::Map<String, serde_json::Value> = (0..=MAX_LOGIT_BIAS_ENTRIES)
        .map(|id| (id.to_string(), serde_json::json!(1.0)))
        .collect();
    let req = completion(serde_json::json!({ "prompt": "a", "logit_bias": bias }));
    let error = handle_completion(
        &served,
        &test_metrics("caps"),
        &req,
        &gives_up_after_two_seconds(),
    )
    .expect_err("10,001 entries are refused");
    assert!(refusal(&error).contains("logit_bias"), "{error:#}");
    assert_eq!(classify_handler_error(&error).0, 400);
    assert_nothing_reached(&queue);
}

/// SC-003. A stop string over the per-string byte cap, and kept strings over the total cap, are each a
/// 400. Mutation: delete the per-string check, or the total check, in `Stop::list`; that case goes red.
#[test]
fn stop_strings_over_the_byte_caps_are_a_400() {
    let long = "x".repeat(MAX_STOP_BYTES + 1);
    let per_string = 1 + MAX_TOTAL_STOP_BYTES / MAX_STOP_BYTES;
    let total: Vec<String> = vec!["y".repeat(MAX_STOP_BYTES); per_string];
    for (what, stop) in [("one string", vec![long]), ("the total", total)] {
        let (served, queue) = served_over(tiny_model(), 1);
        let req = completion(serde_json::json!({ "prompt": "a", "stop": stop }));
        let error = handle_completion(
            &served,
            &test_metrics("caps"),
            &req,
            &gives_up_after_two_seconds(),
        )
        .expect_err("the stop strings are refused");
        assert!(refusal(&error).contains("stop"), "{what}: {error:#}");
        assert_eq!(classify_handler_error(&error).0, 400, "{what}");
        assert_nothing_reached(&queue);
    }
}

/// A `client_gone` that fires after a while: a request that is wrongly admitted waits on a queue no
/// engine reads, and this turns that wait into a failed assertion instead of a hang.
fn gives_up_after_two_seconds() -> impl Fn() -> bool {
    let started = Instant::now();
    move || started.elapsed() > Duration::from_secs(2)
}

fn assert_nothing_reached(queue: &Receiver<EngineMsg>) {
    assert!(
        matches!(
            queue.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "a refused request must not reach the engine"
    );
}

/// SC-004. A front end that panics turns `/health` to `dead` and the request in flight fails with a typed
/// server error well inside the deadline; a request after it fails too. Mutation: remove the
/// `catch_unwind` in `spawn_vision_thread` (run the loop bare); `/health` stays `serving` and the row
/// goes red.
#[test]
fn a_panicking_vision_thread_turns_health_dead_and_fails_the_request() {
    let rig = vision_rig(Behaviour::Panic, Duration::from_secs(30));
    let urls = vec![data_url(&tiny_png())];
    let started = Instant::now();
    let error = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &|| false,
    )
    .expect_err("the front end panicked");
    assert!(
        matches!(
            error.downcast_ref::<ImageError>(),
            Some(ImageError::Failed(_))
        ),
        "{error:#}"
    );
    assert_eq!(
        classify_handler_error(&error),
        (500, "image_front_end_error")
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "failed at once, not at the deadline"
    );
    match rig.health.state() {
        EngineState::Dead(Death::Panicked(reason)) => {
            assert!(reason.contains("fake vision tower broke"), "{reason}");
        }
        other => panic!("/health must be dead, got {other:?}"),
    }
    let after = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &|| false,
    )
    .expect_err("the thread is gone");
    assert_eq!(classify_handler_error(&after).0, 500);
}

/// A front end that never answers fails the request at the deadline with the same typed error.
/// Mutation: drop the deadline test in `await_answer`; the wait never ends and the row times out.
#[test]
fn a_hung_vision_thread_fails_the_request_at_the_deadline() {
    let (gate_tx, gate_rx) = mpsc::channel();
    let rig = vision_rig(Behaviour::Hang(gate_rx), Duration::from_millis(400));
    let urls = vec![data_url(&tiny_png())];
    let started = Instant::now();
    let error = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &gives_up_after_two_seconds(),
    )
    .expect_err("the front end hangs");
    assert_eq!(
        classify_handler_error(&error),
        (500, "image_front_end_error")
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    drop(gate_tx);
}

/// The wait for the vision thread notices a client that hung up. Mutation: drop the `client_gone` poll in
/// `await_answer`; the wait runs to the deadline and the row goes red.
#[test]
fn the_wait_for_the_vision_thread_ends_when_the_client_hangs_up() {
    let (gate_tx, gate_rx) = mpsc::channel();
    let rig = vision_rig(Behaviour::Hang(gate_rx), Duration::from_secs(3));
    let urls = vec![data_url(&tiny_png())];
    let started = Instant::now();
    let error = handle_chat(
        &rig.served,
        &test_metrics("caps"),
        &chat_with_images(&urls),
        &|| true,
    )
    .expect_err("the client is gone");
    assert_eq!(
        error.downcast_ref::<GenerationError>(),
        Some(&GenerationError::ClientCancelled)
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    drop(gate_tx);
}

/// The text of one unseeded or seeded completion at temperature 0.8: the production handler over the
/// production loop, with a fake that draws each token from the request's own sampler over flat logits.
fn sampled_text(seed: Option<u64>) -> String {
    let drawn = Arc::new(Mutex::new(Vec::<u32>::new()));
    let probe_drawn = Arc::clone(&drawn);
    let pick_drawn = Arc::clone(&drawn);
    let flat = vec![0.0f32; 90];
    let engine = RecordingEngine::new(1)
        .probing(move |_, sampler| {
            let token = sampler.pick(&flat).expect("a pick");
            probe_drawn.lock().unwrap().push(token as u32);
        })
        .picking(move |_, _| vec![pick_drawn.lock().unwrap().pop().expect("a drawn token")]);
    let metrics = Arc::new(test_metrics("seed"));
    let (served, thread) = serve_with_fake(engine, tiny_model(), &metrics);
    let mut body = serde_json::json!({ "prompt": "a", "max_tokens": 8, "temperature": 0.8 });
    if let Some(seed) = seed {
        body["seed"] = seed.into();
    }
    let reply = handle_completion(&served, &metrics, &completion(body), &|| false)
        .expect("the completion succeeds");
    thread.finish(served);
    let reply: serde_json::Value = serde_json::from_str(&reply).expect("a JSON reply");
    reply["choices"][0]["text"]
        .as_str()
        .expect("text")
        .to_string()
}

/// SC-005. Two identical requests without `seed` at temperature 0.8 return different text (eight draws
/// over 90 tokens: the chance two agree is 90^-8 per trial), five trials running; two with `seed: 7`
/// match. Mutation: resolve an absent seed to 0 (`unwrap_or(0)` in `resolve_seed`); the unseeded trials
/// are identical and the row goes red.
#[test]
fn unseeded_requests_differ_and_seeded_requests_match() {
    for trial in 0..5 {
        assert_ne!(sampled_text(None), sampled_text(None), "trial {trial}");
    }
    assert_eq!(sampled_text(Some(7)), sampled_text(Some(7)));
    assert_ne!(sampled_text(Some(7)), sampled_text(Some(8)));
}
