//! POOT-1226 SC-002: the out-of-tree "vlm" family ([`vlm`]) serves an image request, and no file under
//! `crates/poot-llm/src` was edited for it: this crate reaches the product through its public surface
//! only (`Registry::builtin_plus`, `ModelHandle::load`, `FrontEndSession`, `Driver::generate_with_inputs`).
//!
//! - The family registers, loads from a written checkpoint, and its handle carries the front end its
//!   registry entry built. A session over it lowers an image and a question to a prompt and the tags the
//!   family's preprocessor derived from the image; two images lower to different tags.
//! - A driver over the fake device generates from that prompt with the lowered inputs, and the bytes
//!   the driver bound on the device are the image's tags: the image reached the model through the
//!   generic path. A model with no front end opens no session.

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{Engine, Executor};
use poot_llm::driver::error::DriverError;
use poot_llm::driver::{Driver, DriverOptions, GenerateRequest, Layout, LoadOptions, ModelHandle};
use poot_llm::{FrontEndSession, GenerationControl, Sampler};
use poot_models::multimodal::front_end::{ChatTurn, FrontEndError, Image, Media, TowerId};
use poot_models::registry::Registry;

use crate::tagged::GENERATED_TAG;
use crate::vlm::{self, PLACEHOLDER, quadrant_tags};
use crate::{Counting, FakeDevice, driver_options, write_checkpoint};

fn registry() -> Registry {
    Registry::builtin_plus(&[crate::tagged::ENTRY, vlm::ENTRY]).unwrap()
}

fn load(dir: &Path) -> Arc<ModelHandle> {
    Arc::new(ModelHandle::load(dir, &registry(), &LoadOptions::default()).unwrap())
}

/// The 8x8 image whose four quadrants have the greys `quadrants`, in reading order.
fn image(quadrants: [u8; 4]) -> Image {
    let rgb = (0..8 * 8)
        .flat_map(|p| {
            let (y, x) = (p / 8, p % 8);
            [quadrants[usize::from(y >= 4) * 2 + usize::from(x >= 4)]; 3]
        })
        .collect();
    Image::from_rgb8(8, 8, rgb).unwrap()
}

/// A session over the handle of a vlm checkpoint written for the test, with no tower to open.
fn session() -> (
    FrontEndSession,
    Arc<ModelHandle>,
    poot_test_util::UniqueTempPath,
) {
    let dir = poot_test_util::unique_temp_path("vlm_checkpoint");
    write_checkpoint(&dir, &vlm::fixture());
    let handle = load(&dir);
    let session = FrontEndSession::open(Arc::clone(&handle), no_tower)
        .unwrap()
        .expect("the vlm family has a front end");
    (session, handle, dir)
}

/// The executor of a tower: the vlm family has none, so no executor is asked for.
fn no_tower(
    _: TowerId,
    _: NonZeroUsize,
) -> Result<(Box<dyn Executor>, DriverOptions), DriverError> {
    unreachable!("the vlm family has no tower")
}

/// SC-002: a registered out-of-tree family with a preprocessor of its own serves an image request.
/// The image is lowered to the family's prompt (four placeholders, the question) and the family's
/// tags (one per quadrant of the image, then the text's), and a second image lowers to other tags.
/// Mutation: build the front end only for the families `poot-llm` knows (drop `front_end` from the
/// parts in `ModelHandle::build` unless the family key is `idefics3`); the handle has no front end and
/// this row is red.
#[test]
fn an_image_request_of_an_out_of_tree_family_is_lowered_by_its_own_preprocessor() {
    let (mut session, handle, _dir) = session();
    assert!(
        handle.front_end().is_some(),
        "the loader keeps the family's front end"
    );
    let (dark, bright) = (image([10, 70, 130, 250]), image([250, 130, 70, 10]));
    let turns = [ChatTurn::user(1, "d r")];
    let first = session.prepare(&turns, &mut [Media::Image(&dark)]).unwrap();
    let second = session
        .prepare(&turns, &mut [Media::Image(&bright)])
        .unwrap();
    for prepared in [&first, &second] {
        assert_eq!(
            prepared.prompt,
            [PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, 3, 17],
            "the placeholders, then the question the model's own tokenizer made"
        );
    }
    assert_eq!(quadrant_tags(&dark), [0, 1, 2, 3]);
    assert_eq!(quadrant_tags(&bright), [3, 2, 1, 0]);
    let tags = |prepared: &poot_models::multimodal::front_end::Prepared| -> Vec<i32> {
        let Some(poot_models::io::InputValue::Tokens(tags)) =
            prepared.inputs.get(crate::tagged::TAG)
        else {
            panic!("the image lowers to the family's tags")
        };
        tags.data().as_i32().unwrap().to_vec()
    };
    assert_eq!(tags(&first), [0, 1, 2, 3, 0, 0]);
    assert_eq!(tags(&second), [3, 2, 1, 0, 0, 0]);
}

/// A request the family's preprocessor cannot take is the client's: two images, or none, are refused
/// before anything runs.
#[test]
fn what_the_family_cannot_take_is_refused_as_the_clients() {
    let (mut session, _handle, _dir) = session();
    let one = image([0; 4]);
    let two = [Media::Image(&one), Media::Image(&one)];
    let refused = |result: Result<_, FrontEndError>| {
        assert!(
            matches!(result, Err(FrontEndError::Refused(_))),
            "{result:?}"
        );
    };
    refused(session.prepare(&[ChatTurn::user(2, "d")], &mut { two }));
    refused(session.prepare(&[ChatTurn::user(0, "d")], &mut []));
    refused(session.prepare(&[ChatTurn::user(1, "d")], &mut []));
}

/// The lowered request reaches the device through the generic path: a driver over the fake device
/// generates from the prompt with the lowered inputs, and the bytes it bound are the image's tags in
/// the prefill pieces and the family's continuation tag at the decode steps.
#[test]
fn the_lowered_images_tags_are_what_the_driver_binds() {
    let (mut session, handle, _dir) = session();
    let bright = image([250, 130, 70, 10]);
    let prepared = session
        .prepare(&[ChatTurn::user(1, "d r")], &mut [Media::Image(&bright)])
        .unwrap();
    let device = FakeDevice::default();
    let executor = Counting {
        inner: Engine::new(device.clone()),
        loads: Rc::new(Cell::new(0)),
    };
    let mut driver = Driver::new(handle, Box::new(executor), driver_options(6, 64))
        .unwrap()
        .serve(Layout::Contiguous);
    let generation = driver
        .generate_with_inputs(
            GenerateRequest {
                prompt: prepared.prompt,
                max_new: 4,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            prepared.inputs,
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(generation.tokens.len(), 4);
    let uploaded = |tags: &[i32]| {
        let want: Vec<u8> = tags.iter().flat_map(|t| t.to_le_bytes()).collect();
        device.writes.borrow().contains(&want)
    };
    assert!(
        uploaded(&[3, 2, 1, 0, 0, 0]),
        "the image's tags were not bound"
    );
    assert!(
        uploaded(&[GENERATED_TAG]),
        "the decode step's tag was not bound"
    );
}

/// A model with no front end opens no session: the text families serve token prompts and chats only.
#[test]
fn a_family_without_a_front_end_opens_no_session() {
    let dir = poot_test_util::unique_temp_path("tagged_checkpoint");
    write_checkpoint(&dir, &crate::tagged::fixture());
    let handle = load(&dir);
    assert!(handle.front_end().is_none());
    let session = FrontEndSession::open(handle, no_tower).unwrap();
    assert!(session.is_none());
}
