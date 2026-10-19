//! A scripted Peano/IRON toolchain for model-free AIE tests (POOT-664, POOT-1117).
//!
//! The compiler's protocol handling and typed failures are what the scripts prove, not machine code: no
//! Peano or IRON driver is installed on the dev box, and a script is not a driver. A test that needs
//! numerical execution or hardware is a device row, not one that uses this.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use poot_runtime_common::{AieShapeAbi, ArgAccess};
use poot_target::AieTargetId;

use crate::{AieFifo, AieLinkRecipe, AieTile, FifoEnd, KernelCache, KernelKey, Toolchain};

/// Publish `bytes` under `key` whatever they are: the stale or foreign entry a cache directory can be
/// left holding. The cache's own `store` is crate-private, so a test that needs a wrong entry on disk
/// (and to see it refused) plants it here.
pub fn plant_entry(cache: &KernelCache, key: &KernelKey, bytes: &[u8]) -> std::io::Result<()> {
    cache.store(key, bytes)
}

fn script(dir: &Path, name: &str, body: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write the scripted driver");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
        .expect("make the scripted driver executable");
    path.to_string_lossy().into_owned()
}

/// A [`Toolchain`] whose Peano (`POOT_AIE_LLC`) and, when `with_iron`, IRON (`POOT_AIE_IRON`) drivers are
/// scripts under `dir` that follow the documented driver protocol and write plausible outputs (the "object"
/// is an ELF header followed by the IR it was compiled from, so it names its kernel symbol).
pub fn scripted_toolchain(dir: &Path, with_iron: bool) -> Toolchain {
    let peano = script(
        dir,
        "peano",
        r#"case "$1" in --version) echo "fake-peano (aie2p)"; exit 0;; esac
for arg; do case "$prev" in -o) out="$arg";; esac; case "$arg" in *.ll) ll="$arg";; esac; prev="$arg"; done
printf '\177ELF' > "$out"; cat "$ll" >> "$out""#,
    );
    let iron = script(
        dir,
        "iron",
        r#"case "$1" in --version) echo "fake-iron"; exit 0;; --platforms) echo npu2; exit 0;; esac
while [ $# -gt 0 ]; do case "$1" in --xclbin) x="$2";; --insts) i="$2";; esac; shift; done
printf 'xclbin2\0payload' > "$x"; printf 'IIII' > "$i""#,
    );
    Toolchain::from_env_with(move |name| match name {
        "POOT_AIE_LLC" => Some(peano.clone()),
        "POOT_AIE_IRON" => with_iron.then(|| iron.clone()),
        _ => None,
    })
}

/// The recipe of a region of one tile whose body takes `abi`'s bindings, in order: each read binding is a
/// host-to-tile FIFO and each written one a tile-to-host FIFO. The shape every region of one equation has,
/// for a test that links a single body against a hand-built ABI.
pub fn single_tile_recipe(target: AieTargetId, abi: &AieShapeAbi) -> AieLinkRecipe {
    let fifos = abi
        .bindings()
        .iter()
        .enumerate()
        .map(|(binding, b)| {
            let (host, tile) = (
                FifoEnd::Host { binding },
                FifoEnd::Tile {
                    tile: 0,
                    port: binding,
                },
            );
            let (producer, consumer) = match b.access() {
                ArgAccess::Read => (host, tile),
                ArgAccess::Write => (tile, host),
            };
            AieFifo {
                producer,
                consumers: vec![consumer],
                depth: AieLinkRecipe::FIFO_DEPTH,
                elements: b.elements(),
                element: b.storage().element(),
            }
        })
        .collect();
    let tile = AieTile {
        ports: (0..abi.bindings().len()).collect(),
    };
    AieLinkRecipe::new(target, abi.clone(), vec![tile], fifos)
        .expect("one tile over a shape ABI is a region")
}
