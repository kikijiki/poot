//! A scripted Peano/IRON toolchain for model-free AIE tests (POOT-664, POOT-1117).
//!
//! The compiler's protocol handling and typed failures are what the scripts prove, not machine code: no
//! Peano or IRON driver is installed on the dev box, and a script is not a driver. A test that needs
//! numerical execution or hardware is a device row, not one that uses this.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use poot_codegen::Toolchain;

fn script(dir: &Path, name: &str, body: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write the scripted driver");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
        .expect("make the scripted driver executable");
    path.to_string_lossy().into_owned()
}

/// A [`Toolchain`] whose Peano (`POOT_AIE_LLC`) and, when `with_iron`, IRON (`POOT_AIE_IRON`) drivers are
/// scripts under `dir` that follow the documented driver protocol and write plausible outputs.
pub fn scripted_toolchain(dir: &Path, with_iron: bool) -> Toolchain {
    let peano = script(
        dir,
        "peano",
        r#"case "$1" in --version) echo "fake-peano (aie2p)"; exit 0;; esac
for last; do :; done
printf '\177ELF' > "$last"; printf 'poot_seq_binary' >> "$last""#,
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
