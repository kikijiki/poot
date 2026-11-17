//! What a built poot binary says about itself: the git sha it was built from, the cargo profile, the enabled
//! cargo features and the compiler. One source for every binary that states it (the benchmark runner's result
//! lines, the server's identity route), so the two cannot name different fields or derive them differently.
//!
//! The build-script half, [`emit`], runs in the binary's own `build.rs` and exports the four values as
//! compile-time environment variables; the runtime half, [`build_identity!`], reads them back in that
//! binary. Both run per binary, so the profile and features are the binary's own, not this crate's.
//!
//! ```ignore
//! // build.rs
//! fn main() { poot_build_identity::emit(); }
//! // main.rs
//! const BUILD: poot_build_identity::BuildIdentity = poot_build_identity::build_identity!();
//! ```

mod emit;

pub use emit::emit;

/// The build a binary came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildIdentity {
    /// The commit the binary was built from, 40 lowercase hex digits, with `-dirty` when a tracked build
    /// input differed from it.
    pub git_sha: &'static str,
    /// The cargo profile: `release` is the only one a benchmark result accepts.
    pub profile: &'static str,
    /// The binary's enabled cargo features, sorted and comma-separated; empty when it has none.
    pub features: &'static str,
    /// `rustc --version` of the compiler the binary was built with.
    pub rustc_version: &'static str,
}

impl BuildIdentity {
    /// The names a result or a route states these four values under, with their values: the one place the
    /// field names exist.
    pub const fn fields(&self) -> [(&'static str, &'static str); 4] {
        [
            ("build_sha", self.git_sha),
            ("build_profile", self.profile),
            ("build_features", self.features),
            ("rustc_version", self.rustc_version),
        ]
    }
}

/// The [`BuildIdentity`] of the binary this expands in. The package needs `build.rs` calling [`emit`].
#[macro_export]
macro_rules! build_identity {
    () => {
        $crate::BuildIdentity {
            git_sha: env!("POOT_BUILD_SHA"),
            profile: env!("POOT_BUILD_PROFILE"),
            features: env!("POOT_BUILD_FEATURES"),
            rustc_version: env!("POOT_RUSTC_VERSION"),
        }
    };
}
