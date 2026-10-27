//! CUDA versions as numbers. A host's CUDA version is `major.minor[.patch]`; two versions are compared field by
//! field as integers, never as text (`13.10` is newer than `13.9`, `"13.2" < "13.10"` as strings is wrong).

use std::fmt;
use std::str::FromStr;

/// A CUDA version, ordered by (major, minor, patch). A missing patch is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CudaVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

impl FromStr for CudaVersion {
    type Err = String;

    /// `major.minor` or `major.minor.patch`, each field decimal digits.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bad = || format!("{text:?} is not a CUDA version (major.minor[.patch], e.g. 13.0)");
        let fields: Vec<u32> = text
            .trim()
            .split('.')
            .map(|field| {
                if field.is_empty() || !field.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(bad());
                }
                field.parse().map_err(|_| bad())
            })
            .collect::<Result<_, _>>()?;
        match fields.as_slice() {
            [major, minor] => Ok(Self {
                major: *major,
                minor: *minor,
                patch: 0,
            }),
            [major, minor, patch] => Ok(Self {
                major: *major,
                minor: *minor,
                patch: *patch,
            }),
            _ => Err(bad()),
        }
    }
}

impl fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.patch == 0 {
            write!(f, "{}.{}", self.major, self.minor)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> CudaVersion {
        text.parse().unwrap()
    }

    /// Versions compare as numbers, through `CudaVersion`'s own ordering (the one the floor check uses). Mutation:
    /// order by the `Display` text instead (a manual `Ord` comparing `to_string()`) and `13.10 > 13.9` goes red.
    #[test]
    fn versions_compare_as_numbers_not_as_text() {
        assert!(v("13.2") >= v("13.0"));
        assert!(v("13.10") > v("13.9"), "13.10 is newer than 13.9");
        assert!(v("12.4") < v("13.0"));
        assert!(v("13.0.1") > v("13.0"));
        assert_eq!(v("13.0"), v("13.0.0"));
        assert!(v("12.9") < v("13.0"));
        assert!(v("13.2") < v("13.10"), "13.2 is older than 13.10");
    }

    #[test]
    fn a_version_must_be_major_dot_minor_with_an_optional_patch() {
        for bad in [
            "",
            "13",
            "13.",
            ".0",
            "13.x",
            "13.0.0.1",
            "v13.0",
            "-1.0",
            "13.0 beta",
        ] {
            assert!(bad.parse::<CudaVersion>().is_err(), "{bad:?}");
        }
        assert_eq!(v(" 13.2 ").to_string(), "13.2");
        assert_eq!(v("12.8.1").to_string(), "12.8.1");
    }
}
