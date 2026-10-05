//! Benchmark manifest model definitions, parsing, and selection.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Model {
    pub(crate) id: String,
    pub(crate) hf_repo: Option<String>,
    /// Weights directory name. The harness reads MODELS_DIR/<local_dir> (bench.py), which differs from
    /// the id for some models (qwen2.5-3b -> qwen2.5-3b-instruct), so downloads must use the same dir.
    pub(crate) local_dir: Option<String>,
    pub(crate) poot_supported: bool,
    /// True if any framework (poot or external) has support = true. A model can be sweepable by
    /// transformers/vllm/llamacpp when poot cannot run it.
    pub(crate) any_framework_supported: bool,
}

impl Model {
    /// The directory name the weights live under (matches the harness): local_dir if set, else the id.
    pub fn dir_name(&self) -> &str {
        self.local_dir.as_deref().unwrap_or(&self.id)
    }
}

/// Parse manifest.toml into the models poot can sweep (id + hf_repo + support flag).
pub(crate) fn parse_manifest_models(toml_src: &str) -> Result<Vec<Model>> {
    #[derive(serde::Deserialize)]
    struct Manifest {
        #[serde(default)]
        model: Vec<MModel>,
    }
    #[derive(serde::Deserialize)]
    struct MModel {
        id: String,
        hf_repo: Option<String>,
        local_dir: Option<String>,
        #[serde(default)]
        frameworks: HashMap<String, FwCfg>,
    }
    #[derive(serde::Deserialize)]
    struct FwCfg {
        #[serde(default)]
        support: bool,
    }
    let m: Manifest = toml::from_str(toml_src).context("parse manifest.toml")?;
    Ok(m.model
        .into_iter()
        .map(|x| {
            let poot_supported = x.frameworks.get("poot").map(|f| f.support).unwrap_or(false);
            let any_framework_supported = x.frameworks.values().any(|f| f.support);
            Model {
                id: x.id,
                hf_repo: x.hf_repo,
                local_dir: x.local_dir,
                poot_supported,
                any_framework_supported,
            }
        })
        .collect())
}

/// Resolve the requested model ids against the manifest. `None` = all poot-supported.
pub(crate) fn select_models(all: &[Model], requested: Option<&str>) -> Result<Vec<Model>> {
    match requested {
        None => Ok(all.iter().filter(|m| m.poot_supported).cloned().collect()),
        Some(list) => {
            let want: Vec<&str> = list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            let mut out = Vec::new();
            for id in want {
                let m = all.iter().find(|m| m.id == id).ok_or_else(|| {
                    anyhow!(
                        "unknown model '{id}'; known: {:?}",
                        all.iter().map(|m| &m.id).collect::<Vec<_>>()
                    )
                })?;
                // An explicit --models request needs only some framework to support the model; the
                // default selection stays poot-supported-only.
                if !m.any_framework_supported {
                    bail!("model '{id}' has no supported frameworks in the manifest");
                }
                out.push(m.clone());
            }
            Ok(out)
        }
    }
}
