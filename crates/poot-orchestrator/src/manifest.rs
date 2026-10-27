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
    /// `hf download --include` patterns: fetch only these files of `hf_repo` (a repo holding many quants of a
    /// model is hundreds of GB). Empty: the whole repo.
    pub(crate) hf_include: Vec<String>,
    /// The file the runners are handed instead of the directory (the first shard of a split GGUF), relative to
    /// the model's directory. Set for a pre-quantized model, whose download is already the format llama.cpp loads.
    pub(crate) weights_file: Option<String>,
    pub(crate) poot_supported: bool,
    /// The manifest marks llama.cpp as supporting the model (an undeclared framework is unsupported).
    pub(crate) llamacpp_supported: bool,
    /// True if any framework (poot or external) has support = true. A model can be sweepable by
    /// transformers/vllm/llamacpp when poot cannot run it.
    pub(crate) any_framework_supported: bool,
}

impl Model {
    /// The directory name the weights live under (matches the harness): local_dir if set, else the id.
    pub(crate) fn dir_name(&self) -> &str {
        self.local_dir.as_deref().unwrap_or(&self.id)
    }

    /// Does the pod pre-convert the HF checkpoint to an f16 GGUF? Only for a llama.cpp-supported model that
    /// downloads as safetensors: a model with a `weights_file` already is a GGUF, and one llama.cpp does not
    /// run (an FP8 checkpoint) has no GGUF to make.
    pub(crate) fn needs_gguf_preconversion(&self) -> bool {
        self.llamacpp_supported && self.weights_file.is_none()
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
        hf_include: Vec<String>,
        weights_file: Option<String>,
        #[serde(default)]
        frameworks: HashMap<String, FwCfg>,
    }
    #[derive(serde::Deserialize)]
    struct FwCfg {
        #[serde(default)]
        support: bool,
    }
    let m: Manifest = toml::from_str(toml_src).context("parse manifest.toml")?;
    if let Some(bad) = m
        .model
        .iter()
        .find(|x| crate::gather::id_ends_in_a_repeat_suffix(&x.id))
    {
        bail!(
            "manifest model id {:?} ends in -r<N>: a run of one repeat names its set without a repeat suffix, so the \
             set could not be told from repeat N of another model; rename the model",
            bad.id
        );
    }
    Ok(m.model
        .into_iter()
        .map(|x| {
            let poot_supported = x.frameworks.get("poot").map(|f| f.support).unwrap_or(false);
            let any_framework_supported = x.frameworks.values().any(|f| f.support);
            let llamacpp_supported = x.frameworks.get("llamacpp").is_some_and(|f| f.support);
            Model {
                id: x.id,
                hf_repo: x.hf_repo,
                local_dir: x.local_dir,
                hf_include: x.hf_include,
                weights_file: x.weights_file,
                poot_supported,
                llamacpp_supported,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A model id ending in `-r<N>` cannot be told, in a one-repeat set's name, from repeat N of another model, so
    /// the manifest refuses it before any pod is rented. Mutation: skip the check and the first assertion goes red.
    #[test]
    fn a_model_id_that_reads_as_a_repeat_suffix_is_refused() {
        let manifest = |id: &str| format!("[[model]]\nid = \"{id}\"\nhf_repo = \"org/m\"\n");
        let err = parse_manifest_models(&manifest("llama-r2")).unwrap_err();
        assert!(format!("{err:#}").contains("llama-r2"), "{err:#}");
        // An id that merely contains "-r" is fine, as is one whose "r" is not followed by digits only.
        for ok in ["llama-rerank", "qwen3-r1x", "model-r", "qwen3.8-27b-fp8"] {
            assert!(parse_manifest_models(&manifest(ok)).is_ok(), "{ok}");
        }
    }
}
