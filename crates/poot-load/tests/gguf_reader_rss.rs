//! Card 543 SC-001 (the GGUF loader RSS gate), restated for the mapped store (POOT-1099): reading a
//! real Q4_K_M or Q8_0 GGUF through [`poot_load::gguf::read_gguf`] holds the file as one mapping, so
//! the store's ANONYMOUS host memory (`RssAnon`) stays bookkeeping-sized - far below the file size -
//! and the peak total RSS (`VmHWM`, which counts the mapped pages validation touched) stays within
//! 1.3x the file size. The driver-level row is `poot-llm/tests/gguf_load_rss.rs`.
//!
//! Mutation (recorded here, never left in the tree; `crates/poot-quant/src/stored.rs`, `slice`):
//! copying the sliced bytes into a new owned region makes every tensor an anonymous copy; the
//! `RssAnon` row goes red (anonymous growth ~ the file size against the 64 MiB limit).

fn proc_status_kib(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    status
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix(field)?.strip_prefix(':')?;
            rest.split_whitespace().next()?.parse().ok()
        })
        .unwrap_or_else(|| panic!("missing {field} in /proc/self/status"))
}

struct IdentityNames;

impl poot_load::gguf::TensorNameMap for IdentityNames {
    fn weight_key(&self, gguf_name: &str) -> Option<poot_quant::weights::WeightKey> {
        Some(poot_quant::weights::WeightKey::from(gguf_name))
    }
}

fn assert_read_gguf_maps_without_anonymous_copies(path: &std::path::Path) {
    let file_bytes = std::fs::metadata(path).expect("stat gguf file").len();

    let baseline_rss = proc_status_kib("VmRSS") * 1024;
    let baseline_anon = proc_status_kib("RssAnon") * 1024;
    let file = poot_quant::StoredBytes::map(std::fs::File::open(path).expect("open gguf file"))
        .expect("map gguf file");
    let index = poot_load::gguf::GgufIndex::parse(&file).expect("parse gguf index");
    let store = poot_load::gguf::read_gguf(
        &index,
        &file,
        &IdentityNames,
        poot_load::plan::default_workers(),
    )
    .expect("read_gguf");
    std::hint::black_box(&store);
    let anon_growth = (proc_status_kib("RssAnon") * 1024).saturating_sub(baseline_anon);
    const ANON_LIMIT: u64 = 64 * 1024 * 1024;
    println!("anon_growth_bytes={anon_growth} anon_limit_bytes={ANON_LIMIT}");
    assert!(
        anon_growth <= ANON_LIMIT,
        "anonymous host memory grew by {anon_growth} bytes for a {file_bytes}-byte mapped checkpoint"
    );
    let peak_rss = proc_status_kib("VmHWM") * 1024;
    let incremental_peak = peak_rss.saturating_sub(baseline_rss);
    let limit = (file_bytes * 13 / 10) + 16 * 1024 * 1024; // 1.3x file size + slack
    println!(
        "card543 gguf reader rss: file={} file_bytes={file_bytes} baseline_rss_bytes={baseline_rss} \
         peak_rss_bytes={peak_rss} incremental_peak_bytes={incremental_peak} limit_bytes={limit}",
        path.display()
    );
    assert!(
        incremental_peak <= limit,
        "incremental peak RSS {incremental_peak} exceeds {limit} bytes (file {file_bytes})"
    );
}

/// Runs `assert_read_gguf_maps_without_anonymous_copies` in a freshly spawned copy of this test binary
/// (a clean RSS baseline), as `bf16_load_rss_gate.rs` does for the driver-level M4 gate.
fn run_in_fresh_process(child_env: &str, test_name: &str, checkpoint: poot_test_util::Checkpoint) {
    let Some(path) = poot_test_util::model_path(checkpoint) else {
        return;
    };

    if std::env::var_os(child_env).is_some() {
        assert_read_gguf_maps_without_anonymous_copies(&path);
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(child_env, "1")
        .output()
        .expect("spawn fresh RSS process");
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    assert!(
        output.status.success(),
        "fresh RSS child process exited {}",
        output.status
    );
}

#[test]
fn read_gguf_of_q4_k_m_maps_without_anonymous_copies() {
    run_in_fresh_process(
        "POOT_CARD543_RSS_CHILD_Q4KM",
        "read_gguf_of_q4_k_m_maps_without_anonymous_copies",
        poot_test_util::checkpoint!("qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    );
}

#[test]
fn read_gguf_of_q8_0_maps_without_anonymous_copies() {
    run_in_fresh_process(
        "POOT_CARD543_RSS_CHILD_Q80",
        "read_gguf_of_q8_0_maps_without_anonymous_copies",
        poot_test_util::checkpoint!("qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"),
    );
}
