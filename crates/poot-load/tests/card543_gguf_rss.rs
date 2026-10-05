//! Card 543 SC-001 (the GGUF loader RSS gate): reading a real Q4_K_M or Q8_0 GGUF through
//! [`poot_load::gguf::read_gguf`] alone peaks at no more than 1.3x the file size in host RSS.
//! `read_gguf` range-reads each tensor's own bytes into a [`poot_quant::weights::WeightStore`] with
//! no widening (card 543); this proves that at the reader itself. The Runner-level row
//! (`Runner::load_gguf`, packed storage end to end) is `poot-llm/tests/card545a_gguf_load_rss.rs`.
//!
//! Mutation (recorded here, never left in the tree; `crates/poot-load/src/gguf.rs`, `read_gguf`):
//! reading the whole file into a buffer kept alive for the whole scan, alongside every range-read
//! tensor buffer (the old `Gguf.bytes` field's lifetime), turns this row red for the Q4_K_M fixture:
//! observed incremental peak RSS 1,006,071,808 bytes (~959.4 MiB) against the 655,597,257-byte
//! (~625.2 MiB) limit (491,400,032-byte file x 1.3 + 16 MiB slack). Removing the extra whole-file
//! read (the landed code) restores green: observed incremental peak 517,160,960 bytes (~493.1 MiB,
//! ~1.05x file size) for Q4_K_M and 700,010,496 bytes (~667.6 MiB, ~1.04x file size) for Q8_0
//! (675,710,816-byte file).

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

fn assert_read_gguf_peaks_under_1_3x_file_size(path: &std::path::Path) {
    let file_bytes = std::fs::metadata(path).expect("stat gguf file").len();

    let baseline_rss = proc_status_kib("VmRSS") * 1024;
    let file = std::fs::File::open(path).expect("open gguf file");
    let index = poot_load::gguf::GgufIndex::open(path).expect("open gguf index");
    let store = poot_load::gguf::read_gguf(&index, &file, &IdentityNames).expect("read_gguf");
    std::hint::black_box(&store);
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

/// Runs `assert_read_gguf_peaks_under_1_3x_file_size` in a freshly spawned copy of this test binary
/// (a clean RSS baseline), as `card540b_rss_gate.rs` does for the Runner-level M4 gate.
fn run_in_fresh_process(child_env: &str, test_name: &str, checkpoint: poot_test_util::Checkpoint) {
    let Some(path) = poot_test_util::model_path(checkpoint) else {
        return;
    };

    if std::env::var_os(child_env).is_some() {
        assert_read_gguf_peaks_under_1_3x_file_size(&path);
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
fn read_gguf_of_q4_k_m_peaks_under_one_point_three_x_file_size_in_host_rss() {
    run_in_fresh_process(
        "POOT_CARD543_RSS_CHILD_Q4KM",
        "read_gguf_of_q4_k_m_peaks_under_one_point_three_x_file_size_in_host_rss",
        poot_test_util::checkpoint!("qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    );
}

#[test]
fn read_gguf_of_q8_0_peaks_under_one_point_three_x_file_size_in_host_rss() {
    run_in_fresh_process(
        "POOT_CARD543_RSS_CHILD_Q80",
        "read_gguf_of_q8_0_peaks_under_one_point_three_x_file_size_in_host_rss",
        poot_test_util::checkpoint!("qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"),
    );
}
