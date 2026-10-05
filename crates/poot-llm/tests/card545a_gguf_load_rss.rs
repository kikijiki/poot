//! Card 545a: `driver::ModelHandle::load` reads a GGUF once. The header comes from the
//! bounded `GgufIndex` read, every tensor is range-read into the `WeightStore` exactly once, and a
//! quantized tensor stays packed (never widened), so loading the real Qwen2.5-0.5B Q8_0 and Q4_K_M
//! files peaks near 1x the file size in host RSS; the excess over the file is the tokenizer.

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

const TEST: &str = "model_handle_load_gguf_peaks_near_one_x_file_size";

#[test]
fn model_handle_load_gguf_peaks_near_one_x_file_size() {
    const CHILD: &str = "POOT_CARD545A_RSS_CHILD";
    const FILE: &str = "POOT_CARD545A_RSS_FILE";

    if let Some(path) = std::env::var_os(CHILD).and(std::env::var_os(FILE)) {
        let file_bytes = std::fs::metadata(&path).expect("stat gguf").len();
        let baseline_rss = proc_status_kib("VmRSS") * 1024;
        let registry = poot_models::registry::Registry::builtin().expect("builtin registry");
        let handle = poot_llm::driver::ModelHandle::load(std::path::Path::new(&path), &registry)
            .expect("load the gguf");
        std::hint::black_box(&handle);
        let peak_rss = proc_status_kib("VmHWM") * 1024;
        let incremental_peak = peak_rss.saturating_sub(baseline_rss);
        // 1.05x the file, plus the tokenizer and the host-computed tables (rope, norms) the file
        // does not store: 96 MiB covers the 151k-token tokenizer with room to spare.
        let limit = file_bytes + file_bytes / 20 + 96 * 1024 * 1024;
        println!(
            "card545a rss: file={} file_bytes={file_bytes} incremental_peak_bytes={incremental_peak} \
             ratio={:.3} limit_bytes={limit}",
            std::path::Path::new(&path).display(),
            incremental_peak as f64 / file_bytes as f64
        );
        assert!(
            incremental_peak <= limit,
            "incremental peak RSS {incremental_peak} exceeds {limit} bytes (file {file_bytes})"
        );
        return;
    }

    for checkpoint in [
        poot_test_util::checkpoint!("qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"),
        poot_test_util::checkpoint!("qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ] {
        let Some(path) = poot_test_util::model_path(checkpoint) else {
            return;
        };
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env(CHILD, "1")
            .env(FILE, &path)
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
}
