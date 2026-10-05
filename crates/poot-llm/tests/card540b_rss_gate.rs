//! Card 540b SC-001 (the M4 RSS gate): loading the bf16 Qwen2.5-0.5B checkpoint through
//! `driver::ModelHandle::load` peaks at no more than 1.5x the file size in host RSS: the handle holds
//! the `WeightStore` (every tensor's bytes as stored, no widening) and the family's weight map over it,
//! never a host f32 copy of a bf16 weight (R475-001/R475-009).

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

#[test]
fn resident_load_of_a_real_bf16_checkpoint_peaks_under_one_point_five_x_file_size() {
    const CHILD: &str = "POOT_CARD540B_RSS_CHILD";
    const DIR: &str = "POOT_CARD540B_RSS_DIR";

    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b")) else {
        return;
    };

    if std::env::var_os(CHILD).is_some() {
        let dir = std::env::var_os(DIR).expect("checkpoint dir");
        let file_bytes = std::fs::metadata(std::path::Path::new(&dir).join("model.safetensors"))
            .expect("stat model.safetensors")
            .len();

        let baseline_rss = proc_status_kib("VmRSS") * 1024;
        let baseline_peak = proc_status_kib("VmHWM") * 1024;
        let registry = poot_models::registry::Registry::builtin().expect("builtin registry");
        let handle = poot_llm::driver::ModelHandle::load(std::path::Path::new(&dir), &registry)
            .expect("load");
        std::hint::black_box(&handle);
        let peak_rss = proc_status_kib("VmHWM") * 1024;
        let incremental_peak = peak_rss.saturating_sub(baseline_rss);
        let limit = (file_bytes * 3 / 2) + 16 * 1024 * 1024;
        println!(
            "card540b rss: file_bytes={file_bytes} baseline_rss_bytes={baseline_rss} \
             baseline_peak_bytes={baseline_peak} peak_rss_bytes={peak_rss} \
             incremental_peak_bytes={incremental_peak} limit_bytes={limit}"
        );
        assert!(
            incremental_peak <= limit,
            "incremental peak RSS {incremental_peak} exceeds {limit} bytes (file {file_bytes})"
        );
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("resident_load_of_a_real_bf16_checkpoint_peaks_under_one_point_five_x_file_size")
        .arg("--nocapture")
        .env(CHILD, "1")
        .env(DIR, &dir)
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
