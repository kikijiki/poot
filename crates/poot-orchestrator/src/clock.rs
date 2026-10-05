//! Timestamp formatting and process-unique execution identifiers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static EXEC_RUN_SEQ: AtomicU64 = AtomicU64::new(0);

pub(crate) fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    iso_from_secs(secs)
}

/// UTC ISO `minutes` from now (the `--keep-warm` expiry). Same format as [`now_iso`], so they compare as strings.
pub(crate) fn iso_in_minutes(minutes: u64) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + (minutes as i64) * 60;
    iso_from_secs(secs)
}

pub(crate) fn new_exec_run_id() -> String {
    let seq = EXEC_RUN_SEQ.fetch_add(1, Ordering::Relaxed);
    exec_run_id_from(SystemTime::now(), std::process::id(), seq)
}

pub(crate) fn exec_run_id_from(time: SystemTime, pid: u32, seq: u64) -> String {
    let duration = time.duration_since(UNIX_EPOCH).unwrap();
    format!(
        "exec-{}-{:09}-{pid}-{seq}",
        duration.as_secs(),
        duration.subsec_nanos()
    )
}

pub(crate) fn iso_from_secs(secs: i64) -> String {
    // ISO-8601 UTC without a date crate (Howard Hinnant's civil-from-days).
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Local UTC offset in seconds, read once from `date +%z` (e.g. "+0900" -> 32400); 0 if `date` is
/// unavailable. The DB stores UTC; the log prefix uses local time.
fn local_offset_secs() -> i64 {
    static OFF: OnceLock<i64> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::process::Command::new("date")
            .arg("+%z")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                let z = s.trim();
                let b = z.as_bytes();
                if b.len() < 5 {
                    return None;
                }
                let sign = if b[0] == b'-' { -1 } else { 1 };
                let hh: i64 = z.get(1..3)?.parse().ok()?;
                let mm: i64 = z.get(3..5)?.parse().ok()?;
                Some(sign * (hh * 3600 + mm * 60))
            })
            .unwrap_or(0)
    })
}

/// Current local wall-clock as `HH:MM:SS` for the log prefix.
pub(crate) fn now_local_hms() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
        + local_offset_secs();
    let tod = secs.rem_euclid(86400);
    format!("{:02}:{:02}:{:02}", tod / 3600, (tod % 3600) / 60, tod % 60)
}
