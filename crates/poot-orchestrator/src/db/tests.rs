use super::*;

mod core;
mod pods;
mod runs;

fn pod(id: &str, status: &str, image: &str, ssh: bool) -> Pod {
    Pod {
        id: id.into(),
        gpu_type: "RTX 3090".into(),
        cloud: "COMMUNITY".into(),
        image: image.into(),
        status: status.into(),
        ssh_host: if ssh { Some("1.2.3.4".into()) } else { None },
        ssh_port: if ssh { Some(22) } else { None },
        current_run: None,
        created_at: "t0".into(),
        updated_at: "t0".into(),
    }
}

fn exec_run(id: &str, status: &str) -> Run {
    Run {
        id: id.into(),
        kind: "exec".into(),
        git_ref: "abc".into(),
        scenario: "--test".into(),
        repeats: 1,
        models: vec![],
        gpu_types: "NVIDIA L40S".into(),
        image: "img".into(),
        status: status.into(),
        pod_id: None,
    }
}

fn owner(pid: i64, start_ticks: &str) -> ExecOwner {
    ExecOwner {
        hostname: "host-a".into(),
        pid,
        boot_id: "boot-a".into(),
        start_ticks: start_ticks.into(),
    }
}
