use super::*;

#[test]
fn pick_idle_pod_matches_idle_image_with_ssh() {
    let pods = vec![
        pod("busy", "busy", "img-a", true),     // wrong status
        pod("nossh", "idle", "img-a", false),   // idle but no endpoint yet
        pod("wrongimg", "idle", "img-b", true), // idle but different image
        pod("good", "idle", "img-a", true),     // <- the adoptable one
        pod("good2", "idle", "img-a", true),    // also adoptable, but `good` is earlier
    ];
    assert_eq!(
        pick_idle_pod(&pods, "img-a").map(|p| p.id.as_str()),
        Some("good")
    );
    assert!(pick_idle_pod(&pods, "img-z").is_none()); // no idle pod for this image
    assert!(pick_idle_pod(&[], "img-a").is_none()); // empty
}

#[test]
fn pods_table_lifecycle_roundtrips() {
    let db = Db::open(":memory:").unwrap();
    db.create_pod_record("p1", "RTX 3090", "COMMUNITY", "img-a")
        .unwrap();
    // provisioning, no endpoint -> not adoptable.
    assert!(pick_idle_pod(&db.live_pods().unwrap(), "img-a").is_none());
    db.set_pod_endpoint("p1", "1.2.3.4", 8700).unwrap();
    db.set_pod_status("p1", "idle").unwrap();
    // now idle + endpoint -> adoptable.
    assert_eq!(
        pick_idle_pod(&db.live_pods().unwrap(), "img-a").map(|p| p.id.clone()),
        Some("p1".to_string())
    );
    // assign -> busy, not adoptable; release -> idle again.
    db.assign_pod("p1", "run-1").unwrap();
    assert!(pick_idle_pod(&db.live_pods().unwrap(), "img-a").is_none());
    let busy = db.live_pods().unwrap();
    assert_eq!(busy[0].status, "busy");
    assert_eq!(busy[0].current_run.as_deref(), Some("run-1"));
    db.release_pod("p1").unwrap();
    assert!(pick_idle_pod(&db.live_pods().unwrap(), "img-a").is_some());
    // terminate -> drops out of live_pods.
    db.set_pod_status("p1", "terminated").unwrap();
    assert!(db.live_pods().unwrap().is_empty());
    assert_eq!(db.all_pods(10).unwrap().len(), 1); // still in history
}
