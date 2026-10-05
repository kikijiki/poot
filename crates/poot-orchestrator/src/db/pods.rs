use super::*;

/// A pod (worker) record, spec 030. Its lifecycle is independent of any run: provisioned, `idle`,
/// assigned to a run (`busy`), released to `idle`, eventually `terminated`. A re-run can adopt an idle pod.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Pod {
    pub id: String,
    pub gpu_type: String,
    pub cloud: String,
    pub image: String,
    pub status: String, // provisioning | idle | busy | terminated
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
    pub current_run: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Pick the oldest `idle` pod with a matching image and an SSH endpoint, if any.
pub fn pick_idle_pod<'a>(pods: &'a [Pod], image: &str) -> Option<&'a Pod> {
    pods.iter().find(|p| {
        p.status == "idle"
            && p.image == image
            && p.ssh_host.as_deref().is_some_and(|h| !h.is_empty())
            && p.ssh_port.is_some()
    })
}

// --- pods (spec 030): workers with a lifecycle independent of runs. ---
impl Db {
    /// Record a freshly created pod (status `provisioning`), called as soon as RunPod returns the id.
    pub fn create_pod_record(
        &self,
        id: &str,
        gpu_type: &str,
        cloud: &str,
        image: &str,
    ) -> Result<()> {
        let now = now_iso();
        self.conn.execute(
            "INSERT OR REPLACE INTO pods (id, gpu_type, cloud, image, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 'provisioning', ?5, ?5)",
            params![id, gpu_type, cloud, image, now],
        )?;
        Ok(())
    }

    pub fn set_pod_status(&self, id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pods SET status=?2, updated_at=?3 WHERE id=?1",
            params![id, status, now_iso()],
        )?;
        Ok(())
    }

    pub fn set_pod_endpoint(&self, id: &str, host: &str, port: u16) -> Result<()> {
        self.conn.execute(
            "UPDATE pods SET ssh_host=?2, ssh_port=?3, updated_at=?4 WHERE id=?1",
            params![id, host, port as i64, now_iso()],
        )?;
        Ok(())
    }

    /// Mark a pod `busy` and record its holder. The run's own `pod_id` is set separately.
    pub fn assign_pod(&self, id: &str, run_id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pods SET status='busy', current_run=?2, updated_at=?3 WHERE id=?1",
            params![id, run_id, now_iso()],
        )?;
        Ok(())
    }

    /// Release a pod after a run: back to `idle`.
    pub fn release_pod(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pods SET status='idle', current_run=NULL, updated_at=?2 WHERE id=?1",
            params![id, now_iso()],
        )?;
        Ok(())
    }

    /// All pods not yet `terminated`.
    pub fn live_pods(&self) -> Result<Vec<Pod>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, gpu_type, cloud, image, status, ssh_host, ssh_port, current_run, created_at, updated_at
             FROM pods WHERE status != 'terminated' ORDER BY created_at",
        )?;
        let rows = stmt
            .query_map([], Self::row_to_pod)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// All pods (any status), newest first.
    pub fn all_pods(&self, limit: i64) -> Result<Vec<Pod>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, gpu_type, cloud, image, status, ssh_host, ssh_port, current_run, created_at, updated_at
             FROM pods ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map([limit], Self::row_to_pod)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    fn row_to_pod(row: &rusqlite::Row) -> rusqlite::Result<Pod> {
        Ok(Pod {
            id: row.get(0)?,
            gpu_type: row.get(1)?,
            cloud: row.get(2)?,
            image: row.get(3)?,
            status: row.get(4)?,
            ssh_host: row.get(5)?,
            ssh_port: row.get::<_, Option<i64>>(6)?.map(|p| p as u16),
            current_run: row.get(7)?,
            created_at: row.get(8)?,
            updated_at: row.get(9)?,
        })
    }
}
