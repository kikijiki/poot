//! The `poot-orchestrator` binary: parses the command line and hands off to
//! [`poot_orchestrator::run_from_args`].

fn main() -> anyhow::Result<()> {
    poot_orchestrator::run_from_args()
}
