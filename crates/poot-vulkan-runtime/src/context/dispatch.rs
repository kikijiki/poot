use crate::*;

impl Context {
    /// Dispatch a compiler-produced `kernel` once (card 608): `buffers` are all the kernel's data
    /// buffers, inputs first then the single output last (see the module doc), checked against
    /// `kernel`'s argument schema and all allocated by this context (else [`RuntimeError::KernelArgs`] /
    /// [`RuntimeError::ForeignObject`]); each is told its full allocation as its element count.
    ///
    /// Builds a fresh pipeline (no cache), records a one-dispatch timed graph, replays it and
    /// fence-waits. `wg` must match the module's `LocalSize`; the grid is `ceil(threads / wg)` per axis.
    /// Returns the device-measured duration when the queue family writes timestamps, else `None`. A
    /// kernel assert that fired is [`RuntimeError::KernelAssertFailed`].
    pub fn dispatch(
        &self,
        kernel: &CompiledKernel,
        wg: [u32; 3],
        threads: [u32; 3],
        buffers: &[&DeviceBuffer],
    ) -> Result<Option<Duration>, RuntimeError> {
        let args: Vec<Binding<'_>> = buffers
            .iter()
            .map(|&buffer| Binding {
                buffer,
                elems: buffer.elem_count(),
            })
            .collect();
        // Refuse a bad dispatch before any pipeline is built.
        self.check_bindings(kernel.args(), wg, threads, &args)?;
        let pipeline = Arc::new(self.build_pipeline(kernel)?);
        let timing = if self.timestamps_supported() {
            GraphTiming::PerDispatch
        } else {
            GraphTiming::Off
        };
        let mut recording = self.begin_graph(timing)?;
        self.record_dispatch(&mut recording, "dispatch", &pipeline, wg, threads, &args)?;
        let graph = self.end_graph(recording)?;
        let replay = graph.replay()?;
        if let Some(fault) = replay.fault {
            return Err(RuntimeError::KernelAssertFailed {
                kernel: fault.kernel,
                code: fault.code,
            });
        }
        Ok(replay.time.map(|t| t.sum_of_dispatch_durations))
    }
}
