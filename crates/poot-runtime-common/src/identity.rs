//! Which adapter and driver a device runs on.

/// The adapter and driver a device runs on, as its own runtime reports them. A benchmark row names them so two
/// results are compared only when they ran on the same adapter under the same driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceIdentity {
    /// The adapter's name (`AMD Radeon 8060S Graphics`).
    pub adapter: String,
    /// The driver stack's version, in the form the runtime reports it.
    pub driver: String,
}
