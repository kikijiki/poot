//! The Khronos validation layer as a pass/fail signal. `POOT_VULKAN_VALIDATION=1` is a test-harness
//! switch (the `device-vulkan` lane sets it): the instance then REQUIRES `VK_LAYER_KHRONOS_validation`
//! and `VK_EXT_debug_utils` (an unavailable layer is [`RuntimeError::ValidationUnavailable`], never a
//! silent pass), a messenger counts every warning and error the layer reports, and dropping the last
//! owner of the device panics if the count is not zero. Execution is otherwise unchanged.

use crate::*;

/// The messages the layer reported (warnings and errors), shared with the native callback.
pub(crate) struct Validation {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    /// Boxed so the callback's user-data pointer is stable for the messenger's lifetime.
    count: Box<AtomicU64>,
}

unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if severity.intersects(
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
    ) {
        // SAFETY: `user_data` is the `AtomicU64` boxed in the `Validation` that owns this messenger,
        // which destroys the messenger before the box is freed.
        unsafe { &*user_data.cast::<AtomicU64>() }.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the layer passes a valid callback-data struct for the duration of the call.
        let message = unsafe { data.as_ref() }
            .and_then(|d| unsafe { d.message_as_c_str() })
            .map(|m| m.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("Vulkan validation {severity:?}: {message}");
    }
    vk::FALSE
}

/// Whether the harness asked for validation.
pub(crate) fn requested() -> bool {
    std::env::var_os("POOT_VULKAN_VALIDATION").is_some()
}

impl Validation {
    /// Start counting this instance's layer messages.
    ///
    /// # Safety
    ///
    /// `instance` must be a live instance created with `VK_EXT_debug_utils` enabled, and the returned
    /// value must be [`Validation::destroy`]ed before the instance is.
    pub(crate) unsafe fn start(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Self, vk::Result> {
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        let count = Box::new(AtomicU64::new(0));
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(on_message))
            .user_data(std::ptr::from_ref::<AtomicU64>(&count).cast_mut().cast());
        // SAFETY: the caller's contract; `info` and the callback outlive the call, and the boxed count
        // outlives the messenger.
        let messenger = unsafe { loader.create_debug_utils_messenger(&info, None)? };
        Ok(Validation {
            loader,
            messenger,
            count,
        })
    }

    pub(crate) fn messages(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Stop the messenger; the count stays readable.
    ///
    /// # Safety
    ///
    /// The instance the messenger belongs to must still be alive.
    pub(crate) unsafe fn destroy(&mut self) {
        // SAFETY: the caller's contract; the messenger is destroyed once (it is nulled after).
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
        self.messenger = vk::DebugUtilsMessengerEXT::null();
    }
}
