//! The unit-test binary's counting global allocator, for behavioral "no source-sized copy" checks
//! ([`poot_test_util::alloc_count`] owns the allocator and the per-thread measurement).

#[global_allocator]
static ALLOCATOR: poot_test_util::alloc_count::CountingAllocator =
    poot_test_util::alloc_count::CountingAllocator;
