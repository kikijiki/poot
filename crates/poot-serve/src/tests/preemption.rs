use super::*;

#[test]
fn select_preemption_victim_picks_fewest_generated_tokens() {
    let slots = vec![
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 5, 20)),
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 1, 20)),
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 3, 20)),
    ];
    assert_eq!(
        select_preemption_victim(&slots),
        Some(1),
        "slot 1 has generated the fewest tokens (1) - the cheapest to discard/replay"
    );
}

#[test]
fn select_preemption_victim_ties_broken_by_lowest_index() {
    let slots = vec![
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 0, 20)),
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 0, 20)),
    ];
    assert_eq!(select_preemption_victim(&slots), Some(0));
}

#[test]
fn select_preemption_victim_none_when_no_active_slots() {
    let slots: Vec<Option<Slot>> = vec![None, None, None];
    assert_eq!(select_preemption_victim(&slots), None);
}

#[test]
fn select_preemption_victim_skips_free_slots_between_active_ones() {
    let slots = vec![
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 9, 20)),
        None,
        Some(preempt_test_slot(preempt_toks(0, 5), 4, 2, 20)),
    ];
    assert_eq!(select_preemption_victim(&slots), Some(2));
}

#[test]
fn preempt_slot_frees_the_victim_and_registers_its_resident_prefix() {
    let mut paged = PagedKvCache::new(2, 8); // 2 slots, 8 shared blocks
    let toks = preempt_toks(0, 2 * BLOCK_SIZE); // 32 tokens
    paged.append(0, 2 * BLOCK_SIZE).unwrap(); // slot 0 reserves 2 full blocks
    // steady-state position: tokens[..pos] (31 tokens) is what preempt_slot will register/keep.
    let mut slots = vec![
        Some(preempt_test_slot(toks.clone(), 2 * BLOCK_SIZE - 1, 20, 20)),
        None,
    ];

    let seq = preempt_slot(&mut slots, &mut paged, 0);
    assert!(slots[0].is_none(), "the victim slot is freed");
    assert_eq!(
        seq.tokens, toks,
        "the full token history is preserved for resumption"
    );
    assert_eq!(
        seq.generated, 20,
        "prior progress carries over for future victim-selection bias"
    );

    // the one full BLOCK_SIZE-aligned block of the resident span survives (cache-pinned); the rest of
    // the pool is reclaimable immediately via evict_prefix.
    assert_eq!(
        paged.evict_prefix(),
        1,
        "exactly the one full registered block is cache-exclusive"
    );
    assert_eq!(
        paged.free_blocks(),
        8,
        "the whole pool is free again after eviction - no leak"
    );
}

#[test]
fn preempt_slot_never_registers_beyond_pos_even_when_more_tokens_are_reserved() {
    // Hazard: a slot admitted this iteration has more tokens reserved in its block table than written; only `tokens[..pos]` is real. Preempting it must not register the unwritten tail as valid KV.
    // Simulated: 2 blocks (32 tokens) reserved, but `pos` says only the first 16 are resident.
    let mut paged = PagedKvCache::new(2, 8);
    let toks = preempt_toks(0, 2 * BLOCK_SIZE);
    paged.append(0, 2 * BLOCK_SIZE).unwrap();
    let mut slots = vec![
        Some(preempt_test_slot(toks.clone(), BLOCK_SIZE, 0, 20)),
        None,
    ];

    preempt_slot(&mut slots, &mut paged, 0);

    // a DIFFERENT admission with the exact same token content must reuse only the resident
    // first block, never the reserved-but-unwritten second one.
    let skip = paged
        .admit_prefix_for(1, &toks, 4, PrefixIdentity::BASE)
        .unwrap();
    assert_eq!(
        skip, BLOCK_SIZE,
        "only the resident first block is reusable; the reserved-but-unwritten second block is not"
    );
}

#[test]
fn admit_prefix_with_preemption_succeeds_via_evict_before_ever_preempting() {
    // Evicting a stale registered prefix alone frees enough room; preemption must not fire (it would interrupt a live request).
    let mut paged = PagedKvCache::new(2, 2); // 2 blocks total
    let stale = preempt_toks(0, BLOCK_SIZE);
    paged
        .admit_prefix_for(0, &stale, 0, PrefixIdentity::BASE)
        .unwrap();
    paged.register_and_reset_for(0, &stale, PrefixIdentity::BASE); // registers 1 block, frees the rest; now cache-exclusive
    assert_eq!(paged.free_blocks(), 1);

    let mut slots: Vec<Option<Slot>> = vec![None, None];
    let (result, requeued) = admit_prefix_with_preemption(
        &mut slots,
        &mut paged,
        0,
        &preempt_toks(100, BLOCK_SIZE),
        4,
        PrefixIdentity::BASE,
    );
    assert!(result.is_ok(), "eviction alone frees the needed block");
    assert!(
        requeued.is_empty(),
        "nothing was preempted - no active slot even existed to preempt"
    );
}

#[test]
fn admit_prefix_with_preemption_preempts_the_cheapest_active_slot_when_eviction_is_not_enough() {
    // 3-block pool (48 tokens). Slot 0 holds a long-running, well-progressed request (2 blocks). The
    // remaining 1 free block is not enough for a new 2-block admission into free slot 1, and nothing
    // is registered in the prefix cache yet (eviction alone cannot help) - preemption of slot 0 is the
    // only way this admission can succeed.
    let mut paged = PagedKvCache::new(2, 3);
    paged.append(0, 30).unwrap(); // 2 blocks (30 tokens reserved)
    let mut slots = vec![
        Some(preempt_test_slot(preempt_toks(0, 25), 24, 15, 20)),
        None,
    ];
    assert_eq!(paged.free_blocks(), 1);

    let (result, requeued) = admit_prefix_with_preemption(
        &mut slots,
        &mut paged,
        1,
        &preempt_toks(1000, 20),
        10,
        PrefixIdentity::BASE,
    );
    let skip = result.expect("admission succeeds after preempting slot 0");
    assert_eq!(
        skip, 0,
        "the new admission shares no prefix with the preempted victim"
    );
    assert_eq!(requeued.len(), 1, "exactly one victim was preempted");
    assert_eq!(
        requeued[0].generated, 15,
        "the preempted sequence's progress is preserved for requeue"
    );
    assert!(slots[0].is_none(), "the victim slot was freed");
    // 3-block pool: preemption + eviction returns all 3 to the free list, the new admission then takes
    // 2 of them (20-token prompt + 10 max_new = 30 tokens -> 2 blocks), leaving exactly 1 free.
    assert_eq!(
        paged.free_blocks(),
        1,
        "the new 2-block admission occupies the freed room, leaving the pool's true remainder"
    );
}

#[test]
fn admit_prefix_with_preemption_fails_cleanly_when_the_request_can_never_fit() {
    // Nothing to preempt and the request exceeds the whole pool: preemption cannot create capacity, so this must fail as before, not loop or panic.
    let mut paged = PagedKvCache::new(1, 1); // 1 block = BLOCK_SIZE tokens total
    let mut slots: Vec<Option<Slot>> = vec![None];
    let (result, requeued) = admit_prefix_with_preemption(
        &mut slots,
        &mut paged,
        0,
        &preempt_toks(0, 2 * BLOCK_SIZE),
        4,
        PrefixIdentity::BASE,
    );
    assert!(result.is_err());
    assert!(requeued.is_empty());
}

#[test]
fn resume_preempted_reuses_the_registered_prefix_when_it_survives() {
    let mut paged = PagedKvCache::new(2, 4);
    let toks = preempt_toks(0, 2 * BLOCK_SIZE);
    paged.append(0, 2 * BLOCK_SIZE).unwrap();
    let mut slots = vec![
        Some(preempt_test_slot(toks.clone(), 2 * BLOCK_SIZE - 1, 10, 20)),
        None,
    ];
    let seq = preempt_slot(&mut slots, &mut paged, 0);

    let resumed =
        resume_preempted(&mut paged, 1, seq).unwrap_or_else(|_| panic!("resume succeeds"));
    assert_eq!(
        resumed.pos, BLOCK_SIZE,
        "the one registered block is reused; only the rest replays"
    );
    assert_eq!(
        resumed.tokens, toks,
        "no data lost across the preemption cycle"
    );
    assert_eq!(
        resumed.generated, 10,
        "prior progress carries over unchanged"
    );
    assert!(
        resumed.prefilled,
        "resumption always takes the token-by-token replay path"
    );
}

#[test]
fn resume_preempted_falls_back_to_full_recompute_when_the_registered_prefix_is_gone() {
    let mut paged = PagedKvCache::new(2, 4);
    let toks = preempt_toks(0, 2 * BLOCK_SIZE);
    paged.append(0, 2 * BLOCK_SIZE).unwrap();
    let mut slots = vec![
        Some(preempt_test_slot(toks.clone(), 2 * BLOCK_SIZE - 1, 10, 20)),
        None,
    ];
    let seq = preempt_slot(&mut slots, &mut paged, 0);
    // simulate a DIFFERENT admission's own evict_prefix reclaiming this registration before resumption.
    assert_eq!(paged.evict_prefix(), 1);

    let resumed = resume_preempted(&mut paged, 1, seq)
        .unwrap_or_else(|_| panic!("resume still succeeds - just recomputes"));
    assert_eq!(
        resumed.pos, 0,
        "nothing survived to match; full recompute from scratch"
    );
    assert_eq!(
        resumed.tokens, toks,
        "still no data lost - correctness holds even on a total cache miss"
    );
}

#[test]
fn resume_preempted_returns_the_sequence_untouched_on_failure_nothing_is_dropped() {
    let mut paged = PagedKvCache::new(1, 1); // 1 block total, nothing reserved yet
    let (reply, _rx) = mpsc::channel();
    let seq = PreemptedSeq {
        tokens: preempt_toks(0, 2 * BLOCK_SIZE), // needs 2 blocks; only 1 exists in the whole pool
        prompt_len: 15,
        generated: 5,
        max_new: 10,
        sampler: Sampler::greedy(),
        stop: vec!["STOP".to_string()],
        stream: true,
        reply,
        decode_start: None,
        last_token: None,
        lora_adapter: LoraAdapterLease::none(),
        mrope: None,
    };
    let original_tokens = seq.tokens.clone();
    match resume_preempted(&mut paged, 0, seq) {
        Ok(_) => panic!("this request can never fit the whole pool - expected an Err"),
        Err(seq_back) => {
            assert_eq!(
                seq_back.tokens, original_tokens,
                "the caller gets the SAME sequence back - nothing silently dropped"
            );
            assert_eq!(seq_back.stop, vec!["STOP".to_string()]);
            assert!(seq_back.stream);
        }
    }
}

/// End-to-end (GPU-free) admit, exhaustion, preempt, requeue, free, resume, deliver, finish cycle of `batch_engine_loop`, through paged-KV/slot bookkeeping with no decode step.
/// Asserts no double-free (the pool returns to full capacity), no orphaned slot state (every slot `None`), and no silent drop (the preempted request's reply channel is still live after resuming).
#[test]
fn recompute_preemption_end_to_end_no_leak_no_orphan_no_silent_drop() {
    const POOL_BLOCKS: usize = 3; // 48 tokens
    let mut paged = PagedKvCache::new(2, POOL_BLOCKS);
    let mut slots: Vec<Option<Slot>> = vec![None, None];
    let mut preempted: VecDeque<PreemptedSeq> = VecDeque::new();

    // --- Request A admits into slot 0 and runs for a while: prompt 10 tokens, 15 generated since
    // (tokens.len() == 25, pos == 24 per the "tokens[..pos] is exactly what's resident" invariant).
    let (reply_a, rx_a) = mpsc::channel();
    paged.append(0, 30).unwrap(); // A's admission reserved prompt(10) + max_new(20) = 30 tokens up front
    slots[0] = Some(Slot {
        tokens: preempt_toks(0, 25),
        prompt_len: 10,
        pos: 24,
        generated: 15,
        max_new: 20,
        sampler: Sampler::greedy(),
        stop: vec![],
        stream: false,
        reply: reply_a,
        prefilled: true,
        decode_start: None,
        last_token: None,
        lora_adapter: LoraAdapterLease::none(),
        mrope: None,
    });
    assert_eq!(
        paged.free_blocks(),
        1,
        "A's 2 blocks leave exactly 1 free in the 3-block pool"
    );

    // --- Request B tries to admit into free slot 1: prompt 20 + max_new 10 = 30 tokens (2 blocks) -
    // more than the 1 free block, and nothing is registered yet, so this must preempt A.
    let (result, freed) = admit_prefix_with_preemption(
        &mut slots,
        &mut paged,
        1,
        &preempt_toks(1000, 20),
        10,
        PrefixIdentity::BASE,
    );
    let skip_b = result.expect("B admits after preempting A");
    assert_eq!(freed.len(), 1, "exactly A was preempted");
    assert!(slots[0].is_none(), "A's slot was freed by the preemption");
    preempted.extend(freed);
    // the caller (mirroring `admit`'s own tail) writes B into the slot the reservation was made for:
    let (reply_b, rx_b) = mpsc::channel();
    slots[1] = Some(Slot {
        tokens: preempt_toks(1000, 20),
        prompt_len: 20,
        pos: skip_b,
        generated: 0,
        max_new: 10,
        sampler: Sampler::greedy(),
        stop: vec![],
        stream: false,
        reply: reply_b,
        prefilled: false,
        decode_start: None,
        last_token: None,
        lora_adapter: LoraAdapterLease::none(),
        mrope: None,
    });

    // --- slot 0 is free again, but the pool is oversubscribed (A's 2 blocks + B's 2 = 4 > 3), which is why A was preempted. A cannot resume until something finishes and returns blocks, so the caller pushes it back on the queue (as `batch_engine_loop` does).
    assert!(
        slots[0].is_none(),
        "slot 0 is free for A to try resuming into"
    );
    assert_eq!(preempted.len(), 1);
    let seq_a = preempted.pop_front().expect("A is queued for resumption");
    let seq_a = match resume_preempted(&mut paged, 0, seq_a) {
        Ok(_) => {
            panic!("A should not fit yet - B is still holding blocks A's full resume needs too")
        }
        Err(seq_a) => seq_a,
    };
    assert_eq!(
        seq_a.tokens.len(),
        25,
        "A's history is intact even after a failed resume attempt"
    );
    preempted.push_front(seq_a);

    // --- B finishes and frees its blocks for real (mirrors the engine's own completion bookkeeping) -
    // Now there is enough room for A to resume.
    slots[1]
        .as_ref()
        .unwrap()
        .reply
        .send(GenEvent::Done(vec![9, 9], vec![], true))
        .expect("B's reply channel works normally too");
    assert!(matches!(rx_b.try_recv(), Ok(GenEvent::Done(toks, _, true)) if toks == vec![9, 9]));
    paged.reset(1);
    slots[1] = None;

    // --- retry resuming A: it fits now.
    let seq_a = preempted.pop_front().expect("A is still queued");
    let resumed_a = resume_preempted(&mut paged, 0, seq_a)
        .unwrap_or_else(|_| panic!("A resumes once B's blocks are back"));
    assert_eq!(
        resumed_a.generated, 15,
        "A's prior progress survived the whole round trip"
    );
    assert_eq!(
        resumed_a.tokens.len(),
        25,
        "A's full token history survived the whole round trip"
    );
    slots[0] = Some(resumed_a);
    assert!(preempted.is_empty());

    // --- no silent drop: A's original reply channel is still live after the preempt/retry/resume round trip.
    slots[0]
        .as_ref()
        .unwrap()
        .reply
        .send(GenEvent::Done(vec![1, 2, 3], vec![], true))
        .expect("A's original reply channel is still connected");
    assert!(
        matches!(rx_a.try_recv(), Ok(GenEvent::Done(toks, _, true)) if toks == vec![1, 2, 3]),
        "the client-facing receiver for A got the message - the SAME channel survived preemption"
    );

    // --- A finishes for real: no orphaned slot state, no leaked blocks across the whole cycle.
    paged.reset(0);
    slots[0] = None;
    assert!(slots.iter().all(|s| s.is_none()), "no orphaned slot state");
    assert_eq!(
        paged.free_blocks(),
        POOL_BLOCKS,
        "every block returned to the pool - no leak across the preemption cycle"
    );
}
