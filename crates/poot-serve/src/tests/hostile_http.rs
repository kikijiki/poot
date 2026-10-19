//! POOT-1221 (R2-C5-001): the HTTP read path on hostile bytes. Every row writes raw bytes to a real
//! socket and runs the production `handle` on the other end, then reads the status line the client
//! gets: the framing and limit refusals (header budget, `Content-Length` forms, `Transfer-Encoding`,
//! body cap, truncated body, host and media-type checks, arrival time) each name their status, and
//! seeded garbage never panics the handler or draws anything but a 4xx. No row reaches the engine: a
//! refused request must not cost it a message. Model-free.
//!
//! These rows live beside the handler, not under `tests/`: `poot-serve` is a binary, and the only
//! backends it starts on are device backends, so an integration test could not reach this path
//! without a GPU.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1221 report):
//! HH-1 `read_capped_line` returns `true` always (no 431); HH-2 `parse_content_length` accepts a sign;
//! HH-3 `read_request_body_from` compares `content_length >= cap`; HH-4 the `Transfer-Encoding` arm is
//! removed.

use super::*;
use crate::http::MAX_HEADER_BYTES;

/// Run `handle` over `request` (written in full, then the write side closed, as a client that has
/// sent everything it will) and return the response text, `""` when the connection closed unanswered.
fn exchange(request: &[u8], limits: &Limits) -> String {
    exchange_with(request, limits, true)
}

fn exchange_with(request: &[u8], limits: &Limits, finish_writing: bool) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let (server, _) = listener.accept().expect("accept");
    // A request larger than the socket buffers is written while the handler reads: on its own thread.
    let mut writer = client.try_clone().unwrap();
    let request = request.to_vec();
    let sending = std::thread::spawn(move || {
        let _ = writer.write_all(&request);
        let _ = writer.flush();
        if finish_writing {
            let _ = writer.shutdown(std::net::Shutdown::Write);
        }
    });
    let (served, queue) = served_over(tiny_model(), 1);
    let policy = LoraAdminPolicy::from_configured_key(std::net::IpAddr::from([127, 0, 0, 1]), None)
        .expect("a loopback policy without a key");
    let _ = handle(
        server,
        Instant::now(),
        &served,
        &EngineHealth::serving_for_test("fake", 1, 64),
        &test_metrics("hostile-http"),
        limits,
        &policy,
    );
    let _ = sending.join();
    let mut response = String::new();
    let _ = client.read_to_string(&mut response);
    assert!(
        queue.try_recv().is_err(),
        "a refused request must not reach the engine queue"
    );
    response
}

fn status_of(response: &str) -> u16 {
    response
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0)
}

/// A POST to the completions route with `headers` (each a full `Name: value` line) and `body`.
fn post(headers: &[&str], body: &str) -> Vec<u8> {
    let mut request = String::from("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n");
    for header in headers {
        request.push_str(header);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    request.push_str(body);
    request.into_bytes()
}

const JSON: &str = "Content-Type: application/json";

#[test]
fn a_well_formed_health_request_is_answered() {
    let response = exchange(
        b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
        &test_limits(),
    );
    assert_eq!(status_of(&response), 200, "{response}");
}

/// The header budget is 64 KiB for the request line and every header together: a newline-free request
/// line, one huge header and many small ones all end at 431, and a head just inside the budget is read.
#[test]
fn a_head_over_the_header_budget_is_refused_with_431() {
    let limits = test_limits();
    let no_newline = vec![b'a'; MAX_HEADER_BYTES + 1];
    assert_eq!(
        status_of(&exchange(&no_newline, &limits)),
        431,
        "request line"
    );

    let one_huge = format!(
        "GET /health HTTP/1.1\r\nHost: localhost\r\nX-Big: {}\r\n\r\n",
        "a".repeat(MAX_HEADER_BYTES)
    );
    assert_eq!(
        status_of(&exchange(one_huge.as_bytes(), &limits)),
        431,
        "one header"
    );

    let many: String = (0..MAX_HEADER_BYTES / 8)
        .map(|i| format!("X-{i:05}: 1234\r\n"))
        .collect();
    let many = format!("GET /health HTTP/1.1\r\nHost: localhost\r\n{many}\r\n");
    assert!(many.len() > MAX_HEADER_BYTES);
    assert_eq!(
        status_of(&exchange(many.as_bytes(), &limits)),
        431,
        "many headers"
    );

    let prefix = "GET /health HTTP/1.1\r\nHost: localhost\r\nX-Pad: ";
    let suffix = "\r\n\r\n";
    let pad = MAX_HEADER_BYTES - prefix.len() - suffix.len() - 1;
    let inside = format!("{prefix}{}{suffix}", "a".repeat(pad));
    assert_eq!(inside.len(), MAX_HEADER_BYTES - 1);
    assert_eq!(
        status_of(&exchange(inside.as_bytes(), &limits)),
        200,
        "one byte inside"
    );
}

/// A `Content-Length` is ASCII digits and nothing else; every other spelling is a 400, so a front
/// proxy and this server cannot disagree on where the body ends. The request is a GET of `/health`,
/// which is answered 200 whenever the head is accepted, so `+0` (a spelling Rust's integer parser
/// reads as 0) is told from a refusal by its status and not by what a body does downstream.
#[test]
fn content_length_spellings_that_are_not_plain_digits_are_refused() {
    let limits = test_limits();
    let health = |value: &str| {
        format!("GET /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: {value}\r\n\r\n")
    };
    assert_eq!(
        status_of(&exchange(health("0").as_bytes(), &limits)),
        200,
        "plain digits"
    );
    for value in [
        "-1",
        "+0",
        "1e3",
        "0x10",
        "5 5",
        "5,5",
        "",
        "\u{0661}",
        "99999999999999999999999",
    ] {
        let response = exchange(health(value).as_bytes(), &limits);
        assert_eq!(
            status_of(&response),
            400,
            "Content-Length {value:?}: {response}"
        );
    }
}

#[test]
fn two_content_lengths_are_refused_even_when_they_agree() {
    let response = exchange(
        &post(&[JSON, "Content-Length: 2", "Content-Length: 2"], "{}"),
        &test_limits(),
    );
    assert_eq!(status_of(&response), 400, "{response}");
}

#[test]
fn any_transfer_encoding_is_refused_with_411() {
    for value in ["chunked", "identity", "gzip, chunked"] {
        let header = format!("Transfer-Encoding: {value}");
        // With a valid length alongside: the encoding alone is the reason for the refusal.
        let response = exchange(
            &post(&[JSON, &header, "Content-Length: 2"], "{}"),
            &test_limits(),
        );
        assert_eq!(status_of(&response), 411, "{value}: {response}");
    }
}

#[test]
fn a_post_without_a_length_is_refused_with_411() {
    let response = exchange(&post(&[JSON], ""), &test_limits());
    assert_eq!(status_of(&response), 411, "{response}");
}

/// The body cap is checked against the declaration before a body byte is read: a request that declares
/// one byte over the cap and sends nothing is a 413 at once, and a body exactly at the cap is read.
#[test]
fn a_declared_body_over_the_cap_is_refused_before_it_is_read() {
    let over = format!("Content-Length: {}", BODY_CAP + 1);
    let response = exchange(&post(&[JSON, &over], ""), &test_limits());
    assert_eq!(status_of(&response), 413, "{response}");

    let at_cap = format!("Content-Length: {BODY_CAP}");
    let body = "x".repeat(BODY_CAP);
    let response = exchange(&post(&[JSON, &at_cap], &body), &test_limits());
    assert_ne!(
        status_of(&response),
        413,
        "a body exactly at the cap is read: {response}"
    );
}

#[test]
fn a_body_shorter_than_its_declaration_is_refused() {
    let response = exchange(&post(&[JSON, "Content-Length: 100"], "{}"), &test_limits());
    assert_eq!(status_of(&response), 400, "{response}");
    assert!(response.contains("body"), "{response}");
}

#[test]
fn a_body_that_is_not_utf8_is_refused() {
    let mut request = post(&[JSON, "Content-Length: 2"], "");
    request.extend([0xC3, 0x28]);
    let response = exchange(&request, &test_limits());
    assert_eq!(status_of(&response), 400, "{response}");
}

#[test]
fn a_post_that_does_not_declare_json_is_refused_with_415() {
    for header in [
        "Content-Type: text/plain",
        "Content-Type: application/x-www-form-urlencoded",
    ] {
        let response = exchange(&post(&[header, "Content-Length: 2"], "{}"), &test_limits());
        assert_eq!(status_of(&response), 415, "{header}: {response}");
    }
    let no_type =
        b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}";
    assert_eq!(status_of(&exchange(no_type, &test_limits())), 415);
}

#[test]
fn a_foreign_or_missing_host_is_refused() {
    let limits = test_limits();
    for host in [
        "evil.example",
        "localhost:1",
        "127.0.0.1.evil.example",
        "[::2]",
    ] {
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: {host}\r\n{JSON}\r\nContent-Length: 2\r\n\r\n{{}}"
        );
        assert_eq!(
            status_of(&exchange(request.as_bytes(), &limits)),
            403,
            "Host {host}"
        );
    }
    let none = format!("POST /v1/completions HTTP/1.1\r\n{JSON}\r\nContent-Length: 2\r\n\r\n{{}}");
    assert_eq!(
        status_of(&exchange(none.as_bytes(), &limits)),
        400,
        "no Host"
    );
    let twice = post(&[JSON, "Host: localhost", "Content-Length: 2"], "{}");
    assert_eq!(status_of(&exchange(&twice, &limits)), 400, "two Hosts");
}

#[test]
fn an_unknown_method_is_refused_with_405() {
    let response = exchange(
        b"BREW /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
        &test_limits(),
    );
    assert_eq!(status_of(&response), 405, "{response}");
}

/// A client that never finishes its head is answered 408 at the request deadline, not held.
#[test]
fn a_head_that_never_ends_is_answered_with_408_at_the_deadline() {
    let limits = test_limits_with(8, Duration::from_millis(300));
    let started = Instant::now();
    let response = exchange_with(b"GET /health HTTP/1.1\r\nHost: loc", &limits, false);
    assert_eq!(status_of(&response), 408, "{response}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

/// Seeded garbage and mutated requests: the handler returns, never panics, and what it answers is a
/// 4xx or nothing.
#[test]
fn seeded_garbage_never_panics_the_handler_or_draws_a_success() {
    let mut s = 0x5EED_1221u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let limits = test_limits_with(8, Duration::from_secs(2));
    let seed_request = post(&[JSON, "Content-Length: 2"], "{}");
    for round in 0..120 {
        let mut bytes = if round % 2 == 0 {
            (0..1 + next() % 300).map(|_| next() as u8).collect()
        } else {
            seed_request.clone()
        };
        if round % 2 == 1 {
            for _ in 0..1 + next() % 5 {
                let at = (next() % bytes.len() as u64) as usize;
                bytes[at] = next() as u8;
            }
        }
        let response = exchange(&bytes, &limits);
        let status = status_of(&response);
        assert!(
            response.is_empty() || (400..500).contains(&status) || status == 200,
            "round {round}: {bytes:?} drew {response}"
        );
    }
}
