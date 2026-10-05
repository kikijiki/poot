//! When generation ends (Card 734): an end-of-sequence token, the `max_new` bound, a stop string in the
//! generated text, or the caller's sink. The loop checks them after each token and never runs another
//! step once one holds.

use std::collections::BTreeSet;

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model sampled one of its end-of-sequence tokens (not part of the output).
    Eos,
    /// `max_new` tokens were generated.
    MaxTokens,
    /// A stop string appeared in the generated text; the output holds the token that completed it.
    Stop,
    /// The token sink asked to stop.
    Cancelled,
}

/// The stop conditions of one request.
#[derive(Debug)]
pub(crate) struct Stops<'a> {
    eos: &'a BTreeSet<u32>,
    max_new: usize,
    strings: Vec<&'a str>,
    longest: usize,
    /// The generated text so far, from which a boundary-straddling stop string is found.
    text: String,
}

impl<'a> Stops<'a> {
    pub(crate) fn new(eos: &'a BTreeSet<u32>, max_new: usize, strings: &'a [String]) -> Self {
        let strings: Vec<&str> = strings
            .iter()
            .map(String::as_str)
            .filter(|s| !s.is_empty())
            .collect();
        let longest = strings.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            eos,
            max_new,
            strings,
            longest,
            text: String::new(),
        }
    }

    pub(crate) fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// Record `piece`, the text the latest token added, and report why generation ends after
    /// `generated` tokens, if it does. Only a bounded tail is kept: the last `longest - 1` bytes (moved
    /// back to a char boundary), which is all a match straddling into the next piece can need.
    pub(crate) fn after_token(&mut self, piece: &str, generated: usize) -> Option<FinishReason> {
        if self.longest > 0 {
            self.text.push_str(piece);
            if self.strings.iter().any(|s| self.text.contains(s)) {
                return Some(FinishReason::Stop);
            }
            let mut from = self.text.len().saturating_sub(self.longest - 1);
            while !self.text.is_char_boundary(from) {
                from -= 1;
            }
            self.text.drain(..from);
        }
        (generated >= self.max_new).then_some(FinishReason::MaxTokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stop string split across two pieces is found when its last piece lands, and a stop string
    /// already past is not found twice: `after_token` reports it at the completing token.
    #[test]
    fn a_stop_string_straddling_two_pieces_ends_at_the_completing_piece() {
        let eos = BTreeSet::new();
        let strings = ["END".to_string()];
        let mut stops = Stops::new(&eos, 100, &strings);
        assert_eq!(stops.after_token("xE", 1), None);
        assert_eq!(stops.after_token("N", 2), None);
        assert_eq!(stops.after_token("D!", 3), Some(FinishReason::Stop));
    }

    /// The kept tail is bounded, and a multi-byte stop string split across pieces (and across the trim
    /// point) still matches. Mutations: trim one byte too much (`longest - 2`), or trim at a raw byte
    /// offset without moving to a char boundary (panics inside the 3-byte character).
    #[test]
    fn the_kept_tail_is_bounded_and_a_multibyte_stop_still_matches_across_pieces() {
        let eos = BTreeSet::new();
        let strings = ["\u{20ac}\u{20ac}!".to_string()]; // 3 + 3 + 1 bytes
        let mut stops = Stops::new(&eos, 100, &strings);
        for _ in 0..50 {
            assert_eq!(stops.after_token("abcdefgh", 1), None);
        }
        assert!(
            stops.text.len() <= 6 + 3,
            "bounded, got {}",
            stops.text.len()
        );
        assert_eq!(stops.after_token("\u{20ac}", 1), None);
        assert_eq!(stops.after_token("\u{20ac}", 2), None);
        assert_eq!(stops.after_token("!", 3), Some(FinishReason::Stop));

        // An ASCII stop of 4 bytes split one byte per piece: the kept tail must hold 3 bytes.
        let strings = ["abcd".to_string()];
        let mut stops = Stops::new(&eos, 100, &strings);
        for (i, piece) in ["x", "a", "b", "c"].iter().enumerate() {
            assert_eq!(stops.after_token(piece, i + 1), None);
        }
        assert_eq!(stops.after_token("d", 5), Some(FinishReason::Stop));

        // Pieces of 3-byte characters put the trim point inside one.
        let strings = ["\u{20ac}\u{20ac}!".to_string()];
        let mut stops = Stops::new(&eos, 100, &strings);
        for _ in 0..3 {
            assert_eq!(stops.after_token("\u{20ac}", 1), None);
        }
        assert_eq!(stops.after_token("a", 1), None);
        assert_eq!(stops.after_token("\u{20ac}", 1), None);
        assert_eq!(stops.after_token("\u{20ac}", 1), None);
        assert_eq!(stops.after_token("!", 1), Some(FinishReason::Stop));
    }

    #[test]
    fn max_new_ends_and_an_empty_stop_never_matches() {
        let eos = BTreeSet::new();
        let strings = [String::new()];
        let mut stops = Stops::new(&eos, 2, &strings);
        assert_eq!(stops.after_token("a", 1), None);
        assert_eq!(stops.after_token("b", 2), Some(FinishReason::MaxTokens));
    }
}
