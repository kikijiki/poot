//! GBNF (llama.cpp-style) context-free grammar constrained decoding.
//!
//! Pushdown engine behind `guided_grammar` / `Constraint::Grammar`. Unlike the regex/DFA path and the
//! JSON acceptor, it accepts an arbitrary user grammar in llama.cpp's GBNF syntax:
//!
//! ```text
//! root  ::= "yes" | "no"
//! ```
//!
//! A grammar is compiled once (`Grammar::parse`) into a flat element encoding, then an incremental
//! `GrammarState` walks it one character at a time: `accept_char` reports whether a character can extend the
//! output while keeping it a prefix of some string in the language, and `is_complete` reports whether the
//! output so far is a full sentence (so EOS is allowed). Because a CFG parse is nondeterministic, the state is
//! a set of parser stacks, expanded and pruned as characters arrive (the model llama.cpp's
//! `llama_grammar` uses).
//!
//! The engine works on Unicode scalars: a token's decoded text is fed one `char` at a time, and char
//! classes and literals are ranges over code points.

use std::collections::HashMap;
use std::sync::Arc;

/// One element of a rule's flat encoding. A rule is a `Vec<Element>`: alternatives are separated by `Alt` and
/// the whole rule ends with `End`. `Char` is a terminal (a class of code points); `Ref` is a nonterminal.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Element {
    /// end of the rule (also terminates the last alternative).
    End,
    /// separates one alternative from the next within a rule.
    Alt,
    /// a reference to another rule (nonterminal), by rule index.
    Ref(usize),
    /// a terminal: a set of inclusive code-point ranges; `negated` matches any code point NOT in the set.
    Char {
        ranges: Vec<(u32, u32)>,
        negated: bool,
    },
}

/// A compiled GBNF grammar: a flat list of rules plus the index of the `root` rule. Immutable and shareable;
/// wrap in `Arc` and derive many `GrammarState`s from it.
#[derive(Clone, Debug)]
pub struct Grammar {
    rules: Vec<Vec<Element>>,
    root: usize,
}

/// A position in the grammar: the `i`-th element of rule `r`.
type Pos = (usize, usize);
/// A single parser stack: the top (last) is the element to process next; entries below are pending
/// continuations.
type Stack = Vec<Pos>;

impl Grammar {
    /// Parse GBNF source into a compiled grammar. Returns a human-readable error string on a syntax error, an
    /// undefined rule reference, or a missing `root` rule.
    pub fn parse(src: &str) -> Result<Grammar, String> {
        Parser::new(src).parse()
    }

    fn is_end_of_seq(&self, r: usize, i: usize) -> bool {
        i >= self.rules[r].len() || matches!(self.rules[r][i], Element::End | Element::Alt)
    }

    /// Expand `stack` until its top is a terminal (`Char`) or it is empty (a complete parse), following rule
    /// references and enumerating their alternatives. Appends every resulting "ready" stack to `out`. `depth`
    /// caps recursion so a left-recursive (invalid) grammar cannot overflow the native stack - such a grammar
    /// under-generates rather than crashing.
    fn advance_stack(&self, stack: Stack, out: &mut Vec<Stack>, depth: usize) {
        if depth == 0 {
            return; // left-recursion / runaway guard
        }
        let Some(&(r, i)) = stack.last() else {
            out.push(stack); // empty stack: a complete parse
            return;
        };
        match &self.rules[r][i] {
            Element::Char { .. } => out.push(stack), // terminal on top: ready to match a character
            Element::Ref(sub) => {
                let sub = *sub;
                // the base is the current stack minus the reference, plus the continuation after it (unless
                // the reference is the last element of its alternative - then it is a tail call).
                let mut base = stack.clone();
                base.pop();
                if !self.is_end_of_seq(r, i + 1) {
                    base.push((r, i + 1));
                }
                // enter each alternative of the referenced rule.
                let mut j = 0usize;
                loop {
                    let mut ns = base.clone();
                    if !self.is_end_of_seq(sub, j) {
                        ns.push((sub, j));
                    }
                    self.advance_stack(ns, out, depth - 1);
                    // skip to the end of this alternative.
                    let mut k = j;
                    while !self.is_end_of_seq(sub, k) {
                        k += 1;
                    }
                    if matches!(self.rules[sub].get(k), Some(Element::Alt)) {
                        j = k + 1; // another alternative follows
                    } else {
                        break;
                    }
                }
            }
            // An Alt/End should not be a stack top (advance never pushes one), but pop defensively.
            Element::Alt | Element::End => {
                let mut ns = stack.clone();
                ns.pop();
                self.advance_stack(ns, out, depth - 1);
            }
        }
    }

    /// The set of ready stacks for the start of the grammar (all alternatives of `root`).
    fn start_stacks(&self) -> Vec<Stack> {
        let mut out = Vec::new();
        let mut j = 0usize;
        loop {
            let mut ns: Stack = Vec::new();
            if !self.is_end_of_seq(self.root, j) {
                ns.push((self.root, j));
            }
            self.advance_stack(ns, &mut out, RECURSION_CAP);
            let mut k = j;
            while !self.is_end_of_seq(self.root, k) {
                k += 1;
            }
            if matches!(self.rules[self.root].get(k), Some(Element::Alt)) {
                j = k + 1;
            } else {
                break;
            }
        }
        dedup_stacks(&mut out);
        out
    }

    /// Build a fresh incremental state at the start of this grammar.
    pub fn start_state(self: &Arc<Self>) -> GrammarState {
        GrammarState {
            grammar: self.clone(),
            stacks: self.start_stacks(),
        }
    }
}

const RECURSION_CAP: usize = 4096;

fn char_in(ranges: &[(u32, u32)], negated: bool, c: u32) -> bool {
    let inside = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
    inside != negated
}

/// Sort and deduplicate a set of stacks so the reachable-parse set stays bounded across characters.
fn dedup_stacks(stacks: &mut Vec<Stack>) {
    stacks.sort_unstable();
    stacks.dedup();
}

/// The incremental parser state: the set of parser stacks currently reachable. Cloneable so token masking can
/// probe a candidate token's characters without disturbing the live state.
#[derive(Clone)]
pub struct GrammarState {
    grammar: Arc<Grammar>,
    stacks: Vec<Stack>,
}

impl GrammarState {
    /// Try to consume one character. Returns false (leaving the state unchanged) if no reachable parse can
    /// accept it; otherwise advances the state and returns true.
    pub fn accept_char(&mut self, ch: char) -> bool {
        let c = ch as u32;
        let mut next: Vec<Stack> = Vec::new();
        for stack in &self.stacks {
            let Some(&(r, i)) = stack.last() else {
                continue; // an empty (complete) stack cannot consume more
            };
            if let Element::Char { ranges, negated } = &self.grammar.rules[r][i]
                && char_in(ranges, *negated, c)
            {
                let mut base = stack.clone();
                base.pop();
                if !self.grammar.is_end_of_seq(r, i + 1) {
                    base.push((r, i + 1));
                }
                self.grammar.advance_stack(base, &mut next, RECURSION_CAP);
            }
        }
        if next.is_empty() {
            return false;
        }
        dedup_stacks(&mut next);
        self.stacks = next;
        true
    }

    /// True when the output so far is a complete sentence of the grammar (some parse stack is empty), so EOS
    /// may be emitted here.
    pub fn is_complete(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// True when no parse can continue and none is complete (dead state). Tests/diagnostics only; the
    /// sampler relies on `accept_char` returning false per candidate.
    #[cfg(test)]
    pub fn is_dead(&self) -> bool {
        self.stacks.is_empty()
    }
}

// ---- GBNF parser -----------------------------------------------------------

struct Parser {
    src: Vec<char>,
    pos: usize,
    rules: Vec<Vec<Element>>,
    names: HashMap<String, usize>,
    /// current group-nesting depth (bounds recursion so an adversarial `((((...))))` cannot overflow the
    /// native stack - `guided_grammar` is untrusted API input).
    depth: usize,
    /// remaining element/rule budget, so a huge or repetition-exploded grammar errors cleanly instead of
    /// exhausting memory.
    budget: usize,
}

/// Maximum group-nesting depth in a grammar.
const MAX_DEPTH: usize = 256;
/// Maximum total grammar elements + rules produced. Bounds per-request memory; a bounded `{m,n}` still
/// multiplies, so the aggregate is capped.
const MAX_BUDGET: usize = 4_000_000;

impl Parser {
    fn new(src: &str) -> Self {
        Parser {
            src: src.chars().collect(),
            pos: 0,
            rules: Vec::new(),
            names: HashMap::new(),
            depth: 0,
            budget: MAX_BUDGET,
        }
    }

    /// Charge `n` against the element/rule budget; error (rather than allocate) once it is exhausted.
    fn charge(&mut self, n: usize) -> Result<(), String> {
        self.budget = self
            .budget
            .checked_sub(n)
            .ok_or_else(|| "grammar exceeds the size budget".to_string())?;
        Ok(())
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// Skip spaces, tabs, newlines, and `# ...` line comments.
    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                Some(' ') | Some('\t') | Some('\r') | Some('\n') => {
                    self.pos += 1;
                }
                Some('#') => {
                    while let Some(c) = self.peek() {
                        self.pos += 1;
                        if c == '\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// Get (or reserve) the rule index for a name; forward references reserve an empty rule to fill later.
    fn rule_id(&mut self, name: &str) -> Result<usize, String> {
        if let Some(&id) = self.names.get(name) {
            Ok(id)
        } else {
            self.charge(1)?;
            let id = self.rules.len();
            self.rules.push(Vec::new());
            self.names.insert(name.to_string(), id);
            Ok(id)
        }
    }

    /// Reserve a fresh anonymous rule (for a desugared group / repetition) and return its index.
    fn alloc_rule(&mut self) -> Result<usize, String> {
        self.charge(1)?;
        let id = self.rules.len();
        self.rules.push(Vec::new());
        Ok(id)
    }

    fn parse(mut self) -> Result<Grammar, String> {
        loop {
            self.skip_ws();
            if self.peek().is_none() {
                break;
            }
            self.parse_rule()?;
        }
        let root = *self
            .names
            .get("root")
            .ok_or_else(|| "grammar has no `root` rule".to_string())?;
        // every referenced/defined rule must be non-empty (an empty rule = referenced but never defined).
        for (name, &id) in &self.names {
            if self.rules[id].is_empty() {
                return Err(format!("rule `{name}` is referenced but never defined"));
            }
        }
        Ok(Grammar {
            rules: self.rules,
            root,
        })
    }

    fn parse_name(&mut self) -> Result<String, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(format!("expected a rule name at offset {}", self.pos));
        }
        Ok(self.src[start..self.pos].iter().collect())
    }

    fn parse_rule(&mut self) -> Result<(), String> {
        let name = self.parse_name()?;
        self.skip_ws();
        // accept `::=` (GBNF) - be lenient and also accept `=`.
        if self.peek() == Some(':') {
            self.expect_str("::=")?;
        } else {
            self.expect_str("=")?;
        }
        let id = self.rule_id(&name)?;
        let mut elems = self.parse_alternates()?;
        elems.push(Element::End);
        // fill the (possibly forward-declared) rule; reject a duplicate definition.
        if !self.rules[id].is_empty() {
            return Err(format!("rule `{name}` is defined more than once"));
        }
        self.set_rule(id, elems)?;
        Ok(())
    }

    fn expect_str(&mut self, s: &str) -> Result<(), String> {
        for want in s.chars() {
            if self.bump() != Some(want) {
                return Err(format!("expected `{s}` at offset {}", self.pos));
            }
        }
        Ok(())
    }

    /// Parse `seq ( "|" seq )*` into a flat element list with `Alt` between alternatives (no trailing End).
    fn parse_alternates(&mut self) -> Result<Vec<Element>, String> {
        let mut out = self.parse_sequence()?;
        loop {
            self.skip_ws();
            if self.peek() == Some('|') {
                self.pos += 1;
                out.push(Element::Alt);
                let seq = self.parse_sequence()?;
                out.extend(seq);
            } else {
                break;
            }
        }
        Ok(out)
    }

    /// Parse a sequence of terms until `|`, `)`, or end-of-input. An empty sequence is legal (an empty
    /// alternative, e.g. the tail of `x?`).
    fn parse_sequence(&mut self) -> Result<Vec<Element>, String> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None | Some('|') | Some(')') => break,
                // a newline-separated next rule: if what follows looks like `name ::=`, stop this rule.
                _ => {}
            }
            if self.at_rule_header() {
                break;
            }
            let term = self.parse_term()?;
            out.extend(term);
        }
        Ok(out)
    }

    /// Lookahead: are we at the start of a new `name ::=` rule header (so the current sequence ends here)?
    fn at_rule_header(&self) -> bool {
        let mut i = self.pos;
        let mut saw = false;
        while let Some(&c) = self.src.get(i) {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                i += 1;
                saw = true;
            } else {
                break;
            }
        }
        if !saw {
            return false;
        }
        while let Some(&c) = self.src.get(i) {
            if c == ' ' || c == '\t' {
                i += 1;
            } else {
                break;
            }
        }
        self.src.get(i) == Some(&':')
            && self.src.get(i + 1) == Some(&':')
            && self.src.get(i + 2) == Some(&'=')
    }

    /// Parse one atom plus an optional `* + ?` postfix, returning its element sequence.
    fn parse_term(&mut self) -> Result<Vec<Element>, String> {
        let atom = self.parse_atom()?;
        self.skip_postfix_ws();
        match self.peek() {
            Some('*') => {
                self.pos += 1;
                Ok(vec![Element::Ref(self.make_star(atom)?)])
            }
            Some('+') => {
                self.pos += 1;
                Ok(vec![Element::Ref(self.make_plus(atom)?)])
            }
            Some('?') => {
                self.pos += 1;
                Ok(vec![Element::Ref(self.make_opt(atom)?)])
            }
            Some('{') => {
                let (min, max) = self.parse_repeat_bounds()?;
                Ok(vec![Element::Ref(self.make_bounded(atom, min, max)?)])
            }
            _ => Ok(atom),
        }
    }

    /// Parse a bounded-repetition suffix `{m}`, `{m,}`, or `{m,n}` (llama.cpp GBNF). Returns `(min, max)`
    /// where `max` is `None` for the open-ended `{m,}`.
    fn parse_repeat_bounds(&mut self) -> Result<(usize, Option<usize>), String> {
        self.pos += 1; // consume '{'
        let min = self.parse_uint()?;
        let max = match self.peek() {
            Some('}') => {
                self.pos += 1;
                Some(min) // {m} - exactly m
            }
            Some(',') => {
                self.pos += 1;
                if self.peek() == Some('}') {
                    self.pos += 1;
                    None // {m,} - m or more
                } else {
                    let n = self.parse_uint()?;
                    if self.bump() != Some('}') {
                        return Err(format!(
                            "expected `}}` in repetition at offset {}",
                            self.pos
                        ));
                    }
                    if n < min {
                        return Err(format!("repetition {{{min},{n}}} has max < min"));
                    }
                    Some(n) // {m,n}
                }
            }
            other => {
                return Err(format!(
                    "expected `,` or `}}` in repetition, got {other:?} at offset {}",
                    self.pos
                ));
            }
        };
        // bound the desugaring so a huge {0,1000000} cannot explode the rule table.
        const MAX_REPEAT: usize = 1024;
        if min > MAX_REPEAT || max.is_some_and(|n| n > MAX_REPEAT) {
            return Err(format!("repetition bound exceeds {MAX_REPEAT}"));
        }
        Ok((min, max))
    }

    fn parse_uint(&mut self) -> Result<usize, String> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!("expected a number at offset {}", self.pos));
        }
        self.src[start..self.pos]
            .iter()
            .collect::<String>()
            .parse()
            .map_err(|_| "repetition count out of range".to_string())
    }

    /// Whitespace between an atom and its postfix must not cross into the next line's rule; we only skip
    /// inline spaces/tabs so a `*` on the same line binds, matching GBNF conventions.
    fn skip_postfix_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Assign `elems` to rule `id`, charging its size against the budget.
    fn set_rule(&mut self, id: usize, elems: Vec<Element>) -> Result<(), String> {
        self.charge(elems.len())?;
        self.rules[id] = elems;
        Ok(())
    }

    /// `R ::= atom R | ` (zero or more).
    fn make_star(&mut self, atom: Vec<Element>) -> Result<usize, String> {
        let id = self.alloc_rule()?;
        let mut elems = atom;
        elems.push(Element::Ref(id));
        elems.push(Element::Alt); // empty second alternative
        elems.push(Element::End);
        self.set_rule(id, elems)?;
        Ok(id)
    }

    /// `R ::= atom R | atom` (one or more).
    fn make_plus(&mut self, atom: Vec<Element>) -> Result<usize, String> {
        let id = self.alloc_rule()?;
        let mut elems = atom.clone();
        elems.push(Element::Ref(id));
        elems.push(Element::Alt);
        elems.extend(atom);
        elems.push(Element::End);
        self.set_rule(id, elems)?;
        Ok(id)
    }

    /// `R ::= atom | ` (optional).
    fn make_opt(&mut self, atom: Vec<Element>) -> Result<usize, String> {
        let id = self.alloc_rule()?;
        let mut elems = atom;
        elems.push(Element::Alt); // empty second alternative
        elems.push(Element::End);
        self.set_rule(id, elems)?;
        Ok(id)
    }

    /// Bounded repetition `atom{min,max}`: `min` required copies of `atom`, then an optional tail matching up
    /// to `max - min` more (or `atom*` when `max` is `None`). Desugars into synthetic rules.
    fn make_bounded(
        &mut self,
        atom: Vec<Element>,
        min: usize,
        max: Option<usize>,
    ) -> Result<usize, String> {
        // charge the min inline copies up front so a huge `atom{min}` errors before it is materialized.
        self.charge(min.saturating_mul(atom.len()))?;
        let tail: Option<usize> = match max {
            None => Some(self.make_star(atom.clone())?), // {min,} = min copies then atom*
            Some(n) if n > min => Some(self.opt_upto(atom.clone(), n - min)?),
            Some(_) => None, // {min} or {min,min} = exactly min copies, no tail
        };
        let id = self.alloc_rule()?;
        let mut elems = Vec::new();
        for _ in 0..min {
            elems.extend(atom.clone());
        }
        if let Some(t) = tail {
            elems.push(Element::Ref(t));
        }
        elems.push(Element::End);
        // the min copies were pre-charged; charge only the tail/End here.
        self.charge(elems.len().saturating_sub(min.saturating_mul(atom.len())))?;
        self.rules[id] = elems;
        Ok(id)
    }

    /// A rule matching between 0 and `k` copies of `atom`: `R_k ::= atom R_{k-1} | ` (empty), `R_0 ::= ` (empty).
    fn opt_upto(&mut self, atom: Vec<Element>, k: usize) -> Result<usize, String> {
        if k == 0 {
            let id = self.alloc_rule()?;
            self.set_rule(id, vec![Element::End])?; // a single empty alternative
            return Ok(id);
        }
        let inner = self.opt_upto(atom.clone(), k - 1)?;
        let id = self.alloc_rule()?;
        let mut elems = atom;
        elems.push(Element::Ref(inner));
        elems.push(Element::Alt); // empty second alternative
        elems.push(Element::End);
        self.set_rule(id, elems)?;
        Ok(id)
    }

    /// Parse a single atom: a string literal, a char class, a group, or a rule reference.
    fn parse_atom(&mut self) -> Result<Vec<Element>, String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => self.parse_string(),
            Some('[') => Ok(vec![self.parse_class()?]),
            // GBNF `.`: any single code point (newline included; use `[^\n]` to exclude it). Encoded as a
            // negated empty range set so `char_in` is true for every code point and postfix operators compose.
            Some('.') => {
                self.pos += 1;
                Ok(vec![Element::Char {
                    ranges: Vec::new(),
                    negated: true,
                }])
            }
            Some('(') => {
                // bound group nesting so a deeply-nested `((((...))))` cannot overflow the native stack.
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    return Err(format!("grammar nesting exceeds {MAX_DEPTH}"));
                }
                self.pos += 1;
                let inner = self.parse_alternates()?;
                self.skip_ws();
                if self.bump() != Some(')') {
                    return Err(format!("expected `)` at offset {}", self.pos));
                }
                self.depth -= 1;
                let id = self.alloc_rule()?;
                let mut elems = inner;
                elems.push(Element::End);
                self.set_rule(id, elems)?;
                Ok(vec![Element::Ref(id)])
            }
            Some(c) if c.is_ascii_alphanumeric() || c == '_' => {
                let name = self.parse_name()?;
                let id = self.rule_id(&name)?;
                Ok(vec![Element::Ref(id)])
            }
            other => Err(format!(
                "unexpected {:?} at offset {} in grammar",
                other, self.pos
            )),
        }
    }

    /// Parse a `"..."` literal into a sequence of single-char terminals.
    fn parse_string(&mut self) -> Result<Vec<Element>, String> {
        self.pos += 1; // opening quote
        let mut out = Vec::new();
        loop {
            match self.bump() {
                None => return Err("unterminated string literal".to_string()),
                Some('"') => break,
                Some('\\') => {
                    let c = self.parse_escape()?;
                    out.push(Element::Char {
                        ranges: vec![(c, c)],
                        negated: false,
                    });
                }
                Some(c) => out.push(Element::Char {
                    ranges: vec![(c as u32, c as u32)],
                    negated: false,
                }),
            }
        }
        Ok(out)
    }

    /// Parse a `[...]` character class into one terminal element.
    fn parse_class(&mut self) -> Result<Element, String> {
        self.pos += 1; // opening bracket
        let negated = if self.peek() == Some('^') {
            self.pos += 1;
            true
        } else {
            false
        };
        let mut ranges = Vec::new();
        loop {
            match self.peek() {
                None => return Err("unterminated character class".to_string()),
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                Some(_) => {
                    let lo = self.parse_class_char()?;
                    // a range `a-z`? (but a trailing `-` before `]` is a literal dash)
                    if self.peek() == Some('-') && self.src.get(self.pos + 1) != Some(&']') {
                        self.pos += 1; // consume '-'
                        let hi = self.parse_class_char()?;
                        ranges.push((lo.min(hi), lo.max(hi)));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
            }
        }
        if ranges.is_empty() {
            return Err("empty character class".to_string());
        }
        Ok(Element::Char { ranges, negated })
    }

    fn parse_class_char(&mut self) -> Result<u32, String> {
        match self.bump() {
            None => Err("unterminated character class".to_string()),
            Some('\\') => self.parse_escape(),
            Some(c) => Ok(c as u32),
        }
    }

    /// Parse the character after a backslash (in a string or class).
    fn parse_escape(&mut self) -> Result<u32, String> {
        match self.bump() {
            None => Err("dangling escape".to_string()),
            Some('n') => Ok('\n' as u32),
            Some('r') => Ok('\r' as u32),
            Some('t') => Ok('\t' as u32),
            Some('\\') => Ok('\\' as u32),
            Some('"') => Ok('"' as u32),
            Some('\'') => Ok('\'' as u32),
            Some('[') => Ok('[' as u32),
            Some(']') => Ok(']' as u32),
            Some('-') => Ok('-' as u32),
            Some('/') => Ok('/' as u32),
            Some('x') => self.parse_hex(2),
            Some('u') => self.parse_hex(4),
            Some('U') => self.parse_hex(8),
            Some(c) => Ok(c as u32), // unknown escape: the literal character
        }
    }

    fn parse_hex(&mut self, n: usize) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..n {
            let d = self
                .bump()
                .and_then(|c| c.to_digit(16))
                .ok_or_else(|| "bad hex escape".to_string())?;
            v = v * 16 + d;
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    fn g(src: &str) -> Arc<Grammar> {
        Arc::new(Grammar::parse(src).expect("parse grammar"))
    }

    /// Does the incremental engine accept `s` as a full sentence?
    fn accepts(gr: &Arc<Grammar>, s: &str) -> bool {
        let mut st = gr.start_state();
        for c in s.chars() {
            if !st.accept_char(c) {
                return false;
            }
        }
        st.is_complete()
    }

    // ---- independent backtracking recognizer (the oracle) --------------------
    //
    // Independent of the incremental stack engine: a memoized recursive matcher that, given a stack of
    // pending grammar positions and an input, returns the set of input indices at which the stack can
    // finish. `recog_full(s)` is then membership; the same routine yields the reachable-index set that
    // validates prefix liveness.

    type Memo = HashMap<(Vec<Pos>, usize), BTreeSet<usize>>;

    fn ends(gr: &Grammar, stack: &[Pos], s: &[char], i: usize, memo: &mut Memo) -> BTreeSet<usize> {
        if let Some(hit) = memo.get(&(stack.to_vec(), i)) {
            return hit.clone();
        }
        let mut res = BTreeSet::new();
        match stack.last() {
            None => {
                res.insert(i); // stack empty: finished here
            }
            Some(&(r, idx)) => {
                let rest = &stack[..stack.len() - 1];
                match &gr.rules[r][idx] {
                    Element::Char { ranges, negated } => {
                        if i < s.len() && char_in(ranges, *negated, s[i] as u32) {
                            let mut cont = rest.to_vec();
                            if !gr.is_end_of_seq(r, idx + 1) {
                                cont.push((r, idx + 1));
                            }
                            res = ends(gr, &cont, s, i + 1, memo);
                        }
                    }
                    Element::Ref(sub) => {
                        let sub = *sub;
                        let mut base = rest.to_vec();
                        if !gr.is_end_of_seq(r, idx + 1) {
                            base.push((r, idx + 1));
                        }
                        let mut j = 0usize;
                        loop {
                            let mut ns = base.clone();
                            if !gr.is_end_of_seq(sub, j) {
                                ns.push((sub, j));
                            }
                            for e in ends(gr, &ns, s, i, memo) {
                                res.insert(e);
                            }
                            let mut k = j;
                            while !gr.is_end_of_seq(sub, k) {
                                k += 1;
                            }
                            if matches!(gr.rules[sub].get(k), Some(Element::Alt)) {
                                j = k + 1;
                            } else {
                                break;
                            }
                        }
                    }
                    Element::Alt | Element::End => {
                        res = ends(gr, rest, s, i, memo);
                    }
                }
            }
        }
        memo.insert((stack.to_vec(), i), res.clone());
        res
    }

    /// The union of finish-index sets over every alternative of `root`.
    fn root_ends(gr: &Grammar, s: &[char], memo: &mut Memo) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        let mut j = 0usize;
        loop {
            let mut ns: Vec<Pos> = Vec::new();
            if !gr.is_end_of_seq(gr.root, j) {
                ns.push((gr.root, j));
            }
            for e in ends(gr, &ns, s, 0, memo) {
                out.insert(e);
            }
            let mut k = j;
            while !gr.is_end_of_seq(gr.root, k) {
                k += 1;
            }
            if matches!(gr.rules[gr.root].get(k), Some(Element::Alt)) {
                j = k + 1;
            } else {
                break;
            }
        }
        out
    }

    /// Independent membership: the grammar derives exactly `s`.
    fn recog_full(gr: &Grammar, s: &[char]) -> bool {
        let mut memo = Memo::new();
        root_ends(gr, s, &mut memo).contains(&s.len())
    }

    #[test]
    fn parses_and_accepts_simple_alternation() {
        let gr = g(r#"root ::= "yes" | "no""#);
        assert!(accepts(&gr, "yes"));
        assert!(accepts(&gr, "no"));
        assert!(!accepts(&gr, "maybe"));
        assert!(!accepts(&gr, "ye")); // incomplete
        assert!(!accepts(&gr, "yess")); // overrun
    }

    #[test]
    fn char_classes_and_repetition() {
        let gr = g(r#"root ::= [a-z]+ [0-9]*"#);
        assert!(accepts(&gr, "abc"));
        assert!(accepts(&gr, "abc12"));
        assert!(accepts(&gr, "z"));
        assert!(!accepts(&gr, "")); // + needs at least one
        assert!(!accepts(&gr, "1")); // must start with a letter
        assert!(!accepts(&gr, "a1b")); // digits only after letters
    }

    #[test]
    fn bounded_repetition() {
        // exactly m
        let g4 = g(r#"root ::= [0-9]{4}"#);
        assert!(accepts(&g4, "2026"));
        assert!(!accepts(&g4, "202")); // too few
        assert!(!accepts(&g4, "20261")); // too many
        // m or more
        let gm = g(r#"root ::= "a"{2,}"#);
        assert!(!accepts(&gm, "a"));
        assert!(accepts(&gm, "aa"));
        assert!(accepts(&gm, "aaaaa"));
        // m to n
        let gmn = g(r#"root ::= "x"{2,3} "!""#);
        assert!(!accepts(&gmn, "x!"));
        assert!(accepts(&gmn, "xx!"));
        assert!(accepts(&gmn, "xxx!"));
        assert!(!accepts(&gmn, "xxxx!"));
        // {0,n} allows the empty case
        let g0 = g(r#"root ::= "-" "a"{0,2} "-""#);
        assert!(accepts(&g0, "--"));
        assert!(accepts(&g0, "-aa-"));
        assert!(!accepts(&g0, "-aaa-"));
        // bad bounds / overflow are parse errors.
        assert!(Grammar::parse(r#"root ::= "a"{3,1}"#).is_err()); // max < min
        assert!(Grammar::parse(r#"root ::= "a"{99999}"#).is_err()); // exceeds cap
        assert!(Grammar::parse(r#"root ::= "a"{}"#).is_err()); // no count
    }

    #[test]
    fn dot_matches_any_single_char() {
        // GBNF `.` consumes exactly one code point of ANY value (including newline + non-ASCII).
        let one = g(r#"root ::= ."#);
        assert!(accepts(&one, "a"));
        assert!(accepts(&one, "Z"));
        assert!(accepts(&one, "7"));
        assert!(accepts(&one, "\n")); // any-char includes newline (use `[^\n]` to exclude)
        assert!(accepts(&one, "\u{03bb}")); // a non-ASCII code point
        assert!(!accepts(&one, "ab")); // exactly one, not more
        assert!(!accepts(&one, "")); // and not zero

        // Postfix repetition composes with `.` for free (it is a single terminal element like `[...]`).
        let four = g(r#"root ::= .{4}"#);
        assert!(accepts(&four, "abcd"));
        assert!(accepts(&four, "1 3.")); // spaces + punctuation are ordinary code points
        assert!(!accepts(&four, "abc"));
        assert!(!accepts(&four, "abcde"));

        // `.` mixes with literals and alternation.
        let mixed = g(r#"root ::= "x" . "z""#);
        assert!(accepts(&mixed, "xyz"));
        assert!(accepts(&mixed, "x_z"));
        assert!(!accepts(&mixed, "xz")); // the middle `.` is mandatory
    }

    #[test]
    fn recursion_balanced_parens() {
        // an arbitrarily deep, context-free language (a regex/DFA cannot count these).
        let gr = g(r#"root ::= "(" root ")" | "" "#);
        assert!(accepts(&gr, ""));
        assert!(accepts(&gr, "()"));
        assert!(accepts(&gr, "((()))"));
        assert!(!accepts(&gr, "(()")); // unbalanced
        assert!(!accepts(&gr, "())"));
    }

    #[test]
    fn optional_and_groups() {
        let gr = g(r#"root ::= "a" ("b" | "c")? "d""#);
        assert!(accepts(&gr, "ad"));
        assert!(accepts(&gr, "abd"));
        assert!(accepts(&gr, "acd"));
        assert!(!accepts(&gr, "abcd"));
        assert!(!accepts(&gr, "abad"));
    }

    #[test]
    fn negated_class() {
        let gr = g(r#"root ::= "\"" [^"]* "\"" "#);
        assert!(accepts(&gr, r#""""#));
        assert!(accepts(&gr, r#""hello world""#));
        assert!(!accepts(&gr, r#""un"terminated""#));
    }

    #[test]
    fn csv_of_numbers_completes_at_each_full_item() {
        let gr = g("root ::= item (\",\" item)*\nitem ::= [0-9]+");
        assert!(accepts(&gr, "0"));
        assert!(accepts(&gr, "12"));
        assert!(accepts(&gr, "1,2,3"));
        assert!(accepts(&gr, "10,20"));
        assert!(!accepts(&gr, "1,")); // trailing comma
        assert!(!accepts(&gr, ",1"));
        assert!(!accepts(&gr, "1,,2"));
    }

    #[test]
    fn errors_on_missing_root_and_undefined_ref() {
        assert!(Grammar::parse(r#"foo ::= "x""#).is_err()); // no root
        assert!(Grammar::parse(r#"root ::= bar"#).is_err()); // bar undefined
        assert!(Grammar::parse(r#"root ::= "a" root ::= "b""#).is_err()); // duplicate def
    }

    #[test]
    fn adversarial_input_errors_instead_of_crashing() {
        // `guided_grammar` is untrusted API input: a malicious grammar must fail cleanly.

        // deeply-nested groups: without a depth guard this recurses ~N frames and aborts the process.
        let deep = format!("root ::= {}\"x\"{}", "(".repeat(4000), ")".repeat(4000));
        assert!(
            Grammar::parse(&deep).is_err(),
            "deep nesting must error, not overflow the stack"
        );
        // but a reasonable nesting depth still parses and works.
        let ok = format!("root ::= {}\"x\"{}", "(".repeat(50), ")".repeat(50));
        let gr = g(&ok);
        assert!(accepts(&gr, "x"));

        // repetition-driven element explosion: a large atom repeated a large bounded number of times must
        // exceed the size budget and error, not allocate gigabytes.
        let big = format!("root ::= \"{}\"{{1024}}", "a".repeat(5000));
        assert!(
            Grammar::parse(&big).is_err(),
            "an over-budget repetition must error"
        );

        // an unterminated deep-nest is also a clean error (not a hang).
        assert!(Grammar::parse(&"(".repeat(1000)).is_err());
    }

    /// Brute-force every string over a small alphabet up to length L and assert the engine's full-match
    /// verdict equals the independent recognizer's. Also assert prefix liveness: for every accepted
    /// string, each prefix is walkable and `is_complete` matches the recognizer.
    #[test]
    fn engine_membership_matches_the_independent_recognizer() {
        let cases: [(&str, &str, usize); 7] = [
            (r#"root ::= "yes" | "no" | "maybe""#, "yesnomab", 4),
            (r#"root ::= [a-c]+ ("!" | "?")"#, "abc!?", 4),
            (r#"root ::= "(" root ")" | "x""#, "()x", 6),
            ("root ::= item (\",\" item)*\nitem ::= [0-9]", "0129,", 5),
            (r#"root ::= "a"? "b"? "c"?"#, "abc", 3),
            (r#"root ::= "a"{2,4}"#, "ab", 5), // bounded repetition
            (r#"root ::= [xy]{1,2} "z"{2}"#, "xyz", 5),
        ];
        for (src, alphabet, max_len) in cases {
            let gr = g(src);
            let alpha: Vec<char> = alphabet.chars().collect();
            let mut any_member = false;
            // enumerate all strings over `alpha` of length 0..=max_len.
            let mut frontier = vec![String::new()];
            for _len in 0..=max_len {
                let mut next = Vec::new();
                for s in &frontier {
                    let chars: Vec<char> = s.chars().collect();
                    let engine = accepts(&gr, s);
                    let oracle = recog_full(&gr, &chars);
                    assert_eq!(engine, oracle, "membership mismatch for {s:?} in {src:?}");
                    if oracle {
                        any_member = true;
                        // every prefix of a member must be walkable and complete-flag-correct.
                        let mut st = gr.start_state();
                        for cut in 0..=chars.len() {
                            let prefix: Vec<char> = chars[..cut].to_vec();
                            let mut memo = Memo::new();
                            let full_here =
                                root_ends(&gr, &prefix, &mut memo).contains(&prefix.len());
                            assert_eq!(
                                st.is_complete(),
                                full_here,
                                "complete-flag mismatch at prefix {prefix:?} of {s:?} in {src:?}"
                            );
                            if cut < chars.len() {
                                assert!(
                                    st.accept_char(chars[cut]),
                                    "engine stalls walking member {s:?} in {src:?}"
                                );
                            }
                        }
                    }
                    if s.chars().count() < max_len {
                        for &c in &alpha {
                            let mut t = s.clone();
                            t.push(c);
                            next.push(t);
                        }
                    }
                }
                frontier = next;
            }
            assert!(
                any_member,
                "no member found for {src:?} - test alphabet too small"
            );
        }
    }
}
