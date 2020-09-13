use std::cmp;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::Write;

use utf8_ranges::{Utf8Range, Utf8Sequences};

use super::utf8_matcher::{find_utf8_path, TransitionTable, Transitions};
use crate::automaton::Automaton;

const STATE_LIMIT: usize = 10_000; // currently at least 20MB >_<

/// An error that occurred while building a Levenshtein automaton.
///
/// This error is only defined when the `levenshtein` crate feature is enabled.
#[derive(Debug)]
pub enum LevenshteinError {
    /// If construction of the automaton reaches some hard-coded limit
    /// on the number of states, then this error is returned.
    ///
    /// The number given is the limit that was exceeded.
    TooManyStates(usize),
}

impl fmt::Display for LevenshteinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            LevenshteinError::TooManyStates(size_limit) => write!(
                f,
                "Levenshtein automaton exceeds size limit of \
                           {} states",
                size_limit
            ),
        }
    }
}

impl std::error::Error for LevenshteinError {}

/// A Unicode aware Levenshtein automaton for running efficient fuzzy queries.
///
/// This is only defined when the `levenshtein` crate feature is enabled.
///
/// A Levenshtein automata is one way to search any finite state transducer
/// for keys that *approximately* match a given query. A Levenshtein automaton
/// approximates this by returning all keys within a certain edit distance of
/// the query. The edit distance is defined by the number of insertions,
/// deletions and substitutions required to turn the query into the key.
/// Insertions, deletions and substitutions are based on
/// **Unicode characters** (where each character is a single Unicode scalar
/// value).
///
/// # Example
///
/// This example shows how to find all keys within an edit distance of `1`
/// from `foo`.
///
/// ```rust
/// use fst::automaton::Levenshtein;
/// use fst::{IntoStreamer, Streamer, Set};
///
/// fn main() {
///     let keys = vec!["fa", "fo", "fob", "focus", "foo", "food", "foul"];
///     let set = Set::from_iter(keys).unwrap();
///
///     let lev = Levenshtein::new("foo", 1).unwrap();
///     let mut stream = set.search(&lev).into_stream();
///
///     let mut keys = vec![];
///     while let Some(key) = stream.next() {
///         keys.push(key.to_vec());
///     }
///     assert_eq!(keys, vec![
///         "fo".as_bytes(),   // 1 deletion
///         "fob".as_bytes(),  // 1 substitution
///         "foo".as_bytes(),  // 0 insertions/deletions/substitutions
///         "food".as_bytes(), // 1 insertion
///     ]);
/// }
/// ```
///
/// This example only uses ASCII characters, but it will work equally well
/// on Unicode characters.
///
/// # Warning: experimental
///
/// While executing this Levenshtein automaton against a finite state
/// transducer will be very fast, *constructing* an automaton may not be.
/// Namely, this implementation is a proof of concept. While I believe the
/// algorithmic complexity is not exponential, the implementation is not speedy
/// and it can use enormous amounts of memory (tens of MB before a hard-coded
/// limit will cause an error to be returned).
///
/// This is important functionality, so one should count on this implementation
/// being vastly improved in the future.
pub struct Levenshtein {
    prog: DynamicLevenshtein,
    dfa: Dfa,
}

impl Levenshtein {
    /// Create a new Levenshtein query.
    ///
    /// The query finds all matching terms that are at most `distance`
    /// edit operations from `query`. (An edit operation may be an insertion,
    /// a deletion or a substitution.)
    ///
    /// If the underlying automaton becomes too big, then an error is returned.
    ///
    /// A `Levenshtein` value satisfies the `Automaton` trait, which means it
    /// can be used with the `search` method of any finite state transducer.
    #[inline]
    pub fn new(
        query: &str,
        distance: u32,
    ) -> Result<Levenshtein, LevenshteinError> {
        let lev = DynamicLevenshtein {
            query: query.to_owned(),
            dist: distance as usize,
        };
        let dfa = DfaBuilder::new(lev.clone()).build()?;
        Ok(Levenshtein { prog: lev, dfa })
    }

    /// Generate graph to `w` for the automaton in DOT language.
    #[inline]
    pub fn to_graphviz(
        &self,
        name: &str,
        w: &mut dyn Write,
    ) -> std::io::Result<()> {
        self.dfa.to_graphviz(name, w)
    }
}

impl fmt::Debug for Levenshtein {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Levenshtein(query: {:?}, distance: {:?})",
            self.prog.query, self.prog.dist
        )
    }
}

#[derive(Clone)]
struct DynamicLevenshtein {
    query: String,
    dist: usize,
}

impl DynamicLevenshtein {
    fn start(&self) -> Vec<usize> {
        (0..self.query.chars().count() + 1).collect()
    }

    fn is_match(&self, state: &[usize]) -> bool {
        state.last().map(|&n| n <= self.dist).unwrap_or(false)
    }

    fn can_match(&self, state: &[usize]) -> bool {
        state.iter().min().map(|&n| n <= self.dist).unwrap_or(false)
    }

    fn accept(&self, state: &[usize], chr: Option<char>) -> Vec<usize> {
        let mut next = vec![state[0] + 1];
        for (i, c) in self.query.chars().enumerate() {
            let cost = if Some(c) == chr { 0 } else { 1 };
            let v = cmp::min(
                cmp::min(next[i] + 1, state[i + 1] + 1),
                state[i] + cost,
            );
            next.push(cmp::min(v, self.dist + 1));
        }
        next
    }
}

impl Automaton for Levenshtein {
    type State = Option<usize>;

    #[inline]
    fn start(&self) -> Option<usize> {
        Some(0)
    }

    #[inline]
    fn is_match(&self, state: &Option<usize>) -> bool {
        state.map(|state| self.dfa.states[state].is_match).unwrap_or(false)
    }

    #[inline]
    fn can_match(&self, state: &Option<usize>) -> bool {
        state.is_some()
    }

    #[inline]
    fn accept(&self, state: &Option<usize>, byte: u8) -> Option<usize> {
        state.and_then(|state| self.dfa.states[state].next[byte as usize])
    }
}

#[derive(Debug)]
struct Dfa {
    states: Vec<State>,
}

impl Dfa {
    pub fn to_graphviz(
        &self,
        name: &str,
        w: &mut dyn Write,
    ) -> std::io::Result<()> {
        let mut transition_table =
            TransitionTable::with_capacity(self.states.len());
        w.write_fmt(format_args!("digraph {} {{\n", name))?;
        for (i, s) in self.states.iter().enumerate() {
            s.to_graphviz(i, w)?;

            // transitions for the state
            let mut transitions = Transitions::default();
            let clusters = s.clusters();
            for cl in clusters
                .iter()
                .filter(|c| c.from + 1 == c.to && c.value.is_some())
            {
                transitions.push((cl.from as u8, cl.value.unwrap()));
            }
            transition_table.push(transitions);
        }

        // Find paths which match a multi-byte utf-8 character, and generate blue dashed lines to
        // visualize them.
        find_utf8_path(&transition_table, &mut |start, end, chr| {
            w.write_fmt(format_args!("S{} -> S{} [color=blue, constrant=false, label=<<font color='blue'>{}</font>>, style=dashed];\n", start, end, chr)).expect("write virtual path error");
        });

        w.write_fmt(format_args!("}}\n"))
    }
}

struct State {
    next: [Option<usize>; 256],
    is_match: bool,
}

struct StateCluster {
    pub from: usize,
    pub to: usize,
    pub value: Option<usize>,
}

impl StateCluster {
    pub fn edge_label(&self) -> String {
        if self.from + 1 == self.to {
            // single char
            let c = self.from;
            let printable = c >= 0x20 && c <= 0x7e;
            if printable {
                format!("0x{:X} ('{}')", self.from, c as u8 as char)
            } else {
                format!("0x{:X}", self.from)
            }
        } else {
            // char range
            format!("[{:X}-{:X}]", self.from, self.to - 1)
        }
    }
    pub fn edge_target(&self) -> String {
        match self.value {
            Some(v) => format!("S{}", v),
            None => "None".to_string(),
        }
    }
    pub fn edge_color(&self) -> &str {
        if self.from + 1 == self.to {
            "red"
        } else {
            "black"
        }
    }
}

impl fmt::Debug for StateCluster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}): ", self.from, self.to)?;
        match self.value {
            Some(x) => write!(f, "{}", x),
            None => write!(f, "None"),
        }
    }
}

impl State {
    pub fn clusters(&self) -> Vec<StateCluster> {
        let mut prev_idx = 0;
        let mut prev = self.next[0];
        let mut clusters = vec![];
        for i in 1..self.next.len() {
            let curr = self.next[i];
            if curr != prev {
                clusters.push(StateCluster {
                    from: prev_idx,
                    to: i,
                    value: prev,
                });
                prev = curr;
                prev_idx = i;
            }
        }
        clusters.push(StateCluster {
            from: prev_idx,
            to: self.next.len(),
            value: *self.next.last().unwrap(),
        });
        clusters
    }
    pub fn to_graphviz(
        &self,
        id: usize,
        w: &mut dyn Write,
    ) -> std::io::Result<()> {
        let shape = if self.is_match { "box" } else { "circle" };
        w.write_fmt(format_args!("S{} [shape=\"{}\"];\n", id, shape))?;
        let clusters = self.clusters();
        for cl in clusters {
            if cl.value.is_none() {
                continue;
            }
            let target = cl.edge_target();
            let label = cl.edge_label();
            let color = cl.edge_color();
            w.write_fmt(format_args!(
                "S{} -> {} [label=<<font color='{}'>{}</font>>, color=\"{}\"];\n",
                id, target, color, label, color
            ))?;
        }
        Ok(())
    }
}

impl PartialEq for State {
    fn eq(&self, other: &State) -> bool {
        if self.is_match != other.is_match {
            return false;
        }
        for i in 0..256 {
            if self.next[i] != other.next[i] {
                return false;
            }
        }
        return true;
    }
}

impl Eq for State {}

impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for n in self.next.iter() {
            match n {
                Some(si) => state.write_usize(*si),
                None => state.write_isize(-1isize),
            }
        }
        state.write_u8(if self.is_match { 1 } else { 0 });
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "State {{")?;
        writeln!(f, "  is_match: {:?}", self.is_match)?;
        for i in 0..256 {
            if let Some(si) = self.next[i] {
                writeln!(f, "  {:?}: {:?}", i, si)?;
            }
        }
        write!(f, "}}")
    }
}

struct DfaBuilder {
    dfa: Dfa,
    lev: DynamicLevenshtein,
    cache: HashMap<Vec<usize>, usize>,
}

impl DfaBuilder {
    fn new(lev: DynamicLevenshtein) -> DfaBuilder {
        DfaBuilder {
            dfa: Dfa { states: Vec::with_capacity(16) },
            lev,
            cache: HashMap::with_capacity(1024),
        }
    }

    fn build(mut self) -> Result<Dfa, LevenshteinError> {
        let mut stack = vec![self.lev.start()];
        let mut seen = HashSet::new();
        let query = self.lev.query.clone(); // temp work around of borrowck
        while let Some(lev_state) = stack.pop() {
            let dfa_si = self.cached_state(&lev_state).unwrap();
            let mismatch = self.add_mismatch_utf8_states(dfa_si, &lev_state);
            if let Some((next_si, lev_next)) = mismatch {
                if !seen.contains(&next_si) {
                    seen.insert(next_si);
                    stack.push(lev_next);
                }
            }
            for (i, c) in query.chars().enumerate() {
                if lev_state[i] > self.lev.dist {
                    continue;
                }
                let lev_next = self.lev.accept(&lev_state, Some(c));
                let next_si = self.cached_state(&lev_next);
                if let Some(next_si) = next_si {
                    self.add_utf8_sequences(true, dfa_si, next_si, c, c);
                    if !seen.contains(&next_si) {
                        seen.insert(next_si);
                        stack.push(lev_next);
                    }
                }
            }
            if self.dfa.states.len() > STATE_LIMIT {
                return Err(LevenshteinError::TooManyStates(STATE_LIMIT));
            }
        }
        Ok(self.dfa)
    }

    fn cached_state(&mut self, lev_state: &[usize]) -> Option<usize> {
        self.cached(lev_state).map(|(si, _)| si)
    }

    fn cached(&mut self, lev_state: &[usize]) -> Option<(usize, bool)> {
        if !self.lev.can_match(lev_state) {
            return None;
        }
        Some(match self.cache.entry(lev_state.to_vec()) {
            Entry::Occupied(v) => (*v.get(), true),
            Entry::Vacant(v) => {
                let is_match = self.lev.is_match(lev_state);
                self.dfa.states.push(State { next: [None; 256], is_match });
                (*v.insert(self.dfa.states.len() - 1), false)
            }
        })
    }

    fn add_mismatch_utf8_states(
        &mut self,
        from_si: usize,
        lev_state: &[usize],
    ) -> Option<(usize, Vec<usize>)> {
        let mismatch_state = self.lev.accept(lev_state, None);
        let to_si = match self.cached(&mismatch_state) {
            None => return None,
            Some((si, _)) => si,
            // Some((si, true)) => return Some((si, mismatch_state)),
            // Some((si, false)) => si,
        };
        self.add_utf8_sequences(false, from_si, to_si, '\u{0}', '\u{10FFFF}');
        return Some((to_si, mismatch_state));
    }

    fn add_utf8_sequences(
        &mut self,
        overwrite: bool,
        from_si: usize,
        to_si: usize,
        from_chr: char,
        to_chr: char,
    ) {
        for seq in Utf8Sequences::new(from_chr, to_chr) {
            let mut fsi = from_si;
            for range in &seq.as_slice()[0..seq.len() - 1] {
                let tsi = self.new_state(false);
                self.add_utf8_range(overwrite, fsi, tsi, range);
                fsi = tsi;
            }
            self.add_utf8_range(
                overwrite,
                fsi,
                to_si,
                &seq.as_slice()[seq.len() - 1],
            );
        }
    }

    fn add_utf8_range(
        &mut self,
        overwrite: bool,
        from: usize,
        to: usize,
        range: &Utf8Range,
    ) {
        for b in range.start as usize..range.end as usize + 1 {
            if overwrite || self.dfa.states[from].next[b].is_none() {
                self.dfa.states[from].next[b] = Some(to);
            }
        }
    }

    fn new_state(&mut self, is_match: bool) -> usize {
        self.dfa.states.push(State { next: [None; 256], is_match });
        self.dfa.states.len() - 1
    }
}
