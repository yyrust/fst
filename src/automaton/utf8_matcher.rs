type Utf8StateId = i32;

const UTF8_INVALID_STATE: Utf8StateId = -1;
const UTF8_TERMINAL_STATE: Utf8StateId = 0;

struct Utf8Transition {
    start: u8,
    end: u8,
    next: Utf8StateId,
}

impl Utf8Transition {
    fn accept(&self, chr: u8) -> bool {
        self.start <= chr && chr <= self.end
    }
}

pub struct Utf8Table {
    trans_table: Vec<Utf8Transition>,
}

impl Utf8Table {
    pub fn new() -> Self {
        type Trans = Utf8Transition;
        let mut trans_table = vec![];
        /*
           [0-7F]
           [C2-DF]                  [80-BF]
           [E0]       [A0-BF]       [80-BF]
           [E1-EC]           [80-BF][80-BF]
           [ED]       [80-9F]       [80-BF]
           [EE-EF]           [80-BF][80-BF]
           [F0]       [90-BF][80-BF][80-BF]
           [F1-F3]    [80-BF][80-BF][80-BF]
           [F4]       [80-8F][80-BF][80-BF]
           (from https://docs.rs/utf8-ranges/1.0.4/utf8_ranges/)
        */
        // state 0, the terminal state
        trans_table.push(Trans {
            start: 0x00,
            end: 0x00,
            next: UTF8_INVALID_STATE,
        });
        // state 1, the only match state
        trans_table.push(Trans { start: 0x80, end: 0xBF, next: 0 });
        // state 2
        trans_table.push(Trans { start: 0x80, end: 0xBF, next: 1 });
        // state 3 from [E0]
        trans_table.push(Trans { start: 0xA0, end: 0xBF, next: 1 });
        // state 4 from [ED]
        trans_table.push(Trans { start: 0x80, end: 0x9F, next: 1 });
        // state 5 from [F0]
        trans_table.push(Trans { start: 0x90, end: 0xBF, next: 2 });
        // state 6 from [F1-F3]
        trans_table.push(Trans { start: 0x80, end: 0xBF, next: 2 });
        // state 7 from [F4]
        trans_table.push(Trans { start: 0x80, end: 0x8F, next: 2 });

        Self { trans_table }
    }

    pub fn start(&self, chr: u8) -> Utf8StateId {
        match chr {
            0x00..=0xC1 => UTF8_INVALID_STATE,
            0xC2..=0xDF => 0,
            0xE0 => 3,
            0xE1..=0xEC => 2,
            0xED => 4,
            0xEE..=0xEF => 2,
            0xF0 => 5,
            0xF1..=0xF3 => 6,
            0xF4 => 7,
            0xF5..=u8::MAX => UTF8_INVALID_STATE,
        }
    }
    pub fn next(&self, state: Utf8StateId, chr: u8) -> Utf8StateId {
        if state < 0 || state >= self.trans_table.len() as Utf8StateId {
            return UTF8_INVALID_STATE;
        }
        let trans = &self.trans_table[state as usize];
        if trans.accept(chr) {
            trans.next
        } else {
            UTF8_INVALID_STATE
        }
    }
    pub fn is_mismatch(state: Utf8StateId) -> bool {
        state < 0
    }
}

enum MatchResult {
    Terminated,
    Continue,
    Mismatch,
}

struct Utf8Matcher<'a> {
    utf8_table: &'a Utf8Table,
    bytes: Vec<u8>,
    states: Vec<Utf8StateId>,
}

impl<'a> Utf8Matcher<'a> {
    fn start(utf8_table: &'a Utf8Table, byte: u8) -> Option<Self> {
        let state = utf8_table.start(byte);
        if Utf8Table::is_mismatch(state) {
            None
        } else {
            Some(Self { utf8_table, bytes: vec![byte], states: vec![state] })
        }
    }
    fn next(&mut self, byte: u8) -> MatchResult {
        self.bytes.push(byte);
        let state = self
            .utf8_table
            .next(*self.states.last().unwrap_or(&UTF8_INVALID_STATE), byte);
        self.states.push(state);
        match state {
            UTF8_TERMINAL_STATE => MatchResult::Terminated,
            UTF8_INVALID_STATE => MatchResult::Mismatch,
            _ => MatchResult::Continue,
        }
    }
    fn decode_char(&self) -> Option<String> {
        Some(String::from_utf8_lossy(&self.bytes).into())
    }
    fn pop(&mut self) {
        self.bytes.pop();
        self.states.pop();
    }
}

type DfaStateId = usize;
pub type Transition = (u8, usize); // (byte of utf-8 seq, dfa state id)
pub type Transitions = Vec<Transition>;
pub type TransitionTable = Vec<Transitions>;

// find paths which match a certain utf-8 character
pub fn find_utf8_path<F: FnMut(DfaStateId, DfaStateId, String)>(
    table: &TransitionTable,
    handle_path: &mut F,
) {
    let utf8_table = Utf8Table::new();
    for (i, trans) in table.iter().enumerate() {
        for t in trans {
            if let Some(mut matcher) = Utf8Matcher::start(&utf8_table, t.0) {
                traverse_utf8_path(table, &mut matcher, i, t.1, handle_path);
            }
        }
    }
}

fn traverse_utf8_path<F: FnMut(DfaStateId, DfaStateId, String)>(
    table: &TransitionTable,
    mut matcher: &mut Utf8Matcher,
    init_state: DfaStateId,
    next: DfaStateId,
    handle_path: &mut F,
) {
    let trans = &table[next];
    for t in trans {
        match matcher.next(t.0) {
            MatchResult::Terminated => {
                if let Some(chr) = matcher.decode_char() {
                    handle_path(init_state, t.1, chr);
                }
            }
            MatchResult::Continue => {
                traverse_utf8_path(
                    table,
                    &mut matcher,
                    init_state,
                    t.1,
                    handle_path,
                );
            }
            MatchResult::Mismatch => {}
        }
        matcher.pop();
    }
}
