//! An exhaustive solver: the exact value of a position, found by plain
//! recursion over every legal move. It shares only the move generator with the
//! real solver (no proof numbers, no endgame analysis, no transposition table),
//! so it serves as ground truth for testing it. It is far too slow for early
//! positions; it gives up with `None` rather than exceed its time or memory.

use crate::game::{Game, GameState, Player};
use serde::Serialize;
use std::collections::HashMap;
use crate::clock::Instant;

/// The value of a position for the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Value {
    Win,
    Draw,
    Loss,
}

/// The most positions remembered, about 10 GB of memory.
const MAX_ENTRIES: usize = 150_000_000;
const FULL: u32 = 0x1ff;

/// Memo key: the whole position. A closed board is stored only as who, if
/// anyone, won it, since its cells can never matter again.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct Key(u64, u64, u64);

fn key(pos: &Game) -> Key {
    let done = pos.global_states.donebits();
    let won_x = pos.global_states.playerbits(Player::X);
    let won_o = pos.global_states.playerbits(Player::O);
    let (mut x, mut o) = (0u128, 0u128);
    for board in 0..9 {
        let bit = 1 << board;
        let (bx, bo) = if done & bit == 0 {
            (
                pos.local_boards.playerbits(Player::X, board),
                pos.local_boards.playerbits(Player::O, board),
            )
        } else {
            // Won by X: all X. Won by O: all O. Drawn: both.
            (
                if won_o & bit == 0 { FULL } else { 0 },
                if won_x & bit == 0 { FULL } else { 0 },
            )
        };
        x |= (bx as u128) << (9 * board);
        o |= (bo as u128) << (9 * board);
    }
    let next = pos.board_to_play().map_or(9, |b| b as u64)
        | match pos.player() {
            Player::X => 0,
            Player::O => 16,
        };
    Key(
        x as u64,
        o as u64,
        (x >> 64) as u64 | ((o >> 64) as u64) << 17 | next << 34,
    )
}

// Memo flags: whether each question has been answered, and the answer.
const WINS_KNOWN: u8 = 1;
const WINS: u8 = 2;
const HOLDS_KNOWN: u8 = 4;
const HOLDS: u8 = 8;

pub struct Oracle {
    memo: HashMap<Key, u8>,
    pub nodes: u64,
    deadline: Option<Instant>,
    /// Why the search gave up, if it did.
    pub stopped: Option<&'static str>,
}

impl Oracle {
    pub fn new(deadline: Option<Instant>) -> Self {
        Oracle {
            memo: HashMap::new(),
            nodes: 0,
            deadline,
            stopped: None,
        }
    }

    /// The exact value for the side to move, or None if the search gave up.
    pub fn value(&mut self, pos: &Game) -> Option<Value> {
        match pos.game_state() {
            GameState::Won(p) if p == pos.player() => return Some(Value::Win),
            GameState::Won(_) => return Some(Value::Loss),
            GameState::Drawn => return Some(Value::Draw),
            GameState::InPlay => (),
        }
        let value = if self.wins(pos) {
            Value::Win
        } else if self.holds(pos) {
            Value::Draw
        } else {
            Value::Loss
        };
        if self.stopped.is_some() {
            None
        } else {
            Some(value)
        }
    }

    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes % (1 << 20) == 0 && self.deadline.map_or(false, |d| Instant::now() >= d) {
            self.stopped = Some("time limit");
        }
    }

    fn remember(&mut self, k: Key, flags: u8) {
        // An answer computed after giving up may be wrong; never keep it.
        if self.stopped.is_some() {
            return;
        }
        if self.memo.len() >= MAX_ENTRIES {
            self.stopped = Some("memory limit");
            return;
        }
        *self.memo.entry(k).or_insert(0) |= flags;
    }

    /// Whether some move ends the game with the side to move winning, or,
    /// if `draw_counts`, drawn.
    fn ends_well(pos: &Game, draw_counts: bool) -> bool {
        let me = pos.player();
        pos.all_moves()
            .any(|m| match pos.make_move(m).unwrap().game_state() {
                GameState::Won(p) => p == me,
                GameState::Drawn => draw_counts,
                GameState::InPlay => false,
            })
    }

    /// Can the side to move force a win?
    fn wins(&mut self, pos: &Game) -> bool {
        let k = key(pos);
        match self.memo.get(&k) {
            Some(&f) if f & WINS_KNOWN != 0 => return f & WINS != 0,
            _ => (),
        }
        self.tick();
        let answer = Self::ends_well(pos, false)
            || pos.all_moves().any(|m| {
                let child = pos.make_move(m).unwrap();
                self.stopped.is_none()
                    && child.game_state() == GameState::InPlay
                    && !self.holds(&child)
            });
        self.remember(k, WINS_KNOWN | if answer { WINS } else { 0 });
        answer
    }

    /// Can the side to move avoid losing, by winning or drawing?
    fn holds(&mut self, pos: &Game) -> bool {
        let k = key(pos);
        match self.memo.get(&k) {
            Some(&f) if f & HOLDS_KNOWN != 0 => return f & HOLDS != 0,
            _ => (),
        }
        self.tick();
        let answer = Self::ends_well(pos, true)
            || pos.all_moves().any(|m| {
                let child = pos.make_move(m).unwrap();
                self.stopped.is_none()
                    && child.game_state() == GameState::InPlay
                    && !self.wins(&child)
            });
        self.remember(k, HOLDS_KNOWN | if answer { HOLDS } else { 0 });
        answer
    }
}
