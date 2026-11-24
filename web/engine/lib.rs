//! The engine for the web page, as WebAssembly: ultimattt's rules, its minimax
//! player and its proof-number solver, and uttt.ai's network and search
//! (`net` and `mcts`).
//!
//! The page writes a request, as text, into the buffer at `buffer()`, calls a
//! function with the request's length, and reads the answer, JSON, from the
//! same buffer: the function returns the answer's length. A request starts
//! with the game's moves, squares 0 to 80 separated by spaces, where square
//! 9 × board + cell, each numbered 0 to 8 from the top left, row by row. Any
//! settings follow, separated by `;`.
//!
//! The page supplies one import, `env.now_ms`, its clock (ultimattt's
//! `clock` module).

mod mcts;
mod net;

use mcts::Tree;
use net::Net;
use serde_json::{json, Value};
use std::ptr::{addr_of, addr_of_mut};
use std::time::Duration;
use ultimattt::clock::Instant;
use ultimattt::game::{CellState, Game, GameState, Move};
use ultimattt::minimax::{self, Minimax, EVAL_LOST, EVAL_WON};
use ultimattt::prove::{dfpn, Evaluation};

const BUFFER: usize = 1 << 16;
static mut IO: [u8; BUFFER] = [0; BUFFER];

/// The minimax players' transposition tables, and the solver's: small enough
/// for a phone.
const MINIMAX_TABLE: usize = 32 << 20;
const PROVE_TABLE: usize = 32 << 20;

/// The network's weights, as the page copies them in, then the network.
static mut NET_BYTES: Vec<u8> = Vec::new();
static mut NET: Option<Net> = None;

/// `think` keeps its search tree while the position stays the same.
static mut THINKER: Option<(String, Tree)> = None;

/// The bot keeps its search tree from one move to the next, as uttt.ai does,
/// with the moves that led to it.
static mut BOT_TREE: Option<(Vec<Move>, Tree)> = None;

#[no_mangle]
pub extern "C" fn buffer() -> *mut u8 {
    addr_of_mut!(IO) as *mut u8
}

/// Room for `n` bytes of network weights: the page copies `net.bin` there,
/// then calls `net_load`.
#[no_mangle]
pub extern "C" fn net_buffer(n: usize) -> *mut u8 {
    unsafe {
        let bytes = &mut *addr_of_mut!(NET_BYTES);
        *bytes = vec![0; n];
        bytes.as_mut_ptr()
    }
}

#[no_mangle]
pub extern "C" fn net_load() -> usize {
    let bytes = unsafe { std::mem::take(&mut *addr_of_mut!(NET_BYTES)) };
    match Net::load(&bytes) {
        Ok(net) => {
            unsafe { *addr_of_mut!(NET) = Some(net) };
            answer(json!({ "ok": true }))
        }
        Err(e) => answer(json!({ "error": e })),
    }
}

fn network() -> Result<&'static Net, Value> {
    unsafe { (*addr_of!(NET)).as_ref() }.ok_or_else(|| json!({ "error": "the network isn't loaded" }))
}

fn request(len: usize) -> Vec<String> {
    let bytes = unsafe { std::slice::from_raw_parts(addr_of!(IO) as *const u8, len.min(BUFFER)) };
    String::from_utf8_lossy(bytes)
        .split(';')
        .map(|s| s.trim().to_owned())
        .collect()
}

fn answer(v: Value) -> usize {
    let text = v.to_string();
    let n = text.len().min(BUFFER);
    unsafe { std::slice::from_raw_parts_mut(buffer(), n).copy_from_slice(&text.as_bytes()[..n]) };
    n
}

fn number(parts: &[String], i: usize) -> f64 {
    parts.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0)
}

fn square(m: Move) -> usize {
    9 * m.global() + m.local()
}

/// The game after `moves`, or why they can't be played.
fn replay(moves: &str) -> Result<Game, String> {
    let mut g = Game::new();
    for (i, word) in moves.split_whitespace().enumerate() {
        let sq: usize = word
            .parse()
            .ok()
            .filter(|&sq| sq < 81)
            .ok_or_else(|| format!("move {} is not a square: {}", i + 1, word))?;
        g = g
            .make_move(Move::from_coords(sq / 9, sq % 9))
            .map_err(|_| format!("move {} (square {}) is not legal", i + 1, sq))?;
    }
    Ok(g)
}

fn legal_moves(g: &Game) -> Vec<Move> {
    if g.game_over() {
        Vec::new()
    } else {
        g.all_moves().collect()
    }
}

fn result(state: GameState) -> Value {
    match state {
        GameState::InPlay => Value::Null,
        GameState::Drawn => json!("draw"),
        GameState::Won(p) => json!(p.as_str()),
    }
}

/// A minimax score for the side to move: "win" or "loss" when the search
/// found the game's end by force, otherwise its estimate.
fn score(s: i64) -> Value {
    if s >= EVAL_WON {
        json!("win")
    } else if s <= EVAL_LOST {
        json!("loss")
    } else {
        json!(s)
    }
}

/// xorshift64*, seeded by the page: enough to vary the easier levels' moves.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[(self.next() % items.len() as u64) as usize]
    }
}

fn minimax_config(depth: Option<i64>, limit: Option<Duration>) -> minimax::Config {
    minimax::Config {
        max_depth: depth,
        limit: limit,
        table_bytes: Some(MINIMAX_TABLE),
        ..Default::default()
    }
}

/// `moves`: the board, where the next move must go, the legal moves, and the
/// result.
#[no_mangle]
pub extern "C" fn state(len: usize) -> usize {
    let parts = request(len);
    let g = match replay(&parts[0]) {
        Ok(g) => g,
        Err(e) => return answer(json!({ "error": e })),
    };
    let cells: String = (0..81)
        .map(|sq| match g.at(sq / 9, sq % 9) {
            CellState::Empty => '.',
            CellState::Played(p) => p.as_str().chars().next().unwrap(),
        })
        .collect();
    let boards: String = (0..9)
        .map(|b| match g.board_state(b) {
            GameState::InPlay => '.',
            GameState::Drawn => 'D',
            GameState::Won(p) => p.as_str().chars().next().unwrap(),
        })
        .collect();
    answer(json!({
        "cells": cells,
        "boards": boards,
        "next": g.player().as_str(),
        "board": if g.game_over() { None } else { g.board_to_play() },
        "legal": legal_moves(&g).into_iter().map(square).collect::<Vec<_>>(),
        "result": result(g.game_state()),
    }))
}

/// `moves; depth; ms; slack; blunder; seed; visits; prove`: the player's
/// move. With `visits`, uttt.ai's search to that many root visits or for
/// `ms` milliseconds, whichever comes first, and then, with `prove`, up to
/// `prove` milliseconds of the solver looking for a move that keeps a proven
/// win or draw. Otherwise minimax: with no
/// slack, a minimax search to `depth` (0 for no limit) that starts no deeper
/// round after `ms` milliseconds (0 for no limit). With a slack, every move is
/// scored to `depth` and one within `slack` of the best is picked at random.
/// `blunder` is the chance of a random move instead.
#[no_mangle]
pub extern "C" fn bot(len: usize) -> usize {
    let parts = request(len);
    let g = match replay(&parts[0]) {
        Ok(g) => g,
        Err(e) => return answer(json!({ "error": e })),
    };
    let moves = legal_moves(&g);
    if moves.is_empty() {
        return answer(json!({ "error": "the game is over" }));
    }
    let depth = number(&parts, 1) as i64;
    let ms = number(&parts, 2);
    let slack = number(&parts, 3) as i64;
    let blunder = number(&parts, 4);
    let mut rng = Rng::new(number(&parts, 5) as u64);
    let visits = number(&parts, 6) as u32;
    let prove_ms = number(&parts, 7);

    if blunder > 0.0 && rng.unit() < blunder {
        return answer(json!({ "move": square(rng.pick(&moves)), "random": true }));
    }
    if visits > 0 {
        let net = match network() {
            Ok(net) => net,
            Err(e) => return answer(e),
        };
        let played: Vec<Move> = parts[0]
            .split_whitespace()
            .map(|w| w.parse::<usize>().unwrap())
            .map(|sq| Move::from_coords(sq / 9, sq % 9))
            .collect();
        let mut tree = match unsafe { (*addr_of_mut!(BOT_TREE)).take() } {
            Some((before, tree)) if played.starts_with(&before) => {
                played[before.len()..].iter().fold(tree, |t, &m| t.advance(m))
            }
            _ => Tree::new(g.clone()),
        };
        let keep = |tree: Tree, m: Move| {
            let mut after = played.clone();
            after.push(m);
            unsafe { *addr_of_mut!(BOT_TREE) = Some((after, tree.advance(m))) };
        };
        // The solver first: late in the game it's usually quick, and then the search isn't needed. It tries the
        // moves in the order the search would (visits, then the network's policy).
        if prove_ms > 0.0 {
            tree.run(net, 1, 0.0, false, &mut rng);
            let mut order: Vec<(u32, f32, Move)> = tree.children().iter().map(|c| (c.visits, c.prior, c.mv)).collect();
            order.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
            let order: Vec<Move> = order.into_iter().map(|(_, _, m)| m).collect();
            if let Some((m, outcome)) = proven_move(&g, Duration::from_secs_f64(prove_ms / 1000.0), &order) {
                keep(tree, m);
                return answer(json!({ "move": square(m), "proven": outcome }));
            }
        }
        tree.run(net, visits, ms, false, &mut rng);
        let m = tree.best(&mut rng);
        let reply = json!({ "move": square(m), "value": tree.root().mean(), "visits": tree.root().visits });
        keep(tree, m);
        return answer(reply);
    }
    if slack > 0 {
        let mut ai = Minimax::with_config(&minimax_config(None, None));
        let scored: Vec<(Move, i64)> = moves
            .iter()
            .map(|&m| (m, -ai.search(&g.make_move(m).unwrap(), depth.max(1) - 1).0))
            .collect();
        let best = scored.iter().map(|&(_, s)| s).max().unwrap();
        let near: Vec<(Move, i64)> = scored.into_iter().filter(|&(_, s)| s >= best - slack).collect();
        let (m, s) = rng.pick(&near);
        return answer(json!({ "move": square(m), "score": score(s), "depth": depth.max(1) }));
    }
    let limit = if ms > 0.0 { Some(Duration::from_secs_f64(ms / 1000.0)) } else { None };
    let mut ai = Minimax::with_config(&minimax_config(if depth > 0 { Some(depth) } else { None }, limit));
    let (pv, stats) = ai.analyze(&g);
    answer(json!({
        "move": square(pv[0]),
        "score": score(stats.last().map_or(0, |s| s.score)),
        "depth": stats.len(),
    }))
}

/// Whether `attacker` can force a win from `g`, in at most `limit`.
fn solve(g: &Game, attacker: ultimattt::game::Player, limit: Duration) -> Evaluation {
    let cfg = dfpn::Config {
        threads: 0,
        table_size: PROVE_TABLE,
        limit: Some(limit),
        attacker: Some(attacker),
        ..Default::default()
    };
    dfpn::DFPN::prove(&cfg, g).value
}

/// A move from `g` that keeps a proven win, or failing that a proven draw,
/// with the result it keeps, trying `order` in turn for about `budget`.
fn proven_move(g: &Game, budget: Duration, order: &[Move]) -> Option<(Move, Value)> {
    let start = Instant::now();
    let left = || budget.checked_sub(start.elapsed()).filter(|d| *d > Duration::from_millis(1));
    let me = g.player();
    let win = match solve(g, me, left()?) {
        Evaluation::True => true,
        Evaluation::False => false,
        Evaluation::Unknown => return None,
    };
    // Lost, or not known to be a draw: nothing to keep.
    if !win && !matches!(solve(g, me.other(), left()?), Evaluation::False) {
        return None;
    }
    for &m in order {
        let child = g.make_move(m).unwrap();
        let keeps = match child.game_state() {
            GameState::Won(p) => p == me,
            GameState::Drawn => !win,
            GameState::InPlay if win => matches!(solve(&child, me, left()?), Evaluation::True),
            GameState::InPlay => matches!(solve(&child, me.other(), left()?), Evaluation::False),
        };
        if keeps {
            return Some((m, if win { json!(me.as_str()) } else { json!("draw") }));
        }
    }
    None
}

/// `moves`: the network's own view of the position: its value for the side
/// to move, and its policy, `[square, probability]` per legal move.
#[no_mangle]
pub extern "C" fn net(len: usize) -> usize {
    let parts = request(len);
    let g = match replay(&parts[0]) {
        Ok(g) => g,
        Err(e) => return answer(json!({ "error": e })),
    };
    let net = match network() {
        Ok(net) => net,
        Err(e) => return answer(e),
    };
    let mut tree = Tree::new(g);
    tree.run(net, 1, 0.0, false, &mut Rng::new(1));
    let policy: Vec<Value> = tree.children().iter().map(|c| json!([square(c.mv), c.prior])).collect();
    answer(json!({ "value": tree.root().mean(), "policy": policy }))
}

/// `moves; visits; seed`: uttt.ai's search of the position, until the root
/// has `visits` visits, carrying on from the last call while the position is
/// the same. Every move is visited at least once, so each has a value: per
/// move, `[square, visits, chance, mean, network]`, the means and the
/// network's value from the side to move's view, -1 to 1.
#[no_mangle]
pub extern "C" fn think(len: usize) -> usize {
    let parts = request(len);
    let g = match replay(&parts[0]) {
        Ok(g) => g,
        Err(e) => return answer(json!({ "error": e })),
    };
    let net = match network() {
        Ok(net) => net,
        Err(e) => return answer(e),
    };
    let mut rng = Rng::new(number(&parts, 2) as u64);
    let slot = unsafe { &mut *addr_of_mut!(THINKER) };
    if slot.as_ref().map_or(true, |(key, _)| *key != parts[0]) {
        *slot = Some((parts[0].clone(), Tree::new(g)));
    }
    let tree = &mut slot.as_mut().unwrap().1;
    tree.run(net, number(&parts, 1) as u32, 0.0, true, &mut rng);
    let chances = tree.chances();
    let moves: Vec<Value> = tree
        .children()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mean = if c.visits > 0 { json!(-c.mean()) } else { Value::Null };
            json!([square(c.mv), c.visits, chances[i], mean, tree.child_value(i)])
        })
        .collect();
    answer(json!({ "visits": tree.root().visits, "value": tree.root().mean(), "moves": moves }))
}

/// `moves; ms`: the game's result with best play, "X", "O" or "draw", proven
/// by the solver within about `ms` milliseconds, or "unknown".
#[no_mangle]
pub extern "C" fn prove(len: usize) -> usize {
    let parts = request(len);
    let g = match replay(&parts[0]) {
        Ok(g) => g,
        Err(e) => return answer(json!({ "error": e })),
    };
    let start = Instant::now();
    if g.game_over() {
        return answer(json!({ "result": result(g.game_state()), "ms": 0 }));
    }
    let budget = Duration::from_secs_f64(number(&parts, 1).max(1.0) / 1000.0);
    let me = g.player();
    let outcome = match solve(&g, me, budget) {
        Evaluation::True => json!(me.as_str()),
        Evaluation::Unknown => json!("unknown"),
        Evaluation::False => {
            let left = budget.checked_sub(start.elapsed()).unwrap_or(Duration::from_millis(1));
            match solve(&g, me.other(), left.max(Duration::from_millis(1))) {
                Evaluation::True => json!(me.other().as_str()),
                Evaluation::False => json!("draw"),
                Evaluation::Unknown => json!("unknown"),
            }
        }
    };
    answer(json!({ "result": outcome, "ms": start.elapsed().as_millis() as u64 }))
}
