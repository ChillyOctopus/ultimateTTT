//! How much would merging equivalent positions shrink the search?
//!
//! From each benchmark position, and from the opening, enumerates every
//! position reachable in a few moves and counts the distinct ones under four
//! notions of "the same position":
//!
//! - current: what the solver's hash already treats as equal (a closed board
//!   reduced to who, if anyone, won it);
//! - dead boards: also ignoring who owns the marks on an open board neither
//!   player can win any more, since only which cells are filled still matters;
//! - symmetry: also identifying the 8 rotations and reflections;
//! - both.
//!
//!   cargo run --release --example equivalence_survey -- ../bench/suite.tsv [depth]

use std::collections::HashSet;
use std::{env, fs};
use ultimattt::game::{notation, CellState, Game, GameState, Player};

const FULL: u16 = 0x1ff;
const LINES: [u16; 8] = [0x7, 0x38, 0x1c0, 0x49, 0x92, 0x124, 0x111, 0x54];

/// The 8 symmetries of a 3x3 grid, as maps from index to index.
fn symmetries() -> [[usize; 9]; 8] {
    let maps: [fn(usize, usize) -> (usize, usize); 8] = [
        |r, c| (r, c),
        |r, c| (c, 2 - r),
        |r, c| (2 - r, 2 - c),
        |r, c| (2 - c, r),
        |r, c| (r, 2 - c),
        |r, c| (2 - r, c),
        |r, c| (c, r),
        |r, c| (2 - c, 2 - r),
    ];
    let mut out = [[0; 9]; 8];
    for (s, f) in maps.iter().enumerate() {
        for i in 0..9 {
            let (r, c) = f(i / 3, i % 3);
            out[s][i] = 3 * r + c;
        }
    }
    out
}

/// Per board (x cells, o cells), then the board to play (9 when free) and the player.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Key([(u16, u16); 9], u8);

fn key(pos: &Game, erase_dead: bool) -> Key {
    let mut boards = [(0, 0); 9];
    for b in 0..9 {
        boards[b] = match pos.board_state(b) {
            GameState::Won(Player::X) => (FULL, 0),
            GameState::Won(Player::O) => (0, FULL),
            GameState::Drawn => (FULL, FULL),
            GameState::InPlay => {
                let (mut x, mut o) = (0, 0);
                for c in 0..9 {
                    match pos.at(b, c) {
                        CellState::Played(Player::X) => x |= 1 << c,
                        CellState::Played(Player::O) => o |= 1 << c,
                        CellState::Empty => (),
                    }
                }
                let dead = LINES.iter().all(|&l| o & l != 0) && LINES.iter().all(|&l| x & l != 0);
                // A dead open board is never full, so (cells, cells) cannot
                // be confused with a drawn board's (FULL, FULL).
                if erase_dead && dead {
                    (x | o, x | o)
                } else {
                    (x, o)
                }
            }
        };
    }
    let player = if pos.player() == Player::O { 16 } else { 0 };
    Key(boards, pos.board_to_play().map_or(9, |b| b as u8) | player)
}

fn canonical(k: Key, sym: &[[usize; 9]; 8]) -> Key {
    let permute = |mask: u16, s: &[usize; 9]| -> u16 {
        (0..9)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| 1 << s[i])
            .fold(0, |a, b| a | b)
    };
    sym.iter()
        .map(|s| {
            let mut boards = [(0, 0); 9];
            for b in 0..9 {
                let (x, o) = k.0[b];
                boards[s[b]] = (permute(x, s), permute(o, s));
            }
            let next = (k.1 & 15) as usize;
            Key(
                boards,
                if next < 9 { s[next] as u8 } else { 9 } | (k.1 & 16),
            )
        })
        .min()
        .unwrap()
}

/// Distinct positions at each depth under the four equivalences.
fn survey(root: &Game, depth: usize, sym: &[[usize; 9]; 8], totals: &mut [[u64; 4]]) {
    let mut frontier = vec![root.clone()];
    for d in 0..depth {
        let mut seen = HashSet::new();
        let mut next = Vec::new();
        for pos in &frontier {
            for m in pos.all_moves() {
                let child = pos.make_move(m).unwrap();
                if child.game_state() == GameState::InPlay && seen.insert(key(&child, false)) {
                    next.push(child);
                }
            }
        }
        let dead: HashSet<_> = next.iter().map(|p| key(p, true)).collect();
        let mirrored: HashSet<_> = seen.iter().map(|&k| canonical(k, sym)).collect();
        let both: HashSet<_> = dead.iter().map(|&k| canonical(k, sym)).collect();
        for (i, n) in [seen.len(), dead.len(), mirrored.len(), both.len()]
            .iter()
            .enumerate()
        {
            totals[d][i] += *n as u64;
        }
        frontier = next;
    }
}

fn report(title: &str, totals: &[[u64; 4]]) {
    println!("{}", title);
    println!("  depth      current   dead boards     symmetry         both");
    for (d, t) in totals.iter().enumerate() {
        let ratio = |i: usize| t[0] as f64 / t[i].max(1) as f64;
        println!(
            "  {:5} {:12} {:12} x{:<4.2} {:7} x{:<4.2} {:7} x{:<4.2}",
            d + 1,
            t[0],
            t[1],
            ratio(1),
            t[2],
            ratio(2),
            t[3],
            ratio(3)
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let depth: usize = args.get(2).map_or(4, |s| s.parse().expect("depth"));
    let sym = symmetries();

    let mut opening = vec![[0; 4]; depth + 1];
    survey(&Game::new(), depth + 1, &sym, &mut opening);
    report("From the opening:", &opening);

    let suite = fs::read_to_string(&args[1]).expect("cannot read the suite file");
    let mut totals = vec![[0; 4]; depth];
    let roots: Vec<_> = suite
        .lines()
        .skip(1)
        .map(|l| l.split('\t').nth(4).unwrap().to_owned())
        .collect();
    for root in &roots {
        survey(&notation::parse(root).unwrap(), depth, &sym, &mut totals);
    }
    report(
        &format!("\nFrom the {} benchmark positions (summed):", roots.len()),
        &totals,
    );
}
