//! Measures how often positions near the benchmark suite have short forced
//! wins or losses that the endgame analysis does not detect, which bounds what a
//! deeper threat analysis could prune.
//!
//! Positions are sampled by playing random moves from each suite position, which
//! resembles the interior of the solver's search tree. Each is checked by
//! exhaustive search to a fixed depth.
//!
//!   cargo run --release --example threat_survey -- ../bench/suite.tsv [samples per position]

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, fs};
use ultimattt::endgame;
use ultimattt::game::{notation, Game, GameState};

/// The side to move can force a win within `plies` plies (an odd number).
fn wins_within(pos: &Game, plies: u32) -> bool {
    let me = pos.player();
    pos.all_moves().any(|m| {
        let child = pos.make_move(m).unwrap();
        match child.game_state() {
            GameState::Won(p) => p == me,
            GameState::Drawn => false,
            GameState::InPlay => {
                plies >= 3
                    && child.all_moves().all(|r| {
                        let g = child.make_move(r).unwrap();
                        g.game_state() == GameState::InPlay && wins_within(&g, plies - 2)
                    })
            }
        }
    })
}

/// Whatever the side to move plays, the opponent can force a win within
/// `plies - 1` more plies (`plies` is even).
fn loses_within(pos: &Game, plies: u32) -> bool {
    pos.all_moves().all(|m| {
        let child = pos.make_move(m).unwrap();
        child.game_state() == GameState::InPlay && wins_within(&child, plies - 1)
    })
}

fn random_descendant(root: &Game, rng: &mut StdRng) -> Game {
    let mut pos = root.clone();
    for _ in 0..rng.gen_range(0, 13) {
        let moves: Vec<_> = pos.all_moves().collect();
        let next = pos.make_move(moves[rng.gen_range(0, moves.len())]).unwrap();
        if next.game_state() != GameState::InPlay {
            break;
        }
        pos = next;
    }
    pos
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let suite = fs::read_to_string(&args[1]).expect("cannot read the suite file");
    let samples: usize = args.get(2).map_or(100, |s| s.parse().expect("samples"));
    let roots: Vec<Game> = suite
        .lines()
        .skip(1)
        .map(|line| notation::parse(line.split('\t').nth(4).expect("position column")).unwrap())
        .collect();

    let mut rng = StdRng::seed_from_u64(1);
    let (mut total, mut said_win, mut said_loss) = (0, 0, 0);
    let mut wins = [0; 3]; // forced wins within 1, 3, 5 plies
    let mut missed_wins = [0; 3];
    let mut losses = [0; 2]; // forced losses within 2, 4 plies
    let mut missed_losses = [0; 2];
    for root in &roots {
        for _ in 0..samples {
            let pos = random_descendant(root, &mut rng);
            let mut stats = Default::default();
            let status = endgame::Analysis::new(&pos, &mut stats).status();
            let me = pos.player();
            let (win, loss) = (status.is_won(me), status.is_won(me.other()));
            total += 1;
            said_win += win as usize;
            said_loss += loss as usize;
            for (i, plies) in [1, 3, 5].iter().enumerate() {
                if wins_within(&pos, *plies) {
                    wins[i] += 1;
                    missed_wins[i] += !win as usize;
                }
            }
            for (i, plies) in [2, 4].iter().enumerate() {
                if loses_within(&pos, *plies) {
                    losses[i] += 1;
                    missed_losses[i] += !loss as usize;
                }
            }
        }
    }

    let pct = |n: usize| 100.0 * n as f64 / total as f64;
    println!(
        "{} positions sampled around {} suite positions",
        total,
        roots.len()
    );
    println!(
        "analysis decides: win {:.1}%, loss {:.1}%",
        pct(said_win),
        pct(said_loss)
    );
    for (i, plies) in [1, 3, 5].iter().enumerate() {
        println!(
            "forced win within {} plies: {:5.1}% of positions; analysis misses {:5.1}%",
            plies,
            pct(wins[i]),
            pct(missed_wins[i])
        );
    }
    for (i, plies) in [2, 4].iter().enumerate() {
        println!(
            "forced loss within {} plies: {:4.1}% of positions; analysis misses {:5.1}%",
            plies,
            pct(losses[i]),
            pct(missed_losses[i])
        );
    }
}
