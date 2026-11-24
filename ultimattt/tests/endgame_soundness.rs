//! Checks every claim made by the endgame analysis (`src/lib/endgame.rs`) against
//! exhaustive search.
//!
//! The solver trusts the analysis without searching: a move it rejects is treated as
//! lost, and a position it decides is not searched further. So a wrong claim becomes a
//! wrong proof. Every claim the analysis makes has a short horizon, so each can be
//! checked exactly:
//!
//! - `evaluate_move` rejects a move: the opponent can then win with their next move.
//!   The solver only asks about moves in positions `status` leaves undecided, so
//!   that is the only place this is checked.
//! - `status` says the side to move wins: they can force a win within three plies.
//! - `status` says the side to move loses: whatever they play, the opponent then wins.
//! - `is_winnable` says a player can never win, whatever is played. That cannot be
//!   checked exhaustively, so check the short horizon: the side to move has no
//!   forced win within three plies, or the opponent cannot win right after any move.
//!
//! Positions come from random games, which produce many arrangements of won boards.
//! Set ENDGAME_FUZZ_ITERS and ENDGAME_FUZZ_SEED to run longer or different checks.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ultimattt::endgame;
use ultimattt::game::{notation, Game, GameState};

/// The side to move can win the whole game with one move.
fn wins_now(pos: &Game) -> bool {
    let me = pos.player();
    pos.all_moves()
        .any(|m| pos.make_move(m).unwrap().game_state() == GameState::Won(me))
}

/// The side to move can force a win within three plies: a move, any reply, a move.
fn wins_within_3(pos: &Game) -> bool {
    let me = pos.player();
    pos.all_moves().any(|m| {
        let child = pos.make_move(m).unwrap();
        match child.game_state() {
            GameState::Won(p) => p == me,
            GameState::Drawn => false,
            GameState::InPlay => child.all_moves().all(|r| {
                let g = child.make_move(r).unwrap();
                g.game_state() == GameState::InPlay && wins_now(&g)
            }),
        }
    })
}

/// Whatever the side to move plays, the opponent can then win with one move.
fn loses_within_2(pos: &Game) -> bool {
    pos.all_moves().all(|m| {
        let child = pos.make_move(m).unwrap();
        child.game_state() == GameState::InPlay && wins_now(&child)
    })
}

/// A random game cut off at a random length, ending before the game is decided.
fn random_position(rng: &mut StdRng) -> Game {
    let mut pos = Game::new();
    for _ in 0..rng.gen_range(10, 81) {
        let moves: Vec<_> = pos.all_moves().collect();
        let next = pos.make_move(moves[rng.gen_range(0, moves.len())]).unwrap();
        if next.game_state() != GameState::InPlay {
            break;
        }
        pos = next;
    }
    pos
}

#[derive(Default)]
struct Report {
    positions: usize,
    wins_claimed: usize,
    losses_claimed: usize,
    moves_rejected: usize,
    wrong_wins: usize,
    wrong_losses: usize,
    wrong_rejections: usize,
    unwinnable_claimed: usize,
    wrong_unwinnable: usize,
    violations: Vec<String>,
}

fn check(pos: &Game, report: &mut Report) {
    let mut stats = Default::default();
    let analysis = endgame::Analysis::new(pos, &mut stats);
    let me = pos.player();
    let status = analysis.status();
    report.positions += 1;

    if status.is_won(me) {
        report.wins_claimed += 1;
        if !wins_within_3(pos) {
            report.wrong_wins += 1;
            report.violations.push(format!(
                "claims a win for the side to move: {}",
                notation::render(pos)
            ));
        }
    }
    if status.is_won(me.other()) {
        report.losses_claimed += 1;
        if !loses_within_2(pos) {
            report.wrong_losses += 1;
            report.violations.push(format!(
                "claims a loss for the side to move: {}",
                notation::render(pos)
            ));
        }
    }
    let (me_dead, other_dead) = (
        !endgame::is_winnable(pos, me),
        !endgame::is_winnable(pos, me.other()),
    );
    if me_dead && wins_within_3(pos) {
        report.wrong_unwinnable += 1;
        report.violations.push(format!(
            "claims a win is impossible for the side to move: {}",
            notation::render(pos)
        ));
    }
    if other_dead
        && pos.all_moves().any(|m| {
            let child = pos.make_move(m).unwrap();
            child.game_state() == GameState::InPlay && wins_now(&child)
        })
    {
        report.wrong_unwinnable += 1;
        report.violations.push(format!(
            "claims a win is impossible for the opponent: {}",
            notation::render(pos)
        ));
    }
    report.unwinnable_claimed += me_dead as usize + other_dead as usize;
    if status.is_won(me) || status.is_won(me.other()) {
        return;
    }
    for m in pos.all_moves() {
        if analysis.evaluate_move(m).is_won(me.other()) {
            report.moves_rejected += 1;
            let child = pos.make_move(m).unwrap();
            if !(child.game_state() == GameState::InPlay && wins_now(&child)) {
                report.wrong_rejections += 1;
                report.violations.push(format!(
                    "rejects {} as losing: {}",
                    notation::render_move(m),
                    notation::render(pos)
                ));
            }
        }
    }
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .map_or(default, |v| v.parse().expect(name))
}

#[test]
fn endgame_claims_hold_on_random_positions() {
    let iters = env_or("ENDGAME_FUZZ_ITERS", 20_000);
    let mut rng = StdRng::seed_from_u64(env_or("ENDGAME_FUZZ_SEED", 1));
    let mut report = Report::default();
    for _ in 0..iters {
        check(&random_position(&mut rng), &mut report);
    }
    eprintln!(
        "checked {} positions: {} wins claimed ({} wrong), {} losses claimed ({} wrong), {} moves rejected ({} wrong), \
         {} cannot-win claims ({} wrong)",
        report.positions,
        report.wins_claimed,
        report.wrong_wins,
        report.losses_claimed,
        report.wrong_losses,
        report.moves_rejected,
        report.wrong_rejections,
        report.unwinnable_claimed,
        report.wrong_unwinnable
    );
    if !report.violations.is_empty() {
        let mut examples = Vec::new();
        for kind in [
            "claims a win for",
            "claims a loss",
            "rejects",
            "claims a win is impossible",
        ]
        .iter()
        {
            examples.extend(
                report
                    .violations
                    .iter()
                    .filter(|v| v.starts_with(kind))
                    .take(4)
                    .cloned(),
            );
        }
        panic!(
            "{} wrong claims, for example:\n  {}",
            report.violations.len(),
            examples.join("\n  ")
        );
    }
}
