//! Checks the proof walk (`dfpn::Config::proof_size`), which follows the proof
//! a search left in its table, checks every step, and counts the positions a
//! search has to expand. On random late positions whose search fits easily in
//! the table, every walk must find a complete proof: nothing missing, and no
//! more positions than the search itself visited. With a table far too small
//! to keep the proof, walks must report missing positions rather than pass.
//!
//! The walk generates moves the way the search does; if the two ever differed,
//! the walk would look for positions the search never solved and report them
//! missing here. Set PROOF_WALK_ITERS and PROOF_WALK_SEED to run longer or
//! different checks.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Duration;
use ultimattt::game::{notation, Game, GameState};
use ultimattt::prove::{dfpn, Evaluation};

/// A random game cut off between ply 35 and the end, before it is decided.
fn random_late_position(rng: &mut StdRng) -> Game {
    let mut pos = Game::new();
    for _ in 0..rng.gen_range(35, 81) {
        let moves: Vec<_> = pos.all_moves().collect();
        let next = pos.make_move(moves[rng.gen_range(0, moves.len())]).unwrap();
        if next.game_state() != GameState::InPlay {
            break;
        }
        pos = next;
    }
    pos
}

fn walk(pos: &Game, table_size: usize, endgame: bool, symmetry: bool) -> dfpn::ProveResult {
    let cfg = dfpn::Config {
        table_size,
        limit: Some(Duration::from_secs(10)),
        endgame,
        symmetry,
        proof_size: true,
        ..Default::default()
    };
    dfpn::DFPN::prove(&cfg, pos)
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .map_or(default, |v| v.parse().expect(name))
}

#[test]
fn walks_complete_proofs() {
    let iters = env_or("PROOF_WALK_ITERS", 100);
    let mut rng = StdRng::seed_from_u64(env_or("PROOF_WALK_SEED", 1));
    let (mut walked, mut strict) = (0, 0);
    for _ in 0..iters {
        let pos = random_late_position(&mut rng);
        for &(endgame, symmetry) in &[(true, true), (false, true), (true, false)] {
            // 16 MiB holds about 500,000 positions.
            let result = walk(&pos, 16 << 20, endgame, symmetry);
            if result.value == Evaluation::Unknown {
                continue;
            }
            let s = &result.stats;
            let at = format!(
                "{} (endgame={} symmetry={})",
                notation::render(&pos),
                endgame,
                symmetry
            );
            assert!(s.proof >= 1, "empty proof at {}", at);
            assert!(
                s.proof <= s.fresh,
                "proof of {} positions, but the search visited only {} at {}",
                s.proof,
                s.fresh,
                at
            );
            walked += 1;
            // A search this small fills under 5% of the table, so the table
            // keeps every solved position.
            if s.fresh <= 25_000 {
                assert_eq!(s.proof_missing, 0, "proof incomplete at {}", at);
                strict += 1;
            }
        }
    }
    assert!(
        strict * 10 >= walked * 8,
        "only {} of {} walks fit the table",
        strict,
        walked
    );
}

#[test]
fn reports_proofs_the_table_lost() {
    let mut rng = StdRng::seed_from_u64(env_or("PROOF_WALK_SEED", 1));
    let (mut lost, mut big) = (0, 0);
    for _ in 0..60 {
        let pos = random_late_position(&mut rng);
        // 64 KiB holds about 2,000 positions.
        let result = walk(&pos, 64 << 10, true, true);
        if result.value == Evaluation::Unknown || result.stats.fresh < 20_000 {
            continue;
        }
        big += 1;
        if result.stats.proof_missing > 0 {
            lost += 1;
        }
    }
    assert!(big >= 5, "only {} searches outgrew the table", big);
    assert!(
        lost * 2 >= big,
        "only {} of {} walks noticed the lost proof",
        lost,
        big
    );
}
