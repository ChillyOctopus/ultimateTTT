//! Checks the proof-number solvers against the exhaustive solver (`oracle`) on
//! random positions late enough for exhaustive search: DFPN with and without
//! the endgame analysis, the parallel PN-DFPN, and plain PN search. They answer "can the side to move
//! force a win?", so a result must be True exactly when the exact value is a win. DFPN also answers
//! for the other player (`attacker`), True exactly when the exact value is a loss.
//!
//! DFPN and PN-DFPN also run guided by a network (`Init::Net`): the small random
//! networks in `tests/data/nnue` and `tests/data/nnue-moves`, the second through
//! its move head (`Config::move_head`), whose guesses are noise. Guidance may
//! change how long a search takes, never its result.
//!
//! This is the rung of the testing ladder between the exhaustive solver, which
//! is simple enough to trust, and the fast solver, which then checks positions
//! too large for exhaustive search. Set SOLVER_FUZZ_ITERS and SOLVER_FUZZ_SEED
//! to run longer or different checks.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use ultimattt::game::{notation, Game, GameState, Player};
use ultimattt::nnue::Network;
use ultimattt::oracle::{Oracle, Value};
use ultimattt::prove::{dfpn, pn, pn_dfpn, Evaluation};

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

fn dfpn_wins(pos: &Game, endgame: bool) -> Evaluation {
    let cfg = dfpn::Config {
        table_size: 16 << 20,
        limit: Some(Duration::from_secs(60)),
        endgame,
        ..Default::default()
    };
    dfpn::DFPN::prove(&cfg, pos).value
}

/// Whether the player not to move can force a win (`dfpn::Config::attacker`).
fn dfpn_opponent_wins(pos: &Game) -> Evaluation {
    let cfg = dfpn::Config {
        table_size: 16 << 20,
        limit: Some(Duration::from_secs(60)),
        attacker: Some(pos.player().other()),
        ..Default::default()
    };
    dfpn::DFPN::prove(&cfg, pos).value
}

/// Starting values from a small random network, from its move head if it has
/// one.
fn guided(cfg: dfpn::Config, net: &Arc<Network>) -> dfpn::Config {
    dfpn::Config {
        init: dfpn::Init::Net(32, 0.0),
        net: Some(net.clone()),
        move_head: net.has_moves(),
        ..cfg
    }
}

/// Whether `attacker`, or else the side to move, can force a win, with DFPN
/// guided by `net`.
fn dfpn_guided_wins(pos: &Game, net: &Arc<Network>, attacker: Option<Player>) -> Evaluation {
    let cfg = dfpn::Config {
        table_size: 16 << 20,
        limit: Some(Duration::from_secs(60)),
        attacker,
        ..Default::default()
    };
    dfpn::DFPN::prove(&guided(cfg, net), pos).value
}

fn pn_dfpn_wins(pos: &Game, net: Option<&Arc<Network>>) -> Evaluation {
    let mut dfpn = dfpn::Config {
        threads: 2,
        table_size: 16 << 20,
        ..Default::default()
    };
    if let Some(net) = net {
        dfpn = guided(dfpn, net);
    }
    let cfg = pn_dfpn::Config {
        limit: Some(Duration::from_secs(60)),
        dfpn,
        ..Default::default()
    };
    pn_dfpn::Prover::prove(&cfg, pos).result
}

fn pn_wins(pos: &Game) -> Evaluation {
    let cfg = pn::Config {
        max_memory: Some(256 << 20),
        limit: Some(Duration::from_secs(60)),
        ..Default::default()
    };
    pn::Prover::prove(&cfg, pos).result
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .map_or(default, |v| v.parse().expect(name))
}

#[derive(Default)]
struct Report {
    checked: usize,
    skipped: usize,
    values: [usize; 3],
    wrong: Vec<String>,
}

fn check_positions(iters: u64, seed: u64) -> Report {
    let mut rng = StdRng::seed_from_u64(seed);
    let load = |path: &str| {
        Arc::new(Network::load(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap())
    };
    let net = load("tests/data/nnue/net.bin");
    let moves = load("tests/data/nnue-moves/net.bin");
    let mut report = Report::default();
    let Report {
        checked,
        skipped,
        values,
        wrong,
    } = &mut report;
    for _ in 0..iters {
        let pos = random_late_position(&mut rng);
        let deadline = Instant::now() + Duration::from_secs(2);
        let exact = match Oracle::new(Some(deadline)).value(&pos) {
            Some(v) => v,
            None => {
                *skipped += 1;
                continue;
            }
        };
        values[exact as usize] += 1;
        *checked += 1;
        let want = if exact == Value::Win {
            Evaluation::True
        } else {
            Evaluation::False
        };
        let want_opponent = if exact == Value::Loss {
            Evaluation::True
        } else {
            Evaluation::False
        };
        let results = [
            ("dfpn", dfpn_wins(&pos, true), want),
            ("dfpn --no-endgame", dfpn_wins(&pos, false), want),
            (
                "dfpn --attacker=opponent",
                dfpn_opponent_wins(&pos),
                want_opponent,
            ),
            ("pn-dfpn", pn_dfpn_wins(&pos, None), want),
            ("pn", pn_wins(&pos), want),
            ("dfpn --init=net", dfpn_guided_wins(&pos, &net, None), want),
            (
                "dfpn --init=net --attacker=opponent",
                dfpn_guided_wins(&pos, &net, Some(pos.player().other())),
                want_opponent,
            ),
            ("pn-dfpn --init=net", pn_dfpn_wins(&pos, Some(&net)), want),
            (
                "dfpn --init=net --move-head",
                dfpn_guided_wins(&pos, &moves, None),
                want,
            ),
            (
                "dfpn --init=net --move-head --attacker=opponent",
                dfpn_guided_wins(&pos, &moves, Some(pos.player().other())),
                want_opponent,
            ),
            (
                "pn-dfpn --init=net --move-head",
                pn_dfpn_wins(&pos, Some(&moves)),
                want,
            ),
        ];
        for (engine, got, want) in results.iter() {
            // A search that ran out of time or memory proves nothing either way.
            if *got != *want && *got != Evaluation::Unknown {
                wrong.push(format!(
                    "{} says {:?}, exact value {:?}: {}",
                    engine,
                    got,
                    exact,
                    notation::render(&pos)
                ));
            }
        }
    }
    report
}

#[test]
fn dfpn_agrees_with_exhaustive_search() {
    // Eight threads, each with its own random positions; more could exhaust
    // memory if several exhaustive searches grow large at once.
    const THREADS: u64 = 8;
    let iters = env_or("SOLVER_FUZZ_ITERS", 1_000);
    let seed = env_or("SOLVER_FUZZ_SEED", 1);
    let handles: Vec<_> = (0..THREADS)
        .map(|t| std::thread::spawn(move || check_positions(iters / THREADS, seed * THREADS + t)))
        .collect();
    let mut total = Report::default();
    for handle in handles {
        let r = handle.join().unwrap();
        total.checked += r.checked;
        total.skipped += r.skipped;
        for i in 0..3 {
            total.values[i] += r.values[i];
        }
        total.wrong.extend(r.wrong);
    }
    let Report {
        checked,
        skipped,
        values,
        wrong,
    } = total;
    eprintln!(
        "checked {} positions (wins {}, draws {}, losses {}), skipped {} needing over 2s of exhaustive search, {} wrong",
        checked, values[0], values[1], values[2], skipped, wrong.len()
    );
    assert!(
        wrong.is_empty(),
        "{} wrong:\n  {}",
        wrong.len(),
        wrong[..wrong.len().min(10)].join("\n  ")
    );
}
