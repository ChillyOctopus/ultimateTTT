//! Checks the solver's network (`ultimattt::nnue`) against PyTorch's outputs
//! on the same positions, as written by `nn/export.py`, and the search's
//! incremental way of computing values against computing them from scratch:
//! for the small random networks in `tests/data/nnue` and
//! `tests/data/nnue-moves` (which has a move head), and for the network in
//! the directory that `NNUE_CHECK` names, if set.

use ultimattt::game::{notation, Game, Player};
use ultimattt::nnue::Network;

use std::fs;
use std::path::{Path, PathBuf};

fn dirs() -> Vec<PathBuf> {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut out = vec![data.join("nnue"), data.join("nnue-moves")];
    out.extend(std::env::var("NNUE_CHECK").ok().map(PathBuf::from));
    out
}

fn positions(dir: &Path) -> Vec<(Game, Vec<f32>)> {
    let text = fs::read_to_string(dir.join("check.txt")).unwrap();
    text.lines()
        .map(|line| {
            let mut fields = line.split(' ');
            let text = fields.next().unwrap();
            let pos = notation::parse(text).unwrap_or_else(|_| panic!("bad position {}", text));
            (pos, fields.map(|f| f.parse().unwrap()).collect())
        })
        .collect()
}

fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1e-4 * want.abs().max(1.0)
}

#[test]
fn matches_pytorch() {
    for dir in dirs() {
        let net = Network::load(dir.join("net.bin"))
            .unwrap_or_else(|e| panic!("{}: {}", dir.display(), e));
        let positions = positions(&dir);
        assert!(positions.len() >= 48);
        for (pos, want) in &positions {
            let text = notation::render(pos);
            let out = net.evaluate(pos);
            assert_eq!(out.moves.len(), if net.has_moves() { 81 } else { 0 });
            assert_eq!(want.len(), 2 + 81 + 2 * out.moves.len(), "{}", text);
            let moves = out.moves.iter().flatten();
            let got = out.value.iter().chain(out.policy.iter()).chain(moves);
            for (i, (&got, &want)) in got.zip(want).enumerate() {
                assert!(
                    close(got, want),
                    "{}: output {} is {}, PyTorch says {}",
                    text,
                    i,
                    got,
                    want
                );
            }
            assert_eq!(net.value(pos), out.value, "{}", text);
        }
        eprintln!("{}: {} positions match", dir.display(), positions.len());
    }
}

/// The search keeps one position's hidden units, updates them from position
/// to position, and computes the values of nearby positions from them
/// (`Network::value_near`), or of a child from its parent's and the move
/// (`Network::value_child`). Its sums are exact, so a value must not depend
/// on the route to it, and it must match the network as trained
/// (`Network::value`).
#[test]
fn incremental_matches_scratch() {
    for dir in dirs() {
        let net = Network::load(dir.join("net.bin")).unwrap();
        let positions: Vec<Game> = positions(&dir).into_iter().map(|(pos, _)| pos).collect();
        // Walks from position to position, never summing from scratch.
        let mut walked = net.accumulator();
        let mut checked = 0;
        for pos in &positions {
            net.update(&mut walked, pos);
            let mut fresh = net.accumulator();
            net.refresh(&mut fresh, pos);
            let children = pos.all_moves().map(|m| pos.make_move(m).unwrap());
            for (i, g) in std::iter::once(pos.clone()).chain(children).enumerate() {
                let got = net.value_near(&walked, &g);
                assert_eq!(got, net.value_near(&fresh, &g), "{}", notation::render(&g));
                assert_eq!(got, net.value_near(&net.accumulator(), &g));
                if i > 0 {
                    assert_eq!(
                        got,
                        net.value_child(&fresh, pos, &g),
                        "{}",
                        notation::render(&g)
                    );
                    // The search's one logit, computed alone, is the same.
                    for attacker in [Player::X, Player::O] {
                        let want = if g.player() == attacker {
                            got[0]
                        } else {
                            -got[1]
                        };
                        let alone = net.success_logit_child(&fresh, pos, &g, attacker);
                        assert_eq!(alone, want, "{}", notation::render(&g));
                    }
                }
                let want = net.value(&g);
                assert!(
                    close(got[0], want[0]) && close(got[1], want[1]),
                    "{}: {:?}, as trained {:?}",
                    notation::render(&g),
                    got,
                    want
                );
                checked += 1;
            }
        }
        eprintln!("{}: {} values match", dir.display(), checked);
    }
}

/// The search gets a position's children's values from its move head, from
/// the position's activations (`Network::activate`). They must match the
/// move head as trained (`Network::evaluate`), whichever route the hidden
/// units took.
#[test]
fn move_head_matches_scratch() {
    let mut tested = 0;
    for dir in dirs() {
        let net = Network::load(dir.join("net.bin")).unwrap();
        if !net.has_moves() {
            continue;
        }
        let positions: Vec<Game> = positions(&dir).into_iter().map(|(pos, _)| pos).collect();
        let mut walked = net.accumulator();
        let (mut act, mut act_walked) = (net.activations(), net.activations());
        let mut checked = 0;
        for pos in &positions {
            net.update(&mut walked, pos);
            let mut fresh = net.accumulator();
            net.refresh(&mut fresh, pos);
            net.activate(&fresh, pos, &mut act);
            net.activate(&walked, pos, &mut act_walked);
            let trained = net.evaluate(pos).moves;
            for square in 0..81 {
                for attacker in [Player::X, Player::O] {
                    let got = net.move_success_logit(&act, pos, square, attacker);
                    let walked = net.move_success_logit(&act_walked, pos, square, attacker);
                    assert_eq!(got, walked, "{} square {}", notation::render(pos), square);
                    let want = if pos.player().other() == attacker {
                        trained[square][0]
                    } else {
                        -trained[square][1]
                    };
                    assert!(
                        close(got, want),
                        "{} square {}: {}, as trained {}",
                        notation::render(pos),
                        square,
                        got,
                        want
                    );
                    checked += 1;
                }
            }
        }
        eprintln!("{}: {} move-head logits match", dir.display(), checked);
        tested += 1;
    }
    assert!(tested > 0, "no network with a move head");
}
