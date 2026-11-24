//! Times the network's pieces on the positions of a suite file: summing a
//! position's hidden units from scratch (`refresh`), moving them from one
//! position to the next (`update`), each child's logit as the search needs
//! it, from its parent's units (`success_logit_child`), and each child's
//! value as trained, from scratch (`value`).
//!
//!   cargo run --release --example nnue_speed -- NET.bin ../bench/suite.tsv

use std::time::Instant;
use ultimattt::game::{notation, Game, Player};
use ultimattt::nnue::{Inputs, Network};

/// Computing both views' inputs and visiting each, per position.
fn inputs_only(positions: &[Game]) -> f64 {
    let rounds = 20000;
    let mut n = 0usize;
    let start = Instant::now();
    for _ in 0..rounds {
        for p in positions {
            Inputs::of(std::hint::black_box(p), Player::X).each(|i| n += i);
            Inputs::of(std::hint::black_box(p), Player::O).each(|i| n += i);
        }
    }
    let ns = start.elapsed().as_nanos() as f64 / (rounds * positions.len()) as f64;
    std::hint::black_box(n);
    ns
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = Network::load(&args[1]).unwrap();
    let positions: Vec<_> = std::fs::read_to_string(&args[2])
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| notation::parse(line.split('\t').nth(4).unwrap()).unwrap())
        .collect();
    let children: Vec<Vec<_>> = positions
        .iter()
        .map(|p| p.all_moves().map(|m| p.make_move(m).unwrap()).collect())
        .collect();
    let rounds = 2000;
    let mut acc = net.accumulator();
    let mut sink = 0.0;

    let start = Instant::now();
    for _ in 0..rounds {
        for p in &positions {
            net.refresh(&mut acc, p);
        }
    }
    let refresh = start.elapsed().as_nanos() as f64 / (rounds * positions.len()) as f64;

    // From each position to one of its children and back.
    let mut update = 0.0;
    for (p, cs) in positions.iter().zip(&children) {
        net.refresh(&mut acc, p);
        let start = Instant::now();
        for _ in 0..rounds {
            net.update(&mut acc, &cs[cs.len() / 2]);
            net.update(&mut acc, p);
        }
        update += start.elapsed().as_nanos() as f64;
    }
    let update = update / (2 * rounds * positions.len()) as f64;

    let n: usize = children.iter().map(|c| c.len()).sum();
    let mut near = 0.0;
    for (p, cs) in positions.iter().zip(&children) {
        net.refresh(&mut acc, p);
        let start = Instant::now();
        for _ in 0..rounds {
            for c in cs {
                sink += net.success_logit_child(&acc, p, c, Player::X);
            }
        }
        near += start.elapsed().as_nanos() as f64;
    }
    let near = near / (rounds * n) as f64;

    let start = Instant::now();
    for _ in 0..rounds / 10 {
        for cs in &children {
            for c in cs {
                sink += net.value(c)[0];
            }
        }
    }
    let scratch = start.elapsed().as_nanos() as f64 / (rounds / 10 * n) as f64;
    println!(
        "inputs of both views, iterated: {:.0} ns",
        inputs_only(&positions)
    );
    println!(
        "hidden {}: refresh {:.0} ns, update by a move {:.0} ns, success_logit_child {:.0} ns per child, value from scratch {:.0} ns ({} positions, {} children; {})",
        net.hidden(),
        refresh,
        update,
        near,
        scratch,
        positions.len(),
        n,
        sink
    );
}
