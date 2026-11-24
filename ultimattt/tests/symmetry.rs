//! Checks the symmetry code against geometry, the game engine, exhaustive
//! search, and the solver run without symmetry pruning.
//!
//! - Each symmetry maps tic-tac-toe lines onto lines, and the 8 form a group.
//! - A random game and its mirror image, replayed move by move, stay mirror
//!   images of each other, so the symmetries agree with the engine's rules.
//! - Mirror images have equal exact values.
//! - On random positions that a symmetry maps onto themselves, DFPN with
//!   symmetry pruning agrees with DFPN without it and with exhaustive search.
//!   Set SYMMETRY_FUZZ_ITERS to check more of them.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::{Duration, Instant};
use ultimattt::game::{notation, Game, GameState, Move};
use ultimattt::oracle::{Oracle, Value};
use ultimattt::prove::{dfpn, Evaluation};
use ultimattt::symmetry::{maps_onto, permute_cells, permute_move, stabilizer, SYMMETRIES};

const LINES: [u16; 8] = [0x7, 0x38, 0x1c0, 0x49, 0x92, 0x124, 0x111, 0x54];

#[test]
fn symmetries_map_lines_onto_lines_and_form_a_group() {
    for s in 0..8 {
        for &line in LINES.iter() {
            assert!(
                LINES.contains(&permute_cells(line, s)),
                "symmetry {} breaks line {:x}",
                s,
                line
            );
        }
        for t in 0..8 {
            let composed: Vec<usize> = (0..9).map(|i| SYMMETRIES[t][SYMMETRIES[s][i]]).collect();
            assert!(
                SYMMETRIES.iter().any(|u| u[..] == composed[..]),
                "{} then {} is not a symmetry",
                s,
                t
            );
        }
    }
}

fn random_game(rng: &mut StdRng, plies: usize) -> Vec<Move> {
    let mut pos = Game::new();
    let mut moves = Vec::new();
    for _ in 0..plies {
        let legal: Vec<_> = pos.all_moves().collect();
        let m = legal[rng.gen_range(0, legal.len())];
        let next = pos.make_move(m).unwrap();
        if next.game_state() != GameState::InPlay {
            break;
        }
        moves.push(m);
        pos = next;
    }
    moves
}

fn play(moves: &[Move]) -> Game {
    moves.iter().fold(Game::new(), |pos, &m| {
        pos.make_move(m).expect("mirrored move is illegal")
    })
}

#[test]
fn mirrored_games_stay_mirror_images() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..2_000 {
        let plies = rng.gen_range(1, 70);
        let moves = random_game(&mut rng, plies);
        let pos = play(&moves);
        for s in 0..8 {
            let mirrored: Vec<_> = moves.iter().map(|&m| permute_move(m, s)).collect();
            let image = play(&mirrored);
            assert!(
                maps_onto(&pos, &image, s),
                "symmetry {} disagrees with the engine on {:?}",
                s,
                moves
            );
        }
    }
}

#[test]
fn mirror_images_have_equal_values() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut checked = 0;
    while checked < 300 {
        let plies = rng.gen_range(45, 81);
        let moves = random_game(&mut rng, plies);
        let s = rng.gen_range(1, 8);
        let deadline = || Some(Instant::now() + Duration::from_secs(2));
        let a = Oracle::new(deadline()).value(&play(&moves));
        let mirrored: Vec<_> = moves.iter().map(|&m| permute_move(m, s)).collect();
        let b = Oracle::new(deadline()).value(&play(&mirrored));
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a, b, "mirror images differ in value: {:?}", moves);
            checked += 1;
        }
    }
}

/// A random position that symmetry `s` maps onto itself, or None when the
/// random fill is not a valid position still in play.
fn random_symmetric_position(rng: &mut StdRng, s: usize, fill: f64) -> Option<Game> {
    let map = &SYMMETRIES[s];
    let mut cells = [[b'.'; 9]; 9];
    let mut done = [[false; 9]; 9];
    for b in 0..9 {
        for c in 0..9 {
            if done[b][c] {
                continue;
            }
            let mark = if rng.gen::<f64>() >= fill {
                b'.'
            } else if rng.gen::<bool>() {
                b'X'
            } else {
                b'O'
            };
            // Give the whole orbit of (b, c) under the symmetry the same mark.
            let (mut bb, mut cc) = (b, c);
            loop {
                cells[bb][cc] = mark;
                done[bb][cc] = true;
                let next = (map[bb], map[cc]);
                if next == (b, c) {
                    break;
                }
                bb = next.0;
                cc = next.1;
            }
        }
    }
    let mask = |b: usize, mark: u8| {
        (0..9)
            .filter(|&c| cells[b][c] == mark)
            .fold(0u16, |m, c| m | 1 << c)
    };
    let won = |cells: u16| LINES.iter().any(|&l| cells & l == l);
    let mut global = [b'.'; 9];
    for b in 0..9 {
        let (x, o) = (mask(b, b'X'), mask(b, b'O'));
        global[b] = match (won(x), won(o)) {
            (true, true) => return None,
            (true, false) => b'X',
            (false, true) => b'O',
            _ if x | o == 0x1ff => b'#',
            _ => b'.',
        };
    }
    let owned = |who: u8| {
        (0..9)
            .filter(|&b| global[b] == who)
            .fold(0u16, |m, b| m | 1 << b)
    };
    if won(owned(b'X')) || won(owned(b'O')) || global.iter().all(|&g| g != b'.') {
        return None;
    }
    let count = |mark: u8| cells.iter().flatten().filter(|&&c| c == mark).count();
    let player = match count(b'X') as i64 - count(b'O') as i64 {
        0 => 'X',
        1 => 'O',
        _ => return None,
    };
    let fixed: Vec<usize> = (0..9)
        .filter(|&b| map[b] == b && global[b] == b'.')
        .collect();
    if !fixed.is_empty() && rng.gen::<bool>() {
        global[fixed[rng.gen_range(0, fixed.len())]] = b'@';
    }
    let boards: Vec<String> = cells
        .iter()
        .map(|b| String::from_utf8(b.to_vec()).unwrap())
        .collect();
    let text = format!(
        "{};{};{}",
        player,
        String::from_utf8(global.to_vec()).unwrap(),
        boards.join("/")
    );
    let pos = notation::parse(&text).unwrap();
    assert!(
        stabilizer(&pos) & (1 << s) != 0,
        "generated position is not symmetric: {}",
        text
    );
    Some(pos)
}

fn dfpn(pos: &Game, symmetry: bool) -> (Evaluation, usize) {
    let cfg = dfpn::Config {
        table_size: 64 << 20,
        limit: Some(Duration::from_secs(10)),
        symmetry,
        ..Default::default()
    };
    let result = dfpn::DFPN::prove(&cfg, pos);
    (result.value, result.stats.mid)
}

/// Checks `count` random symmetric positions; returns how many were checked,
/// how many also exhaustively, and the nodes searched with and without symmetry.
fn check_symmetric_positions(count: usize, seed: u64) -> (usize, usize, usize, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut checked, mut with_oracle, mut nodes_with, mut nodes_without) = (0, 0, 0, 0);
    while checked < count {
        let s = rng.gen_range(1, 8);
        let fill = rng.gen_range(0.55, 0.8);
        let pos = match random_symmetric_position(&mut rng, s, fill) {
            Some(p) => p,
            None => continue,
        };
        let (with, n_with) = dfpn(&pos, true);
        let (without, n_without) = dfpn(&pos, false);
        if with == Evaluation::Unknown || without == Evaluation::Unknown {
            continue;
        }
        assert_eq!(
            with,
            without,
            "symmetry changes the result: {}",
            notation::render(&pos)
        );
        if let Some(v) = Oracle::new(Some(Instant::now() + Duration::from_secs(2))).value(&pos) {
            let want = if v == Value::Win {
                Evaluation::True
            } else {
                Evaluation::False
            };
            assert_eq!(
                with,
                want,
                "wrong result {:?}, exact {:?}: {}",
                with,
                v,
                notation::render(&pos)
            );
            with_oracle += 1;
        }
        checked += 1;
        nodes_with += n_with;
        nodes_without += n_without;
    }
    (checked, with_oracle, nodes_with, nodes_without)
}

#[test]
fn symmetry_pruning_keeps_results() {
    const THREADS: usize = 8;
    let count = std::env::var("SYMMETRY_FUZZ_ITERS").map_or(200, |v| v.parse().unwrap());
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            std::thread::spawn(move || check_symmetric_positions(count / THREADS, 3 + t as u64))
        })
        .collect();
    let (mut checked, mut with_oracle, mut nodes_with, mut nodes_without) = (0, 0, 0, 0);
    for handle in handles {
        let r = handle.join().unwrap();
        checked += r.0;
        with_oracle += r.1;
        nodes_with += r.2;
        nodes_without += r.3;
    }
    eprintln!(
        "{} symmetric positions agree ({} also checked exhaustively); nodes with symmetry {}, without {} (x{:.3})",
        checked,
        with_oracle,
        nodes_with,
        nodes_without,
        nodes_with as f64 / nodes_without as f64
    );
}
