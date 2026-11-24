//! Checks fast engine helpers against the plain move generator, and that the
//! cells inside a closed board never matter.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ultimattt::game::{notation, Game, GameState, Player};

#[test]
fn move_count_matches_move_generation() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..20_000 {
        let mut pos = Game::new();
        loop {
            let moves: Vec<_> = pos.all_moves().collect();
            assert_eq!(
                pos.move_count() as usize,
                moves.len(),
                "{}",
                notation::render(&pos)
            );
            let next = pos.make_move(moves[rng.gen_range(0, moves.len())]).unwrap();
            if next.game_state() != GameState::InPlay {
                assert_eq!(next.move_count(), 0, "a finished game has no moves");
                break;
            }
            pos = next;
        }
    }
}

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [6, 4, 2],
];

/// A random filling of one local board with the given result: a line for
/// the winner and none for the loser, or a full board with no line.
fn random_board(rng: &mut StdRng, state: GameState) -> String {
    loop {
        let mut cells: Vec<u8> = (0..9)
            .map(|_| match rng.gen_range(0, 3) {
                0 if state != GameState::Drawn => b'.',
                1 => b'O',
                _ => b'X',
            })
            .collect();
        if let GameState::Won(p) = state {
            let mark = if p == Player::X { b'X' } else { b'O' };
            for &c in LINES[rng.gen_range(0, 8)].iter() {
                cells[c] = mark;
            }
        }
        let has_line = |mark: u8| LINES.iter().any(|l| l.iter().all(|&c| cells[c] == mark));
        let ok = match state {
            GameState::Won(Player::X) => !has_line(b'O'),
            GameState::Won(Player::O) => !has_line(b'X'),
            _ => !has_line(b'X') && !has_line(b'O'),
        };
        if ok {
            return String::from_utf8(cells).unwrap();
        }
    }
}

#[test]
fn closed_board_cells_never_matter() {
    let mut rng = StdRng::seed_from_u64(2);
    let (mut checked, mut changed) = (0, 0);
    while checked < 5_000 {
        let mut pos = Game::new();
        for _ in 0..rng.gen_range(10, 60) {
            let moves: Vec<_> = pos.all_moves().collect();
            let next = pos.make_move(moves[rng.gen_range(0, moves.len())]).unwrap();
            if next.game_over() {
                break;
            }
            pos = next;
        }
        let closed: Vec<usize> = (0..9)
            .filter(|&b| pos.board_state(b) != GameState::InPlay)
            .collect();
        if pos.game_over() || closed.is_empty() {
            continue;
        }
        // Refill every closed board with other cells, keeping each result.
        let text = notation::render(&pos);
        let parts: Vec<&str> = text.split(';').collect();
        let mut boards: Vec<String> = parts[2].split('/').map(|b| b.to_owned()).collect();
        for &b in closed.iter() {
            boards[b] = random_board(&mut rng, pos.board_state(b));
        }
        let refilled = format!("{};{};{}", parts[0], parts[1], boards.join("/"));
        let mut other = notation::parse(&refilled).unwrap();
        if refilled != text {
            changed += 1;
        }
        // A different result on a closed board is a different position.
        let b = closed[0];
        let relabeled_state = match pos.board_state(b) {
            GameState::Won(Player::X) => GameState::Won(Player::O),
            GameState::Won(Player::O) => GameState::Drawn,
            _ => GameState::Won(Player::X),
        };
        let mut global: Vec<u8> = parts[1].bytes().collect();
        global[b] = match relabeled_state {
            GameState::Won(Player::X) => b'X',
            GameState::Won(Player::O) => b'O',
            _ => b'#',
        };
        boards[b] = random_board(&mut rng, relabeled_state);
        let relabeled = notation::parse(&format!(
            "{};{};{}",
            parts[0],
            String::from_utf8(global).unwrap(),
            boards.join("/")
        ))
        .unwrap();
        assert!(!pos.equivalent(&relabeled), "{}", text);
        assert_ne!(pos.zobrist(), relabeled.zobrist(), "{}", text);
        // Play the same random game out from both: the same moves stay
        // legal, and the two hash alike, to the end.
        loop {
            assert!(pos.equivalent(&other), "{} vs {}", text, refilled);
            assert_eq!(pos.zobrist(), other.zobrist(), "{} vs {}", text, refilled);
            assert_eq!(
                pos.game_state(),
                other.game_state(),
                "{} vs {}",
                text,
                refilled
            );
            if pos.game_over() {
                break;
            }
            let moves: Vec<_> = pos.all_moves().collect();
            assert_eq!(moves, other.all_moves().collect::<Vec<_>>(), "{}", refilled);
            let m = moves[rng.gen_range(0, moves.len())];
            pos = pos.make_move(m).unwrap();
            other = other.make_move(m).unwrap();
        }
        checked += 1;
    }
    assert!(
        changed > checked * 9 / 10,
        "only {} refills changed a cell",
        changed
    );
}
