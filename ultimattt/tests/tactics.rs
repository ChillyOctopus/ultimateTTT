//! Checks the bitboard tactics against brute force on random positions: the
//! engine plays each move, and a board is winnable when some empty cell, read
//! one at a time, would complete a line.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ultimattt::game::{notation, CellState, Game, GameState, Player};
use ultimattt::tactics::{move_safety, threatened_boards, winning_cells};

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

fn random_position(rng: &mut StdRng) -> Game {
    let mut pos = Game::new();
    for _ in 0..rng.gen_range(0, 75) {
        let moves: Vec<_> = pos.all_moves().collect();
        let next = pos.make_move(moves[rng.gen_range(0, moves.len())]).unwrap();
        if next.game_over() {
            break;
        }
        pos = next;
    }
    pos
}

/// The empty cells of `board` where a mark of `who` would complete a line.
fn brute_winning_cells(pos: &Game, board: usize, who: Player) -> u32 {
    let mut cells = 0;
    for cell in (0..9).filter(|&c| pos.at(board, c) == CellState::Empty) {
        let owns = |c: usize| c == cell || pos.at(board, c) == CellState::Played(who);
        if LINES.iter().any(|line| line.iter().all(|&c| owns(c))) {
            cells |= 1 << cell;
        }
    }
    cells
}

/// Whether the side to move in `pos` can win the board it must play in, or
/// has a free move.
fn brute_handed_something(pos: &Game) -> bool {
    match pos.board_to_play() {
        None => true,
        Some(b) => brute_winning_cells(pos, b, pos.player()) != 0,
    }
}

#[test]
fn tactics_match_brute_force() {
    let mut rng = StdRng::seed_from_u64(1);
    let (mut positions, mut risky_moves, mut moves) = (0, 0, 0);
    while positions < 20_000 {
        let pos = random_position(&mut rng);
        if pos.game_over() {
            continue;
        }
        positions += 1;
        let text = notation::render(&pos);
        for who in [Player::X, Player::O] {
            let mut boards = 0;
            for b in 0..9 {
                if pos.board_state(b) != GameState::InPlay {
                    continue;
                }
                let want = brute_winning_cells(&pos, b, who);
                assert_eq!(winning_cells(&pos, b, who), want, "board {} of {}", b, text);
                if want != 0 {
                    boards |= 1 << b;
                }
            }
            assert_eq!(threatened_boards(&pos, who), boards, "{}", text);
        }

        let (mut safe, mut risky) = (0, 0);
        for m in pos.all_moves() {
            if brute_handed_something(&pos.make_move(m).unwrap()) {
                risky += 1;
            } else {
                safe += 1;
            }
        }
        assert_eq!(move_safety(&pos), (safe, risky), "{}", text);
        risky_moves += risky;
        moves += safe + risky;
    }
    eprintln!(
        "{} positions agree; {} of {} moves hand the opponent something",
        positions, risky_moves, moves
    );
}
