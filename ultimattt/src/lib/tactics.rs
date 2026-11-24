//! Cheap tactical facts about a position, read off the bitboards with table
//! lookups: where each player could win a local board with one move, and
//! which of the side to move's moves hand the opponent something.

use crate::game::{Game, Player};

const LINES: [u32; 8] = [0x7, 0x38, 0x1c0, 0x49, 0x92, 0x124, 0x111, 0x54];
const FULL_BOARD: u32 = 0x1ff;

const fn completes_table() -> [u16; 512] {
    let mut out = [0u16; 512];
    let mut cells = 0;
    while cells < 512 {
        let mut l = 0;
        while l < 8 {
            if (cells as u32 & LINES[l]).count_ones() == 2 {
                out[cells] |= (LINES[l] & !(cells as u32)) as u16;
            }
            l += 1;
        }
        cells += 1;
    }
    out
}

/// `COMPLETES[cells]` holds the cells that would complete a line for a player
/// who has `cells` on a board, empty or not.
static COMPLETES: [u16; 512] = completes_table();

/// The empty cells of `board` where `who` would win it.
pub fn winning_cells(pos: &Game, board: usize, who: Player) -> u32 {
    COMPLETES[pos.local_boards.playerbits(who, board) as usize] as u32
        & pos.local_boards.free_squares(board)
}

/// The open boards that `who` could win with one move there.
pub fn threatened_boards(pos: &Game, who: Player) -> u32 {
    let open = !pos.global_states.donebits() & FULL_BOARD;
    (0..9)
        .filter(|&b| open & (1 << b) != 0 && winning_cells(pos, b, who) != 0)
        .fold(0, |boards, b| boards | 1 << b)
}

/// The side to move's legal moves, counted as (safe, risky). A move is risky
/// when it hands the opponent something: it sends them to a board they can
/// win with their reply, or to a closed board, which gives them a free move.
pub fn move_safety(pos: &Game) -> (u32, u32) {
    if pos.game_over() {
        return (0, 0);
    }
    let (me, them) = (pos.player(), pos.player().other());
    let open = !pos.global_states.donebits() & FULL_BOARD;
    let gifts = !open & FULL_BOARD | threatened_boards(pos, them);
    let boards = pos.board_to_play().map_or(open, |b| 1 << b);
    let (mut safe, mut risky) = (0, 0);
    for b in (0..9).filter(|&b| boards & (1 << b) != 0) {
        let empty = pos.local_boards.free_squares(b);
        let mut bad = empty & gifts & !(1 << b);
        if empty & (1 << b) != 0 {
            // Cell `b` sends the opponent back to this board as the move
            // leaves it: closed, if the move wins or fills it, and otherwise
            // without the cell the move takes.
            let mine = pos.local_boards.playerbits(me, b);
            let closes = COMPLETES[mine as usize] as u32 & (1 << b) != 0 || empty == 1 << b;
            if closes || winning_cells(pos, b, them) & !(1 << b) != 0 {
                bad |= 1 << b;
            }
        }
        risky += bad.count_ones();
        safe += (empty & !bad).count_ones();
    }
    (safe, risky)
}
