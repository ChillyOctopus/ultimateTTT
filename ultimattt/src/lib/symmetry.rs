//! The 8 symmetries of Ultimate Tic Tac Toe: the rotations and reflections of
//! the 3x3 grid, applied to the global board and to every local board at once.
//! When one of them maps a position onto itself, it pairs up moves whose
//! results are mirror images, so they have equal values and a solver need
//! search only one move of each group.

use crate::game::{Game, Move, Player};

/// `SYMMETRIES[s][i]` is the image of grid index `i` (3 * row + column) under
/// symmetry `s`. Symmetry 0 is the identity.
pub const SYMMETRIES: [[usize; 9]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8], // identity
    [2, 5, 8, 1, 4, 7, 0, 3, 6], // rotate 90 degrees clockwise
    [8, 7, 6, 5, 4, 3, 2, 1, 0], // rotate 180 degrees
    [6, 3, 0, 7, 4, 1, 8, 5, 2], // rotate 270 degrees
    [2, 1, 0, 5, 4, 3, 8, 7, 6], // mirror left to right
    [6, 7, 8, 3, 4, 5, 0, 1, 2], // mirror top to bottom
    [0, 3, 6, 1, 4, 7, 2, 5, 8], // reflect in the main diagonal
    [8, 5, 2, 7, 4, 1, 6, 3, 0], // reflect in the other diagonal
];

const fn permute_table(map: [usize; 9]) -> [u16; 512] {
    let mut out = [0u16; 512];
    let mut cells = 0;
    while cells < 512 {
        let mut i = 0;
        while i < 9 {
            if cells & (1 << i) != 0 {
                out[cells] |= 1 << map[i];
            }
            i += 1;
        }
        cells += 1;
    }
    out
}

static PERMUTE: [[u16; 512]; 8] = [
    permute_table(SYMMETRIES[0]),
    permute_table(SYMMETRIES[1]),
    permute_table(SYMMETRIES[2]),
    permute_table(SYMMETRIES[3]),
    permute_table(SYMMETRIES[4]),
    permute_table(SYMMETRIES[5]),
    permute_table(SYMMETRIES[6]),
    permute_table(SYMMETRIES[7]),
];

/// A board's 9-bit cell mask, moved by symmetry `s`.
pub fn permute_cells(cells: u16, s: usize) -> u16 {
    PERMUTE[s][cells as usize & 0x1ff]
}

/// A move, moved by symmetry `s`.
pub fn permute_move(m: Move, s: usize) -> Move {
    Move::from_coords(SYMMETRIES[s][m.global()], SYMMETRIES[s][m.local()])
}

/// Whether symmetry `s` maps position `p` onto position `q`: the same player
/// to move, the board to play mapped onto `q`'s, and each board mapped onto
/// the matching board of `q`. Closed boards are compared only by who won
/// them, since their cells no longer matter.
pub fn maps_onto(p: &Game, q: &Game, s: usize) -> bool {
    let map = &SYMMETRIES[s];
    let (pg, qg) = (&p.global_states, &q.global_states);
    let moved = |cells: u32| permute_cells(cells as u16, s) as u32;
    // The closed boards first: three masks, which reject almost every
    // asymmetric position without looking at a single cell.
    if p.player() != q.player()
        || p.board_to_play().map(|b| map[b]) != q.board_to_play()
        || moved(pg.xbits()) != qg.xbits()
        || moved(pg.obits()) != qg.obits()
        || moved(pg.drawbits()) != qg.drawbits()
    {
        return false;
    }
    let open = !pg.donebits() & 0x1ff;
    (0..9).filter(|b| open & (1 << b) != 0).all(|b| {
        [Player::X, Player::O].iter().all(|&who| {
            moved(p.local_boards.playerbits(who, b)) == q.local_boards.playerbits(who, map[b])
        })
    })
}

/// The symmetries other than the identity that map `pos` onto itself, as a
/// bitmask. They always form a group, together with the identity.
pub fn stabilizer(pos: &Game) -> u8 {
    (1..8)
        .filter(|&s| maps_onto(pos, pos, s))
        .fold(0, |set, s| set | 1 << s)
}

/// Whether `m` represents its group of mirror-image moves under the
/// symmetries in `stabilizer`: the member with the smallest index.
pub fn is_representative(m: Move, stabilizer: u8) -> bool {
    let index = |m: Move| 9 * m.global() + m.local();
    (1..8)
        .filter(|s| stabilizer & (1 << s) != 0)
        .all(|s| index(permute_move(m, s)) >= index(m))
}
