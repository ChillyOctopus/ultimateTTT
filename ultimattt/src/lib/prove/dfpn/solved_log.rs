//! Writes the positions a search settles to a file, one fixed-size record per
//! position, as exact training data for networks that guide the search
//! (ROADMAP.md, step 3). `nn/uttt_data.py` reads the records.
//!
//! Records are 80 bytes, little-endian. Boards and squares are numbered 0 to 8
//! row by row, and square `s` of board `b` is bit `9 * b + s`:
//!
//! | Offset | Bytes | Field |
//! |---|---|---|
//! | 0 | 8 | Zobrist hash of the position |
//! | 8 | 16 | squares X holds |
//! | 24 | 16 | squares O holds |
//! | 40 | 4 | bits 0-8: boards X won; 9-17: boards O won; 18-26: drawn boards |
//! | 44 | 1 | flags: 1, O is to move; 2, O is the attacker; 4, the side to move gets what it plays for (the attacker can force a win, or the defender can stop it); 8, a move that does so was known before the position was searched, so move order did not matter there |
//! | 45 | 1 | the board the next move must be in, or 9 for any |
//! | 46 | 1 | with flag 4, the move the proof uses (`9 * board + square`); otherwise 255 |
//! | 47 | 1 | that move's place among the moves searched, in the order their starting values put them (0 = tried first); otherwise 255 |
//! | 48 | 1 | safe moves of the side to move (`tactics::move_safety`) |
//! | 49 | 1 | risky moves |
//! | 50 | 1 | number of moves searched |
//! | 51 | 1 | the move the starting values put first, or 255 if none |
//! | 52 | 4 | zero |
//! | 56 | 8 | work: search steps spent on the position and below it |
//! | 64 | 16 | the moves searched, as bits |

use crate::game::{self, Player};
use crate::tactics;

use std::fs::File;
use std::io::{self, BufWriter, Write};

const RECORD_BYTES: usize = 80;

pub(in crate::prove) struct SolvedLog {
    out: BufWriter<File>,
    /// A position is logged when the top 32 bits of its hash fall below
    /// this, so each position is either always logged or never.
    threshold: u64,
}

impl SolvedLog {
    /// Logs about a `sample` fraction of the settled positions.
    pub(in crate::prove) fn create(path: &str, sample: f64) -> io::Result<Self> {
        Ok(SolvedLog {
            out: BufWriter::with_capacity(1 << 20, File::create(path)?),
            threshold: (sample.clamp(0.0, 1.0) * (1u64 << 32) as f64) as u64,
        })
    }

    pub(in crate::prove) fn wants(&self, hash: u64) -> bool {
        (hash >> 32) < self.threshold
    }

    /// Writes one settled position. `moves` holds the moves searched, in the
    /// order they were generated, each with the disproof number its position
    /// started with; DFPN tries the smallest first, the earliest on ties.
    pub(in crate::prove) fn write(
        &mut self,
        pos: &game::Game,
        attacker: Player,
        won: bool,
        known: bool,
        proof_move: game::Move,
        work: u64,
        moves: &[(game::Move, u32)],
    ) -> io::Result<()> {
        let square = |m: game::Move| (9 * m.global() + m.local()) as u8;
        let (mut x, mut o) = (0u128, 0u128);
        for b in 0..9 {
            x |= (pos.local_boards.playerbits(Player::X, b) as u128) << (9 * b);
            o |= (pos.local_boards.playerbits(Player::O, b) as u128) << (9 * b);
        }
        let states = &pos.global_states;
        let boards = states.xbits() | states.obits() << 9 | states.drawbits() << 18;
        let flags = (pos.player() == Player::O) as u8
            | ((attacker == Player::O) as u8) << 1
            | (won as u8) << 2
            | (known as u8) << 3;
        let (proof_square, rank) = match moves.iter().position(|&(m, _)| won && m == proof_move) {
            Some(i) => {
                let start = moves[i].1;
                let before = moves
                    .iter()
                    .enumerate()
                    .filter(|&(j, &(_, s))| s < start || (s == start && j < i))
                    .count();
                (square(proof_move), before as u8)
            }
            None => (255, 255),
        };
        let first = moves
            .iter()
            .enumerate()
            .min_by_key(|&(j, &(_, s))| (s, j))
            .map_or(255, |(_, &(m, _))| square(m));
        let (safe, risky) = tactics::move_safety(pos);
        let searched = moves
            .iter()
            .fold(0u128, |bits, &(m, _)| bits | 1 << square(m));

        let mut rec = [0u8; RECORD_BYTES];
        rec[0..8].copy_from_slice(&pos.zobrist().to_le_bytes());
        rec[8..24].copy_from_slice(&x.to_le_bytes());
        rec[24..40].copy_from_slice(&o.to_le_bytes());
        rec[40..44].copy_from_slice(&boards.to_le_bytes());
        rec[44] = flags;
        rec[45] = pos.board_to_play().map_or(9, |b| b as u8);
        rec[46] = proof_square;
        rec[47] = rank;
        rec[48] = safe as u8;
        rec[49] = risky as u8;
        rec[50] = moves.len() as u8;
        rec[51] = first;
        rec[56..64].copy_from_slice(&work.to_le_bytes());
        rec[64..80].copy_from_slice(&searched.to_le_bytes());
        self.out.write_all(&rec)
    }

    pub(in crate::prove) fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}
