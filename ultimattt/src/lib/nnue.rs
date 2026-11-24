//! A network that guides the proof search (ROADMAP.md, step 3): an NNUE
//! trained by `nn/train.py` on positions the solver settled, and written by
//! `nn/export.py`, whose docstring gives the file format.
//!
//! It sees only the position, from one player's view (`Inputs`): the
//! squares each player holds in open boards, which boards each player won,
//! and which were drawn. One layer, shared by both views, turns each view's
//! inputs into `hidden` units. The side to move's units, then its opponent's,
//! pass through a squared clipped ReLU into two heads. The value head has a
//! pair of logits for each of 10 buckets, chosen by where the next move must
//! go: that the side to move can force a win, and that its opponent can. The
//! policy head has a logit for each of the 81 squares: that it is the move a
//! proof uses. A network may also have a move head, learned from the value
//! head: for each square, the value logits of the position after a move
//! there, so that one evaluation of a position gives all of its children's.
//!
//! `value` and `evaluate` compute everything from the position in floating
//! point, as PyTorch does. The search instead keeps an `Accumulator` with the
//! hidden units of one position, and gets the values of positions near it
//! (`value_near`), or of a child from its parent's units and the move
//! (`value_child`), by adding and subtracting only the weights of the inputs
//! that differ. For that, the first layer's weights are scaled by a power of
//! two and rounded to integers, so the sums are exact whichever positions
//! they pass through, and a value never depends on the search's path to it.
//! For the move head, the search computes a position's clipped and squared
//! units once (`activate`) and each child's logit from them
//! (`move_success_logit`). These loops run 16 lanes at a time, compiled for
//! AVX-512 or AVX2 when the processor has them.

use crate::game::{Game, Player};

use smallvec::SmallVec;
use std::convert::TryInto;
use std::fs;
use std::io;
use std::path::Path;
use crate::simd::{f32x16, i32x16};

pub const INPUTS: usize = 189;
pub const BUCKETS: usize = 10;
const MAX_HIDDEN: usize = 2048;
const LANES: usize = 16;
/// The most inputs active at once from one view: 81 squares and 9 boards.
const MAX_ACTIVE: f64 = 90.0;

pub struct Network {
    hidden: usize,
    // The weights as trained, for `value` and `evaluate`.
    /// [INPUTS][hidden]
    ft_weight: Vec<f32>,
    ft_bias: Vec<f32>,
    /// [2 * BUCKETS][2 * hidden]
    value_weight: Vec<f32>,
    value_bias: Vec<f32>,
    /// [81][2 * hidden]
    policy_weight: Vec<f32>,
    policy_bias: Vec<f32>,
    /// [2 * 81][2 * hidden], or empty: row 2m + k gives logit k of the
    /// position after a move to square m, as the value head would.
    move_weight: Vec<f32>,
    move_bias: Vec<f32>,
    // For the search, in 16-lane vectors, which align rows to cache lines:
    // the first layer times 2^`shift`, rounded, and the value weights
    // divided by 2^(2 · `shift`), which undoes that scale on the squared
    // units.
    shift: u32,
    /// [INPUTS][hidden / 16]
    ft_weight_int: Vec<i32x16>,
    ft_bias_int: Vec<i32x16>,
    /// [2 * BUCKETS][2 * hidden / 16]
    value_weight_scaled: Vec<f32x16>,
    /// [2 * 81][2 * hidden / 16], scaled as the value weights are.
    move_weight_scaled: Vec<f32x16>,
    simd: Simd,
}

/// The widest vector instructions the processor has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Simd {
    Avx512,
    Avx2,
    Generic,
}

impl Simd {
    fn detect() -> Simd {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Simd::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Simd::Avx2;
            }
        }
        Simd::Generic
    }
}

/// The network's outputs for a position, as logits.
pub struct Output {
    /// That the side to move can force a win, and that its opponent can.
    pub value: [f32; 2],
    /// For each square, numbered `9 * board + cell`.
    pub policy: [f32; 81],
    /// With a move head, for each square: the value logits of the position
    /// after a move there (from that position's side to move's view).
    /// Empty otherwise.
    pub moves: Vec<[f32; 2]>,
}

const fn open_squares_table() -> [u128; 512] {
    let mut out = [0u128; 512];
    let mut open = 0;
    while open < 512 {
        let mut b = 0;
        while b < 9 {
            if open & (1 << b) != 0 {
                out[open] |= 0x1ff << (9 * b);
            }
            b += 1;
        }
        open += 1;
    }
    out
}

/// `OPEN_SQUARES[boards]`: the squares, as bit `9 * board + cell`, of the
/// boards in the set `boards`.
static OPEN_SQUARES: [u128; 512] = open_squares_table();

/// The inputs active in a position from one player's view: the squares,
/// numbered `9 * board + cell`, that the player holds (inputs 0-80) and
/// that its opponent holds (81-161) in open boards; and the boards the
/// player won (162-170), its opponent won (171-179), and drawn (180-188).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inputs {
    mine: u128,
    theirs: u128,
    /// Bits 0-8: boards won by the player; 9-17: by its opponent; 18-26:
    /// drawn.
    boards: u32,
}

impl Inputs {
    pub fn of(pos: &Game, view: Player) -> Inputs {
        let states = &pos.global_states;
        let open = OPEN_SQUARES[(!states.donebits() & 0x1ff) as usize];
        Inputs {
            mine: pos.local_boards.squares(view) & open,
            theirs: pos.local_boards.squares(view.other()) & open,
            boards: states.playerbits(view)
                | states.playerbits(view.other()) << 9
                | states.drawbits() << 18,
        }
    }

    /// The inputs in `self` and not in `other`.
    fn without(self, other: Inputs) -> Inputs {
        Inputs {
            mine: self.mine & !other.mine,
            theirs: self.theirs & !other.theirs,
            boards: self.boards & !other.boards,
        }
    }

    /// Calls `f` with each input's index, in increasing order.
    pub fn each(self, mut f: impl FnMut(usize)) {
        each_bit(self.mine, |i| f(i));
        each_bit(self.theirs, |i| f(81 + i));
        each_bit(self.boards as u128, |i| f(162 + i));
    }
}

fn each_bit(mut bits: u128, mut f: impl FnMut(usize)) {
    while bits != 0 {
        f(bits.trailing_zeros() as usize);
        bits &= bits - 1;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The sum of the lanes, added pairwise in a fixed order: faster than
/// `horizontal_sum`, which adds them one after another.
#[inline(always)]
fn sum_lanes(v: f32x16) -> f32 {
    let a = v.to_array();
    let mut b = [0f32; 8];
    for i in 0..8 {
        b[i] = a[i] + a[i + 8];
    }
    let mut c = [0f32; 4];
    for i in 0..4 {
        c[i] = b[i] + b[i + 4];
    }
    (c[0] + c[2]) + (c[1] + c[3])
}

/// Both views' hidden units before the activation, scaled as the network's
/// integer first layer is, for one position; and the inputs they sum.
#[derive(Clone)]
pub struct Accumulator {
    /// Indexed by `Player::as_bit`.
    units: [Vec<i32x16>; 2],
    inputs: [Inputs; 2],
}

/// Offsets, in vectors, of rows of the first layer's weights.
type Rows = SmallVec<[usize; 24]>;

/// One position's units as the heads see them, from both views, for the
/// move head (`Network::activate`).
pub struct Activations {
    units: Vec<f32x16>,
}

fn vectors<T: Copy + Default, V>(v: &[T], f: impl Fn([T; LANES]) -> V) -> Vec<V> {
    v.chunks_exact(LANES)
        .map(|c| {
            let mut lanes = [T::default(); LANES];
            lanes.copy_from_slice(c);
            f(lanes)
        })
        .collect()
}

impl Network {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Network> {
        let bytes = fs::read(path)?;
        let bad = |why: String| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not a network file: {}", why),
            )
        };
        if bytes.len() < 24 || &bytes[..8] != b"UTTTNNUE" {
            return Err(bad("no UTTTNNUE header".to_owned()));
        }
        let word = |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap());
        let (version, inputs, hidden, buckets) = (word(0), word(1), word(2) as usize, word(3));
        // Version 2 adds the move head.
        if !(version == 1 || version == 2)
            || inputs as usize != INPUTS
            || buckets as usize != BUCKETS
            || hidden == 0
            || hidden > MAX_HIDDEN
            || hidden % LANES != 0
        {
            return Err(bad(format!(
                "version {} with {} inputs, {} hidden units and {} buckets",
                version, inputs, hidden, buckets
            )));
        }
        let moves = if version == 2 { 2 * 81 } else { 0 };
        let sizes = [
            INPUTS * hidden,
            hidden,
            2 * BUCKETS * 2 * hidden,
            2 * BUCKETS,
            81 * 2 * hidden,
            81,
            moves * 2 * hidden,
            moves,
        ];
        if bytes.len() != 24 + 4 * sizes.iter().sum::<usize>() {
            return Err(bad(format!("{} bytes", bytes.len())));
        }
        let mut floats = bytes[24..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        let mut arrays = sizes.iter().map(|&n| floats.by_ref().take(n).collect());
        let mut next = || -> Vec<f32> { arrays.next().unwrap() };
        let (ft_weight, ft_bias) = (next(), next());
        let (value_weight, value_bias) = (next(), next());
        let (policy_weight, policy_bias) = (next(), next());
        let (move_weight, move_bias) = (next(), next());

        // Scale the first layer as finely as 32-bit sums allow, even if every
        // unit took its largest weight from every input, but no finer than
        // f32 can hold exactly.
        let largest = |v: &[f32]| v.iter().fold(0f64, |m, &x| m.max(x.abs() as f64));
        let bound = largest(&ft_bias) + MAX_ACTIVE * largest(&ft_weight);
        let shift = (i32::MAX as f64 / bound.max(1.0)).log2().floor().min(24.0);
        if !(shift >= 8.0) {
            return Err(bad(format!(
                "first-layer weights up to {} are too large",
                bound
            )));
        }
        let shift = shift as u32;
        let scale = (1u32 << shift) as f32;
        let quantize = |v: &[f32]| {
            vectors(v, |lanes| {
                i32x16::from_array(lanes.map(|x| (x * scale).round() as i32))
            })
        };
        let unscale = |v: &[f32]| {
            vectors(v, |lanes| {
                f32x16::from_array(lanes.map(|w| w / scale / scale))
            })
        };
        Ok(Network {
            hidden,
            shift,
            ft_weight_int: quantize(&ft_weight),
            ft_bias_int: quantize(&ft_bias),
            value_weight_scaled: unscale(&value_weight),
            move_weight_scaled: unscale(&move_weight),
            move_weight,
            move_bias,
            ft_weight,
            ft_bias,
            value_weight,
            value_bias,
            policy_weight,
            policy_bias,
            simd: Simd::detect(),
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Whether the network has a move head.
    pub fn has_moves(&self) -> bool {
        !self.move_bias.is_empty()
    }

    /// One view's hidden units before the activation: the feature biases
    /// plus the weights of the inputs active from `view`'s side.
    fn accumulate(&self, pos: &Game, view: Player, acc: &mut [f32]) {
        let h = self.hidden;
        acc.copy_from_slice(&self.ft_bias);
        Inputs::of(pos, view).each(|i| {
            for (a, w) in acc.iter_mut().zip(&self.ft_weight[i * h..(i + 1) * h]) {
                *a += w;
            }
        });
    }

    /// Calls `f` with what the heads see: the side to move's hidden units,
    /// then its opponent's, each clipped to [0, 1] and squared.
    fn with_activations<T>(&self, pos: &Game, f: impl FnOnce(&[f32]) -> T) -> T {
        let h = self.hidden;
        let mut a = vec![0f32; 2 * h];
        self.accumulate(pos, pos.player(), &mut a[..h]);
        self.accumulate(pos, pos.player().other(), &mut a[h..]);
        for x in a.iter_mut() {
            let c = x.clamp(0.0, 1.0);
            *x = c * c;
        }
        f(&a)
    }

    fn value_head(&self, pos: &Game, a: &[f32]) -> [f32; 2] {
        let row = 2 * pos.board_to_play().unwrap_or(9);
        let n = 2 * self.hidden;
        let logit = |r: usize| self.value_bias[r] + dot(&self.value_weight[r * n..(r + 1) * n], a);
        [logit(row), logit(row + 1)]
    }

    /// The value logits, as trained: that the side to move can force a win,
    /// and that its opponent can.
    pub fn value(&self, pos: &Game) -> [f32; 2] {
        self.with_activations(pos, |a| self.value_head(pos, a))
    }

    /// Both heads' outputs, as trained.
    pub fn evaluate(&self, pos: &Game) -> Output {
        self.with_activations(pos, |a| {
            let n = 2 * self.hidden;
            let mut policy = [0.0; 81];
            for (m, p) in policy.iter_mut().enumerate() {
                *p = self.policy_bias[m] + dot(&self.policy_weight[m * n..(m + 1) * n], a);
            }
            let moves = (0..self.move_bias.len() / 2)
                .map(|m| {
                    [0, 1].map(|k| {
                        let r = 2 * m + k;
                        self.move_bias[r] + dot(&self.move_weight[r * n..(r + 1) * n], a)
                    })
                })
                .collect();
            Output {
                value: self.value_head(pos, a),
                policy,
                moves,
            }
        })
    }

    /// The logit of the chance that the side to move gets what it plays for
    /// (a forced win when it is `attacker`, a stop to one otherwise), as the
    /// search computes it.
    pub fn success_logit(&self, pos: &Game, attacker: Player) -> f32 {
        self.success_logit_near(&self.accumulator(), pos, attacker)
    }

    /// `success_logit`, from an accumulator holding a position near `pos`.
    pub fn success_logit_near(&self, acc: &Accumulator, pos: &Game, attacker: Player) -> f32 {
        let diffs = self.near_diffs(acc, pos);
        self.success_from(acc, pos, &diffs, attacker)
    }

    /// `success_logit` for `child`, a move from `parent`, when `acc` holds
    /// `parent` (see `value_child`).
    pub fn success_logit_child(
        &self,
        acc: &Accumulator,
        parent: &Game,
        child: &Game,
        attacker: Player,
    ) -> f32 {
        let diffs = self.child_diffs(acc, parent, child);
        self.success_from(acc, child, &diffs, attacker)
    }

    /// The one value logit `success_logit` needs, computed alone.
    fn success_from(
        &self,
        acc: &Accumulator,
        pos: &Game,
        diffs: &[(Rows, Rows); 2],
        attacker: Player,
    ) -> f32 {
        if pos.player() == attacker {
            self.value_from(acc, pos, diffs, [0])[0]
        } else {
            -self.value_from(acc, pos, diffs, [1])[0]
        }
    }

    /// Room for one position's activations, for `activate`.
    pub fn activations(&self) -> Activations {
        Activations {
            units: vec![f32x16::splat(0.0); 2 * self.hidden / LANES],
        }
    }

    /// Sets `out` to the activations of `pos`, whose units `acc` holds.
    pub fn activate(&self, acc: &Accumulator, pos: &Game, out: &mut Activations) {
        debug_assert_eq!(
            acc.inputs,
            [Inputs::of(pos, Player::X), Inputs::of(pos, Player::O)]
        );
        assert_eq!(out.units.len(), 2 * self.hidden / LANES);
        // Safe: `out` and both views' units have the lengths the kernel reads.
        unsafe {
            match self.simd {
                Simd::Avx512 => self.activate_avx512(acc, pos, &mut out.units),
                Simd::Avx2 => self.activate_avx2(acc, pos, &mut out.units),
                Simd::Generic => self.activate_kernel(acc, pos, &mut out.units),
            }
        }
    }

    /// The move head's `success_logit` for the position after a move to
    /// `square` (`9 * board + cell`) from `parent`, whose activations `act`
    /// holds. The network must have a move head.
    pub fn move_success_logit(
        &self,
        act: &Activations,
        parent: &Game,
        square: usize,
        attacker: Player,
    ) -> f32 {
        assert!(square < 81 && self.has_moves() && act.units.len() == 2 * self.hidden / LANES);
        // Safe: the row is within the move weights, and `act` has a row's
        // length.
        let logit = |k: usize| unsafe {
            match self.simd {
                Simd::Avx512 => self.move_avx512(act, 2 * square + k),
                Simd::Avx2 => self.move_avx2(act, 2 * square + k),
                Simd::Generic => self.move_kernel(act, 2 * square + k),
            }
        };
        if parent.player().other() == attacker {
            logit(0)
        } else {
            -logit(1)
        }
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx512f"))]
    unsafe fn activate_avx512(&self, acc: &Accumulator, pos: &Game, out: &mut [f32x16]) {
        self.activate_kernel(acc, pos, out)
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
    unsafe fn activate_avx2(&self, acc: &Accumulator, pos: &Game, out: &mut [f32x16]) {
        self.activate_kernel(acc, pos, out)
    }

    /// Both views' units, the side to move's first, clipped to [0, 2^`shift`]
    /// and squared, as `accumulate_heads` computes them.
    #[inline(always)]
    unsafe fn activate_kernel(&self, acc: &Accumulator, pos: &Game, out: &mut [f32x16]) {
        let n = self.hidden / LANES;
        let (zero, top) = (
            f32x16::splat(0.0),
            f32x16::splat((1u32 << self.shift) as f32),
        );
        let me = pos.player();
        for (half, view) in [me, me.other()].iter().enumerate() {
            let units = &acc.units[view.as_bit()];
            for j in 0..n {
                let c = f32x16::round_from_int(*units.get_unchecked(j))
                    .max(zero)
                    .min(top);
                *out.get_unchecked_mut(half * n + j) = c * c;
            }
        }
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx512f"))]
    unsafe fn move_avx512(&self, act: &Activations, row: usize) -> f32 {
        self.move_kernel(act, row)
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
    unsafe fn move_avx2(&self, act: &Activations, row: usize) -> f32 {
        self.move_kernel(act, row)
    }

    /// Logit `row` of the move head, from a position's activations.
    #[inline(always)]
    unsafe fn move_kernel(&self, act: &Activations, row: usize) -> f32 {
        let a = &act.units;
        let n = a.len();
        let w = self
            .move_weight_scaled
            .get_unchecked(row * n..(row + 1) * n);
        let mut sums = [f32x16::splat(0.0); 4];
        for base in (0..n).step_by(4) {
            for (q, sum) in sums.iter_mut().enumerate() {
                let j = base + q;
                if j < n {
                    *sum += *a.get_unchecked(j) * *w.get_unchecked(j);
                }
            }
        }
        self.move_bias[row] + sum_lanes((sums[0] + sums[1]) + (sums[2] + sums[3]))
    }

    /// An accumulator holding no inputs.
    pub fn accumulator(&self) -> Accumulator {
        Accumulator {
            units: [self.ft_bias_int.clone(), self.ft_bias_int.clone()],
            inputs: [Inputs::default(); 2],
        }
    }

    /// The offsets of the weight rows of the inputs in `new` and not in
    /// `old`, and of those in `old` and not in `new`.
    fn rows(&self, new: Inputs, old: Inputs) -> (Rows, Rows) {
        let n = self.hidden / LANES;
        let (mut add, mut sub) = (Rows::new(), Rows::new());
        new.without(old).each(|i| add.push(i * n));
        old.without(new).each(|i| sub.push(i * n));
        (add, sub)
    }

    /// Moves `acc` to `pos`, adding the weights of the inputs `pos` has and
    /// `acc`'s position lacked, and subtracting those of the inputs it lost:
    /// cheap when the two positions are a few moves apart.
    pub fn update(&self, acc: &mut Accumulator, pos: &Game) {
        for &view in &[Player::X, Player::O] {
            let inputs = Inputs::of(pos, view);
            let (add, sub) = self.rows(inputs, acc.inputs[view.as_bit()]);
            let units = &mut acc.units[view.as_bit()];
            // Safe: every offset is a row of `ft_weight_int`, and `units` has
            // a row's length.
            unsafe {
                match self.simd {
                    Simd::Avx512 => self.apply_avx512(units, &add, &sub),
                    Simd::Avx2 => self.apply_avx2(units, &add, &sub),
                    Simd::Generic => self.apply(units, &add, &sub),
                }
            }
            acc.inputs[view.as_bit()] = inputs;
        }
    }

    /// Sets `acc` to `pos`, summing from the biases.
    pub fn refresh(&self, acc: &mut Accumulator, pos: &Game) {
        for units in acc.units.iter_mut() {
            units.copy_from_slice(&self.ft_bias_int);
        }
        acc.inputs = [Inputs::default(); 2];
        self.update(acc, pos);
    }

    /// Adds the rows at offsets `add` to `units` and subtracts those at
    /// `sub`.
    #[inline(always)]
    unsafe fn apply(&self, units: &mut [i32x16], add: &[usize], sub: &[usize]) {
        let n = units.len();
        debug_assert_eq!(n, self.hidden / LANES);
        let w = &self.ft_weight_int;
        for &r in add {
            for (x, y) in units.iter_mut().zip(w.get_unchecked(r..r + n)) {
                *x += *y;
            }
        }
        for &r in sub {
            for (x, y) in units.iter_mut().zip(w.get_unchecked(r..r + n)) {
                *x -= *y;
            }
        }
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx512f"))]
    unsafe fn apply_avx512(&self, units: &mut [i32x16], add: &[usize], sub: &[usize]) {
        self.apply(units, add, sub)
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
    unsafe fn apply_avx2(&self, units: &mut [i32x16], add: &[usize], sub: &[usize]) {
        self.apply(units, add, sub)
    }

    /// The value logits of `pos` as the search computes them, from `acc`
    /// holding the hidden units of any position: only the weights of the
    /// inputs that differ are added or subtracted, so this is cheap when the
    /// positions are near. The result does not depend on which position
    /// `acc` holds.
    pub fn value_near(&self, acc: &Accumulator, pos: &Game) -> [f32; 2] {
        let diffs = self.near_diffs(acc, pos);
        self.value_from(acc, pos, &diffs, [0, 1])
    }

    /// `value_near` for `child`, a move from `parent`, when `acc` holds
    /// `parent`: the inputs that differ are found from the move alone.
    pub fn value_child(&self, acc: &Accumulator, parent: &Game, child: &Game) -> [f32; 2] {
        let diffs = self.child_diffs(acc, parent, child);
        self.value_from(acc, child, &diffs, [0, 1])
    }

    /// The rows that turn `acc`'s units into `pos`'s, for `pos`'s side to
    /// move and then its opponent: the rows to add, and the rows to subtract.
    fn near_diffs(&self, acc: &Accumulator, pos: &Game) -> [(Rows, Rows); 2] {
        let me = pos.player();
        [me, me.other()].map(|v| self.rows(Inputs::of(pos, v), acc.inputs[v.as_bit()]))
    }

    /// `near_diffs` for `child`, a move from `parent`, when `acc` holds
    /// `parent`, found from the move alone.
    fn child_diffs(&self, acc: &Accumulator, parent: &Game, child: &Game) -> [(Rows, Rows); 2] {
        debug_assert_eq!(
            acc.inputs,
            [Inputs::of(parent, Player::X), Inputs::of(parent, Player::O)]
        );
        let occupied =
            |g: &Game| g.local_boards.squares(Player::X) | g.local_boards.squares(Player::O);
        let added = occupied(child) & !occupied(parent);
        debug_assert_eq!(added.count_ones(), 1, "not a move from `parent`");
        let square = added.trailing_zeros() as usize;
        let board = square / 9;
        let mover = parent.player();
        let n = self.hidden / LANES;
        // [the child's side to move, the mover], as `value_from` takes them.
        let mut diffs = [(Rows::new(), Rows::new()), (Rows::new(), Rows::new())];
        let (to_move, moved) = diffs.split_at_mut(1);
        let views = [(mover.other(), &mut to_move[0]), (mover, &mut moved[0])];
        if child.global_states.in_play(board) {
            for (view, (add, _)) in views {
                add.push((if view == mover { square } else { 81 + square }) * n);
            }
        } else {
            // The board closed: its squares leave both views, and its result
            // enters them.
            let cells = 0x1ffu128 << (9 * board);
            let drawn = child.global_states.drawbits() & (1 << board) != 0;
            for (view, (add, sub)) in views {
                let had = acc.inputs[view.as_bit()];
                Inputs {
                    mine: had.mine & cells,
                    theirs: had.theirs & cells,
                    boards: 0,
                }
                .each(|i| sub.push(i * n));
                let result = if drawn {
                    180
                } else if view == mover {
                    162
                } else {
                    171
                };
                add.push((result + board) * n);
            }
        }
        diffs
    }

    /// Value logits of `pos` from `acc`, where `diffs` holds the rows to add
    /// to and subtract from `acc`'s units for `pos`'s side to move, then for
    /// its opponent: logit k of `pos`'s bucket for each k in `outputs` (0,
    /// the side to move can force a win; 1, its opponent can). Each logit is
    /// computed the same way whichever others are asked for.
    fn value_from<const K: usize>(
        &self,
        acc: &Accumulator,
        pos: &Game,
        diffs: &[(Rows, Rows); 2],
        outputs: [usize; K],
    ) -> [f32; K] {
        // Safe: the kernel reads only rows of the weights, within `hidden`.
        unsafe {
            match self.simd {
                Simd::Avx512 => self.value_avx512(acc, pos, diffs, outputs),
                Simd::Avx2 => self.value_avx2(acc, pos, diffs, outputs),
                Simd::Generic => self.value_kernel(acc, pos, diffs, outputs),
            }
        }
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx512f"))]
    unsafe fn value_avx512<const K: usize>(
        &self,
        acc: &Accumulator,
        pos: &Game,
        diffs: &[(Rows, Rows); 2],
        outputs: [usize; K],
    ) -> [f32; K] {
        self.value_kernel(acc, pos, diffs, outputs)
    }

    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
    unsafe fn value_avx2<const K: usize>(
        &self,
        acc: &Accumulator,
        pos: &Game,
        diffs: &[(Rows, Rows); 2],
        outputs: [usize; K],
    ) -> [f32; K] {
        self.value_kernel(acc, pos, diffs, outputs)
    }

    #[inline(always)]
    unsafe fn value_kernel<const K: usize>(
        &self,
        acc: &Accumulator,
        pos: &Game,
        diffs: &[(Rows, Rows); 2],
        outputs: [usize; K],
    ) -> [f32; K] {
        let n = self.hidden / LANES;
        let rows = outputs.map(|k| 2 * pos.board_to_play().unwrap_or(9) + k);
        let (w, v) = (&self.ft_weight_int, &self.value_weight_scaled);
        let top = f32x16::splat((1u32 << self.shift) as f32);
        // Four partial sums per head, so that consecutive vectors' products
        // add up in parallel.
        let mut sums = [[f32x16::splat(0.0); 4]; K];
        let me = pos.player();
        for (half, (&view, (add, sub))) in [me, me.other()].iter().zip(diffs).enumerate() {
            let units = &acc.units[view.as_bit()];
            let heads = rows.map(|r| v.get_unchecked((2 * r + half) * n..(2 * r + half + 1) * n));
            if let ([r], []) = (&add[..], &sub[..]) {
                let r = *r;
                accumulate_heads(&mut sums, heads, top, n, |j| {
                    *units.get_unchecked(j) + *w.get_unchecked(r + j)
                });
            } else {
                accumulate_heads(&mut sums, heads, top, n, |j| {
                    let mut a = *units.get_unchecked(j);
                    for &r in add.iter() {
                        a += *w.get_unchecked(r + j);
                    }
                    for &r in sub.iter() {
                        a -= *w.get_unchecked(r + j);
                    }
                    a
                });
            }
        }
        let mut out = [0.0; K];
        for ((o, &r), s) in out.iter_mut().zip(&rows).zip(&sums) {
            *o = self.value_bias[r] + sum_lanes((s[0] + s[1]) + (s[2] + s[3]));
        }
        out
    }
}

/// Adds one view's share of the value heads to `sums`: each of the `n`
/// vectors of hidden units that `unit` gives, clipped to [0, `top`] and
/// squared, times the heads' weights. Vector j goes to partial sum j % 4.
#[inline(always)]
unsafe fn accumulate_heads<const K: usize>(
    sums: &mut [[f32x16; 4]; K],
    heads: [&[f32x16]; K],
    top: f32x16,
    n: usize,
    unit: impl Fn(usize) -> i32x16,
) {
    let zero = f32x16::splat(0.0);
    for base in (0..n).step_by(4) {
        for q in 0..4 {
            let j = base + q;
            if j < n {
                let c = f32x16::round_from_int(unit(j)).max(zero).min(top);
                let c = c * c;
                for (sum, head) in sums.iter_mut().zip(&heads) {
                    sum[q] += c * *head.get_unchecked(j);
                }
            }
        }
    }
}
