//! uttt.ai's policy-value network (Apache-2.0, Arkadiusz Nowaczyński),
//! evaluated in plain Rust. The page fetches its weights, `net.bin`, written
//! by `web/tools/export_net.py`, which also documents the layout and checks
//! the same arithmetic against onnxruntime.
//!
//! A port of uttt.ai's `PolicyValueNetwork` (utttpy/selfplay/policy_value_network.py),
//! under the Apache License 2.0: web/LICENSE-uttt.ai.
//!
//! Activations are kept per square: the network's 3×3 grid is the nine
//! boards, and its 9×9 grid is the 81 squares, so square 9 × board + cell is
//! board `board`'s position `cell` at either scale.

use ultimattt::game::{CellState, Game, Move};

const MAGIC: &[u8] = b"UTTTNET1";

struct Dense {
    w: Vec<f32>,
    b: Vec<f32>,
    inputs: usize,
}

impl Dense {
    /// `out[o] = b[o] + w[o] · x`.
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        for (o, y) in out.iter_mut().enumerate() {
            *y = self.b[o] + dot(&self.w[o * self.inputs..(o + 1) * self.inputs], x);
        }
    }

    /// Row `row` alone: one output of a layer stored as several blocks of rows.
    fn row(&self, row: usize, x: &[f32]) -> f32 {
        dot(&self.w[row * self.inputs..(row + 1) * self.inputs], x)
    }
}

/// A residual block: 256 → 512 (1×1), a depthwise 3×3 that folds the nine
/// boards into one, and a transposed 3×3 back to nine boards of 256.
struct Block {
    expand: Dense,
    depthwise: Dense,
    spread: Dense,
}

pub struct Net {
    first: Dense,
    second: [Dense; 2],
    blocks: [Block; 3],
    policy: Dense,
    upsample: Dense,
    logits: [Dense; 2],
    value_heads: [(Dense, Dense); 2],
    value_fc: [Dense; 2],
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        for k in 0..8 {
            acc[k] += a[8 * i + k] * b[8 * i + k];
        }
    }
    let mut sum = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for i in 8 * chunks..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn elu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x.exp_m1()
    }
}

fn half(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let v = match exp {
        0 => return (if sign != 0 { -1.0 } else { 1.0 }) * frac as f32 / (1u32 << 24) as f32,
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(v)
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<Vec<f32>, String> {
        if self.bytes.len() < 2 * n {
            return Err("net.bin is too short".into());
        }
        let (head, rest) = self.bytes.split_at(2 * n);
        self.bytes = rest;
        Ok(head.chunks_exact(2).map(|c| half(u16::from_le_bytes([c[0], c[1]]))).collect())
    }

    fn dense(&mut self, outputs: usize, inputs: usize) -> Result<Dense, String> {
        Ok(Dense { w: self.take(outputs * inputs)?, b: self.take(outputs)?, inputs })
    }

    /// A transposed convolution or a depthwise one, stored as `rows` rows of
    /// `inputs` with `bias` biases.
    fn blocked(&mut self, rows: usize, inputs: usize, bias: usize) -> Result<Dense, String> {
        Ok(Dense { w: self.take(rows * inputs)?, b: self.take(bias)?, inputs })
    }
}

impl Net {
    pub fn load(bytes: &[u8]) -> Result<Net, String> {
        if !bytes.starts_with(MAGIC) {
            return Err("net.bin is not a network".into());
        }
        let mut r = Reader { bytes: &bytes[MAGIC.len()..] };
        let first = r.dense(128, 36)?;
        let second = [r.dense(128, 128)?, r.dense(256, 128)?];
        let mut block = || -> Result<Block, String> {
            Ok(Block {
                expand: r.dense(512, 256)?,
                depthwise: r.blocked(512, 9, 512)?,
                spread: r.blocked(9 * 256, 512, 256)?,
            })
        };
        let blocks = [block()?, block()?, block()?];
        let policy = r.dense(256, 640)?;
        let upsample = r.blocked(9 * 256, 256, 256)?;
        let logits = [r.dense(128, 260)?, r.dense(1, 128)?];
        let mut head = || -> Result<(Dense, Dense), String> { Ok((r.dense(256, 256)?, r.blocked(256, 9, 256)?)) };
        let value_heads = [head()?, head()?];
        let value_fc = [r.dense(128, 512)?, r.dense(1, 128)?];
        if !r.bytes.is_empty() {
            return Err("net.bin is too long".into());
        }
        Ok(Net { first, second, blocks, policy, upsample, logits, value_heads, value_fc })
    }

    /// The network's input for `g`: per square, the side to move's mark, the
    /// other side's, +1 or -1 for X or O to move, and whether it's legal.
    fn planes(g: &Game, legal: &[Move]) -> [[f32; 4]; 81] {
        let me = g.player();
        let turn = if me.as_str() == "X" { 1.0 } else { -1.0 };
        let mut p = [[0.0, 0.0, turn, 0.0]; 81];
        for sq in 0..81 {
            if let CellState::Played(who) = g.at(sq / 9, sq % 9) {
                p[sq][if who == me { 0 } else { 1 }] = 1.0;
            }
        }
        for m in legal {
            p[9 * m.global() + m.local()][3] = 1.0;
        }
        p
    }

    /// The policy's logit for each move in `legal`, and the value for the side
    /// to move, -1 to 1.
    pub fn eval(&self, g: &Game, legal: &[Move]) -> (Vec<f32>, f32) {
        let planes = Net::planes(g, legal);

        let mut x1 = [[0f32; 128]; 9];
        for b in 0..9 {
            let mut patch = [0f32; 36];
            for i in 0..4 {
                for c in 0..9 {
                    patch[9 * i + c] = planes[9 * b + c][i];
                }
            }
            self.first.apply(&patch, &mut x1[b]);
            x1[b].iter_mut().for_each(|v| *v = elu(*v));
        }

        let mut x2 = [[0f32; 256]; 9];
        for b in 0..9 {
            let mut h = [0f32; 128];
            self.second[0].apply(&x1[b], &mut h);
            h.iter_mut().for_each(|v| *v = elu(*v));
            self.second[1].apply(&h, &mut x2[b]);
            x2[b].iter_mut().for_each(|v| *v = v.max(0.0));
        }

        let mut x = x2;
        for block in &self.blocks {
            let mut h = vec![[0f32; 512]; 9];
            for b in 0..9 {
                block.expand.apply(&x[b], &mut h[b]);
                h[b].iter_mut().for_each(|v| *v = elu(*v));
            }
            let mut d = [0f32; 512];
            for (c, y) in d.iter_mut().enumerate() {
                let w = &block.depthwise.w[9 * c..9 * c + 9];
                *y = elu(block.depthwise.b[c] + (0..9).map(|b| w[b] * h[b][c]).sum::<f32>());
            }
            for b in 0..9 {
                for o in 0..256 {
                    let v = block.spread.b[o] + block.spread.row(256 * b + o, &d) + x[b][o];
                    x[b][o] = v.max(0.0);
                }
            }
        }
        let x3 = x;

        let mut p = [[0f32; 256]; 9];
        for b in 0..9 {
            let mut cat = [0f32; 640];
            cat[..128].copy_from_slice(&x1[b]);
            cat[128..384].copy_from_slice(&x2[b]);
            cat[384..].copy_from_slice(&x3[b]);
            self.policy.apply(&cat, &mut p[b]);
            p[b].iter_mut().for_each(|v| *v = elu(*v));
        }

        // The policy at each legal square only: the rest is never used.
        let logits = legal
            .iter()
            .map(|m| {
                let (b, c) = (m.global(), m.local());
                let mut cat = [0f32; 260];
                cat[..4].copy_from_slice(&planes[9 * b + c]);
                for o in 0..256 {
                    cat[4 + o] = elu(self.upsample.b[o] + self.upsample.row(256 * c + o, &p[b]));
                }
                let mut h = [0f32; 128];
                self.logits[0].apply(&cat, &mut h);
                h.iter_mut().for_each(|v| *v = elu(*v));
                let mut l = [0f32; 1];
                self.logits[1].apply(&h, &mut l);
                l[0].clamp(-32.0, 32.0)
            })
            .collect();

        let mut v = [0f32; 512];
        for (k, (src, (conv, depthwise))) in [&x2, &x3].into_iter().zip(&self.value_heads).enumerate() {
            let mut h = [[0f32; 256]; 9];
            for b in 0..9 {
                conv.apply(&src[b], &mut h[b]);
            }
            for c in 0..256 {
                let w = &depthwise.w[9 * c..9 * c + 9];
                v[256 * k + c] = elu(depthwise.b[c] + (0..9).map(|b| w[b] * h[b][c]).sum::<f32>());
            }
        }
        let mut h = [0f32; 128];
        self.value_fc[0].apply(&v, &mut h);
        h.iter_mut().for_each(|v| *v = elu(*v));
        let mut out = [0f32; 1];
        self.value_fc[1].apply(&h, &mut out);
        (logits, out[0].tanh())
    }
}
