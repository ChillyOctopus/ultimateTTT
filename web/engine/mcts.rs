//! uttt.ai's search: Monte Carlo tree search guided by its network (PUCT),
//! following its website's `NeuralMonteCarloTreeSearch.js`. Single-threaded
//! and sequential: each simulation walks down the tree by the PUCT score,
//! expands the leaf it reaches with the network's policy, and backs up the
//! network's value (or the result, at the game's end).
//!
//! A port of uttt.ai's code, under the Apache License 2.0: web/LICENSE-uttt.ai.

use crate::net::Net;
use crate::Rng;
use ultimattt::clock::Instant;
use ultimattt::game::{Game, GameState, Move};

/// uttt.ai's exploration strength.
pub const EXPLORATION: f32 = 2.0;

#[derive(Clone, Copy)]
pub struct Node {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    sum: f32,
    first: u32,
    count: u32,
}

impl Node {
    fn new(mv: Move, prior: f32) -> Node {
        Node { mv, prior, visits: 0, sum: 0.0, first: 0, count: 0 }
    }

    /// The mean value for the side to move at this node.
    pub fn mean(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.sum / self.visits as f32
        }
    }
}

pub struct Tree {
    game: Game,
    nodes: Vec<Node>,
    /// Each node's network value, for the analysis: none until it's expanded.
    values: Vec<Option<f32>>,
}

impl Tree {
    pub fn new(game: Game) -> Tree {
        Tree { game, nodes: vec![Node::new(Move::none(), 1.0)], values: vec![None] }
    }

    /// The tree after root move `mv`: its subtree, kept, as uttt.ai keeps it
    /// from one move to the next. A new tree if `mv` isn't a root move.
    pub fn advance(self, mv: Move) -> Tree {
        let game = self.game.make_move(mv).expect("a legal move");
        let Some(start) = self.children().iter().position(|c| c.mv == mv) else {
            return Tree::new(game);
        };
        let start = self.nodes[0].first as usize + start;
        let mut nodes = vec![self.nodes[start]];
        let mut values = vec![self.values[start]];
        let mut i = 0;
        while i < nodes.len() {
            let (first, count) = (nodes[i].first as usize, nodes[i].count as usize);
            if count > 0 {
                nodes[i].first = nodes.len() as u32;
                nodes.extend_from_slice(&self.nodes[first..first + count]);
                values.extend_from_slice(&self.values[first..first + count]);
            }
            i += 1;
        }
        Tree { game, nodes, values }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    pub fn children(&self) -> &[Node] {
        let r = &self.nodes[0];
        &self.nodes[r.first as usize..(r.first + r.count) as usize]
    }

    /// The network's own value for root child `i`, from the root's side, once
    /// it has been visited.
    pub fn child_value(&self, i: usize) -> Option<f32> {
        self.values[self.nodes[0].first as usize + i].map(|v| -v)
    }

    /// Simulate until the root has `visits` visits, or `ms` milliseconds have
    /// passed (0 for no limit). With `wide`, every root move is visited at
    /// least once first, so each has a value to show.
    pub fn run(&mut self, net: &Net, visits: u32, ms: f64, wide: bool, rng: &mut Rng) {
        let start = Instant::now();
        if self.game.game_over() {
            return;
        }
        if self.nodes[0].visits == 0 {
            self.simulate(net, None, rng);
        }
        if wide {
            for i in 0..self.nodes[0].count as usize {
                if self.nodes[self.nodes[0].first as usize + i].visits == 0 {
                    self.simulate(net, Some(i), rng);
                }
            }
        }
        while self.nodes[0].visits < visits && (ms <= 0.0 || start.elapsed().as_secs_f64() * 1000.0 < ms) {
            self.simulate(net, None, rng);
        }
    }

    fn select(&self, parent: usize, rng: &mut Rng) -> usize {
        let p = &self.nodes[parent];
        let root_n = (p.visits as f32).sqrt();
        let mut best = f32::NEG_INFINITY;
        let mut ties: Vec<usize> = Vec::new();
        for i in p.first as usize..(p.first + p.count) as usize {
            let c = &self.nodes[i];
            let score = -c.mean() + EXPLORATION * c.prior.max(0.01) * root_n / (c.visits as f32 + 1.0);
            if score > best {
                best = score;
                ties.clear();
                ties.push(i);
            } else if score == best {
                ties.push(i);
            }
        }
        rng.pick(&ties)
    }

    fn simulate(&mut self, net: &Net, forced: Option<usize>, rng: &mut Rng) {
        let mut path = vec![0usize];
        let mut g = self.game.clone();
        let mut node = 0;
        while self.nodes[node].count > 0 {
            node = match (forced, path.len()) {
                (Some(i), 1) => self.nodes[0].first as usize + i,
                _ => self.select(node, rng),
            };
            g = g.make_move(self.nodes[node].mv).expect("the tree holds legal moves");
            path.push(node);
        }
        let value = match g.game_state() {
            GameState::Drawn => 0.0,
            GameState::Won(_) => -1.0,
            GameState::InPlay => self.expand(net, node, &g),
        };
        let mut sign = 1.0;
        for &n in path.iter().rev() {
            self.nodes[n].visits += 1;
            self.nodes[n].sum += sign * value;
            sign = -sign;
        }
    }

    fn expand(&mut self, net: &Net, node: usize, g: &Game) -> f32 {
        let legal: Vec<Move> = g.all_moves().collect();
        let (logits, value) = net.eval(g, &legal);
        let top = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
        let total: f32 = exps.iter().sum();
        self.nodes[node].first = self.nodes.len() as u32;
        self.nodes[node].count = legal.len() as u32;
        for (m, e) in legal.into_iter().zip(exps) {
            self.nodes.push(Node::new(m, e / total));
            self.values.push(None);
        }
        self.values[node] = Some(value);
        value
    }

    /// uttt.ai's chance for each root move: its visits, blended with the
    /// network's prior while the visits are few.
    pub fn chances(&self) -> Vec<f32> {
        let n = self.nodes[0].visits as f32;
        let weight = (100.0 - 0.1 * n).max(0.0);
        let denominator = (n + weight - 1.0).max(1.0);
        self.children().iter().map(|c| (c.visits as f32 + c.prior * weight) / denominator).collect()
    }

    /// The move uttt.ai would play: the highest chance, ties broken at random.
    pub fn best(&self, rng: &mut Rng) -> Move {
        let chances = self.chances();
        let top = chances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ties: Vec<Move> = self.children().iter().zip(&chances).filter(|(_, &p)| p >= top).map(|(c, _)| c.mv).collect();
        rng.pick(&ties)
    }
}
