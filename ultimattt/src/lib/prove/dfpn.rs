// Modified by the ultimateTTT project (2026): the single-threaded
// search honors `limit`, `endgame: false` turns off the endgame analysis,
// `Stats::fresh` counts first visits, `Stats::profile` breaks the work down by
// kind of position, `symmetry` skips mirror-image moves, `init` chooses how
// new positions' proof numbers start, by rules or by a network (`net`, and
// its move head with `move_head`),
// `proof_size` walks and checks the proof a search found, `log_solved` writes
// the positions a search settles as training data, and `attacker` lets either
// player be the one trying to win. It times itself with `clock::Instant`,
// which also works in WebAssembly.

use crate::endgame;
use crate::game;
use crate::nnue;
use crate::progress::Ticker;
use crate::prove;
use crate::prove::spdfpn;
use crate::prove::{Bounds, INFINITY};
use crate::symmetry;
use crate::table;
use crate::tactics;
use crate::util;

use hdrhistogram::Histogram;
use serde::Serialize;

use std::cmp::{max, min};
use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use crate::clock::Instant;
use std::time::Duration;
use std::{fs, io, mem};

mod solved_log;
use solved_log::SolvedLog;

#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub mid: usize,
    /// MID calls on a position with no recorded work: a first visit, or one
    /// whose table entry was lost. The other calls re-expand earlier work.
    pub fresh: usize,
    pub generated_children: usize,
    pub try_calls: usize,
    pub jobs: usize,
    pub solved: usize,
    pub endgame_solve: usize,
    pub endgame_move: usize,
    /// Moves skipped because a symmetry of the position maps them onto a
    /// move that is searched.
    pub symmetric_moves: usize,
    /// With `Config::proof_size`: the positions in the proof (or disproof)
    /// the search found that a search has to expand, and the positions that
    /// proof needs but the table no longer holds as solved. See
    /// `MID::proof_size`.
    pub proof: usize,
    pub proof_missing: usize,
    #[serde(serialize_with = "util::serialize_histogram")]
    pub branch: Histogram<u64>,
    #[serde(serialize_with = "util::serialize_histogram")]
    pub open_boards: Histogram<u64>,
    #[serde(flatten)]
    pub tt: table::Stats,
    pub endgame: endgame::Stats,
    pub profile: Profile,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            mid: 0,
            fresh: 0,
            generated_children: 0,
            try_calls: 0,
            jobs: 0,
            solved: 0,
            endgame_solve: 0,
            endgame_move: 0,
            symmetric_moves: 0,
            proof: 0,
            proof_missing: 0,
            branch: Histogram::new_with_max(81, 3).unwrap(),
            open_boards: Histogram::new_with_max(9, 2).unwrap(),
            tt: Default::default(),
            endgame: Default::default(),
            profile: Default::default(),
        }
    }
}

impl Stats {
    pub fn merge(&self, other: &Stats) -> Stats {
        Stats {
            mid: self.mid + other.mid,
            fresh: self.fresh + other.fresh,
            generated_children: self.generated_children + other.generated_children,
            jobs: self.jobs + other.jobs,
            try_calls: self.try_calls + other.try_calls,
            solved: self.solved + other.solved,
            endgame_solve: self.endgame_solve + other.endgame_solve,
            endgame_move: self.endgame_move + other.endgame_move,
            symmetric_moves: self.symmetric_moves + other.symmetric_moves,
            proof: self.proof + other.proof,
            proof_missing: self.proof_missing + other.proof_missing,
            branch: util::merge_histogram(&self.branch, &other.branch),
            open_boards: util::merge_histogram(&self.open_boards, &other.open_boards),
            tt: self.tt.merge(&other.tt),
            endgame: self.endgame.merge(&other.endgame),
            profile: self.profile.merge(&other.profile),
        }
    }
}

/// Where the search's work goes, by kind of position. Each MID call is one
/// unit of work.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Profile {
    /// MID calls and children generated, indexed by `Profile::kind`.
    pub mids: [u64; 16],
    pub children: [u64; 16],
    /// MID calls by the number of local boards still open.
    pub mids_by_open: [u64; 10],
}

impl Profile {
    /// Bit 0: the side to move has a free move. Bit 1: the side to move is
    /// the defender (an AND node). Bits 2 and 3: the side to move, and its
    /// opponent, have critical boards.
    pub fn kind(pos: &game::Game, analysis: &endgame::Analysis, attacker: game::Player) -> usize {
        let (mine, theirs) = analysis.threats();
        pos.board_to_play().is_none() as usize
            | ((pos.player() != attacker) as usize) << 1
            | (mine as usize) << 2
            | (theirs as usize) << 3
    }

    pub fn merge(&self, other: &Profile) -> Profile {
        let mut out = self.clone();
        for i in 0..16 {
            out.mids[i] += other.mids[i];
            out.children[i] += other.children[i];
        }
        for i in 0..10 {
            out.mids_by_open[i] += other.mids_by_open[i];
        }
        out
    }
}

#[repr(C)]
pub(in crate::prove) struct Entry {
    pub(in crate::prove) bounds: Bounds,  // 8
    pub(in crate::prove) hash: u64,       // 8
    pub(in crate::prove) work: u64,       // 8
    pub(in crate::prove) sync: AtomicU32, // 4
    pub(in crate::prove) pv: game::Move,  // 1
    pub(in crate::prove) child: u8,       // 1
}

pub fn sizeof_entry() -> usize {
    mem::size_of::<Entry>()
}

impl Clone for Entry {
    fn clone(&self) -> Self {
        Entry {
            bounds: self.bounds.clone(),
            hash: self.hash,
            work: self.work,
            sync: AtomicU32::new(0),
            pv: self.pv,
            child: self.child,
        }
    }
}

impl table::Entry for Entry {
    fn hash(&self) -> u64 {
        self.hash
    }

    fn better_than(&self, other: &Entry) -> bool {
        if self.hash == other.hash {
            if self.bounds.solved() != other.bounds.solved() {
                return self.bounds.solved();
            }
        }
        self.work >= other.work
    }

    fn valid(&self) -> bool {
        self.work != std::u64::MAX
    }
}

impl table::AtomicEntry for Entry {
    unsafe fn lock(e: *const Self) -> *const AtomicU32 {
        &(*e).sync
    }

    unsafe fn write(dst: *mut Self, v: &Self) {
        (*dst).bounds = v.bounds;
        (*dst).hash = v.hash;
        (*dst).work = v.work;
        (*dst).pv = v.pv;
        (*dst).child = v.child;
    }
}

impl Default for Entry {
    fn default() -> Self {
        Entry {
            bounds: Bounds { phi: 0, delta: 0 },
            hash: 0,
            work: std::u64::MAX,
            pv: game::Move::none(),
            child: std::u8::MAX,
            sync: AtomicU32::new(0),
        }
    }
}

#[derive(Clone)]
pub struct Config {
    pub threads: usize,
    pub table_size: usize,
    pub limit: Option<Duration>,
    pub debug: usize,
    pub epsilon: f64,
    pub max_work_per_job: u64,
    pub dump_table: Option<String>,
    pub load_table: Option<String>,
    pub dump_interval: Duration,
    pub probe_hash: Option<u64>,
    pub probe_log: String,
    pub endgame: bool,
    pub symmetry: bool,
    pub init: Init,
    /// The network `Init::Net` starts positions from.
    pub net: Option<Arc<nnue::Network>>,
    /// With `Init::Net`, start a position's children from the network's move
    /// head, evaluated once for the position, instead of evaluating each
    /// child. The network must have a move head.
    pub move_head: bool,
    /// After a single-threaded search, walk the proof it found and count its
    /// positions (`Stats::proof`).
    pub proof_size: bool,
    /// Write the positions a single-threaded search settles to this file (see
    /// `solved_log`), keeping about a `log_sample` fraction of them.
    pub log_solved: Option<String>,
    pub log_sample: f64,
    /// The player trying to force a win; the side to move when `None`.
    /// Single-threaded search only.
    pub attacker: Option<game::Player>,
}

/// How a position's proof numbers start before it is searched. They set the
/// order in which DFPN tries moves.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Init {
    /// (1, 1) for every position.
    Unity,
    /// Proving needs one good move, disproving needs every move refuted:
    /// (1, number of legal moves). Moves that leave the opponent few replies
    /// come first, and moves that hand them a free move come last.
    Mobility,
    /// Mobility weighted by tactics: (`unit`, `unit` per safe move plus 1 per
    /// risky move), where a risky move hands the opponent a free move or a
    /// board they can win (see `tactics::move_safety`). Refuting a position
    /// mostly means refuting its safe moves, so DFPN tries first the moves
    /// that leave the opponent few safe replies. That includes a move that
    /// gives a free move, when most replies would hand something back. Unit 1
    /// is Mobility.
    Safe(u32),
    /// From the network (`Config::net`): (1 + `unit` · c(p), 1 + `unit` ·
    /// c(1 − p)), rounded, where p is its chance that the side to move gets
    /// what it plays for. With `exponent` 0, c(q) = −ln q: proof numbers add
    /// up where every move must succeed and take the least where one must,
    /// as these costs do for independent chances. A positive exponent a
    /// makes unlikely outcomes costlier: c(q) = (q^−a − 1) / a. The logit
    /// of p is first rounded to a multiple of 1/32 within ±32, so that the
    /// search can look the bounds up.
    Net(u32, f32),
}

impl std::str::FromStr for Init {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unity" => Ok(Init::Unity),
            "mobility" => Ok(Init::Mobility),
            "safe" => Ok(Init::Safe(8)),
            "net" => Ok(Init::Net(32, 0.5)),
            _ => {
                let unit = |text: &str| text.parse().ok().filter(|&unit: &u32| unit >= 1);
                let net = |rest: &str| {
                    let mut parts = rest.splitn(2, ':');
                    let unit = unit(parts.next()?)?;
                    let exponent = match parts.next() {
                        Some(a) => a.parse().ok().filter(|&a: &f32| a >= 0.0)?,
                        None => 0.0,
                    };
                    Some(Init::Net(unit, exponent))
                };
                s.strip_prefix("safe:")
                    .and_then(unit)
                    .map(Init::Safe)
                    .or_else(|| s.strip_prefix("net:").and_then(net))
                    .ok_or_else(|| format!("unknown initialization: `{}`", s))
            }
        }
    }
}

/// The proof numbers a position in play starts with, when `attacker` is the
/// player trying to force a win.
pub fn initial_bounds(cfg: &Config, pos: &game::Game, attacker: game::Player) -> Bounds {
    match cfg.init {
        Init::Unity => Bounds::unity(),
        Init::Mobility => Bounds {
            phi: 1,
            delta: pos.move_count().max(1),
        },
        Init::Safe(unit) => {
            let (safe, risky) = tactics::move_safety(pos);
            Bounds {
                phi: unit,
                delta: (unit * safe + risky).max(1),
            }
        }
        Init::Net(unit, exponent) => {
            let net = cfg.net.as_ref().expect("--init=net needs a network");
            net_bounds(unit, exponent, net.success_logit(pos, attacker))
        }
    }
}

/// `Init::Net` rounds logits to a multiple of 1 / `NET_GRID` within
/// ±`NET_RANGE`.
const NET_GRID: f32 = 32.0;
const NET_RANGE: f32 = 32.0;

/// The place of logit `s`, once rounded, among the logits `Init::Net` uses.
fn net_index(s: f32) -> usize {
    ((s.clamp(-NET_RANGE, NET_RANGE) * NET_GRID).round() + NET_RANGE * NET_GRID) as usize
}

/// `Init::Net`'s bounds, from the logit `s` of the network's chance that the
/// side to move gets what it plays for.
fn net_bounds(unit: u32, exponent: f32, s: f32) -> Bounds {
    net_bounds_at(unit, exponent, net_index(s))
}

/// `Init::Net`'s bounds at a logit's place (`net_index`).
fn net_bounds_at(unit: u32, exponent: f32, index: usize) -> Bounds {
    let s = (index as f64 - (NET_RANGE * NET_GRID) as f64) / NET_GRID as f64;
    let a = exponent as f64;
    let cost = |x: f64| {
        // With p = sigmoid(s), −ln p = softplus(−s) and −ln(1 − p) = softplus(s).
        let nll = x.max(0.0) + (-x.abs()).exp().ln_1p();
        let c = if a > 0.0 { (a * nll).exp_m1() / a } else { nll };
        1 + (unit as f64 * c).round().min(1e6) as u32
    };
    Bounds {
        phi: cost(-s),
        delta: cost(s),
    }
}

/// What a search guided by the network keeps between positions: the hidden
/// units of the last position whose children it started, and `Init::Net`'s
/// bounds for every logit, which took as long to compute as the network.
pub(in crate::prove) struct Guide {
    /// The hash of the position `acc` holds.
    hash: Option<u64>,
    acc: nnue::Accumulator,
    /// For the move head: that position's activations, once computed.
    act: nnue::Activations,
    activated: bool,
    bounds: Vec<Bounds>,
}

impl Guide {
    fn new(net: &nnue::Network, unit: u32, exponent: f32) -> Guide {
        let places = 2 * (NET_RANGE * NET_GRID) as usize + 1;
        Guide {
            hash: None,
            acc: net.accumulator(),
            act: net.activations(),
            activated: false,
            bounds: (0..places)
                .map(|i| net_bounds_at(unit, exponent, i))
                .collect(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            threads: 0,
            table_size: table::DEFAULT_TABLE_SIZE,
            limit: None,
            debug: 0,
            epsilon: 0.5,
            max_work_per_job: 500,
            dump_table: None,
            load_table: None,
            dump_interval: Duration::from_secs(30),
            probe_hash: None,
            probe_log: "probe.csv".to_owned(),
            endgame: true,
            symmetry: true,
            init: Init::Safe(8),
            net: None,
            move_head: false,
            proof_size: false,
            log_solved: None,
            log_sample: 1.0,
            attacker: None,
        }
    }
}

pub struct ProveResult {
    pub value: prove::Evaluation,
    pub bounds: Bounds,
    pub stats: Stats,
    pub duration: Duration,
    pub work: u64,
    pub pv: Vec<game::Move>,
}

pub struct DFPN {
    start: Instant,
    cfg: Config,
    stats: Stats,
    root: game::Game,
}

#[derive(Clone)]
pub(in crate::prove) struct Child {
    pub(in crate::prove) position: game::Game,
    pub(in crate::prove) r#move: game::Move,
    pub(in crate::prove) entry: Entry,
}

const CHECK_TICK_WORK: u64 = 500000;
pub static TICK_TIME: Duration = Duration::from_millis(500);

impl DFPN {
    pub fn prove(cfg: &Config, g: &game::Game) -> ProveResult {
        let start = Instant::now();
        let mut prover = DFPN {
            root: g.clone(),
            start: start,
            cfg: cfg.clone(),
            stats: Default::default(),
        };
        let (result, pv, work) = prover.run();
        // The root's bounds are from its side to move's view: proof means
        // that player gets what it plays for, a win as the attacker or a stop
        // to one as the defender.
        let (attacker_wins, attacker_fails) = if cfg.attacker.map_or(true, |a| a == g.player()) {
            (result.bounds.phi, result.bounds.delta)
        } else {
            (result.bounds.delta, result.bounds.phi)
        };
        ProveResult {
            value: if attacker_wins == 0 {
                prove::Evaluation::True
            } else if attacker_fails == 0 {
                prove::Evaluation::False
            } else {
                prove::Evaluation::Unknown
            },
            work: work,
            bounds: result.bounds,
            stats: prover.stats.clone(),
            duration: Instant::now().duration_since(prover.start),
            pv: pv,
        }
    }

    fn run(&mut self) -> (Entry, Vec<game::Move>, u64) {
        let mut probe = Probe::from_config(&self.cfg);

        if self.cfg.threads == 0 {
            let table = if let Some(ref path) = self.cfg.load_table {
                table::TranspositionTable::<_, 4>::from_file(path).expect("invalid table file")
            } else {
                table::TranspositionTable::<_, 4>::with_memory(self.cfg.table_size)
            };
            let mut worker = MID {
                id: 0,
                cfg: &self.cfg,
                player: self.cfg.attacker.unwrap_or(self.root.player()),
                table: table,
                stats: Default::default(),
                stack: Vec::new(),
                guide: None,
                solved_log: self.cfg.log_solved.as_ref().map(|path| {
                    SolvedLog::create(path, self.cfg.log_sample)
                        .expect("unable to create the --log-solved file")
                }),
                probe: |stats: &Stats, pos: &game::Game, data: &Entry, children: &[Child]| {
                    if let Some(ref mut p) = probe {
                        if data.hash != p.hash {
                            return;
                        }
                        let now = Instant::now();
                        if now < p.tick && !data.bounds.solved() {
                            return;
                        }
                        p.do_probe(
                            now + Duration::from_millis(10),
                            stats.mid,
                            pos,
                            data,
                            children,
                        );
                    }
                },
            };
            let mut root = Entry {
                hash: self.root.zobrist(),
                bounds: Bounds::unity(),
                child: 0xff,
                pv: game::Move::none(),
                work: 0,
                sync: AtomicU32::new(0),
            };
            let mut work = 0;
            let mut dump_tick = Ticker::new(self.cfg.dump_interval);
            while !root.bounds.solved() {
                let (out, this_work, _) = worker.mid(
                    Bounds {
                        phi: INFINITY / 2,
                        delta: INFINITY / 2,
                    },
                    CHECK_TICK_WORK,
                    root,
                    &self.root,
                );
                root = out;
                work += this_work;
                let now = Instant::now();
                if self.cfg.debug > 0 {
                    let elapsed = now - self.start;
                    eprintln!(
                        "t={}.{:03}s mid={} root={:?}",
                        elapsed.as_secs(),
                        elapsed.subsec_millis(),
                        worker.stats.mid,
                        root.bounds,
                    );
                }

                if let Some(_) = self.cfg.dump_table {
                    if dump_tick.tick() {
                        dump_table(&self.cfg, &worker.table).expect("dump_table failed");
                    }
                }
                if self
                    .cfg
                    .limit
                    .map_or(false, |limit| now - self.start >= limit)
                {
                    break;
                }
            }
            self.stats = worker.stats.clone();
            let pv = extract_pv(&worker.cfg, &mut worker.table, &self.root);
            self.stats.tt = worker.table.stats();
            if self.cfg.proof_size && root.bounds.solved() {
                let (proof, missing) = worker.proof_size(&self.root, root.bounds.phi == 0);
                self.stats.proof = proof;
                self.stats.proof_missing = missing;
            }
            if let Some(ref mut log) = worker.solved_log {
                log.flush().expect("writing the --log-solved file");
            }
            dump_table(&self.cfg, &worker.table).expect("final dump_table");
            (root, pv, work)
        } else {
            assert!(
                self.cfg.attacker.map_or(true, |a| a == self.root.player()),
                "a parallel search's attacker is the side to move"
            );
            let (stats, root, pv, work) = spdfpn::run(self.start, &self.cfg, &self.root, probe);
            self.stats = stats;
            (root, pv, work)
        }
    }
}

pub(in crate::prove) struct Probe {
    pub(in crate::prove) tick: Instant,
    pub(in crate::prove) hash: u64,
    pub(in crate::prove) out: fs::File,
}

impl Probe {
    fn write_header(&mut self) {
        write!(
            self.out,
            "mid,notation,node_work,node_phi,node_delta,child,chash,work,phi,delta\n"
        )
        .expect("probe header");
    }

    pub(in crate::prove) fn do_probe(
        &mut self,
        tick: Instant,
        mid: usize,
        pos: &game::Game,
        data: &Entry,
        children: &[Child],
    ) {
        self.tick = tick;

        for (i, ch) in children.iter().enumerate() {
            write!(
                self.out,
                "{},{},{},{},{},{},{},{},{},{}\n",
                mid,
                game::notation::render(pos),
                data.work,
                data.bounds.phi,
                data.bounds.delta,
                i,
                ch.entry.hash,
                ch.entry.work,
                ch.entry.bounds.phi,
                ch.entry.bounds.delta,
            )
            .expect("probe line");
        }
    }

    pub(in crate::prove) fn from_config(cfg: &Config) -> Option<Self> {
        cfg.probe_hash.map(|h| {
            let probelog = fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&cfg.probe_log)
                .expect("open probe log");
            let mut probe = Probe {
                hash: h,
                tick: Instant::now(),
                out: probelog,
            };
            probe.write_header();
            probe
        })
    }
}

pub(in crate::prove) fn compute_bounds(children: &[Child]) -> Bounds {
    let mut out = Bounds {
        phi: INFINITY,
        delta: 0,
    };
    for ch in children.iter() {
        out.phi = min(out.phi, ch.entry.bounds.delta);
        out.delta = out.delta.saturating_add(ch.entry.bounds.phi);
    }
    out.delta = min(out.delta, INFINITY);
    return out;
}

// returns (child index, Bounds)
pub(in crate::prove) fn select_child(
    children: &[Child],
    bounds: Bounds,
    data: &mut Entry,
    epsilon: f64,
) -> (usize, Bounds) {
    let (mut delta_1, mut delta_2) = (INFINITY, INFINITY);
    let mut idx = children.len();
    for (i, ch) in children.iter().enumerate() {
        if ch.entry.bounds.delta < delta_1 {
            delta_2 = delta_1;
            delta_1 = ch.entry.bounds.delta;
            idx = i;
        } else if ch.entry.bounds.delta < delta_2 {
            delta_2 = ch.entry.bounds.delta;
        }
    }
    if data.child != std::u8::MAX && data.child != (idx as u8) {
        let child_bounds = thresholds(
            epsilon,
            bounds,
            data.bounds,
            children[data.child as usize].entry.bounds.phi,
            delta_1,
        );
        if children[data.child as usize].entry.bounds.delta < child_bounds.delta {
            return (data.child as usize, child_bounds);
        }
    }
    data.child = idx as u8;

    let child_bounds = thresholds(
        epsilon,
        bounds,
        data.bounds,
        children[idx].entry.bounds.phi,
        delta_2,
    );
    (idx, child_bounds)
}

pub fn thresholds(epsilon: f64, bounds: Bounds, nd: Bounds, phi_1: u32, delta_2: u32) -> Bounds {
    Bounds {
        phi: bounds.delta + phi_1 - nd.delta,
        delta: min(
            bounds.phi,
            max(delta_2 + 1, (delta_2 as f64 * (1.0 + epsilon)) as u32),
        ),
    }
}

pub(in crate::prove) fn populate_pv(data: &mut Entry, children: &[Child]) {
    // If the position is solved, store the PV. For winning moves,
    // find the shortest path to victory; for losing, the
    // most-delaying
    let ch = if data.bounds.phi == 0 {
        children
            .iter()
            .filter(|e| e.entry.bounds.delta == 0)
            .min_by_key(|e| e.entry.work)
            .expect("won node, no winning move")
    } else if data.bounds.delta == 0 {
        children
            .iter()
            .max_by_key(|e| e.entry.work)
            .expect("lost node, no move")
    } else {
        return;
    };
    data.pv = ch.r#move;
}

pub(in crate::prove) fn dump_table<T: table::Table<Entry>>(
    cfg: &Config,
    table: &T,
) -> io::Result<()> {
    if let Some(ref path) = cfg.dump_table {
        let before = Instant::now();
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(false)
            .truncate(false)
            .write(true)
            .open(path)?;
        table.dump(&mut f)?;

        let elapsed = Instant::now().duration_since(before);
        if cfg.debug > 0 {
            eprintln!(
                "Checkpointed table[{}] in {}.{:03}s",
                path,
                elapsed.as_secs(),
                elapsed.subsec_millis(),
            );
        }
    }
    Ok(())
}

pub(in crate::prove) fn extract_pv<T: table::Table<Entry>>(
    cfg: &Config,
    table: &mut T,
    root: &game::Game,
) -> Vec<game::Move> {
    let mut pv = Vec::new();
    let mut g = root.clone();
    let mut prev = false;
    loop {
        match g.game_state() {
            game::GameState::InPlay => (),
            _ => {
                if cfg.debug > 0 {
                    eprintln!("PV terminated depth={} game={:?}", pv.len(), g.game_state());
                }
                break;
            }
        };
        if let Some(ent) = table.lookup(g.zobrist()) {
            if ent.bounds.phi != 0 && ent.bounds.delta != 0 {
                assert!(
                    pv.len() == 0,
                    "PV contains unproven positions d={} bounds=({}, {}) work={}",
                    pv.len(),
                    ent.bounds.phi,
                    ent.bounds.delta,
                    ent.work,
                );
                break;
            }
            let won = ent.bounds.phi == 0;
            if pv.len() != 0 {
                assert!(
                    won == !prev,
                    "inconsistent game tree d={} prev={} bounds=({}, {})",
                    pv.len(),
                    prev,
                    ent.bounds.phi,
                    ent.bounds.delta
                );
            }
            prev = won;
            if ent.pv.is_none() {
                if cfg.debug > 0 {
                    eprintln!("PV terminated, no move");
                }
                break;
            }
            pv.push(ent.pv);
            g = g.make_move(ent.pv).unwrap_or_else(|_| {
                panic!("PV contained illegal move depth={} m={}", pv.len(), ent.pv)
            });
        } else {
            if cfg.debug > 0 {
                eprintln!("PV terminated depth={} no entry", pv.len());
            }
            break;
        }
    }
    pv
}

pub(in crate::prove) trait ProbeFn: FnMut(&Stats, &game::Game, &Entry, &[Child]) {}
impl<F: FnMut(&Stats, &game::Game, &Entry, &[Child])> ProbeFn for F {}

pub(in crate::prove) struct MID<'a, Table, Probe>
where
    Table: table::Table<Entry>,
    Probe: ProbeFn,
{
    pub(in crate::prove) id: usize,
    pub(in crate::prove) cfg: &'a Config,
    pub(in crate::prove) table: Table,
    pub(in crate::prove) player: game::Player,
    pub(in crate::prove) stack: Vec<game::Move>,
    pub(in crate::prove) probe: Probe,
    pub(in crate::prove) solved_log: Option<SolvedLog>,
    /// With `Init::Net`, what the search keeps of the network's work.
    pub(in crate::prove) guide: Option<Guide>,
    pub(in crate::prove) stats: Stats,
}

impl<'a, Table, Probe> MID<'a, Table, Probe>
where
    Table: table::Table<Entry>,
    Probe: ProbeFn,
{
    pub(in crate::prove) fn mid(
        &mut self,
        bounds: Bounds,
        max_work: u64,
        mut data: Entry,
        pos: &game::Game,
    ) -> (Entry, u64, Vec<Child>) {
        let depth = self.stack.len();
        if self.cfg.debug > 6 {
            eprintln!(
                "{:4$}[{}]mid[{}]: mid={} d={} bounds=({}, {}) max_work={}",
                "",
                self.id,
                self.stack
                    .last()
                    .map(|&m| game::notation::render_move(m))
                    .unwrap_or_else(|| "<root>".to_owned()),
                self.stats.mid,
                depth,
                bounds.phi,
                bounds.delta,
                max_work,
            );
        }
        self.stats.mid += 1;
        if data.work == 0 {
            self.stats.fresh += 1;
        }

        if data.bounds.exceeded(bounds) {
            return (data, 0, Vec::new());
        }

        self.stats
            .open_boards
            .record(pos.open_boards() as u64)
            .unwrap();

        let analysis = if self.cfg.endgame {
            endgame::Analysis::new(pos, &mut self.stats.endgame)
        } else {
            endgame::Analysis::none(pos)
        };
        let kind = Profile::kind(pos, &analysis, self.player);
        self.stats.profile.mids[kind] += 1;
        self.stats.profile.mids_by_open[pos.open_boards() as usize] += 1;

        let terminal = match pos.game_state() {
            game::GameState::InPlay => {
                let proof = analysis.status();
                if !proof.is_winnable(self.player) {
                    self.stats.endgame_solve += 1;
                    Some(pos.player() != self.player)
                } else if proof.is_won(self.player) {
                    self.stats.endgame_solve += 1;
                    Some(pos.player() == self.player)
                } else {
                    None
                }
            }
            game::GameState::Drawn => Some(pos.player() != self.player),
            game::GameState::Won(p) => Some(p == pos.player()),
        };
        if let Some(v) = terminal {
            if v {
                data.bounds = Bounds::winning();
            } else {
                data.bounds = Bounds::losing();
            }
            data.work = 1;
            self.table.store(&data);
            return (data, 1, Vec::new());
        }

        let mut local_work = 1;

        // Moves that a symmetry of the position maps onto each other lead to
        // mirror-image positions of equal value; search one of each.
        let stabilizer = if self.cfg.symmetry {
            symmetry::stabilizer(pos)
        } else {
            0
        };
        let mut children = Vec::new();
        // Whether a move already known to win for the side to move cut move
        // generation short.
        let mut stopped = false;
        for m in pos.all_moves() {
            if stabilizer != 0 && !symmetry::is_representative(m, stabilizer) {
                self.stats.symmetric_moves += 1;
                continue;
            }
            let eval = analysis.evaluate_move(m);
            if eval.is_won(pos.player().other()) && children.len() > 0 {
                self.stats.endgame_move += 1;
                continue;
            }
            let g = pos.make_move(m).expect("all_moves returned illegal move");
            let data = self.ttlookup_or_default(pos, m, &g, eval);
            let bounds = data.bounds;
            children.push(Child {
                position: g,
                r#move: m,
                entry: data,
            });
            if bounds.delta == 0 {
                stopped = true;
                break;
            }
        }
        self.stats.generated_children += children.len();
        self.stats.profile.children[kind] += children.len() as u64;
        self.stats.branch.record(children.len() as u64).unwrap();

        loop {
            data.bounds = compute_bounds(&children);
            (self.probe)(&self.stats, &pos, &data, &children);

            if local_work >= max_work || data.bounds.exceeded(bounds) {
                break;
            }
            let (best_idx, child_bounds) =
                select_child(&children, bounds, &mut data, self.cfg.epsilon);
            let child = &children[best_idx];
            self.stack.push(children[best_idx].r#move);
            let (child_entry, child_work, _) = self.mid(
                child_bounds,
                max_work - local_work,
                children[best_idx].entry.clone(),
                &child.position,
            );
            children[best_idx].entry = child_entry;
            self.stack.pop();
            local_work += child_work;
            data.work += child_work;
        }

        if data.bounds.solved() {
            self.stats.solved += 1;
        }

        populate_pv(&mut data, &children);
        if data.bounds.solved()
            && self
                .solved_log
                .as_ref()
                .map_or(false, |log| log.wants(data.hash))
        {
            let moves: Vec<_> = children
                .iter()
                .map(|c| {
                    let eval = analysis.evaluate_move(c.r#move);
                    (
                        c.r#move,
                        self.start_bounds(pos, c.r#move, &c.position, eval).delta,
                    )
                })
                .collect();
            let won = data.bounds.phi == 0;
            self.solved_log
                .as_mut()
                .unwrap()
                .write(pos, self.player, won, stopped, data.pv, data.work, &moves)
                .expect("writing the --log-solved file");
        }
        let did_store = self.table.store(&data);
        if self.cfg.debug > 6 {
            eprintln!(
                "{:depth$}[{id}]exit mid bounds={bounds:?} local_work={local_work} store={did_store}",
                "",
                id=self.id,
                depth = self.stack.len(),
                bounds = data.bounds,
                local_work = local_work,
                did_store = did_store,
            );
        }
        (data, local_work, children)
    }

    /// The exact bounds of a position when the game is over or the analysis
    /// settles the move into it, with a flag saying whether the analysis did.
    fn settled_bounds(&self, g: &game::Game, eval: prove::Status) -> Option<(Bounds, bool)> {
        match g.game_state() {
            game::GameState::Won(p) => {
                if p == g.player() {
                    Some((Bounds::winning(), false))
                } else {
                    Some((Bounds::losing(), false))
                }
            }
            game::GameState::Drawn => {
                if g.player() == self.player {
                    Some((Bounds::losing(), false))
                } else {
                    Some((Bounds::winning(), false))
                }
            }
            game::GameState::InPlay => {
                if eval.is_won(g.player()) {
                    Some((Bounds::winning(), true))
                } else if eval.is_won(g.player().other()) {
                    Some((Bounds::losing(), true))
                } else {
                    None
                }
            }
        }
    }

    /// The bounds `g`, the position after move `m` from `parent`, starts
    /// with before it is searched: exact when the game is over or the
    /// analysis settles the move into it, otherwise `new_bounds`.
    fn start_bounds(
        &mut self,
        parent: &game::Game,
        m: game::Move,
        g: &game::Game,
        eval: prove::Status,
    ) -> Bounds {
        match self.settled_bounds(g, eval) {
            Some((bounds, _)) => bounds,
            None => self.new_bounds(parent, m, g),
        }
    }

    /// The starting bounds of `g`, the position after move `m` from
    /// `parent`. With `Init::Net`, the network's value of `g` comes from
    /// `parent`'s hidden units, which are updated once for all of its
    /// children from the last position whose children were started: from
    /// the move head (`Config::move_head`), or from `g`'s own units, which
    /// differ from `parent`'s only by the move
    /// (`nnue::Network::success_logit_child`).
    fn new_bounds(&mut self, parent: &game::Game, m: game::Move, g: &game::Game) -> Bounds {
        let (unit, exponent, net) = match (self.cfg.init, &self.cfg.net) {
            (Init::Net(unit, exponent), Some(net)) => (unit, exponent, net),
            _ => return initial_bounds(self.cfg, g, self.player),
        };
        let guide = self
            .guide
            .get_or_insert_with(|| Guide::new(net, unit, exponent));
        if guide.hash != Some(parent.zobrist()) {
            net.update(&mut guide.acc, parent);
            guide.hash = Some(parent.zobrist());
            guide.activated = false;
        }
        let s = if self.cfg.move_head {
            if !guide.activated {
                net.activate(&guide.acc, parent, &mut guide.act);
                guide.activated = true;
            }
            let square = 9 * m.global() + m.local();
            net.move_success_logit(&guide.act, parent, square, self.player)
        } else {
            net.success_logit_child(&guide.acc, parent, g, self.player)
        };
        guide.bounds[net_index(s)]
    }

    /// The table entry of `g`, a move from `parent`, or a new one with its
    /// starting bounds. Settled positions are not looked up, and the starting
    /// bounds of the others are computed only when the table has no entry.
    pub(in crate::prove) fn ttlookup_or_default(
        &mut self,
        parent: &game::Game,
        m: game::Move,
        g: &game::Game,
        eval: prove::Status,
    ) -> Entry {
        let new = |bounds| Entry {
            bounds: bounds,
            hash: g.zobrist(),
            work: 0,
            ..Default::default()
        };
        if let Some((bounds, analysed)) = self.settled_bounds(g, eval) {
            if analysed {
                self.stats.endgame_move += 1;
            }
            return new(bounds);
        }
        match self.table.lookup(g.zobrist()) {
            Some(entry) => entry,
            None => new(self.new_bounds(parent, m, g)),
        }
    }

    /// Walks the proof (or disproof) that the search left in the table and
    /// checks each step of it against the rules. Returns how many of its
    /// positions a search has to expand, and how many it needs that the table
    /// no longer holds as solved. Moves are generated as in `mid`, with the
    /// same exact analysis, so a walk with nothing missing has checked a
    /// complete proof of the result. Where the side to move wins, the walk
    /// follows the winning move with the least recorded work; where it loses,
    /// every move. A perfectly guided search would expand little more than a
    /// proof, so this count against the search's work shows how much better
    /// guidance could save.
    pub(in crate::prove) fn proof_size(
        &mut self,
        root: &game::Game,
        root_won: bool,
    ) -> (usize, usize) {
        let mut seen = HashSet::new();
        let mut stack = vec![(root.clone(), root_won)];
        let (mut size, mut missing) = (0, 0);
        while let Some((pos, won)) = stack.pop() {
            if !seen.insert(pos.zobrist()) {
                continue;
            }
            let analysis = if self.cfg.endgame {
                endgame::Analysis::new(&pos, &mut self.stats.endgame)
            } else {
                endgame::Analysis::none(&pos)
            };
            // Positions that `mid` settles on entry, without generating moves.
            let terminal = match pos.game_state() {
                game::GameState::InPlay => {
                    let proof = analysis.status();
                    if !proof.is_winnable(self.player) {
                        Some(pos.player() != self.player)
                    } else if proof.is_won(self.player) {
                        Some(pos.player() == self.player)
                    } else {
                        None
                    }
                }
                game::GameState::Drawn => Some(pos.player() != self.player),
                game::GameState::Won(p) => Some(p == pos.player()),
            };
            if let Some(v) = terminal {
                assert_eq!(
                    v,
                    won,
                    "proof contradicts the analysis at {}",
                    game::notation::render(&pos)
                );
                size += 1;
                continue;
            }
            match self.table.lookup(pos.zobrist()) {
                Some(e) if e.bounds.solved() => assert_eq!(
                    e.bounds.phi == 0,
                    won,
                    "proof contradicts the table at {}",
                    game::notation::render(&pos)
                ),
                _ => {
                    missing += 1;
                    continue;
                }
            }
            size += 1;

            let stabilizer = if self.cfg.symmetry {
                symmetry::stabilizer(&pos)
            } else {
                0
            };
            let mut children = Vec::new();
            for m in pos.all_moves() {
                if stabilizer != 0 && !symmetry::is_representative(m, stabilizer) {
                    continue;
                }
                let eval = analysis.evaluate_move(m);
                if eval.is_won(pos.player().other()) && children.len() > 0 {
                    continue;
                }
                let g = pos.make_move(m).expect("all_moves returned illegal move");
                // Settled by the rules or the analysis, without a search.
                let settled = match g.game_state() {
                    game::GameState::InPlay => {
                        eval.is_won(g.player()) || eval.is_won(g.player().other())
                    }
                    _ => true,
                };
                let entry = self.ttlookup_or_default(&pos, m, &g, eval);
                children.push((g, entry, settled));
            }
            if won {
                // One move that leaves the opponent lost.
                let best = children
                    .into_iter()
                    .filter(|(_, e, _)| e.bounds.delta == 0)
                    .min_by_key(|(_, e, _)| e.work);
                match best {
                    Some((g, _, false)) => stack.push((g, false)),
                    Some(_) => (),
                    None => missing += 1,
                }
            } else {
                // Every move must leave the opponent winning.
                for (g, e, settled) in children {
                    if !settled {
                        stack.push((g, true));
                    } else {
                        assert!(
                            e.bounds.phi == 0,
                            "proof contradicts the analysis at {}",
                            game::notation::render(&g)
                        );
                    }
                }
            }
        }
        (size, missing)
    }
}
