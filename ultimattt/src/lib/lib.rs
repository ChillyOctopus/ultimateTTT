// Modified by the ultimateTTT project (2026): adds the `nnue`,
// `oracle`, `symmetry` and `tactics` modules, and `clock` and `simd`, so that
// stable Rust builds it, for WebAssembly too.

pub mod clock;
pub mod endgame;
pub mod game;
pub mod minimax;
pub mod nnue;
pub mod oracle;
pub mod progress;
pub mod protocol;
pub mod simd;
pub mod prove;
pub mod symmetry;
pub mod table;
pub mod tactics;
pub mod util;
