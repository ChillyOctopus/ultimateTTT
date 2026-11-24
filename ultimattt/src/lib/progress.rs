// Modified by the ultimateTTT project (2026): uses `clock::Instant`, which
// also works in WebAssembly.

use crate::clock::Instant;
use std::time::Duration;

pub struct Ticker {
    interval: Duration,
    tick: Instant,
}

impl Ticker {
    pub fn new(interval: Duration) -> Self {
        Ticker {
            interval,
            tick: Instant::now() + interval,
        }
    }

    pub fn tick(&mut self) -> bool {
        let now = Instant::now();
        if now > self.tick {
            self.tick = now + self.interval;
            return true;
        }
        return false;
    }
}

pub struct Counter<const N: usize> {
    count: usize,
}

impl<const N: usize> Counter<N> {
    pub fn new() -> Self {
        Counter { count: N }
    }

    pub fn tick(&mut self) -> bool {
        self.count -= 1;
        if self.count == 0 {
            self.count = N;
            return true;
        }
        return false;
    }
}
