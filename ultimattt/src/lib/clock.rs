// Added by the ultimateTTT project (2026): the time limits' clock.
// `std::time::Instant` natively. In WebAssembly, where `Instant::now()`
// panics, the page passes in its own clock (`performance.now()`) as the
// import `env.now_ms`.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use self::wasm::Instant;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::ops::{Add, Sub};
    use std::time::Duration;

    #[link(wasm_import_module = "env")]
    extern "C" {
        fn now_ms() -> f64;
    }

    /// Milliseconds since the page loaded.
    #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Instant {
            Instant(unsafe { now_ms() })
        }
        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64((self.0 - earlier.0).max(0.0) / 1000.0)
        }
        pub fn elapsed(&self) -> Duration {
            Instant::now().duration_since(*self)
        }
    }

    impl Add<Duration> for Instant {
        type Output = Instant;
        fn add(self, d: Duration) -> Instant {
            Instant(self.0 + d.as_secs_f64() * 1000.0)
        }
    }

    impl Sub for Instant {
        type Output = Duration;
        fn sub(self, earlier: Instant) -> Duration {
            self.duration_since(earlier)
        }
    }
}
