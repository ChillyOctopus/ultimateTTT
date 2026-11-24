// Added by the ultimateTTT project (2026): stands in for the parts of
// `std::simd` the engine used, which needs a nightly compiler. Plain arrays
// with lane-by-lane operations, which the compiler vectorizes itself, so
// stable Rust builds the engine for native code and for WebAssembly alike.

#![allow(non_camel_case_types)]

use std::ops::{Add, AddAssign, BitAnd, Mul, Sub, SubAssign};

/// The lane-by-lane comparison of two vectors.
pub struct Mask<const N: usize>([bool; N]);

impl<const N: usize> Mask<N> {
    #[inline(always)]
    pub fn any(&self) -> bool {
        self.0.iter().any(|&b| b)
    }
}

macro_rules! vector {
    ($name:ident, $t:ty, $n:expr) => {
        #[derive(Copy, Clone, Debug, PartialEq)]
        #[repr(C, align(64))]
        pub struct $name([$t; $n]);

        impl $name {
            #[inline(always)]
            pub const fn from_array(a: [$t; $n]) -> Self {
                $name(a)
            }
            #[inline(always)]
            pub const fn splat(x: $t) -> Self {
                $name([x; $n])
            }
            #[inline(always)]
            pub fn to_array(self) -> [$t; $n] {
                self.0
            }
            #[inline(always)]
            pub fn lanes_eq(self, other: Self) -> Mask<$n> {
                let mut m = [false; $n];
                for i in 0..$n {
                    m[i] = self.0[i] == other.0[i];
                }
                Mask(m)
            }
            #[inline(always)]
            fn zip(self, other: Self, f: impl Fn($t, $t) -> $t) -> Self {
                let mut out = self.0;
                for i in 0..$n {
                    out[i] = f(self.0[i], other.0[i]);
                }
                $name(out)
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                self.zip(o, |a, b| a + b)
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                self.zip(o, |a, b| a - b)
            }
        }
        impl Mul for $name {
            type Output = Self;
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                self.zip(o, |a, b| a * b)
            }
        }
        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }
        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) {
                *self = *self - o;
            }
        }
    };
}

vector!(u16x8, u16, 8);
vector!(i32x16, i32, 16);
vector!(f32x16, f32, 16);

impl BitAnd for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, o: Self) -> Self {
        self.zip(o, |a, b| a & b)
    }
}

impl f32x16 {
    #[inline(always)]
    pub fn round_from_int(v: i32x16) -> Self {
        f32x16(v.0.map(|x| x as f32))
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        self.zip(o, f32::max)
    }
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        self.zip(o, f32::min)
    }
}
