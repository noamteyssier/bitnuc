//! 2-bit nucleotide encoding.
//!
//! # Format
//!
//! The encoded representation is a byte sequence: byte `k` holds bases
//! `4k..4k+4`, with base `j` occupying bits `2*(j % 4)..2*(j % 4) + 2` of its
//! byte (`A=00`, `C=01`, `G=10`, `T=11`). A sequence of `n` bases occupies
//! exactly `n.div_ceil(4)` bytes, and unused bits of a trailing partial byte
//! are always zero.
//!
//! This definition is endianness-free.
//!
//! Endianness is introduced when reinterpreting to a `u64` kmer.
//! This is used by [`as_2bit`] and [`from_2bit`] which are pinned to
//! little-endian internally.
//!
//! # Ambiguous bases
//!
//! Encoding makes no guarantees for bytes outside `ACGTacgt`: invalid bases
//! map to an unspecified (position-dependent) code rather than an error.
//! Detect ambiguous bases separately if you need to handle them.

mod decode;
mod encode;
mod hamming;
mod kmer;

pub use decode::{decode, decode_resize};
pub use encode::{encode, encode_resize};
pub use hamming::hdist_scalar;
#[allow(deprecated)]
pub use kmer::as_2bit_lossy;
pub use kmer::{as_2bit, from_2bit, from_2bit_alloc};

use fearless_simd::Level;

/// Selects the SIMD level for encode/decode dispatch.
///
/// With the `nosimd` feature, forces the scalar fallback level instead of
/// runtime feature detection.
#[inline]
pub(crate) fn level() -> Level {
    #[cfg(feature = "nosimd")]
    {
        Level::fallback()
    }
    #[cfg(not(feature = "nosimd"))]
    {
        Level::new()
    }
}
