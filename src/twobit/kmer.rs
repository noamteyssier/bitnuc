//! The `u64` kmer boundary.
//!
//! These are the only functions in the crate that reinterpret encoded bytes
//! as `u64` values. The conversion is pinned to little-endian
//! (`u64::{from,to}_le_bytes`), so packed values are bit-identical on every
//! platform and to the values produced by bitnuc 0.4.x. On little-endian
//! targets the conversion compiles to a plain load/store.

use super::{decode, encode};
use crate::BitnucError;

/// Packs a sequence of up to 32 bases into a `u64`, 2 bits per base.
///
/// Base `i` occupies bits `2i..2i + 2` (`A=00`, `C=01`, `G=10`, `T=11`);
/// unused high bits are zero. Bases outside `ACGTacgt` map to an unspecified
/// code — see the [module docs](crate::twobit#ambiguous-bases).
///
/// # Errors
///
/// Returns [`BitnucError::SequenceTooLong`] if `seq` is longer than 32 bases.
///
/// # Examples
///
/// ```rust
/// let packed = bitnuc::as_2bit(b"ACGT")?;
/// assert_eq!(packed, 0b11100100);
/// # Ok::<(), bitnuc::BitnucError>(())
/// ```
pub fn as_2bit(seq: &[u8]) -> Result<u64, BitnucError> {
    if seq.len() > 32 {
        return Err(BitnucError::SequenceTooLong(seq.len()));
    }
    let mut buf = [0u8; 8];
    encode(seq, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Deprecated alias for [`as_2bit`].
///
/// Base validation was removed from the encoder, so the lossy and non-lossy
/// variants are now the same function.
#[deprecated(since = "0.5.0", note = "encoding is always lossy now; use `as_2bit`")]
pub fn as_2bit_lossy(seq: &[u8]) -> Result<u64, BitnucError> {
    as_2bit(seq)
}

/// Unpacks `n` bases (at most 32) from a 2-bit packed `u64`, appending the
/// ASCII bases to `seq`.
///
/// The buffer is appended to — not overwritten — matching the 0.4.x
/// semantics, so a reusable buffer must be cleared between sequences.
///
/// # Errors
///
/// Returns [`BitnucError::InvalidLength`] if `n` is greater than 32.
///
/// # Examples
///
/// ```rust
/// let packed = bitnuc::as_2bit(b"ACGT")?;
/// let mut seq = Vec::new();
/// bitnuc::from_2bit(packed, 4, &mut seq)?;
/// assert_eq!(&seq, b"ACGT");
/// # Ok::<(), bitnuc::BitnucError>(())
/// ```
pub fn from_2bit(packed: u64, n: usize, seq: &mut Vec<u8>) -> Result<(), BitnucError> {
    if n > 32 {
        return Err(BitnucError::InvalidLength(n));
    }
    let bytes = packed.to_le_bytes();
    let start = seq.len();
    seq.resize(start + n, 0);
    decode(&bytes, n, &mut seq[start..])
}

/// Unpacks `n` bases (at most 32) from a 2-bit packed `u64` into a new `Vec`.
///
/// # Errors
///
/// Returns [`BitnucError::InvalidLength`] if `n` is greater than 32.
pub fn from_2bit_alloc(packed: u64, n: usize) -> Result<Vec<u8>, BitnucError> {
    let mut seq = Vec::with_capacity(n);
    from_2bit(packed, n, &mut seq)?;
    Ok(seq)
}
