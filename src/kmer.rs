//! The `u64` kmer boundary.
//!
//! These are the only functions in the crate that reinterpret encoded bytes
//! as `u64` values. The conversion is pinned to little-endian
//! (`u64::{from,to}_le_bytes`), so packed values are bit-identical on every
//! platform and to the values produced by bitnuc 0.4.x. On little-endian
//! targets the conversion compiles to a plain load/store.

use crate::{BitnucError, decode::decode, encode::encode};

/// Packs a sequence of up to 32 bases into a `u64`, 2 bits per base.
///
/// Base `i` occupies bits `2i..2i + 2` (`A=00`, `C=01`, `G=10`, `T=11`);
/// unused high bits are zero. Bases outside `ACGTacgt` map to an unspecified
/// code — see the [crate docs](crate#encoded-format).
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

/// Unpacks a 2-bit packed `u64` into a stack array of 32 ASCII bases.
///
/// All 32 positions decode; the caller slices to their sequence length.
/// Positions past the packed sequence decode the kmer's unused high bits,
/// i.e. `b'A'` for values produced by [`as_2bit`].
///
/// # Examples
///
/// ```rust
/// let packed = bitnuc::as_2bit(b"ACGT")?;
/// let unpacked = bitnuc::from_2bit(packed);
/// assert_eq!(&unpacked[..4], b"ACGT");
/// # Ok::<(), bitnuc::BitnucError>(())
/// ```
pub fn from_2bit(packed: u64) -> [u8; 32] {
    // always decode all 32 bases in SIMD with a stack buffer
    let mut dbuf = [0u8; 32];
    decode(&packed.to_le_bytes(), 32, &mut dbuf)
        .expect("8 encoded bytes always decode into a 32-byte buffer");
    dbuf
}
