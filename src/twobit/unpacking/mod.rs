#[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
mod aarch64;
#[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
mod avx;
mod naive;

use crate::NucleotideError;

/// Converts an arbitrary sized 2-bit packed representation back into a nucleotide sequence.
pub fn from_2bit_multi(
    ebuf: &[u64],
    n_bases: usize,
    dbuf: &mut Vec<u8>,
) -> Result<(), NucleotideError> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return aarch64::fast_decode(ebuf, n_bases, dbuf);
    } else {
        // Fall back to naive implemention if SIMD feature is not enabled
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx2") {
        return unsafe { avx::from_2bit_multi_simd(ebuf, n_bases, dbuf) };
    } else {
        // Fall back to naive implemention if SIMD feature is not enabled
    }

    // Calculate the number of chunks and the remainder
    let n_chunks = n_bases.div_ceil(32);
    let rem = match n_bases % 32 {
        0 => 32, // Full chunk
        rem => rem,
    };

    // Process all chunks except the last one
    ebuf.iter()
        .take(n_chunks - 1)
        .try_for_each(|component| from_2bit(*component, 32, dbuf))?;

    // Process the last one with the remainder
    ebuf.get(n_chunks - 1)
        .map_or(Err(NucleotideError::InvalidLength(n_bases)), |&component| {
            from_2bit(component, rem, dbuf)
        })?;

    Ok(())
}

/// Converts a 2-bit packed representation back into a nucleotide sequence.
///
/// This function reverses the packing performed by `as_2bit`.
///
/// # Arguments
///
/// * `packed` - A u64 containing the 2-bit packed sequence
/// * `expected_size` - The number of bases to unpack
///
/// # Returns
///
/// Returns a `Vec<u8>` containing the ASCII sequence.
///
/// # Errors
///
/// Returns `NucleotideError::InvalidLength` if `expected_size` is greater than 32
/// (as a u64 can only store 32 * 2 bits).
///
/// # Examples
///
/// Basic unpacking:
/// ```rust
/// use bitnuc::{as_2bit, from_2bit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Pack and unpack
/// let packed = as_2bit(b"ACGT")?;
/// let mut unpacked = Vec::new();
/// from_2bit(packed, 4, &mut unpacked)?;
/// assert_eq!(&unpacked, b"ACGT");
/// unpacked.clear();
///
/// // Partial unpacking
/// from_2bit(packed, 2, &mut unpacked)?;
/// assert_eq!(&unpacked, b"AC");
/// # Ok(())
/// # }
/// ```
///
/// Error handling:
/// ```rust
/// use bitnuc::{from_2bit, NucleotideError};
///
/// # fn main() {
/// // Length too long
/// assert!(matches!(
///     from_2bit(0, 33, &mut Vec::new()),
///     Err(NucleotideError::InvalidLength(33))
/// ));
/// # }
/// ```
///
/// # Implementation Details
///
/// The bases are unpacked from least significant to most significant bits:
/// ```rust
/// use bitnuc::{as_2bit, from_2bit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let packed = 0b11100100; // "ACGT" in 2-bit encoding
/// let mut seq = Vec::new();
/// from_2bit(packed, 4, &mut seq)?;
/// assert_eq!(seq[0], b'A'); // 00
/// assert_eq!(seq[1], b'C'); // 01
/// assert_eq!(seq[2], b'G'); // 10
/// assert_eq!(seq[3], b'T'); // 11
/// # Ok(())
/// # }
/// ```
pub fn from_2bit(
    packed: u64,
    expected_size: usize,
    sequence: &mut Vec<u8>,
) -> Result<(), NucleotideError> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { aarch64::from_2bit_simd(packed, expected_size, sequence) }
    } else {
        naive::from_2bit(packed, expected_size, sequence)
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx2") {
        unsafe { avx::from_2bit_simd(packed, expected_size, sequence) }
    } else {
        naive::from_2bit(packed, expected_size, sequence)
    }

    // Fall back to naive implemention if:
    // - SIMD is disabled via nosimd feature
    // - or SIMD feature is not enabled
    // - or required CPU features aren't availabe
    #[cfg(any(
        feature = "nosimd",
        all(not(target_arch = "aarch64"), not(target_arch = "x86_64"),)
    ))]
    naive::from_2bit(packed, expected_size, sequence)
}

/// This calls from_2bit but allocates a new Vec to store the result.
///
/// # Arguments
///
/// * `packed` - A u64 containing the 2-bit packed sequence
/// * `expected_size` - The number of bases to unpack
///
/// # Returns
///
/// Returns a `Vec<u8>` containing the ASCII sequence.
///
/// # Errors
///
/// Returns `NucleotideError::InvalidLength` if `expected_size` is greater than 32
///
/// # Examples
///
/// Basic unpacking:
///
/// ```rust
/// use bitnuc::from_2bit_alloc;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let packed = 0b11100100; // "ACGT" in 2-bit encoding
/// let seq = from_2bit_alloc(packed, 4)?;
/// assert_eq!(&seq, b"ACGT");
/// # Ok(())
/// # }
/// ```
pub fn from_2bit_alloc(packed: u64, expected_size: usize) -> Result<Vec<u8>, NucleotideError> {
    let mut sequence = Vec::with_capacity(expected_size);
    from_2bit(packed, expected_size, &mut sequence)?;
    Ok(sequence)
}

#[cfg(test)]
mod testing {
    use super::*;

    #[test]
    fn test_from_2bit_valid_sequence() {
        let tests = vec![
            (0b11100100, 4, b"ACGT"),
            (0b00000000, 4, b"AAAA"),
            (0b11111111, 4, b"TTTT"),
        ];

        let mut unpacked = Vec::new();
        for (input, size, expected) in tests {
            from_2bit(input, size, &mut unpacked).unwrap();
            assert_eq!(unpacked, expected);
            unpacked.clear();
        }
    }

    #[test]
    fn test_example_case_from_2bit() {
        let packed = 71620941647064936;
        let slen = 28;
        let expected = b"AGGCTTGAGGCCCATTCTCTGATCGTTT";
        let observed = from_2bit_alloc(packed, slen).unwrap();

        let obs_str = std::str::from_utf8(&observed).unwrap();
        let exp_str = std::str::from_utf8(expected).unwrap();
        assert_eq!(obs_str, exp_str);
        assert_eq!(&observed, expected);
    }
}
