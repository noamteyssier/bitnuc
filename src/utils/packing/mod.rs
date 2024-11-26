use crate::NucleotideError;

#[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
mod aarch64;
#[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
mod avx;
mod naive;

/// Converts a nucleotide sequence into a 2-bit packed representation.
///
/// Each nucleotide is encoded using 2 bits:
/// - A/a = 00
/// - C/c = 01
/// - G/g = 10
/// - T/t = 11
///
/// The bases are packed from least significant to most significant bits.
/// For example, "ACGT" becomes 0b11100100.
///
/// # Arguments
///
/// * `seq` - A byte slice containing ASCII nucleotides (A,C,G,T, case insensitive)
///
/// # Returns
///
/// Returns a `u64` containing the packed representation.
///
/// # Errors
///
/// Returns `NucleotideError::InvalidBase` if the sequence contains any characters
/// other than A,C,G,T (case insensitive).
///
/// Returns `NucleotideError::SequenceTooLong` if the input sequence is longer
/// than 32 bases (as a u64 can only store 32 * 2 bits).
///
/// # Examples
///
/// Basic packing:
/// ```rust
/// use bitnuc::as_2bit;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let packed = as_2bit(b"ACGT")?;
/// assert_eq!(packed, 0b11100100);
/// # Ok(())
/// # }
/// ```
///
/// Case insensitivity:
/// ```rust
/// use bitnuc::as_2bit;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// assert_eq!(as_2bit(b"acgt")?, as_2bit(b"ACGT")?);
/// # Ok(())
/// # }
/// ```
///
/// Error handling:
/// ```rust
/// use bitnuc::{as_2bit, NucleotideError};
///
/// # fn main() {
/// // Invalid base
/// assert!(matches!(
///     as_2bit(b"ACGN"),
///     Err(NucleotideError::InvalidBase(b'N'))
/// ));
///
/// // Sequence too long
/// let long_seq = vec![b'A'; 33];
/// assert!(matches!(
///     as_2bit(&long_seq),
///     Err(NucleotideError::SequenceTooLong(33))
/// ));
/// # }
/// ```
#[inline(always)]
pub fn as_2bit(seq: &[u8]) -> Result<u64, NucleotideError> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if is_aarch64_feature_detected!("neon") {
        return aarch64::as_2bit(seq);
    } else {
        return naive::as_2bit(seq);
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx") {
        return avx::as_2bit(seq);
    } else {
        return naive::as_2bit(seq);
    }

    // #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64",)))]
    #[cfg(any(
        feature = "nosimd",
        all(not(target_arch = "aarch64"), not(target_arch = "x86_64"))
    ))]
    return naive::as_2bit(seq);
}

#[cfg(test)]
mod testing {
    use super::*;

    #[test]
    fn test_as_2bit_valid_sequence() {
        let tests = vec![
            (b"ACGT", 0b11100100),
            (b"AAAA", 0b00000000),
            (b"TTTT", 0b11111111),
            (b"GGGG", 0b10101010),
            (b"CCCC", 0b01010101),
        ];

        for (input, expected) in tests {
            assert_eq!(as_2bit(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_as_2bit_alignments() {
        let tests = vec![(b"ACTGGAAAATTTTAAGG", 0b1010000011111111000000001010110100)];
        for (input, expected) in tests {
            assert_eq!(as_2bit(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_as_2bit_lowercase() {
        assert_eq!(as_2bit(b"acgt").unwrap(), as_2bit(b"ACGT").unwrap());
    }

    #[test]
    fn test_as_2bit_invalid_base() {
        let result = as_2bit(b"ACGN");
        assert!(matches!(result, Err(NucleotideError::InvalidBase(b'N'))));
    }

    #[test]
    fn test_as_2bit_sequence_too_long() {
        let long_seq = vec![b'A'; 33];
        assert!(matches!(
            as_2bit(&long_seq),
            Err(NucleotideError::SequenceTooLong(33))
        ));
    }
}
