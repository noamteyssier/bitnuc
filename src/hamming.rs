use crate::BitnucError;

// Masks for the lower and upper bit of each 2-bit group
const LOWER_BITS: u64 = 0x5555555555555555;
const UPPER_BITS: u64 = 0xAAAAAAAAAAAAAAAA;

/// Calculates the hamming distance between two 2-bit packed `u64` kmers
/// (as produced by [`as_2bit`](super::as_2bit)) of length `len` bases.
///
/// # Errors
///
/// Returns [`BitnucError::InvalidLength`] if `len` is greater than 32.
#[inline]
pub fn hdist_scalar(u: u64, v: u64, len: usize) -> Result<u32, BitnucError> {
    if len > 32 {
        return Err(BitnucError::InvalidLength(len));
    }

    if len == 0 || u == v {
        return Ok(0);
    }

    // Mask to the valid region (2 bits per base)
    let valid_bits = len * 2;
    let mask = if valid_bits == 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };

    let diff = (u ^ v) & mask;

    // A base differs if either of its two bits differs
    let lower_diffs = diff & LOWER_BITS;
    let upper_diffs = (diff & UPPER_BITS) >> 1;
    let combined_diffs = lower_diffs | upper_diffs;

    Ok(combined_diffs.count_ones())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::as_2bit;

    #[test]
    fn test_hdist_scalar_validation() {
        assert!(hdist_scalar(0, 0, 33).is_err()); // Too long
        assert!(hdist_scalar(0, 0, 0).is_ok()); // Empty sequences
        assert!(hdist_scalar(0, 0, 32).is_ok()); // Max length
    }

    #[test]
    fn test_hdist_scalar_identical() {
        assert_eq!(hdist_scalar(0, 0, 1).unwrap(), 0);
        assert_eq!(hdist_scalar(0xFFFFFFFF, 0xFFFFFFFF, 16).unwrap(), 0);
        assert_eq!(
            hdist_scalar(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 32).unwrap(),
            0
        );
    }

    #[test]
    fn test_hdist_scalar_masks_beyond_len() {
        // Differences beyond `len` bases must not count
        let u = 0b0000u64;
        let v = 0b1100u64; // differs only at base 1
        assert_eq!(hdist_scalar(u, v, 1).unwrap(), 0);
        assert_eq!(hdist_scalar(u, v, 2).unwrap(), 1);
    }

    #[test]
    fn test_hdist_scalar_full_sequences() {
        let test_cases: Vec<(&[u8], &[u8], u32)> = vec![
            (b"AAAA", b"AAAA", 0),
            (b"AAAA", b"AAAT", 1),
            (b"AAAA", b"AATT", 2),
            (b"AAAA", b"ATTT", 3),
            (b"AAAA", b"TTTT", 4),
            (b"ACTGACTG", b"TGCATGCA", 8),
        ];

        for (seq1, seq2, expected) in test_cases {
            let u = as_2bit(seq1).unwrap();
            let v = as_2bit(seq2).unwrap();
            assert_eq!(
                hdist_scalar(u, v, seq1.len()).unwrap(),
                expected,
                "Failed for sequences {:?} and {:?}",
                std::str::from_utf8(seq1).unwrap(),
                std::str::from_utf8(seq2).unwrap()
            );
        }
    }
}
