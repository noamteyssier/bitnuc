use crate::NucleotideError;

// Create masks for lower and upper bits of each 2-bit group
const LOWER_BITS: u64 = 0x5555555555555555;
const UPPER_BITS: u64 = 0xAAAAAAAAAAAAAAAA;

/// Calculate hamming distance between two 2-bit encoded u64 values
/// Each u64 can contain up to 32 bases (2 bits per base)
/// len must be <= 32
#[inline]
pub fn hdist_scalar(u: u64, v: u64, len: usize) -> Result<u32, NucleotideError> {
    // Validate length
    if len > 32 {
        return Err(NucleotideError::InvalidLength(len));
    }

    // For empty sequences, distance is 0
    if len == 0 || u == v {
        return Ok(0);
    }

    // Calculate number of valid bits (2 bits per base)
    let valid_bits = len * 2;

    // Create mask for valid bits
    let mask = if valid_bits == 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };

    // XOR to find differences and mask to valid region
    let diff = (u ^ v) & mask;

    if diff == 0 {
        return Ok(0);
    }

    // Get differences in lower and upper bits, masked to valid region
    let lower_diffs = diff & LOWER_BITS & mask;
    let upper_diffs = (diff & UPPER_BITS & mask) >> 1;

    // Combine differences - if either or both bits differ, count as one difference
    let combined_diffs = lower_diffs | upper_diffs;

    // Count number of 1 bits in result
    Ok(combined_diffs.count_ones())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdist_scalar_validation() {
        // Test length validation
        assert!(hdist_scalar(0, 0, 33).is_err()); // Too long
        assert!(hdist_scalar(0, 0, 0).is_ok()); // Empty sequences
        assert!(hdist_scalar(0, 0, 32).is_ok()); // Max length
    }

    #[test]
    fn test_hdist_scalar_identical() {
        // Test identical sequences
        assert_eq!(hdist_scalar(0, 0, 1), Ok(0)); // Single base
        assert_eq!(hdist_scalar(0xFFFFFFFF, 0xFFFFFFFF, 16), Ok(0)); // 16 bases
        assert_eq!(
            hdist_scalar(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 32),
            Ok(0)
        );
        // 32 bases
    }

    #[test]
    fn test_hdist_scalar_small_sequences() {
        // Test sequence: A=00, C=01, G=10, T=11

        // Convert "AC" (0001) to u64
        let ac = 0b0001u64;
        // Convert "AG" (0010) to u64
        let ag = 0b0010u64;
        // Convert "AT" (0011) to u64
        let at = 0b0011u64;

        assert_eq!(hdist_scalar(ac, ag, 2), Ok(1)); // AC vs AG = 1 difference
        assert_eq!(hdist_scalar(ac, at, 2), Ok(1)); // AC vs AT = 1 difference
        assert_eq!(hdist_scalar(ag, at, 2), Ok(1)); // AG vs AT = 1 difference
    }

    #[test]
    fn test_hdist_scalar_full_sequences() {
        // Test cases with known distances
        let test_cases: Vec<(&[u8], &[u8], u32)> = vec![
            (b"AAAA", b"AAAA", 0),
            (b"AAAA", b"AAAT", 1),
            (b"AAAA", b"AATT", 2),
            (b"AAAA", b"ATTT", 3),
            (b"AAAA", b"TTTT", 4),
            (b"ACTGACTG", b"TGCATGCA", 8),
        ];

        for (seq1, seq2, expected) in test_cases {
            let u = crate::as_2bit(seq1).unwrap();
            let v = crate::as_2bit(seq2).unwrap();
            let len = seq1.len();

            assert_eq!(
                hdist_scalar(u, v, len),
                Ok(expected),
                "Failed for sequences {:?} and {:?}",
                std::str::from_utf8(seq1).unwrap(),
                std::str::from_utf8(seq2).unwrap()
            );
        }
    }
}
