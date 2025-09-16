use crate::Error;

/// Splits a packed nucleotide sequence into two subsequences at the given index.
///
/// # Arguments
/// * `ebuf` - The encoded sequence buffer
/// * `slen` - The total length of the sequence in bases
/// * `idx` - The position to split at (included in right sequence)
/// * `lbuf` - Buffer to store the left subsequence
/// * `rbuf` - Buffer to store the right subsequence
///
/// # Errors
/// Returns an error if the index is out of bounds
pub fn split_packed(
    ebuf: &[u64],
    slen: usize,
    idx: usize,
    lbuf: &mut Vec<u64>,
    rbuf: &mut Vec<u64>,
) -> Result<(), Error> {
    // Validate inputs
    if idx > slen {
        return Err(Error::IndexOutOfBounds {
            index: idx,
            length: slen,
        });
    }

    // Clear output buffers
    lbuf.clear();
    rbuf.clear();

    // Handle edge cases first
    if idx == 0 {
        // Left sequence is empty, right gets everything
        rbuf.extend_from_slice(ebuf);
        return Ok(());
    }
    if idx == slen {
        // Right sequence is empty, left gets everything
        lbuf.extend_from_slice(ebuf);
        return Ok(());
    }

    // Handle empty input case
    if ebuf.is_empty() {
        return Ok(());
    }

    // Calculate required buffer sizes for the general case
    let left_chunks = (idx / 32) + 1; // Include chunk containing split
    let right_chunks = if idx == slen {
        0
    } else {
        (slen - idx).div_ceil(32)
    };

    // Reserve space in output buffers
    lbuf.reserve(left_chunks);
    rbuf.reserve(right_chunks);

    // Handle the general case
    let chunk_idx = idx / 32; // Which u64 contains the split point
    let bit_idx = (idx % 32) * 2; // Which bit position within that u64

    // Copy full chunks to left buffer
    if chunk_idx > 0 && chunk_idx <= ebuf.len() {
        lbuf.extend_from_slice(&ebuf[..chunk_idx]);
    }

    // Handle the split chunk
    let split_mask = if bit_idx == 0 {
        0
    } else {
        (1u64 << bit_idx) - 1
    };
    lbuf.push(ebuf[chunk_idx] & split_mask);

    // Handle remaining bits for right buffer
    let right_shift = bit_idx;
    let mut carry = 0u64;

    for curr in ebuf.iter().skip(chunk_idx) {
        // Combine previous carry with current shifted value
        let shifted = carry | (curr >> right_shift);
        rbuf.push(shifted);

        // Save bits that will be needed for next iteration
        carry = if right_shift == 0 {
            0
        } else {
            curr << (64 - right_shift)
        };
    }

    // Handle the final carry if needed
    if carry != 0 && rbuf.len() < right_chunks {
        rbuf.push(carry);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twobit::{decode, encode};

    #[test]
    fn test_split_basic() {
        // Test sequence: "ACTGACTG"
        let seq = b"ACTGACTG";
        let mut ebuf = Vec::new();
        encode(seq, &mut ebuf).unwrap();

        let mut lbuf = Vec::new();
        let mut rbuf = Vec::new();

        // Split after 4 bases
        split_packed(&ebuf, seq.len(), 4, &mut lbuf, &mut rbuf).unwrap();

        assert_eq!(lbuf.len(), 1);
        assert_eq!(rbuf.len(), 1);

        // Verify left side
        let mut left = Vec::new();
        decode(&lbuf, 4, &mut left).unwrap();
        assert_eq!(&left, b"ACTG");

        // Verify right side
        let mut right = Vec::new();
        decode(&rbuf, 4, &mut right).unwrap();
        assert_eq!(&right, b"ACTG");
    }

    #[test]
    fn test_split_edge_cases() {
        let seq = b"ACTG";
        let mut ebuf = Vec::new();
        encode(seq, &mut ebuf).unwrap();

        let mut lbuf = Vec::new();
        let mut rbuf = Vec::new();

        // Split at start
        split_packed(&ebuf, seq.len(), 0, &mut lbuf, &mut rbuf).unwrap();
        let mut decoded = Vec::new();
        decode(&rbuf, seq.len(), &mut decoded).unwrap();
        assert_eq!(&decoded, seq);

        assert_eq!(lbuf.len(), 0);
        assert_eq!(rbuf.len(), 1);

        // Split at end
        split_packed(&ebuf, seq.len(), seq.len(), &mut lbuf, &mut rbuf).unwrap();
        let mut decoded = Vec::new();
        decode(&lbuf, seq.len(), &mut decoded).unwrap();
        assert_eq!(&decoded, seq);

        assert_eq!(lbuf.len(), 1);
        assert_eq!(rbuf.len(), 0);
    }

    #[test]
    fn test_split_odd_lengths() {
        let seq = b"ACTGACTGAC"; // 10 bases
        let mut ebuf = Vec::new();
        encode(seq, &mut ebuf).unwrap();

        let mut lbuf = Vec::new();
        let mut rbuf = Vec::new();

        // Split at position 7
        split_packed(&ebuf, seq.len(), 7, &mut lbuf, &mut rbuf).unwrap();

        // Assert the expected number of chunks are found in both buffers
        assert_eq!(lbuf.len(), 1);
        assert_eq!(rbuf.len(), 1);

        let mut left = Vec::new();
        decode(&lbuf, 7, &mut left).unwrap();
        assert_eq!(&left, b"ACTGACT");

        let mut right = Vec::new();
        decode(&rbuf, 3, &mut right).unwrap();
        assert_eq!(&right, b"GAC");
    }

    #[test]
    fn test_split_at_chunk_boundary() {
        let seq = b"ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"; // 40 bases
        let mut ebuf = Vec::new();
        encode(seq, &mut ebuf).unwrap();

        let mut lbuf = Vec::new();
        let mut rbuf = Vec::new();

        // Split at position 32 (chunk boundary)
        split_packed(&ebuf, seq.len(), 32, &mut lbuf, &mut rbuf).unwrap();

        // Assert the expected number of chunks are found in both buffers
        assert_eq!(lbuf.len(), 2);
        assert_eq!(rbuf.len(), 1);

        let mut left = Vec::new();
        decode(&lbuf, 32, &mut left).unwrap();
        assert_eq!(&left, &seq[..32]);

        let mut right = Vec::new();
        decode(&rbuf, 8, &mut right).unwrap();
        assert_eq!(&right, &seq[32..]);
    }

    #[test]
    fn test_invalid_inputs() {
        let seq = b"ACTG";
        let mut ebuf = Vec::new();
        encode(seq, &mut ebuf).unwrap();

        let mut lbuf = Vec::new();
        let mut rbuf = Vec::new();

        // Out of bounds index
        assert!(split_packed(&ebuf, seq.len(), seq.len() + 1, &mut lbuf, &mut rbuf).is_err());
    }
}
