use crate::NucleotideError;

#[inline(always)]
pub fn as_4bit(seq: &[u8]) -> Result<u64, NucleotideError> {
    if seq.len() > 16 {
        // 16 bases * 4 bits = 64 bits max
        return Err(NucleotideError::SequenceTooLong(seq.len()));
    }
    let mut packed = 0u64;
    for (i, &base) in seq.iter().enumerate() {
        let bits = match base {
            b'A' | b'a' => 0b0000,
            b'C' | b'c' => 0b0001,
            b'G' | b'g' => 0b0010,
            b'T' | b't' => 0b0011,
            // IUPAC ambiguous nucleotide codes - all mapped to 0b1111 (15)
            b'N' | b'n' => 0b1111, // Any base
            b'R' | b'r' => 0b1111, // A or G (puRine)
            b'Y' | b'y' => 0b1111, // C or T (pYrimidine)
            b'S' | b's' => 0b1111, // G or C (Strong)
            b'W' | b'w' => 0b1111, // A or T (Weak)
            b'K' | b'k' => 0b1111, // G or T (Keto)
            b'M' | b'm' => 0b1111, // A or C (aMino)
            b'B' | b'b' => 0b1111, // C, G, or T (not A)
            b'D' | b'd' => 0b1111, // A, G, or T (not C)
            b'H' | b'h' => 0b1111, // A, C, or T (not G)
            b'V' | b'v' => 0b1111, // A, C, or G (not T)
            invalid => return Err(NucleotideError::InvalidBase(invalid)),
        };
        packed |= (bits as u64) << (i * 4);
    }
    Ok(packed)
}

pub fn encode_internal(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), NucleotideError> {
    // Clear the buffer
    ebuf.clear();

    // Calculate the number of chunks (16 bases per u64 with 4-bit encoding)
    let n_chunks = sequence.len().div_ceil(16);

    let mut l_bounds = 0;
    for _ in 0..n_chunks - 1 {
        let r_bounds = l_bounds + 16;
        let chunk = &sequence[l_bounds..r_bounds];

        let bits = as_4bit(chunk)?;
        ebuf.push(bits);
        l_bounds = r_bounds;
    }

    let bits = as_4bit(&sequence[l_bounds..])?;
    ebuf.push(bits);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_4bit_basic() {
        // Test basic nucleotides
        let tests: Vec<(&'static [u8], u64)> = vec![
            (b"A", 0b0000),
            (b"C", 0b0001),
            (b"G", 0b0010),
            (b"T", 0b0011),
            (b"ACGT", 0b0011001000010000), // T=3, G=2, C=1, A=0
        ];

        for (input, expected) in tests {
            assert_eq!(as_4bit(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_as_4bit_ambiguous() {
        // Test ambiguous bases - all should encode to 0b1111
        let ambiguous_bases = [
            b'N', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V',
        ];

        for &base in &ambiguous_bases {
            let result = as_4bit(&[base]).unwrap();
            assert_eq!(
                result, 0b1111,
                "Base {} should encode to 0b1111",
                base as char
            );
        }
    }

    #[test]
    fn test_as_4bit_case_insensitive() {
        assert_eq!(as_4bit(b"acgt").unwrap(), as_4bit(b"ACGT").unwrap());
        assert_eq!(as_4bit(b"nrys").unwrap(), as_4bit(b"NRYS").unwrap());
    }

    #[test]
    fn test_as_4bit_sequence_too_long() {
        let long_seq = vec![b'A'; 17];
        assert!(matches!(
            as_4bit(&long_seq),
            Err(NucleotideError::SequenceTooLong(17))
        ));
    }

    #[test]
    fn test_as_4bit_invalid_base() {
        let result = as_4bit(b"ACGX");
        assert!(matches!(result, Err(NucleotideError::InvalidBase(b'X'))));
    }

    #[test]
    fn test_as_4bit_longer_sequences() {
        let seq = b"ACGTACGTACGTACGT"; // 16 bases - maximum for u64
        let result = as_4bit(seq);
        assert!(result.is_ok());

        // Test bit extraction for verification
        let packed = result.unwrap();
        for (i, &expected_base) in seq.iter().enumerate() {
            let extracted_bits = (packed >> (i * 4)) & 0b1111;
            let expected_bits = match expected_base {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => panic!("Unexpected base in test"),
            };
            assert_eq!(extracted_bits, expected_bits, "Mismatch at position {}", i);
        }
    }

    #[test]
    fn test_encode_internal() {
        // Test encoding a sequence longer than 16 bases
        let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT"; // 32 bases
        let mut ebuf = Vec::new();

        encode_internal(sequence, &mut ebuf).unwrap();

        // Should create 2 chunks (32 bases / 16 bases per chunk)
        assert_eq!(ebuf.len(), 2);

        // First chunk should be first 16 bases
        let first_chunk_expected = as_4bit(&sequence[..16]).unwrap();
        assert_eq!(ebuf[0], first_chunk_expected);

        // Second chunk should be remaining 16 bases
        let second_chunk_expected = as_4bit(&sequence[16..]).unwrap();
        assert_eq!(ebuf[1], second_chunk_expected);
    }

    #[test]
    fn test_encode_internal_partial_chunk() {
        // Test encoding a sequence that doesn't fill complete chunks
        let sequence = b"ACGTACGTACGTACGTACGT"; // 20 bases (1 full chunk + 4 partial)
        let mut ebuf = Vec::new();

        encode_internal(sequence, &mut ebuf).unwrap();

        // Should create 2 chunks
        assert_eq!(ebuf.len(), 2);

        // First chunk should be first 16 bases
        let first_chunk_expected = as_4bit(&sequence[..16]).unwrap();
        assert_eq!(ebuf[0], first_chunk_expected);

        // Second chunk should be remaining 4 bases
        let second_chunk_expected = as_4bit(&sequence[16..]).unwrap();
        assert_eq!(ebuf[1], second_chunk_expected);
    }
}
