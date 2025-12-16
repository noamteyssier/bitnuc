use std::arch::x86_64::*;

use crate::Error;

#[repr(u8)]
enum NucleotideBits4 {
    A = 0b0000,
    C = 0b0001,
    G = 0b0010,
    T = 0b0011,
    N = 0b1111, // Using 0b1111 for ambiguous bases
}

#[repr(align(32))]
struct SimdConstants4 {
    zeros: __m256i,
    ones: __m256i,
    twos: __m256i,
    threes: __m256i,
    ns: __m256i, // For N bases
}

impl SimdConstants4 {
    #[allow(unsafe_op_in_unsafe_fn)]
    #[inline(always)]
    unsafe fn new() -> Self {
        Self {
            zeros: _mm256_set1_epi8(NucleotideBits4::A as i8),
            ones: _mm256_set1_epi8(NucleotideBits4::C as i8),
            twos: _mm256_set1_epi8(NucleotideBits4::G as i8),
            threes: _mm256_set1_epi8(NucleotideBits4::T as i8),
            ns: _mm256_set1_epi8(NucleotideBits4::N as i8),
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn create_dual_pattern_mask(chunk: __m256i, upper: i8, lower: i8) -> __m256i {
    _mm256_or_si256(
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(upper)),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(lower)),
    )
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn create_ambiguous_mask(chunk: __m256i) -> __m256i {
    // Create masks for all ambiguous IUPAC codes
    let n_mask = create_dual_pattern_mask(chunk, b'N' as i8, b'n' as i8);
    let r_mask = create_dual_pattern_mask(chunk, b'R' as i8, b'r' as i8); // A or G
    let y_mask = create_dual_pattern_mask(chunk, b'Y' as i8, b'y' as i8); // C or T
    let s_mask = create_dual_pattern_mask(chunk, b'S' as i8, b's' as i8); // G or C
    let w_mask = create_dual_pattern_mask(chunk, b'W' as i8, b'w' as i8); // A or T
    let k_mask = create_dual_pattern_mask(chunk, b'K' as i8, b'k' as i8); // G or T
    let m_mask = create_dual_pattern_mask(chunk, b'M' as i8, b'm' as i8); // A or C
    let b_mask = create_dual_pattern_mask(chunk, b'B' as i8, b'b' as i8); // C,G,T
    let d_mask = create_dual_pattern_mask(chunk, b'D' as i8, b'd' as i8); // A,G,T
    let h_mask = create_dual_pattern_mask(chunk, b'H' as i8, b'h' as i8); // A,C,T
    let v_mask = create_dual_pattern_mask(chunk, b'V' as i8, b'v' as i8); // A,C,G

    // Combine all ambiguous masks
    _mm256_or_si256(
        n_mask,
        _mm256_or_si256(
            r_mask,
            _mm256_or_si256(
                y_mask,
                _mm256_or_si256(
                    s_mask,
                    _mm256_or_si256(
                        w_mask,
                        _mm256_or_si256(
                            k_mask,
                            _mm256_or_si256(
                                m_mask,
                                _mm256_or_si256(
                                    b_mask,
                                    _mm256_or_si256(d_mask, _mm256_or_si256(h_mask, v_mask)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn set_bits_4bit(
    c_mask: __m256i,
    g_mask: __m256i,
    t_mask: __m256i,
    n_mask: __m256i,
    constants: &SimdConstants4,
) -> __m256i {
    let mut result = constants.zeros;

    // Set bits based on nucleotide masks
    result = _mm256_or_si256(
        _mm256_and_si256(c_mask, constants.ones),
        _mm256_andnot_si256(c_mask, result),
    );
    result = _mm256_or_si256(
        _mm256_and_si256(g_mask, constants.twos),
        _mm256_andnot_si256(g_mask, result),
    );
    result = _mm256_or_si256(
        _mm256_and_si256(t_mask, constants.threes),
        _mm256_andnot_si256(t_mask, result),
    );
    result = _mm256_or_si256(
        _mm256_and_si256(n_mask, constants.ns),
        _mm256_andnot_si256(n_mask, result),
    );

    result
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn process_simd_chunk_4bit(chunk: __m256i, constants: &SimdConstants4) -> __m256i {
    let c_mask = create_dual_pattern_mask(chunk, b'C' as i8, b'c' as i8);
    let g_mask = create_dual_pattern_mask(chunk, b'G' as i8, b'g' as i8);
    let t_mask = create_dual_pattern_mask(chunk, b'T' as i8, b't' as i8);
    let n_mask = create_ambiguous_mask(chunk);

    set_bits_4bit(c_mask, g_mask, t_mask, n_mask, constants)
}

pub fn as_4bit(seq: &[u8]) -> Result<u64, Error> {
    if seq.len() > 16 {
        // 16 bases * 4 bits = 64 bits
        return Err(Error::SequenceTooLong(seq.len()));
    }

    // Use naive implementation for small sequences
    if seq.len() < 8 {
        return naive_4bit::as_4bit(seq);
    }

    // Validate all bases
    if let Some(&invalid) = seq.iter().find(|&&b| !is_valid_nucleotide_4bit(b)) {
        return Err(Error::InvalidBase(invalid));
    }

    let mut packed = 0u64;
    let len = seq.len();
    let simd_len = len - (len % 8); // Process 8 bases at a time

    unsafe {
        let constants = SimdConstants4::new();

        for chunk_idx in (0..simd_len).step_by(8) {
            let chunk = _mm256_loadu_si256(seq[chunk_idx..].as_ptr() as *const __m256i);
            let result = process_simd_chunk_4bit(chunk, &constants);

            let mut temp = [0u8; 32];
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result);

            // Pack 4-bit values (only use first 8 bytes)
            for (i, &val) in temp.iter().take(8).enumerate() {
                packed |= (val as u64) << ((chunk_idx + i) * 4);
            }
        }

        // Handle remaining bases
        for (i, &base) in seq.iter().skip(simd_len).enumerate() {
            let bits = match base {
                b'A' | b'a' => NucleotideBits4::A as u64,
                b'C' | b'c' => NucleotideBits4::C as u64,
                b'G' | b'g' => NucleotideBits4::G as u64,
                b'T' | b't' => NucleotideBits4::T as u64,
                _ => NucleotideBits4::N as u64, // All other bases become N
            };
            packed |= bits << ((simd_len + i) * 4);
        }
    }

    Ok(packed)
}

#[inline(always)]
fn is_valid_nucleotide_4bit(base: u8) -> bool {
    matches!(
        base,
        b'A' | b'a'
            | b'C'
            | b'c'
            | b'G'
            | b'g'
            | b'T'
            | b't'
            | b'N'
            | b'n'
            | b'R'
            | b'r'
            | b'Y'
            | b'y'
            | b'S'
            | b's'
            | b'W'
            | b'w'
            | b'K'
            | b'k'
            | b'M'
            | b'm'
            | b'B'
            | b'b'
            | b'D'
            | b'd'
            | b'H'
            | b'h'
            | b'V'
            | b'v'
    )
}

pub fn encode_internal(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), Error> {
    ebuf.clear();

    // Calculate number of chunks (16 bases per u64 with 4-bit encoding)
    let n_chunks = sequence.len().div_ceil(16);

    let mut l_bounds = 0;
    for _ in 0..n_chunks - 1 {
        let r_bounds = l_bounds + 16;
        let chunk = &sequence[l_bounds..r_bounds];

        let bits = as_4bit(chunk)?;
        ebuf.push(bits);
        l_bounds = r_bounds;
    }

    // Handle the final chunk
    let bits = as_4bit(&sequence[l_bounds..])?;
    ebuf.push(bits);

    Ok(())
}

// Naive implementation module for small sequences
mod naive_4bit {
    use super::NucleotideBits4;
    use crate::Error;

    #[inline(always)]
    pub fn as_4bit(seq: &[u8]) -> Result<u64, Error> {
        if seq.len() > 16 {
            return Err(Error::SequenceTooLong(seq.len()));
        }

        let mut packed = 0u64;
        for (i, &base) in seq.iter().enumerate() {
            let bits = match base {
                b'A' | b'a' => NucleotideBits4::A as u64,
                b'C' | b'c' => NucleotideBits4::C as u64,
                b'G' | b'g' => NucleotideBits4::G as u64,
                b'T' | b't' => NucleotideBits4::T as u64,
                // Map all ambiguous bases to N (0b1111)
                b'N' | b'n' | b'R' | b'r' | b'Y' | b'y' | b'S' | b's' | b'W' | b'w' | b'K'
                | b'k' | b'M' | b'm' | b'B' | b'b' | b'D' | b'd' | b'H' | b'h' | b'V' | b'v' => {
                    NucleotideBits4::N as u64
                }
                invalid => return Err(Error::InvalidBase(invalid)),
            };
            packed |= bits << (i * 4);
        }
        Ok(packed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_4bit_basic_encoding() {
        let tests = vec![
            (b"ACGT", 0b0011001000010000), // T=3, G=2, C=1, A=0
            (b"AAAA", 0b0000000000000000),
            (b"TTTT", 0b0011001100110011),
            (b"NNNN", 0b1111111111111111),
        ];

        for (input, expected) in tests {
            assert_eq!(as_4bit(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_4bit_ambiguous_bases() {
        // Test that ambiguous bases are encoded as N (0b1111)
        let seq_with_n = b"ACGN";
        let seq_with_r = b"ACGR"; // R should become N

        let n_result = as_4bit(seq_with_n).unwrap();
        let r_result = as_4bit(seq_with_r).unwrap();

        assert_eq!(n_result, r_result); // Both should have same encoding

        // Extract the last 4 bits (4th base)
        let last_bits_n = (n_result >> 12) & 0b1111;
        let last_bits_r = (r_result >> 12) & 0b1111;

        assert_eq!(last_bits_n, 0b1111);
        assert_eq!(last_bits_r, 0b1111);
    }

    #[test]
    fn test_4bit_sequence_too_long() {
        let long_seq = vec![b'A'; 17];
        assert!(matches!(
            as_4bit(&long_seq),
            Err(Error::SequenceTooLong(17))
        ));
    }

    #[test]
    fn test_4bit_case_insensitive() {
        assert_eq!(as_4bit(b"acgt").unwrap(), as_4bit(b"ACGT").unwrap());
        assert_eq!(as_4bit(b"nrys").unwrap(), as_4bit(b"NRYS").unwrap());
    }
}
