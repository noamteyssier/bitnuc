use super::naive;
use crate::NucleotideError;
use std::arch::x86_64::*;

#[repr(u8)]
enum NucleotideBits {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
}

#[repr(align(32))] // Changed to 32-byte alignment for AVX2
struct SimdConstants {
    zeros: __m256i,
    ones: __m256i,
    twos: __m256i,
    threes: __m256i,
}

impl SimdConstants {
    #[inline(always)]
    unsafe fn new() -> Self {
        Self {
            zeros: _mm256_set1_epi8(NucleotideBits::A as i8),
            ones: _mm256_set1_epi8(NucleotideBits::C as i8),
            twos: _mm256_set1_epi8(NucleotideBits::G as i8),
            threes: _mm256_set1_epi8(NucleotideBits::T as i8),
        }
    }
}

#[inline(always)]
unsafe fn create_dual_pattern_mask(chunk: __m256i, upper: i8, lower: i8) -> __m256i {
    _mm256_or_si256(
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(upper)),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(lower)),
    )
}

#[inline(always)]
unsafe fn set_bits(
    c_mask: __m256i,
    g_mask: __m256i,
    t_mask: __m256i,
    constants: &SimdConstants,
) -> __m256i {
    let mut result = constants.zeros;

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

    result
}

#[inline(always)]
unsafe fn process_simd_chunk(chunk: __m256i, constants: &SimdConstants) -> __m256i {
    let (c_mask, g_mask, t_mask) = (
        create_dual_pattern_mask(chunk, b'C' as i8, b'c' as i8),
        create_dual_pattern_mask(chunk, b'G' as i8, b'g' as i8),
        create_dual_pattern_mask(chunk, b'T' as i8, b't' as i8),
    );
    set_bits(c_mask, g_mask, t_mask, constants)
}

pub fn as_2bit(seq: &[u8]) -> Result<u64, NucleotideError> {
    if seq.len() > 32 {
        return Err(NucleotideError::SequenceTooLong(seq.len()));
    }

    // Increased minimum length for AVX2
    if seq.len() < 16 {
        return naive::as_2bit(seq);
    }

    if let Some(&invalid) = seq
        .iter()
        .find(|&&b| !matches!(b, b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't'))
    {
        return Err(NucleotideError::InvalidBase(invalid));
    }

    let mut packed = 0u64;
    let len = seq.len();
    // Process 16 bytes at a time for 256-bit operations
    let simd_len = len - (len % 16);

    unsafe {
        let constants = SimdConstants::new();

        for chunk_idx in (0..simd_len).step_by(16) {
            // Use 256-bit load
            let chunk = _mm256_loadu_si256(seq[chunk_idx..].as_ptr() as *const __m256i);
            let result = process_simd_chunk(chunk, &constants);

            let mut temp = [0u8; 32]; // Increased to 32 bytes for 256-bit
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result);

            for (i, &val) in temp.iter().take(16).enumerate() {
                packed |= (val as u64) << ((chunk_idx + i) * 2);
            }
        }

        // Handle remaining bases
        for (i, &base) in seq.iter().skip(simd_len).enumerate() {
            let bits = match base {
                b'A' | b'a' => NucleotideBits::A as u64,
                b'C' | b'c' => NucleotideBits::C as u64,
                b'G' | b'g' => NucleotideBits::G as u64,
                b'T' | b't' => NucleotideBits::T as u64,
                _ => unreachable!(),
            };
            packed |= bits << ((simd_len + i) * 2);
        }
    }

    Ok(packed)
}

pub fn encode_internal(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), NucleotideError> {
    // Clear the buffer
    ebuf.clear();

    // Calculate the number of chunks
    let n_chunks = sequence.len().div_ceil(32);

    let mut l_bounds = 0;
    for _ in 0..n_chunks - 1 {
        let r_bounds = l_bounds + 32;
        let chunk = &sequence[l_bounds..r_bounds];

        let bits = as_2bit(chunk)?;
        ebuf.push(bits);
        l_bounds = r_bounds;
    }

    let bits = as_2bit(&sequence[l_bounds..])?;
    ebuf.push(bits);

    Ok(())
}
