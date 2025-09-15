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

// Using 128-bit SSE2 instead of 256-bit AVX2 to better match data size
#[repr(align(16))]
struct SimdConstants {
    zeros: __m128i,
    ones: __m128i,
    twos: __m128i,
    threes: __m128i,
}

impl SimdConstants {
    #[inline(always)]
    unsafe fn new() -> Self {
        Self {
            zeros: _mm_set1_epi8(NucleotideBits::A as i8),
            ones: _mm_set1_epi8(NucleotideBits::C as i8),
            twos: _mm_set1_epi8(NucleotideBits::G as i8),
            threes: _mm_set1_epi8(NucleotideBits::T as i8),
        }
    }
}

#[inline(always)]
unsafe fn create_dual_pattern_mask(chunk: __m128i, upper: i8, lower: i8) -> __m128i {
    _mm_or_si128(
        _mm_cmpeq_epi8(chunk, _mm_set1_epi8(upper)),
        _mm_cmpeq_epi8(chunk, _mm_set1_epi8(lower)),
    )
}

// Optimized bit selection using AND+OR instead of blend
#[inline(always)]
unsafe fn set_bits(
    c_mask: __m128i,
    g_mask: __m128i,
    t_mask: __m128i,
    constants: &SimdConstants,
) -> __m128i {
    let mut result = constants.zeros;

    // Using AND+OR operations which can be more efficient than blend
    result = _mm_or_si128(
        _mm_and_si128(c_mask, constants.ones),
        _mm_andnot_si128(c_mask, result),
    );
    result = _mm_or_si128(
        _mm_and_si128(g_mask, constants.twos),
        _mm_andnot_si128(g_mask, result),
    );
    result = _mm_or_si128(
        _mm_and_si128(t_mask, constants.threes),
        _mm_andnot_si128(t_mask, result),
    );

    result
}

#[inline(always)]
unsafe fn process_simd_chunk(chunk: __m128i, constants: &SimdConstants) -> __m128i {
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

    // Keep the same threshold as your AARCH64 version
    if seq.len() < 8 {
        return naive::as_2bit(seq);
    }

    // Pre-validate bases
    if let Some(&invalid) = seq
        .iter()
        .find(|&&b| !matches!(b, b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't'))
    {
        return Err(NucleotideError::InvalidBase(invalid));
    }

    let mut packed = 0u64;
    let len = seq.len();
    // Process 8 bytes at a time like the AARCH64 version
    let simd_len = len - (len % 8);

    unsafe {
        let constants = SimdConstants::new();

        for chunk_idx in (0..simd_len).step_by(8) {
            // Use 128-bit load instead of 256-bit
            let chunk = _mm_loadu_si128(seq[chunk_idx..].as_ptr() as *const __m128i);
            let result = process_simd_chunk(chunk, &constants);

            let mut temp = [0u8; 16];
            _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, result);

            for (i, &val) in temp.iter().take(8).enumerate() {
                packed |= (val as u64) << ((chunk_idx + i) * 2);
            }
        }

        // Handle remaining bases the same way as AARCH64
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
