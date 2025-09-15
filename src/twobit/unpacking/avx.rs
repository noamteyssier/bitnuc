use crate::NucleotideError;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn unpack_8_bases(packed: u64, lookup: __m128i) -> __m128i {
    let mut indices = [0u8; 16];

    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 2)) & 0b11) as u8;
    }
    let index_vec = _mm_loadu_si128(indices.as_ptr() as *const __m128i);
    _mm_shuffle_epi8(lookup, index_vec)
}

#[inline(always)]
unsafe fn unpack_16_bases(packed: u64, lookup: __m128i) -> __m128i {
    let mut indices = [0u8; 16];
    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 2)) & 0b11) as u8;
    }
    let index_vec = _mm_loadu_si128(indices.as_ptr() as *const __m128i);
    _mm_shuffle_epi8(lookup, index_vec)
}

#[inline(always)]
unsafe fn unpack_32_bases(packed: u64, lookup: __m256i) -> __m256i {
    let mut indices = [0u8; 32];
    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 2)) & 0b11) as u8;
    }
    let index_vec = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
    _mm256_shuffle_epi8(lookup, index_vec)
}

#[inline(always)]
unsafe fn process_remainder(packed: u64, start: usize, end: usize, sequence: &mut Vec<u8>) {
    static LOOKUP: [u8; 4] = [b'A', b'C', b'G', b'T'];
    let count = end - start;
    let old_len = sequence.len();
    sequence.reserve(count);

    let ptr = sequence.as_mut_ptr().add(old_len);
    for i in 0..count {
        let bits = (packed >> ((start + i) * 2)) & 0b11;
        *ptr.add(i) = LOOKUP[bits as usize];
    }
    sequence.set_len(old_len + count);
}

pub unsafe fn from_2bit_simd(
    packed: u64,
    expected_size: usize,
    sequence: &mut Vec<u8>,
) -> Result<(), NucleotideError> {
    if expected_size > 32 {
        return Err(NucleotideError::InvalidLength(expected_size));
    }

    sequence.reserve(expected_size);

    if expected_size >= 32 {
        // 32 bases at a time
        let lookup = _mm256_setr_epi8(
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8,
            b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8,
            b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8,
            b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8,
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8,
        );
        let result = unpack_32_bases(packed, lookup);
        let mut temp = [0u8; 32];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result);
        sequence.extend_from_slice(&temp[..expected_size]);
    } else if expected_size >= 16 {
        // 16 bases at a time
        let lookup = _mm_setr_epi8(
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8,
            b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8,
            b'G' as i8, b'T' as i8,
        );
        let simd_chunks = expected_size / 16;
        for chunk in 0..simd_chunks {
            let chunk_data = packed >> (chunk * 32);
            let result = unpack_16_bases(chunk_data, lookup);
            let mut temp = [0u8; 16];
            _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, result);
            sequence.extend_from_slice(&temp[..16]);
        }
        let remaining_start = simd_chunks * 16;
        process_remainder(packed, remaining_start, expected_size, sequence);
    } else if expected_size >= 8 {
        // 8 bases at a time
        let lookup = _mm_setr_epi8(
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8,
            b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8,
            b'G' as i8, b'T' as i8,
        );
        let simd_chunks = expected_size / 8;
        for chunk in 0..simd_chunks {
            let chunk_data = packed >> (chunk * 16);
            let result = unpack_8_bases(chunk_data, lookup);
            let mut temp = [0u8; 16];
            _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, result);
            sequence.extend_from_slice(&temp[..8]);
        }
        let remaining_start = simd_chunks * 8;
        process_remainder(packed, remaining_start, expected_size, sequence);
    } else {
        // Small sequences are handled by the naive implementation
        process_remainder(packed, 0, expected_size, sequence);
    }

    Ok(())
}

#[inline(always)]
pub unsafe fn from_2bit_multi_simd(
    ebuf: &[u64],
    n_bases: usize,
    sequence: &mut Vec<u8>,
) -> Result<(), NucleotideError> {
    sequence.reserve(n_bases);

    // Set up SIMD lookup table once for all chunks
    let lookup = _mm256_setr_epi8(
        b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8,
        b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8,
        b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8,
        b'C' as i8, b'G' as i8, b'T' as i8, b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8,
        b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8,
    );

    // Process full 32-base chunks
    let full_chunks = n_bases / 32;
    let mut temp = [0u8; 32];

    for &chunk in ebuf.iter().take(full_chunks) {
        let result = unpack_32_bases(chunk, lookup);
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result);
        sequence.extend_from_slice(&temp);
    }

    // Handle remaining bases if any
    let remaining_bases = n_bases % 32;
    if remaining_bases > 0 {
        let last_chunk = ebuf[full_chunks];
        let result = unpack_32_bases(last_chunk, lookup);
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result);
        sequence.extend_from_slice(&temp[..remaining_bases]);
    }

    Ok(())
}

#[cfg(test)]
mod testing {
    use super::*;
    use crate::as_2bit;

    #[test]
    fn test_from_2bit_simd() {
        let expected = b"ACTGACTGACTGACTGACTG";
        let packed = as_2bit(expected).unwrap();
        let mut observed = Vec::new();
        unsafe {
            from_2bit_simd(packed, 20, &mut observed).unwrap();
        }
        assert_eq!(&observed, expected);
    }

    #[test]
    fn test_various_lengths() {
        for len in 1..=32 {
            let input = b"ACTGACTGACTGACTGACTGACTGACTGACTG";
            let packed = as_2bit(&input[..len]).unwrap();
            let mut observed = Vec::new();
            unsafe {
                from_2bit_simd(packed, len, &mut observed).unwrap();
            }
            assert_eq!(&observed, &input[..len], "Failed at length {}", len);
        }
    }

    #[test]
    fn test_append() {
        let sequence = b"ACTGACTGACTGACTGACTG";
        let packed = as_2bit(sequence).unwrap();
        let mut observed = Vec::new();
        unsafe {
            from_2bit_simd(packed, 10, &mut observed).unwrap();
            from_2bit_simd(packed, 10, &mut observed).unwrap();
        }
        let expected = b"ACTGACTGACACTGACTGAC"; // repeated out of phase
        assert_eq!(&observed, expected);
    }
}
