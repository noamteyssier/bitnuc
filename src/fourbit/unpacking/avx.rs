use std::arch::x86_64::*;

use crate::Error;

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_4_bases(packed: u64, lookup: __m128i) -> __m128i {
    let mut indices = [0u8; 16];

    for (i, v) in indices.iter_mut().take(4).enumerate() {
        *v = ((packed >> (i * 4)) & 0b1111) as u8;
    }
    let index_vec = _mm_loadu_si128(indices.as_ptr() as *const __m128i);
    _mm_shuffle_epi8(lookup, index_vec)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_8_bases(packed: u64, lookup: __m128i) -> __m128i {
    let mut indices = [0u8; 16];

    for (i, v) in indices.iter_mut().take(8).enumerate() {
        *v = ((packed >> (i * 4)) & 0b1111) as u8;
    }
    let index_vec = _mm_loadu_si128(indices.as_ptr() as *const __m128i);
    _mm_shuffle_epi8(lookup, index_vec)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_16_bases(packed: u64, lookup: __m128i) -> __m128i {
    let mut indices = [0u8; 16];
    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 4)) & 0b1111) as u8;
    }
    let index_vec = _mm_loadu_si128(indices.as_ptr() as *const __m128i);
    _mm_shuffle_epi8(lookup, index_vec)
}

/// Unpack 32 4-bit encoded bases from two u64s into 32 ASCII nucleotides
#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_32_bases(packed1: u64, packed2: u64, lookup: __m256i) -> __m256i {
    // Extract nibbles from both u64s into a 32-byte array
    let mut indices = [0u8; 32];

    // Process first u64 (bases 0-15)
    let bytes1 = packed1.to_le_bytes();
    for i in 0..8 {
        let byte = bytes1[i];
        indices[i * 2] = byte & 0x0F; // Low nibble
        indices[i * 2 + 1] = byte >> 4; // High nibble
    }

    // Process second u64 (bases 16-31)
    let bytes2 = packed2.to_le_bytes();
    for i in 0..8 {
        let byte = bytes2[i];
        indices[16 + i * 2] = byte & 0x0F;
        indices[16 + i * 2 + 1] = byte >> 4;
    }

    // Use SIMD shuffle to convert indices to nucleotides
    let index_vec = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
    _mm256_shuffle_epi8(lookup, index_vec)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn process_remainder_4bit(packed: u64, start: usize, end: usize, sequence: &mut Vec<u8>) {
    static LOOKUP: [u8; 16] = [
        b'A', b'C', b'G', b'T', // 0-3: Standard bases
        b'N', b'N', b'N', b'N', // 4-7: Reserved/unused (map to N)
        b'N', b'N', b'N', b'N', // 8-11: Reserved/unused (map to N)
        b'N', b'N', b'N',
        b'N', // 12-14: Reserved/unused (map to N)
              // 15: N (ambiguous base)
    ];
    let count = end - start;
    let old_len = sequence.len();
    sequence.reserve(count);

    let ptr = sequence.as_mut_ptr().add(old_len);
    for i in 0..count {
        let bits = (packed >> ((start + i) * 4)) & 0b1111;
        *ptr.add(i) = LOOKUP[bits as usize];
    }
    sequence.set_len(old_len + count);
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn from_4bit_simd(
    packed: u64,
    expected_size: usize,
    sequence: &mut Vec<u8>,
) -> Result<(), Error> {
    if expected_size > 16 {
        return Err(Error::InvalidLength(expected_size));
    }

    sequence.reserve(expected_size);

    if expected_size >= 16 {
        // 16 bases at a time (maximum for 4-bit encoding in u64)
        let lookup = _mm_setr_epi8(
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, // 0-3
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 4-7
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 8-11
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 12-15
        );
        let result = unpack_16_bases(packed, lookup);
        let mut temp = [0u8; 16];
        _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, result);
        sequence.extend_from_slice(&temp[..expected_size]);
    } else if expected_size >= 8 {
        // 8 bases at a time
        let lookup = _mm_setr_epi8(
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, // 0-3
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 4-7
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 8-11
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 12-15
        );
        let simd_chunks = expected_size / 8;
        for chunk in 0..simd_chunks {
            let chunk_data = packed >> (chunk * 32);
            let result = unpack_8_bases(chunk_data, lookup);
            let mut temp = [0u8; 16];
            _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, result);
            sequence.extend_from_slice(&temp[..8]);
        }
        let remaining_start = simd_chunks * 8;
        process_remainder_4bit(packed, remaining_start, expected_size, sequence);
    } else if expected_size >= 4 {
        // 4 bases at a time
        let lookup = _mm_setr_epi8(
            b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, // 0-3
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 4-7
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 8-11
            b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, // 12-15
        );
        let simd_chunks = expected_size / 4;
        for chunk in 0..simd_chunks {
            let chunk_data = packed >> (chunk * 16);
            let result = unpack_4_bases(chunk_data, lookup);
            let mut temp = [0u8; 16];
            _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, result);
            sequence.extend_from_slice(&temp[..4]);
        }
        let remaining_start = simd_chunks * 4;
        process_remainder_4bit(packed, remaining_start, expected_size, sequence);
    } else {
        // Small sequences are handled by the naive implementation
        process_remainder_4bit(packed, 0, expected_size, sequence);
    }

    Ok(())
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
pub unsafe fn decode_internal(
    ebuf: &[u64],
    n_bases: usize,
    sequence: &mut Vec<u8>,
) -> Result<(), Error> {
    // Setup lookup table (repeated twice for both 128-bit lanes)
    let lookup = _mm256_setr_epi8(
        b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'N' as i8, b'N' as i8, b'N' as i8,
        b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8,
        b'N' as i8, b'N' as i8, // Repeat for upper 128 bits
        b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'N' as i8, b'N' as i8, b'N' as i8,
        b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8, b'N' as i8,
        b'N' as i8, b'N' as i8,
    );

    // Pre-allocate and get write pointer
    let old_len = sequence.len();
    sequence.reserve(n_bases);
    let mut out_ptr = sequence.as_mut_ptr().add(old_len);

    // Process 32 bases at a time (2 u64s per iteration)
    let full_chunks = n_bases / 32;

    for i in 0..full_chunks {
        let packed1 = ebuf[i * 2];
        let packed2 = ebuf[i * 2 + 1];
        let result = unpack_32_bases(packed1, packed2, lookup);
        _mm256_storeu_si256(out_ptr as *mut __m256i, result);
        out_ptr = out_ptr.add(32);
    }

    // Handle remaining bases (0-31)
    let remaining_bases = n_bases % 32;
    if remaining_bases > 0 {
        let offset = full_chunks * 2;
        let packed1 = ebuf.get(offset).copied().unwrap_or(0);
        let packed2 = ebuf.get(offset + 1).copied().unwrap_or(0);

        let result = unpack_32_bases(packed1, packed2, lookup);
        let mut temp = [0u8; 32];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result);
        std::ptr::copy_nonoverlapping(temp.as_ptr(), out_ptr, remaining_bases);
    }

    sequence.set_len(old_len + n_bases);
    Ok(())
}

#[cfg(test)]
mod testing {
    use super::*;
    use crate::as_4bit;

    #[test]
    fn test_from_4bit_simd_basic() {
        let expected = b"ACGT";
        let packed = as_4bit(expected).unwrap();
        let mut observed = Vec::new();
        unsafe {
            from_4bit_simd(packed, 4, &mut observed).unwrap();
        }
        assert_eq!(&observed, expected);
    }

    #[test]
    fn test_from_4bit_simd_with_n() {
        let expected = b"ACGN";
        let packed = as_4bit(expected).unwrap();
        let mut observed = Vec::new();
        unsafe {
            from_4bit_simd(packed, 4, &mut observed).unwrap();
        }
        assert_eq!(&observed, expected);
    }

    #[test]
    fn test_from_4bit_simd_max_length() {
        let expected = b"ACGTACGTACGTACGT"; // 16 bases - maximum for u64 with 4-bit
        let packed = as_4bit(expected).unwrap();
        let mut observed = Vec::new();
        unsafe {
            from_4bit_simd(packed, 16, &mut observed).unwrap();
        }
        assert_eq!(&observed, expected);
    }

    #[test]
    fn test_various_lengths() {
        for len in 1..=16 {
            let input = b"ACGTACGTACGTACGT";
            let packed = as_4bit(&input[..len]).unwrap();
            let mut observed = Vec::new();
            unsafe {
                from_4bit_simd(packed, len, &mut observed).unwrap();
            }
            assert_eq!(&observed, &input[..len], "Failed at length {}", len);
        }
    }

    #[test]
    fn test_append() {
        let sequence = b"ACGTACGTACGTACGT";
        let packed = as_4bit(sequence).unwrap();
        let mut observed = Vec::new();
        unsafe {
            from_4bit_simd(packed, 8, &mut observed).unwrap();
            from_4bit_simd(packed, 8, &mut observed).unwrap();
        }
        let expected = b"ACGTACGTACGTACGT"; // Two copies of the first 8 bases
        assert_eq!(&observed, expected);
    }

    #[test]
    fn test_multi_chunk_decoding() {
        // Test with multiple chunks
        let sequence1 = b"ACGTACGTACGTACGT"; // 16 bases
        let sequence2 = b"TGCATGCATGCATGCA"; // 16 bases
        let mut ebuf = Vec::new();

        // Encode both sequences
        let packed1 = as_4bit(sequence1).unwrap();
        let packed2 = as_4bit(sequence2).unwrap();
        ebuf.push(packed1);
        ebuf.push(packed2);

        // Decode using multi function
        let mut decoded = Vec::new();
        unsafe {
            decode_internal(&ebuf, 32, &mut decoded).unwrap();
        }

        let expected: Vec<u8> = sequence1.iter().chain(sequence2.iter()).cloned().collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_partial_last_chunk() {
        // Test with partial last chunk
        let sequence = b"ACGTACGTACGTACGTACGT"; // 20 bases (1 full + 4 partial)
        let mut ebuf = Vec::new();

        // This would create 2 chunks: first 16 bases, then remaining 4
        let packed1 = as_4bit(&sequence[..16]).unwrap();
        let packed2 = as_4bit(&sequence[16..]).unwrap();
        ebuf.push(packed1);
        ebuf.push(packed2);

        let mut decoded = Vec::new();
        unsafe {
            decode_internal(&ebuf, 20, &mut decoded).unwrap();
        }

        assert_eq!(&decoded, sequence);
    }
}
