use std::arch::aarch64::*;

use crate::Error;

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_4_bases(packed: u64, lookup: uint8x16_t) -> uint8x8_t {
    let mut indices = [0u8; 8];

    for (i, v) in indices.iter_mut().take(4).enumerate() {
        *v = ((packed >> (i * 4)) & 0b1111) as u8;
    }
    let index_vec = vld1_u8(indices.as_ptr());
    vqtbl1_u8(lookup, index_vec) // Use vqtbl1_u8 for 16-element lookup
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_8_bases(packed: u64, lookup: uint8x16_t) -> uint8x8_t {
    let mut indices = [0u8; 8];

    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 4)) & 0b1111) as u8;
    }
    let index_vec = vld1_u8(indices.as_ptr());
    // Use 16-byte lookup table for 4-bit indices
    vqtbl1_u8(lookup, index_vec)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn unpack_16_bases(packed: u64, lookup: uint8x16_t) -> uint8x16_t {
    let mut indices = [0u8; 16];
    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 4)) & 0b1111) as u8;
    }
    let index_vec = vld1q_u8(indices.as_ptr());
    vqtbl1q_u8(lookup, index_vec)
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
        let lookup = vld1q_u8(
            [
                b'A', b'C', b'G', b'T', // 0-3
                b'N', b'N', b'N', b'N', // 4-7
                b'N', b'N', b'N', b'N', // 8-11
                b'N', b'N', b'N', b'N', // 12-15
            ]
            .as_ptr(),
        );
        let result = unpack_16_bases(packed, lookup);
        let mut temp = [0u8; 16];
        vst1q_u8(temp.as_mut_ptr(), result);
        sequence.extend_from_slice(&temp[..expected_size]);
    } else if expected_size >= 8 {
        // 8 bases at a time
        let lookup = vld1q_u8(
            [
                b'A', b'C', b'G', b'T', // 0-3
                b'N', b'N', b'N', b'N', // 4-7
                b'N', b'N', b'N', b'N', // 8-11
                b'N', b'N', b'N', b'N', // 12-15
            ]
            .as_ptr(),
        );
        let simd_chunks = expected_size / 8;
        for chunk in 0..simd_chunks {
            let chunk_data = packed >> (chunk * 32);
            let result = unpack_8_bases(chunk_data, lookup);
            let mut temp = [0u8; 8];
            vst1_u8(temp.as_mut_ptr(), result);
            sequence.extend_from_slice(&temp[..8]);
        }
        let remaining_start = simd_chunks * 8;
        process_remainder_4bit(packed, remaining_start, expected_size, sequence);
    } else if expected_size >= 4 {
        // 4 bases at a time
        let lookup = vld1q_u8(
            // Use 16-element lookup table
            [
                b'A', b'C', b'G', b'T', // 0-3
                b'N', b'N', b'N', b'N', // 4-7
                b'N', b'N', b'N', b'N', // 8-11
                b'N', b'N', b'N', b'N', // 12-15
            ]
            .as_ptr(),
        );
        let simd_chunks = expected_size / 4;
        for chunk in 0..simd_chunks {
            let chunk_data = packed >> (chunk * 16);
            let result = unpack_4_bases(chunk_data, lookup);
            let mut temp = [0u8; 8];
            vst1_u8(temp.as_mut_ptr(), result);
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
    sequence.reserve(n_bases);

    // Set up SIMD lookup table once for all chunks
    let lookup = vld1q_u8(
        [
            b'A', b'C', b'G', b'T', // 0-3
            b'N', b'N', b'N', b'N', // 4-7
            b'N', b'N', b'N', b'N', // 8-11
            b'N', b'N', b'N', b'N', // 12-15
        ]
        .as_ptr(),
    );

    // Process full 16-base chunks
    let full_chunks = n_bases / 16;
    let mut temp = [0u8; 16];

    for &chunk in ebuf.iter().take(full_chunks) {
        let result = unpack_16_bases(chunk, lookup);
        vst1q_u8(temp.as_mut_ptr(), result);
        sequence.extend_from_slice(&temp);
    }

    // Handle remaining bases if any
    let remaining_bases = n_bases % 16;
    if remaining_bases > 0 {
        let last_chunk = ebuf[full_chunks];
        let result = unpack_16_bases(last_chunk, lookup);
        vst1q_u8(temp.as_mut_ptr(), result);
        sequence.extend_from_slice(&temp[..remaining_bases]);
    }

    Ok(())
}

#[cfg(test)]
mod testing {
    use super::*;
    use crate::fourbit::as_4bit;

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
