use crate::NucleotideError;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn unpack_8_bases(packed: u64, lookup: uint8x8_t) -> uint8x8_t {
    // Create indices array for 8 bases
    let mut indices = [0u8; 8];
    for (i, v) in indices.iter_mut().enumerate() {
        *v = ((packed >> (i * 2)) & 0b11) as u8;
    }

    // Load indices and perform table lookup
    let index_vec = vld1_u8(indices.as_ptr());
    vtbl1_u8(lookup, index_vec)
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

    // Create lookup table for base conversion
    let lookup = vld1_u8([b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T'].as_ptr());

    // Process in chunks of 8 bases
    let simd_chunks = expected_size / 8;
    for chunk in 0..simd_chunks {
        let chunk_data = packed >> (chunk * 16);
        let result = unpack_8_bases(chunk_data, lookup);
        let mut temp = [0u8; 8];
        vst1_u8(temp.as_mut_ptr(), result);
        sequence.extend_from_slice(&temp);
    }

    // Handle remaining bases
    let remaining_start = simd_chunks * 8;
    if remaining_start < expected_size {
        process_remainder(packed, remaining_start, expected_size, sequence);
    }

    Ok(())
}

/// Decode 16 packed 2‑bit codes (`u32`) to ASCII (`A`, `C`, `G`, `T`).
#[inline(always)]
pub unsafe fn decode_16_nucleotides(encoded: u32, dst: *mut u8) {
    // 1. Broadcast the word to four lanes
    let val = vdupq_n_u32(encoded);
    let mask = vdupq_n_u32(3);

    // 2. Extract 2‑bit fields (negative counts = right shift)
    #[inline(always)]
    const fn shv(a: i32, b: i32, c: i32, d: i32) -> int32x4_t {
        unsafe { core::mem::transmute([a, b, c, d]) }
    }
    let c0 = vandq_u32(vshlq_u32(val, shv(0, -2, -4, -6)), mask);
    let c1 = vandq_u32(vshlq_u32(val, shv(-8, -10, -12, -14)), mask);
    let c2 = vandq_u32(vshlq_u32(val, shv(-16, -18, -20, -22)), mask);
    let c3 = vandq_u32(vshlq_u32(val, shv(-24, -26, -28, -30)), mask);

    // 3. Narrow u32 → u8 and assemble 16 indices
    let idx: uint8x16_t = vcombine_u8(
        vmovn_u16(vcombine_u16(vmovn_u32(c0), vmovn_u32(c1))),
        vmovn_u16(vcombine_u16(vmovn_u32(c2), vmovn_u32(c3))),
    );

    // 4. LUT: 0→A, 1→C, 2→G, 3→T
    let lut: uint8x16_t = core::mem::transmute([
        b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T', b'A', b'C', b'G',
        b'T',
    ]);
    let ascii = vqtbl1q_u8(lut, idx);

    // 5. Store
    vst1q_u8(dst, ascii);
}

/// Decode a packed 2‑bit stream (`u64` words) back to ASCII nucleotides.
pub unsafe fn decode_nucleotides_simd(
    input: &[u64],
    len: usize,
    output: &mut [u8],
) -> Result<(), NucleotideError> {
    if len > output.len() {
        return Err(NucleotideError::InvalidLength(len));
    }

    let chunk = 32;
    let chunks = len / chunk;

    for i in 0..chunks {
        let w = input.get(i).copied().unwrap_or(0);
        decode_16_nucleotides(w as u32, output.as_mut_ptr().add(i * chunk));
        decode_16_nucleotides((w >> 32) as u32, output.as_mut_ptr().add(i * chunk + 16));
    }

    // Scalar tail
    let lut = [b'A', b'C', b'G', b'T'];
    for j in (chunks * chunk)..len {
        let idx = ((input[j / 32] >> (2 * (j % 32))) & 3) as usize;
        output[j] = lut[idx];
    }
    Ok(())
}

pub fn fast_decode(enc: &[u64], len: usize, out: &mut Vec<u8>) -> Result<(), NucleotideError> {
    out.resize(len, 0);
    unsafe { decode_nucleotides_simd(enc, len, out) }
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
        let expected = b"ACTGACTGACACTGACTGAC"; // Two copies of the first 10 bases
        assert_eq!(&observed, expected);
    }
}
