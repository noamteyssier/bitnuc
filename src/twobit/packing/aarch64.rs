use super::naive;
use crate::error::NucleotideError;
use std::arch::aarch64::*;

/// Represents the 2-bit encoding for each nucleotide
#[repr(u8)]
enum NucleotideBits {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
}

/// A reusable structure holding common SIMD constants
#[repr(align(16))] // Ensure proper alignment for SIMD
struct SimdConstants {
    zeros: uint8x8_t,
    ones: uint8x8_t,
    twos: uint8x8_t,
    threes: uint8x8_t,
}

impl SimdConstants {
    #[inline(always)]
    unsafe fn new() -> Self {
        Self {
            zeros: vdup_n_u8(NucleotideBits::A as u8),
            ones: vdup_n_u8(NucleotideBits::C as u8),
            twos: vdup_n_u8(NucleotideBits::G as u8),
            threes: vdup_n_u8(NucleotideBits::T as u8),
        }
    }
}

/// Creates a bitmask for matching both upper and lowercase versions of a nucleotide
///
/// This function combines equality comparisons for both cases of a nucleotide
/// to create a single mask where matching positions are set to all 1s.
#[inline(always)]
unsafe fn create_dual_pattern_mask(chunk: uint8x8_t, upper: u8, lower: u8) -> uint8x8_t {
    vorr_u8(
        vceq_u8(chunk, vdup_n_u8(upper)),
        vceq_u8(chunk, vdup_n_u8(lower)),
    )
}

/// Sets the appropriate 2-bit patterns based on nucleotide masks
#[inline(always)]
unsafe fn set_bits(
    c_mask: uint8x8_t,
    g_mask: uint8x8_t,
    t_mask: uint8x8_t,
    constants: &SimdConstants,
) -> uint8x8_t {
    let mut result = constants.zeros;
    // Use BSL (bit select) to set appropriate values based on masks
    result = vbsl_u8(c_mask, constants.ones, result);
    result = vbsl_u8(g_mask, constants.twos, result);
    result = vbsl_u8(t_mask, constants.threes, result);
    result
}

/// Processes a single SIMD chunk of 8 nucleotides
#[inline(always)]
unsafe fn process_simd_chunk(chunk: uint8x8_t, constants: &SimdConstants) -> uint8x8_t {
    let (c_mask, g_mask, t_mask) = (
        create_dual_pattern_mask(chunk, b'C', b'c'),
        create_dual_pattern_mask(chunk, b'G', b'g'),
        create_dual_pattern_mask(chunk, b'T', b't'),
    );
    set_bits(c_mask, g_mask, t_mask, constants)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn as_2bit(seq: &[u8]) -> Result<u64, NucleotideError> {
    if seq.len() > 32 {
        return Err(NucleotideError::SequenceTooLong(seq.len()));
    }

    if seq.len() < 8 {
        return naive::as_2bit(seq);
    }

    // Pre-validate all bases using SIMD when possible
    if let Some(&invalid) = seq
        .iter()
        .find(|&&b| !matches!(b, b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't'))
    {
        return Err(NucleotideError::InvalidBase(invalid));
    }

    let mut packed = 0u64;
    let len = seq.len();
    let simd_len = len - (len % 8);

    unsafe {
        let constants = SimdConstants::new();

        // Process 8 nucleotides at a time using SIMD
        for chunk_idx in (0..simd_len).step_by(8) {
            let chunk = vld1_u8(seq[chunk_idx..].as_ptr());
            let result = process_simd_chunk(chunk, &constants);

            // Store and pack results
            let mut temp = [0u8; 8];
            vst1_u8(temp.as_mut_ptr(), result);

            for (i, &val) in temp.iter().enumerate() {
                packed |= (val as u64) << ((chunk_idx + i) * 2);
            }
        }

        // Handle remaining nucleotides
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

/// Encode 16 ASCII nucleotides (`A`, `C`, `G`, `T`) into a single `u32`.
///
/// Output layout: nt0 → bits 0‑1 … nt15 → bits 30‑31 (little‑endian).
#[inline(always)]
pub unsafe fn encode_16_nucleotides(nucs: uint8x16_t) -> u32 {
    // 1. ASCII → 2‑bit codes: code = ((b >> 1) ^ (b >> 2)) & 3
    let t1 = vshrq_n_u8(nucs, 1);
    let t2 = vshrq_n_u8(nucs, 2);
    let code = vandq_u8(veorq_u8(t1, t2), vdupq_n_u8(3));

    // 2. Pack two codes into one 4‑bit nibble
    let even = vuzp1q_u8(code, code); // c0, c2, …, c14
    let odd = vuzp2q_u8(code, code); // c1, c3, …, c15
    let nibbles = vorrq_u8(even, vshlq_n_u8(odd, 2));

    // 3. Pack two nibbles into one byte
    let even_b = vuzp1q_u8(nibbles, nibbles); // p0, p2, p4, p6
    let odd_b = vuzp2q_u8(nibbles, nibbles); // p1, p3, p5, p7
    let packed = vorrq_u8(even_b, vshlq_n_u8(odd_b, 4));

    // 4. Return the first lane (lower 32 bits)
    vgetq_lane_u32(vreinterpretq_u32_u8(packed), 0)
}

/// Return `true` if every byte in `v` is a valid nucleotide (case‑insensitive).
#[inline(always)]
unsafe fn valid_block(v: uint8x16_t) -> bool {
    let lower = vorrq_u8(v, vdupq_n_u8(0x20));
    let is_a = vceqq_u8(lower, vdupq_n_u8(b'a'));
    let is_c = vceqq_u8(lower, vdupq_n_u8(b'c'));
    let is_g = vceqq_u8(lower, vdupq_n_u8(b'g'));
    let is_t = vceqq_u8(lower, vdupq_n_u8(b't'));
    let ok = vorrq_u8(is_a, vorrq_u8(is_c, vorrq_u8(is_g, is_t)));
    vminvq_u8(ok) == 0xFF
}

/// Encode an arbitrary‑length ASCII slice into packed 2‑bit words (`u64`).
///
/// * 32 nt per word.
/// * `output` must be large enough; otherwise `Err(())` is returned.
/// * On any invalid byte the function zero‑fills `output` and returns `Err(())`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn encode_nucleotides_simd(
    input: &[u8],
    output: &mut [u64],
) -> Result<(), NucleotideError> {
    // If less than 32 nt, we can with the default method before SIMD overhead
    if input.len() < 32 {
        let tail = as_2bit(input)?;
        output[0] = tail;
        return Ok(());
    }

    output.fill(0);

    let mut ip = input.as_ptr();
    let mut left = input.len();
    let mut out = output.as_mut_ptr();

    // Vector loop: 32 nt → 1 u64
    while left >= 32 {
        let v0 = vld1q_u8(ip);
        let v1 = vld1q_u8(ip.add(16));
        if !valid_block(v0) || !valid_block(v1) {
            return Err(NucleotideError::InvalidBase(*ip));
        }
        *out = (encode_16_nucleotides(v0) as u64) | ((encode_16_nucleotides(v1) as u64) << 32);

        ip = ip.add(32);
        left -= 32;
        out = out.add(1);
    }

    // Scalar tail (≤ 31 nt)
    if left != 0 {
        let mut tail = 0u64;
        for i in 0..left {
            tail |= match *ip.add(i) | 0x20 {
                b'a' => 0u64,
                b'c' => 1u64,
                b'g' => 2u64,
                b't' => 3u64,
                _ => return Err(NucleotideError::InvalidBase(*ip.add(i))),
            } << (2 * i);
        }
        *out = tail;
    }
    Ok(())
}

#[inline(always)]
pub fn encode_internal(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), NucleotideError> {
    if sequence.len() < 32 {
        // Use the naive method for small sequences
        let bits = as_2bit(sequence)?;
        ebuf.push(bits);
        return Ok(());
    }

    // If the sequence is large enough and SIMD is supported, use SIMD acceleration
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe {
            // resize the buffer to fit the number of chunks
            let n_chunks = sequence.len().div_ceil(32);
            ebuf.resize(n_chunks, 0);
            encode_nucleotides_simd(sequence, ebuf)?;
        }
        return Ok(());
    }
    Ok(())
}
