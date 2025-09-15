use std::arch::aarch64::*;

use super::naive;
use crate::NucleotideError;

#[repr(u8)]
enum NucleotideBits4 {
    A = 0b0000,
    C = 0b0001,
    G = 0b0010,
    T = 0b0011,
    N = 0b1111, // Using 0b1111 for ambiguous bases
}

/// A reusable structure holding common SIMD constants
#[repr(align(16))]
struct SimdConstants4 {
    zeros: uint8x8_t,
    ones: uint8x8_t,
    twos: uint8x8_t,
    threes: uint8x8_t,
    ns: uint8x8_t, // For N bases
}

impl SimdConstants4 {
    #[inline(always)]
    unsafe fn new() -> Self {
        Self {
            zeros: vdup_n_u8(NucleotideBits4::A as u8),
            ones: vdup_n_u8(NucleotideBits4::C as u8),
            twos: vdup_n_u8(NucleotideBits4::G as u8),
            threes: vdup_n_u8(NucleotideBits4::T as u8),
            ns: vdup_n_u8(NucleotideBits4::N as u8),
        }
    }
}

/// Creates a bitmask for matching both upper and lowercase versions of a nucleotide
#[inline(always)]
unsafe fn create_dual_pattern_mask(chunk: uint8x8_t, upper: u8, lower: u8) -> uint8x8_t {
    vorr_u8(
        vceq_u8(chunk, vdup_n_u8(upper)),
        vceq_u8(chunk, vdup_n_u8(lower)),
    )
}

/// Creates a mask for all ambiguous IUPAC codes
#[inline(always)]
unsafe fn create_ambiguous_mask(chunk: uint8x8_t) -> uint8x8_t {
    // Create masks for all ambiguous IUPAC codes
    let n_mask = create_dual_pattern_mask(chunk, b'N', b'n');
    let r_mask = create_dual_pattern_mask(chunk, b'R', b'r'); // A or G
    let y_mask = create_dual_pattern_mask(chunk, b'Y', b'y'); // C or T
    let s_mask = create_dual_pattern_mask(chunk, b'S', b's'); // G or C
    let w_mask = create_dual_pattern_mask(chunk, b'W', b'w'); // A or T
    let k_mask = create_dual_pattern_mask(chunk, b'K', b'k'); // G or T
    let m_mask = create_dual_pattern_mask(chunk, b'M', b'm'); // A or C
    let b_mask = create_dual_pattern_mask(chunk, b'B', b'b'); // C,G,T
    let d_mask = create_dual_pattern_mask(chunk, b'D', b'd'); // A,G,T
    let h_mask = create_dual_pattern_mask(chunk, b'H', b'h'); // A,C,T
    let v_mask = create_dual_pattern_mask(chunk, b'V', b'v'); // A,C,G

    // Combine all ambiguous masks
    vorr_u8(
        n_mask,
        vorr_u8(
            r_mask,
            vorr_u8(
                y_mask,
                vorr_u8(
                    s_mask,
                    vorr_u8(
                        w_mask,
                        vorr_u8(
                            k_mask,
                            vorr_u8(
                                m_mask,
                                vorr_u8(b_mask, vorr_u8(d_mask, vorr_u8(h_mask, v_mask))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
}

/// Sets the appropriate 4-bit patterns based on nucleotide masks
#[inline(always)]
unsafe fn set_bits_4bit(
    c_mask: uint8x8_t,
    g_mask: uint8x8_t,
    t_mask: uint8x8_t,
    n_mask: uint8x8_t,
    constants: &SimdConstants4,
) -> uint8x8_t {
    let mut result = constants.zeros;

    // Use BSL (bit select) to set appropriate values based on masks
    result = vbsl_u8(c_mask, constants.ones, result);
    result = vbsl_u8(g_mask, constants.twos, result);
    result = vbsl_u8(t_mask, constants.threes, result);
    result = vbsl_u8(n_mask, constants.ns, result);

    result
}

/// Processes a single SIMD chunk of 8 nucleotides
#[inline(always)]
unsafe fn process_simd_chunk_4bit(chunk: uint8x8_t, constants: &SimdConstants4) -> uint8x8_t {
    let c_mask = create_dual_pattern_mask(chunk, b'C', b'c');
    let g_mask = create_dual_pattern_mask(chunk, b'G', b'g');
    let t_mask = create_dual_pattern_mask(chunk, b'T', b't');
    let n_mask = create_ambiguous_mask(chunk);

    set_bits_4bit(c_mask, g_mask, t_mask, n_mask, constants)
}

pub fn as_4bit(seq: &[u8]) -> Result<u64, NucleotideError> {
    if seq.len() > 16 {
        // 16 bases * 4 bits = 64 bits
        return Err(NucleotideError::SequenceTooLong(seq.len()));
    }

    // Use naive implementation for small sequences
    if seq.len() < 8 {
        return naive::as_4bit(seq);
    }

    // Validate all bases
    if let Some(&invalid) = seq.iter().find(|&&b| !is_valid_nucleotide_4bit(b)) {
        return Err(NucleotideError::InvalidBase(invalid));
    }

    let mut packed = 0u64;
    let len = seq.len();
    let simd_len = len - (len % 8); // Process 8 bases at a time

    unsafe {
        let constants = SimdConstants4::new();

        for chunk_idx in (0..simd_len).step_by(8) {
            let chunk = vld1_u8(seq[chunk_idx..].as_ptr());
            let result = process_simd_chunk_4bit(chunk, &constants);

            // Store and pack results
            let mut temp = [0u8; 8];
            vst1_u8(temp.as_mut_ptr(), result);

            // Pack 4-bit values
            for (i, &val) in temp.iter().enumerate() {
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

/// Encode 8 ASCII nucleotides into a single u32 using 4-bit encoding
#[inline(always)]
pub unsafe fn encode_8_nucleotides_4bit(nucs: uint8x8_t) -> u32 {
    let constants = SimdConstants4::new();

    // Process the nucleotides to get 4-bit codes
    let result = process_simd_chunk_4bit(nucs, &constants);

    // Extract the results and pack them into a u32
    let mut temp = [0u8; 8];
    vst1_u8(temp.as_mut_ptr(), result);

    let mut packed = 0u32;
    for (i, &val) in temp.iter().enumerate() {
        packed |= (val as u32) << (i * 4);
    }

    packed
}

/// Return true if every byte in v is a valid nucleotide (case-insensitive) or ambiguous code
#[inline(always)]
unsafe fn valid_block_4bit(v: uint8x16_t) -> bool {
    // Create masks for all valid nucleotides and ambiguous codes
    let lower = vorrq_u8(v, vdupq_n_u8(0x20)); // Convert to lowercase

    let is_a = vceqq_u8(lower, vdupq_n_u8(b'a'));
    let is_c = vceqq_u8(lower, vdupq_n_u8(b'c'));
    let is_g = vceqq_u8(lower, vdupq_n_u8(b'g'));
    let is_t = vceqq_u8(lower, vdupq_n_u8(b't'));
    let is_n = vceqq_u8(lower, vdupq_n_u8(b'n'));
    let is_r = vceqq_u8(lower, vdupq_n_u8(b'r'));
    let is_y = vceqq_u8(lower, vdupq_n_u8(b'y'));
    let is_s = vceqq_u8(lower, vdupq_n_u8(b's'));
    let is_w = vceqq_u8(lower, vdupq_n_u8(b'w'));
    let is_k = vceqq_u8(lower, vdupq_n_u8(b'k'));
    let is_m = vceqq_u8(lower, vdupq_n_u8(b'm'));
    let is_b = vceqq_u8(lower, vdupq_n_u8(b'b'));
    let is_d = vceqq_u8(lower, vdupq_n_u8(b'd'));
    let is_h = vceqq_u8(lower, vdupq_n_u8(b'h'));
    let is_v = vceqq_u8(lower, vdupq_n_u8(b'v'));

    let ok = vorrq_u8(
        is_a,
        vorrq_u8(
            is_c,
            vorrq_u8(
                is_g,
                vorrq_u8(
                    is_t,
                    vorrq_u8(
                        is_n,
                        vorrq_u8(
                            is_r,
                            vorrq_u8(
                                is_y,
                                vorrq_u8(
                                    is_s,
                                    vorrq_u8(
                                        is_w,
                                        vorrq_u8(
                                            is_k,
                                            vorrq_u8(
                                                is_m,
                                                vorrq_u8(
                                                    is_b,
                                                    vorrq_u8(is_d, vorrq_u8(is_h, is_v)),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );

    vminvq_u8(ok) == 0xFF
}

/// Encode an arbitrary-length ASCII slice into packed 4-bit words (u64)
/// 16 nt per word
#[inline(always)]
pub unsafe fn encode_nucleotides_simd_4bit(
    input: &[u8],
    output: &mut [u64],
) -> Result<(), NucleotideError> {
    // If less than 16 nt, use the default method before SIMD overhead
    if input.len() < 16 {
        let tail = as_4bit(input)?;
        output[0] = tail;
        return Ok(());
    }

    output.fill(0);

    let mut ip = input.as_ptr();
    let mut left = input.len();
    let mut out = output.as_mut_ptr();

    // Vector loop: 16 nt → 1 u64
    while left >= 16 {
        let v = vld1q_u8(ip);
        if !valid_block_4bit(v) {
            return Err(NucleotideError::InvalidBase(*ip));
        }

        // Split into two 8-element chunks
        let low = vget_low_u8(v);
        let high = vget_high_u8(v);

        let low_packed = encode_8_nucleotides_4bit(low);
        let high_packed = encode_8_nucleotides_4bit(high);

        *out = (low_packed as u64) | ((high_packed as u64) << 32);

        ip = ip.add(16);
        left -= 16;
        out = out.add(1);
    }

    // Scalar tail (≤ 15 nt)
    if left != 0 {
        let mut tail = 0u64;
        for i in 0..left {
            tail |= match *ip.add(i) | 0x20 {
                b'a' => 0u64,
                b'c' => 1u64,
                b'g' => 2u64,
                b't' => 3u64,
                // All ambiguous bases become N (0b1111 = 15)
                b'n' | b'r' | b'y' | b's' | b'w' | b'k' | b'm' | b'b' | b'd' | b'h' | b'v' => 15u64,
                _ => return Err(NucleotideError::InvalidBase(*ip.add(i))),
            } << (4 * i);
        }
        *out = tail;
    }
    Ok(())
}

pub fn encode_internal(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), NucleotideError> {
    if sequence.len() < 16 {
        // Use the naive method for small sequences
        let bits = naive::as_4bit(sequence)?;
        ebuf.clear();
        ebuf.push(bits);
        return Ok(());
    }

    // If the sequence is large enough and SIMD is supported, use SIMD acceleration
    unsafe {
        // resize the buffer to fit the number of chunks (16 bases per u64 with 4-bit encoding)
        let n_chunks = sequence.len().div_ceil(16);
        ebuf.resize(n_chunks, 0);
        encode_nucleotides_simd_4bit(sequence, ebuf)?;
    }
    Ok(())
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
            Err(NucleotideError::SequenceTooLong(17))
        ));
    }

    #[test]
    fn test_4bit_case_insensitive() {
        assert_eq!(as_4bit(b"acgt").unwrap(), as_4bit(b"ACGT").unwrap());
        assert_eq!(as_4bit(b"nrys").unwrap(), as_4bit(b"NRYS").unwrap());
    }
}
