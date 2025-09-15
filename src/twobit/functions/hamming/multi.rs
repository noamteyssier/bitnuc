#[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
use std::arch::x86_64::*;

use crate::NucleotideError;

use super::hdist_scalar;

#[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
#[inline]
unsafe fn hdist_multi_avx2(ebuf1: &[u64], ebuf2: &[u64], full_chunks: usize) -> u32 {
    let mut total = 0u32;

    // Process 4 chunks (128 bases) at a time using AVX2
    let quad_chunks = full_chunks / 4;

    // Constants for bit manipulation
    let lower_bits = _mm256_set1_epi64x(0x5555555555555555u64 as i64);
    let upper_bits = _mm256_set1_epi64x(0xAAAAAAAAAAAAAAAAu64 as i64);

    for i in 0..quad_chunks {
        // Load 4 chunks (256 bits) from each buffer
        let u_ptr = ebuf1.as_ptr().add(i * 4) as *const i64;
        let v_ptr = ebuf2.as_ptr().add(i * 4) as *const i64;

        let u_vec = _mm256_loadu_si256(u_ptr as *const __m256i);
        let v_vec = _mm256_loadu_si256(v_ptr as *const __m256i);

        // XOR to find differences
        let diff = _mm256_xor_si256(u_vec, v_vec);

        // Skip if no differences
        if _mm256_testz_si256(diff, diff) == 1 {
            continue;
        }

        // Get differences in lower and upper bits
        let lower_diffs = _mm256_and_si256(diff, lower_bits);
        let upper_diffs = _mm256_srli_epi64(_mm256_and_si256(diff, upper_bits), 1);

        // Combine differences
        let combined = _mm256_or_si256(lower_diffs, upper_diffs);

        // Extract and count bits from each 64-bit lane with fixed indices
        let lane0 = _mm256_extract_epi64(combined, 0) as u64;
        let lane1 = _mm256_extract_epi64(combined, 1) as u64;
        let lane2 = _mm256_extract_epi64(combined, 2) as u64;
        let lane3 = _mm256_extract_epi64(combined, 3) as u64;

        total += lane0.count_ones() + lane1.count_ones() + lane2.count_ones() + lane3.count_ones();
    }

    // Handle remaining full chunks
    let remaining_start = quad_chunks * 4;
    for (sub_ebuf1, sub_ebuf2) in ebuf1
        .iter()
        .zip(ebuf2.iter())
        .skip(remaining_start)
        .take(full_chunks - remaining_start)
    {
        let chunk_dist = hdist_scalar(*sub_ebuf1, *sub_ebuf2, 32).unwrap_or(0);
        total += chunk_dist;
    }

    total
}

#[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
#[inline]
unsafe fn hdist_multi_neon(ebuf1: &[u64], ebuf2: &[u64], full_chunks: usize) -> u32 {
    let mut total = 0u32;

    // Process 2 chunks (64 bases) at a time using NEON
    let dual_chunks = full_chunks / 2;

    // Constants for bit manipulation
    let lower_bits = vdupq_n_u64(0x5555555555555555);
    let upper_bits = vdupq_n_u64(0xAAAAAAAAAAAAAAAA);

    for i in 0..dual_chunks {
        // Load 2 chunks (128 bits) from each buffer
        let u_ptr = ebuf1.as_ptr().add(i * 2);
        let v_ptr = ebuf2.as_ptr().add(i * 2);

        let u_vec = vld1q_u64(u_ptr);
        let v_vec = vld1q_u64(v_ptr);

        // XOR to find differences
        let diff = veorq_u64(u_vec, v_vec);

        // Skip if no differences
        if vgetq_lane_u64::<0>(diff) == 0 && vgetq_lane_u64::<1>(diff) == 0 {
            continue;
        }

        // Get differences in lower and upper bits
        let lower_diffs = vandq_u64(diff, lower_bits);
        let upper_diffs = vshrq_n_u64(vandq_u64(diff, upper_bits), 1);

        // Combine differences
        let combined = vorrq_u64(lower_diffs, upper_diffs);

        // Extract and count bits from each 64-bit lane
        total += vgetq_lane_u64::<0>(combined).count_ones();
        total += vgetq_lane_u64::<1>(combined).count_ones();
    }

    // Handle remaining full chunks
    let remaining_start = dual_chunks * 2;
    for i in remaining_start..full_chunks {
        let chunk_dist = hdist_scalar(ebuf1[i], ebuf2[i], 32).unwrap_or(0);
        total += chunk_dist;
    }

    total
}

/// Calculate hamming distance between two 2-bit encoded sequences
/// Each u64 contains up to 32 bases (2 bits per base)
#[inline]
pub fn hdist(ebuf1: &[u64], ebuf2: &[u64], n_bases: usize) -> Result<u32, NucleotideError> {
    // Validate buffer sizes
    let expected_chunks = n_bases.div_ceil(32);
    if ebuf1.len() < expected_chunks || ebuf2.len() < expected_chunks {
        return Err(NucleotideError::InvalidLength(n_bases));
    }

    let full_chunks = n_bases / 32;
    let mut total_dist = 0u32;

    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    unsafe {
        if std::arch::is_aarch64_feature_detected!("neon") && full_chunks >= 2 {
            total_dist = hdist_multi_neon(ebuf1, ebuf2, full_chunks);
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    unsafe {
        if is_x86_feature_detected!("avx2") && full_chunks >= 4 {
            total_dist = hdist_multi_avx2(ebuf1, ebuf2, full_chunks);
        }
    }

    // If SIMD is not available, use the naive implementation
    if total_dist == 0 && full_chunks > 0 {
        for (scal_ebuf1, scal_ebuf2) in ebuf1.iter().zip(ebuf2.iter()).take(full_chunks) {
            total_dist += hdist_scalar(*scal_ebuf1, *scal_ebuf2, 32)?;
        }
    }

    // Handle remaining bases
    let remaining_bases = n_bases % 32;
    if remaining_bases > 0 {
        total_dist += hdist_scalar(ebuf1[full_chunks], ebuf2[full_chunks], remaining_bases)?;
    }

    Ok(total_dist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twobit::encode_alloc;

    #[test]
    fn test_hdist_multi_validation() {
        // Test with buffers that are too small
        let buf1 = vec![0u64; 1];
        let buf2 = vec![0u64; 1];
        assert!(hdist(&buf1, &buf2, 64).is_err());
    }

    #[test]
    fn test_hdist_multi_identical() {
        let seq = b"ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"; // 64 bases
        let buf = encode_alloc(seq).unwrap();
        assert_eq!(hdist(&buf, &buf, seq.len()), Ok(0));
    }

    #[test]
    fn test_hdist_multi_full_chunks() {
        // Multiple of 128 bases for AVX2 (4 chunks) and 64 bases for NEON (2 chunks)
        let seq1 = vec![b'A'; 128];
        let seq2 = vec![b'T'; 128];
        let buf1 = encode_alloc(&seq1).unwrap();
        let buf2 = encode_alloc(&seq2).unwrap();
        assert_eq!(hdist(&buf1, &buf2, seq1.len()), Ok(128));
    }

    #[test]
    fn test_hdist_multi_various_lengths() {
        // Test lengths that exercise different SIMD paths
        for len in 1..=256 {
            let seq1 = vec![b'A'; len];
            let seq2 = vec![b'T'; len];
            let buf1 = encode_alloc(&seq1).unwrap();
            let buf2 = encode_alloc(&seq2).unwrap();
            assert_eq!(
                hdist(&buf1, &buf2, len),
                Ok(len as u32),
                "Failed for length {}",
                len
            );
        }
    }
}
