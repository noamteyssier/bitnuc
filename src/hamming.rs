use std::ops::{BitAnd, BitOr, BitXor, Shr};

use fearless_simd::{Level, Simd, SimdBase, dispatch, u8x16, u8x32, u8x64, u64x2, u64x4, u64x8};

use crate::{BitnucError, resize};

// Masks for the lower and upper bit of each 2-bit group
const LOWER_BITS: u64 = 0x5555555555555555;
const UPPER_BITS: u64 = 0xAAAAAAAAAAAAAAAA;

pub fn hdist(u: &[u8], v: &[u8], len: usize) -> Result<usize, BitnucError> {
    if len.div_ceil(4) > u.len() {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: len.div_ceil(4),
            actual: u.len(),
        });
    }
    if len.div_ceil(4) > v.len() {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: len.div_ceil(4),
            actual: v.len(),
        });
    }
    if u.len() != v.len() {
        return Err(BitnucError::EncodingBuffersAreDifferentLengths {
            u_len: u.len(),
            v_len: v.len(),
        });
    }

    let level = Level::new();
    let dist = dispatch!(level, simd => hdist_simd(simd, u, v, len));

    Ok(dist)
}

fn hdist_simd<S: Simd>(simd: S, u: &[u8], v: &[u8], len: usize) -> usize {
    // number of bytes which are *fully* valid (i.e. contain 4 bases each)
    let packed_bytes = len / 4;

    let mut dist = 0;
    let mut i = 0;
    while i + 64 <= packed_bytes {
        hdist_lanes::<S, u8x64<S>>(
            simd,
            u8x64::from_slice(simd, &u[i..i + 64]),
            u8x64::from_slice(simd, &v[i..i + 64]),
            &mut dist,
        );
        i += 64;
    }

    while i + 32 <= packed_bytes {
        hdist_lanes::<S, u8x32<S>>(
            simd,
            u8x32::from_slice(simd, &u[i..i + 32]),
            u8x32::from_slice(simd, &v[i..i + 32]),
            &mut dist,
        );
        i += 32;
    }

    while i + 16 <= packed_bytes {
        hdist_lanes::<S, u8x16<S>>(
            simd,
            u8x16::from_slice(simd, &u[i..i + 16]),
            u8x16::from_slice(simd, &v[i..i + 16]),
            &mut dist,
        );
        i += 16;
    }

    while i < packed_bytes {
        let diff = u[i] ^ v[i];
        let lo = diff & 0x55;
        let hi = (diff & 0xAA) >> 1;
        let combined = lo | hi;

        dist += combined.count_ones() as usize;

        i += 1;
    }

    if i != u.len() && len % 4 != 0 {
        // Handle the last byte if it contains fewer than 4 bases
        let remaining_bases = len % 4;
        let mask = (1u8 << (remaining_bases * 2)) - 1;
        let diff = (u[i] ^ v[i]) & mask;

        let lo = diff & 0x55;
        let hi = (diff & 0xAA) >> 1;
        let combined = lo | hi;

        dist += combined.count_ones() as usize;
    }

    dist
}

fn hdist_lanes<S, V>(simd: S, u: V, v: V, d: &mut usize)
where
    S: Simd,
    V: SimdBase<S, Element = u8>
        + BitXor<Output = V>
        + BitAnd<Output = V>
        + Shr<u32, Output = V>
        + BitOr<Output = V>,
{
    let diff = v ^ u;

    // A base differs if either of its two bits differs
    let lower_diffs = diff & V::simd_from(simd, 0x55);
    let upper_diffs = (diff & V::simd_from(simd, 0xAA)) >> 1;
    let combined_diffs = lower_diffs | upper_diffs;

    // unfortunately need to run popcount on each lane separately
    // since fearless_simd doesn't have a popcount/lane implementation.
    for c in combined_diffs.as_slice() {
        *d += c.count_ones() as usize;
    }
}

/// Calculates the hamming distance between two 2-bit packed `u64` kmers
/// (as produced by [`as_2bit`](super::as_2bit)) of length `len` bases.
///
/// # Errors
///
/// Returns [`BitnucError::InvalidLength`] if `len` is greater than 32.
#[inline]
pub fn hdist_scalar(u: u64, v: u64, len: usize) -> Result<u32, BitnucError> {
    if len > 32 {
        return Err(BitnucError::InvalidLength(len));
    }

    if len == 0 || u == v {
        return Ok(0);
    }

    // Mask to the valid region (2 bits per base)
    let valid_bits = len * 2;
    let mask = if valid_bits == 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };

    let diff = (u ^ v) & mask;

    // A base differs if either of its two bits differs
    let lower_diffs = diff & LOWER_BITS;
    let upper_diffs = (diff & UPPER_BITS) >> 1;
    let combined_diffs = lower_diffs | upper_diffs;

    Ok(combined_diffs.count_ones())
}

/// Calculates the hamming distance between every pair of 2-bit packed `u64`
/// kmers in `items`, writing the results into `into`.
///
/// Distances are laid out in condensed row-major upper-triangle order: the
/// distance between `items[i]` and `items[j]` (for `i < j`) lands at index
/// `i * n - i * (i + 1) / 2 + (j - i - 1)`, matching scipy's `pdist`. The
/// first `n * (n - 1) / 2` elements of `into` are valid after the call; the
/// buffer is grown as needed but never shrunk.
///
/// # Errors
///
/// Returns [`BitnucError::InvalidLength`] if `len` is greater than 32.
pub fn hdist_pairwise(items: &[u64], len: usize, into: &mut [usize]) -> Result<(), BitnucError> {
    if len > 32 {
        return Err(BitnucError::InvalidLength(len));
    }

    let n_distances = items.len() * items.len().saturating_sub(1) / 2;
    if into.len() < n_distances {
        return Err(BitnucError::PairwiseDistanceBufferTooSmall {
            expected: n_distances,
            actual: into.len(),
        });
    }

    let level = Level::new();
    dispatch!(level, simd => hdist_pairwise_simd(simd, items, len, into));

    Ok(())
}

/// Calculates the hamming distance between every pair of 2-bit packed `u64`
/// kmers in `items`, writing the results into `into`. Grows the `into` buffer
/// as needed to hold the results, but never shrinks it.
///
/// Distances are laid out in condensed row-major upper-triangle order: the
/// distance between `items[i]` and `items[j]` (for `i < j`) lands at index
/// `i * n - i * (i + 1) / 2 + (j - i - 1)`, matching scipy's `pdist`. The
/// first `n * (n - 1) / 2` elements of `into` are valid after the call; the
/// buffer is grown as needed but never shrunk.
///
/// # Errors
///
/// Returns [`BitnucError::InvalidLength`] if `len` is greater than 32.
pub fn hdist_pairwise_resize(
    items: &[u64],
    len: usize,
    into: &mut Vec<usize>,
) -> Result<(), BitnucError> {
    if len > 32 {
        return Err(BitnucError::InvalidLength(len));
    }

    let n_distances = items.len() * items.len().saturating_sub(1) / 2;
    resize::resize(into, n_distances);

    let level = Level::new();
    dispatch!(level, simd => hdist_pairwise_simd(simd, items, len, into));

    Ok(())
}

fn hdist_pairwise_simd<S: Simd>(simd: S, items: &[u64], len: usize, into: &mut [usize]) {
    let valid_bits = len * 2;
    let mask = if valid_bits == 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };

    let mut out = 0;

    for (i, &u) in items.iter().enumerate() {
        let rest = &items[i + 1..];
        let mut j = 0;

        while j + 8 <= rest.len() {
            hamming_lanes::<S, u64x8<S>>(simd, u, &rest[j..j + 8], mask, &mut into[out..out + 8]);
            out += 8;
            j += 8;
        }

        while j + 4 <= rest.len() {
            hamming_lanes::<S, u64x4<S>>(simd, u, &rest[j..j + 4], mask, &mut into[out..out + 4]);
            out += 4;
            j += 4;
        }

        while j + 2 <= rest.len() {
            hamming_lanes::<S, u64x2<S>>(simd, u, &rest[j..j + 2], mask, &mut into[out..out + 2]);
            out += 2;
            j += 2;
        }

        while j < rest.len() {
            let diff = (u ^ rest[j]) & mask;

            let lower_diffs = diff & LOWER_BITS;
            let upper_diffs = (diff & UPPER_BITS) >> 1;
            let combined_diffs = lower_diffs | upper_diffs;

            into[out] = combined_diffs.count_ones() as usize;
            out += 1;
            j += 1;
        }
    }

    debug_assert_eq!(out, items.len() * items.len().saturating_sub(1) / 2);
}

#[inline(always)]
fn hamming_lanes<S, V>(simd: S, u: u64, v: &[u64], mask: u64, out: &mut [usize])
where
    S: Simd,
    V: SimdBase<S, Element = u64>
        + BitXor<Output = V>
        + BitAnd<Output = V>
        + Shr<u32, Output = V>
        + BitOr<Output = V>,
{
    let v = V::from_slice(simd, v);
    let diff = (v ^ V::simd_from(simd, u)) & V::simd_from(simd, mask);

    // A base differs if either of its two bits differs
    let lower_diffs = diff & V::simd_from(simd, LOWER_BITS);
    let upper_diffs = (diff & V::simd_from(simd, UPPER_BITS)) >> 1;
    let combined_diffs = lower_diffs | upper_diffs;

    // unfortunately need to run popcount on each lane separately
    // since fearless_simd doesn't have a popcount/lane implementation.
    for (idx, c) in combined_diffs.as_slice().iter().enumerate() {
        out[idx] = c.count_ones() as usize;
    }
}

#[cfg(test)]
mod hdist_packed {
    use std::collections::HashSet;

    use rand::{Rng, RngExt, make_rng, rngs::SmallRng, seq::IndexedRandom};

    use crate::encode_resize;

    use super::*;

    fn generate_sequence<R: Rng>(n: usize, rng: &mut R) -> Vec<u8> {
        (0..n).map(|_| *b"ACGT".choose(rng).unwrap()).collect()
    }

    fn edit_sequence<R: Rng>(seq: &mut [u8], n_errors: usize, rng: &mut R) {
        let len = seq.len();
        if len == 0 {
            return;
        }

        let mut seen_pos = HashSet::new();
        for _ in 0..(n_errors.min(len)) {
            let idx = {
                loop {
                    let idx = rng.random_range(0..len);
                    if !seen_pos.contains(&idx) {
                        seen_pos.insert(idx);
                        break idx;
                    }
                }
            };

            let new_base = {
                let cur_base = seq[idx];
                loop {
                    let new_base = *b"ACGT".choose(rng).unwrap();
                    if new_base != cur_base {
                        break new_base;
                    }
                }
            };

            seq[idx] = new_base;
        }
    }

    #[test]
    fn test_hdist_packed() {
        let mut rng: SmallRng = make_rng();

        let mut ebuf1 = Vec::new();
        let mut ebuf2 = Vec::new();
        for size in [1, 10, 100, 1_000, 10_000] {
            let seq1 = generate_sequence(size, &mut rng);

            for n_errors in [1, 2, 5, 10, 100] {
                if n_errors > size {
                    continue;
                }

                let mut seq2 = seq1.clone();
                edit_sequence(&mut seq2, n_errors, &mut rng);

                encode_resize(&seq1, &mut ebuf1);
                encode_resize(&seq2, &mut ebuf2);

                let dist = hdist(&ebuf1, &ebuf2, size).unwrap();
                assert_eq!(
                    dist, n_errors,
                    "Failed for size {size} with {n_errors} errors"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::as_2bit;

    #[test]
    fn test_hdist_scalar_validation() {
        assert!(hdist_scalar(0, 0, 33).is_err()); // Too long
        assert!(hdist_scalar(0, 0, 0).is_ok()); // Empty sequences
        assert!(hdist_scalar(0, 0, 32).is_ok()); // Max length
    }

    #[test]
    fn test_hdist_scalar_identical() {
        assert_eq!(hdist_scalar(0, 0, 1).unwrap(), 0);
        assert_eq!(hdist_scalar(0xFFFFFFFF, 0xFFFFFFFF, 16).unwrap(), 0);
        assert_eq!(
            hdist_scalar(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 32).unwrap(),
            0
        );
    }

    #[test]
    fn test_hdist_scalar_masks_beyond_len() {
        // Differences beyond `len` bases must not count
        let u = 0b0000u64;
        let v = 0b1100u64; // differs only at base 1
        assert_eq!(hdist_scalar(u, v, 1).unwrap(), 0);
        assert_eq!(hdist_scalar(u, v, 2).unwrap(), 1);
    }

    #[test]
    fn test_hdist_pairwise_matches_scalar() {
        // xorshift64 for deterministic pseudo-random packed kmers
        let mut state = 0x243F6A8885A308D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for n in 0..40usize {
            for len in [0, 1, 7, 16, 31, 32] {
                let items: Vec<u64> = (0..n).map(|_| next()).collect();
                let mut into = Vec::new();
                hdist_pairwise_resize(&items, len, &mut into).unwrap();

                let mut into_prebuilt = vec![0usize; n * n.saturating_sub(1) / 2];
                hdist_pairwise(&items, len, &mut into_prebuilt).unwrap();

                let n_distances = n * n.saturating_sub(1) / 2;
                assert_eq!(into.len(), n_distances);

                let mut k = 0;
                for i in 0..n {
                    for j in i + 1..n {
                        assert_eq!(
                            into[k],
                            hdist_scalar(items[i], items[j], len).unwrap() as usize,
                            "mismatch at pair ({i}, {j}) with n={n}, len={len}"
                        );

                        assert_eq!(
                            into[k], into_prebuilt[k],
                            "mismatch between resize and prebuilt at pair ({i}, {j}) with n={n}, len={len}"
                        );
                        k += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn test_hdist_pairwise_validation() {
        let mut into = Vec::new();
        assert!(hdist_pairwise_resize(&[0, 1], 33, &mut into).is_err());
        assert!(into.is_empty()); // buffer untouched on error

        assert!(hdist_pairwise_resize(&[], 4, &mut into).is_ok());
        assert!(hdist_pairwise_resize(&[0], 4, &mut into).is_ok());
    }

    #[test]
    fn test_hdist_pairwise_oversized_buffer() {
        // A previously-larger buffer is reused without shrinking; the valid
        // distances occupy the prefix.
        let mut into = vec![usize::MAX; 100];
        hdist_pairwise_resize(&[0b00, 0b01, 0b11], 1, &mut into).unwrap();
        assert_eq!(into.len(), 100);
        assert_eq!(&into[..3], &[1, 1, 1]);
        assert_eq!(into[3], usize::MAX);
    }

    #[test]
    fn test_hdist_scalar_full_sequences() {
        let test_cases: Vec<(&[u8], &[u8], u32)> = vec![
            (b"AAAA", b"AAAA", 0),
            (b"AAAA", b"AAAT", 1),
            (b"AAAA", b"AATT", 2),
            (b"AAAA", b"ATTT", 3),
            (b"AAAA", b"TTTT", 4),
            (b"ACTGACTG", b"TGCATGCA", 8),
        ];

        for (seq1, seq2, expected) in test_cases {
            let u = as_2bit(seq1).unwrap();
            let v = as_2bit(seq2).unwrap();
            assert_eq!(
                hdist_scalar(u, v, seq1.len()).unwrap(),
                expected,
                "Failed for sequences {:?} and {:?}",
                std::str::from_utf8(seq1).unwrap(),
                std::str::from_utf8(seq2).unwrap()
            );
        }
    }
}
