use std::ops::BitOr;

use fearless_simd::{Level, Simd, dispatch, prelude::*, u8x16, u8x32, u8x64};

/// Identifies all ambiguous bases (non-ACGTacgt) and appends
/// their positions to a user-supplied buffer.
pub fn ambiguous_bases(seq: &[u8], pos: &mut Vec<usize>) {
    let level = Level::new();
    dispatch!(level, simd => ambiguous_bases_inner(simd, seq, pos))
}

#[inline(always)]
fn ambiguous_bases_inner<S: Simd>(simd: S, seq: &[u8], pos: &mut Vec<usize>) {
    let mut i = 0; // current index in seq
    while i + 64 <= seq.len() {
        find_in_lane::<S, u8x64<S>>(simd, &seq[i..i + 64], pos, i);
        i += 64
    }
    if i + 32 <= seq.len() {
        find_in_lane::<S, u8x32<S>>(simd, &seq[i..i + 32], pos, i);
        i += 32
    }
    if i + 16 <= seq.len() {
        find_in_lane::<S, u8x16<S>>(simd, &seq[i..i + 16], pos, i);
        i += 16
    }
    scalar_fallback(i, seq, pos);
}

#[inline(always)]
fn find_in_lane<S, V>(simd: S, chunk: &[u8], pos: &mut Vec<usize>, offset: usize)
where
    S: Simd,
    V: SimdBase<S, Element = u8> + BitOr<Output = V>,
{
    let vec = V::from_slice(simd, chunk);
    let lowercase = vec | V::splat(simd, 0x20);

    let is_canonical = lowercase.simd_eq(b'a')
        | lowercase.simd_eq(b'c')
        | lowercase.simd_eq(b'g')
        | lowercase.simd_eq(b't');

    let bits = (!is_canonical).to_bitmask();
    write_bitmask(offset, bits, pos);
}

#[inline(always)]
fn write_bitmask(offset: usize, mut bits: u64, pos: &mut Vec<usize>) {
    if bits == 0 {
        return; // skip popcnt/extend overhead on clean chunks
    }
    let cnt = bits.count_ones(); // popcnt to get number of elements
    pos.extend(
        (0..cnt) // reserve elements upfront
            .map(|_| {
                let p = offset + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                p
            }),
    );
}

#[inline(always)]
fn scalar_fallback(offset: usize, seq: &[u8], pos: &mut Vec<usize>) {
    for (idx, b) in seq[offset..].iter().enumerate() {
        if !matches!(*b | 0x20, b'a' | b'c' | b'g' | b't') {
            pos.push(offset + idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn is_canonical(b: u8) -> bool {
        matches!(b, b'A' | b'C' | b'G' | b'T' | b'a' | b'c' | b'g' | b't')
    }

    fn scalar_reference(seq: &[u8]) -> Vec<usize> {
        seq.iter()
            .enumerate()
            .filter(|&(_, &b)| !is_canonical(b))
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn test_empty() {
        let mut pos = Vec::new();
        ambiguous_bases(b"", &mut pos);
        assert!(pos.is_empty());
    }

    #[test]
    fn test_all_canonical() {
        let seq: Vec<u8> = b"ACGTacgt".repeat(20); // 160 bases: SIMD + tail
        let mut pos = Vec::new();
        ambiguous_bases(&seq, &mut pos);
        assert!(pos.is_empty());
    }

    #[test]
    fn test_known_positions() {
        let mut seq: Vec<u8> = b"ACGT".repeat(32); // 128 bases
        seq[0] = b'N';
        seq[63] = b'n';
        seq[64] = b'-';
        seq[127] = b'U';
        let mut pos = Vec::new();
        ambiguous_bases(&seq, &mut pos);
        assert_eq!(pos, vec![0, 63, 64, 127]);
    }

    #[test]
    fn test_tail_only() {
        let mut pos = Vec::new();
        ambiguous_bases(b"ACGNT", &mut pos);
        assert_eq!(pos, vec![3]);
    }

    #[test]
    fn test_appends_without_clearing() {
        let mut pos = vec![999];
        ambiguous_bases(b"N", &mut pos);
        assert_eq!(pos, vec![999, 0]);
    }

    #[test]
    fn test_all_bytes_against_scalar() {
        // Every byte value, at every lane offset within a chunk
        let seq: Vec<u8> = (0..=255u8).cycle().take(64 * 8 + 17).collect();
        let mut pos = Vec::new();
        ambiguous_bases(&seq, &mut pos);
        assert_eq!(pos, scalar_reference(&seq));
    }

    #[test]
    fn test_length_sweep_against_scalar() {
        // Cross every 64/32/16/scalar branch combination, with hits in each
        let base: Vec<u8> = (0..=255u8).cycle().take(256).collect();
        for len in 0..=160 {
            let seq: Vec<u8> = base.iter().cycle().take(len).copied().collect();
            let mut pos = Vec::new();
            ambiguous_bases(&seq, &mut pos);
            assert_eq!(pos, scalar_reference(&seq), "failed at len={len}");
        }
    }
}
