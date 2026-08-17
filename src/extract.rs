use std::ops::{BitOr, Range, Shl, Shr};

use fearless_simd::{Level, Simd, SimdBase, dispatch, u8x16, u8x32, u8x64};

use crate::{BitnucError, resize};

/// extracts the bits belonging to the given basepair `range` from `packed` into `into`
///
/// Note: This is moving the two-bit encoded bits and *not* converting back to ASCII
pub fn extract(packed: &[u8], range: Range<usize>, into: &mut [u8]) -> Result<(), BitnucError> {
    // Range extends beyond what is possible in the packed buffer
    if range.end.div_ceil(4) > packed.len() {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: range.end.div_ceil(4),
            actual: packed.len(),
        });
    }

    // Range span is larger than the available buffer
    if range.len().div_ceil(4) > into.len() {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: range.len().div_ceil(4),
            actual: into.len(),
        });
    }

    // set the range of the output buffer to the exact size needed for the range
    let into = &mut into[..range.len().div_ceil(4)];

    // no-op
    if range.is_empty() {
        return Ok(());
    }

    let level = Level::new();
    dispatch!(
        level,
        simd =>  extract_inner(simd, packed, range, into)
    );

    Ok(())
}

pub fn extract_resize(
    packed: &[u8],
    range: Range<usize>,
    into: &mut Vec<u8>,
) -> Result<(), BitnucError> {
    // no-op (without resizing)
    if range.is_empty() {
        return Ok(());
    }

    // Range extends beyond what is possible in the packed buffer
    if range.end.div_ceil(4) > packed.len() {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: range.end.div_ceil(4),
            actual: packed.len(),
        });
    }
    resize::resize(into, range.len().div_ceil(4));

    let level = Level::new();
    dispatch!(
        level,
        simd =>  extract_inner(simd, packed, range, into)
    );
    Ok(())
}

#[inline(always)]
fn extract_inner<S: Simd>(simd: S, packed: &[u8], range: Range<usize>, into: &mut [u8]) {
    let start_word = range.start / 4;
    let end_word = (range.end - 1) / 4;
    let packed_subset = &packed[start_word..=end_word];

    // Phase offset into the packed word (u32 for SIMD shift operations)
    let offbit = 2 * (range.start % 4) as u32;

    if offbit == 0 {
        into[..packed_subset.len()].copy_from_slice(packed_subset);
    } else {
        let mut idx = 0;

        while idx + 65 <= into.len() {
            extract_lanes::<S, u8x64<S>>(simd, packed_subset, offbit, idx, 64, into);
            idx += 64;
        }

        if idx + 33 <= into.len() {
            extract_lanes::<S, u8x32<S>>(simd, packed_subset, offbit, idx, 32, into);
            idx += 32;
        }

        if idx + 17 <= into.len() {
            extract_lanes::<S, u8x16<S>>(simd, packed_subset, offbit, idx, 16, into);
            idx += 16;
        }

        // scalar fallback
        while idx < into.len() {
            let w_i = packed_subset[idx];
            let w_i1 = packed_subset.get(idx + 1).copied().unwrap_or(0);
            let w_o = (w_i >> offbit) | (w_i1 << (8 - offbit));
            into[idx] = w_o;
            idx += 1
        }
    }

    // zero out the unused high bits of the last packed word
    if !range.len().is_multiple_of(4) {
        let used_bits = 2 * (range.len() % 4);
        into[into.len() - 1] &= (1u8 << used_bits) - 1; // bitmask out unused bits
    }
}

/// Generic implementation of the core extraction logic.
///
/// `packed` is the packed sequence to extract from.
/// `offbit` is the number of bits to shift left/right.
/// `offset` is the offset into `packed` to start extracting from.
/// `size` is the number of packed bytes to extract.
/// `into` is the output buffer.
///
/// Note: `packed` must be at least `offset + size + 1` bytes long.
#[inline(always)]
fn extract_lanes<S, V>(
    simd: S,
    packed: &[u8],
    offbit: u32,
    offset: usize,
    size: usize,
    into: &mut [u8],
) where
    S: Simd,
    V: SimdBase<S, Element = u8> + Shl<u32, Output = V> + Shr<u32, Output = V> + BitOr<Output = V>,
{
    let w_i = V::from_slice(simd, &packed[offset..offset + size]);
    let w_i1 = V::from_slice(simd, &packed[offset + 1..offset + size + 1]);
    let w_o = (w_i >> offbit) | (w_i1 << (8 - offbit));
    into[offset..offset + size].copy_from_slice(w_o.as_slice());
}

#[cfg(test)]
mod testing {
    use rand::{Rng, RngExt, rngs::SmallRng};

    use super::{extract, extract_resize};
    use crate::{decode_resize, encode_resize};

    fn gen_seq<R: Rng>(n: usize, rng: &mut R) -> Vec<u8> {
        (0..n).map(|_| b"ACGT"[rng.random_range(0..4)]).collect()
    }

    #[test]
    fn test_extract() {
        let mut rng: SmallRng = rand::make_rng();

        let n_sequences = 100;
        let max_len = 1000;

        let mut nproc = 0;
        while nproc < n_sequences {
            let seq_len = rng.random_range(0..max_len);
            let endpoint = rng.random_range(0..=seq_len);
            let startpoint = rng.random_range(0..=endpoint);
            let range = startpoint..endpoint;

            // generate a random sequence
            let seq = gen_seq(seq_len, &mut rng);

            // encode the sequence
            let mut packed = Vec::default();
            encode_resize(&seq, &mut packed);

            // extract a sub-sequence within a range
            let mut extracted = Vec::default();
            extract_resize(&packed, range.clone(), &mut extracted).unwrap();

            // decode the extracted sub-sequence
            let mut decoded = Vec::default();
            decode_resize(&extracted, range.len(), &mut decoded).unwrap();

            // verify the decoded sequence matches the original
            assert_eq!(decoded, seq[range.clone()]);

            // decode the extracted sub-sequence to the ceiling to check for leaked bits
            //
            // if the span isn't a multiple of 4 there will be a remainder that must be zeroed out
            if range.len() % 4 != 0 {
                let mut decoded = Vec::default();
                decode_resize(&extracted, range.len().next_multiple_of(4), &mut decoded).unwrap();

                // verify the decoded sequence matches the original up to the range
                assert_eq!(decoded[..range.len()], seq[range.clone()]);

                // verify the remaining bits are zero (a.k.a. 'A')
                assert_eq!(&decoded[range.len()..], &vec![b'A'; 4 - range.len() % 4]);
            }

            nproc += 1;
        }
    }

    #[test]
    fn test_extract_oversized_buffer() {
        let mut rng: SmallRng = rand::make_rng();

        const BUF_LEN: usize = 32;
        let mut buf = [0u8; BUF_LEN];

        for seq_len in [1usize, 2, 3, 5, 8, 9, 17, 65, 128, 130] {
            let seq = gen_seq(seq_len, &mut rng);
            let mut packed = Vec::default();
            encode_resize(&seq, &mut packed);

            for start in 0..=seq_len {
                for end in start..=seq_len.min(start + BUF_LEN * 4) {
                    let range = start..end;
                    let out_len = range.len().div_ceil(4);

                    // poison the buffer so stale bits surface in the tail-masking check
                    buf.fill(0xFF);
                    extract(&packed, range.clone(), &mut buf).unwrap();

                    let mut decoded = Vec::default();
                    decode_resize(&buf[..out_len], range.len(), &mut decoded).unwrap();

                    assert_eq!(
                        decoded,
                        seq[range.clone()],
                        "mismatch at seq_len={seq_len}, range={range:?}"
                    );

                    // unused high bits of the last written byte must be zero
                    if range.len() % 4 != 0 {
                        let used_bits = 2 * (range.len() % 4);
                        let mask = !((1u8 << used_bits) - 1);
                        let last = buf[out_len - 1];
                        assert_eq!(
                            last & mask,
                            0,
                            "unused bits not zero at seq_len={seq_len}, range={range:?}, last byte={last:#010b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_extract_exhaustive() {
        // exhaustive over every (start, end) pair for a handful of seq lengths,
        // covering all four residues of start%4 and end%4 crossed together
        let mut rng: SmallRng = rand::make_rng();

        for seq_len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 65, 130] {
            let seq = gen_seq(seq_len, &mut rng);
            let mut packed = Vec::default();
            encode_resize(&seq, &mut packed);

            for start in 0..=seq_len {
                for end in start..=seq_len {
                    let range = start..end;
                    let mut extracted = Vec::default();
                    extract_resize(&packed, range.clone(), &mut extracted).unwrap();

                    let mut decoded = Vec::default();
                    decode_resize(&extracted, range.len(), &mut decoded).unwrap();

                    assert_eq!(
                        decoded,
                        seq[range.clone()],
                        "mismatch at seq_len={seq_len}, range={range:?}"
                    );

                    // unused high bits of the last packed byte must be zero
                    if range.len() % 4 != 0 {
                        let used_bits = 2 * (range.len() % 4);
                        let mask = !((1u8 << used_bits) - 1);
                        let last = *extracted.last().unwrap();
                        assert_eq!(
                            last & mask,
                            0,
                            "unused bits not zero at seq_len={seq_len}, range={range:?}, last byte={last:#010b}"
                        );
                    }
                }
            }
        }
    }
}
