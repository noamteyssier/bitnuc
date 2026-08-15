use std::ops::Range;

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
    if (range.end - range.start).div_ceil(4) > into.len() {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: range.len().div_ceil(4),
            actual: into.len(),
        });
    }

    // no-op
    if range.is_empty() {
        return Ok(());
    }

    extract_inner(packed, range, into);

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
    extract_inner(packed, range, into);
    Ok(())
}

#[inline(always)]
fn extract_inner(packed: &[u8], range: Range<usize>, into: &mut [u8]) {
    let start_word = range.start / 4;
    let end_word = (range.end - 1) / 4;
    let packed_subset = &packed[start_word..=end_word];

    // Phase offset into the packed word
    let offbit = 2 * (range.start % 4);

    // fast-path: no need to shift bits
    if offbit == 0 {
        into[..packed_subset.len()].copy_from_slice(packed_subset);
    } else {
        // main loop: shift bits between packed words
        for idx in 0..into.len() {
            let w_i = packed_subset[idx]; // current packed word
            let w_i1 = packed_subset.get(idx + 1).copied().unwrap_or(0); // next packed word, if any
            into[idx] = (w_i >> offbit) | (w_i1 << (8 - offbit));
        }
    }

    // zero out the unused high bits of the last packed word
    if !range.len().is_multiple_of(4) {
        let used_bits = 2 * (range.len() % 4);
        into[into.len() - 1] &= (1u8 << used_bits) - 1; // bitmask out unused bits
    }
}

#[cfg(test)]
mod testing {
    use rand::{Rng, RngExt, rngs::SmallRng};

    use super::extract_resize;
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
