#![doc = include_str!("../README.md")]

mod decode;
mod encode;
mod error;
mod hamming;
mod kmer;

pub use decode::{decode, decode_resize};
pub use encode::{encode, encode_resize};
pub use error::BitnucError;
pub use hamming::hdist_scalar;
pub use kmer::{as_2bit, from_2bit};

#[cfg(test)]
mod testing {
    use rand::{make_rng, rngs::SmallRng, seq::IndexedRandom};

    use crate::{BitnucError, as_2bit, from_2bit, *};

    const CHARS: [u8; 4] = *b"ACGT";

    fn generate_sequence(n: usize) -> Vec<u8> {
        let mut rng: SmallRng = make_rng();
        (0..n).map(|_| *CHARS.choose(&mut rng).unwrap()).collect()
    }

    /// Byte `k` holds bases `4k..4k+4`, base `j` at bits `2*(j % 4)`.
    #[test]
    fn golden_byte_layout() {
        let mut ebuf = Vec::new();

        // A=00, C=01, G=10, T=11 packed LSB-first within the byte
        encode_resize(b"ACGT", &mut ebuf);
        assert_eq!(ebuf, [0b11100100]);

        encode_resize(b"TGCA", &mut ebuf);
        assert_eq!(ebuf, [0b00011011]);

        // Second byte holds bases 4..8
        encode_resize(b"AAAATGCA", &mut ebuf);
        assert_eq!(ebuf, [0b00000000, 0b00011011]);
    }

    /// Unused bits of a trailing partial byte are always zero.
    #[test]
    fn golden_partial_byte_padding() {
        let mut ebuf = Vec::new();

        encode_resize(b"TTTTT", &mut ebuf);
        assert_eq!(ebuf, [0xFF, 0b00000011]);

        // Padding is deterministic even when the buffer held stale data
        let mut ebuf = vec![0xFFu8; 2];
        encode(b"TTTTT", &mut ebuf).unwrap();
        assert_eq!(ebuf, [0xFF, 0b00000011]);
    }

    #[test]
    fn lowercase_matches_uppercase() {
        let mut upper = Vec::new();
        let mut lower = Vec::new();
        encode_resize(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGT", &mut upper);
        encode_resize(b"acgtacgtacgtacgtacgtacgtacgtacgtacgt", &mut lower);
        assert_eq!(upper, lower);
    }

    /// Exercises every dispatch path: 64-base blocks, the 32/16/8-base steps,
    /// and the scalar tail.
    #[test]
    fn roundtrip_all_lengths() {
        let mut ebuf = Vec::new();
        let mut dbuf = Vec::new();

        for len in 0..=1025 {
            let seq = generate_sequence(len);

            encode_resize(&seq, &mut ebuf);
            assert_eq!(ebuf.len(), len.div_ceil(4));

            decode_resize(&ebuf, len, &mut dbuf).unwrap();
            assert_eq!(&dbuf[..len], &seq, "roundtrip failed at len {len}");
        }
    }

    /// `as_2bit` is exactly the little-endian interpretation of the byte
    /// encoding — the only place the u64 boundary is crossed.
    #[test]
    fn kmer_is_le_view_of_bytes() {
        let mut ebuf = Vec::new();

        for len in 0..=32 {
            let seq = generate_sequence(len);

            let packed = as_2bit(&seq).unwrap();

            encode_resize(&seq, &mut ebuf);
            let mut word = [0u8; 8];
            word[..ebuf.len()].copy_from_slice(&ebuf);
            assert_eq!(packed, u64::from_le_bytes(word), "mismatch at len {len}");
        }
    }

    /// Golden values from the 0.4.x u64 packing (base `i` at bits `2i`).
    #[test]
    fn kmer_golden_values() {
        assert_eq!(as_2bit(b"ACGT").unwrap(), 0b11100100);
        assert_eq!(as_2bit(b"A").unwrap(), 0);
        assert_eq!(as_2bit(b"T").unwrap(), 0b11);
        assert_eq!(
            as_2bit(b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT").unwrap(),
            u64::MAX
        );
        // High bits beyond the sequence are zero
        assert_eq!(as_2bit(b"TTTT").unwrap(), 0xFF);
    }

    /// The whole-buffer encoding padded to 8-byte words matches the legacy
    /// one-u64-per-32-bases packing serialized little-endian (bq compatibility).
    #[test]
    fn padded_bytes_match_legacy_word_stream() {
        let mut ebuf = Vec::new();

        for len in [1, 31, 32, 33, 64, 100, 129] {
            let seq = generate_sequence(len);

            // New path, padded to whole words
            encode_resize(&seq, &mut ebuf);
            ebuf.resize(ebuf.len().next_multiple_of(8), 0);

            // Legacy path: one u64 per 32 bases, written little-endian
            let legacy: Vec<u8> = seq
                .chunks(32)
                .flat_map(|chunk| as_2bit(chunk).unwrap().to_le_bytes())
                .collect();

            assert_eq!(ebuf, legacy, "word-stream mismatch at len {len}");
        }
    }

    /// `from_2bit` decodes the full kmer; positions past the packed sequence
    /// are `b'A'` for values produced by `as_2bit`.
    #[test]
    fn from_2bit_stack_array() {
        let packed = as_2bit(b"TGCA").unwrap();
        let unpacked = from_2bit(packed);
        assert_eq!(&unpacked[..4], b"TGCA");
        assert_eq!(&unpacked[4..], [b'A'; 28]);
    }

    #[test]
    fn from_2bit_partial() {
        let packed = as_2bit(b"ACGT").unwrap();
        assert_eq!(&from_2bit(packed)[..2], b"AC");
        assert_eq!(&from_2bit(packed)[..3], b"ACG");
    }

    #[test]
    fn kmer_errors() {
        let long = vec![b'A'; 33];
        assert!(matches!(
            as_2bit(&long).unwrap_err(),
            BitnucError::SequenceTooLong(33)
        ));
    }

    #[test]
    fn buffer_size_errors() {
        let mut small = [0u8; 1];
        assert!(matches!(
            encode(b"ACGTA", &mut small).unwrap_err(),
            BitnucError::EncodingBufferTooSmall {
                expected: 2,
                actual: 1
            }
        ));

        let ebuf = [0u8; 1];
        let mut out = [0u8; 2];
        assert!(matches!(
            decode(&ebuf, 4, &mut out).unwrap_err(),
            BitnucError::DecodingBufferTooSmall {
                expected: 4,
                actual: 2
            }
        ));
        let mut out = [0u8; 8];
        assert!(matches!(
            decode(&ebuf, 8, &mut out).unwrap_err(),
            BitnucError::EncodingBufferTooSmall {
                expected: 2,
                actual: 1
            }
        ));
    }
}
