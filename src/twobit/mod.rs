pub mod functions;
pub mod packing;
pub mod unpacking;

pub use functions::{hdist, hdist_scalar, split_packed};
pub use packing::{as_2bit, encode_internal};
pub use unpacking::{from_2bit, from_2bit_alloc, from_2bit_multi};

use crate::NucleotideError;

/// Encode a sequence into a buffer of 2-bit encoded nucleotides.
///
/// # Arguments
///
/// * `sequence` - The nucleotide sequence to encode.
/// * `ebuf` - The buffer to write the encoded nucleotides to.
///
/// # Errors
///
/// If the sequence cannot be encoded, an error is returned.
pub fn encode(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), NucleotideError> {
    encode_internal(sequence, ebuf)?;
    Ok(())
}

/// Encode a sequence into a buffer of 2-bit encoded nucleotides.
///
/// This function allocates a new buffer to store the encoded nucleotides.
///
/// # Arguments
///
/// * `sequence` - The nucleotide sequence to encode.
///
/// # Errors
///
/// If the sequence cannot be encoded, an error is returned.
pub fn encode_alloc(sequence: &[u8]) -> Result<Vec<u64>, NucleotideError> {
    let mut ebuf = Vec::new();
    encode(sequence, &mut ebuf)?;
    Ok(ebuf)
}

/// Unpacks a 2-bit packed sequence into a nucleotide sequence.
///
/// The sequence is a collection of u64 values, where each u64 contains up to 32 nucleotides.
///
/// It is expected that the sequence is packed fully (32bp) until the final u64, which may contain
/// fewer than 32 nucleotides.
///
/// # Arguments
///
/// * `ebuf` - The buffer containing the packed nucleotides.
/// * `n_bases` - The number of nucleotides to unpack.
/// * `dbuf` - The buffer to write the unpacked nucleotides to.
///
/// # Errors
///
/// If the sequence cannot be unpacked, an error is returned.
pub fn decode(ebuf: &[u64], n_bases: usize, dbuf: &mut Vec<u8>) -> Result<(), NucleotideError> {
    from_2bit_multi(ebuf, n_bases, dbuf)
}

#[cfg(test)]
mod testing {
    use super::*;
    use crate::NucleotideError;
    use nucgen::Sequence;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let test_cases = [
            b"A".as_slice(),
            b"C".as_slice(),
            b"G".as_slice(),
            b"T".as_slice(),
            b"AC".as_slice(),
            b"GT".as_slice(),
            b"ACG".as_slice(),
            b"TGC".as_slice(),
            b"ACGT".as_slice(),
            b"TGCA".as_slice(),
            b"ACGTACGT".as_slice(),
            b"AAAA".as_slice(),
            b"CCCC".as_slice(),
            b"GGGG".as_slice(),
            b"TTTT".as_slice(),
        ];
        let mut unpacked = Vec::new();

        for &seq in &test_cases {
            let packed = as_2bit(seq).unwrap();
            from_2bit(packed, seq.len(), &mut unpacked).unwrap();
            assert_eq!(seq, unpacked.as_slice());
            unpacked.clear();
        }
    }

    #[test]
    fn test_partial_unpack() {
        let packed = as_2bit(b"ACGT").unwrap();
        let mut unpacked = Vec::new();

        from_2bit(packed, 2, &mut unpacked).unwrap();
        assert_eq!(unpacked, b"AC");
        unpacked.clear();

        from_2bit(packed, 3, &mut unpacked).unwrap();
        assert_eq!(unpacked, b"ACG");
        unpacked.clear();
    }

    #[test]
    fn test_large_sequence_round_trip() -> Result<(), NucleotideError> {
        let mut rng = rand::thread_rng();
        let mut seq = Sequence::new();

        let sizes = 1..=1000;

        for len in sizes {
            seq.fill_buffer(&mut rng, len);

            let mut ebuf = Vec::new();
            encode(seq.bytes(), &mut ebuf)?;

            let mut unpacked = Vec::new();
            decode(&ebuf, len, &mut unpacked)?;

            assert_eq!(seq.bytes(), unpacked.as_slice());
        }

        Ok(())
    }
}
