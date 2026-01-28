//! # bitnuc
//!
//! A library for efficient nucleotide sequence manipulation using 2-bit encoding.
//!
//! ## Features
//!
//! - 2-bit nucleotide encoding (A=00, C=01, G=10, T=11)
//! - 4-bit nucleotide encoding (A=0000, C=0001, G=0010, T=0011, N=1111)
//! - Direct bit manipulation functions for custom implementations
//! - Higher-level sequence type with additional analysis features
//!
//! ## Low-Level Packing Functions
//!
//! For direct bit manipulation, use the `as_2bit` and `from_2bit` functions:
//!
//! ```rust
//! use bitnuc::{as_2bit, from_2bit, from_2bit_alloc};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Pack a sequence into a u64
//!     let packed = as_2bit(b"ACGT")?;
//!     assert_eq!(packed, 0b11100100);
//!
//!     // Unpack back to a sequence using a reusable buffer
//!     let mut unpacked = Vec::new();
//!     from_2bit(packed, 4, &mut unpacked)?;
//!     assert_eq!(&unpacked, b"ACGT");
//!     unpacked.clear();
//!
//!     // Unpack back to a sequence with a reallocation
//!     let unpacked = from_2bit_alloc(packed, 4)?;
//!     assert_eq!(&unpacked, b"ACGT");
//!
//!     Ok(())
//! }
//! ```
//!
//! These functions are useful when you need to:
//! - Implement custom sequence storage
//! - Manipulate sequences at the bit level
//! - Integrate with other bioinformatics tools
//! - Copy sequences more efficiently
//! - Hash sequences more efficiently
//!
//! For example, packing multiple short sequences:
//!
//! ```rust
//! use bitnuc::{as_2bit, from_2bit};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Pack multiple 4-mers into u64s
//!     let kmers = [b"ACGT", b"TGCA", b"GGCC"];
//!     let packed: Vec<u64> = kmers
//!         .into_iter()
//!         .map(|kmer| as_2bit(kmer))
//!         .collect::<Result<_, _>>()?;
//!
//!     // Unpack when needed
//!     let mut unpacked = Vec::new();
//!     from_2bit(packed[0], 4, &mut unpacked)?;
//!     assert_eq!(&unpacked, b"ACGT");
//!     Ok(())
//! }
//! ```
//!
//! ## Mid-Level Encoding Functions
//!
//! For more control over encoding and decoding, use the `encode` and `decode` functions:
//!
//! These will handle sequences of any length, padding the last u64 with zeros if needed.
//!
//! We'll use the [`nucgen`](https://crates.io/crates/nucgen) crate to generate random sequences for testing:
//!
//! ```rust
//! use bitnuc::twobit::{encode, decode};
//! use nucgen::Sequence;
//!
//! let mut rng = rand::thread_rng();
//! let mut seq = Sequence::new();
//! let seq_len = 1000;
//!
//! // Generate a random sequence
//! seq.fill_buffer(&mut rng, seq_len);
//!
//! // Encode the sequence
//! let mut ebuf = Vec::new(); // Buffer for encoded sequence
//! encode(seq.bytes(), &mut ebuf);
//!
//! // Decode the sequence
//! let mut dbuf = Vec::new(); // Buffer for decoded sequence
//! decode(&ebuf, seq_len, &mut dbuf);
//!
//! // Check that the decoded sequence matches the original
//! assert_eq!(seq.bytes(), &dbuf);
//! ```
//!
//! Note that the `encode` function will always encode a full u64.
//! If you have a sequence that is not a multiple of 32 bases, the final u64 will be backed up to the remainder,
//! and the rest of the bits will be set to zero.
//!
//! Decoding will ignore these zero bits and return the original sequence.
//!
//!
//! ## High-Level Sequence Type
//!
//! For more complex sequence manipulation, use the [`BitNuc`] type:
//!
//! ```rust
//! use bitnuc::BitNuc;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let seq: &[u8] = b"ACGTACGT";
//!     let mut packed = BitNuc::new_2bit();
//!     packed.fill(seq);
//!
//!     let mut dbuf = Vec::new(); // Buffer for decoded sequence
//!     packed.decode_into(&mut dbuf);
//!
//!     // Check that the decoded sequence matches the original
//!     assert_eq!(seq, &dbuf);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Memory Usage
//!
//! The 2-bit encoding provides significant memory savings:
//!
//! ```text
//! Standard encoding: 1 byte per base
//! ACGT = 4 bytes = 32 bits
//!
//! 2-bit encoding: 2 bits per base
//! ACGT = 8 bits
//! ```
//!
//! This means you can store 4 times as many sequences in the same amount of memory.
//!
//! ## Error Handling
//!
//! All operations that could fail return a [`Result`] with [`Error`]:
//!
//! ```rust
//! use bitnuc::{as_2bit, Error};
//!
//! // Invalid nucleotide
//! let err = as_2bit(b"ACGN").unwrap_err();
//! assert!(matches!(err, Error::InvalidBase(b'N')));
//!
//! // Sequence too long
//! let long_seq = vec![b'A'; 33];
//! let err = as_2bit(&long_seq).unwrap_err();
//! assert!(matches!(err, Error::SequenceTooLong(33)));
//! ```
//!
//! ## Performance Considerations
//!
//! When working with many short sequences (like k-mers), using `as_2bit` and `from_2bit`
//! directly can be more efficient than creating [`BitNuc`] instances:
//!
//! ```rust
//! use bitnuc::{as_2bit, from_2bit};
//! use std::collections::HashMap;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Efficient k-mer counting
//!     let mut kmer_counts = HashMap::new();
//!
//!     // Pack k-mers directly into u64s
//!     let sequence = b"ACGTACGT";
//!     for window in sequence.windows(4) {
//!         let packed = as_2bit(window)?;
//!         *kmer_counts.entry(packed).or_insert(0) += 1;
//!     }
//!
//!     // Count of "ACGT"
//!     let acgt_packed = as_2bit(b"ACGT")?;
//!     assert_eq!(kmer_counts.get(&acgt_packed), Some(&2));
//!     Ok(())
//! }
//! ```
//!
//! If you are unpacking many sequences, consider reusing a buffer to avoid reallocations:
//!
//! ```rust
//! use bitnuc::{as_2bit, from_2bit};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//!     // Pack a sequence
//!     let packed = as_2bit(b"ACGT")?;
//!
//!     // Reusable buffer for unpacking
//!     let mut unpacked = Vec::new();
//!     from_2bit(packed, 4, &mut unpacked)?;
//!     assert_eq!(&unpacked, b"ACGT");
//!     unpacked.clear();
//!
//!     // Pack another sequence
//!     let packed = as_2bit(b"TGCA")?;
//!     from_2bit(packed, 4, &mut unpacked)?;
//!     assert_eq!(&unpacked, b"TGCA");
//!     Ok(())
//! }
//! ```
//!
//!
//! See the documentation for [`as_2bit`] and [`from_2bit`] for more details on
//! working with packed sequences directly.

mod error;
pub mod fourbit;
pub mod twobit;
mod types;

pub use error::Error;
pub use types::{BitNuc, BitSize};

pub use fourbit::{as_4bit, from_4bit, from_4bit_alloc};
pub use twobit::{as_2bit, as_2bit_lossy, from_2bit, from_2bit_alloc};

#[cfg(test)]
mod testing {
    use crate::BitNuc;

    #[test]
    fn test_sequence_creation_2bit() {
        let seq = b"ACGTACGT";
        let mut packed = BitNuc::new_2bit();

        // Pack the sequence
        packed.fill(seq).unwrap();

        // Test basic properties
        assert_eq!(seq.len(), 8);

        // Test decoding
        let decoded = packed.decode_alloc().unwrap();
        assert_eq!(&decoded, seq);
    }

    #[test]
    fn test_sequence_replacement_2bit() {
        let seq_a = b"ACGT";
        let seq_b = b"TGCA";

        let mut packed = BitNuc::new_2bit();

        // Pack sequence A
        packed.fill(seq_a).unwrap();

        // Test basic properties
        assert_eq!(seq_a.len(), 4);

        // Test decoding
        let decoded = packed.decode_alloc().unwrap();
        assert_eq!(&decoded, seq_a);

        // Replace with sequence B
        packed.fill(seq_b).unwrap();

        // Test basic properties
        assert_eq!(seq_b.len(), 4);

        // Test decoding
        let decoded = packed.decode_alloc().unwrap();
        assert_eq!(&decoded, seq_b);
    }

    #[test]
    fn test_sequence_creation_4bit() {
        let seq = b"ACGTACGT";
        let mut packed = BitNuc::new_4bit();

        // Pack the sequence
        packed.fill(seq).unwrap();

        // Test basic properties
        assert_eq!(seq.len(), 8);

        // Test decoding
        let decoded = packed.decode_alloc().unwrap();
        assert_eq!(&decoded, seq);
    }

    #[test]
    fn test_sequence_replacement_4bit() {
        let seq_a = b"ACGT";
        let seq_b = b"TGCA";

        let mut packed = BitNuc::new_4bit();

        // Pack sequence A
        packed.fill(seq_a).unwrap();

        // Test basic properties
        assert_eq!(seq_a.len(), 4);

        // Test decoding
        let decoded = packed.decode_alloc().unwrap();
        assert_eq!(&decoded, seq_a);

        // Replace with sequence B
        packed.fill(seq_b).unwrap();

        // Test basic properties
        assert_eq!(seq_b.len(), 4);

        // Test decoding
        let decoded = packed.decode_alloc().unwrap();
        assert_eq!(&decoded, seq_b);
    }
}
