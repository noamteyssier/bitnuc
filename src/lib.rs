//! # bitnuc
//!
//! A library for efficient nucleotide sequence manipulation using 2-bit encoding.
//!
//! ## Features
//!
//! - Byte-native 2-bit nucleotide encoding (A=00, C=01, G=10, T=11) with a
//!   SIMD implementation dispatched at runtime via
//!   [`fearless_simd`](https://docs.rs/fearless_simd)
//! - A little-endian-pinned `u64` kmer boundary ([`as_2bit`] / [`from_2bit`])
//!   for hashing and fixed-width storage of sequences up to 32 bases
//!
//! ## Encoded Format
//!
//! Sequences encode to plain bytes: byte `k` holds bases `4k..4k+4`, base `j`
//! at bits `2*(j % 4)` of its byte. A sequence of `n` bases occupies exactly
//! `n.div_ceil(4)` bytes and trailing pad bits are always zero. Because the
//! format is defined on bytes, it is endianness-free — the encoded buffer is
//! the on-disk format. See the [`twobit`] module docs for details.
//!
//! Encoding is lossy for bytes outside `ACGTacgt`: invalid bases map to an
//! unspecified code rather than an error. If you need to preserve ambiguous
//! bases, detect them separately and handle them out-of-band.
//!
//! ## Encoding and Decoding
//!
//! The core functions operate on byte slices ([`twobit::encode`] /
//! [`twobit::decode`]), with `_resize` variants that manage the buffer length
//! for you:
//!
//! ```rust
//! use bitnuc::twobit::{encode_resize, decode_resize};
//!
//! # fn main() -> Result<(), bitnuc::BitnucError> {
//! let seq = b"ACGTACGTAC"; // 10 bases -> 3 encoded bytes
//!
//! let mut ebuf = Vec::new();
//! encode_resize(seq, &mut ebuf);
//! assert_eq!(ebuf.len(), 3);
//!
//! let mut dbuf = Vec::new();
//! decode_resize(&ebuf, seq.len(), &mut dbuf)?;
//! assert_eq!(&dbuf, seq);
//! # Ok(())
//! # }
//! ```
//!
//! The slice-based variants write into caller-provided buffers, which lets
//! consumers control allocation and padding (e.g. file formats that pad
//! encoded sequences to 8-byte words):
//!
//! ```rust
//! use bitnuc::twobit::{encode, decode};
//!
//! # fn main() -> Result<(), bitnuc::BitnucError> {
//! let seq = b"ACGTACGTAC";
//!
//! // Pad the encoded buffer to an 8-byte multiple: the layout is identical
//! // to the legacy u64 packing serialized little-endian
//! let mut ebuf = vec![0u8; seq.len().div_ceil(4).next_multiple_of(8)];
//! encode(seq, &mut ebuf)?;
//!
//! let mut dbuf = vec![0u8; seq.len()];
//! decode(&ebuf, seq.len(), &mut dbuf)?;
//! assert_eq!(&dbuf, seq);
//! # Ok(())
//! # }
//! ```
//!
//! ## u64 Kmer Packing
//!
//! For hashing and fixed-width storage of short sequences (barcodes, UMIs,
//! k-mers up to 32 bases), [`as_2bit`] and [`from_2bit`] pack to and from a
//! `u64`. These are the only functions that cross the byte/u64 boundary, and
//! they are pinned to little-endian internally, so the values are
//! bit-identical on every platform (and to bitnuc 0.4.x):
//!
//! ```rust
//! use bitnuc::{as_2bit, from_2bit};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), bitnuc::BitnucError> {
//! let packed = as_2bit(b"ACGT")?;
//! assert_eq!(packed, 0b11100100);
//!
//! // Efficient k-mer counting
//! let mut kmer_counts = HashMap::new();
//! for window in b"ACGTACGT".windows(4) {
//!     *kmer_counts.entry(as_2bit(window)?).or_insert(0) += 1;
//! }
//! assert_eq!(kmer_counts.get(&packed), Some(&2));
//!
//! // Unpacking appends to the buffer; clear it between sequences when reusing
//! let mut unpacked = Vec::new();
//! from_2bit(packed, 4, &mut unpacked)?;
//! assert_eq!(&unpacked, b"ACGT");
//! # Ok(())
//! # }
//! ```
//!
//! Packed kmers can be compared with [`twobit::hdist_scalar`]:
//!
//! ```rust
//! use bitnuc::{as_2bit, twobit::hdist_scalar};
//!
//! # fn main() -> Result<(), bitnuc::BitnucError> {
//! let u = as_2bit(b"ACGT")?;
//! let v = as_2bit(b"ACGA")?;
//! assert_eq!(hdist_scalar(u, v, 4)?, 1);
//! # Ok(())
//! # }
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
//! ACGT = 1 byte = 8 bits
//! ```
//!
//! ## Error Handling
//!
//! Operations that can fail return a [`Result`] with [`BitnucError`]:
//!
//! ```rust
//! use bitnuc::{as_2bit, BitnucError};
//!
//! // Sequence too long for a u64
//! let long_seq = vec![b'A'; 33];
//! let err = as_2bit(&long_seq).unwrap_err();
//! assert!(matches!(err, BitnucError::SequenceTooLong(33)));
//! ```

mod error;
pub mod twobit;

pub use error::BitnucError;

#[allow(deprecated)]
pub use twobit::as_2bit_lossy;
pub use twobit::{as_2bit, from_2bit, from_2bit_alloc};
