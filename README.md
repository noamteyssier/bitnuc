# bitnuc

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
![actions status](https://github.com/noamteyssier/bitnuc/workflows/Rust/badge.svg)
[![Crates.io](https://img.shields.io/crates/d/bitnuc?color=orange&label=crates.io)](https://crates.io/crates/bitnuc)
[![docs.rs](https://img.shields.io/docsrs/bitnuc?color=green&label=docs.rs)](https://docs.rs/bitnuc/latest/bitnuc/)

A library for efficient nucleotide sequence manipulation using 2-bit encoding.

## Summary

This a SIMD-accelerated two-bit encoding library for nucleotide sequences.
It is meant to have fast encode/decode routines for small and large sequences and to provide a fairly unstructured interface for working with nucleotide sequences in memory.

It provides:

- 2-bit nucleotide encoding (A=00, C=01, G=10, T=11) with SIMD dispatched at runtime via [`fearless_simd`](https://docs.rs/fearless_simd)
- A little-endian-pinned `u64` kmer boundary (`as_2bit` / `from_2bit`) for hashing and fixed-width integer storage of sequences up to 32 bases
- Ambiguous base detection (`ambiguous_bases`) for tracking non-`ACGTacgt` bases

## Encoded Format

Sequences encode to plain bytes: byte `k` holds bases `4k..4k+4`, base `j` at
bits `2*(j % 4)` of its byte. A sequence of `n` bases occupies exactly
`n.div_ceil(4)` bytes and trailing pad bits are always zero.

```text
Input Sequence: [A][C][G][T]
Output Buffer: [11 10 01 00]

[]: byte boundary
Bit pairs are MSB-first: base j occupies bits 2j..2j+1,
so the first base is the rightmost pair (A=00).
```

> Note: Encoding is lossy for bytes outside `ACGTacgt`: invalid bases map to an unspecified code rather than an error. If you need to preserve ambiguous bases, detect and track them separately (see `ambiguous_bases`).

## Encoding and Decoding

The core functions operate on byte slices, with `_resize` variants that
manage the buffer length for you:

```rust
use bitnuc::{encode_resize, decode_resize};

fn main() -> Result<(), bitnuc::BitnucError> {
    let seq = b"ACGTACGTAC"; // 10 bases -> 3 encoded bytes

    let mut ebuf = Vec::new();
    encode_resize(seq, &mut ebuf);
    assert_eq!(ebuf.len(), 3);

    let mut dbuf = Vec::new();
    decode_resize(&ebuf, seq.len(), &mut dbuf)?;
    assert_eq!(&dbuf, seq);
    Ok(())
}
```

The slice-based variants write into caller-provided buffers, which lets
consumers control allocation and padding (e.g. file formats that pad encoded
sequences to 8-byte words):

```rust
use bitnuc::{encode, decode};

fn main() -> Result<(), bitnuc::BitnucError> {
    let seq = b"ACGTACGTAC";

    // Pad the encoded buffer to an 8-byte multiple: the layout is identical
    // to the legacy u64 packing serialized little-endian
    let mut ebuf = vec![0u8; seq.len().div_ceil(4).next_multiple_of(8)];
    encode(seq, &mut ebuf)?;

    let mut dbuf = vec![0u8; seq.len()];
    decode(&ebuf, seq.len(), &mut dbuf)?;
    assert_eq!(&dbuf, seq);
    Ok(())
}
```

## u64 Kmer Packing

For hashing and fixed-width storage of short sequences (barcodes, UMIs,
k-mers up to 32 bases), `as_2bit` and `from_2bit` pack to and from a `u64`.

> Note: These are pinned to **little-endian** internally.

```rust
use bitnuc::{as_2bit, from_2bit};
use std::collections::HashMap;

fn main() -> Result<(), bitnuc::BitnucError> {
    let packed = as_2bit(b"ACGT")?;
    assert_eq!(packed, 0b11100100);

    // Efficient k-mer counting
    let mut kmer_counts = HashMap::new();
    for window in b"ACGTACGT".windows(4) {
        *kmer_counts.entry(as_2bit(window)?).or_insert(0) += 1;
    }
    assert_eq!(kmer_counts.get(&packed), Some(&2));

    // Unpacking returns a stack array of all 32 bases; slice to your length
    let unpacked = from_2bit(packed);
    assert_eq!(&unpacked[..4], b"ACGT");
    Ok(())
}
```

Packed kmers can be compared with `hdist_scalar`:

```rust
use bitnuc::{as_2bit, hdist_scalar};

fn main() -> Result<(), bitnuc::BitnucError> {
    let u = as_2bit(b"ACGT")?;
    let v = as_2bit(b"ACGA")?;
    assert_eq!(hdist_scalar(u, v, 4)?, 1);
    Ok(())
}
```

## Identifying ambiguous bases

Ambiguous bases (non-`ACGT`) bases are unable to be represented with this two-bit encoding scheme.
The encoding algorithm also remaps lowercase to upper case (so `acgt` -> `ACGT` internally).
Use `ambiguous_bases` to track all positions of unrepresentable nucleotides.
The position buffer is generic over its element type (`usize`, `u64`, or `u32`):

```rust
use bitnuc::ambiguous_bases;

fn main() {
    let seq = b"ACgTNACYAaTH"; // has unrepresentable bases (N/Y/H)

    let mut pos: Vec<usize> = Vec::default();
    ambiguous_bases(seq, &mut pos);

    assert_eq!(
        pos,
        vec![4, 7, 11],
    );
}
```

> Note: `ambiguous_bases` only tracks non-`ACGTacgt` bases.
> It does not identify lowercase letters which are also not representable but
> which are remapped to their uppercase variants through encoding/decoding.

## Memory Usage

The 2-bit encoding provides significant memory savings:

```text
Standard encoding: 1 byte per base
ACGT = 4 bytes = 32 bits

2-bit encoding: 2 bits per base
ACGT = 1 byte = 8 bits
```

## Performance

Throughput by sequence length, measured on an Apple M3 Pro with
`target-cpu=native` (criterion mean, 1 byte per base):

|     bp | encode (GB/s) | decode (GB/s) |
| -----: | ------------: | ------------: |
|     10 |           2.3 |           2.2 |
|    100 |          12.5 |          13.8 |
|   1000 |          35.4 |          29.7 |
|  10000 |          38.2 |          34.0 |
| 100000 |          38.6 |          33.5 |

To regenerate the table on your machine:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench simd_comparison -- coding_2bit
uv run scripts/perf_table.py
```

## SIMD Acceleration

The 2-bit `encode` and `decode` are SIMD accelerated via
[`fearless_simd`](https://docs.rs/fearless_simd), with the instruction set
(NEON, SSE, AVX2, AVX-512) selected at runtime.

## Related Work

I highly recommend checking out [packed-seq](https://github.com/rust-seq/packed-seq).
They are currently the highest performance 2-bit encoding library in Rust as far as I can tell.
They follow a different bit-packing scheme than this library and they can shave off a few instructions in their SIMD routines.

If you're interested in 2-bit encoding in general make sure to check out [cute-nucleotides](https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides) which has an excellent overview of different algorithms and their performance characteristics.
