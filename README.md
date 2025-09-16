# bitnuc

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
![actions status](https://github.com/noamteyssier/bitnuc/workflows/Rust/badge.svg)
[![Crates.io](https://img.shields.io/crates/d/bitnuc?color=orange&label=crates.io)](https://crates.io/crates/bitnuc)
[![docs.rs](https://img.shields.io/docsrs/bitnuc?color=green&label=docs.rs)](https://docs.rs/bitnuc/latest/bitnuc/)

A library for efficient nucleotide sequence manipulation using 2-bit encoding.

## Features

- 2-bit nucleotide encoding (A=00, C=01, G=10, T=11)
- 4-bit nucleotide encoding (A=0000, C=0001, G=0010, T=0011, N=1111)
- Direct bit manipulation functions for custom implementations
- Higher-level sequence type with additional analysis features

## Low-Level Packing Functions

For direct bit manipulation, use the `as_2bit` and `from_2bit` functions:

```rust
use bitnuc::{as_2bit, from_2bit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Pack a sequence into a u64
    let packed = as_2bit(b"ACGT")?;
    assert_eq!(packed, 0b11100100);

    // Unpack back to a sequence
    let mut unpacked = Vec::new(); // Allocate a reusable buffer
    from_2bit(packed, 4, &mut unpacked)?;
    assert_eq!(&unpacked, b"ACGT");
    unpacked.clear(); // Reuse the buffer
    Ok(())
}
```

These functions are useful when you need to:

- Implement custom sequence storage
- Manipulate sequences at the bit level
- Integrate with other bioinformatics tools
- Copy sequences more efficiently
- Hash sequences more efficiently

For example, packing multiple short sequences:

```rust
use bitnuc::{as_2bit, from_2bit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Pack multiple 4-mers into u64s
    let kmers = [b"ACGT", b"TGCA", b"GGCC"];
    let packed: Vec<u64> = kmers
        .into_iter()
        .map(|kmer| as_2bit(kmer))
        .collect::<Result<_, _>>()?;

    // Unpack when needed
    let mut kmers = Vec::new();
    from_2bit(packed[0], 4, &mut kmers)?;
    assert_eq!(&kmers, b"ACGT");
    Ok(())
}
```

## Mid-Level Encoding Functions

For more control over encoding and decoding, use the `encode` and `decode` functions:

These will handle sequences of any length, padding the last u64 with zeros if needed.

We'll use the [`nucgen`](https://crates.io/crates/nucgen) crate to generate random sequences for testing:

```rust
use bitnuc::{encode, decode};
use nucgen::Sequence;

let mut rng = rand::thread_rng();
let mut seq = Sequence::new();
let seq_len = 1000;

// Generate a random sequence
seq.fill_buffer(&mut rng, seq_len);

// Encode the sequence
let mut ebuf = Vec::new(); // Buffer for encoded sequence
encode(seq.bytes(), &mut ebuf);

// Decode the sequence
let mut dbuf = Vec::new(); // Buffer for decoded sequence
decode(&ebuf, seq_len, &mut dbuf);

// Check that the decoded sequence matches the original
assert_eq!(seq.bytes(), &dbuf);
```

Note that the `encode` function will always encode a full u64.
If you have a sequence that is not a multiple of 32 bases, the final u64 will be backed up to the remainder,
and the rest of the bits will be set to zero.

Decoding will ignore these zero bits and return the original sequence.

## High-Level Sequence Type

For more complex sequence manipulation, use the [`PackedSequence`] type:

```rust
use bitnuc::{PackedSequence, GCContent, BaseCount};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seq = PackedSequence::new(b"ACGTACGT")?;

    // Sequence analysis
    println!("GC Content: {}%", seq.gc_content());
    let [a_count, c_count, g_count, t_count] = seq.base_counts();

    // Slicing
    let subseq = seq.slice(1..5)?;
    assert_eq!(&subseq, b"CGTA");
    Ok(())
}
```

## Memory Usage

The 2-bit encoding provides significant memory savings:

```text
Standard encoding: 1 byte per base
ACGT = 4 bytes = 32 bits

2-bit encoding: 2 bits per base
ACGT = 8 bits
```

This means you can store 4 times as many sequences in the same amount of memory.

## Error Handling

All operations that could fail return a [`Result`] with [`Error`]:

```rust
use bitnuc::{as_2bit, Error};

// Invalid nucleotide
let err = as_2bit(b"ACGN").unwrap_err();
assert!(matches!(err, Error::InvalidBase(b'N')));

// Sequence too long
let long_seq = vec![b'A'; 33];
let err = as_2bit(&long_seq).unwrap_err();
assert!(matches!(err, Error::SequenceTooLong(33)));
```

## SIMD Acceleration

`as_2bit`, `from_2bit`, `as_4bit`, `from_4bit`, and both twobit and fourbit `encode`, and `decode` are optionally SIMD accelerated depending on the architecture of your system.
By default, SIMD instructions are used, but they can be shut-off using the `nosimd` feature flag.

For increased performance and to really take advantage of the SIMD I recommend compiling with:

```bash
RUSTFLAGS="-C target-cpu=native"
```

or to add these flags to your project via the cargo build config:

```toml
# ./cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]
```
