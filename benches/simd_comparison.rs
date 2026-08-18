use std::hint::black_box;

use bitnuc::{
    ambiguous_bases, as_2bit, decode, encode, encode_resize, extract, from_2bit,
    hdist_pairwise_resize, hdist_scalar,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn generate_sequence(length: usize) -> Vec<u8> {
    let bases = *b"ACGT";
    (0..length).map(|i| bases[i % 4]).collect()
}

fn bench_packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("packing");

    // Test different sequence lengths
    for size in [4, 8, 16, 24, 32].iter() {
        let seq = generate_sequence(*size);

        group.bench_with_input(BenchmarkId::new("packing_2bit", size), &seq, |b, seq| {
            b.iter(|| as_2bit(seq))
        });
    }

    group.finish();
}

fn bench_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");

    // Test different sequence lengths
    for size in [10, 100, 1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        let seq = generate_sequence(*size);

        let mut ebuf = vec![0u8; size.div_ceil(4)];
        group.bench_with_input(BenchmarkId::new("encoding_2bit", size), &seq, |b, seq| {
            b.iter(|| encode(seq, &mut ebuf).unwrap())
        });
    }

    group.finish();
}

fn bench_unpacking(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpacking");

    // from_2bit always decodes the full 32-base kmer, so one entry suffices
    let packed = as_2bit(&generate_sequence(32)).unwrap();
    group.bench_with_input("unpacking_2bit", &packed, |b, packed| {
        b.iter(|| from_2bit(*packed))
    });

    group.finish();
}

fn bench_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoding");

    // Test different sequence lengths
    let sizes = [10, 100, 1_000, 10_000, 100_000];
    for size in sizes.iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        let seq = generate_sequence(*size);

        let mut packed_2bit = Vec::new();
        encode_resize(&seq, &mut packed_2bit);
        let mut dbuf = vec![0u8; *size];
        group.bench_with_input(
            BenchmarkId::new("decoding_2bit", size),
            &packed_2bit,
            |b, packed| b.iter(|| decode(packed, *size, &mut dbuf).unwrap()),
        );
    }

    group.finish();
}

/// Canonical sequence with an `N` every `n_every` bases (0 = fully canonical)
fn generate_ambiguous_sequence(length: usize, n_every: usize) -> Vec<u8> {
    let bases = *b"ACGT";
    (0..length)
        .map(|i| {
            if n_every != 0 && i % n_every == n_every - 1 {
                b'N'
            } else {
                bases[i % 4]
            }
        })
        .collect()
}

/// Scalar baseline for `ambiguous_bases`
fn ambiguous_bases_scalar(seq: &[u8], pos: &mut Vec<usize>) {
    for (i, b) in seq.iter().enumerate() {
        if !matches!(*b | 0x20, b'a' | b'c' | b'g' | b't') {
            pos.push(i);
        }
    }
}

fn bench_ambiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("ambiguous_bases");

    for (label, n_every) in [("0pct", 0), ("1pct", 100), ("25pct", 4)] {
        for size in [1_000, 100_000] {
            group.throughput(Throughput::Bytes(size as u64));
            let seq = generate_ambiguous_sequence(size, n_every);
            let mut pos = Vec::with_capacity(size);

            group.bench_with_input(
                BenchmarkId::new(format!("simd_{label}"), size),
                &seq,
                |b, seq| {
                    b.iter(|| {
                        pos.clear();
                        ambiguous_bases(seq, &mut pos);
                        black_box(&pos);
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_{label}"), size),
                &seq,
                |b, seq| {
                    b.iter(|| {
                        pos.clear();
                        ambiguous_bases_scalar(seq, &mut pos);
                        black_box(&pos);
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract");

    // Test different sequence lengths, at both:
    // - byte-aligned start (`offbit == 0` memcpy fast path)
    // - byte-unaligned start (the shift-and-merge path in SIMD)
    for size in [10, 100, 1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        let seq = generate_sequence(*size);
        let mut packed = Vec::new();
        encode_resize(&seq, &mut packed);

        let aligned_range = 0..*size;
        let mut aligned_out = vec![0u8; aligned_range.len().div_ceil(4)];
        group.bench_with_input(
            BenchmarkId::new("extract_aligned", size),
            &packed,
            |b, packed| {
                b.iter(|| {
                    extract(packed, aligned_range.clone(), &mut aligned_out).unwrap();
                    black_box(&aligned_out);
                })
            },
        );

        if *size > 4 {
            let unaligned_range = 1..(*size - 1);
            let mut unaligned_out = vec![0u8; unaligned_range.len().div_ceil(4)];
            group.bench_with_input(
                BenchmarkId::new("extract_unaligned", size),
                &packed,
                |b, packed| {
                    b.iter(|| {
                        extract(packed, unaligned_range.clone(), &mut unaligned_out).unwrap();
                        black_box(&unaligned_out);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Deterministic pseudo-random packed kmers (xorshift64)
fn generate_kmers(n: usize) -> Vec<u64> {
    let mut state = 0x243F6A8885A308D3u64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        })
        .collect()
}

/// Naive scalar baseline for `hdist_pairwise_resize`
fn hdist_pairwise_scalar(items: &[u64], len: usize, into: &mut [usize]) {
    let mut out = 0;
    for (i, &u) in items.iter().enumerate() {
        for &v in &items[i + 1..] {
            into[out] = hdist_scalar(u, v, len).unwrap() as usize;
            out += 1;
        }
    }
}

fn bench_hdist_pairwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdist_pairwise");
    let len = 32;

    for n in [8, 16, 128, 1024] {
        let n_distances = n * (n - 1) / 2;
        group.throughput(Throughput::Elements(n_distances as u64));
        let items = generate_kmers(n);
        let mut into = vec![0usize; n_distances];

        group.bench_with_input(BenchmarkId::new("simd", n), &items, |b, items| {
            b.iter(|| {
                hdist_pairwise_resize(items, len, &mut into).unwrap();
                black_box(&into);
            })
        });
        group.bench_with_input(BenchmarkId::new("scalar", n), &items, |b, items| {
            b.iter(|| {
                hdist_pairwise_scalar(items, len, &mut into);
                black_box(&into);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_packing,
    bench_encoding,
    bench_unpacking,
    bench_decoding,
    bench_ambiguous,
    bench_extract,
    bench_hdist_pairwise,
);
criterion_main!(benches);
