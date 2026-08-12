use bitnuc::{as_2bit, decode, encode, encode_resize, from_2bit};
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
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].iter() {
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
    let sizes = [32, 128, 512, 2048];
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

criterion_group!(
    benches,
    bench_packing,
    bench_encoding,
    bench_unpacking,
    bench_decoding
);
criterion_main!(benches);
