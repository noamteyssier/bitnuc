use bitnuc::{as_2bit, as_4bit, fourbit, from_2bit, from_4bit, twobit};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn generate_sequence(length: usize) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..length).map(|i| bases[i % 4]).collect()
}

fn bench_packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("packing");

    let impl_type = if cfg!(feature = "nosimd") {
        "nosimd"
    } else {
        "simd"
    };

    // Test different sequence lengths
    for size in [4, 8, 16, 24, 32].iter() {
        let seq = generate_sequence(*size);

        group.bench_with_input(
            BenchmarkId::new(format!("packing_2bit_{}", impl_type), size),
            &seq,
            |b, seq| b.iter(|| as_2bit(seq)),
        );

        if *size <= 16 {
            group.bench_with_input(
                BenchmarkId::new(format!("packing_4bit_{}", impl_type), size),
                &seq,
                |b, seq| b.iter(|| as_4bit(seq)),
            );
        }
    }

    group.finish();
}

fn bench_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("packing");

    let impl_type = if cfg!(feature = "nosimd") {
        "nosimd"
    } else {
        "simd"
    };

    // Test different sequence lengths
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].iter() {
        let seq = generate_sequence(*size);
        let mut ebuf = Vec::new();
        group.bench_with_input(
            BenchmarkId::new(format!("encoding_2bit_{}", impl_type), size),
            &seq,
            |b, seq| {
                b.iter(|| {
                    ebuf.clear();
                    twobit::encode(seq, &mut ebuf).unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new(format!("encoding_4bit_{}", impl_type), size),
            &seq,
            |b, seq| {
                b.iter(|| {
                    ebuf.clear();
                    fourbit::encode(seq, &mut ebuf).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_unpacking(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpacking");

    let impl_type = if cfg!(feature = "nosimd") {
        "no_simd"
    } else {
        "simd"
    };

    // Test different sequence lengths
    for size in [4, 8, 16, 24, 32].iter() {
        let seq = generate_sequence(*size);

        let packed_2b = as_2bit(&seq).unwrap();
        group.bench_with_input(
            BenchmarkId::new(format!("unpacking_2bit_{}", impl_type), size),
            &packed_2b,
            |b, packed| b.iter(|| from_2bit(*packed, *size, &mut Vec::new()).unwrap()),
        );

        if *size <= 16 {
            let packed_4b = as_4bit(&seq).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("unpacking_4bit_{}", impl_type), size),
                &packed_4b,
                |b, packed| b.iter(|| from_4bit(*packed, *size, &mut Vec::new()).unwrap()),
            );
        }
    }

    group.finish();
}

fn bench_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpacking");

    let impl_type = if cfg!(feature = "nosimd") {
        "no_simd"
    } else {
        "simd"
    };

    // Test different sequence lengths
    let sizes = [32, 128, 512, 2048];
    for size in sizes.iter() {
        let seq = generate_sequence(*size);
        let packed_2bit = twobit::encode_alloc(&seq).unwrap();
        let packed_4bit = fourbit::encode_alloc(&seq).unwrap();
        let mut dbuf = Vec::with_capacity(*size);

        group.bench_with_input(
            BenchmarkId::new(format!("decoding_2bit_{}", impl_type), size),
            &packed_2bit,
            |b, packed| {
                b.iter(|| {
                    dbuf.clear();
                    twobit::decode(packed, *size, &mut dbuf).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("decoding_4bit_{}", impl_type), size),
            &packed_4bit,
            |b, packed| {
                b.iter(|| {
                    dbuf.clear();
                    fourbit::decode(packed, *size, &mut dbuf).unwrap()
                })
            },
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
