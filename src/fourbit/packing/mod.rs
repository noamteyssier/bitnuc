mod naive;

#[cfg(target_arch = "x86_64")]
mod avx;

#[cfg(target_arch = "aarch64")]
mod neon;

pub use crate::Error;

pub fn as_4bit(seq: &[u8]) -> Result<u64, Error> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        neon::as_4bit(seq)
    } else {
        naive::as_4bit(seq)
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx2") {
        // Use 256 bit instructions
        avx::as_4bit(seq)
    } else {
        // Use 64 bit instructions
        naive::as_4bit(seq)
    }

    // Fall back to naive implemention if:
    // - SIMD is disabled via nosimd feature
    // - or SIMD feature is not enabled
    // - or required CPU features aren't availabe
    #[cfg(any(
        feature = "nosimd",
        all(not(target_arch = "aarch64"), not(target_arch = "x86_64"),)
    ))]
    naive::as_4bit(seq)
}

pub fn encode(seq: &[u8], ebuf: &mut Vec<u64>) -> Result<(), Error> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        neon::encode_internal(seq, ebuf)
    } else {
        naive::encode_internal(seq, ebuf)
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx2") {
        // Use 256 bit instructions
        avx::encode_internal(seq, ebuf)
    } else {
        // Use 64 bit instructions
        naive::encode_internal(seq, ebuf)
    }

    // Fall back to naive implemention if:
    // - SIMD is disabled via nosimd feature
    // - or SIMD feature is not enabled
    // - or required CPU features aren't availabe
    #[cfg(any(
        feature = "nosimd",
        all(not(target_arch = "aarch64"), not(target_arch = "x86_64"),)
    ))]
    naive::encode_internal(seq, ebuf)
}

pub fn encode_alloc(seq: &[u8]) -> Result<Vec<u64>, Error> {
    let mut ebuf = Vec::new();
    encode(seq, &mut ebuf)?;
    Ok(ebuf)
}
