mod naive;

#[cfg(target_arch = "x86_64")]
mod avx;

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::Error;

pub fn from_4bit(ebuf: u64, len: usize, dbuf: &mut Vec<u8>) -> Result<(), Error> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon::from_4bit_simd(ebuf, len, dbuf) }
    } else {
        naive::from_4bit(ebuf, len, dbuf)
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx2") {
        unsafe { avx::from_4bit_simd(ebuf, len, dbuf) }
    } else {
        naive::from_4bit(ebuf, len, dbuf)
    }

    // Fall back to naive implemention if:
    // - SIMD is disabled via nosimd feature
    // - or SIMD feature is not enabled
    // - or required CPU features aren't availabe
    #[cfg(any(
        feature = "nosimd",
        all(not(target_arch = "aarch64"), not(target_arch = "x86_64"),)
    ))]
    naive::from_4bit(ebuf, len, dbuf)
}

pub fn from_4bit_alloc(packed: u64, len: usize) -> Result<Vec<u8>, Error> {
    let mut dbuf = Vec::default();
    from_4bit(packed, len, &mut dbuf)?;
    Ok(dbuf)
}

pub fn decode(ebuf: &[u64], len: usize, dbuf: &mut Vec<u8>) -> Result<(), Error> {
    #[cfg(all(target_arch = "aarch64", not(feature = "nosimd")))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon::decode_internal(ebuf, len, dbuf) }
    } else {
        naive::decode_internal(ebuf, len, dbuf)
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "nosimd")))]
    if is_x86_feature_detected!("avx2") {
        unsafe { avx::decode_internal(ebuf, len, dbuf) }
    } else {
        naive::decode_internal(ebuf, len, dbuf)
    }

    // Fall back to naive implemention if:
    // - SIMD is disabled via nosimd feature
    // - or SIMD feature is not enabled
    // - or required CPU features aren't availabe
    #[cfg(any(
        feature = "nosimd",
        all(not(target_arch = "aarch64"), not(target_arch = "x86_64"),)
    ))]
    naive::decode_internal(ebuf, len, dbuf)
}
