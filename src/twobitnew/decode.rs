use core::ops::{BitAnd, BitOr, Shl};

use fearless_simd::{Level, Simd, dispatch, prelude::*, u8x16, u8x32, u8x64};

use crate::BitnucError;

/// The code -> ASCII table; only indices 0-3 are ever selected, the rest pad
/// the vector to the 16 bytes a table-lookup shuffle expects. Wider vectors
/// replicate this per 128-bit block (`block_splat`) so the lookup stays a
/// single-register `tbl`/`pshufb` per block instead of a full-width shuffle.
const DECODE_LUT: [u8; 16] = *b"ACGT\0\0\0\0\0\0\0\0\0\0\0\0";

/// Packed byte -> its four ASCII bases as a little-endian u32, built at
/// compile time. Collapses the scalar tail from per-base to per-packed-byte.
const DECODE_WORDS: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut p = 0;
    while p < 256 {
        let mut word = 0u32;
        let mut k = 0;
        while k < 4 {
            word |= (DECODE_LUT[(p >> (2 * k)) & 3] as u32) << (8 * k);
            k += 1;
        }
        table[p] = word;
        p += 1;
    }
    table
};

/// Decodes packed bytes (one per u32 lane) into ASCII bases, generic over the
/// u32 vector width: the exact inverse of `pack_lanes`. The shift cascade runs
/// backwards (scatter instead of gather), spreading each lane's four 2-bit
/// codes into its four bytes; the cross-lane spill from the shifts lands
/// outside the mask. The block-local byte shuffle (`tbl` on NEON, `pshufb` on
/// x86) then maps every code through the LUT, which `unpack_lanes` replicated
/// into each 128-bit block of the table vector.
#[inline(always)]
fn spread_lanes<S, V>(simd: S, q: V, table: V::ByteVector) -> V::ByteVector
where
    S: Simd,
    V: SimdBase<S, Element = u32> + Shl<u32, Output = V> + BitOr<Output = V> + BitAnd<Output = V>,
{
    let x = q | (q << 12);
    let x = x | (x << 6);
    let codes = x & V::simd_from(simd, 0x03030303u32);
    table.swizzle_dyn_within_blocks(codes.bitcast::<V::ByteVector>())
}

/// Decodes `P::N` packed bytes into `P::N * 4` ASCII bases, generic over the
/// byte vector width: the mirror of `pack_lanes`. Two rounds of `widen` (one
/// register in, two out — narrow's inverse) put each packed byte alone in a
/// u32 lane; each quarter then spreads and table-looks-up back to ASCII.
///
/// `Q` is the twice-widened u32 vector. Naming it as a parameter (pinned by
/// the `Widened = Q` equality) keeps the bounds readable, and `ByteVector = P`
/// tells the compiler that spreading a quarter yields `P` again, so its full
/// `P::N` bytes store straight into `out`.
#[inline(always)]
fn unpack_lanes<S, P, Q>(simd: S, chunk: &[u8], lut_block: u8x16<S>, out: &mut [u8])
where
    S: Simd,
    P: SimdWiden<S> + SimdBase<S, Element = u8, Block = u8x16<S>>,
    P::Widened: SimdWiden<S, Widened = Q>,
    Q: SimdBase<S, Element = u32, ByteVector = P>
        + Shl<u32, Output = Q>
        + BitOr<Output = Q>
        + BitAnd<Output = Q>,
{
    let table = P::block_splat(lut_block);
    let packed = P::from_slice(simd, chunk);
    let (lo, hi) = packed.widen();
    let (q0, q1) = lo.widen();
    let (q2, q3) = hi.widen();

    for (idx, q) in [q0, q1, q2, q3].into_iter().enumerate() {
        let start = idx * P::N;
        let end = start + P::N;
        spread_lanes(simd, q, table).store_slice(&mut out[start..end]);
    }
}

#[inline(always)]
fn decode_inner<S: Simd>(simd: S, ebuf: &[u8], n: usize, seq: &mut [u8]) {
    let lut_block = u8x16::from_slice(simd, &DECODE_LUT);
    let mut i = 0; // current base in the output sequence
    let mut b = 0; // current byte in the encoded buffer

    // main loop: 256 bases at a time
    while i + 256 <= n {
        unpack_lanes::<S, u8x64<S>, _>(simd, &ebuf[b..b + 64], lut_block, &mut seq[i..i + 256]);
        i += 256;
        b += 64;
    }

    // at most one 128-base step
    if i + 128 <= n {
        unpack_lanes::<S, u8x32<S>, _>(simd, &ebuf[b..b + 32], lut_block, &mut seq[i..i + 128]);
        i += 128;
        b += 32;
    }

    // at most one 64-base step
    if i + 64 <= n {
        unpack_lanes::<S, u8x16<S>, _>(simd, &ebuf[b..b + 16], lut_block, &mut seq[i..i + 64]);
        i += 64;
        b += 16;
    }

    // tail: 0..=63 remaining bases, whole packed bytes first (4 bases per
    // table lookup), then the final partial byte per-base
    while i + 4 <= n {
        let word = DECODE_WORDS[ebuf[b] as usize];
        seq[i..i + 4].copy_from_slice(&word.to_le_bytes());
        i += 4;
        b += 1;
    }
    for j in i..n {
        let code = (ebuf[j / 4] >> (2 * (j % 4))) & 0b11;
        seq[j] = DECODE_LUT[code as usize];
    }
}

/// Decodes a 2-bit encoded buffer back into `n` ASCII nucleotides.
pub fn decode(ebuf: &[u8], n: usize, seq: &mut [u8]) -> Result<(), BitnucError> {
    if seq.len() < n {
        return Err(BitnucError::DecodingBufferTooSmall {
            expected: n,
            actual: seq.len(),
        });
    }

    if ebuf.len() < n.div_ceil(4) {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: n.div_ceil(4),
            actual: ebuf.len(),
        });
    }

    let level = Level::new();
    dispatch!(level, simd => decode_inner(simd, ebuf, n, seq));
    Ok(())
}

/// Decodes a 2-bit encoded buffer back into `n` ASCII nucleotides
/// with a resizable `seq` buffer.
///
/// Used when passing in a reusable buffer across multiple sequences and you
/// don't want to pass in slices.
///
/// Safety:
/// Uses an uninit-value trick internally but it is safe as
/// values are immediately overwritten and no uninit values
/// are returned to the user.
pub fn decode_resize(ebuf: &[u8], n: usize, seq: &mut Vec<u8>) -> Result<(), BitnucError> {
    if seq.len() < n {
        let diff = n - seq.len();
        seq.reserve(diff);
        unsafe {
            seq.set_len(n);
        }
    }

    if ebuf.len() < n.div_ceil(4) {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: n.div_ceil(4),
            actual: ebuf.len(),
        });
    }

    let level = Level::new();
    dispatch!(level, simd => decode_inner(simd, ebuf, n, &mut seq[..n]));
    Ok(())
}
