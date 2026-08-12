use core::ops::{BitAnd, BitOr, Shr};

use fearless_simd::{Level, Simd, dispatch, prelude::*, u8x16, u32x4, u32x8, u32x16};

use crate::BitnucError;

/// Index = ascii & 6. This table projects back to expected representation.
///
/// | Base | &6 | LUT |
/// | --- | --- | --- |
/// | A   | 0   | 0   |
/// | C   | 2   | 1   |
/// | G   | 6   | 2   |
/// | T   | 4   | 3   |
///
/// Note: all unexpected values project to A
const ENCODE_LUT: [u8; 16] = [
    0, 0, 1, 0, 3, 0, 2, 0, // idx 0=A, 2=C, 4=T, 6=G
    0, 0, 0, 0, 0, 0, 0, 0,
];

/// `pack_lanes` with table-lookup extraction: `& 6` + one block-local byte
/// shuffle replace the `((v>>1) ^ (v>>2)) & 3` hash, cutting the extraction
/// from 4 vector ops to 2 (the trick packed-seq gets for free from its
/// bits-in-place mapping, recovered here for our mapping via the LUT). The
/// tbl output is already exact 2-bit codes, so the cascade needs no mask.
///
/// Pack lanes with table-lookup extraction:
///
/// - Project to SIMD
/// - Calculate index with &6
/// - Project from index to ordinal values
/// - Cascade OR and SHL to pack characters into lowest byte
/// - Truncate to lowest byte and write to out
#[inline(always)]
fn pack_lanes<S, V>(simd: S, chunk: &[u8], lut_block: u8x16<S>, out: &mut [u8])
where
    S: Simd,
    V: SimdNarrow<S>
        + SimdBase<S, Element = u32>
        + Shr<u32, Output = V>
        + BitAnd<Output = V>
        + BitOr<Output = V>,
    V::ByteVector: SimdBase<S, Element = u8, Block = u8x16<S>> + BitAnd<Output = V::ByteVector>,
    V::Narrowed: SimdNarrow<S>,
    <V::Narrowed as SimdNarrow<S>>::Narrowed: SimdBase<S, Element = u8>,
{
    let code: V = {
        let table = V::ByteVector::block_splat(lut_block); // build LUT (optimized out)
        let ascii = V::ByteVector::from_slice(simd, chunk); // map to lanes
        let idx = ascii & V::ByteVector::simd_from(simd, 6u8); // calculate index
        table.swizzle_dyn_within_blocks(idx).bitcast() // reindex codes
    };

    // The cascade of ORs and shifts packs the 2-bit codes into the low 8 bits of each u32 lane.
    let code = {
        let code = code & V::simd_from(simd, 0x03030303u32); // no-op, proves disjointness to LLVM which fuses each `| >>`
        let code = code | (code >> 6);
        code | (code >> 12)
    };

    let bytes = {
        let halves = code.narrow(code); // narrows u32 -> u16 (first `N` lanes are relevant)
        let quads = halves.narrow(halves); // narrows u16 -> u8 (first `N` lanes are relevant)
        quads
    };

    out.copy_from_slice(&bytes.as_slice()[..V::N]);
}

/// Packs 8 ASCII bases into 2 packed bytes using the same bitwise operations
/// as `pack_lanes`.
///
/// This is a fallback for the 8-base case which can be covered by portable_simd's
/// u32x2 but is unavailable in `fearless_simd` because it only covers 128-bit
/// registers. Follows a SIMD within a register (SWAR) approach, using a u64 to
/// hold the 8 bases and performing the same bitwise operations to pack them into 2 bytes.
#[inline(always)]
fn pack_8bp_swar(chunk: &[u8], ebuf: &mut [u8]) {
    let v = u64::from_le_bytes(chunk.try_into().unwrap());

    let code = {
        let r1 = v >> 1; // rotate 1 right
        let r2 = v >> 2; // rotate 2 right
        (r1 ^ r2) & 0x0303_0303_0303_0303 // global bitwise AND to get 2-bit codes
    };

    // OR and shift cascade to pack the 2-bit codes into the low 8 bits of each u32 lane
    let packed = {
        let code = code | (code >> 6);
        code | (code >> 12)
    };

    ebuf[0] = packed as u8; // update first word
    ebuf[1] = (packed >> 32) as u8; // update second word
}

#[inline(always)]
fn encode_inner<S: Simd>(simd: S, seq: &[u8], ebuf: &mut [u8]) {
    let lut_block = u8x16::from_slice(simd, &ENCODE_LUT);
    let mut i = 0; // current index in the sequence
    let mut b = 0; // current byte in the buffer

    // main loop: 64 bases at a time
    while i + 64 <= seq.len() {
        pack_lanes::<S, u32x16<S>>(simd, &seq[i..i + 64], lut_block, &mut ebuf[b..b + 16]);
        i += 64;
        b += 16;
    }

    // at most one 32-base step
    if i + 32 <= seq.len() {
        pack_lanes::<S, u32x8<S>>(simd, &seq[i..i + 32], lut_block, &mut ebuf[b..b + 8]);
        i += 32;
        b += 8;
    }

    // at most one 16-base step
    if i + 16 <= seq.len() {
        pack_lanes::<S, u32x4<S>>(simd, &seq[i..i + 16], lut_block, &mut ebuf[b..b + 4]);
        i += 16;
        b += 4;
    }

    // at most one 8-base step (SWAR keeps the arithmetic hash; tbl has no
    // scalar equivalent and the tail never dominates)
    if i + 8 <= seq.len() {
        pack_8bp_swar(&seq[i..i + 8], &mut ebuf[b..b + 2]);
        i += 8;
        b += 2;
    }

    // scalar tail: <8 remaining bases; these OR into their bytes, so zero
    // exactly the bytes the tail touches (everything before b is fully written)
    if i < seq.len() {
        ebuf[b..seq.len().div_ceil(4)].fill(0);
        for (j, &base) in seq[i..].iter().enumerate() {
            let code = ((base >> 1) ^ (base >> 2)) & 3; // (r1^r2)&3
            ebuf[b + j / 4] |= code << (2 * (j % 4)); // directly place packed code
        }
    }
}

/// Two-bit encodes an input sequence into an encoding buffer
pub fn encode(seq: &[u8], ebuf: &mut [u8]) -> Result<(), BitnucError> {
    let n_bytes = seq.len().div_ceil(4);
    if ebuf.len() < n_bytes {
        return Err(BitnucError::EncodingBufferTooSmall {
            expected: n_bytes,
            actual: ebuf.len(),
        });
    }

    let level = Level::new();
    dispatch!(level, simd => encode_inner(simd, seq, ebuf));
    Ok(())
}

/// Two-bit encodes an input sequence into an encoding buffer
/// which can be resized.
///
/// Used for passing in a reusable vec across sequences with
/// different sizes and you don't want to calculate it.
///
/// # Safety:
/// Uses an uninit-value trick internally but it is safe as
/// values are immediately overwritten and no uninit values
/// are returned to the user.
#[allow(clippy::uninit_vec)]
pub fn encode_resize(seq: &[u8], ebuf: &mut Vec<u8>) {
    let n_bytes = seq.len().div_ceil(4);
    if ebuf.len() < n_bytes {
        ebuf.reserve(n_bytes - ebuf.len());
        unsafe {
            ebuf.set_len(n_bytes); // currently uninit values
        }
    }

    let level = Level::new();
    dispatch!(level, simd => encode_inner(simd, seq, &mut ebuf[..n_bytes]));
}
