use crate::error::NucleotideError;

#[inline(always)]
pub fn as_2bit(seq: &[u8]) -> Result<u64, NucleotideError> {
    if seq.len() > 32 {
        return Err(NucleotideError::SequenceTooLong(seq.len()));
    }
    let mut packed = 0u64;
    for (i, &base) in seq.iter().enumerate() {
        let bits = match base {
            b'A' | b'a' => 0b00,
            b'C' | b'c' => 0b01,
            b'G' | b'g' => 0b10,
            b'T' | b't' => 0b11,
            invalid => return Err(NucleotideError::InvalidBase(invalid)),
        };
        packed |= (bits as u64) << (i * 2);
    }
    Ok(packed)
}

pub fn encode_internal(sequence: &[u8], ebuf: &mut Vec<u64>) -> Result<(), NucleotideError> {
    // Clear the buffer
    ebuf.clear();

    // Calculate the number of chunks
    let n_chunks = sequence.len().div_ceil(32);

    let mut l_bounds = 0;
    for _ in 0..n_chunks - 1 {
        let r_bounds = l_bounds + 32;
        let chunk = &sequence[l_bounds..r_bounds];

        let bits = as_2bit(chunk)?;
        ebuf.push(bits);
        l_bounds = r_bounds;
    }

    let bits = as_2bit(&sequence[l_bounds..])?;
    ebuf.push(bits);

    Ok(())
}
