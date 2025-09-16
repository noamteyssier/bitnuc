use crate::Error;

pub fn from_4bit(packed: u64, expected_size: usize, sequence: &mut Vec<u8>) -> Result<(), Error> {
    if expected_size > 32 {
        return Err(Error::InvalidLength(expected_size));
    }

    for i in 0..expected_size {
        let bits = (packed >> (i * 4)) & 0xF;
        let base = match bits {
            0b0000 => b'A',
            0b0001 => b'C',
            0b0010 => b'G',
            0b0011 => b'T',
            _ => b'N',
        };
        sequence.push(base);
    }

    Ok(())
}

pub fn decode_internal(ebuf: &[u64], len: usize, dbuf: &mut Vec<u8>) -> Result<(), Error> {
    let n_chunks = len.div_ceil(16);
    let rem = match len % 16 {
        0 => 16, // full chunk
        rem => rem,
    };

    // Ensure the packed data is the correct length
    if n_chunks != ebuf.len() {
        return Err(Error::InvalidLength(len));
    }

    // Process all chunks except the last one
    ebuf.iter()
        .take(n_chunks - 1)
        .try_for_each(|component| from_4bit(*component, 16, dbuf))?;

    // Process last chunk with remainder
    ebuf.get(n_chunks - 1)
        .map_or(Err(Error::InvalidLength(len)), |&component| {
            from_4bit(component, rem, dbuf)
        })?;

    Ok(())
}
