use crate::NucleotideError;

pub fn from_2bit(
    packed: u64,
    expected_size: usize,
    sequence: &mut Vec<u8>,
) -> Result<(), NucleotideError> {
    if expected_size > 32 {
        return Err(NucleotideError::InvalidLength(expected_size));
    }

    for i in 0..expected_size {
        let bits = (packed >> (i * 2)) & 0b11;
        let base = match bits {
            0b00 => b'A',
            0b01 => b'C',
            0b10 => b'G',
            0b11 => b'T',
            _ => unreachable!(),
        };
        sequence.push(base);
    }

    Ok(())
}
