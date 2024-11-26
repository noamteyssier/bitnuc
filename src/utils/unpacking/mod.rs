use crate::NucleotideError;

/// Converts a 2-bit packed representation back into a nucleotide sequence.
///
/// This function reverses the packing performed by `as_2bit`.
///
/// # Arguments
///
/// * `packed` - A u64 containing the 2-bit packed sequence
/// * `expected_size` - The number of bases to unpack
///
/// # Returns
///
/// Returns a `Vec<u8>` containing the ASCII sequence.
///
/// # Errors
///
/// Returns `NucleotideError::InvalidLength` if `expected_size` is greater than 32
/// (as a u64 can only store 32 * 2 bits).
///
/// # Examples
///
/// Basic unpacking:
/// ```rust
/// use bitnuc::{as_2bit, from_2bit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Pack and unpack
/// let packed = as_2bit(b"ACGT")?;
/// let unpacked = from_2bit(packed, 4)?;
/// assert_eq!(&unpacked, b"ACGT");
///
/// // Partial unpacking
/// let partial = from_2bit(packed, 2)?;
/// assert_eq!(&partial, b"AC");
/// # Ok(())
/// # }
/// ```
///
/// Error handling:
/// ```rust
/// use bitnuc::{from_2bit, NucleotideError};
///
/// # fn main() {
/// // Length too long
/// assert!(matches!(
///     from_2bit(0, 33),
///     Err(NucleotideError::InvalidLength(33))
/// ));
/// # }
/// ```
///
/// # Implementation Details
///
/// The bases are unpacked from least significant to most significant bits:
/// ```rust
/// use bitnuc::{as_2bit, from_2bit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let packed = 0b11100100; // "ACGT" in 2-bit encoding
/// let seq = from_2bit(packed, 4)?;
/// assert_eq!(seq[0], b'A'); // 00
/// assert_eq!(seq[1], b'C'); // 01
/// assert_eq!(seq[2], b'G'); // 10
/// assert_eq!(seq[3], b'T'); // 11
/// # Ok(())
/// # }
/// ```
pub fn from_2bit(packed: u64, expected_size: usize) -> Result<Vec<u8>, NucleotideError> {
    if expected_size > 32 {
        return Err(NucleotideError::InvalidLength(expected_size));
    }

    let mut sequence = Vec::with_capacity(expected_size);

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

    Ok(sequence)
}

#[cfg(test)]
mod testing {
    use super::*;

    #[test]
    fn test_from_2bit_valid_sequence() {
        let tests = vec![
            (0b11100100, 4, b"ACGT"),
            (0b00000000, 4, b"AAAA"),
            (0b11111111, 4, b"TTTT"),
        ];

        for (input, size, expected) in tests {
            assert_eq!(from_2bit(input, size).unwrap(), expected);
        }
    }
}