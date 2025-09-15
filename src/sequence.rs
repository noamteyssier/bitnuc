use crate::error::NucleotideError;
use crate::twobit::encode;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct PackedSequence {
    data: Vec<u64>,
    length: usize,
}

impl PackedSequence {
    /// Creates a new `PackedSequence` from a byte slice containing nucleotides.
    ///
    /// The input sequence must contain only valid nucleotides (A, C, G, T, case insensitive).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"ACGT")?;
    /// assert_eq!(seq.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `NucleotideError::InvalidBase` if the input contains non-ACGT characters:
    ///
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() {
    /// let result = PackedSequence::new(b"ACGN");
    /// assert!(result.is_err());
    /// # }
    /// ```
    pub fn new(seq: &[u8]) -> Result<Self, NucleotideError> {
        let mut data = Vec::new();
        if seq.is_empty() {
            // Skip encoding for empty sequences
        } else {
            encode(seq, &mut data)?;
        }

        Ok(Self {
            data,
            length: seq.len(),
        })
    }

    /// Returns the number of bases in the sequence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"ACGT")?;
    /// assert_eq!(seq.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the sequence contains no bases.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"")?;
    /// assert!(seq.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the nucleotide at the given position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"ACGT")?;
    /// assert_eq!(seq.get(0)?, b'A');
    /// assert_eq!(seq.get(3)?, b'T');
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `NucleotideError::IndexOutOfBounds` if the index is past the end of the sequence:
    ///
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() {
    /// # let seq = PackedSequence::new(b"ACGT").unwrap();
    /// let result = seq.get(4);
    /// assert!(result.is_err());
    /// # }
    /// ```
    pub fn get(&self, index: usize) -> Result<u8, NucleotideError> {
        if index >= self.length {
            return Err(NucleotideError::IndexOutOfBounds {
                index,
                length: self.length,
            });
        }

        let chunk_idx = index / 32;
        let bit_idx = (index % 32) * 2;
        let bits = (self.data[chunk_idx] >> bit_idx) & 0b11;

        Ok(match bits {
            0b00 => b'A',
            0b01 => b'C',
            0b10 => b'G',
            0b11 => b'T',
            _ => unreachable!(),
        })
    }

    /// Returns a subsequence within the given range.
    ///
    /// The range is exclusive of the end bound, matching Rust's standard range behavior.
    ///
    /// # Examples
    ///
    /// Basic slicing:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"ACGTACGT")?;
    ///
    /// // Get middle section
    /// assert_eq!(seq.slice(1..5)?, b"CGTA");
    ///
    /// // Get prefix
    /// assert_eq!(seq.slice(0..3)?, b"ACG");
    ///
    /// // Get suffix
    /// assert_eq!(seq.slice(5..8)?, b"CGT");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Empty slices are allowed:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"ACGT")?;
    /// assert_eq!(seq.slice(2..2)?, b"");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `NucleotideError::InvalidRange` in these cases:
    ///
    /// Start index greater than end index:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() {
    /// # let seq = PackedSequence::new(b"ACGT").unwrap();
    /// let result = seq.slice(3..2);
    /// assert!(result.is_err());
    /// # }
    /// ```
    ///
    /// Range extends past end of sequence:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() {
    /// # let seq = PackedSequence::new(b"ACGT").unwrap();
    /// let result = seq.slice(2..5);
    /// assert!(result.is_err());
    /// # }
    /// ```
    pub fn slice(&self, range: Range<usize>) -> Result<Vec<u8>, NucleotideError> {
        if range.start > range.end || range.end > self.length {
            return Err(NucleotideError::InvalidRange {
                start: range.start,
                end: range.end,
                length: self.length,
            });
        }

        let mut result = Vec::with_capacity(range.end - range.start);
        for i in range {
            result.push(self.get(i)?);
        }
        Ok(result)
    }

    /// Converts the entire packed sequence back to a vector of bytes.
    ///
    /// This is equivalent to `slice(0..len())` but may be more efficient
    /// for full sequence conversion.
    ///
    /// # Examples
    ///
    /// Basic conversion:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"ACGT")?;
    /// assert_eq!(seq.to_vec()?, b"ACGT");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Working with longer sequences:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let original = b"ACGTACGTACGTACGT".to_vec();
    /// let seq = PackedSequence::new(&original)?;
    /// assert_eq!(seq.to_vec()?, original);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Empty sequences:
    /// ```rust
    /// use bitnuc::PackedSequence;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let seq = PackedSequence::new(b"")?;
    /// assert_eq!(seq.to_vec()?, b"");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance
    ///
    /// This method allocates a new vector and performs a full copy of the sequence.
    /// For frequent access to subsequences, consider using `slice()` or individual
    /// base access via `get()` instead.
    pub fn to_vec(&self) -> Result<Vec<u8>, NucleotideError> {
        self.slice(0..self.length)
    }
}
#[cfg(test)]
mod tests {
    use crate::error::NucleotideError;
    use crate::sequence::PackedSequence;
    use std::collections::HashSet;

    #[test]
    fn test_new_sequence() {
        let seq = PackedSequence::new(b"ACGT").unwrap();
        assert_eq!(seq.len(), 4);
        assert_eq!(seq.to_vec().unwrap(), b"ACGT");
    }

    #[test]
    fn test_sequence_get() {
        let seq = PackedSequence::new(b"ACGT").unwrap();
        assert_eq!(seq.get(0).unwrap(), b'A');
        assert_eq!(seq.get(1).unwrap(), b'C');
        assert_eq!(seq.get(2).unwrap(), b'G');
        assert_eq!(seq.get(3).unwrap(), b'T');
    }

    #[test]
    fn test_sequence_get_out_of_bounds() {
        let seq = PackedSequence::new(b"ACGT").unwrap();
        assert!(matches!(
            seq.get(4),
            Err(NucleotideError::IndexOutOfBounds {
                index: 4,
                length: 4
            })
        ));
    }

    #[test]
    fn test_sequence_slice() {
        let seq = PackedSequence::new(b"ACGTACGT").unwrap();
        assert_eq!(seq.slice(1..5).unwrap(), b"CGTA");
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_sequence_invalid_slice() {
        let seq = PackedSequence::new(b"ACGT").unwrap();
        assert!(matches!(
            seq.slice(3..2),
            Err(NucleotideError::InvalidRange {
                start: 3,
                end: 2,
                length: 4
            })
        ));
    }

    #[test]
    fn test_sequence_equality() {
        let seq1 = PackedSequence::new(b"ACGT").unwrap();
        let seq2 = PackedSequence::new(b"ACGT").unwrap();
        let seq3 = PackedSequence::new(b"TGCA").unwrap();

        assert_eq!(seq1, seq2);
        assert_ne!(seq1, seq3);
    }

    #[test]
    fn test_sequence_hashability() {
        let mut set = HashSet::new();
        let seq1 = PackedSequence::new(b"ACGT").unwrap();
        let seq2 = PackedSequence::new(b"ACGT").unwrap();
        let seq3 = PackedSequence::new(b"TGCA").unwrap();

        set.insert(seq1.clone());
        assert!(set.contains(&seq2));
        assert!(!set.contains(&seq3));
    }
}
