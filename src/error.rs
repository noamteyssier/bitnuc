use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidBase(u8),
    SequenceTooLong(usize),
    InvalidLength(usize),
    IndexOutOfBounds {
        index: usize,
        length: usize,
    },
    InvalidRange {
        start: usize,
        end: usize,
        length: usize,
    },
    Unsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidBase(b) => write!(f, "Invalid nucleotide base: {}", b),
            Error::SequenceTooLong(len) => {
                write!(f, "Sequence length {} exceeds maximum", len)
            }
            Error::InvalidLength(len) => write!(f, "Invalid length: {}", len),
            Error::IndexOutOfBounds { index, length } => {
                write!(
                    f,
                    "Index {} out of bounds for sequence of length {}",
                    index, length
                )
            }
            Error::InvalidRange { start, end, length } => {
                write!(
                    f,
                    "Invalid range {}..{} for sequence of length {}",
                    start, end, length
                )
            }
            Error::Unsupported => write!(f, "Unsupported architecture"),
        }
    }
}

impl std::error::Error for Error {}
