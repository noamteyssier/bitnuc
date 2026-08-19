use thiserror::Error;

#[derive(Error, Debug)]
pub enum BitnucError {
    #[error("Failure: {0}")]
    DynamicError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[error(
        "Encoding buffer is too small - expected at least {expected} bytes, but got {actual} bytes"
    )]
    EncodingBufferTooSmall { expected: usize, actual: usize },

    #[error(
        "Decoding buffer is too small - expected at least {expected} bytes, but got {actual} bytes"
    )]
    DecodingBufferTooSmall { expected: usize, actual: usize },

    #[error("Sequence length {0} exceeds the 32-base maximum for u64 packing")]
    SequenceTooLong(usize),

    #[error("Invalid length: {0}")]
    InvalidLength(usize),

    #[error(
        "Pairwise distance buffer is too small - expected at least {expected} entries, but got {actual} entries"
    )]
    PairwiseDistanceBufferTooSmall { expected: usize, actual: usize },

    #[error("Encoding buffers are different lengths - u_len: {u_len}, v_len: {v_len}")]
    EncodingBuffersAreDifferentLengths { u_len: usize, v_len: usize },
}
