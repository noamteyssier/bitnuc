mod packing;
mod unpacking;

pub use packing::{as_4bit, encode, encode_alloc};
pub use unpacking::{decode, from_4bit, from_4bit_alloc};

#[cfg(test)]
mod testing {
    use crate::Error;

    use super::*;

    #[test]
    fn test_roundtrip_4bit() -> Result<(), Error> {
        let input: &[u8] = b"ACGT";
        let mut output = Vec::new();
        let packed = as_4bit(input)?;
        from_4bit(packed, input.len(), &mut output)?;
        assert_eq!(input, &output);
        Ok(())
    }

    #[test]
    fn test_roundtrip_4bit_16bp() -> Result<(), Error> {
        let input: &[u8] = b"ACGTACGTACGTACGT";
        let mut output = Vec::new();
        let packed = as_4bit(input)?;
        from_4bit(packed, input.len(), &mut output)?;
        assert_eq!(input, &output);
        Ok(())
    }

    #[test]
    fn test_roundtrip_4bit_with_ambiguous() -> Result<(), Error> {
        let input: &[u8] = b"ACGTNACGTN";
        let mut output = Vec::new();
        let packed = as_4bit(input)?;
        from_4bit(packed, input.len(), &mut output)?;
        assert_eq!(input, &output);
        Ok(())
    }

    #[test]
    fn test_roundtrip_4bit_17bp() -> Result<(), Error> {
        let input: &[u8] = b"ACGTACGTACGTACGTA";
        let packed = as_4bit(input);
        assert!(packed.is_err());
        Ok(())
    }

    #[test]
    fn test_roundtrip_encode_decode_4bit_16bp() -> Result<(), Error> {
        let input: &[u8] = b"ACGTACGTACGTACGT";
        let mut ebuf = Vec::new();
        let mut dbuf = Vec::new();
        encode(input, &mut ebuf)?;
        decode(&ebuf, input.len(), &mut dbuf)?;
        assert_eq!(input, &dbuf);
        Ok(())
    }

    #[test]
    fn test_roundtrip_encode_decode_4bit_17bp() -> Result<(), Error> {
        let input: &[u8] = b"ACGTACGTACGTACGTA";
        let mut ebuf = Vec::new();
        let mut dbuf = Vec::new();
        encode(input, &mut ebuf)?;
        decode(&ebuf, input.len(), &mut dbuf)?;
        assert_eq!(input, &dbuf);
        Ok(())
    }

    #[test]
    fn test_roundtrip_encode_decode_4bit_128bp() -> Result<(), Error> {
        let input = vec![b'N'; 128];
        let mut ebuf = Vec::new();
        let mut dbuf = Vec::new();
        encode(&input, &mut ebuf)?;
        decode(&ebuf, input.len(), &mut dbuf)?;
        assert_eq!(&input, &dbuf);
        Ok(())
    }

    #[test]
    fn test_roundtrip_encode_decode_4bit_1024bp() -> Result<(), Error> {
        let input = vec![b'N'; 1024];
        let mut ebuf = Vec::new();
        let mut dbuf = Vec::new();
        encode(&input, &mut ebuf)?;
        decode(&ebuf, input.len(), &mut dbuf)?;
        assert_eq!(&input, &dbuf);
        Ok(())
    }
}
