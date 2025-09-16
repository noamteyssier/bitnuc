use crate::{fourbit, twobit, Error};

#[derive(Debug, PartialEq, Eq, Clone, Hash, Default, Copy)]
pub enum BitSize {
    #[default]
    Two,
    Four,
}
impl Into<u8> for BitSize {
    fn into(self) -> u8 {
        match self {
            BitSize::Two => 2,
            BitSize::Four => 4,
        }
    }
}
impl BitSize {
    pub fn encode(&self, seq: &[u8], data: &mut Vec<u64>) -> Result<(), Error> {
        match self {
            BitSize::Two => twobit::encode(seq, data),
            BitSize::Four => fourbit::encode(seq, data),
        }
    }
    pub fn decode(&self, data: &[u64], length: usize, buf: &mut Vec<u8>) -> Result<(), Error> {
        match self {
            BitSize::Two => twobit::decode(data, length, buf),
            BitSize::Four => fourbit::decode(data, length, buf),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Default)]
pub struct BitNuc {
    data: Vec<u64>,
    length: usize,
    size: BitSize,
}

impl BitNuc {
    pub fn new(size: BitSize) -> Self {
        BitNuc {
            data: Vec::new(),
            length: 0,
            size,
        }
    }

    pub fn new_2bit() -> Self {
        BitNuc::new(BitSize::Two)
    }

    pub fn new_4bit() -> Self {
        BitNuc::new(BitSize::Four)
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.length = 0;
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn fill(&mut self, seq: &[u8]) -> Result<(), Error> {
        self.clear();
        match self.size {
            BitSize::Two => twobit::encode(seq, &mut self.data),
            BitSize::Four => fourbit::encode(seq, &mut self.data),
        }?;
        self.length = seq.len();
        Ok(())
    }

    pub fn decode_into(&self, buf: &mut Vec<u8>) -> Result<(), Error> {
        match self.size {
            BitSize::Two => twobit::decode(&self.data, self.length, buf),
            BitSize::Four => fourbit::decode(&self.data, self.length, buf),
        }
    }

    pub fn decode_alloc(&self) -> Result<Vec<u8>, Error> {
        let mut buf = Vec::with_capacity(self.length);
        self.decode_into(&mut buf)?;
        Ok(buf)
    }
}
