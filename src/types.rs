use crate::{fourbit, twobit, NucleotideError};

#[derive(Debug, PartialEq, Eq, Clone, Hash, Default, Copy)]
pub enum NucSize {
    #[default]
    Two,
    Four,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Default)]
pub struct BitNuc {
    data: Vec<u64>,
    length: usize,
    size: NucSize,
}

impl BitNuc {
    pub fn new(size: NucSize) -> Self {
        BitNuc {
            data: Vec::new(),
            length: 0,
            size,
        }
    }

    pub fn new_2bit() -> Self {
        BitNuc::new(NucSize::Two)
    }

    pub fn new_4bit() -> Self {
        BitNuc::new(NucSize::Four)
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

    pub fn fill(&mut self, seq: &[u8]) -> Result<(), NucleotideError> {
        self.clear();
        match self.size {
            NucSize::Two => twobit::encode(seq, &mut self.data),
            NucSize::Four => fourbit::encode(seq, &mut self.data),
        }?;
        self.length = seq.len();
        Ok(())
    }

    pub fn decode_into(&self, buf: &mut Vec<u8>) -> Result<(), NucleotideError> {
        match self.size {
            NucSize::Two => twobit::decode(&self.data, self.length, buf),
            NucSize::Four => fourbit::decode(&self.data, self.length, buf),
        }
    }

    pub fn decode_alloc(&self) -> Result<Vec<u8>, NucleotideError> {
        let mut buf = Vec::with_capacity(self.length);
        self.decode_into(&mut buf)?;
        Ok(buf)
    }
}
