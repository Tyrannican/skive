use bytes::{Buf, Bytes};
use flate2::{
    read::{ZlibDecoder, ZlibEncoder},
    Compression,
};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    sync::{mpsc::channel, Arc},
};
use threadpool::ThreadPool;

pub struct Slicer;

#[derive(Debug)]
pub struct BinBlock {
    pub zlib_data: Vec<u8>,
    pub chksum: String,
    pub blk_size: usize,
    pub id: usize,
}

impl Slicer {
    pub fn slice(input: &[u8], block_size: usize) -> std::io::Result<Vec<BinBlock>> {
        let mut slices = Vec::new();
        for (idx, slice) in input.chunks(block_size).enumerate() {
            match BinBlock::new(slice, idx) {
                Ok(blk) => slices.push(blk),
                Err(e) => return Err(e),
            }
        }

        Ok(slices)
    }

    pub fn par_slice(
        input: &[u8],
        block_size: usize,
        num_threads: usize,
    ) -> std::io::Result<Vec<BinBlock>> {
        let (tx, rx) = channel::<std::io::Result<BinBlock>>();
        let tx = Arc::new(tx);
        let pool = ThreadPool::new(num_threads);

        for (idx, block) in input.chunks(block_size).enumerate() {
            let tx_clone = Arc::clone(&tx);
            let block = block.to_vec();
            pool.execute(move || {
                tx_clone
                    .send(BinBlock::new(&block, idx))
                    .expect("error sending bin block to channel");
                drop(tx_clone);
            });
        }

        drop(tx);

        let mut zipped_blocks = Vec::with_capacity(input.len() / block_size);
        while let Ok(blk) = rx.recv() {
            match blk {
                Ok(blk) => zipped_blocks.push(blk),
                Err(e) => return Err(e),
            }
        }

        Ok(zipped_blocks)
    }

    pub fn par_slice_ordered(
        input: &[u8],
        block_size: usize,
        num_threads: usize,
    ) -> std::io::Result<Vec<BinBlock>> {
        let mut blocks = Self::par_slice(input, block_size, num_threads)?;
        blocks.sort_by_key(|b| b.id);

        Ok(blocks)
    }
}

impl BinBlock {
    pub fn new(input: &[u8], id: usize) -> std::io::Result<Self> {
        let size = input.len();
        let mut hasher = Sha256::new();
        hasher.update(input);
        let chksum = hex::encode(hasher.finalize());

        let mut zlib_data = Vec::new();
        let mut encoder = ZlibEncoder::new(input, Compression::fast());
        encoder.read_to_end(&mut zlib_data)?;

        Ok(Self {
            zlib_data,
            chksum,
            blk_size: size,
            id,
        })
    }

    pub fn decompress(&self) -> std::io::Result<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(&*self.zlib_data);
        let mut output = Vec::with_capacity(self.blk_size);
        std::io::copy(&mut decoder, &mut output)?;

        let mut hasher = Sha256::new();
        hasher.update(&output);
        let chksum = hex::encode(hasher.finalize());
        assert_eq!(chksum, self.chksum);

        Ok(output)
    }

    pub fn into_bytes(self) -> Result<Vec<u8>, String> {
        // 4 bytes for the ID
        // 4 Bytes for the block size
        // 4 bytes for the length of the compressed data
        // 32 Bytes for the hash
        // Rest for the data
        let mut output = Vec::new();
        let id: [u8; 4] = u32::to_be_bytes(self.id as u32);
        let blk_size: [u8; 4] = u32::to_be_bytes(self.blk_size as u32);
        let cmp_size: [u8; 4] = u32::to_be_bytes(self.zlib_data.len() as u32);
        let hash = match hex::decode(self.chksum) {
            Ok(hash) => hash,
            Err(e) => return Err(e.to_string()),
        };

        assert_eq!(hash.len(), 32);
        output
            .write_all(&id)
            .map_err(|_| "id write error".to_string())?;
        output
            .write_all(&blk_size)
            .map_err(|_| "block size write error".to_string())?;
        output
            .write_all(&cmp_size)
            .map_err(|_| "compressed size write error".to_string())?;
        output
            .write_all(&hash)
            .map_err(|_| "hash write error".to_string())?;
        output
            .write_all(&self.zlib_data)
            .map_err(|_| "compressed data write error".to_string())?;

        Ok(output)
    }

    pub fn from_bytes(input: &[u8]) -> std::io::Result<Self> {
        let mut reader = Bytes::copy_from_slice(input);
        let id = reader.get_u32() as usize;
        let blk_size = reader.get_u32() as usize;
        let cmp_size = reader.get_u32() as usize;
        let mut hash = Vec::with_capacity(32);
        for _ in 0..32 {
            hash.push(reader.get_u8());
        }

        let mut zlib_data = Vec::new();
        let mut r = reader.reader();
        r.read_to_end(&mut zlib_data)?;

        assert_eq!(cmp_size, zlib_data.len());

        Ok(Self {
            id,
            blk_size,
            chksum: hex::encode(hash),
            zlib_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    const GB: usize = 1024 * 1024 * 1024;

    fn generate_test_array(size: usize) -> Vec<u8> {
        let mut test: Vec<u8> = Vec::with_capacity(size);
        for _ in 0..size {
            test.push(thread_rng().gen())
        }

        test
    }

    fn par_generate_test_array(size: usize, num_threads: usize) -> Vec<u8> {
        let mut test = Vec::with_capacity(size);
        let (tx, rx) = channel::<Vec<u8>>();
        let tx = Arc::new(tx);
        let pool = ThreadPool::new(num_threads);
        for _ in 0..num_threads {
            let tx_clone = Arc::clone(&tx);
            pool.execute(move || {
                let internal_size = size / num_threads;
                let mut slice: Vec<u8> = Vec::with_capacity(internal_size);
                for _ in 0..internal_size {
                    slice.push(thread_rng().gen())
                }
                tx_clone.send(slice).expect("error sending to channel");
                drop(tx_clone);
            });
        }

        drop(tx);
        while let Ok(slice) = rx.recv() {
            test.extend_from_slice(&slice);
        }

        test
    }

    #[test]
    fn slices_fine() {
        let arr = generate_test_array(32 * KB);
        let slices = Slicer::slice(&arr, 2 * KB).unwrap();
        assert_eq!(slices.len(), 16);
    }

    #[test]
    fn slices_in_order() {
        let arr = generate_test_array(32 * KB);
        let slices = Slicer::slice(&arr, 2 * KB).unwrap();
        for (idx, slice) in slices.into_iter().enumerate() {
            assert_eq!(idx, slice.id);
            assert_eq!(slice.blk_size, 2 * KB);
        }
    }

    #[test]
    fn slices_to_bytes_and_back_again() {
        let arr = generate_test_array(32 * KB);
        let slices = Slicer::slice(&arr, 2 * KB).unwrap();
        for (idx, slice) in slices.into_iter().enumerate() {
            let og_decomp = slice.decompress().unwrap();
            let b = slice.into_bytes().unwrap();
            let reconstructed = BinBlock::from_bytes(&b).unwrap();
            let r_decomp = reconstructed.decompress().unwrap();
            assert_eq!(idx, reconstructed.id);
            assert_eq!(og_decomp, r_decomp);
        }
    }

    #[test]
    fn decodes_slice_fine() {
        let arr = generate_test_array(32 * KB);
        let slices = Slicer::slice(&arr, 2 * KB).unwrap();
        for slice in slices.into_iter() {
            let expected_hash = slice.chksum.clone();
            let decoded = slice.decompress().expect("error decoding chunk");
            assert_eq!(decoded.len(), 2 * KB);

            let mut hasher = Sha256::new();
            hasher.update(decoded);
            let hash = hex::encode(hasher.finalize());
            assert_eq!(hash, expected_hash);
        }
    }

    #[test]
    fn par_slice_fine() {
        let arr = par_generate_test_array(2 * GB, 16);
        let slices = Slicer::par_slice(&arr, 4 * MB, 12);
        assert!(slices.is_ok());
        let slices = slices.unwrap();
        let size = slices.len();
        assert_eq!(slices.len(), arr.len() / (4 * MB));
        for slice in slices.into_iter() {
            assert!(slice.id < size, "slice id: {}", slice.id);
            assert_eq!(slice.blk_size, 4 * MB);
        }
    }

    #[test]
    fn par_slice_fine_ordered() {
        let arr = par_generate_test_array(2 * GB, 16);
        let slices = Slicer::par_slice_ordered(&arr, 4 * MB, 12);
        assert!(slices.is_ok());
        let slices = slices.unwrap();
        assert_eq!(slices.len(), arr.len() / (4 * MB));
        for (idx, slice) in slices.into_iter().enumerate() {
            assert_eq!(idx, slice.id);
            assert_eq!(slice.blk_size, 4 * MB);
        }
    }
}
