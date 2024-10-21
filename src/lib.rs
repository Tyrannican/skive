//! A byte array slicer that slices data into sized chunks and ZLib compresses them.
//!
//! It takes a `u8` slice and splits it into evenly sized chunks (or less if the last
//! chunk is less than the given size). These chunks are then compressed using the
//! [`ZlibEncoder`] and given an ID. These can then be processed / utilised in a way
//! that fits your purpose.
//!
//! Each generated chunk is in the form a a [`BinBlock`] which holds the compressed data as well as
//! some metadata (block ID, chunk size, compressed size, and block hash). Each block supplies
//! functions which allow the chunk to be turned back into a [`Vec<u8>`] and to be constructed from
//! a `u8` slice as well. Each block checksum is computed as a [`Sha256`] hash which is hex-encoded.
//! When decompressed, this checksum is recomputed to ensure the data is valid.
//!
//! A parallel version is also supplied which uses the [`ThreadPool`] struct to process
//! larger inputs in less time. It behaves the same way as above. There exists an unordered
//! version, [`Slicer::par_slice`], and an ordered version [`Slicer::par_slice_ordered`] which
//! orders the output chunks by their ID.
//!
//! # Bin Block Format
//!
//! A [`BinBlock`] is the output component of the [`Slicer`] operations. It holds the compressed
//! data as well as the following set of metadata:
//!
//! * A Block ID which signifies the Block number (Max value: `2^32 - 1`)
//! * The Block size which is the size of the chunked data (before compression)
//!     * This will be the same for all blocks except the last block in cases where the final chunk
//!     is < `2^n` (Max value: `2^32 - 1`)
//! * The compressed size which is the size of the compressed data. (Max Value: `cmp_data <
//! block_size <= 2^32 -1`)
//! * The [`Sha256`] hash of the uncompressed data. This is used to calculate a checksum.
//! * The compressed data itself (Compressed using ZLib)
//!
//! Each block can be converted to a [`Vec<u8>`] so it can be stored on disk or elsewhere with the
//! [`BinBlock::into_bytes()`] function. The storage format is as follows (Big Endian):
//!
//! * 4 Bytes for the Block ID (`u32`)
//! * 4 Bytes for the Block size (`u32`)
//! * 4 Bytes for the compressed data size (`u32`)
//! * 32 Bytes for the Hash
//! * Remainder of the slice is the compressed data
//!
//! # Examples
//!
//! ## Sequential Operation
//!
//! Here you can chunk the data sequentially which will work well enough for small / medium sizes:
//!
//! ```rust,no_run
//! use skive::Slicer;
//! use std::fs;
//!
//! fn main() -> std::io::Result<()> {
//!     let some_file = fs::read("some-file.pdf")?;
//!     
//!     // We want the data sliced into 2Mb chunks then compressed
//!     let cmp_blocks = Slicer::slice(&some_file, 2 * 1024 * 1024)?;
//!
//!     // Now we can convert the blocks into bytes and send them on their way
//!     for block in cmp_blocks {
//!         let data = block.into_bytes().expect("unable to convert block to bytes");
//!         /* Send them across a network or something */
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Parallel Operation
//!
//! Here you can use the parallel slicer which will slice the data using the [`ThreadPool`] crate.
//! You can specify the number of threads you wish to run concurrently and the pool will queue up
//! operations for you automatically:
//!
//! ```rust,no_run
//! use skive::Slicer;
//! use std::fs;
//!
//! fn main() -> std::io::Result<()> {
//!     let some_large_file = fs::read("some-huge-file-like-a-video.mp4")?;
//!
//!     // We want to slice the data into 4Mb chunks across 8 threads and compresses them in
//!     // parallel
//!     let cmp_blocks = Slicer::par_slice(&some_large_file, 4 * 1024 * 1024, 8)?;
//!
//!     // Now we can convert the blocks into bytes and send them on their way
//!     for block in cmp_blocks {
//!         let data = block.into_bytes().expect("unable to convert block to bytes");
//!         /* Send them across a network or something */
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
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

/// Slices `u8` slices into [`BinBlock`] arrays
pub struct Slicer;

/// A slice of the underlying data that is compressed with ZLib and given an ID and hash
#[derive(Debug)]
#[repr(C)]
pub struct BinBlock {
    /// ID of the block from the original data
    pub id: usize,

    /// Size of the uncompressed data
    pub blk_size: usize,

    /// Size of the compressed data
    pub cmp_size: usize,

    /// Hash of the uncompressed data used for checksum validation
    pub hash: [u8; 32],

    /// The compressed ZLib data
    pub compressed_data: Vec<u8>,
}

impl Slicer {
    /// Sequential slicer of input data.
    ///
    /// Slices the data in `block_size` chunks and compresses it, returning a [`Vec<BinBlock>`]
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

    /// Parallel slicer of input data
    ///
    /// Slices the data concurrently into `block_size` chunks across `num_threads` threads and
    /// compresses it, returning a [`Vec<BinBlock>`]
    ///
    /// The data returned here is in no particular order.
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

    /// Performs the same operation as [`Slicer::par_slice`] but orders the returned blocks by
    /// their Block ID.
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
    /// Creates a new [`BinBlock`] from the input data and block ID.
    ///
    /// The data is compressed and a hash is calculated using [`Sha256`]
    pub fn new(input: &[u8], id: usize) -> std::io::Result<Self> {
        let size = input.len();
        let mut hasher = Sha256::new();
        hasher.update(input);
        let hash: [u8; 32] = hasher.finalize().into();

        let mut compressed_data = Vec::new();
        let mut encoder = ZlibEncoder::new(input, Compression::fast());
        encoder.read_to_end(&mut compressed_data)?;

        Ok(Self {
            id,
            blk_size: size,
            cmp_size: compressed_data.len(),
            hash,
            compressed_data,
        })
    }

    /// Generates the Hex-encoded value of the block hash.
    pub fn checksum(&self) -> String {
        hex::encode(self.hash)
    }

    /// Decompresses the compressed data and returns the original data.
    ///
    /// A validation check is performed here using the checksum to ensure data validity
    pub fn decompress(&self) -> std::io::Result<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(&*self.compressed_data);
        let mut output = Vec::with_capacity(self.blk_size);
        std::io::copy(&mut decoder, &mut output)?;

        let mut hasher = Sha256::new();
        hasher.update(&output);
        let chksum = hex::encode(hasher.finalize());
        assert_eq!(chksum, self.checksum());

        Ok(output)
    }

    /// Converts the [`BinBlock`] into a [`Vec<u8>`] using the block format.
    ///
    /// See documentation [here](index.html#bin-block-format)
    pub fn into_bytes(self) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        let id: [u8; 4] = u32::to_be_bytes(self.id as u32);
        let blk_size: [u8; 4] = u32::to_be_bytes(self.blk_size as u32);
        let cmp_size: [u8; 4] = u32::to_be_bytes(self.compressed_data.len() as u32);

        assert_eq!(self.hash.len(), 32);
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
            .write_all(&self.hash)
            .map_err(|_| "hash write error".to_string())?;
        output
            .write_all(&self.compressed_data)
            .map_err(|_| "compressed data write error".to_string())?;

        Ok(output)
    }

    /// Reads a chunk of data and attempts to construct a [`BinBlock`] from it.
    ///
    /// It reconstructs a [`BinBlock`] accoring to the block format as described
    /// [here](index.html#bin-block-format)
    pub fn from_bytes(input: &[u8]) -> std::io::Result<Self> {
        let mut reader = Bytes::copy_from_slice(input);
        let id = reader.get_u32() as usize;
        let blk_size = reader.get_u32() as usize;
        let cmp_size = reader.get_u32() as usize;

        let mut hash = [0; 32];
        let mut r = reader.reader();
        r.read_exact(&mut hash)?;

        let mut compressed_data = Vec::new();
        r.read_to_end(&mut compressed_data)?;

        assert_eq!(cmp_size, compressed_data.len());

        Ok(Self {
            id,
            blk_size,
            cmp_size,
            hash,
            compressed_data,
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
    fn uneven_slices() {
        let arr = generate_test_array(32 * KB + 9837);
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
            let expected_hash = slice.checksum();
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
