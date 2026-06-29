//! Readers for the `.fvecs` / `.ivecs` formats used by SIFT1M (corpus-texmex)
//! and many other classic ANN datasets.
//!
//! Layout is a flat sequence of records, each:
//!   `[i32 dim][dim * <f32 | i32>]`   (little-endian)
//! The leading `dim` is repeated on every record; we assume it is constant
//! across the file (true for every dataset in this family).

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Float vectors stored contiguously, row-major: row `i` is
/// `data[i * dim .. (i + 1) * dim]`.
pub struct Vectors {
    pub data: Vec<f32>,
    pub dim: usize,
}

impl Vectors {
    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Integer vectors (used for ground-truth neighbor indices in `.ivecs`).
pub struct IntVectors {
    pub data: Vec<i32>,
    pub dim: usize,
}

impl IntVectors {
    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[i32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

fn read_all(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buf)?;
    Ok(buf)
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Parse the record framing common to `.fvecs` and `.ivecs`. Returns the
/// dimension and the count of records, validating the file is well-formed.
fn frame(buf: &[u8]) -> io::Result<(usize, usize)> {
    if buf.is_empty() {
        return Ok((0, 0));
    }
    if buf.len() < 4 {
        return Err(bad("file too short to contain a dimension header"));
    }
    let dim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if dim == 0 {
        return Err(bad("first record declares dimension 0"));
    }
    let record = 4 + dim * 4; // bytes per record (header + dim components)
    if buf.len() % record != 0 {
        return Err(bad("file size is not a multiple of the record size"));
    }
    Ok((dim, buf.len() / record))
}

/// Read an `.fvecs` file (32-bit float components).
pub fn read_fvecs(path: impl AsRef<Path>) -> io::Result<Vectors> {
    let buf = read_all(path)?;
    let (dim, n) = frame(&buf)?;
    let record = 4 + dim * 4;
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let off = i * record;
        let d = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        if d != dim {
            return Err(bad("record dimension differs from first record"));
        }
        for c in buf[off + 4..off + record].chunks_exact(4) {
            data.push(f32::from_le_bytes(c.try_into().unwrap()));
        }
    }
    Ok(Vectors { data, dim })
}

/// Write integer vectors as `.ivecs` (per row: `[i32 len][len × i32]`).
/// Used to persist generated ground truth (top-k neighbor ids per query).
pub fn write_ivecs(path: impl AsRef<Path>, rows: &[Vec<u32>]) -> io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    for r in rows {
        buf.extend_from_slice(&(r.len() as i32).to_le_bytes());
        for &v in r {
            buf.extend_from_slice(&(v as i32).to_le_bytes());
        }
    }
    std::fs::write(path, buf)
}

/// Read an `.ivecs` file (32-bit signed int components).
pub fn read_ivecs(path: impl AsRef<Path>) -> io::Result<IntVectors> {
    let buf = read_all(path)?;
    let (dim, n) = frame(&buf)?;
    let record = 4 + dim * 4;
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let off = i * record;
        let d = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        if d != dim {
            return Err(bad("record dimension differs from first record"));
        }
        for c in buf[off + 4..off + record].chunks_exact(4) {
            data.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
    }
    Ok(IntVectors { data, dim })
}
