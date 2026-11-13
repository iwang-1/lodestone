//! On-disk index files: a small, versioned, little-endian binary format with a CRC-32 trailer.
//!
//! Layout of every file: `magic (8 bytes) | format version (u32) | body | crc32 (u32)`, where the CRC
//! covers everything before it. Files are written to a temporary sibling, fsynced, and renamed into
//! place, so a crash during `save` leaves either the old file or the new one, never a torn mix. Loading
//! verifies the magic, the version, and the checksum before decoding, then validates every structural
//! invariant of the decoded index, so a corrupt or truncated file is an error, not a panic or a wrong
//! answer.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

/// Errors from saving or loading an index file.
#[derive(Debug)]
pub enum PersistError {
    /// Reading or writing the file failed.
    Io(io::Error),
    /// The file does not start with the expected magic bytes (wrong file type).
    BadMagic,
    /// The file was written by a format version this build does not read.
    UnsupportedVersion(u32),
    /// The stored checksum does not match the contents (corruption or truncation).
    Checksum,
    /// The contents decode, but violate an invariant of the index.
    Corrupt(String),
}

impl fmt::Display for PersistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistError::Io(e) => write!(f, "index file I/O error: {e}"),
            PersistError::BadMagic => write!(f, "not a lodestone index file (bad magic)"),
            PersistError::UnsupportedVersion(v) => {
                write!(f, "unsupported index format version {v}")
            }
            PersistError::Checksum => write!(f, "index file checksum mismatch"),
            PersistError::Corrupt(m) => write!(f, "corrupt index file: {m}"),
        }
    }
}

impl std::error::Error for PersistError {}

impl From<io::Error> for PersistError {
    fn from(e: io::Error) -> Self {
        PersistError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, PersistError>;

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320), table-driven.
pub fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    let mut crc = !0u32;
    for &b in data {
        crc = TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Little-endian encoder for an index body.
#[derive(Default)]
pub struct Encoder {
    buf: Vec<u8>,
}

impl Encoder {
    pub fn new(magic: &[u8; 8], version: u32) -> Self {
        let mut e = Encoder::default();
        e.buf.extend_from_slice(magic);
        e.u32(version);
        e
    }
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    /// A length-prefixed run of `f32`s.
    pub fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        self.buf.reserve(v.len() * 4);
        for x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    /// A length-prefixed run of `u32`s.
    pub fn u32s(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        self.buf.reserve(v.len() * 4);
        for x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    /// A length-prefixed run of bytes.
    pub fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.buf.extend_from_slice(v);
    }

    /// Append the CRC trailer and write the file atomically: temp sibling, fsync, rename.
    pub fn write_atomic(mut self, path: &Path) -> Result<()> {
        let crc = crc32(&self.buf);
        self.buf.extend_from_slice(&crc.to_le_bytes());
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = Path::new(&tmp);
        {
            let mut f = File::create(tmp)?;
            f.write_all(&self.buf)?;
            f.sync_all()?;
        }
        fs::rename(tmp, path)?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            // Make the rename itself durable. Not supported on every platform; ignore failures there.
            if let Ok(d) = File::open(dir) {
                let _ = d.sync_all();
            }
        }
        Ok(())
    }
}

/// Bounds-checked little-endian decoder over a verified file body.
pub struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// Verify magic, version, and checksum, and position the decoder at the start of the body.
    pub fn open(file: &'a [u8], magic: &[u8; 8], version: u32) -> Result<Self> {
        if file.len() < 8 || &file[..8] != magic {
            return Err(PersistError::BadMagic);
        }
        if file.len() < 8 + 4 + 4 {
            return Err(PersistError::Checksum);
        }
        let (content, trailer) = file.split_at(file.len() - 4);
        let stored = u32::from_le_bytes(trailer.try_into().expect("4 bytes"));
        if crc32(content) != stored {
            return Err(PersistError::Checksum);
        }
        let mut d = Decoder {
            buf: content,
            pos: 8,
        };
        let v = d.u32()?;
        if v != version {
            return Err(PersistError::UnsupportedVersion(v));
        }
        Ok(d)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.buf.len() - self.pos < n {
            return Err(PersistError::Corrupt("unexpected end of data".into()));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    /// A length as `usize`, rejecting values that cannot fit in the remaining bytes at `elem` bytes each.
    fn len(&mut self, elem: usize) -> Result<usize> {
        let n = self.u64()?;
        let remaining = (self.buf.len() - self.pos) as u64;
        if n.checked_mul(elem as u64)
            .is_none_or(|bytes| bytes > remaining)
        {
            return Err(PersistError::Corrupt(format!(
                "length {n} exceeds file size"
            )));
        }
        Ok(n as usize)
    }
    pub fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.len(4)?;
        let raw = self.take(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
            .collect())
    }
    pub fn u32s(&mut self) -> Result<Vec<u32>> {
        let n = self.len(4)?;
        let raw = self.take(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4")))
            .collect())
    }
    pub fn bytes(&mut self) -> Result<Vec<u8>> {
        let n = self.len(1)?;
        Ok(self.take(n)?.to_vec())
    }
    /// Error unless every byte of the body was consumed.
    pub fn finish(self) -> Result<()> {
        if self.pos != self.buf.len() {
            return Err(PersistError::Corrupt("trailing bytes after index".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_known_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn decoder_rejects_lengths_larger_than_the_file() {
        let mut e = Encoder::new(b"TESTFILE", 1);
        e.u64(u64::MAX); // a bogus f32s length
        let mut file = e.buf.clone();
        let crc = crc32(&file);
        file.extend_from_slice(&crc.to_le_bytes());
        let mut d = Decoder::open(&file, b"TESTFILE", 1).unwrap();
        assert!(matches!(d.f32s(), Err(PersistError::Corrupt(_))));
    }
}
