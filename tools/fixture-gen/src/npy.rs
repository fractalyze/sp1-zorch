//! NumPy `.npy` / `.npz` writers emitting raw **Montgomery** `u32` — the on-disk
//! form the committed `gpu_fibonacci` fixtures use (device `.bin` buffers are
//! already Montgomery, and `KoalaBear` is a transparent Montgomery `u32`). This
//! deliberately differs from the openvm reference generator, which emits
//! *canonical* u32; here the byte-match compares Montgomery limbs directly
//! (see the repo `CLAUDE.md`: "Compare Montgomery-form u32 bytes directly").

use std::fs;
use std::io::Write;
use std::path::Path;

use slop_algebra::AbstractExtensionField;
use sp1_gpu_utils::{Ext, Felt};

/// Raw Montgomery `u32` of a base-field element. Mirrors the fork's
/// `transmute_copy::<Felt, u32>` (`KoalaBear` is `#[repr(transparent)]` over its
/// Montgomery `u32` representation), so the emitted bytes match the device
/// `.bin` buffers exactly.
#[inline]
pub fn felt_mont_u32(f: Felt) -> u32 {
    const _: () = assert!(core::mem::size_of::<Felt>() == core::mem::size_of::<u32>());
    // SAFETY: `Felt` is a transparent Montgomery `u32`; sizes asserted equal above.
    unsafe { core::mem::transmute_copy::<Felt, u32>(&f) }
}

/// Extension-field element → its 4 Montgomery base limbs, in basis order.
#[inline]
pub fn ext_mont_limbs(e: Ext) -> [u32; 4] {
    let s = <Ext as AbstractExtensionField<Felt>>::as_base_slice(&e);
    debug_assert_eq!(s.len(), 4, "Ext is degree 4 over KoalaBear");
    core::array::from_fn(|i| felt_mont_u32(s[i]))
}

/// Flatten a slice of EF values to row-major Montgomery limbs `(n, 4)`.
pub fn ext_rows_mont(es: &[Ext]) -> Vec<u32> {
    let mut out = Vec::with_capacity(es.len() * 4);
    for &e in es {
        out.extend_from_slice(&ext_mont_limbs(e));
    }
    out
}

/// Flatten a slice of base-field values to Montgomery `u32`.
pub fn felt_row_mont(fs: &[Felt]) -> Vec<u32> {
    fs.iter().map(|&f| felt_mont_u32(f)).collect()
}

// ---------------------------------------------------------------------------
// .npy v1.0 (hand-rolled — no third-party npy dependency, matching openvm).
// ---------------------------------------------------------------------------

/// Build the `.npy` v1.0 header for `descr`/`shape`. The 10 preamble bytes
/// (6 magic + 2 version + 2 header-len) plus the dict + padding + `\n` are a
/// multiple of 64, exactly as NumPy writes them.
fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    let unpadded = 10 + dict.len() + 1; // +1 for the trailing '\n'
    let padded = unpadded.div_ceil(64) * 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + (padded - unpadded), b' ');
    header.push(b'\n');

    let header_len = u16::try_from(header.len()).expect("npy header < 64KiB");
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]); // version 1.0
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header);
    out
}

fn npy_bytes(descr: &str, elem: usize, shape: &[usize], payload: &[u8]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    assert_eq!(
        payload.len(),
        n * elem,
        "npy shape {shape:?} vs payload {}",
        payload.len()
    );
    let mut bytes = npy_header(descr, shape);
    bytes.extend_from_slice(payload);
    bytes
}

fn u32_payload(data: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(data.len() * 4);
    for &x in data {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn i64_payload(data: &[i64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(data.len() * 8);
    for &x in data {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// `<u4` little-endian array.
pub fn write_npy_u32(path: &Path, shape: &[usize], data: &[u32]) -> std::io::Result<()> {
    fs::write(path, npy_bytes("<u4", 4, shape, &u32_payload(data)))
}

/// `|u1` array (raw device-buffer bytes already in Montgomery form).
pub fn write_npy_u8(path: &Path, shape: &[usize], data: &[u8]) -> std::io::Result<()> {
    fs::write(path, npy_bytes("|u1", 1, shape, data))
}

/// `<i8` array (e.g. `chip_final_lens`).
pub fn write_npy_i64(path: &Path, shape: &[usize], data: &[i64]) -> std::io::Result<()> {
    fs::write(path, npy_bytes("<i8", 8, shape, &i64_payload(data)))
}

// ---------------------------------------------------------------------------
// .npz — a ZIP of `<key>.npy` members, written deterministically (ZIP_STORED,
// fixed mtime) so the same fork pin reproduces byte-identical files.
// ---------------------------------------------------------------------------

/// One named array destined for an `.npz` member.
pub struct NpzEntry {
    pub key: String,
    pub shape: Vec<usize>,
    pub data: Vec<u32>,
}

impl NpzEntry {
    pub fn u32(key: impl Into<String>, shape: Vec<usize>, data: Vec<u32>) -> Self {
        Self {
            key: key.into(),
            shape,
            data,
        }
    }
}

/// Write `entries` as an uncompressed, reproducible `.npz` (each array stored as
/// `<key>.npy`, matching `numpy.savez`). All members are `<u4`.
pub fn write_npz(path: &Path, entries: &[NpzEntry]) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    // ZIP_STORED + a fixed DOS timestamp → deterministic across runs.
    let opts: zip::write::FileOptions<()> = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .last_modified_time(zip::DateTime::default());
    for e in entries {
        zip.start_file(format!("{}.npy", e.key), opts)
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        zip.write_all(&npy_bytes("<u4", 4, &e.shape, &u32_payload(&e.data)))?;
    }
    zip.finish()
        .map_err(|err| std::io::Error::other(err.to_string()))?;
    Ok(())
}
