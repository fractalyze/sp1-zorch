//! NumPy `.npy` / `.npz` writers emitting raw **Montgomery** `u32` — the on-disk
//! form the committed `gpu_fibonacci` fixtures use (device `.bin` buffers are
//! already Montgomery, and `KoalaBear` is a transparent Montgomery `u32`). This
//! deliberately differs from the openvm reference generator, which emits
//! *canonical* u32; here the byte-match compares Montgomery limbs directly
//! (see the repo `CLAUDE.md`: "Compare Montgomery-form u32 bytes directly").

use std::fs;
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

/// CRC-32 (IEEE 802.3, the ZIP checksum) of `data`.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Write `entries` as an uncompressed, reproducible `.npz` (each array stored as
/// `<key>.npy`, all `<u4`), byte-identical to `numpy.savez`.
///
/// numpy opens every member with `force_zip64=True`, which produces an asymmetric
/// layout we reproduce by hand (the `zip` crate's `large_file` Zip64s the central
/// directory too, which numpy does not):
///  * each **local** header is Zip64 — version-needed 45, sizes written as the
///    `0xffffffff` sentinel, followed by a 20-byte Zip64 extra field carrying the
///    real (uncompressed, compressed) `u64` sizes;
///  * each **central directory** record keeps the real 32-bit sizes and *no*
///    extra field, with `version-made-by = 0x032d` (Unix host, v4.5) and
///    `external_attr = 0x01800000` — the exact constants CPython's `zipfile`
///    emits for a freshly-named member.
/// Stored (no compression), fixed 1980-01-01 timestamp → deterministic output.
pub fn write_npz(path: &Path, entries: &[NpzEntry]) -> std::io::Result<()> {
    const DOS_DATE_1980: u16 = 0x0021;
    let mut out: Vec<u8> = Vec::new();
    let mut central: Vec<u8> = Vec::new();

    for e in entries {
        let name = format!("{}.npy", e.key);
        let data = npy_bytes("<u4", 4, &e.shape, &u32_payload(&e.data));
        let crc = crc32(&data);
        let offset = out.len() as u32;
        let size = data.len();

        // Local file header — Zip64-forced (sizes deferred to the extra field).
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&45u16.to_le_bytes()); // version needed (Zip64)
        out.extend_from_slice(&0u16.to_le_bytes()); // general-purpose flags
        out.extend_from_slice(&0u16.to_le_bytes()); // compression = stored
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&DOS_DATE_1980.to_le_bytes()); // mod date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // compressed size sentinel
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // uncompressed size sentinel
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // Zip64 extra field length
        out.extend_from_slice(name.as_bytes());
        // Zip64 extra field: id 0x0001, 16-byte payload (uncompressed, compressed).
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(&(size as u64).to_le_bytes());
        out.extend_from_slice(&(size as u64).to_le_bytes());
        out.extend_from_slice(&data);

        // Central directory record — real 32-bit sizes, no extra field.
        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&0x032du16.to_le_bytes()); // version made by (Unix, 45)
        central.extend_from_slice(&45u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&0u16.to_le_bytes()); // compression = stored
        central.extend_from_slice(&0u16.to_le_bytes()); // mod time
        central.extend_from_slice(&DOS_DATE_1980.to_le_bytes()); // mod date
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(size as u32).to_le_bytes());
        central.extend_from_slice(&(size as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        central.extend_from_slice(&0u16.to_le_bytes()); // comment length
        central.extend_from_slice(&0u16.to_le_bytes()); // disk number start
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0x0180_0000u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }

    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend_from_slice(&central);

    // End of central directory record (all sizes/counts fit in 32/16 bits, so no
    // Zip64 EOCD — matching numpy for these small archives).
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes()); // disk number
    out.extend_from_slice(&0u16.to_le_bytes()); // disk with central dir
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes()); // entries this disk
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes()); // entries total
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment length

    fs::write(path, out)
}
