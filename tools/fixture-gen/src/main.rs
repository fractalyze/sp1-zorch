//! Rust generator for the `gpu_fibonacci` byte-match fixtures.
//!
//! Drives the SP1 GPU shard prove once, in-process, with a recording challenger,
//! then emits the zerocheck and jagged(+open) fixtures directly — replacing the
//! prior dump-and-Python-convert pipeline (sp1-zorch #76 / PR #83) and the fork's
//! `fri.rs` Fiat-Shamir dumps (sp1 #29). See issue #86 and `README.md`.
//!
//! CLI (the prove is in-process, so there is no separate `--dump` step):
//! ```text
//! fixture-gen --zerocheck-out <zerocheck/testdata/gpu_fibonacci>
//!             --out           <jagged/testdata/gpu_fibonacci>
//!             [--dump-cache <file>]   # after proving, cache the capture
//!             [--from-cache <file>]   # load a cache, skip the GPU prove
//! ```
//! At least one of `--out` / `--zerocheck-out` is required. `--out` emits the
//! jagged-eval pieces and the stacked-open pieces (open augments the jagged dir).
//!
//! `--from-cache` lets the emit stages be iterated offline (the GPU prove is the
//! slow, shared-resource step); `--dump-cache` captures one prove for reuse.

mod components;
mod driver;
mod emit;
mod npy;
mod recorder;

use std::path::PathBuf;

struct Args {
    out: Option<PathBuf>,
    zerocheck_out: Option<PathBuf>,
    dump_cache: Option<PathBuf>,
    from_cache: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut out = None;
    let mut zerocheck_out = None;
    let mut dump_cache = None;
    let mut from_cache = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(it.next().expect("--out needs a path"))),
            "--zerocheck-out" => {
                zerocheck_out = Some(PathBuf::from(
                    it.next().expect("--zerocheck-out needs a path"),
                ))
            }
            "--dump-cache" => {
                dump_cache = Some(PathBuf::from(it.next().expect("--dump-cache needs a path")))
            }
            "--from-cache" => {
                from_cache = Some(PathBuf::from(it.next().expect("--from-cache needs a path")))
            }
            other => panic!(
                "unknown arg {other:?}; usage: --out <dir> --zerocheck-out <dir> \
                 [--dump-cache <file>] [--from-cache <file>]"
            ),
        }
    }
    if out.is_none() && zerocheck_out.is_none() {
        panic!("at least one of --out / --zerocheck-out is required");
    }
    Args {
        out,
        zerocheck_out,
        dump_cache,
        from_cache,
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = parse_args();

    // The GPU prove is the slow, shared-resource step; `--from-cache` reuses a
    // prior capture so emit can be iterated offline.
    let captured = match &args.from_cache {
        Some(path) => {
            eprintln!("loading capture from {}", path.display());
            driver::load_cache(path).expect("load --from-cache")
        }
        None => {
            // One prove feeds both fixture dirs (the dense `D` is shared).
            let captured = driver::prove_and_capture().await;
            if let Some(path) = &args.dump_cache {
                driver::save_cache(&captured, path).expect("write --dump-cache");
                eprintln!("capture cached to {}", path.display());
            }
            captured
        }
    };

    let n_samples = captured.log.is_sample.iter().filter(|b| **b).count();
    eprintln!("=== capture summary ===");
    eprintln!(
        "log: values={} (base samples={}, => ~{} EF), bit_samples={}, grind_witnesses={}",
        captured.log.values.len(),
        n_samples,
        n_samples / 4,
        captured.log.bit_samples.len(),
        captured.log.grind_witnesses.len(),
    );
    eprintln!(
        "host_dense={}, public_values={}, chips={}",
        captured.host_dense.len(),
        captured.public_values.len(),
        captured.chip_names.len(),
    );

    if let Some(dir) = &args.zerocheck_out {
        emit::zerocheck(&captured, dir).expect("emit zerocheck fixtures");
        println!("zerocheck fixtures written to {}", dir.display());
    }
    if let Some(dir) = &args.out {
        emit::jagged(&captured, dir).expect("emit jagged fixtures");
        emit::open(&captured, dir).expect("emit open fixtures");
        println!("jagged + open fixtures written to {}", dir.display());
    }
}
