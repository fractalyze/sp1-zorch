//! Rust generator for the `gpu_fibonacci` byte-match fixtures.
//!
//! Drives the SP1 GPU shard prove once, in-process, with a recording challenger,
//! then emits the zerocheck and jagged(+open) fixtures directly — replacing the
//! `dump + convert.py` pipeline (sp1-zorch #76 / PR #83) and the fork's `fri.rs`
//! Fiat-Shamir dumps (sp1 #29). See issue #86 and `README.md`.
//!
//! CLI (mirrors `convert.py`, minus `--dump` since the prove is in-process):
//! ```text
//! fixture-gen --zerocheck-out <zerocheck/testdata/gpu_fibonacci>
//!             --out           <jagged/testdata/gpu_fibonacci>
//! ```
//! At least one of `--out` / `--zerocheck-out` is required. `--out` emits the
//! jagged-eval pieces and the stacked-open pieces (open augments the jagged dir).

// Scaffold (issue #86): the npy/recorder helpers forward-declare the API that
// Phase-3 emission consumes; remove this once `emit::{zerocheck,jagged,open}`
// are wired and exercise them.
#![allow(dead_code)]

mod components;
mod driver;
mod emit;
mod npy;
mod recorder;

use std::path::PathBuf;

struct Args {
    out: Option<PathBuf>,
    zerocheck_out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut out = None;
    let mut zerocheck_out = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(it.next().expect("--out needs a path"))),
            "--zerocheck-out" => {
                zerocheck_out = Some(PathBuf::from(
                    it.next().expect("--zerocheck-out needs a path"),
                ))
            }
            other => panic!("unknown arg {other:?}; usage: --out <dir> --zerocheck-out <dir>"),
        }
    }
    if out.is_none() && zerocheck_out.is_none() {
        panic!("at least one of --out / --zerocheck-out is required");
    }
    Args { out, zerocheck_out }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = parse_args();

    // One prove feeds both fixture dirs (the dense `D` is shared).
    let captured = driver::prove_and_capture().await;

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
