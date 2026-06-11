//! Drive one `gpu_fibonacci` shard prove in-process and capture everything the
//! fixtures need: the `ShardProof`, the recorder log (host-sampled challenges),
//! the packed dense `D` buffer, and the public values.
//!
//! Mirrors the fork's `test_prove_shard_fibonacci` but from a downstream crate:
//!  * The prover is built via the pub `CudaShardProver::new` with `GC =
//!    RecordingGC` and a `pow_bits = 0` basefold prover (deterministic witness).
//!  * The prove goes through the pub `AirProver::setup_and_prove_shard`, which
//!    builds the challenger via `RecordingGC::default_challenger()`; we recover
//!    its log from [`crate::recorder::last_log`] (the caller-supplied-challenger
//!    entry `prove_shard_with_data` is `pub(crate)` and unreachable downstream).
//!  * `setup_and_prove_shard` consumes the trace data, so the dense `D` buffer is
//!    read back from a separate, deterministic `full_tracegen` (same fork pin →
//!    identical bytes) on a clone of the record.
//!
//! Heavy + GPU-bound — runs one fibonacci shard prove (plus one extra tracegen
//! for the dense). Off-peak only.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use slop_basefold::{BasefoldVerifier, FriConfig};
use slop_futures::queue::WorkerQueue;
use sp1_core_machine::io::SP1Stdin;
use sp1_gpu_air::codegen_cuda_eval;
use sp1_gpu_basefold::FriCudaProver;
use sp1_gpu_cudart::{run_in_place, PinnedBuffer};
use sp1_gpu_jagged_tracegen::test_utils::tracegen_setup;
use sp1_gpu_jagged_tracegen::{full_tracegen, CORE_MAX_TRACE_SIZE};
use sp1_gpu_logup_gkr::Interactions;
use sp1_gpu_merkle_tree::CudaTcsProver; // brings `RecordingMerkleProver::new` into scope
use sp1_gpu_shard_prover::CudaShardProver;
use sp1_gpu_utils::Felt;
use sp1_hypercube::air::MachineAir; // brings `chip.name()` into scope
use sp1_hypercube::prover::{AirProver, ProverSemaphore};
use sp1_hypercube::{SP1PcsProof, ShardProof};

use crate::components::{RecordingComponents, RecordingMerkleProver};
use crate::recorder::{last_log, RecorderLog, RecordingGC};

/// The concrete shard proof this generator produces.
pub type RecordingShardProof = ShardProof<RecordingGC, SP1PcsProof<RecordingGC>>;

/// Everything one prove yields that the fixture emitters consume.
///
/// `Serialize`/`Deserialize` back the `--dump-cache` / `--from-cache` loop: one
/// off-peak GPU prove dumps this, then the emitters iterate offline against it.
#[derive(Serialize, Deserialize)]
pub struct Captured {
    /// The full shard proof (zerocheck / jagged / basefold sub-proofs).
    pub proof: RecordingShardProof,
    /// Captured host transcript (raw): the emitter reads it positionally for the
    /// EF challenges; `bit_samples` → FRI query indices, `grind_witnesses` → PoW
    /// witness.
    pub log: RecorderLog,
    /// Packed prep‖main dense `D`, raw Montgomery `Felt` (the emitters split it
    /// at the round-0 boundary into prep_dense / main_dense).
    pub host_dense: Vec<Felt>,
    /// Shard public values, raw Montgomery `Felt`.
    pub public_values: Vec<Felt>,
    /// Chip names (schedule order), for `meta.json` + region dicts.
    pub chip_names: Vec<String>,
}

/// FRI parameters for deterministic fixtures: 52 queries, log_blowup 2,
/// **pow_bits = 0** — identical to the fork test under SP1_DUMP_PHASES.
pub fn deterministic_fri_config() -> FriConfig<Felt> {
    FriConfig::new(2, 52, 0)
}

/// Run the fibonacci shard prove and capture the fixture inputs/outputs.
pub async fn prove_and_capture() -> Captured {
    let (machine, record, program) =
        tracegen_setup::setup(&test_artifacts::FIBONACCI_ELF, SP1Stdin::new()).await;
    let record_for_dense = record.clone();
    // `run_in_place` returns a `TaskHandle<T>` whose owned value needs a parent
    // scope to extract; capture the result into a slot instead.
    let slot: Arc<Mutex<Option<Captured>>> = Arc::new(Mutex::new(None));
    let slot_w = slot.clone();

    run_in_place(|scope| async move {
        // Per-chip interactions + zerocheck codegen cache (as in the fork test).
        let mut all_interactions = BTreeMap::new();
        for chip in machine.chips().iter() {
            let host = Interactions::new(chip.sends(), chip.receives());
            let device = host.copy_to_device(&scope).unwrap();
            all_interactions.insert(chip.name().to_string(), Arc::new(device));
        }
        let mut cache = BTreeMap::new();
        for chip in machine.chips().iter() {
            cache.insert(
                chip.name().to_string(),
                codegen_cuda_eval(chip.air.as_ref()),
            );
        }

        // pow_bits = 0 basefold prover (deterministic PoW witness).
        let verifier = BasefoldVerifier::<RecordingGC>::new(deterministic_fri_config(), 2);
        let basefold_prover = FriCudaProver::<RecordingGC, _, Felt>::new(
            RecordingMerkleProver::new(&scope),
            verifier.fri_config,
            tracegen_setup::LOG_STACKING_HEIGHT,
        );

        let trace_buffers = Arc::new(WorkerQueue::new(vec![PinnedBuffer::<Felt>::with_capacity(
            CORE_MAX_TRACE_SIZE as usize,
        )]));

        let prover = CudaShardProver::<RecordingGC, RecordingComponents>::new(
            trace_buffers,
            tracegen_setup::CORE_MAX_LOG_ROW_COUNT,
            basefold_prover,
            machine.clone(),
            CORE_MAX_TRACE_SIZE as usize,
            scope.clone(),
            all_interactions,
            cache,
            false,
            false,
        );

        // Dense readback: a standalone, deterministic tracegen (identical bytes to
        // what the prove regenerates internally).
        let dense_queue = Arc::new(WorkerQueue::new(vec![PinnedBuffer::<Felt>::with_capacity(
            CORE_MAX_TRACE_SIZE as usize,
        )]));
        let dense_buffer = dense_queue.pop().await.unwrap();
        let (public_values, trace_data, _chips, _permit) = full_tracegen(
            &machine,
            program.clone(),
            Arc::new(record_for_dense),
            &dense_buffer,
            CORE_MAX_TRACE_SIZE as usize,
            tracegen_setup::LOG_STACKING_HEIGHT,
            tracegen_setup::CORE_MAX_LOG_ROW_COUNT,
            &scope,
            ProverSemaphore::new(1),
            true,
        )
        .await;
        // SAFETY: device→host copy of the packed dense `D` buffer (Montgomery Felt).
        let host_dense: Vec<Felt> = unsafe { trace_data.0.dense().dense.copy_into_host_vec() };

        // Prove: builds `RecordingGC::default_challenger()` internally, which
        // registers its log handle for `last_log()`.
        let (_vk, shard_proof, _permit) = prover
            .setup_and_prove_shard(program, record, None, ProverSemaphore::new(1))
            .await;

        let log = last_log()
            .map(|h| std::mem::take(&mut *h.lock().unwrap()))
            .unwrap_or_default();
        let chip_names = machine
            .chips()
            .iter()
            .map(|c| c.name().to_string())
            .collect();

        *slot_w.lock().unwrap() = Some(Captured {
            proof: shard_proof,
            log,
            host_dense,
            public_values,
            chip_names,
        });
    })
    .await;

    let captured = slot.lock().unwrap().take();
    captured.expect("prove_and_capture produced no result")
}

/// Serialize a [`Captured`] to `path` (JSON) so emit can be iterated without a
/// fresh GPU prove. Same fork pin → identical capture → identical fixtures.
pub fn save_cache(c: &Captured, path: &Path) -> std::io::Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    serde_json::to_writer(f, c).map_err(std::io::Error::other)
}

/// Load a [`Captured`] previously written by [`save_cache`].
pub fn load_cache(path: &Path) -> std::io::Result<Captured> {
    let f = std::io::BufReader::new(std::fs::File::open(path)?);
    serde_json::from_reader(f).map_err(std::io::Error::other)
}
