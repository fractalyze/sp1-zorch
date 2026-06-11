//! Drive one `gpu_fibonacci` shard prove in-process and capture everything the
//! fixtures need: the `ShardProof`, the recorder log (host-sampled challenges),
//! the packed dense `D` buffer, and the public values.
//!
//! This mirrors the fork's `test_prove_shard_fibonacci` (`sp1-gpu shard_prover`)
//! but from a downstream crate. Two seams differ from the in-crate test:
//!
//!  1. **Prove invocation (open).** The test builds a `pub(crate)`
//!     `CudaShardProverInner` and calls `inner.prove_shard_with_data(data,
//!     challenger)` — also the only entry that accepts a caller-supplied
//!     challenger. Neither is reachable downstream. Plan: build the `pub`
//!     `CudaShardProver` with a pow_bits=0 basefold prover and drive its `pub`
//!     high-level prove API, which constructs the challenger via
//!     `RecordingGC::default_challenger()`; recover its log from
//!     [`crate::recorder::last_log`]. Fallback: add a 3-line `pub` forwarder to
//!     `CudaShardProver` in the Phase-4 fork commit and pass the challenger
//!     explicitly. Settle in the build window.
//!  2. **Dense readback.** `full_tracegen` returns the trace data holding the
//!     dense `D` buffer; copy it to host (`copy_into_host_vec`) before setup
//!     consumes it, instead of reading a `.bin` dump.
//!
//! The body below is the intended flow as a recipe; the concrete prover/tracegen
//! signatures are wired once the crate compiles (build window). Heavy + GPU-bound
//! (one cold CUDA build of `sp1-gpu-shard-prover` + one prove on the RTX 5090) —
//! run off-peak.

use slop_basefold::FriConfig;
use sp1_gpu_utils::{Ext, Felt};

use crate::recorder::RecorderLog;

/// Everything one prove yields that the fixture emitters consume.
pub struct Captured {
    /// The full shard proof. Concrete type:
    /// `ShardProof<RecordingGC, <SP1InnerPcs as MultilinearPcsVerifier<RecordingGC>>::Proof>`.
    /// Boxed-opaque until the prove call is wired.
    pub proof: ShardProofOpaque,
    /// Captured host transcript (raw): the emitter reads it positionally for the
    /// EF challenges; `bit_samples` → FRI query indices, `grind_witnesses` → PoW
    /// witness.
    pub log: RecorderLog,
    /// Packed prep‖main dense `D`, raw Montgomery `Felt` (the emitters split it
    /// at the round-0 boundary into prep_dense / main_dense).
    pub host_dense: Vec<Felt>,
    /// Shard public values, raw Montgomery `Felt`.
    pub public_values: Vec<Felt>,
    /// Chip schedule / names, for `meta.json` + region dicts.
    pub chip_names: Vec<String>,
}

/// Placeholder for the concrete `ShardProof<…>` until the prove call is wired.
pub type ShardProofOpaque = ();

/// Convenience EF/Felt re-exports for the emit modules.
pub type CapturedEf = Ext;

/// FRI parameters for deterministic fixtures: 52 queries, log_blowup 2,
/// **pow_bits = 0** — identical to the fork test under SP1_DUMP_PHASES.
pub fn deterministic_fri_config() -> FriConfig<Felt> {
    FriConfig::new(2, 52, 0)
}

/// Run the fibonacci shard prove and capture the fixture inputs/outputs.
///
/// Intended flow (wired in the build window):
/// ```text
/// let (machine, record, program) =
///     tracegen_setup::setup(&FIBONACCI_ELF, SP1Stdin::new()).await;          // pub
/// run_in_place(|scope| async move {
///     let (public_values, jagged_trace_data, shard_chips, permit) =
///         full_tracegen(&machine, program.clone(), Arc::new(record), &buffer,
///             CORE_MAX_TRACE_SIZE, LOG_STACKING_HEIGHT, CORE_MAX_LOG_ROW_COUNT,
///             &scope, ProverSemaphore::new(1), true).await;
///     let host_dense = unsafe { jagged_trace_data.dense().dense.copy_into_host_vec() };
///     // per-chip Interactions::copy_to_device + codegen_cuda_eval cache ...
///     let fri = deterministic_fri_config();
///     let verifier = BasefoldVerifier::<RecordingGC>::new(fri, 2);
///     let basefold_prover = FriCudaProver::<RecordingGC, _, Felt>::new(
///         Poseidon2SP1Field16CudaProver::new(&scope), verifier.fri_config, LOG_STACKING_HEIGHT);
///     let prover = CudaShardProver::<RecordingGC, RecordingComponents>::new(
///         trace_buffers, CORE_MAX_LOG_ROW_COUNT, basefold_prover, machine.clone(),
///         CORE_MAX_TRACE_SIZE, scope.clone(), all_interactions, cache, false, false);
///     let proof = prover./*pub high-level prove*/(program, jagged_trace_data,
///         public_values.clone(), ...).await;        // builds RecordingGC::default_challenger()
///     let log = last_log().map(|h| std::mem::take(&mut *h.lock().unwrap())).unwrap_or_default();
///     Captured { proof, log, host_dense, public_values, chip_names }
/// }).await
/// ```
pub async fn prove_and_capture() -> Captured {
    todo!("wire the fibonacci prove (build window) — see the module + flow doc above")
}
