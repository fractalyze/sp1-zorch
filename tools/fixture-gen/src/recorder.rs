//! A recording Fiat–Shamir challenger plus the `IopCtx` / prover-component
//! wiring that lets this crate drive the SP1 GPU shard prover **in-process**
//! while capturing every host-sampled challenge — replacing the fork's scattered
//! `SP1_DUMP_PHASES` Fiat–Shamir dumps (`sp1#29`, `basefold/src/fri.rs`).
//!
//! Verified against the fork (`envs/zorch4/sp1`, 2026-06-11):
//!
//! * The shard prover is generic over `GC: IopCtx<F = Felt, EF = Ext>` and the
//!   challenger enters only via `GC::default_challenger()`. So we define our own
//!   [`RecordingGC`] whose `Challenger` is `RecordingChallenger<KoalaBearDuplexChallenger>`,
//!   plus [`RecordingComponents`] mirroring the fork's `TestProverComponentsImpl`.
//!   The device challenger stays `sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>`,
//!   so the on-device branching-program kernels are unaffected. **No prover-source fork.**
//!
//! * The host newtype logs every base-field draw (`is_sample`), every
//!   `sample_bits` (FRI query indices) and every `grind`/`grind_device` (PoW
//!   witness). EF challenges decompose to 4 consecutive base samples
//!   (`sample_ext_element` → `sample_vec(4)`), so [`RecorderLog::ef_samples`]
//!   reconstructs them. The jagged-eval per-round `alpha`s are sampled inside the
//!   CUDA kernel and bypass the newtype — but those values are the jagged
//!   sumcheck *points*, recovered from the proof struct, so no device readback is
//!   needed.
//!
//! * The device-challenger conversion impls (`FromChallenger` /
//!   `FromHostChallengerSync`) only exist for the bare
//!   `slop_challenger::DuplexChallenger`. The two delegating impls for
//!   `RecordingChallenger` live here: the orphan rule permits them because
//!   `RecordingChallenger` (a local type) appears as a trait type parameter even
//!   though the `Self` type (the device challenger) is foreign.

use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use slop_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, FromChallenger,
    GrindingChallenger, IopCtx,
};
use slop_koala_bear::{my_kb_16_perm, KoalaBearDegree4Duplex};
use sp1_gpu_basefold::DeviceGrindingChallenger;
use sp1_gpu_challenger::{FromHostChallengerSync, KoalaBearDuplexChallenger};
use sp1_gpu_cudart::TaskScope;
use sp1_gpu_utils::{Ext, Felt};

/// The full host-challenger transcript captured during one prove — pure storage.
///
/// `values`/`is_sample` are the flat base-field stream (openvm `TranscriptLog`
/// shape). `bit_samples` and `grind_witnesses` capture the two draws that do
/// **not** flow through the base `sample()` path on this newtype (they are
/// delegated to the inner challenger).
///
/// EF challenges are reconstructed by the emitter (Phase 3) reading this log
/// **positionally** against the protocol order — a cursor that asserts the
/// `is_sample` flag at each step and cross-checks proof-struct values, like the
/// openvm reference's `walk_*_log`. (A global "collect all samples, chunk by 4"
/// is intentionally avoided: `len % 4 == 0` cannot catch a misaligned interleave.)
#[derive(Default, Debug)]
pub struct RecorderLog {
    /// Every observed (`is_sample=false`) / base-sampled (`is_sample=true`)
    /// field element, in protocol order.
    pub values: Vec<Felt>,
    pub is_sample: Vec<bool>,
    /// Every `sample_bits` result (FRI query indices), in order.
    pub bit_samples: Vec<usize>,
    /// Every `grind` / `grind_device` PoW witness, in order.
    pub grind_witnesses: Vec<Felt>,
}

/// A shared, clonable handle to a [`RecorderLog`]. Cloning the challenger shares
/// the same log (Arc), so the log captured during `prove_shard_with_data` (which
/// takes the challenger by value) is still readable afterwards through a handle
/// kept by the driver.
pub type LogHandle = Arc<Mutex<RecorderLog>>;

/// Every challenger handed out by [`RecordingGC::default_challenger`] registers
/// its log handle here. This lets the driver recover the recorder log even when
/// the prover constructs the challenger internally (the `AirProver` pub path) —
/// `prove_shard_with_data`, the entry that accepts a caller-supplied challenger,
/// is `pub(crate)` on `CudaShardProverInner` and unreachable downstream. (If a
/// tiny pub forwarder is added to the fork instead, the driver can pass an
/// explicit challenger and skip this registry.)
static LOG_REGISTRY: OnceLock<Mutex<Vec<LogHandle>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<LogHandle>> {
    LOG_REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// The handle of the most recently constructed recording challenger, i.e. the
/// one the just-completed prove used. Returns `None` if no prove has run.
pub fn last_log() -> Option<LogHandle> {
    registry().lock().unwrap().last().cloned()
}

/// A newtype over a host challenger `C` that records every host-trait draw.
#[derive(Clone)]
pub struct RecordingChallenger<C> {
    pub inner: C,
    pub log: LogHandle,
}

impl<C> RecordingChallenger<C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            log: Arc::new(Mutex::new(RecorderLog::default())),
        }
    }

    /// A handle to the shared log; keep this before moving the challenger into
    /// the prover, then read it back after the prove returns.
    pub fn log_handle(&self) -> LogHandle {
        Arc::clone(&self.log)
    }

    fn push_value(&self, v: Felt, is_sample: bool) {
        let mut l = self.log.lock().unwrap();
        l.values.push(v);
        l.is_sample.push(is_sample);
    }
}

impl<C: CanObserve<Felt>> CanObserve<Felt> for RecordingChallenger<C> {
    fn observe(&mut self, value: Felt) {
        self.inner.observe(value);
        self.push_value(value, false);
    }
}

impl<C: CanObserve<[Felt; 8]>> CanObserve<[Felt; 8]> for RecordingChallenger<C> {
    fn observe(&mut self, value: [Felt; 8]) {
        self.inner.observe(value);
        for v in value {
            self.push_value(v, false);
        }
    }
}

impl<C: CanSample<Felt>> CanSample<Felt> for RecordingChallenger<C> {
    fn sample(&mut self) -> Felt {
        let v = self.inner.sample();
        self.push_value(v, true);
        v
    }
}

impl<C: CanSampleBits<usize>> CanSampleBits<usize> for RecordingChallenger<C> {
    fn sample_bits(&mut self, bits: usize) -> usize {
        // The inner draw bypasses our base `sample()`; record the result instead.
        let b = self.inner.sample_bits(bits);
        self.log.lock().unwrap().bit_samples.push(b);
        b
    }
}

// `FieldChallenger` is a marker here: its `sample_ext_element`/`observe_ext_element`
// defaults decompose into base `sample()`/`observe()` on `self`, which we already
// record. `Sync` is satisfied by `RecordingChallenger` (inner + Arc<Mutex<..>>).
impl<C: FieldChallenger<Felt>> FieldChallenger<Felt> for RecordingChallenger<C> {}

impl<C: GrindingChallenger<Witness = Felt>> GrindingChallenger for RecordingChallenger<C> {
    type Witness = Felt;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        let w = self.inner.grind(bits);
        self.log.lock().unwrap().grind_witnesses.push(w);
        w
    }
}

impl<C: DeviceGrindingChallenger<Witness = Felt>> DeviceGrindingChallenger
    for RecordingChallenger<C>
{
    fn grind_device(&mut self, bits: usize, scope: &TaskScope) -> Self::Witness {
        let w = self.inner.grind_device(bits, scope);
        self.log.lock().unwrap().grind_witnesses.push(w);
        w
    }
}

// --- Device-challenger conversions (orphan-legal: `RecordingChallenger` is a
// --- local type in the trait parameter list; the existing impls cover only the
// --- bare `slop_challenger::DuplexChallenger`, so we delegate to those).

impl FromChallenger<RecordingChallenger<KoalaBearDuplexChallenger>, TaskScope>
    for sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>
{
    fn from_challenger(
        challenger: &RecordingChallenger<KoalaBearDuplexChallenger>,
        backend: &TaskScope,
    ) -> Self {
        <Self as FromChallenger<KoalaBearDuplexChallenger, TaskScope>>::from_challenger(
            &challenger.inner,
            backend,
        )
    }
}

impl FromHostChallengerSync<RecordingChallenger<KoalaBearDuplexChallenger>>
    for sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>
{
    fn from_host_challenger_sync(
        challenger: &RecordingChallenger<KoalaBearDuplexChallenger>,
        backend: &TaskScope,
    ) -> Self {
        <Self as FromHostChallengerSync<KoalaBearDuplexChallenger>>::from_host_challenger_sync(
            &challenger.inner,
            backend,
        )
    }
}

/// Our fixture-gen `IopCtx`: identical to the fork's `KoalaBearDegree4Duplex`
/// except the challenger is the recording newtype.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingGC;

impl IopCtx for RecordingGC {
    type F = Felt;
    type EF = Ext;
    type Digest = [Felt; 8];
    type Challenger = RecordingChallenger<KoalaBearDuplexChallenger>;
    type Hasher = <KoalaBearDegree4Duplex as IopCtx>::Hasher;
    type Compressor = <KoalaBearDegree4Duplex as IopCtx>::Compressor;

    fn default_hasher_and_compressor() -> (Self::Hasher, Self::Compressor) {
        KoalaBearDegree4Duplex::default_hasher_and_compressor()
    }

    fn default_challenger() -> Self::Challenger {
        // `my_kb_16_perm()` == the perm the fork's `KoalaBearDuplexChallenger`
        // and `TestGC::default_challenger()` both use (SP1DiffusionMatrix ==
        // DiffusionMatrixKoalaBear), so this is the identical challenger, wrapped.
        let challenger = RecordingChallenger::new(KoalaBearDuplexChallenger::new(my_kb_16_perm()));
        registry().lock().unwrap().push(challenger.log_handle());
        challenger
    }
}

// A degenerate sanity check that the perm types unify, kept so a mismatch shows
// up at compile time rather than as a silent byte-match failure.
const _: fn() = || {
    let _: DuplexChallenger<Felt, slop_koala_bear::KoalaPerm, 16, 8> =
        KoalaBearDuplexChallenger::new(my_kb_16_perm());
};
