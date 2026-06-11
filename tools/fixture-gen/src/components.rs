//! Prover components for [`RecordingGC`] — a copy of the fork's
//! `TestProverComponentsImpl` with `GC = RecordingGC`. The device challenger is
//! unchanged (`sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>`), so the
//! `AsMutRawChallenger` / `BranchingProgramKernel` machinery keyed on it keeps
//! working; only the *host* challenger (`GC::Challenger`) is our recording
//! newtype.

use sp1_core_machine::riscv::RiscvAir;
use sp1_gpu_cudart::TaskScope;
use sp1_gpu_merkle_tree::{
    MerkleTreeSingleLayerProver, Poseidon2SP1Field16Hasher, Poseidon2SP1Field16Kernels,
};
use sp1_gpu_shard_prover::CudaShardProverComponents;
use sp1_gpu_utils::Felt;
use sp1_hypercube::SP1Pcs;

use crate::recorder::RecordingGC;

/// The fork's `Poseidon2SP1Field16CudaProver` Merkle prover, but parameterized by
/// `RecordingGC` instead of the baked-in `SP1GlobalContext` (the single-layer
/// kernels are GC-independent — see the merkle_tree `single_layer.rs`
/// generalization to `GC: IopCtx<F = SP1Field>`).
pub type RecordingMerkleProver = MerkleTreeSingleLayerProver<
    RecordingGC,
    Felt,
    Poseidon2SP1Field16Kernels,
    Poseidon2SP1Field16Hasher,
    16,
>;

pub struct RecordingComponents;

impl CudaShardProverComponents<RecordingGC> for RecordingComponents {
    type P = RecordingMerkleProver;
    type Air = RiscvAir<Felt>;
    type C = SP1Pcs<RecordingGC>;
    type DeviceChallenger = sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>;
}
