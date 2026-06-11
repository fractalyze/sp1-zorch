//! Prover components for [`RecordingGC`] — a copy of the fork's
//! `TestProverComponentsImpl` with `GC = RecordingGC`. The device challenger is
//! unchanged (`sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>`), so the
//! `AsMutRawChallenger` / `BranchingProgramKernel` machinery keyed on it keeps
//! working; only the *host* challenger (`GC::Challenger`) is our recording
//! newtype.

use sp1_core_machine::riscv::RiscvAir;
use sp1_gpu_cudart::TaskScope;
use sp1_gpu_merkle_tree::Poseidon2SP1Field16CudaProver;
use sp1_gpu_shard_prover::CudaShardProverComponents;
use sp1_gpu_utils::Felt;
use sp1_hypercube::SP1InnerPcs;

use crate::recorder::RecordingGC;

pub struct RecordingComponents;

impl CudaShardProverComponents<RecordingGC> for RecordingComponents {
    type P = Poseidon2SP1Field16CudaProver;
    type Air = RiscvAir<Felt>;
    type C = SP1InnerPcs;
    type DeviceChallenger = sp1_gpu_challenger::DuplexChallenger<Felt, TaskScope>;
}
