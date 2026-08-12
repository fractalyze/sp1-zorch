"""Launcher shim: the staged runner lives in the wheel-shipped package.

The implementation is ``sp1_zorch.shard_prover.staged_prove_shard`` so wheel
consumers (``warm_shard_cache`` workers, zkvm-prover prewarm) can import it
without a repo checkout; this file only keeps the historical
``bazel run //tools:staged_prove_shard`` entry point working.
"""

from sp1_zorch.shard_prover.staged_prove_shard import app, main

if __name__ == "__main__":
    app.run(main)
