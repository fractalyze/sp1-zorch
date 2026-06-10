# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""ctypes binding for SP1's host-side shard verifier.

``sp1_verify_shard`` consumes host byte buffers (bincode vk + proof) and
returns a status code — no device memory, no JAX FFI custom-call plumbing.
The shared library is SP1's ``libsp1_gpu_jax_ffi.so``; sp1-zorch does not
vendor it (it ships with the SP1 reference checkout), so the path comes
from ``SP1_JAX_FFI_LIB``.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import c_char_p, c_int32, c_size_t

_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        path = os.environ.get("SP1_JAX_FFI_LIB", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                "SP1_JAX_FFI_LIB does not point at libsp1_gpu_jax_ffi.so "
                f"(got {path!r}); the library is vendored in SP1 reference "
                "checkouts (e.g. whir-zorch third_party/sp1/)."
            )
        lib = ctypes.CDLL(path)
        lib.sp1_verify_shard.restype = c_int32
        lib.sp1_verify_shard.argtypes = [
            c_char_p,  # vk_bytes
            c_size_t,  # vk_len
            c_char_p,  # proof_bytes
            c_size_t,  # proof_len
            c_size_t,  # log_blowup
            c_size_t,  # num_queries
            c_size_t,  # pow_bits (basefold FRI)
            c_size_t,  # gkr_pow_bits (LogUp-GKR)
        ]
        _lib = lib
    return _lib


def sp1_verify_shard(
    vk_bincode: bytes,
    proof_bincode: bytes,
    *,
    log_blowup: int,
    num_queries: int,
    pow_bits: int,
    gkr_pow_bits: int,
) -> None:
    """Verify a bincode shard proof; the parameters must match the prover's.

    Raises ``RuntimeError`` on malformed bincode (code 2) or a verification
    failure (any other non-zero code).
    """
    ret = _get_lib().sp1_verify_shard(
        vk_bincode,
        len(vk_bincode),
        proof_bincode,
        len(proof_bincode),
        log_blowup,
        num_queries,
        pow_bits,
        gkr_pow_bits,
    )
    if ret == 0:
        return
    if ret == 2:
        raise RuntimeError("sp1_verify_shard: bincode deserialization failed")
    raise RuntimeError(f"sp1_verify_shard: verification failed (code={ret})")


__all__ = ["sp1_verify_shard"]
