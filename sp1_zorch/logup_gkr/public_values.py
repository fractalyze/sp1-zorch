# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1's ``Record::eval_public_values`` as a consumer-evaluable predicate.

The LogUp-GKR output-layer acceptance leg: SP1 checks the GKR circuit's
``sum(num_i / den_i)`` against ``-local_interaction_digest``, where the digest
comes from running ``eval_public_values`` through a
``VerifierPublicValuesConstraintFolder``. The folder does two things at once
(SP1 ``crates/hypercube/src/folder.rs`` @ ``e2c02f376``):

* every ``assert_zero`` Horner-folds into an ``accumulator`` under the head's
  discarded public-values challenge; a well-formed public-values vector drives
  it to zero (``crates/hypercube/src/logup_gkr/verifier.rs::verify_public_values``).
* every ``send``/``receive`` fingerprints the message with the same
  ``alpha``/``betas`` as the trace interactions and accumulates
  ``+/- multiplicity / fingerprint`` into ``local_interaction_digest`` — the
  bus mass of the interactions that ride the public-values vector (the global
  memory / page-prot init and finalize controls, the global-sum accumulation,
  and the byte range checks of ``eval_state``/``eval_committed_value_digest``).

Pinned for diffing against SP1 ``crates/core/executor/src/record.rs`` @
``e2c02f376`` (``eval_public_values`` and its ``eval_*`` helpers). The accumulator
fold is order-sensitive in general but order-free for a valid vector (a Horner
fold of all-zeros is zero regardless of the challenge), so the port mirrors
SP1's call order for readability without depending on it for correctness.

Per-element field values are lifted into the extension field up front so all
arithmetic runs in EF exactly as SP1's ``Expr = EF`` folder does; the embedding
is a ring homomorphism, so a constraint that is zero over the base field stays
zero here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import frx.numpy as jnp
from frx import Array
from zk_dtypes import koalabear_mont as F

# Public-values vector layout (SP1 ``PublicValues<[F;4], [F;3], [F;4], F>`` in
# ``#[repr(C)]`` order). The proof ships ``PROOF_MAX_NUM_PVS`` (187) elements;
# ``eval_public_values`` reads the first ``SP1_PROOF_NUM_PV_ELTS`` and asserts
# the trailing ``empty`` padding is zero.
SP1_PROOF_NUM_PV_ELTS = 160
PV_DIGEST_NUM_WORDS = 8

# ``InteractionKind`` discriminants (SP1 ``crates/hypercube/src/lookup/
# interaction.rs``); only the kinds that appear in ``eval_public_values``.
_KIND_BYTE = 5
_KIND_STATE = 7
_KIND_GLOBAL_ACCUMULATION = 13
_KIND_MEMORY_GLOBAL_INIT = 14
_KIND_MEMORY_GLOBAL_FINALIZE = 15
_KIND_PAGE_PROT_GLOBAL_INIT = 20
_KIND_PAGE_PROT_GLOBAL_FINALIZE = 21

# ``ByteOpcode`` discriminants (SP1 ``crates/core/executor/src/opcode.rs``).
_BYTE_U8RANGE = 3
_BYTE_RANGE = 6

# ``SepticDigest::zero()`` — the curve point every global cumulative sum starts
# from (SP1 ``crates/hypercube/src/septic_digest.rs``, derived from sqrt(2)).
_CURVE_CUMULATIVE_SUM_START_X = (
    0x1414213, 0x5623730, 0x9504880, 0x1688724, 0x2096980, 0x7856967, 0x1875376,
)
_CURVE_CUMULATIVE_SUM_START_Y = (
    2020310104, 1513506566, 1843922297, 2003644209, 805967281, 1882435203, 1623804682,
)


@dataclass(frozen=True)
class _Pv:
    """Named views into the extension-field-lifted public-values vector.

    Each accessor returns the EF-embedded slice at the SP1 ``#[repr(C)]`` field
    offset; word-typed fields (committed-value digests) keep their ``[word][limb]``
    shape so the eval helpers index them exactly as the Rust does.
    """

    v: Array  # (SP1_PROOF_NUM_PV_ELTS,) EF

    def _w1(self, base: int) -> Array:  # [PV_DIGEST_NUM_WORDS][4]
        return self.v[base : base + PV_DIGEST_NUM_WORDS * 4].reshape(
            PV_DIGEST_NUM_WORDS, 4
        )

    @property
    def prev_committed_value_digest(self) -> Array:
        return self._w1(0)

    @property
    def committed_value_digest(self) -> Array:
        return self._w1(32)

    @property
    def prev_deferred_proofs_digest(self) -> Array:
        return self.v[64:72]

    @property
    def deferred_proofs_digest(self) -> Array:
        return self.v[72:80]

    @property
    def pc_start(self) -> Array:
        return self.v[80:83]

    @property
    def next_pc(self) -> Array:
        return self.v[83:86]

    @property
    def prev_exit_code(self) -> Array:
        return self.v[86]

    @property
    def exit_code(self) -> Array:
        return self.v[87]

    @property
    def is_execution_shard(self) -> Array:
        return self.v[88]

    @property
    def previous_init_addr(self) -> Array:
        return self.v[89:92]

    @property
    def last_init_addr(self) -> Array:
        return self.v[92:95]

    @property
    def previous_finalize_addr(self) -> Array:
        return self.v[95:98]

    @property
    def last_finalize_addr(self) -> Array:
        return self.v[98:101]

    @property
    def previous_init_page_idx(self) -> Array:
        return self.v[101:104]

    @property
    def last_init_page_idx(self) -> Array:
        return self.v[104:107]

    @property
    def previous_finalize_page_idx(self) -> Array:
        return self.v[107:110]

    @property
    def last_finalize_page_idx(self) -> Array:
        return self.v[110:113]

    @property
    def initial_timestamp(self) -> Array:
        return self.v[113:117]

    @property
    def last_timestamp(self) -> Array:
        return self.v[117:121]

    @property
    def is_timestamp_high_eq(self) -> Array:
        return self.v[121]

    @property
    def inv_timestamp_high(self) -> Array:
        return self.v[122]

    @property
    def is_timestamp_low_eq(self) -> Array:
        return self.v[123]

    @property
    def inv_timestamp_low(self) -> Array:
        return self.v[124]

    @property
    def global_init_count(self) -> Array:
        return self.v[125]

    @property
    def global_finalize_count(self) -> Array:
        return self.v[126]

    @property
    def global_page_prot_init_count(self) -> Array:
        return self.v[127]

    @property
    def global_page_prot_finalize_count(self) -> Array:
        return self.v[128]

    @property
    def global_count(self) -> Array:
        return self.v[129]

    @property
    def global_cumulative_sum(self) -> Array:  # (14,): x[7] || y[7]
        return self.v[130:144]

    @property
    def prev_commit_syscall(self) -> Array:
        return self.v[144]

    @property
    def commit_syscall(self) -> Array:
        return self.v[145]

    @property
    def prev_commit_deferred_syscall(self) -> Array:
        return self.v[146]

    @property
    def commit_deferred_syscall(self) -> Array:
        return self.v[147]

    @property
    def initial_timestamp_inv(self) -> Array:
        return self.v[148]

    @property
    def last_timestamp_inv(self) -> Array:
        return self.v[149]

    @property
    def is_first_execution_shard(self) -> Array:
        return self.v[150]

    @property
    def is_untrusted_programs_enabled(self) -> Array:
        return self.v[151]

    @property
    def empty(self) -> Array:
        return self.v[156:160]


class _Folder:
    """The folding state of SP1's ``VerifierPublicValuesConstraintFolder``.

    ``assert_zero`` Horner-folds the constraint under ``pv_challenge``; ``send``
    and ``receive`` fingerprint with ``alpha``/``betas`` and accumulate the
    interaction's signed bus mass. ``when``/``when_not`` return a filtered view
    whose asserts multiply by the (negated) condition, mirroring SP1's
    ``FilteredAirBuilder``.
    """

    def __init__(self, pv_challenge: Array, alpha: Array, betas: Array) -> None:
        ef = alpha.dtype
        self._pv_challenge = pv_challenge
        self._alpha = alpha
        self._betas = betas
        self._ef = ef
        self._one = jnp.ones((), ef)
        self.accumulator = jnp.zeros((), ef)
        self.local_interaction_digest = jnp.zeros((), ef)

    # --- constraint folding -------------------------------------------------

    def assert_zero(self, x: Array) -> None:
        self.accumulator = self.accumulator * self._pv_challenge + x

    def assert_one(self, x: Array) -> None:
        self.assert_zero(x - self._one)

    def assert_eq(self, a: Array, b: Array) -> None:
        self.assert_zero(a - b)

    def assert_bool(self, x: Array) -> None:
        self.assert_zero(x * (x - self._one))

    def assert_all_zero(self, xs: Array) -> None:
        for i in range(xs.shape[0]):
            self.assert_zero(xs[i])

    def assert_all_eq(self, a: Array, b: Array) -> None:
        for i in range(a.shape[0]):
            self.assert_zero(a[i] - b[i])

    def when(self, cond: Array) -> "_Filtered":
        return _Filtered(self, cond)

    def when_not(self, cond: Array) -> "_Filtered":
        return _Filtered(self, self._one - cond)

    # --- interaction folding ------------------------------------------------

    def _fingerprint(self, values: Sequence[Array], kind: int) -> Array:
        denominator = self._alpha + self._betas[0] * kind
        for i, value in enumerate(values):
            denominator = denominator + self._betas[i + 1] * value
        return denominator

    def send(self, values: Sequence[Array], multiplicity: Array, kind: int) -> None:
        self.local_interaction_digest += multiplicity / self._fingerprint(values, kind)

    def receive(self, values: Sequence[Array], multiplicity: Array, kind: int) -> None:
        self.local_interaction_digest -= multiplicity / self._fingerprint(values, kind)

    # --- byte / state interaction helpers (SP1 ``ByteAirBuilder`` etc.) ------

    def send_byte(self, opcode: Array, a: Array, b: Array, c: Array, mult: Array) -> None:
        self.send([opcode, a, b, c], mult, _KIND_BYTE)

    def send_state(self, clk_high: Array, clk_low: Array, pc: Array, mult: Array) -> None:
        self.send([clk_high, clk_low, pc[0], pc[1], pc[2]], mult, _KIND_STATE)

    def receive_state(
        self, clk_high: Array, clk_low: Array, pc: Array, mult: Array
    ) -> None:
        self.receive([clk_high, clk_low, pc[0], pc[1], pc[2]], mult, _KIND_STATE)

    def const(self, value: int) -> Array:
        """An EF-embedded canonical field constant."""
        return jnp.array(value, F) * self._one

    @property
    def one(self) -> Array:
        return self._one

    @property
    def zero(self) -> Array:
        return jnp.zeros((), self._ef)


@dataclass(frozen=True)
class _Filtered:
    """SP1's ``FilteredAirBuilder``: asserts scale by the condition; interaction
    multiplicities scale too (the page-prot controls fire under
    ``is_untrusted_programs_enabled``)."""

    folder: _Folder
    cond: Array

    def assert_zero(self, x: Array) -> None:
        self.folder.assert_zero(self.cond * x)

    def assert_one(self, x: Array) -> None:
        self.assert_zero(x - self.folder.one)

    def assert_eq(self, a: Array, b: Array) -> None:
        self.assert_zero(a - b)

    def assert_all_zero(self, xs: Array) -> None:
        for i in range(xs.shape[0]):
            self.assert_zero(xs[i])

    def assert_all_eq(self, a: Array, b: Array) -> None:
        for i in range(a.shape[0]):
            self.assert_zero(a[i] - b[i])


def _eval_state(pv: _Pv, f: _Folder) -> None:
    pow8 = f.const(1 << 8)
    pow16 = f.const(1 << 16)
    initial_timestamp_high = pv.initial_timestamp[1] + pv.initial_timestamp[0] * pow8
    initial_timestamp_low = pv.initial_timestamp[3] + pv.initial_timestamp[2] * pow16
    last_timestamp_high = pv.last_timestamp[1] + pv.last_timestamp[0] * pow8
    last_timestamp_low = pv.last_timestamp[3] + pv.last_timestamp[2] * pow16
    inv8 = f.const(8)

    # Range-check the timestamp limbs.
    f.send_byte(f.const(_BYTE_RANGE), pv.initial_timestamp[0], f.const(16), f.zero, f.one)
    f.send_byte(
        f.const(_BYTE_RANGE),
        (pv.initial_timestamp[3] - f.one) / inv8,
        f.const(13),
        f.zero,
        f.one,
    )
    f.send_byte(f.const(_BYTE_RANGE), pv.last_timestamp[0], f.const(16), f.zero, f.one)
    f.send_byte(
        f.const(_BYTE_RANGE),
        (pv.last_timestamp[3] - f.one) / inv8,
        f.const(13),
        f.zero,
        f.one,
    )
    f.send_byte(
        f.const(_BYTE_U8RANGE), f.zero, pv.initial_timestamp[1], pv.initial_timestamp[2], f.one
    )
    f.send_byte(
        f.const(_BYTE_U8RANGE), f.zero, pv.last_timestamp[1], pv.last_timestamp[2], f.one
    )

    # Range-check the initial / final program-counter limbs.
    for i in range(3):
        f.send_byte(f.const(_BYTE_RANGE), pv.pc_start[i], f.const(16), f.zero, f.one)
        f.send_byte(f.const(_BYTE_RANGE), pv.next_pc[i], f.const(16), f.zero, f.one)

    # Send / receive the initial and last state.
    f.send_state(initial_timestamp_high, initial_timestamp_low, pv.pc_start, f.one)
    f.receive_state(last_timestamp_high, last_timestamp_low, pv.next_pc, f.one)

    # Non-execution shards keep timestamp and pc fixed.
    is_execution_shard = pv.is_execution_shard
    f.assert_bool(is_execution_shard)
    f.when_not(is_execution_shard).assert_eq(initial_timestamp_low, last_timestamp_low)
    f.when_not(is_execution_shard).assert_eq(initial_timestamp_high, last_timestamp_high)
    f.when_not(is_execution_shard).assert_all_eq(pv.pc_start, pv.next_pc)

    # IsZeroOperation on the high timestamp bits.
    f.assert_bool(pv.is_timestamp_high_eq)
    f.assert_eq(
        (last_timestamp_high - initial_timestamp_high) * pv.inv_timestamp_high,
        f.one - pv.is_timestamp_high_eq,
    )
    f.assert_zero(
        (last_timestamp_high - initial_timestamp_high) * pv.is_timestamp_high_eq
    )

    # IsZeroOperation on the low timestamp bits.
    f.assert_bool(pv.is_timestamp_low_eq)
    f.assert_eq(
        (last_timestamp_low - initial_timestamp_low) * pv.inv_timestamp_low,
        f.one - pv.is_timestamp_low_eq,
    )
    f.assert_zero((last_timestamp_low - initial_timestamp_low) * pv.is_timestamp_low_eq)

    # Execution shards have distinct timestamps.
    f.assert_eq(
        f.one - is_execution_shard,
        pv.is_timestamp_high_eq * pv.is_timestamp_low_eq,
    )
    f.when(is_execution_shard).assert_eq(
        (last_timestamp_high + last_timestamp_low - f.one) * pv.last_timestamp_inv,
        f.one,
    )


def _eval_first_execution_shard(pv: _Pv, f: _Folder) -> None:
    first = pv.is_first_execution_shard
    f.assert_bool(first)
    f.when(first).assert_all_eq(
        pv.initial_timestamp, jnp.stack([f.zero, f.zero, f.zero, f.one])
    )
    f.when(first).assert_one(pv.is_execution_shard)
    for i in range(PV_DIGEST_NUM_WORDS):
        f.when(first).assert_all_zero(pv.prev_committed_value_digest[i])
    f.when(first).assert_all_zero(pv.prev_deferred_proofs_digest)
    f.when(first).assert_zero(pv.prev_exit_code)
    f.when(first).assert_all_zero(pv.previous_init_addr)
    f.when(first).assert_all_zero(pv.previous_finalize_addr)
    f.when(first).assert_all_zero(pv.previous_init_page_idx)
    f.when(first).assert_all_zero(pv.previous_finalize_page_idx)
    f.when(first).assert_zero(pv.prev_commit_syscall)
    f.when(first).assert_zero(pv.prev_commit_deferred_syscall)


def _eval_exit_code(pv: _Pv, f: _Folder) -> None:
    is_execution_shard = pv.is_execution_shard
    f.assert_zero(pv.prev_exit_code * (pv.exit_code - pv.prev_exit_code))
    f.when_not(is_execution_shard).assert_eq(pv.prev_exit_code, pv.exit_code)


def _eval_committed_value_digest(pv: _Pv, f: _Folder) -> None:
    is_execution_shard = pv.is_execution_shard
    prev = pv.prev_committed_value_digest
    cur = pv.committed_value_digest
    for i in range(PV_DIGEST_NUM_WORDS):
        f.send_byte(f.const(_BYTE_U8RANGE), f.zero, prev[i][0], prev[i][1], f.one)
        f.send_byte(f.const(_BYTE_U8RANGE), f.zero, prev[i][2], prev[i][3], f.one)
        f.send_byte(f.const(_BYTE_U8RANGE), f.zero, cur[i][0], cur[i][1], f.one)
        f.send_byte(f.const(_BYTE_U8RANGE), f.zero, cur[i][2], cur[i][3], f.one)

    f.assert_bool(pv.prev_commit_syscall)
    f.assert_bool(pv.commit_syscall)
    f.when(pv.prev_commit_syscall).assert_one(pv.commit_syscall)
    f.when_not(is_execution_shard).assert_eq(pv.prev_commit_syscall, pv.commit_syscall)
    for i in range(PV_DIGEST_NUM_WORDS):
        f.when_not(is_execution_shard).assert_all_eq(prev[i], cur[i])
    for word in range(PV_DIGEST_NUM_WORDS):
        for limb in range(4):
            for i in range(PV_DIGEST_NUM_WORDS):
                f.when(prev[word][limb]).assert_all_eq(prev[i], cur[i])
    for i in range(PV_DIGEST_NUM_WORDS):
        f.when(pv.prev_commit_syscall).assert_all_eq(prev[i], cur[i])


def _eval_deferred_proofs_digest(pv: _Pv, f: _Folder) -> None:
    is_execution_shard = pv.is_execution_shard
    prev = pv.prev_deferred_proofs_digest
    cur = pv.deferred_proofs_digest
    f.assert_bool(pv.prev_commit_deferred_syscall)
    f.assert_bool(pv.commit_deferred_syscall)
    f.when(pv.prev_commit_deferred_syscall).assert_one(pv.commit_deferred_syscall)
    f.when_not(is_execution_shard).assert_eq(
        pv.prev_commit_deferred_syscall, pv.commit_deferred_syscall
    )
    f.when_not(is_execution_shard).assert_all_eq(prev, cur)
    for limb in range(prev.shape[0]):
        f.when(prev[limb]).assert_all_eq(prev, cur)
    f.when(pv.prev_commit_deferred_syscall).assert_all_eq(prev, cur)


def _eval_global_sum(pv: _Pv, f: _Folder) -> None:
    initial_x = [f.const(v) for v in _CURVE_CUMULATIVE_SUM_START_X]
    initial_y = [f.const(v) for v in _CURVE_CUMULATIVE_SUM_START_Y]
    f.send([f.zero, *initial_x, *initial_y], f.one, _KIND_GLOBAL_ACCUMULATION)
    gcs = pv.global_cumulative_sum
    f.receive(
        [pv.global_count, *[gcs[i] for i in range(14)]],
        f.one,
        _KIND_GLOBAL_ACCUMULATION,
    )


def _eval_global_memory_init(pv: _Pv, f: _Folder) -> None:
    for i in range(3):
        f.send_byte(f.const(_BYTE_RANGE), pv.previous_init_addr[i], f.const(16), f.zero, f.one)
        f.send_byte(f.const(_BYTE_RANGE), pv.last_init_addr[i], f.const(16), f.zero, f.one)
    f.send(
        [f.zero, *[pv.previous_init_addr[i] for i in range(3)], f.one],
        f.one,
        _KIND_MEMORY_GLOBAL_INIT,
    )
    f.receive(
        [pv.global_init_count, *[pv.last_init_addr[i] for i in range(3)], f.one],
        f.one,
        _KIND_MEMORY_GLOBAL_INIT,
    )


def _eval_global_memory_finalize(pv: _Pv, f: _Folder) -> None:
    for i in range(3):
        f.send_byte(
            f.const(_BYTE_RANGE), pv.previous_finalize_addr[i], f.const(16), f.zero, f.one
        )
        f.send_byte(f.const(_BYTE_RANGE), pv.last_finalize_addr[i], f.const(16), f.zero, f.one)
    f.send(
        [f.zero, *[pv.previous_finalize_addr[i] for i in range(3)], f.one],
        f.one,
        _KIND_MEMORY_GLOBAL_FINALIZE,
    )
    f.receive(
        [pv.global_finalize_count, *[pv.last_finalize_addr[i] for i in range(3)], f.one],
        f.one,
        _KIND_MEMORY_GLOBAL_FINALIZE,
    )


def _eval_global_page_prot_init(pv: _Pv, f: _Folder) -> None:
    enabled = pv.is_untrusted_programs_enabled
    f.assert_bool(enabled)
    f.send(
        [f.zero, *[pv.previous_init_page_idx[i] for i in range(3)], f.one],
        enabled,
        _KIND_PAGE_PROT_GLOBAL_INIT,
    )
    f.receive(
        [pv.global_page_prot_init_count, *[pv.last_init_page_idx[i] for i in range(3)], f.one],
        enabled,
        _KIND_PAGE_PROT_GLOBAL_INIT,
    )


def _eval_global_page_prot_finalize(pv: _Pv, f: _Folder) -> None:
    enabled = pv.is_untrusted_programs_enabled
    f.assert_bool(enabled)
    f.send(
        [f.zero, *[pv.previous_finalize_page_idx[i] for i in range(3)], f.one],
        enabled,
        _KIND_PAGE_PROT_GLOBAL_FINALIZE,
    )
    f.receive(
        [
            pv.global_page_prot_finalize_count,
            *[pv.last_finalize_page_idx[i] for i in range(3)],
            f.one,
        ],
        enabled,
        _KIND_PAGE_PROT_GLOBAL_FINALIZE,
    )


def eval_public_values(
    public_values: Array, pv_challenge: Array, alpha: Array, betas: Array
) -> tuple[Array, Array]:
    """Run SP1's ``eval_public_values`` over ``public_values`` and return
    ``(accumulator, local_interaction_digest)``.

    ``pv_challenge`` is the head's discarded public-values challenge (the
    constraint-folding RLC seed); ``alpha`` and ``betas`` are the head's
    interaction-fingerprint challenges (the same tensor the trace interactions
    use). The accumulator is zero for a well-formed vector; the GKR output
    cumulative sum equals ``-local_interaction_digest`` when the bus balances.
    """
    if public_values.shape[0] < SP1_PROOF_NUM_PV_ELTS:
        raise ValueError(
            f"public values vector has {public_values.shape[0]} elements, need "
            f"at least {SP1_PROOF_NUM_PV_ELTS}"
        )
    if betas.shape[0] < 16:
        raise ValueError(
            "the GlobalAccumulation interaction needs 16 betas (1 kind + 15 "
            f"values); got {betas.shape[0]}"
        )
    one_ef = jnp.ones((), alpha.dtype)
    pv = _Pv(public_values[:SP1_PROOF_NUM_PV_ELTS] * one_ef)
    f = _Folder(pv_challenge, alpha, betas)

    for var in range(pv.empty.shape[0]):
        f.assert_zero(pv.empty[var])

    _eval_state(pv, f)
    _eval_first_execution_shard(pv, f)
    _eval_exit_code(pv, f)
    _eval_committed_value_digest(pv, f)
    _eval_deferred_proofs_digest(pv, f)
    _eval_global_sum(pv, f)
    _eval_global_memory_init(pv, f)
    _eval_global_memory_finalize(pv, f)
    _eval_global_page_prot_init(pv, f)
    _eval_global_page_prot_finalize(pv, f)

    return f.accumulator, f.local_interaction_digest
