# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bincode serializer for the SP1 shard-proof wire format.

Produces byte buffers compatible with Rust's ``bincode::deserialize`` under
bincode's default (legacy) config: little-endian, fixed 8-byte ``u64`` length
prefixes, no varint.

KoalaBear's serde impl emits **canonical** u32, never the Montgomery raw form
the device arrays carry. The encoders assemble the wire *structure* on the
host — every reshape/ravel/row-index/EF->BF reinterpret is a numpy op over a
chunk list — and the one field-semantic step, the Mont->canonical convert
(plus the wire's derived final-poly evals, which need field arithmetic),
runs as a single jitted zone over one concatenated flat
(:func:`_canonical_zone`) — exactly one compiled program per proof shape.
The wire bytes are pinned by the goldens in ``serialize_test``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import efinfo

from sp1_zorch.types import ChipOpenedValues, MachineVerifyingKey

if TYPE_CHECKING:
    from zorch.pcs.jagged.open import Opening, StackedOpenProof
    from zorch.pcs.jagged.prover import JaggedEvalMsg
    from zorch.pcs.jagged.region import JaggedRegion

    from sp1_zorch.types import (
        LogupGkrProof,
        ShardWitness,
        TraceEvaluationClaim,
    )
from sp1_zorch.types import (
    ShardClaim,
    ShardProof,
)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", int(v))


def _usize(v: int) -> bytes:
    return _u64(v)


def _vec_prefix(length: int) -> bytes:
    return _u64(length)


@dataclass(frozen=True)
class _FieldChunk:
    """A run of Montgomery base-field limbs awaiting the canonical convert."""

    limbs: np.ndarray  # 1-D, base-field dtype, host


@dataclass(frozen=True)
class _EvalChunk:
    """A derived wire value: ``poly(coeffs)`` at ``alpha``, evaluated in the
    canonical zone (field arithmetic cannot run in numpy)."""

    coeffs: np.ndarray  # (n_coeffs,), field dtype, host
    alpha: np.ndarray  # (1,), same field dtype, host


def _host_limbs(arr: Array) -> np.ndarray:
    """Host base-field Montgomery limbs of any field array (any shape).

    The numpy ``.view`` is the same reinterpretation as
    ``lax.bitcast_convert_type`` — both read the extension element's base
    limbs in memory order — so the limb stream matches the old device
    bitcast byte-for-byte.
    """
    a = np.ascontiguousarray(np.atleast_1d(np.asarray(arr)))
    if a.dtype.itemsize > 4:
        a = a.view(efinfo(a.dtype).base_field_dtype)
    return a.reshape(-1)


def _field_chunk(arr: Array) -> _FieldChunk:
    return _FieldChunk(_host_limbs(arr))


def _limb_count(dtype: Any) -> int:
    return dtype.itemsize // 4


@frx.jit
def _canonical_zone(flat: Array, eval_parts: tuple[tuple[Array, Array], ...]) -> Array:
    """The proof's one compiled dispatch: every collected Montgomery limb,
    plus the derived final-poly evals appended behind them, converted to
    canonical u32 by a single ``convert_element_type``."""
    parts = [flat]
    for coeffs, alpha in eval_parts:
        v = _eval_poly_at(coeffs, alpha)
        if v.dtype.itemsize > 4:
            v = lax.bitcast_convert_type(v, efinfo(v.dtype).base_field_dtype)
        parts.append(fnp.ravel(v))
    return lax.convert_element_type(fnp.concatenate(parts), np.uint32)


def _flush_chunks(chunks: Sequence[Any]) -> bytes:
    """Emit the wire bytes for a chunk list: one canonical-zone dispatch for
    every field limb and derived eval, then a pure-bytes splice."""
    field_parts = [c.limbs for c in chunks if isinstance(c, _FieldChunk)]
    eval_parts = tuple((c.coeffs, c.alpha) for c in chunks if isinstance(c, _EvalChunk))
    if not field_parts and not eval_parts:
        return b"".join(chunks)

    if field_parts:
        flat = np.concatenate(field_parts)
    else:
        coeffs = eval_parts[0][0]
        base = (
            efinfo(coeffs.dtype).base_field_dtype
            if coeffs.dtype.itemsize > 4
            else coeffs.dtype
        )
        flat = np.empty((0,), dtype=base)
    out = np.asarray(_canonical_zone(flat, eval_parts)).tobytes()

    parts: list[bytes] = []
    pos = 0
    eval_pos = sum(p.shape[0] for p in field_parts)
    for c in chunks:
        if isinstance(c, _FieldChunk):
            n = c.limbs.shape[0]
            parts.append(out[4 * pos : 4 * (pos + n)])
            pos += n
        elif isinstance(c, _EvalChunk):
            n = _limb_count(c.coeffs.dtype)
            parts.append(out[4 * eval_pos : 4 * (eval_pos + n)])
            eval_pos += n
        else:
            parts.append(c)
    return b"".join(parts)


def _field_bytes(arr: Array) -> bytes:
    """Canonical LE bytes for any base- or extension-field array (any shape)."""
    return _flush_chunks([_field_chunk(arr)])


def _eval_poly_at(coeffs_row: Array, alpha: Array) -> Array:
    """Evaluate a univariate polynomial (coefficient form) at alpha via Horner.

    Field arithmetic — runs inside :func:`_canonical_zone` (or eagerly in
    tests), never in numpy.
    """
    result = fnp.zeros((), dtype=coeffs_row.dtype)
    for i in range(int(coeffs_row.shape[0]) - 1, -1, -1):
        result = result * alpha + coeffs_row[i]
    return result


def _tensor_chunks(arr: Array, dimensions: list[int]) -> list[Any]:
    """Encode ``Tensor<T>``: ``{storage: Vec<T>, dimensions: Vec<usize>}``."""
    flat = _field_chunk(arr)
    n = int(np.atleast_1d(np.asarray(arr)).size)
    return [
        _vec_prefix(n),
        flat,
        _vec_prefix(len(dimensions)),
        b"".join(_usize(d) for d in dimensions),
    ]


def _encode_tensor(arr: Array, dimensions: list[int]) -> bytes:
    return _flush_chunks(_tensor_chunks(arr, dimensions))


def _point_chunks(arr: Array) -> list[Any]:
    """Encode ``Point<T> = {values: Buffer<T>}`` = ``Vec<T>``."""
    flat = np.atleast_1d(np.asarray(arr))
    return [_vec_prefix(int(flat.shape[0])), _field_chunk(flat)]


def _encode_point(arr: Array) -> bytes:
    return _flush_chunks(_point_chunks(arr))


def _partial_sumcheck_chunks(
    round_polys: Array, claimed_sum: Array, point: Array
) -> list[Any]:
    """Encode ``PartialSumcheckProof<EF>``: ``{univariate_polys: Vec<Vec<EF>>,
    claimed_sum: EF, point_and_eval: (Point<EF>, EF)}``. The wire's eval is
    the last round polynomial at ``point[0]`` — the final fold's value, with
    ``point[0]`` the last challenge in SP1's insert-at-front point order."""
    polys = np.ascontiguousarray(np.asarray(round_polys))
    pt = np.ascontiguousarray(np.atleast_1d(np.asarray(point)))
    n_rounds = int(polys.shape[0])
    n_coeffs = int(polys.shape[1])

    parts: list[Any] = [_vec_prefix(n_rounds)]
    for r in range(n_rounds):
        parts.append(_vec_prefix(n_coeffs))
        parts.append(_field_chunk(polys[r]))

    parts.append(_field_chunk(claimed_sum))
    parts.extend(_point_chunks(pt))
    parts.append(_EvalChunk(np.ascontiguousarray(polys[-1]), pt[0:1]))
    return parts


def _encode_partial_sumcheck_proof(
    round_polys: Array, claimed_sum: Array, point: Array
) -> bytes:
    return _flush_chunks(_partial_sumcheck_chunks(round_polys, claimed_sum, point))


def _logup_gkr_proof_chunks(proof: LogupGkrProof, max_log_row_count: int) -> list[Any]:
    """Encode ``LogupGkrProof<F, EF>`` (rust field order: circuit_output,
    round_proofs, logup_evaluations, witness — the last is `pow_witness` here).

    ``proof`` is ``sp1_zorch.logup_gkr.prover.LogupGkrProof``; the wire's
    per-layer ``point_and_eval`` reads each round proof's ``point``, retained
    by zorch at prove time.
    """
    parts: list[Any] = []

    n_num = int(np.atleast_1d(np.asarray(proof.circuit_output.numerator)).shape[0])
    parts.extend(_tensor_chunks(proof.circuit_output.numerator, [n_num, 1]))
    n_den = int(np.atleast_1d(np.asarray(proof.circuit_output.denominator)).shape[0])
    parts.extend(_tensor_chunks(proof.circuit_output.denominator, [n_den, 1]))

    parts.append(_vec_prefix(len(proof.round_proofs)))
    for rp in proof.round_proofs:
        parts.append(_field_chunk(rp.numerator_0))
        parts.append(_field_chunk(rp.numerator_1))
        parts.append(_field_chunk(rp.denominator_0))
        parts.append(_field_chunk(rp.denominator_1))
        parts.extend(_partial_sumcheck_chunks(rp.round_polys, rp.claim, rp.point))

    # SP1's eval_point has exactly max_log_row_count dims after all GKR
    # rounds. The prover-side point may overshoot — trim to the tail.
    gkr_point = np.atleast_1d(np.asarray(proof.eval_point))
    if gkr_point.shape[0] > max_log_row_count:
        gkr_point = gkr_point[-max_log_row_count:]
    parts.extend(_point_chunks(gkr_point))

    chip_map = proof.chip_openings
    parts.append(_vec_prefix(len(chip_map)))
    for name in sorted(chip_map):  # BTreeMap: ascending key order
        name_bytes = name.encode("utf-8")
        parts.append(_vec_prefix(len(name_bytes)))
        parts.append(name_bytes)
        ce = chip_map[name]
        n_main = int(np.atleast_1d(np.asarray(ce.main)).shape[0])
        parts.extend(_tensor_chunks(ce.main, [n_main]))
        if ce.preprocessed is not None:
            parts.append(b"\x01")
            n_prep = int(np.atleast_1d(np.asarray(ce.preprocessed)).shape[0])
            parts.extend(_tensor_chunks(ce.preprocessed, [n_prep]))
        else:
            parts.append(b"\x00")

    parts.append(_field_chunk(proof.pow_witness))
    return parts


def _encode_logup_gkr_proof(proof: LogupGkrProof, max_log_row_count: int) -> bytes:
    return _flush_chunks(_logup_gkr_proof_chunks(proof, max_log_row_count))


def _digest_chunks(arr: Any) -> list[Any]:
    """Encode ``GC::Digest = [F; 8]`` = 8 × canonical u32."""
    if hasattr(arr, "dtype"):
        return [_FieldChunk(_host_limbs(arr)[:8])]
    return [struct.pack(f"<{len(arr)}I", *[int(x) for x in arr])[:32]]


def _encode_digest(arr: Any) -> bytes:
    return _flush_chunks(_digest_chunks(arr))


def _chip_opened_values_chunks(
    cov: ChipOpenedValues, max_log_row_count: int
) -> list[Any]:
    """Encode ``ChipOpenedValues<F, EF>``. A chip without a preprocessed
    trace serializes an EMPTY ``Vec`` — unlike the GKR chip openings, whose
    missing prep is an ``Option`` tag byte."""
    parts: list[Any] = []

    if cov.preprocessed_evals is not None:
        n = int(cov.preprocessed_evals.shape[0])
        parts.append(_vec_prefix(n))
        parts.append(_field_chunk(cov.preprocessed_evals))
    else:
        parts.append(_vec_prefix(0))

    n = int(cov.main_evals.shape[0])
    parts.append(_vec_prefix(n))
    parts.append(_field_chunk(cov.main_evals))

    n_bits = max_log_row_count + 1
    degree_bits = [(cov.degree >> bit) & 1 for bit in range(n_bits - 1, -1, -1)]
    parts.append(_vec_prefix(n_bits))
    parts.append(struct.pack(f"<{n_bits}I", *degree_bits))

    return parts


def _encode_chip_opened_values(cov: ChipOpenedValues, max_log_row_count: int) -> bytes:
    return _flush_chunks(_chip_opened_values_chunks(cov, max_log_row_count))


def _shard_opened_values_chunks(
    chip_opened_values: Sequence[ChipOpenedValues],
    chip_names: Sequence[str],
    max_log_row_count: int,
) -> list[Any]:
    """Encode ``ShardOpenedValues<F, EF> = {chips: BTreeMap<String,
    ChipOpenedValues>}`` — ascending chip-name order."""
    sorted_pairs = sorted(zip(chip_names, chip_opened_values, strict=True))
    parts: list[Any] = [_vec_prefix(len(sorted_pairs))]
    for name, cov in sorted_pairs:
        name_bytes = name.encode("utf-8")
        parts.append(_vec_prefix(len(name_bytes)))
        parts.append(name_bytes)
        parts.extend(_chip_opened_values_chunks(cov, max_log_row_count))
    return parts


def _encode_shard_opened_values(
    chip_opened_values: Sequence[ChipOpenedValues],
    chip_names: Sequence[str],
    max_log_row_count: int,
) -> bytes:
    return _flush_chunks(
        _shard_opened_values_chunks(chip_opened_values, chip_names, max_log_row_count)
    )


def _batch_openings_chunks(opening: Opening, root_digest: Array) -> list[Any]:
    """Encode ``MerkleTreeOpeningAndProof<GC>`` from one vmapped SMCS batch
    opening: the opened rows as ``Tensor<F>`` with dimensions ``[num_queries,
    width]``, then the proof — the **raw** Merkle root (the sibling paths
    reconstruct it, not the separator-bound commitment the transcript
    observes), depth, width, and the sibling digests as a ``Tensor`` with
    dimensions ``[num_queries, depth]`` (query-major)."""
    rows, paths = opening
    num_queries, width = (int(s) for s in rows.shape)
    depth = len(paths)

    parts: list[Any] = _tensor_chunks(rows, [num_queries, width])

    parts.extend(_digest_chunks(root_digest))
    parts.append(_usize(depth))
    parts.append(_usize(width))
    parts.append(_vec_prefix(num_queries * depth))
    # ``paths`` is level-major, one (Q, digest) array per tree level; the
    # wire wants every query's full path contiguously.
    if depth > 0:
        parts.append(_field_chunk(np.stack([np.asarray(p) for p in paths], axis=1)))
    parts.append(_vec_prefix(2))
    parts.append(_usize(num_queries))
    parts.append(_usize(depth))
    return parts


def _pack_batch_openings(opening: Opening, root_digest: Array) -> bytes:
    return _flush_chunks(_batch_openings_chunks(opening, root_digest))


def _basefold_proof_chunks(
    open_proof: StackedOpenProof, component_raw_roots: Sequence[Array]
) -> list[Any]:
    """Encode ``BasefoldProof<GC>``.

    ``component_raw_roots`` are the commit-time raw Merkle roots of the
    committed rounds, in the same order as ``component_openings`` — the
    proof retains only the fold layers' raw roots (``fri_raw_roots``), so
    the commit stage supplies the component ones.
    """
    parts: list[Any] = []

    # Vec<(EF, EF)>: one pair-count prefix, then the (s(0), s(1)) pairs
    # contiguous — exactly the (num_vars, 2) array's row-major bytes.
    msgs = open_proof.univariate_messages
    parts.append(_vec_prefix(int(msgs.shape[0])))
    parts.append(_field_chunk(msgs))

    # Vec<Digest>: each row is exactly [F; 8], so the stacked array's bytes
    # are the digests back to back.
    fri_commitments = open_proof.fri_commitments
    parts.append(_vec_prefix(int(fri_commitments.shape[0])))
    parts.append(_field_chunk(fri_commitments))

    comp = open_proof.component_openings
    parts.append(_vec_prefix(len(comp)))
    for opening, raw_root in zip(comp, component_raw_roots, strict=True):
        parts.extend(_batch_openings_chunks(opening, raw_root))

    query = open_proof.query_openings
    parts.append(_vec_prefix(len(query)))
    fri_raw_roots = np.asarray(open_proof.fri_raw_roots)
    for i, opening in enumerate(query):
        parts.extend(_batch_openings_chunks(opening, fri_raw_roots[i]))

    parts.append(_field_chunk(open_proof.final_poly))
    parts.append(_field_chunk(open_proof.pow_witness))
    return parts


def _encode_basefold_proof(
    open_proof: StackedOpenProof, component_raw_roots: Sequence[Array]
) -> bytes:
    return _flush_chunks(_basefold_proof_chunks(open_proof, component_raw_roots))


def _evaluation_proof_chunks(
    eval_msg: JaggedEvalMsg,
    open_proof: StackedOpenProof,
    component_raw_roots: Sequence[Array],
    component_commitments: Sequence[Array],
    row_counts_and_column_counts: Sequence[Sequence[tuple[int, int]]],
    max_log_row_count: int,
) -> list[Any]:
    """Encode ``JaggedPcsProof<GC, StackedBasefoldProof<GC>>``.

    ``row_counts_and_column_counts`` is the per-committed-round ``(row_count,
    column_count)`` chip layout (``JaggedRegion`` order, stacking dummies
    included). ``component_raw_roots`` are the rounds' pre-binding raw Merkle
    roots the batch openings reconstruct from their sibling paths.
    ``component_commitments`` are the SMCS commitments (``smcs.commit()`` output,
    before structure binding) the wire's ``original_commitments`` carries — SP1's
    recursion applies ``bind_structure`` to each and checks it against the vk
    (round 0 is the preprocessed commit), so a raw root here fails that check.
    ``log_m`` is the outer sumcheck's round count, read off the proof.
    """
    parts: list[Any] = _basefold_proof_chunks(open_proof, component_raw_roots)

    parts.append(_vec_prefix(len(open_proof.batch_evals)))
    for evals in open_proof.batch_evals:
        n_evals = int(np.atleast_1d(np.asarray(evals)).shape[0])
        parts.extend(_tensor_chunks(evals, [n_evals]))

    outer_polys = eval_msg.outer_sumcheck_polys
    parts.extend(
        _partial_sumcheck_chunks(
            outer_polys,
            eval_msg.outer_sumcheck_claim,
            eval_msg.outer_sumcheck_point,
        )
    )

    parts.extend(
        _partial_sumcheck_chunks(
            eval_msg.inner_sumcheck_polys,
            eval_msg.inner_claimed_sum,
            eval_msg.inner_point,
        )
    )

    parts.append(_vec_prefix(len(row_counts_and_column_counts)))
    for round_counts in row_counts_and_column_counts:
        parts.append(_vec_prefix(len(round_counts)))
        for row_count, column_count in round_counts:
            parts.append(_usize(row_count))
            parts.append(_usize(column_count))

    parts.append(_vec_prefix(len(component_commitments)))
    for commitment in component_commitments:
        parts.extend(_digest_chunks(commitment))

    parts.append(_field_chunk(eval_msg.dense_eval))
    parts.append(_usize(max_log_row_count))
    parts.append(_usize(int(outer_polys.shape[0])))
    return parts


def _encode_evaluation_proof(
    eval_msg: JaggedEvalMsg,
    open_proof: StackedOpenProof,
    component_raw_roots: Sequence[Array],
    component_commitments: Sequence[Array],
    row_counts_and_column_counts: Sequence[Sequence[tuple[int, int]]],
    *,
    max_log_row_count: int,
) -> bytes:
    return _flush_chunks(
        _evaluation_proof_chunks(
            eval_msg,
            open_proof,
            component_raw_roots,
            component_commitments,
            row_counts_and_column_counts,
            max_log_row_count,
        )
    )


def encode_vk(vk: MachineVerifyingKey) -> bytes:
    """Encode ``MachineVerifyingKey<SP1GlobalContext>`` to bincode.

    Serde field order is pc_start, initial_global_cumulative_sum (SepticDigest
    = x then y), preprocessed_commit, enable_untrusted_programs — NOT the
    transcript ``observe_into`` order, which leads with the commit.
    """
    return _flush_chunks(
        [
            _field_chunk(vk.pc_start),
            _field_chunk(vk.cum_sum_x),
            _field_chunk(vk.cum_sum_y),
            _field_chunk(vk.preprocessed_commit),
            struct.pack("<I", int(vk.enable_untrusted)),
        ]
    )


def chip_opened_values(
    evaluation: TraceEvaluationClaim, main: JaggedRegion
) -> list[ChipOpenedValues]:
    """Convert the zerocheck reduced claim's opened values to the wire's
    per-chip shape. The split off the final folded traces is the zerocheck
    stage's (``zerocheck.prover.split_opened_values`` — one view shared with
    the transcript absorbs and the jagged-eval claims); ``degree`` is the
    chip's live row count — the height whose bits the wire spells out.
    """
    values = []
    for i, name in enumerate(main.chip_names):
        ev = evaluation.opened_values[name]
        values.append(
            ChipOpenedValues(
                preprocessed_evals=ev.preprocessed,
                main_evals=ev.main,
                degree=int(main.chip_heights[i]),
            )
        )
    return values


def encode_shard_proof(
    claim: ShardClaim,
    witness: ShardWitness,
    proof: ShardProof,
    evaluation: TraceEvaluationClaim,
    commit_digest_layers: tuple[list[Array], ...],
    *,
    max_log_row_count: int,
) -> bytes:
    """Encode ``ShardProof<SP1GlobalContext, SP1PcsProofInner>`` to bincode.

    Takes the shard statement and its proof, plus the two prover-only products
    the wire needs but no claim carries: the zerocheck reduced claim's opened
    values and the commit-time digest trees. The per-round SMCS commitments
    ride the jagged proof, since the verifier cannot derive them. Serde field
    order: public values, main commitment, LogUp-GKR proof, zerocheck partial
    sumcheck, shard opened values, evaluation proof.

    The whole proof flushes as ONE chunk list — a single
    :func:`_canonical_zone` dispatch (one compiled program per proof shape),
    not a per-array eager convert.
    """
    gkr_proof = proof.gkr
    zerocheck_proof = proof.zerocheck
    jagged_proof = proof.jagged
    parts: list[Any] = _point_chunks(claim.public_values)

    parts.append(_field_chunk(proof.commitment))

    parts.extend(_logup_gkr_proof_chunks(gkr_proof, max_log_row_count))

    # The wire's zerocheck point is SP1's insert-at-front order — the
    # accumulated round challenges reversed, the same z_row order the
    # jagged-eval stage consumes.
    parts.extend(
        _partial_sumcheck_chunks(
            zerocheck_proof.msgs.round_poly,
            zerocheck_proof.claimed_sum,
            np.atleast_1d(np.asarray(zerocheck_proof.msgs.challenge))[::-1],
        )
    )

    parts.extend(
        _shard_opened_values_chunks(
            chip_opened_values(evaluation, witness.main_region),
            list(witness.main_region.chip_names),
            max_log_row_count,
        )
    )

    # Committed-round order is [prep, main] — the order the PCS commit half
    # wrote its StackedRounds in.
    regions = [
        region
        for region in (witness.prep_region, witness.main_region)
        if region is not None
    ]
    component_raw_roots = [
        digest_layers[-1][0] for digest_layers in commit_digest_layers
    ]
    # smcs_commitments = the SMCS commitment (pre-structure-binding), retained
    # off the PCS commit in the same [prep, main] order as commit_digest_layers.
    component_commitments = jagged_proof.smcs_commitments.in_round_order()
    row_column_counts = [
        list(zip(region.row_counts, region.column_counts, strict=True))
        for region in regions
    ]
    parts.extend(
        _evaluation_proof_chunks(
            jagged_proof.eval,
            jagged_proof.open,
            component_raw_roots,
            component_commitments,
            row_column_counts,
            max_log_row_count,
        )
    )
    return _flush_chunks(parts)
