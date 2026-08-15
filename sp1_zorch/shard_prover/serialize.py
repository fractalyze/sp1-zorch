# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bincode serializer for the SP1 shard-proof wire format.

Produces byte buffers compatible with Rust's ``bincode::deserialize`` under
bincode's default (legacy) config: little-endian, fixed 8-byte ``u64`` length
prefixes, no varint.

KoalaBear's serde impl emits **canonical** u32, never the Montgomery raw form
the device arrays carry — ``_canonical_u32`` converts via
``lax.convert_element_type(..., uint32)``. Extension-field elements flatten to
their base-field limbs before conversion.

The wire interleaves those field bytes with host-computed length prefixes, but
a proof's arrays are all read at once: every encoder here splits into a
``*_arrays`` half naming what it reads and a ``*_bytes`` half assembling the
result from already-pulled segments, so one ``_field_bytes_many`` serves a
whole section. The prover holds the device across the encode, so the number of
pulls — not their volume — is what the rest of the pipeline waits behind.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import efinfo

from sp1_zorch.types import ChipOpenedValues, MachineVerifyingKey

if TYPE_CHECKING:
    from zorch.logup_gkr.jagged_prover import JaggedLayerProof
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


def _canonical_u32(arr: Array) -> Array:
    """Flat canonical ``uint32`` for any base- or extension-field array."""
    a = fnp.atleast_1d(arr)
    if a.dtype.itemsize > 4:
        a = lax.bitcast_convert_type(a, efinfo(a.dtype).base_field_dtype)
    return lax.convert_element_type(fnp.ravel(a), np.uint32)


def _field_bytes(arr: Array) -> bytes:
    """Canonical LE bytes for any base- or extension-field array (any shape)."""
    return np.asarray(_canonical_u32(arr)).tobytes()


def _field_bytes_many(arrays: Sequence[Array]) -> Iterator[bytes]:
    """``_field_bytes`` for several arrays across ONE device-to-host pull.

    Yields one segment per input, in order. Concatenating on device first
    trades a few hundred synchronisation points for one. The cost is device
    residency: the section's canonical parts and the joined buffer are live
    together, ~2.7 MB for a core shard's evaluation proof, and the caller's
    own array list holds ~1.2 MB more on top of that.
    """
    if not arrays:
        return iter(())
    parts = [_canonical_u32(a) for a in arrays]
    joined = np.asarray(fnp.concatenate(parts)).tobytes()

    def _split() -> Iterator[bytes]:
        offset = 0
        for part in parts:
            end = offset + int(part.shape[0]) * 4
            yield joined[offset:end]
            offset = end

    return _split()


def _leading_len(arr: Array) -> int:
    """The ``Vec``/``Point`` length the wire prefixes an array with."""
    return int(fnp.atleast_1d(arr).shape[0])


def _eval_poly_at(coeffs: Array, alpha: Array) -> Array:
    """Evaluate univariate polynomials (coefficient form along the last axis)
    at alpha via Horner.

    Batched over any leading axes: a Horner step is a dispatch per coefficient,
    so a layer chain whose polynomials share a degree evaluates in one sweep
    rather than one sweep per layer.
    """
    result = fnp.zeros(alpha.shape, dtype=coeffs.dtype)
    for i in range(int(coeffs.shape[-1]) - 1, -1, -1):
        result = result * alpha + coeffs[..., i]
    return result


def _tensor_bytes(storage: bytes, count: int, dimensions: list[int]) -> bytes:
    """Encode ``Tensor<T>``: ``{storage: Vec<T>, dimensions: Vec<usize>}``."""
    return (
        _vec_prefix(count)
        + storage
        + _vec_prefix(len(dimensions))
        + b"".join(_usize(d) for d in dimensions)
    )


def _encode_tensor(arr: Array, dimensions: list[int]) -> bytes:
    return _tensor_bytes(_field_bytes(arr), int(arr.size), dimensions)


def _point_bytes(values: bytes, count: int) -> bytes:
    """Encode ``Point<T> = {values: Buffer<T>}`` = ``Vec<T>``."""
    return _vec_prefix(count) + values


def _encode_point(arr: Array) -> bytes:
    return _point_bytes(_field_bytes(arr), _leading_len(arr))


def _partial_sumcheck_arrays(
    round_polys: Array,
    claimed_sum: Array,
    point: Array,
    final_eval: Array | None = None,
) -> list[Array]:
    """What a ``PartialSumcheckProof<EF>`` reads. The wire's eval is the last
    round polynomial at ``point[0]`` — the final fold's value, with
    ``point[0]`` the last challenge in SP1's insert-at-front point order. A
    caller holding several sumchecks of one degree passes it in, having swept
    them together."""
    if final_eval is None:
        final_eval = _eval_poly_at(round_polys[-1], point[0])
    return [round_polys, claimed_sum, point, final_eval]


def _partial_sumcheck_bytes(
    round_polys: Array, point: Array, segments: Iterator[bytes]
) -> bytes:
    """Encode ``PartialSumcheckProof<EF>``: ``{univariate_polys: Vec<Vec<EF>>,
    claimed_sum: EF, point_and_eval: (Point<EF>, EF)}``. ``round_polys`` is
    rectangular, so its row-major bytes already carry every round's Vec back
    to back — the rounds only need their length prefixes woven in."""
    n_rounds = int(round_polys.shape[0])
    n_coeffs = int(round_polys.shape[1])
    polys, claimed_sum, point_bytes, final_eval = (next(segments) for _ in range(4))
    stride = len(polys) // n_rounds
    coeff_prefix = _vec_prefix(n_coeffs)

    parts = [_vec_prefix(n_rounds)]
    for r in range(n_rounds):
        parts.append(coeff_prefix)
        parts.append(polys[r * stride : (r + 1) * stride])

    parts.append(claimed_sum)
    parts.append(_point_bytes(point_bytes, _leading_len(point)))
    parts.append(final_eval)
    return b"".join(parts)


def _encode_partial_sumcheck_proof(
    round_polys: Array, claimed_sum: Array, point: Array
) -> bytes:
    arrays = _partial_sumcheck_arrays(round_polys, claimed_sum, point)
    return _partial_sumcheck_bytes(round_polys, point, _field_bytes_many(arrays))


def _gkr_final_evals(round_proofs: Sequence[JaggedLayerProof]) -> list[Array]:
    """Each layer's wire eval, from one Horner sweep across the chain — the
    jagged layers share a round-poly degree, so a mismatched chain raises in
    ``stack`` rather than reaching the wire."""
    if not round_proofs:
        return []
    evals = _eval_poly_at(
        fnp.stack([rp.round_polys[-1] for rp in round_proofs]),
        fnp.stack([rp.point[0] for rp in round_proofs]),
    )
    return [evals[i] for i in range(len(round_proofs))]


def _gkr_point(proof: LogupGkrProof, max_log_row_count: int) -> Array:
    # SP1's eval_point has exactly max_log_row_count dims after all GKR
    # rounds. The prover-side point may overshoot — trim to the tail.
    point = proof.eval_point
    if point.shape[0] > max_log_row_count:
        return point[-max_log_row_count:]
    return point


def _encode_logup_gkr_proof(proof: LogupGkrProof, max_log_row_count: int) -> bytes:
    """Encode ``LogupGkrProof<F, EF>`` (rust field order: circuit_output,
    round_proofs, logup_evaluations, witness — the last is `pow_witness` here).

    ``proof`` is ``sp1_zorch.logup_gkr.prover.LogupGkrProof``; the wire's
    per-layer ``point_and_eval`` reads each round proof's ``point``, retained
    by zorch at prove time.
    """
    output = proof.circuit_output
    point = _gkr_point(proof, max_log_row_count)
    names = sorted(proof.chip_openings)  # BTreeMap: ascending key order

    arrays: list[Array] = [output.numerator, output.denominator]
    for rp, final_eval in zip(
        proof.round_proofs, _gkr_final_evals(proof.round_proofs), strict=True
    ):
        arrays += [rp.numerator_0, rp.numerator_1, rp.denominator_0, rp.denominator_1]
        arrays += _partial_sumcheck_arrays(
            rp.round_polys, rp.claim, rp.point, final_eval
        )
    arrays.append(point)
    for name in names:
        ce = proof.chip_openings[name]
        arrays.append(ce.main)
        if ce.preprocessed is not None:
            arrays.append(ce.preprocessed)
    arrays.append(proof.pow_witness)
    seg = _field_bytes_many(arrays)

    parts = [
        _tensor_bytes(
            next(seg), int(output.numerator.size), [_leading_len(output.numerator), 1]
        ),
        _tensor_bytes(
            next(seg),
            int(output.denominator.size),
            [_leading_len(output.denominator), 1],
        ),
        _vec_prefix(len(proof.round_proofs)),
    ]
    for rp in proof.round_proofs:
        parts += [next(seg), next(seg), next(seg), next(seg)]
        parts.append(_partial_sumcheck_bytes(rp.round_polys, rp.point, seg))

    parts.append(_point_bytes(next(seg), _leading_len(point)))

    parts.append(_vec_prefix(len(names)))
    for name in names:
        name_bytes = name.encode("utf-8")
        parts.append(_vec_prefix(len(name_bytes)))
        parts.append(name_bytes)
        ce = proof.chip_openings[name]
        parts.append(
            _tensor_bytes(next(seg), int(ce.main.size), [_leading_len(ce.main)])
        )
        if ce.preprocessed is not None:
            parts.append(b"\x01")
            parts.append(
                _tensor_bytes(
                    next(seg),
                    int(ce.preprocessed.size),
                    [_leading_len(ce.preprocessed)],
                )
            )
        else:
            parts.append(b"\x00")

    parts.append(next(seg))
    return b"".join(parts)


def _encode_digest(arr: Any) -> bytes:
    """Encode ``GC::Digest = [F; 8]`` = 8 × canonical u32."""
    if hasattr(arr, "dtype"):
        return _field_bytes(arr)[:32]
    return struct.pack(f"<{len(arr)}I", *[int(x) for x in arr])[:32]


def _chip_opened_values_arrays(cov: ChipOpenedValues) -> list[Array]:
    if cov.preprocessed_evals is None:
        return [cov.main_evals]
    return [cov.preprocessed_evals, cov.main_evals]


def _chip_opened_values_bytes(
    cov: ChipOpenedValues, max_log_row_count: int, segments: Iterator[bytes]
) -> bytes:
    """Encode ``ChipOpenedValues<F, EF>``. A chip without a preprocessed
    trace serializes an EMPTY ``Vec`` — unlike the GKR chip openings, whose
    missing prep is an ``Option`` tag byte."""
    parts = []

    if cov.preprocessed_evals is not None:
        parts.append(_vec_prefix(int(cov.preprocessed_evals.shape[0])))
        parts.append(next(segments))
    else:
        parts.append(_vec_prefix(0))

    parts.append(_vec_prefix(int(cov.main_evals.shape[0])))
    parts.append(next(segments))

    n_bits = max_log_row_count + 1
    degree_bits = [(cov.degree >> bit) & 1 for bit in range(n_bits - 1, -1, -1)]
    parts.append(_vec_prefix(n_bits))
    parts.append(struct.pack(f"<{n_bits}I", *degree_bits))

    return b"".join(parts)


def _encode_chip_opened_values(cov: ChipOpenedValues, max_log_row_count: int) -> bytes:
    segments = _field_bytes_many(_chip_opened_values_arrays(cov))
    return _chip_opened_values_bytes(cov, max_log_row_count, segments)


def _encode_shard_opened_values(
    chip_opened_values: Sequence[ChipOpenedValues],
    chip_names: Sequence[str],
    max_log_row_count: int,
) -> bytes:
    """Encode ``ShardOpenedValues<F, EF> = {chips: BTreeMap<String,
    ChipOpenedValues>}`` — ascending chip-name order."""
    sorted_pairs = sorted(zip(chip_names, chip_opened_values, strict=True))
    arrays: list[Array] = []
    for _, cov in sorted_pairs:
        arrays += _chip_opened_values_arrays(cov)
    seg = _field_bytes_many(arrays)

    parts = [_vec_prefix(len(sorted_pairs))]
    for name, cov in sorted_pairs:
        name_bytes = name.encode("utf-8")
        parts.append(_vec_prefix(len(name_bytes)))
        parts.append(name_bytes)
        parts.append(_chip_opened_values_bytes(cov, max_log_row_count, seg))
    return b"".join(parts)


def _batch_opening_arrays(opening: Opening, root_digest: Array) -> list[Array]:
    """What one vmapped SMCS batch opening reads: the opened rows, the raw
    Merkle root, and the sibling paths concatenated level-major."""
    rows, paths = opening
    arrays = [rows, root_digest]
    if paths:
        arrays.append(fnp.concatenate(paths, axis=0))
    return arrays


def _batch_opening_bytes(opening: Opening, segments: Iterator[bytes]) -> bytes:
    """Encode ``MerkleTreeOpeningAndProof<GC>``: the opened rows as
    ``Tensor<F>`` with dimensions ``[num_queries, width]``, then the proof —
    the **raw** Merkle root (the sibling paths reconstruct it, not the
    separator-bound commitment the transcript observes), depth, width, and the
    sibling digests as a ``Tensor`` with dimensions ``[num_queries, depth]``
    (query-major)."""
    rows, paths = opening
    num_queries, width = (int(s) for s in rows.shape)
    depth = len(paths)

    parts = [_tensor_bytes(next(segments), int(rows.size), [num_queries, width])]
    parts.append(next(segments)[:32])
    parts.append(_usize(depth))
    parts.append(_usize(width))
    parts.append(_vec_prefix(num_queries * depth))
    if depth > 0:
        # `paths` arrives level-major, one (num_queries, digest) array per tree
        # level, and the wire wants each query's full path contiguously. The
        # transpose rides the pulled u32 rather than the device: a `stack`
        # would cost one lay-in kernel per level, ~300 per shard proof.
        level_major = np.frombuffer(next(segments), dtype=np.uint32)
        paths_bytes = level_major.reshape(depth, num_queries, -1).transpose(1, 0, 2)
        parts.append(paths_bytes.tobytes())
    parts.append(_vec_prefix(2))
    parts.append(_usize(num_queries))
    parts.append(_usize(depth))
    return b"".join(parts)


def _pack_batch_openings(opening: Opening, root_digest: Array) -> bytes:
    segments = _field_bytes_many(_batch_opening_arrays(opening, root_digest))
    return _batch_opening_bytes(opening, segments)


def _basefold_arrays(
    open_proof: StackedOpenProof, component_raw_roots: Sequence[Array]
) -> list[Array]:
    arrays: list[Array] = [open_proof.univariate_messages, open_proof.fri_commitments]
    for opening, raw_root in zip(
        open_proof.component_openings, component_raw_roots, strict=True
    ):
        arrays += _batch_opening_arrays(opening, raw_root)
    for i, opening in enumerate(open_proof.query_openings):
        arrays += _batch_opening_arrays(opening, open_proof.fri_raw_roots[i])
    arrays.append(open_proof.final_poly)
    arrays.append(open_proof.pow_witness)
    return arrays


def _basefold_bytes(
    open_proof: StackedOpenProof,
    segments: Iterator[bytes],
) -> bytes:
    """Encode ``BasefoldProof<GC>``."""
    # Vec<(EF, EF)>: one pair-count prefix, then the (s(0), s(1)) pairs
    # contiguous — exactly the (num_vars, 2) array's row-major bytes.
    parts = [_vec_prefix(int(open_proof.univariate_messages.shape[0])), next(segments)]

    # Vec<Digest>: each row is exactly [F; 8], so the stacked array's bytes
    # are the digests back to back.
    parts.append(_vec_prefix(int(open_proof.fri_commitments.shape[0])))
    parts.append(next(segments))

    parts.append(_vec_prefix(len(open_proof.component_openings)))
    for opening in open_proof.component_openings:
        parts.append(_batch_opening_bytes(opening, segments))

    parts.append(_vec_prefix(len(open_proof.query_openings)))
    for opening in open_proof.query_openings:
        parts.append(_batch_opening_bytes(opening, segments))

    parts.append(next(segments))
    parts.append(next(segments))
    return b"".join(parts)


def _encode_basefold_proof(
    open_proof: StackedOpenProof, component_raw_roots: Sequence[Array]
) -> bytes:
    """Encode ``BasefoldProof<GC>``.

    ``component_raw_roots`` are the commit-time raw Merkle roots of the
    committed rounds, in the same order as ``component_openings`` — the
    proof retains only the fold layers' raw roots (``fri_raw_roots``), so
    the commit stage supplies the component ones.
    """
    arrays = _basefold_arrays(open_proof, component_raw_roots)
    return _basefold_bytes(open_proof, _field_bytes_many(arrays))


def _encode_evaluation_proof(
    eval_msg: JaggedEvalMsg,
    open_proof: StackedOpenProof,
    component_raw_roots: Sequence[Array],
    component_commitments: Sequence[Array],
    row_counts_and_column_counts: Sequence[Sequence[tuple[int, int]]],
    max_log_row_count: int,
) -> bytes:
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
    outer_polys = eval_msg.outer_sumcheck_polys
    arrays = _basefold_arrays(open_proof, component_raw_roots)
    arrays += list(open_proof.batch_evals)
    arrays += _partial_sumcheck_arrays(
        outer_polys, eval_msg.outer_sumcheck_claim, eval_msg.outer_sumcheck_point
    )
    arrays += _partial_sumcheck_arrays(
        eval_msg.inner_sumcheck_polys, eval_msg.inner_claimed_sum, eval_msg.inner_point
    )
    arrays.append(eval_msg.dense_eval)
    seg = _field_bytes_many(arrays)

    parts = [_basefold_bytes(open_proof, seg)]

    parts.append(_vec_prefix(len(open_proof.batch_evals)))
    for evals in open_proof.batch_evals:
        parts.append(_tensor_bytes(next(seg), int(evals.size), [_leading_len(evals)]))

    parts.append(
        _partial_sumcheck_bytes(outer_polys, eval_msg.outer_sumcheck_point, seg)
    )
    parts.append(
        _partial_sumcheck_bytes(
            eval_msg.inner_sumcheck_polys, eval_msg.inner_point, seg
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
        parts.append(_encode_digest(commitment))

    parts.append(next(seg))
    parts.append(_usize(max_log_row_count))
    parts.append(_usize(int(outer_polys.shape[0])))
    return b"".join(parts)


def encode_vk(vk: MachineVerifyingKey) -> bytes:
    """Encode ``MachineVerifyingKey<SP1GlobalContext>`` to bincode.

    Serde field order is pc_start, initial_global_cumulative_sum (SepticDigest
    = x then y), preprocessed_commit, enable_untrusted_programs — NOT the
    transcript ``observe_into`` order, which leads with the commit.
    """
    seg = _field_bytes_many(
        [vk.pc_start, vk.cum_sum_x, vk.cum_sum_y, vk.preprocessed_commit]
    )
    return b"".join([*seg, struct.pack("<I", int(vk.enable_untrusted))])


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
    ride the jagged proof, since the verifier cannot derive them. Serde field order: public values,
    main commitment, LogUp-GKR proof, zerocheck partial sumcheck, shard opened
    values, evaluation proof.
    """
    gkr_proof = proof.gkr
    zerocheck_proof = proof.zerocheck
    jagged_proof = proof.jagged

    # The wire's zerocheck point is SP1's insert-at-front order — the
    # accumulated round challenges reversed, the same z_row order the
    # jagged-eval stage consumes.
    zerocheck_point = zerocheck_proof.msgs.challenge[::-1]
    zerocheck_arrays = _partial_sumcheck_arrays(
        zerocheck_proof.msgs.round_poly, zerocheck_proof.claimed_sum, zerocheck_point
    )
    seg = _field_bytes_many([claim.public_values, proof.commitment, *zerocheck_arrays])

    parts = [_point_bytes(next(seg), _leading_len(claim.public_values))]
    parts.append(next(seg))
    parts.append(_encode_logup_gkr_proof(gkr_proof, max_log_row_count))
    parts.append(
        _partial_sumcheck_bytes(zerocheck_proof.msgs.round_poly, zerocheck_point, seg)
    )

    parts.append(
        _encode_shard_opened_values(
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
    parts.append(
        _encode_evaluation_proof(
            jagged_proof.eval,
            jagged_proof.open,
            component_raw_roots,
            component_commitments,
            row_column_counts,
            max_log_row_count=max_log_row_count,
        )
    )
    return b"".join(parts)
