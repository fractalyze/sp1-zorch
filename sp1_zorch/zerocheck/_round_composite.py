# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`zorch.sumcheck.round` marker, variant=zerocheck (fractalyze/zorch#394).

Wraps the SP1 jagged-zerocheck round reduce (eq-weighted sum of the summand
value + geq zero-extension correction + Gruen degree-4 assembly). The summand
VALUE is an operand — `constraint_eval` (constraint + the SP1 column/opening
term) stays external, so the marker encodes only the generic jagged-zerocheck
round shape; the SP1-specific column term never enters it. An unclaimed marker
decomposes inline, byte-identical to the plain reduce."""
from __future__ import annotations

import frx.numpy as fnp
from zorch._composite import composite
from zorch.poly.geq import VirtualGeq
from zorch.poly.eq import eq_factor
from zorch.sumcheck.gruen import round_coeffs_from_matrix
from zorch.sumcheck.prover import SUMCHECK_ROUND_MARKER, SUMCHECK_ROUND_MARKER_VERSION

# The PJRT plugin only recognizes the sumcheck round marker for variant
# dense/jagged (LogUp-GKR); the variant=zerocheck recognizer is a later zorch#394
# slice than this producer. Emitting the marker before the plugin knows the
# variant hard-errors on GPU (missing composite attributes / unknown variant),
# while CPU decomposes inline and hides it. Keep the marker off -- run the
# byte-identical inline decomposition -- and flip this to True once the plugin
# lands the recognizer.
_MARK_ZEROCHECK_ROUNDS = False


def _decomp(v0, v2, v4, eq, interp, claim, last, eq_adj, padded_row_adj,
            nr_live, vgeq_threshold, vgeq_geq_coeff, vgeq_eq_coeff, **_attrs):
    """Byte-exact fallback (the emitter replaces this) for a LIVE chip.
    Rebuilds VirtualGeq from its leaves; reproduces `_reduce_and_assemble`
    exactly."""
    ef = last.dtype
    # VirtualGeq fields, in order: (threshold, geq_coefficient, eq_coefficient)
    # — see zorch/poly/geq.py; built in jagged.py as VirtualGeq(nr, one, zero).
    vgeq = VirtualGeq(vgeq_threshold, vgeq_geq_coeff, vgeq_eq_coeff)
    y_raw = (fnp.sum(v0 * eq), fnp.sum(v2 * eq), fnp.sum(v4 * eq))
    # A runtime-empty chip (nr_live == 0 — a cap-class chip the program never
    # exercised) has no live row: the last-live-row index (nr_live+1)//2 - 1 is
    # -1, which corrupts the padding correction and leaks a spurious term for a
    # constraint that is nonzero on an all-zero row (e.g. DivRem). Clamp the
    # index and vanish the reduce to the trivial claim identity — byte-identical
    # to the exact path's static `is_zero_chip` branch (its width-0 buffer
    # reaches _zero_chip_poly by another route), at no extra Gruen dot.
    empty = nr_live == 0
    threshold_half = fnp.maximum((nr_live + 1) // 2 - 1, 0)
    msb_lagrange = eq_adj * eq[threshold_half]

    def corr(y, t_val):
        eq_last = eq_factor(t_val, last)
        vg = vgeq.fix_last_variable(t_val).eval_at(threshold_half)
        val = eq_last * (y * eq_adj - padded_row_adj * vg * msb_lagrange)
        return fnp.where(empty, fnp.zeros((), ef), val)

    zero, two, four = fnp.zeros((), ef), fnp.array(2, ef), fnp.array(4, ef)
    y0, y2, y4 = corr(y_raw[0], zero), corr(y_raw[1], two), corr(y_raw[2], four)
    return round_coeffs_from_matrix(interp, y0, claim, (y2, y4))


def zerocheck_round_poly(vals, eq, interp, claim, last, eq_adj, padded_row_adj,
                         nr_live, vgeq):
    """Emit the variant=zerocheck marker around one LIVE chip's round reduce
    (gated: see `_MARK_ZEROCHECK_ROUNDS` -- default runs the inline decomposition,
    byte-identical, until the zkx plugin ships the variant=zerocheck emitter)."""
    v0, v2, v4 = vals
    operands = (
        v0, v2, v4, eq, interp, claim, last, eq_adj, padded_row_adj,
        nr_live, vgeq.threshold, vgeq.geq_coefficient, vgeq.eq_coefficient,
    )
    if not _MARK_ZEROCHECK_ROUNDS:
        return _decomp(*operands)
    return composite(
        _decomp, *operands,
        name=SUMCHECK_ROUND_MARKER, version=SUMCHECK_ROUND_MARKER_VERSION,
        phase="mid", variant="zerocheck", degree=4, poly_form="coefficient",
    )
