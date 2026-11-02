"""Nested polynomial subspaces for coarse-to-fine TT coefficient learning."""
import numbers

import jax.numpy as jnp
import numpy as np


def prepare_mode_schedule(sizes, steps, schedule=None):
    """Validate (first update, active counts) stages, ending at full resolution.

    Counts include the constant coefficient and can be one integer for every
    axis or one integer per axis. No TT rank or coefficient array is resized.
    """
    if not isinstance(steps, numbers.Integral) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    sizes = tuple(sizes)
    if schedule is None:
        return ((0, sizes),)
    stages = []
    for start, counts in schedule:
        if not isinstance(start, numbers.Integral) or start < 0 or start >= max(steps, 1):
            raise ValueError('stage starts must be update indices before the end of training')
        counts = (counts,)*len(sizes) if isinstance(counts, numbers.Integral) else tuple(counts)
        if len(counts) != len(sizes) or any(not isinstance(n, numbers.Integral) or not 1 <= n <= full
                                         for n, full in zip(counts, sizes, strict=True)):
            raise ValueError('active mode counts must be integers within the template sizes')
        counts = tuple(map(int, counts))
        if stages and (start <= stages[-1][0] or any(n < old for n, old in zip(counts, stages[-1][1], strict=True))):
            raise ValueError('stage starts must increase and active counts must not decrease')
        stages.append((int(start), counts))
    if not stages or stages[0][0] != 0 or stages[-1][1] != sizes:
        raise ValueError('schedule must start at zero and finish with every template mode active')
    return tuple(stages)


def mode_masks(sizes, counts):
    return tuple(jnp.asarray(np.arange(n) < active) for n, active in zip(sizes, counts, strict=True))


def mask_modes(arrays, masks):
    return tuple(a*mask for a, mask in zip(arrays, masks, strict=True))
