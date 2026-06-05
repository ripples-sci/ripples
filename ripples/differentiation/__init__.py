"""
Differentiation - (`ripples.differentiation`)
===============

Numerical differentiation for scalar functions of any variable number.
This submodule computes:

- The n-th partial-derivative tensor of f at one or many points
  (gradients, Hessians, and full mixed-partials tensors of any order).

- The Hessian-vector product H(x) @ v of f at x along v, without ever
  forming the full Hessian.

Three numerical strategies are available, in increasing order of accuracy
and decreasing order of generality: plain central finite differences,
central finite differences refined by Richardson extrapolation, and the
complex-step method. The choice is exposed as a parameter of the public
functions.


Public API
----------
- ***nth_numerical_derivative***
    Builds a callable that evaluates the `derivative_order`-th partial
    derivative tensor of `function_to_differentiate` at any point(s),
    using the selected numerical strategy.

- ***numerical_hessian_vector_product***
    Computes `H(point) @ vector` for a scalar function f, given its
    gradient, via a 1-D central-difference stencil along the direction
    of `vector`. Never forms the full Hessian.

Every public entry point returns a `DifferentiationResult` (defined
in `_numerical_differentiation_utils.py`), the result is immutable, behaves
like a read-only `numpy.ndarray` of the derivative, and carries the optional
truncation-error estimate together with the full configuration metadata.


Folder contents and recommended reading order
---------------------------------------------
The following list contains all the files in this folder, they are in the
recommended reading order, from the lowest-level utilities up to the unified
entry point:

1. `__init__.py` (this file)
    Lists the all the files of this module and re-exports the public ones
    they appear under `ripples.optimization`.

2. `_numerical_differentiation_utils.py`
    Building blocks. Defines:

    - `FLOAT_EPSILON`: machine epsilon for 64 bit floats.
    - `_solve_gauss_jordan_fraction`: exact-rational linear solver
      used to compute central-difference coefficients without
      floating-point contamination.
    - `_effective_point_number`: enforces the parity rule between
      derivative order and stencil count.
    - `_validate_nth_numerical_derivative_parameters`: input
      validation and normalisation for the main builder.
    - `_resolve_step_size`: translates 'auto', scalar, and tuple
      step-size inputs into a single per-coordinate array.
    - `_PRECOMPUTED_CENTRAL_DIFFERENCES_COEFFICIENTS`: O(1) lookup
      table for the most common stencils.
    - ***DifferentiationResult***: the output class returned by every
      public function of this submodule.

3. `_numerical_differentiation.py`
    The numerical strategies themselves, built on the just explained
    utilities. Defines:

    - `_central_difference_coefficients` and
      `_central_difference_stencil`: stencil construction.
    - `_estimate_truncation_error_constant` and
      `_auto_step_size`: error analysis and step-size selection.
    - `_evaluate_component_central_difference`,
      `_evaluate_component_richardson`,
      `_evaluate_component_complex_step`,
      `_evaluate_component_dispatcher`: per-component evaluators
      for the three numerical strategies and the dispatcher that
      routes to the right one.
    - ***nth_numerical_derivative***: The main entry point of the module.
    - `_hessian_vector_product_central_difference`: directional
      stencil used by the HVP path.
    - ***numerical_hessian_vector_product***: The Hessian-vector product
      entry point.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.
#
# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

from ._numerical_differentiation import (
    nth_numerical_derivative,
    numerical_hessian_vector_product
)
from ._numerical_differentiation_utils import DifferentiationResult



__all__ = [
    'nth_numerical_derivative',
    'numerical_hessian_vector_product',
    'DifferentiationResult'
]



def __dir__() -> list[str]:
    """
    Restrict tab-completion and `dir(ripples.differentiation)` to the
    public surface declared in `__all__`.
    """

    return sorted(__all__)
