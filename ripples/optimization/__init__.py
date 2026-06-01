"""
Optimization - (`ripples.optimization`)
============

Numerical optimization for scalar functions of any variable number. A
single unified entry point (***minimizer***) dispatches to every
algorithm supported by the submodule. The methods cover:

- First-order local methods: Nesterov-accelerated gradient descent, Adam.

- Nonlinear conjugate gradient (Polak-Ribière+).

- Quasi-Newton methods: BFGS, L-BFGS.

- Newton-CG and trust-region methods: trust-ncg, trust-lanczos.

- Global optimizers: DIRECT (deterministic, derivative-free), dua
annealing (stochastic).

- Constrained optimization through an Augmented-Lagrangian wrapper
around any of the above as the inner solver.

Gradients and Hessians are taken either from the user (analytical callables) or
from `ripples.differentiation` (finite-differences callbacks), so the same call
shape works whether or not analytical derivatives are available.

Public API
----------
- ***minimizer***
    The unified entry point. Selects the algorithm through the `method`
    parameter and returns an ***OptimizationResult***.

- ***OptimizationResult***
    The output class. Behaves like a read-only `numpy.ndarray` of the final
    parameters and additionally carries the cost, evaluation counts, elapsed
    time, termination reason, and every configuration setting that produced the
    result. Exposes conversion helpers (`as_array`, `as_float`, `to_list`,
    `to_dict`), equality (`==`, `allclose`), and a `summary`.


Folder contents and recommended reading order
---------------------------------------------
The following list contains all the files in this folder, they are in the
recommended reading order, from the lowest-level utilities up to the unified
entry point:

1. `__init__.py` (this file)
       Lists the all the files of this module and re-exports the public ones
       they appear under `ripples.differentiation`.

2. `_utils.py`
       Foundations shared by every optimizer module. Defines:
       - `FLOAT_EPSILON`: machine epsilon for 64 bit floats.
       - Method-classification constants (`AVAILABLE_METHODS`,
         `TRUST_REGION_METHODS`, `METHODS_THAT_REQUIRE_GRADIENT`,
         `METHODS_THAT_REQUIRE_HESSIAN`, `GLOBAL_OPTIMIZATION_METHODS`,
         `LOCAL_OPTIMIZATION_METHODS`, `METHODS_WITH_INNER_MINIMIZER`,
         `METHODS_THAT_REQUIRE_BOUNDS`).
       - `_initial_check`: pre-algorithm validation of the starting
         point.
       - `_check_termination`: standard convergence check.
       - `_normalize_settings_dict`: callable/array normalization so
         configuration dicts can be represented and compared.
       - `OptimizationResult`: the output class returned by every
         public function of this submodule.

3. `_steepest_descent_optimization.py`
       First-order methods that do not need a line search: Nesterov
       accelerated gradient descent and Adam. It is the simplest family of
       methods and serves as a reference for the more sophisticated algorithms
       that follow.

4. `_line_search_utils.py`
       Strong-Wolfe line search (bracketing + zoom phase) with full
       parameter validation. Used by every method in in
       `_line_search_optimization.py`.

5. `_line_search_optimization.py`
       Methods that use the line search as their step-length calculator:
       nonlinear conjugate gradient (Polak-Ribière+), BFGS, L-BFGS, and the
       Newton-CG variant.

6. `_trust_region_utils.py`
       Trust-region radius update, ratio thresholds, and the step-to-boundary
       solver that places an iterate exactly on the trust-region boundary.

7. `_trust_region_optimization.py`
       Trust-region methods: trust-ncg (Steihaug-Toint truncated CG subproblem
       solver) and trust-lanczos (Lanczos subproblem solver, robust under
       indefinite Hessians).

8. `_global_optimization.py`
       Global optimizers: DIRECT (deterministic, derivative-free) and dual
       annealing (generalised simulated annealing with a periodic local
       refiner). Both need box bounds.

9. `_constrained_optimization.py`
       Augmented-Lagrangian outer loop for problems with equality and
       inequality constraints (and optional bounds). Wraps any of the
       previously described methods as the inner unconstrained solver.

10. `_minimizer.py`
       The unified entry point ***minimizer*** that brings every algorithm
       above together. Validates the user's inputs, fills in defaults, builds
       numerical gradient and Hessian-vector callables when analytical ones are
       not provided, dispatches to the chosen algorithm, and packages the
       outcome into an ***OptimizationResult***.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.
#
# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

from ._minimizer import minimizer
from ._utils import OptimizationResult



__all__ = [
    'minimizer',
    'OptimizationResult',
]



def __dir__() -> list[str]:
    """
    Restrict tab-completion and `dir(ripples.optimization)` to the public
    surface declared in `__all__`.
    """

    return sorted(__all__)
