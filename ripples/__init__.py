"""
Ripples
=======

A scientific computation library that aims to provide a unified, numerically
accurate, and readable framework for multi-discipline numerical work.

The public API is exposed at the top level, so the intended workflow is::

    >>> import ripples
    >>> ripples.<public_function>(...)

enough to reach any feature of the library.


Submodules
----------
Each submodule groups one discipline:

ripples.differentiation
    Numerical differentiation for scalar functions of any variable number.
    Computes gradients, Hessians, and full mixed-partials tensors of any order
    through plain central finite differences, Richardson-extrapolated central
    differences, or the complex-step method. Also exposes a Hessian-vector
    product that never forms the full Hessian.

    Public functions / classes
        - ***nth_numerical_derivative***
        - ***numerical_hessian_vector_product***
        - ***DifferentiationResult***

ripples.optimization
    Numerical optimization for scalar functions of any variable number. A
    single unified entry point (***minimizer***) dispatches to local
    first-order methods (Nesterov, Adam), nonlinear conjugate gradient,
    quasi-Newton methods (BFGS, L-BFGS), Newton-CG, trust-region methods
    (trust-ncg, trust-lanczos), and global optimizers (DIRECT, dual annealing).
    Supports bounds and constraints through an Augmented Lagrangian wrapper.

    Public functions / classes
        - ***minimizer***
        - ***OptimizationResult***

The recommended reading order, file by file, is documented in the `__init__.py`
file of each submodule.


Examples
--------
The following snippets shown for each function are minimal, for the full
parameter list and complete usage guide read their own docstring.


### Differentiation


***nth_numerical_derivative***
    Builds a callable that evaluates the `derivative_order`-th partial
    derivative tensor of a user function at one or many points.

    >>> import numpy as np
    >>> import ripples
    >>>
    >>> # Example 1 - first derivative of a 1-D function at a single point.
    >>> derivative_of_sin = ripples.nth_numerical_derivative(
    ...     np.sin, derivative_order=1,
    ... )
    >>> derivative_of_sin(0.0) # ~ cos(0) = 1.0
    >>>
    >>> # Example 2 - full Hessian of a 2-D function at one point.
    >>> def rosenbrock(x):
    ...     return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    >>>
    >>> hessian_of_rosenbrock = ripples.nth_numerical_derivative(
    ...     rosenbrock, derivative_order=2,
    ... )
    >>> hessian_of_rosenbrock(np.array([1.0, 1.0])) # 2x2 ndarray


***numerical_hessian_vector_product***
    Computes `H(point) @ vector` directly, without ever forming
    `H`, from the gradient of the user function.

    >>> import numpy as np
    >>> import ripples
    >>>
    >>> # Example 1 - HVP of the quadratic 0.5 * x^T A x at any point.
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> def gradient_of_quadratic(x):
    ...     return A @ x
    >>>
    >>> ripples.numerical_hessian_vector_product(
    ...     gradient_of_quadratic,
    ...     point=np.array([1.0, -2.0]),
    ...     vector=np.array([1.0, 1.0]),
    ... )                                   # ~ A @ [1, 1] = [5, 4]
    >>>
    >>> # Example 2 - HVP of a nonlinear function at a generic point.
    >>> def gradient_of_nonlinear(x):
    ...     return np.array([
    ...         np.cos(x[0]) * np.exp(x[1]) + 2.0 * x[0] * x[1],
    ...         np.sin(x[0]) * np.exp(x[1]) + x[0] ** 2,
    ...     ])
    >>>
    >>> ripples.numerical_hessian_vector_product(
    ...     gradient_of_nonlinear,
    ...     point=np.array([0.7, 0.5]),
    ...     vector=np.array([0.3, -0.6]),
    ... )                                   # 1-D ndarray of length 2


### Optimization


***minimizer***
    Minimizes a scalar function of any variable number under a unified
    interface. The same call shape works for every supported method;
    `method='trust_ncg'` is the robust default for smooth problems.

    >>> import numpy as np
    >>> import ripples
    >>>
    >>> # Example 1 - minimize a 2-D quadratic with the default method.
    >>> def quadratic(x):
    ...     return (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
    >>>
    >>> result = ripples.minimizer(
    ...     function=quadratic,
    ...     initial_params=[0.0, 0.0],
    ... )
    >>> result.final_params                 # ~ [3., -1.]
    >>> result.final_cost                   # ~ 0.0
    >>>
    >>> # Example 2 - global optimization on a multi-modal function.
    >>> def rastrigin(x):
    ...     return 10.0 * len(x) + np.sum(
    ...         x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)
    ...     )
    >>>
    >>> result = ripples.minimizer(
    ...     function=rastrigin,
    ...     method='annealing',
    ...     bounds=([-5.12, -5.12], [5.12, 5.12]),
    ... )
    >>> result.final_params                 # ~ [0., 0.]


***OptimizationResult***
    Output class returned by ***minimizer***. Behaves like a read-only
    `numpy.ndarray` of the final parameters and additionally carries the full
    record of the run (cost, evaluation counts, elapsed time, termination
    reason) and every configuration setting.

    >>> import ripples
    >>>
    >>> # Example 1 - inspect the most common attributes.
    >>> result = ripples.minimizer(
    ...     function=lambda x: x[0] ** 2 + x[1] ** 2,
    ...     initial_params=[1.0, 1.0],
    ... )
    >>> result.success
    >>> result.final_params                 # read-only ndarray
    >>> result.final_cost
    >>> result.iteration_number
    >>>
    >>> # Example 2 - summary and conversion helpers.
    >>> print(result)                       # compact summary
    >>> print(result.summary('full'))       # full configuration
    >>> result.to_dict()                    # JSON-friendly dict


Author
------
Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>, unique
author, maintainer, and responsible for overall design, numerical
methods implementation, and API consistency.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author, maintainer, and responsible for overall design, numerical
# methods implementation, and API consistency.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Proper attribution to the original author must be preserved in
# all copies or substantial portions of this software.

__version__ = '0.1.0'
__author__ = 'Álvaro Cátedra Sánchez'
__email__ = 'alvaro.catedra.sanchez@gmail.com'
__license__ = 'Apache-2.0'
__copyright__ = 'Copyright (c) Álvaro Cátedra Sánchez'



# Differentiation
from .differentiation import (
    nth_numerical_derivative,
    numerical_hessian_vector_product,
    DifferentiationResult
)

# Optimization
from .optimization import (
    minimizer,
    OptimizationResult
)

# Test
from ._test import test




_submodules = (
    'differentiation',
    'optimization',
)

_public_functions = (
    'nth_numerical_derivative',
    'numerical_hessian_vector_product',
    'DifferentiationResult',
    'minimizer',
    'OptimizationResult',
)

_dispatcher = (
    'test',
)

_metadata = (
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__copyright__',
)

__all__ = list(_submodules + _public_functions + _dispatcher + _metadata)



def __dir__() -> list[str]:
    """
    Custom directory listing for `ripples`.

    Restricts tab-completion and `dir(ripples)` to the public surface
    declared in `__all__`.

    Returns
    -------
    list of str
        Sorted copy of `__all__`.
    """

    return sorted(__all__)
