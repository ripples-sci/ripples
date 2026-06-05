"""
Numerical differentiation module for scalars functions.

This module:

- Computes the n-th derivative of f at x, (gradients, Hessians, and full
mixed-partials tensors of any order).

- And a Hessian-vector product of the Hessian of f on a vector v at x.


Three numerical strategies are offered, in increasing order of accuracy
and decreasing order of generality:

- Plain central finite differences, applicable to any sufficiently smooth
function and any derivative order.

- Central finite differences refined by Richardson extrapolation, which
trades extra function evaluations for several additional digits of
accuracy and a Romberg-style per-component error bound.

- The complex-step method, restricted to first derivatives of functions that
admit an analytic extension over the complex plane, which delivers
machine-precision results from a single evaluation.

The choice between the three is exposed as parameters of the public functions
and is documented in detail in each one's docstring.

Contains
--------
_central_difference_coefficients
    Generates the coefficients for an equally spaced grid of points for the
    central finite-difference formula. Hits the precomputed lookup table for
    the most common cases and falls back to an exact rational linear-system
    solve otherwise.

_central_difference_stencil
    Builds the central-difference stencil (offsets paired with weights) for the
    `derivative_order`-th derivative on a `point_number`-point symmetric grid.

_estimate_truncation_error_constant
    Returns the truncation-error constant t_c that multiplies h^acc in the
    leading-order error term of the central-difference formula.

_auto_step_size
    Computes per-coordinate absolute step sizes for a given evaluation point,
    balancing truncation error O(h^acc) against floating-point roundoff
    O(eps / h^n).

_evaluate_component_central_difference
    Computes one component of the derivative tensor using the plain
    central-difference stencil.

_evaluate_component_richardson
    Refines a single derivative tensor component by Richardson extrapolation,
    returning both the refined value and a Romberg-style truncation-error upper
    bound.

_evaluate_component_complex_step
    Computes one first-derivative component at `point` using the complex-step
    method.

_evaluate_component_dispatcher
    Evaluates one component of the derivative tensor at `point`, dispatching to
    the complex-step, the plain central-difference, or the
    Richardson-extrapolation path according to the resolved configuration.

nth_numerical_derivative
    Builds a callable that evaluates the `derivative_order`-th partial
    derivative of `function_to_differentiate` using central
    finite-difference or complex-step differentiation. The main public
    entry point of the module.

_hessian_vector_product_central_difference
    Computes H(point) @ unit_direction using a 1-D central-difference stencil
    applied to `gradient_function` along `unit_direction`.

numerical_hessian_vector_product
    Computes the Hessian-vector product H(`point`) @ `vector` of a scalar
    function f, given its gradient `gradient_function`, without ever forming
    the Hessian using a central-difference stencil.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from fractions import Fraction
from math import factorial
from warnings import warn
from itertools import product
from functools import lru_cache
from collections import Counter
from typing import Union, Callable, Optional, Tuple, Literal

from ._numerical_differentiation_utils import (
    FLOAT_EPSILON, _PRECOMPUTED_CENTRAL_DIFFERENCES_COEFFICIENTS,
    _validate_nth_numerical_derivative_parameters, _resolve_step_size,
    _solve_gauss_jordan_fraction, _effective_point_number, DifferentiationResult
)



# Cache the last 4096 different coefficient calls, it is intended that it is
# sufficient in most cases but with an upper bound to prevent it from growing
@lru_cache(maxsize=4096)
def _central_difference_coefficients(
    derivative_order: int, point_number: int, return_fractions: bool = False
) -> Union[Tuple[float, ...], Tuple[Fraction, ...]]:
    """
    Generates the coefficients for an equally spaced grid of points for the
    central finite-difference formula.

    The evaluation of a function at these points allows to approximate its
    `derivative_order`-th derivative with
    O(h^(`point_number` - `derivative_order` + 1)) accuracy.

    Parameters
    ----------
    derivative_order : int
        Order of the derivative to approximate (1 = first, 2 = second, ...).
        Must be a positive integer.
    point_number : int
        Desired point number to be used in the formula. Must be a positive even
        integer (2, 4, 6, ...) when `derivative_order` is odd and a positive
        odd integer (3, 5, 7, ...) when `derivative_order` is even.
        `point_number`  >=  `derivative_order` + 1 must also be
        satisfied [1].
    return_fractions : bool
        If True, returns a numpy array full of Fractions. If False returns a
        numpy array full of floats. Default is False.

    Returns
    -------
    Tuple of floats or Tuple of Fractions
        - For odd derivative orders:
        the half-stencil coefficients [c_1, ..., c_kmax] for the formula:

            f^{derivative_order}(x) ≈
                (1/h^derivative_order) *
                sum_{k=1}^{kmax}  c_k * [f(x + k * h) - f(x - k * h)]

        where kmax = point_number / 2.

        - For even derivative orders:
        the coefficients [c_0, c_1, ..., c_kmax] (c_0 is the center
        coefficient, corresponding to f(x)) for the formula:

            f^{derivative_order}(x) ≈
                (1/h^derivative_order) *
                [c_0 * f(x) +
                 sum_{k=1}^{kmax}  c_k * (f(x + k * h) + f(x - k * h))]

        where kmax = (point_number - 1) / 2.

        If `return_fractions` is True, returns a Tuple full of Fractions. If
        False returns a Tuple array full of floats.

    Raises
    ------
    ValueError :
        - When `derivative_order` < 1.
        - When `point_number` < `derivative_order` + 1.
        - When `derivative_order` and `point_number` have the same
        parity (both even or both odd).

    Mathematical Description
    ------------------------
    Letting:

    - n = `derivative_order`
    - p = `point_number`

    Given a smooth function f: R -> R at a point x in R, the goal is to find
    coefficients c_1, ..., c_kmax (for odd n) or c_0, c_1, ..., c_kmax
    (for even n), such that a weighted combination of f evaluated at the grid
    points {x, x +- h, x +- 2*h, ...} gives an O(h^{p+1-n})-accurate
    approximation of f^{n}(x).

    where
    - kmax = p / 2 when n is odd
    or
    - kmax = (p - 1) / 2 when n is even


    Starting by the Taylor expansions of f(x +- k * h):

        f(x + k * h) = sum_{j=0}^{inf}  (k * h)^j / j! * f^{j}(x)
                     = f(x) +
                       k * h * f'(x) +
                       ((k * h)^2 / 2!) * f''(x) +
                       ((k * h)^3 / 3!) * f'''(x) +
                       ((k * h)^4 / 4!) * f''''(x) +
                       ...

        f(x - k * h) = sum_{j=0}^{inf}  (-k * h)^j / j! * f^{j}(x)
                     = f(x) -
                       k * h * f'(x) +
                       ((k * h)^2 / 2!) * f''(x) -
                       ((k * h)^3 / 3!) * f'''(x) +
                       ((k * h)^4 / 4!) * f''''(x) -
                       ...

    where:
    - h is the step size

    ---

    ### For odd-order derivatives (n = 1, 3, 5, ..)

    When subtracting the two expansions all even powers cancel and only odd
    powers of h remain:

        f(x + k * h) - f(x - k * h) = 2 * (k * h * f'(x) +
                                           (k^3 * h^3 / 3!) * f'''(x) +
                                           (k^5 * h^5 / 5!) * f^{5}(x) +
                                           ...)

    given this difference, now for some coefficients c_k = [c_1, ..., c_kmax]
    one can write the following weighted grid:

        sum_{k=1}^{kmax}  c_k * [f(x + k h) - f(x - k h)]
            = sum_{k=1}^{kmax}  c_k * 2 * (k * h * f'(x) +
                                           (k^3 * h^3 / 3!) * f'''(x) +
                                           (k^5 * h^5 / 5!) * f^{5}(x) +
                                           ...)

            = 2 * (sum_{k=1}^{kmax}  (c_k * k * h) * f'(x) +
                   sum_{k=1}^{kmax}  (c_k * k^3 * h^3 / 3!) * f'''(x) +
                   sum_{k=1}^{kmax}  (c_k * k^5 * h^5 / 5!) * f^{5}(x) +
                   ...)

            = 2 * (h * (sum_{k=1}^{kmax}  c_k * k) * f'(x) +
                   (h^3 / 3!) * (sum_{k=1}^{kmax}  c_k * k^3) * f'''(x) +
                   (h^5 / 5!) * (sum_{k=1}^{kmax}  c_k * k^5) * f^{5}(x) +
                   ...)

    notice that this grid is a function of the unknowns coefficients c_k to be
    determined and it has all the odd-order derivatives of the function.

    One wants this last expansion to equal exactly h^n * f^{n}(x) (so that when
    dividing the whole expression by h^n, to obtain f^{n}(x)). Therefore,
    force the coefficient for the n-th derivative to be exactly 1, and all
    other coefficients to be zero:

        2 * (1 / n!) * sum_{k=1}^{kmax}  c_k * k^n = 1

    solving for the c_k terms gives:

        sum_{k=1}^{kmax}  c_k * k^n = n!/2

    all the others summand's k-dependent part must be equal to zero (the last
    derived expression is added for clarity):

        sum_{k=1}^{kmax}  c_k * k = 0                         [equation 1]

        sum_{k=1}^{kmax}  c_k * k^3 = 0,                      [equation 2]

        sum_{k=1}^{kmax}  c_k * k^5 = 0,                      [equation 3]

        ...,

        sum_{k=1}^{kmax}  c_k * k^n = n!/2,                   [equation (n+1)/2]

        ...


    So, the number of unknowns kmax = p / 2 must be at least (n+1)/2 (the
    minimum number of equations) for system to be solvable exactly, that is:

        p/2 >= (n+1)/2

        p >= n+1

    which is imposed in the code. The derived equations can be expressed in
    matrix form in the following way:

        [ 1^1      2^1      ...  (p/2)^1     ]   [ c_1         ]   [ 0    ]
        [ 1^3      2^3      ...  (p/2)^3     ]   [ c_2         ]   [ 0    ]
        [ 1^5      2^5      ...  (p/2)^5     ]   [ c_3         ]   [ 0    ]
        [ ...      ...      ...  ...         ] * [ ...         ] = [ ...  ]
        [ 1^n      2^n      ...  (p/2)^n     ]   [ c_{(n+1)/2} ]   [ n!/2 ]
        [ ...      ...      ...  ...         ]   [ ...         ]   [ ...  ]
        [ 1^(p-1)  2^(p-1)  ...  (p/2)^(p-1) ]   [ c_{p/2}     ]   [ 0    ]


    This system allows computing the coefficients multiplied by h^n for a
    central finite-difference odd-order nth-derivative which has p points with
    the form:

        f^{n}(x) ≈ (1/h^n) * sum_{k=1}^{p/2}  c_k *
                                              [f(x + k * h) - f(x - k * h)]

    ---

    ### For even-order derivatives (n = 2, 4, 6, ..)

    When the two expansions are summed all odd_terms cancel and only even
    powers of h remain:

        f(x + k * h) + f(x - k * h) = 2 * (f(x) +
                                           (k * h)^2 / 2!) * f''(x) +
                                           ((k * h)^4 / 4!) * f''''(x) +
                                           ((k * h)^6 / 6!) * f^{6}(x) +
                                           ...)

    Given this sum, now for some coefficients c_k = [c_0, c_1, ..., c_kmax]
    one can write the following weighted grid. Note that c_0 only applies to
    f(x), as f(x) only appears once:

        c_0 * f(x) + sum_{k=1}^{kmax}  c_k * [f(x + k h) + f(x - k h)]

            = c_0 * f(x) +
              sum_{k=1}^{kmax}  c_k * 2 * (f(x) +
                                           (k * h)^2 / 2!) * f''(x) +
                                           ((k * h)^4 / 4!) * f''''(x) +
                                           ((k * h)^6 / 6!) * f^{6}(x) +
                                           ...)

            = (c_0 + 2 * (c_1 + ... + c_kmax) ) * f(x) +
              sum_{k=1}^{kmax}  c_k * 2 * ((k * h)^2 / 2!) * f''(x) +
                                           ((k * h)^4 / 4!) * f''''(x) +
                                           ((k * h)^6 / 6!) * f^{6}(x) +
                                           ...)

            = (c_0 + 2 * (c_1 + ... + c_kmax) ) * f(x) +
               2 * (sum_{k=1}^{kmax}  (c_k * k^2 * h^2 / 2!) * f''(x) +
               sum_{k=1}^{kmax}  (c_k * k^4 * h^4 / 4!) * f''''(x) +
               sum_{k=1}^{kmax}  (c_k * k^6 * h^6 / 6!) * f^{6}(x) +
               ...)

            = (c_0 + 2 * (c_1 + ... + c_kmax) ) * f(x) +
               2 * ((h^2 / 2!) * (sum_{k=1}^{kmax}  c_k * k^2) * f''(x) +
               (h^4 / 4!) * (sum_{k=1}^{kmax}  c_k * k^4) * f''''(x) +
               (h^6 / 6!) * (sum_{k=1}^{kmax}  c_k * k^6) * f^{6}(x) +
               ...)

    notice that this grid is a function of the unknowns coefficients c_k to be
    determined and it has all the even-order derivatives of the function.

    Again, the last expansion hast to be equal exactly to h^n * f^{n}(x)
    (so that when dividing the whole expression by h^n, to obtain f^{n}(x)).
    Therefore, force the coefficient for the n-th derivative to be exactly 1,
    and all other coefficients to be zero:

        2 * (1 / n!) * sum_{k=1}^{kmax}  c_k * k^n = 1

    solving for the c_k terms gives:

        sum_{k=1}^{kmax}  c_k * k^n = n!/2

    all the others summands k-dependent part must be equal to zero (the last
    derived expression is added for clarity), thus independent of h:

        c_0 + 2*(c_1 + ... + c_kmax) = 0                        [equation 0]

        sum_{k=1}^{kmax}  c_k * k^2 = 0                         [equation 1]

        sum_{k=1}^{kmax}  c_k * k^4 = 0,                        [equation 2]

        sum_{k=1}^{kmax}  c_k * k^6 = 0,                        [equation 3]

        ...,

        sum_{k=1}^{kmax}  c_k * k^n = n!/2,                     [equation n/2]

        ...


    So, the number of unknowns kmax = 1 + (p - 1) / 2 (summing 1 because of
    equation 0) must be at least 1 + n/2 (the minimum number of equations) for
    system to be solvable exactly, that is:

        1 + (p-1)/2 >= 1 + n/2

        p-1 >= n

        p >= n + 1

    this condition is imposed in the code and is exactly the same as the
    one derived in the previous odd-order section.

    Now the last equations can be expressed in matrix form in the following way:

        [ 1    2        2        ...  2          ]   [ c_0    ]   [ 0    ]
        [ 0    1^2      2^2      ...  kmax^2     ]   [ c_1    ]   [ 0    ]
        [ 0    1^4      2^4      ...  kmax^4     ]   [ c_2    ]   [ 0    ]
        [ 0    1^6      2^6      ...  kmax^6     ] * [ c_3    ] = [ 0    ]
        [ 0    ...      ...      ...  ...        ]   [ ...    ]   [ ...  ]
        [ 0    1^n      2^n      ...  kmax^n     ]   [ c_{n/2}]   [ n!/2 ]
        [ ...  ...      ...      ...  ...        ]   [ ...    ]   [ ...  ]
        [ 0    1^(p-1)  2^(p-1)  ...  kmax^(p-1) ]   [ c_kmax ]   [ 0    ]

    where
    - kmax = (p - 1) / 2

    this system allows to compute the coefficients for a central
    finite-difference even-order nth-derivative which has p points with the
    form:

        f^{n}(x) ≈ (1/h^n) *
                   (c_0 * f(x) +
                    sum_{k=1}^{(p-1)/2}  c_k * [f(x + k * h) + f(x - k * h)]

    ---

    - Notice how in both cases, the formulation isolates h^n on the right-hand
    side of the system, so the system solves for c_k * h^n. The output of this
    Python function only depends of the grid configuration (the integers k) and
    strictly independent of the step size h.


    - The matrix system is specifically constructed to cancel out all Taylor
    series terms up to the power p - 1, while forcing the n-th derivative term
    to evaluate exactly to 1. Because the stencil is entirely symmetric (or
    antisymmetric), the next non-vanishing term in the Taylor expansion will
    strictly have the degree of at least p + 1, though higher-order
    cancellation may occur in symmetric cases.

    Consequently, evaluating the weighted grid yields the true derivative scaled
    by h^n, plus an error summation strictly proportional to
    h^{p+1} * f^{p+1}(x). When dividing the entire expression by h^n to
    isolate f^{n}(x), the leading truncation error becomes:

        Error = O(h^{p + 1 - n})

    Since p and n must have opposite parity (one is even, the other is odd), the
    subtraction p - n is always an odd number. Therefore, the accuracy order
    p + 1 - n is always a strictly positive even integer (2, 4, 6, ...).
    This property verifies that central finite difference strictly increase in
    accuracy in stepwise jumps of 2.


    - The matrix formulations derived above are generalized Vandermonde matrices
    where the elements take the form A_{j, i} = i^{q_j}. As with all
    Vandermonde-like matrices, they become exponentially ill-conditioned as the
    dimension grows. This code uses exact Fraction arithmetic avoiding this
    issue entirely, producing coefficients that are correct regardless of
    `point_number`.

    References
    ----------
    [1] Randall J. LeVeque (2007).
        **Finite Difference Methods for Ordinary and Partial Differential
        Equations: Steady-State and Time-Dependent Problems**
        SIAM (Society for Industrial and Applied Mathematics).
        Chapter 1.
        https://doi.org/10.1137/1.9780898717839
    """

    # Two caches are used, @lru_cache and the following one, because
    # in cases where the user calls this module using multiple derivative
    # orders, with their different point numbers, @lru_cache only is not
    # sufficient for such common calls, using the following one makes this
    # function much more efficient for the first calls.
    # @lru_cache must still be used in order to cache higher order than 20 or
    # higher point numbers than derivative_order + 11
    cache_check = _PRECOMPUTED_CENTRAL_DIFFERENCES_COEFFICIENTS.get(
        (derivative_order, point_number, return_fractions), None
    )
    if cache_check is not None:
        return cache_check


    # Odd Derivative
    if derivative_order % 2 == 1:
        # floor division (//) is used because it outputs an integer
        half_stencil_size = point_number // 2

        # antisymmetric half-stencil
        Matrix  = [
            [Fraction(0)] * half_stencil_size for _ in range(half_stencil_size)
        ]
        right_hs = [Fraction(0)] * half_stencil_size

        for row in range(half_stencil_size):
            # odd powers: 1, 3, 5, ..., point_number - 1
            power = 2 * row + 1

            for column in range(half_stencil_size):
                # stencil offset k = 1, ..., point_number / 2
                stencil_offset = column + 1
                Matrix[row][column] = Fraction(stencil_offset ** power)

        # Desired odd-derivative row, at (derivative_order+1)/2 in 1-based
        # index (as in Mathematical Description), so in 0-based index it is
        # (derivative_order+1)/2 - 1 which is equivalent to
        # derivative_order/2 - 1/2 = derivative_order // 2
        target_row = derivative_order // 2
        right_hs[target_row] = Fraction(factorial(derivative_order), 2)

    # Even Derivative
    else:
        # add one because of equation 0, c_0
        half_stencil_size_plus_1 = point_number // 2 + 1

        # symmetric half-stencil + centre weight
        Matrix  = [
            [Fraction(0)] * half_stencil_size_plus_1
            for _ in range(half_stencil_size_plus_1)
        ]
        right_hs = [Fraction(0)] * half_stencil_size_plus_1

        # Row 0, constant term must be zero
        Matrix[0][0] = Fraction(1)
        for col in range(1, half_stencil_size_plus_1):
            Matrix[0][col] = Fraction(2)

        for row in range(1, half_stencil_size_plus_1):
            # even powers: 2, 4, 6, ..., point_number - 1
            power = 2 * row

            for col in range(1, half_stencil_size_plus_1):
                # stencil offset k = 1, ..., (point_number - 1) / 2
                stencil_offset = col
                Matrix[row][col] = Fraction(stencil_offset ** power)

        # Desired even-derivative row
        # (Mathematical description is already in 0-based index)
        target_row = derivative_order // 2
        right_hs[target_row] = Fraction(factorial(derivative_order), 2)


    # Solve the system using the Gauss-Jordan method with Fractions
    solution = _solve_gauss_jordan_fraction(Matrix, right_hs)

    if not return_fractions:
        return tuple([float(x) for x in solution])
    else:
        return tuple(solution)



# Cache stencils: each (derivative_order, effective_point_number) pair is built
# at most once per call regardless of how many tensor components need it.
@lru_cache(maxsize=8192)
def _central_difference_stencil(
    derivative_order: int,
    point_number: int
) -> Tuple[Tuple[int, float], ...]:
    """
    Builds the central-difference stencil for the `derivative_order`-th
    derivative.

    Parameters
    ----------
    derivative_order : int
        Differentiation order for which the stencil is generated.
    point_number : int
        Base stencil point count. The effective count may be
        `point_number` + 1 when `derivative_order` and `point_number`
        share the same parity.

    Returns
    -------
    tuple of (int, float)
        Pairs (offset i, coefficient c_i) satisfying

            f^(derivative_order)(x) ≈
                (1/h^derivative_order) * sum_i  c_i * f(x + i*h)

    Mathematical Description
    ------------------------
    The effective point number is chosen to satisfy the parity constraint of
    ***_central_difference_coefficients***: `derivative_order` and the
    effective point must have opposite parity. If `point_number` and
    `derivative_order` already have opposite parity,
    effective_point_number = `point_number`; otherwise
    effective_point_number = `point_number` + 1`.

    ---

    ### c_{-k} = -c_k for odd n and c_{-k} = +c_k for even n.

    Starting by knowing that c_k is the half-stencil weight returned by
    ***_central_difference_coefficients***, in that function the
    finite-difference formula is expressed as a sum over positive k only,
    paired with either the antisymmetric difference
    [f(x + k*h) - f(x - k*h)] when n is odd, or the symmetric sum
    [f(x + k*h) + f(x - k*h)] when n is even.

    One can rewrite these last formulas as a single weighted sum over all
    integer offsets:

        f^{n}(x) ≈ (1/h^n) * sum_{k=-kmax}^{kmax}  c_k * f(x + k*h),

    where the half-stencil c_k is extended to negative indices. So, the exact
    relation between the negative-k and positive-k weights is derived below,
    separately for each parity:



    - For odd n (kmax = p/2, c_0 not present)

    The derived formula for the derivative in
    ***_central_difference_coefficients*** is:

        f^{n}(x) ≈ (1/h^n) * sum_{k=1}^{kmax}  c_k *
                                               [f(x + k*h) - f(x - k*h)]

    Expanding the bracket splits the sum into two:

        f^{n}(x) ≈ (1/h^n) * (sum_{k=1}^{kmax}  c_k * f(x + k*h) -
                              sum_{k=1}^{kmax}  c_k * f(x - k*h))

    Looking now at the second summand of the last expression one can make the
    following change:

        sum_{k=1}^{kmax}  c_k * f(x - k*h)
            = sum_{k=-kmax}^{-1}  c_{-k} * f(x + k*h)

    Substituting this result and grouping the two sums:

        f^{n}(x) ≈ (1/h^n) * (sum_{k=1}^{kmax}  c_k * f(x + k*h) -
                              sum_{k=-kmax}^{-1}  c_{-k} * f(x + k*h))
                 = (1/h^n) * (sum_{k=1}^{kmax}  c_k * f(x + k*h) +
                              sum_{k=-kmax}^{-1}  -c_{-k} * f(x + k*h))

    Now, one can compare term-by-term with the unified form
    sum_{k=-kmax}^{kmax}  c_k * f(x + k*h), the weight at any
    negative offset k must equal -c_{-k}, that is:

        c_{-k} = -c_k    for k = 1, ..., kmax

    This is the Antisymmetric property of odd-order central stencils:
    the weight changes sign under k -> -k.



    - For even n (kmax = (p-1)/2, c_0 present)

    The derived formula for the derivative in
    ***_central_difference_coefficients*** is:

        f^{n}(x) ≈ (1/h^n) * (c_0 * f(x) +
                              sum_{k=1}^{kmax}  c_k * [f(x + k*h) + f(x - k*h)])

    Expanding the bracket splits the sum into two:

        f^{n}(x) ≈ (1/h^n) * (c_0 * f(x) +
                              sum_{k=1}^{kmax}  c_k * f(x + k*h) +
                              sum_{k=1}^{kmax}  c_k * f(x - k*h))

    Looking now at the second summand of the last expression one can make the
    following change:

        sum_{k=1}^{kmax}  c_k * f(x - k*h)
            = sum_{k=-kmax}^{-1}  c_{-k} * f(x + k*h)

    Substituting this result and grouping the two sums:

        f^{n}(x) ≈ (1/h^n) * (c_0 * f(x) +
                              sum_{k=1}^{kmax}  c_k * f(x + k*h) +
                              sum_{k=-kmax}^{-1}  c_{-k} * f(x + k*h))

    Now, one can compare term-by-term with the unified form
    sum_{k=-kmax}^{kmax}  c_k * f(x + k*h), the weight at any
    negative offset k must equal c_{-k}, that is:

        c_{-k} = +c_k    for k = 1, ..., kmax

    This is the symmetric property of even-order central stencils:
    the weight is invariant under k -> -k.
    """

    effective_point_number = _effective_point_number(
        derivative_order, point_number
    )

    weights = _central_difference_coefficients(
        derivative_order, effective_point_number, return_fractions=False
    )

    if derivative_order % 2 == 1:
        # Antisymmetric half-stencil:
        # c_k * [f(x+k*h) - f(x-k*h)]  ->
        # [(+k, +c_k), (-k, -c_k)]
        stencil = []
        for index, weight in enumerate(weights):
            stencil_offset = index + 1 # k
            stencil.append((+stencil_offset, float(+weight)))
            stencil.append((-stencil_offset, float(-weight)))
    else:
        # Symmetric stencil:
        # c_0*f(x) + c_k*(f(x+k*h)+f(x-k*h))  ->
        # [(0,c_0),(+k,+c_k),(-k,+c_k)]
        stencil = [(0, float(+weights[0]))]
        for index, weight in enumerate(weights[1:]):
            stencil_offset = index + 1 # k
            stencil.append((+stencil_offset, float(+weight)))
            stencil.append((-stencil_offset, float(+weight)))

    return tuple(stencil)



@lru_cache(maxsize=8192)
def _estimate_truncation_error_constant(
    derivative_order: int,
    point_number: int,
) -> float:
    """
    Returns the truncation-error constant t_c.

    The truncation error is (see ***_auto_step_size***):

        |E_t(h)| ≈ t_c * h^acc * |f^{n+acc}(x)| * (1 + O(h^2))

    Parameters
    ----------
    derivative_order : int
        Order n of the derivative the stencil approximates. Must be a
        positive integer.
    point_number : int
        Base stencil point count.

    Returns
    -------
    float
        Strictly positive leading-order truncation-error constant t_c,
        rounded once at the very end from an exact rational computation
        (machine precision regardless of `point_number`).

    Mathematical Description
    ------------------------
    Letting:

    - n = `derivative_order`

    - p = `point_number` or `point_number` + 1, depending on parity:
        - If `point_number` and n have the same parity
        then p = `point_number` + 1
        - If `point_number` and n have opposite parity
        then p = `point_number`

        (it is assumed that the provided arguments satisfy
        `point_number` >= max(2, n))

    - acc = p - n + 1, the formal accuracy order of the stencil

    - c_k, the half-stencil weights returned by
      ***_central_difference_coefficients***. They define the full-stencil
      coefficients c_k at integer offset k by:

        - when n is odd then c_0 is not present so the coefficients are:
        c_k with k = -kmax, ...,-1, 1, ..., kmax; kmax = p/2.

        - when n is even the coefficients are
        c_k with k = -kmax, ..., kmax; kmax = (p-1)/2

    ---

    In order to understand where the leading error comes from, consider
    a central finite-difference scheme:

        f^{n}(x) ≈ (1/h^n) * sum_{k=-kmax}^{kmax}  c_k * f(x + k*h)

    where, f(x + k*h) has a Taylor expansion of:

        f(x + k*h) = sum_{j=0}^{inf}  (k*h)^j / j! * f^{j}(x)

    substituting this last expansion into the scheme and later swapping
    the two sums:

        f^{n}(x) ≈ (1/h^n) * sum_{k=-kmax}^{kmax}  c_k * f(x + k*h)
            = (1/h^n) * (sum_{k=-kmax}^{kmax}  c_k *
                         (sum_{j=0}^{inf}  (k*h)^j/j! * f^{j}(x)))
            = (1/h^n) * sum_{j=0}^{inf}  (h^j / j!) * f^{j}(x) *
                                         (sum_{k=-kmax}^{kmax}  c_k * k^j)


    Now, recalling the Vandermonde system solved in
    ***_central_difference_coefficients***, the coefficients (once extended
    to the full stencil through the symmetry/antisymmetry relations
    c_{-k} = +-c_k derived in ***_central_difference_stencil***) satisfy the
    exact conditions:

        sum_k  c_k * k^j = n!    if j = n
        sum_k  c_k * k^j = 0     otherwise, for 0 <= j <= p-1

    These make every Taylor term vanish except for j = n, so dividing the
    scheme by h^n recovers f^{n}(x) exactly up to higher-order remainders.
    Moreover, the central symmetry (or antisymmetry) of the stencil makes
    every term of parity opposite to n vanish identically for every j (not
    just j <= p-1). The first remainder that is not forced to zero is
    therefore at j = p + 1, the smallest j > p - 1 that shares the parity
    of n.

    Substituting j = p + 1 back and dividing by h^n gives the truncation error:

        E_t(h) = (h^acc / (p+1)!) * sum_k  c_k * k^(p+1) * f^{p+1}(x) +
                 O(h^(acc+2))

    ---

    Because, as enforced, p always has opposite parity to n, p + 1 always has
    the same parity as n. In both parity cases the matching parity of
    k^(p+1) and of the weights c_k makes their product symmetric in k, which
    gives the same result for sum_{k=-kmax}^{kmax}  c_k * k^(p+1):

    - When n is odd then p+1 is odd, that is, (-k)^(p+1) = -k^(p+1) and
    c_{-k} = -c_k [this last equation is derived in
    _central_difference_stencil]:

        sum_{k=-kmax}^{kmax}  c_k * k^(p+1)

            = sum_{k=1}^{kmax}  c_k * k^(p+1) +
              sum_{k=-kmax}^{-1}  c_k * k^(p+1)

            = sum_{k=1}^{kmax}  c_k * k^(p+1) +
              sum_{k=1}^{kmax}  (-c_k) * (-k)^(p+1)

            = sum_{k=1}^{kmax}  c_k * k^(p+1) +
              sum_{k=1}^{kmax}  c_k * k^(p+1)

            = 2 * sum_{k=1}^{kmax}  c_k * k^(p+1)

    - When n is even then p+1 is even, that is, (-k)^(p+1) = +k^(p+1) and
    c_{-k} = +c_k [this last equation is derived in
    _central_difference_stencil]:

        sum_{k=-kmax}^{kmax}  c_k * k^(p+1)

            = c_0 * 0^(p+1) + sum_{k=1}^{kmax}  c_k * k^(p+1) +
              sum_{k=-kmax}^{-1}  c_k * k^(p+1)

            = sum_{k=1}^{kmax}  c_k * k^(p+1) +
              sum_{k=1}^{kmax}  c_k * (-k)^(p+1)

            = sum_{k=1}^{kmax}  c_k * k^(p+1) + sum_{k=1}^{kmax}  c_k * k^(p+1)

            = 2 * sum_{k=1}^{kmax}  c_k * k^(p+1)


    In both cases:

        sum_{k=-kmax}^{kmax}  c_k * k^(p+1)
            = 2 * sum_{k=1}^{kmax}  c_k * k^(p+1)

    so the leading-error constant has the same expression regardless of the
    parity of n.

    ---

    Substituting the last result into the truncation error expression
    and taking absolute value:

        |E_t(h)| ≈ (2 * |sum_{k=1}^{kmax}  c_k * k^(p+1)| / (p+1)!) * h^acc *
                   |f^{p+1}(x)| * (1 + O(h^2))

    which has a strictly positive constant part:

        t_c = 2 * |sum_{k=1}^{kmax}  c_k * k^(p+1)| / (p+1)! > 0

    This is exactly what this function computes and returns.

    Notes
    -----
    Looking at the output of this function
    (2 * |sum_{k=1}^{kmax}  c_k * k^(p+1)| / (p+1)!) one notices that the
    numerator grows exponentially with p and the denominator grows factorially
    with p, so for `point_number` >= ~14 a numerical round-off error is
    structurally unavoidable. For this reason the result is computed in exact
    Fraction arithmetic using the exact c_k from
    ***_central_difference_coefficients***, and rounded once at the very end
    after dividing by (p+1)!.

    References
    ----------
    [1] Randall J. LeVeque (2007).
        **Finite Difference Methods for Ordinary and Partial Differential
        Equations: Steady-State and Time-Dependent Problems**
        SIAM (Society for Industrial and Applied Mathematics).
        Chapter 1.
        https://doi.org/10.1137/1.9780898717839
    """

    effective_point_number = _effective_point_number(
        derivative_order, point_number
    )

    leading_error_order = effective_point_number + 1

    weights = _central_difference_coefficients(
        derivative_order, effective_point_number, return_fractions=True
    )

    # when derivative_order is even, c_0 is ignored
    half_stencil_weights = (
        weights if derivative_order % 2 == 1 else weights[1:]
    )

    half_stencil_sum = Fraction(0)
    for index, coefficient in enumerate(half_stencil_weights):
        # sum_{k=1}^{kmax}  c_k * k^(p+1)
        offset = index + 1 # k

        # coefficient is a Fraction, and Fraction * int is a Fraction
        half_stencil_sum += (
            coefficient * offset ** leading_error_order
        )

    # half_stencil_sum is a Fraction, and Fraction / int is a Fraction
    return float(abs(2 * half_stencil_sum) / factorial(leading_error_order))



@lru_cache(maxsize=8192)
def _auto_step_size(
    variable_number: int,
    indexes_counter_tuple: Tuple[Tuple[int, int], ...],
    point_number: int,
) -> np.ndarray:
    """
    Computes per-coordinate absolute step sizes for a given evaluation point.

    Parameters
    ----------
    variable_number : int
        Number of dimensions of the user-provided function.
    indexes_counter_tuple : tuple of (int, int)
        A tuple that encodes, for each active coordinate, the pair
        (coordinate index, per-coordinate differentiation order). For example:

            ((0, 1), (1, 2))   ->   d^{3}f(x) / dx_0 dx_1^{2}.
    point_number : int
        Base stencil point count. The formal accuracy order achieved along
        an active coordinate of per-coordinate derivative order m is
        acc = effective_point_number - m + 1, where the effective point
        number equals `point_number` when it has opposite parity to m,
        otherwise `point_number` + 1.

    Returns
    -------
    np.ndarray
        Shape (variable_number,). Read-only. Entries at active coordinates hold
        the per-coordinate optimal step h_k; entries for inactive coordinates
        are np.inf. Entries for inactive coordinates are never accessed
        during computation.

    Mathematical Description
    ------------------------
    The derivation below is presented for a single coordinate; the
    implementation simply applies the same formula independently along each
    active coordinate using that coordinate's per-coordinate differentiation
    order from `indexes_counter_tuple`.

    Letting:

    - n = per-coordinate derivative order along the coordinate being
      considered.

    - p = `point_number` or `point_number` + 1, depending on parity:
        - If `point_number` and n have the same parity
        then p = `point_number` + 1
        - If `point_number` and n have opposite parity
        then p = `point_number`

        (it is assumed that the provided arguments satisfy
        `point_number` >= max(2, n))

    - acc = p - n + 1

    ---

    ### Optimal step size

    The computation of the optimal step size comes from the minimization of
    the total numerical error E(h), which in a finite-difference
    approximation is the sum of the truncation error (from the Taylor-series
    approximation) and the roundoff error (from floating-point precision
    limits).

    For a central-difference scheme of accuracy O(h^acc) the n-th derivative
    has:

    - Truncation error E_t ≈ t_c * |f^{n+acc}(x)| * h^acc
    - Roundoff error E_r ≈ eps * coefficient_sum * |f(x)| / h^n

    where:

    - t_c is the leading-order truncation-error constant computed by
    ***_estimate_truncation_error_constant***. It depends only on the stencil
    (n and p), not on h or on f.
    - eps is the machine epsilon (~2.2e-16 for a 64-bit float).
    - coefficient_sum = sum_{k=-kmax}^{kmax}  |c_k|, the sum of the absolute
    values of every stencil weight returned by
    ***_central_difference_stencil***.

    For the purpose of minimizing E(h) with respect to h, every other
    quantity is treated as a constant in h, so the total-error model is:

        E(h) = t_c * |f^{n+acc}(x)| * h^acc +
               eps * coefficient_sum * |f(x)| / h^n

    To minimize it, take the derivative with respect to h and set it to zero:

        dE/dh = acc * t_c * |f^{n+acc}(x)| * h^(acc-1) -
                n * eps * coefficient_sum * |f(x)| * h^(-(n+1)) = 0

    Rearranging:

        acc * t_c * |f^{n+acc}(x)| * h^(acc-1)
            = n * eps * coefficient_sum * |f(x)| * h^(-(n+1))

    multiplying both sides by h^(n+1):

        acc * t_c * |f^{n+acc}(x)| * h^(n+acc)
            = n * eps * coefficient_sum * |f(x)|

    solving for h^(n+acc) gives:

        h^(n+acc)
            = (n * coefficient_sum * |f(x)|) /
              (acc * t_c * |f^{n+acc}(x)|) * eps

    which can be regrouped as:

        h_optimal^(n+acc)
            = (n * coefficient_sum * eps / (acc * t_c)) *
              (|f(x)| / |f^{n+acc}(x)|)

    Taking the (n+acc)-th root finally yields the optimal step:

        h_optimal
            = ((n * coefficient_sum * eps / (acc * t_c)) *
               (|f(x)| / |f^{n+acc}(x)|)) ^ (1 / (n+acc))


    Looking at the just derived formula:

    - The stencil factor:

        (n * coefficient_sum * eps / (acc * t_c)) ^ (1 / (n+acc))

    depends only on the stencil and on machine precision; this is what this
    function returns.

    - The function dependent factor:

        (|f(x)| / |f^{n+acc}(x)|) ^ (1 / (n+acc))

    is not evaluated here, it is estimated in
    ***_evaluate_component_dispatcher***.

    References
    ----------
    [1] William H. Press, Saul A. Teukolsky, William T. Vetterling &
        Brian P. Flannery (2007).
        **Numerical Recipes: The Art of Scientific Computing, 3rd edition.**
        Cambridge University Press.
        Section 5.7.
        ISBN: 978-0-521-88068-8.
    """

    # Filled with inf to later indicate if there is one variable that the
    # function is not derived with respect to
    output_step_size = np.full(variable_number, np.inf)

    for (current_variable_index,
            current_variable_derivative_order) in indexes_counter_tuple:

        effective_point_number = _effective_point_number(
            current_variable_derivative_order, point_number
        )
        local_accuracy = (
            effective_point_number - current_variable_derivative_order + 1
        )


        # t_c the truncation error constant. The magnitude of the derivative
        # is computed in _evaluate_component_dispatcher in order to
        # preserve the cache.
        truncation_error_constant = _estimate_truncation_error_constant(
            current_variable_derivative_order, point_number
        )


        # roundoff amplification estimation, As the following function is
        # cached, it is only computed once, so it has almost no
        # computational cost
        stencil = _central_difference_stencil(
            current_variable_derivative_order, point_number
        )
        coefficient_sum = sum(
            abs(coefficient) for _, coefficient in stencil
        )


        step_size_scalar = (
            current_variable_derivative_order * coefficient_sum /
            (local_accuracy * truncation_error_constant)
        )
        exponent = (
            1.0 / (current_variable_derivative_order + local_accuracy)
        )
        current_variable_step_size = (
            (step_size_scalar * FLOAT_EPSILON) ** exponent
        )


        output_step_size[current_variable_index] = (
            current_variable_step_size
        )

    # Prevent the caller from modifying the output in place, it would change
    # @lru_cache
    output_step_size.setflags(write=False)

    return output_step_size



def _evaluate_component_central_difference(
    user_function: Callable[[np.ndarray], float],
    point: np.ndarray,
    indexes_counter: Counter[int],
    point_number: int,
    step_size_array: np.ndarray,
) -> float:
    """
    Computes one component of the derivative tensor at `point`.

    Parameters
    ----------
    user_function : Callable[[np.ndarray], float]
        Function to differentiate; must accept a 1-D array of shape (D,)
        and return a scalar.
    point : np.ndarray
        Evaluation point, shape (D,).
    indexes_counter : Counter of int
        A Counter that encodes each active coordinate index to its
        per-coordinate differentiation order. For example:

            Counter({0: 1})          ->  df(x) / dx_0
            Counter({0: 1, 1: 1})    ->  d^{2}f(x) / dx_0 dx_1
            Counter({1: 2})          ->  d^{2}f(x) / dx_1^{2}
            Counter({0: 1, 1: 2})    ->  d^{3}f(x) / dx_0 dx_1^{2}

        It is assumed that the coordinate indexes are valid.
    point_number : int
        Base stencil point count.
    step_size_array : np.ndarray
        Shape (D,). Step size h_k for each active coordinate k. np.inf
        for inactive coordinates (which are never read during computation).

    Returns
    -------
    float
        Numerical approximation of the requested partial derivative.

    Mathematical Description
    ------------------------
    A multi-variable partial derivative can be built by combining independent
    1-D derivative rules along each coordinate direction [3], for example:

        d^{x_0x_1}f = d^{x_0}(d^{x_1}f(x))

    meaning that the second cross-derivative with 2 point in each partial
    derivative is [1][2]:

        d^{x_0}(d^{x_1}f) = d^{x_0}(f_x_1)

            ≈ (0.5*f_x_1(x_0+h_0,x_1) - 0.5*f_x_1(x_0-h_0,x_1)) / h_0

            = (0.5*((0.5*f(x_0+h_0,x_1+h_1) - 0.5*f(x_0+h_0,x_1-h_1)) / h_1) -
               0.5*((0.5*f(x_0-h_0,x_1+h_1) - 0.5*f(x_0-h_0,x_1-h_1)) / h_1)) /
               h_0

            = (0.25*f(x_0+h_0,x_1+h_1) - 0.25*f(x_0+h_0,x_1-h_1) -
               0.25*f(x_0-h_0,x_1+h_1) + 0.25*f(x_0-h_0,x_1-h_1)) / (h_0*h_1)


    In 1-D, a central-difference stencil approximates a derivative as a
    weighted sum of function evaluations at points shifted by multiples of
    h:

        f'(x) ≈ (1/h) * sum_{i}  c_i * f(x + i*h)

    For a mixed partial, this is done simultaneously along every involved
    coordinate. Each combination of shifts from every coordinate's stencil
    defines one evaluation point, and its contribution to the total is the
    product of all per-coordinate coefficients times the function value
    there.

    Summing all combinations and dividing by the multiplication:

        ∏_{k}  h_k^(m_k)

    where m_k is the order of differentiation along coordinate k (because
    each coordinate may carry a different step size). This gives the mixed
    partial derivatives, this is the Cartesian-product composition of 1-D
    stencils.

    References
    ----------
    [1] William H. Press, Saul A. Teukolsky, William T. Vetterling &
        Brian P. Flannery (2007).
        **Numerical Recipes: The Art of Scientific Computing, 3rd edition.**
        Cambridge University Press.
        Section 5.7.
        ISBN: 978-0-521-88068-8.

    [2] Wikipedia contributors.
        **Finite Differences.**
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

    [3] Randall J. LeVeque (2007).
        **Finite Difference Methods for Ordinary and Partial Differential
        Equations: Steady-State and Time-Dependent Problems**
        SIAM (Society for Industrial and Applied Mathematics).
        Chapter 1.
        https://doi.org/10.1137/1.9780898717839
    """

    # Each unique coordinate only needs one stencil, of order equal to
    # how many times that coordinate appears in indexes_tuple.
    #
    # Example: indexes_counter = {0: 1, 1: 2} meaning that the a first-order
    # stencil along x_0 and second-order stencil along x_1 are computed
    unique_active_coords = list(indexes_counter.keys())


    # Generate a 1D stencil for each coordinate based on its derivative
    # order. Each stencil is a list of (offset, coefficient) pairs (i, c_i)
    # used to approximate a derivative such that:
    #
    # f^(m)(x) ≈ (1/h^m) * sum_{(i, c_i)}  c_i * f(x + i*h)
    #
    # where m is the derivative order for that coordinate. Offsets can be
    # negative, zero, or positive, covering both sides of x symmetrically.
    #
    # Example with point_number=2, idx_tuple=(0, 1, 1):
    #
    # - x_0: (m=1, m is odd so the stencil is antisymmetric):
    #        [(+1, +0.5), (-1, -0.5)], meaning:  0.5*f(x+h) - 0.5*f(x-h)
    #
    # - x_1: (m=2, m is even so the stencil is symmetric with centre point):
    #        [(0, -2.0), (+1, +1.0), (-1, +1.0)], meaning
    #        -2*f(x) + f(x+h) + f(x-h)
    stencils = [
        _central_difference_stencil(indexes_counter[coord], point_number)
        for coord in unique_active_coords
    ]


    # Taking the Cartesian product of all per-coordinate stencils generates
    # every combination of shifts across all active coordinates. Each
    # combination defines:
    #
    # - one evaluation point + perturbation     (a shifted copy of point)
    # - one combined weight     (product of all per-coordinate coefficients)
    #
    # For example, the idx_tuple = (0, 1) with point_number=2 means:
    #
    # x_0 stencil: [(+1, +0.5), (-1, -0.5)]
    # x_1 stencil: [(+1, +0.5), (-1, -0.5)]
    #
    # which Cartesian product gives 4 combinations:
    #
    # 1.: ((+1,+0.5), (+1,+0.5))  ->  point + [ h,  h], weight = +0.25
    # 2.: ((+1,+0.5), (-1,-0.5))  ->  point + [ h, -h], weight = -0.25
    # 3.: ((-1,-0.5), (+1,+0.5))  ->  point + [-h,  h], weight = -0.25
    # 4.: ((-1,-0.5), (-1,-0.5))  ->  point + [-h, -h], weight = +0.25
    #
    # In this case the total variable will be:
    #
    # Total = (+0.25*f(x+h,y+h) - 0.25*f(x+h,y-h)
    #         - 0.25*f(x-h,y+h) + 0.25*f(x-h,y-h)) / h^2
    #
    # which is exactly the standard mixed second-derivative formula [1].
    total = 0.0
    perturbation = np.zeros(point.shape, dtype=float)
    for combo in product(*stencils):
        weight = 1.0
        # Fill the already-allocated perturbation array with zeros
        perturbation.fill(0)

        for (offset, coeff), coord in zip(
            combo, unique_active_coords, strict=True
        ):
            # Accumulate the scalar weight and the displacement vector for
            # this particular combination of per-coordinate shifts
            weight *= coeff

            perturbation[coord] += offset * step_size_array[coord]

        # In case the user function stores the input-array, at each iteration a
        # new evaluated_point must be created, if not the user storage would be
        # misleading
        evaluated_point = point + perturbation
        total += weight * float(user_function(evaluated_point))


    # Each per-coordinate stencil contributes one factor of 1/h^m_k. The
    # combined formula therefore approximates h^n * f^(n)(x), so dividing
    # by h^n (where n = derivative_order = len(idx_tuple)) recovers the
    # derivative itself.
    denominator = 1.0
    for coord, order in indexes_counter.items():
        denominator *= step_size_array[coord] ** order

    return total / denominator



def _evaluate_component_richardson(
    user_function: Callable[[np.ndarray], float],
    point: np.ndarray,
    indexes_counter: Counter[int],
    point_number: int,
    base_step_size_array: np.ndarray,
    step_size_is_auto: bool,
    accuracy: int,
    maximum_richardson_equations: int,
) -> Tuple[float, float]:
    """
    Applies Richardson extrapolation to one derivative tensor component at
    `point`.

    Parameters
    ----------
    user_function : Callable[[np.ndarray], float]
        Function to differentiate.
    point : np.ndarray
        Evaluation point, shape (D,).
    indexes_counter : Counter[int]
        A Counter that encodes each active coordinate index to its
        per-coordinate differentiation order. For example:

            Counter({0: 1})          ->  df(x) / dx_0
            Counter({0: 1, 1: 1})    ->  d^{2}f(x) / dx_0 dx_1
            Counter({1: 2})          ->  d^{2}f(x) / dx_1^{2}
            Counter({0: 1, 1: 2})    ->  d^{3}f(x) / dx_0 dx_1^{2}

        It is assumed that the coordinate indexes are valid.
    point_number : int
        Base stencil point count.
    base_step_size_array : np.ndarray
        Shape (D,). Per-coordinate step sizes h_k for each active coordinate k
        which will serve as the base in order to perform the Richardson
        Extrapolation (see Mathematical Description).

        np.inf for inactive coordinates (which are never read during
        computation).
    step_size_is_auto : bool
        True if the user-specified step size was set to 'auto'. False otherwise.
    accuracy : int
        Base stencil accuracy order acc = `point_number` -
        `derivative_order` + 1.
    maximum_richardson_equations : int
        Maximum number of step-size equations, up to
        `maximum_richardson_equations` - 1 elimination passes are performed.

    Returns
    -------
    Tuple[float, float]
        First, Richardson-extrapolated estimate of the derivative component with
        accuracy O(h^{acc + 2*k_max}), where
        acc = `point_number` - `derivative_order` + 1
        k_max = `maximum_richardson_equations` - 1.

        Second, the estimated residual truncation error of the returned
        value. It is nan when no Richardson estimate could be formed
        (for example when the base stencil is non-finite).

    Mathematical Description
    ------------------------
    Letting:

    - n = `derivative_order`

    - p = `point_number` or `point_number` + 1, depending on parity:
        -If `point_number` and `derivative_order` has the same parity
        then p = `point_number` + 1
        -If `point_number` and `derivative_order` has opposite parity
        then p = `point_number`

        (it is assumed that provided arguments satisfy:
        `point_number >= max(2, derivative_order+1)`)

    - acc = p - n + 1

    ---

    ### Richardson Extrapolation

    Richardson Extrapolation allows to recursively eliminate the greatest
    truncation error made by the Taylor Series approximation used.

    As the approximated derivative is computed using a central-difference
    stencil of accuracy O(h^acc), the Taylor Series derivation guarantees
    that consecutive truncation error terms are spaced two powers of h apart,
    so the full expansion is:

        F(h) = D + C_0 * h^acc + C_1 * h^(acc+2) + C_2 * h^(acc+4) + ...

    where:
    - F is the truncated value given by the central-difference stencil
    - D is the exact derivative value
    - C_i are an unknown constants depending on higher derivatives

    This truncation-error structure is inherent to the Taylor Series
    derivation of central difference, independent of derivative order.


    When the extrapolation is performed one starts by computing the value of the
    truncated value given by the finite-difference stencil at:

    - k_max := `maximum_richardson_equations` - 1

    successively halved step sizes, that is:

        for k = 0, 1, ..., k_max; evaluate F(h/2^k):

            k = 0  ->  F(h), the approximation using step size h
            k = 1  ->  F(h/2), the approximation using step size h/2
            ...
            k = k_max  ->  F(h/2^k_max), the approximation using step size
                h/2^k_max

    from now on, truncated values given by then finite-difference stencil will
    be expressed by:

        F_{j,k} = D + C_j * (h/2^k)^(acc+2*j) +
                  C_{j+1} * (h/2^k)^(acc+2*(j+1)) +
                  C_{j+2} * (h/2^k)^(acc+2*(j+2)) + ...
                = D + O((h/2^k)^(acc+2*j))

    meaning that:
    - Its row index j is its discarded part proportional to (h/2^k)^(acc+2*j) or
    a higher, so O((h/2^k)^(acc+2*j)).
    - Its column index k indicates that it comes from F_{0,k} which was computed
    using a step size h/2^k.


    The procedure of the recursive extrapolation is done in k_max parts,
    iterating k at each one so the accuracy is increased by eliminating the
    leading truncation error term C_j * (h/2^k)^(acc+2*j),

    to illustrate the process, consider the following Extrapolation table with
    each row representing a different level of accuracy (j; acc+2*j) and
    each column representing the step size used to obtain that column's first
    truncated value (k; h/2^k) [2]:

               k=0      k=1      k=2      k=3      ...  k=k_max
        j=0    F_{0,0}  F_{0,1}  F_{0,2}  F_{0,3}  ...  F_{0,k_max}
        j=1             F_{1,1}  F_{1,2}  F_{1,3}  ...  F_{1,k_max}
        j=2                      F_{2,2}  F_{2,3}  ...  F_{2,k_max}
        j=3                               F_{3,3}  ...  F_{3,k_max}
        ...                                        ...  ...
        k_max                                           F_{k_max, k_max}

    it is built row by row starting from the successively halved step truncated
    values which all are at j=0.
    Now the recursive formula described below is used to go down row by row:

    - Row one, F_{1,1} is computed from F_{0,0} and F_{0,1}, F_{1,2} is
    computed from F_{0,1} and F_{0,2}, and so on until reaching F_{1,k_max}.

    - Row two, F_{2,2} is computed from F_{1,1} and F_{1,2}, F_{2,3} is
    computed from F_{1,2} and F_{1,3}, and so on until reaching F_{2,k_max}.

    - This is repeated until reaching the row j=k_max, where only one entry is
    left, F_{k_max, k_max}, which is the output and the returned value as it
    has the desired accuracy acc+2*k_max.

    ---

    The recursive formula used to eliminate the leading error comes from
    the truncated values obtained by the central-difference stencil:

        F_{0,0}, the truncated derivative using step size h
        F_{0,1}, the truncated derivative using step size h/2
        ...
        F_{0,k}, the truncated derivative using step size h/2^k
        ...
        F_{0,k_max}, the truncated derivative using step size h/2^k_max

    which Taylor expansions are:

        F_{0,0} = D + C_0 * h^acc + O(h^(acc+2))
        F_{0,1} = D + C_0 * h^acc / 2^acc + O(h^(acc+2))
        ...
        F_{0,k} = D + C_0 * h^acc / 2^(k*acc) + O(h^(acc+2))
        ...
        F_{0,k_max} = D + C_0 * h^acc / 2^(k_max*acc) + O(h^(acc+2))

    now these expansions when at the j-th row change to:

        F_{j,0} = D + C_j * h^(acc+2*j) + O(h^(acc+2*(j+1)))
        F_{j,1} = D + C_j * h^(acc+2*j) / 2^(acc+2*j) + O(h^(acc+2*(j+1)))
        ...
        F_{j,k} = D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j)) + O(h^(acc+2*(j+1)))
        ...
        F_{j,k_max} = D + C_j * h^(acc+2*j) / (2^(k_max*(acc+2*j))) +
                      O(h^(acc+2*(j+1)))



    Forming now a linear combination that extrapolates for two truncated values

    - The first is, for an arbitrary pair of j and k, F_{j,k} with k satisfying:

        1 <= k <= k_max

    - The second is its left-adjacent (in the table), F_{j,k-1},

    then with coefficients alpha and beta, one has:

        alpha * F_{j,k} + beta * F_{j,k-1}
            = alpha * (D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j))) +
              beta * (D + C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j)))

    from the right hand side, one can extract that

    - the exact derivative term is (alpha + beta) * D
    - the error term is C_j * h^(acc+2*j) * (alpha / 2^(k*(acc+2*j)) +
                                             beta / 2^((k-1)*(acc+2*j)))

    as the goal is to gain accuracy:

    - Set exact derivative to be preserverd: alpha + beta = 1
    - Set error term to be eliminated:

        alpha / 2^(k*(acc+2*j)) + beta / 2^((k-1)*(acc+2*j)) = 0
        alpha / 2^(k*(acc+2*j)) + beta / 2^(k*(acc+2*j)-(acc+2*j)) = 0
        alpha / 2^(k*(acc+2*j)) + beta * 2^(acc+2*j) / 2^(k*(acc+2*j)) = 0
        (alpha + beta * 2^(acc+2*j)) / 2^(k*(acc+2*j)) = 0
        alpha + beta * 2^(acc+2*j) = 0

    So one has a system of two equations:

        alpha + beta = 1
        alpha + beta * 2^(acc+2*j) = 0

    which solution is:

        beta * 2^(acc+2*j) + 1 - beta = 0
        beta * (2^(acc+2*j) - 1) + 1 = 0
        beta = - 1 / (2^(acc+2*j) - 1)

        alpha = 1 + 1 / (2^(acc+2*j) - 1)
        alpha = (2^(acc+2*j) - 1 + 1) / (2^(acc+2*j) - 1)
        alpha = 2^(acc+2*j) / (2^(acc+2*j) - 1)

    so:

        alpha = 2^(acc+2*j) / (2^(acc+2*j) - 1)
        beta = - 1 / (2^(acc+2*j) - 1)

    Inserting this into the linear combination equation:

        alpha * F_{j,k} + beta * F_{j,k-1}
            = alpha * (D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j))) +
              beta * (D + C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j)))

    directly gives:

        2^(acc+2*j) / (2^(acc+2*j) - 1) * F_{j,k} -
        1/(2^(acc+2*j) - 1) * F_{j,k-1}
            = (2^(acc+2*j) / (2^(acc+2*j) - 1)) *
              (D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j))) -
              1/(2^(acc+2*j) - 1) *
              (D + C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j)))

    multiplying both sides by (2^(acc+2*j) - 1):

        2^(acc+2*j) * F_{j,k} - F_{j,k-1}
            = 2^(acc+2*j) * (D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j))) -
                             (D + C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j)))

    now solving for D gives, as just enforced by the linear system,
    the cancellation of the error term:

        2^(acc+2*j) * F_{j,k} - F_{j,k-1}
            = 2^(acc+2*j) * (D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j))) -
              (D + C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j)))

            = 2^(acc+2*j) * D - D + (C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j)) -
                                    (C_j * h^(acc+2*j) / 2^((k-1)*(acc+2*j))))

            = 2^(acc+2*j) * D - D

            = D*(2^(acc+2*j) - 1)

    so solving for the exact derivative D value after k iterations is:

        D = (2^(acc+2*j) * F_{j,k} - F_{j,k-1}) / (2^(acc+2*j) - 1) +
            O(h^(acc+2*(j+1)))

    which means that the computed entry in the table would be:

        F_{j+1,k} = (2^(acc+2*j) * F_{j,k} - F_{j,k-1}) / (2^(acc+2*j) - 1)

    which is equivalent to:

        F_{j,k} = (2^(acc+2*(j-1)) * F_{j-1,k} - F_{j-1,k-1}) /
                  (2^(acc+2*(j-1)) - 1)


    This final result is the recursive formula used in the Richardson
    extrapolation which is implemented recursively up to k_max
    (= `maximum_richardson_equations` - 1). It increases accuracy by
    eliminating at every iteration j the term proportional to h^(acc+2*j), so
    the remaining error term at each iteration is O(h^(acc+2*(j+1))).

    ---

    ### Romberg error estimation [1, Section 4.5][2, Algorithm dfridr.h]

    The error estimation used in this function comes from the
    Romberg-integration which provides from the aplication of the Richardson
    extrapolation in order to improve the accuracy of a trapezoidal integration,
    as showed above and in [1, Section 4.5] the extrapolation can be performed
    at any aproximation with a truncation-error with the form:

        sum_{j=1}^{m-1}  C_j * h^(2*j) + O(h^(2*m))

    that is, with only even powers of h. It is performed until
    |F_{j,j} - F_{j-1,j-1}|, which are the best approximating values at row j
    and row j-1 respectively, are within a given tolerance.


    The following is a proof of how the Richardson extrapolation truncation
    error has an upper bound and that the logic used in Romberg-integration can
    be applied here too:

    Start by considering an arbitrary extrapolation table component:

        F_{j,k} = D + C_j * h^(acc+2*j) / 2^(k*(acc+2*j)) +
                  O(h^(acc+2*(j+1)))

    it has only even powers of h and at row j the component with the least
    truncation error is F_{j,j}:

        F_{j,j} = D + C_j * h^(acc+2*j) / 2^(j*(acc+2*j)) +
                  O(h^(acc+2*(j+1)))

    now consider the previous row, j-1, component with the least truncation
    error:

        F_{j-1,j-1} = D + C_{j-1} * h^(acc+2*(j-1)) / 2^((j-1)*(acc+2*(j-1))) +
                      O(h^(acc+2*j))

    when one substract the last 2 expression, gets:

        F_{j,j} - F_{j-1,j-1}

            = D + C_j * h^(acc+2*j) / 2^(j*(acc+2*j)) -
              (D + C_{j-1} * h^(acc+2*(j-1)) / 2^((j-1)*(acc+2*(j-1))))

            = C_j * h^(acc+2*j) / 2^(j*(acc+2*j)) -
              C_{j-1} * h^(acc+2*(j-1)) / 2^((j-1)*(acc+2*(j-1)))

    for a sufficiently small h, the second term dominates because it has
    two powers of h lower than the first one, so:

        |F_{j,j} - F_{j-1,j-1}|
            ≈ |C_{j-1} / 2^((j-1)*(acc+2*(j-1)))| * h^(acc+2*(j-1))
            = |F_{j-1,j-1} - D|

    meaning that the absolute difference value of consecutive diagonal entries
    asymptotically approximates the error of the lower-order entry
    F_{j-1,j-1}.

    Because F_{j,j} is more accurate than F_{j-1,j-1} by h^2, it can be shown
    that their difference is also a strict (and conservative) upper bound on
    the error of F_{j,j}:

        |F_{j,j} - D| = |(F_{j,j} - F_{j-1,j-1}) + (F_{j-1,j-1} - D)|

    this last expression can be modified by knowing that for two arbitrary real
    numbers x and y, |x+y| <= |x|+|y|, so:

        |F_{j,j} - D| = |(F_{j,j} - F_{j-1,j-1}) + (F_{j-1,j-1} - D)|
                      <= |F_{j,j} - F_{j-1,j-1}| + |F_{j-1,j-1} - D|
                      ≈ 2 * |F_{j,j} - F_{j-1,j-1}|

    resulting in the consecutive-diagonal difference being:

        |F_{j,j} - D| <= 2 * |F_{j,j} - F_{j-1,j-1}|

    which proves that the error estimation of |F_{j,j} - D| is conservative
    and has an upper bound of |F_{j,j} - F_{j-1,j-1}| up to a factor of 2.


    This is the classical Romberg-like error estimate, originally derived
    for Romberg integration [1, Section 4.5] and later applied in numerical
    differentiation [2, Section 5.7].

    ---

    ### Step-size scaling when step_size is 'auto'

    When `step_size_is_auto` is True, `base_step_size_array` is chosen by
    ***_auto_step_size*** which computes h_auto so that truncation and roundoff
    errors are roughly balanced:

        E_t(h_auto) ≈ t_c * |f^{n+acc}(x)| * h^acc            (truncation)
        E_r(h_auto) ≈ eps * coefficient_sum * |f(x)| / h^n    (roundoff)

    Richardson extrapolation requires evaluating the stencil at successively
    halved step sizes h_auto, h_auto/2, ..., h_auto/2^k_max. Because h_auto is
    at the truncation/roundoff boundary, the finest evaluated step
    h_auto/2^k_max will be inside the roundoff-dominated regime.

    In that regime, halving the step doubles the roundoff contribution relative
    to truncation, so instead of canceling truncation error, each Richardson
    table row amplifies noise, making the extrapolated result actually worse
    than the base stencil.


    This problem can be addressed by scaling the base step (h_auto) so that
    even the finest evaluated step (h_auto / 2^k_max) is in the
    truncation-dominated regime within a pre-defined safety margin:

    now, defining R, the desired ratio of truncation error to roundoff error at
    the finest evaluated step:

        R := E_t(h_finest) / E_r(h_finest)

    Since both errors are balanced at h_auto, their ratio at any arbitray step
    size h is:

        E_t(h) / E_r(h) = (t_c * |f^{n+acc}(x)| * h^acc) /
                          (eps * coefficient_sum * |f(x)| / h^n)
                        = (t_c * |f^{n+acc}(x)|) /
                          (eps * coefficient_sum * |f(x)|) * h^(acc+n)

    defining:

        K := (t_c * |f^{n+acc}(x)|) / (eps * coefficient_sum * |f(x)|)

    one gets that:

        E_t(h) / E_r(h) = K * h^(acc+n)

    Now, recalling that for the base step size of this function when step size
    is set to 'auto', h_auto, both error have aproximately the same magnitude:

        E_t(h_auto) / E_r(h_auto) ≈ 1

    which can be introduced in the previous expression getting:

        E_t(h_auto) / E_r(h_auto) = K * (h^(acc+n)) = 1

    solving for K, one gets that:

        K = 1/h^(acc+n)

    this last result allows one to substitue K in the equation for an arbitrary
    h, so the ratio of both errors becomes:

        E_t(h) / E_r(h) = K * h^(acc+n) = (h / h_auto)^(n+acc)

    in this last equation, setting h to h_finest gives:

        E_t(h_finest) / E_r(h_finest) = (h_finest / h_auto)^(n+acc) = R

    solving for h_finest:

        (h_finest / h_auto)^(n+acc) = R
        h_finest / h_auto = R^(1/(n+acc))

    defining now:

        safety_factor := R^(1/(n+acc))

    the finest step is therefore:

        h_finest = safety_factor * h_auto


    Giving all these results, the step from which the Richardson extrapolation
    starts halving said step k_max times is:

        h_start = 2^k_max * h_finest = 2^k_max * h_auto * safety_factor
                = 2^k_max * h_auto * R^(1/(n+acc))


    It is to be noted that as n grows, then safety_factor = R^(1/(n+acc))
    becomes smaller, and as it is dived by h^n it is made more roundoff
    sensitive, so a modest step increase already yields a large truncation to
    roundoff ratio and h_start does not need to be pushed far from h_auto.

    Notes
    -----
    As the 2^k_max factor grows exponentially, for k_max > ~8 the scaled
    h_start can lead to evaluation points outside the function's domain or
    beyond the region where the Taylor coefficients C_k are slowly varying,
    producing nan / inf at some of them.

    References
    ----------
    [1] Richard L. Burden, Douglas J. Faires & Annete M. Burden (2016).
        **Numerical Analysis, 10th edition.**
        Cengage Learning.
        Sections 4.2 & 4.5.
        ISBN: 978-1-305-25366-7.

    [2] William H. Press, Saul A. Teukolsky, William T. Vetterling &
        Brian P. Flannery (2007).
        **Numerical Recipes: The Art of Scientific Computing, 3rd edition.**
        Cambridge University Press.
        Section 5.7.
        ISBN: 978-0-521-88068-8.
    """

    # k_max in the docstring, renamed here for clarity
    max_columns = maximum_richardson_equations - 1

    # Only the current row is needed as it can be modified in place, so
    # current_extrapolation_table_row is a 1D array which can be built from the
    # previous values inside it and can overwritten in place.
    current_extrapolation_table_row = np.zeros(max_columns+1, dtype=float)

    # This variable is to signal that all columns are non finite as an output
    # of the _fill_first_extrapolation_row function
    _ALL_COLUMNS_NON_FINITE = -100



    def _fill_first_extrapolation_row(
        start_step_size_array: np.ndarray,
        last_column: int,
    ) -> int:
        """
        Fill the extrapolation table:

            current_extrapolation_table_row[0: `last_column` + 1]

        from column 0 to column `last_column` (included) in place with the
        first row estimates F_{0,k}, evaluated at the step sizes
        `start_step_size_array` / 2**k for k = 0, 1, ..., `last_column`.

        Returns the index of the first column whose estimate is non-finite, or
        _ALL_COLUMNS_NON_FINITE when every column up to `last_column` is finite.
        """

        for column_k in range(last_column + 1):
            step_halving_factor = 2.0 ** column_k

            evaluation_step_size_array = (
                start_step_size_array / step_halving_factor
            )

            current_extrapolation_table_row[column_k] = (
                _evaluate_component_central_difference(
                    user_function=user_function,
                    point=point,
                    indexes_counter=indexes_counter,
                    point_number=point_number,
                    step_size_array=evaluation_step_size_array,
                )
            )

            if not np.isfinite(current_extrapolation_table_row[column_k]):
                return column_k

        # last_column was reached by the loop
        return _ALL_COLUMNS_NON_FINITE



    # Fill the first row F_{0, k}.
    if step_size_is_auto:
        total_derivative_order = sum(indexes_counter.values())

        # A value of 100 has been chosen to balance:
        # When the ratio is below ~30, it leaves a truncation error so low that
        # Richardson can't gain more accuracy.
        # When the ratio is above ~1000, h_start is so far from the evaluation
        # point that fast-growing functions (for example, exp(x*y) mixed
        # partials derivatives) or functions with restricted domains (for
        # example log near the origin) produce NaNs or finite-difference
        # cancellation.
        target_truncation_to_roundoff_ratio = 100.0 # R
        safety_factor = (
            target_truncation_to_roundoff_ratio
            ** (1.0 / (total_derivative_order + accuracy))
        )


        last_used_max_columns = -1
        # Try to fill the first row up to max_columns, as the 'auto' start step
        # depends on 2^max_columns, every time a non-finite column is found then
        # max_columns is capped at the previous column, so the start step must
        # be recomputed and the row refilled.
        # At most maximum_richardson_equations - 2 iterations can be ran.
        for iteration in range(maximum_richardson_equations - 1):

            # If max_columns hasn't changed since the last fill, the
            # start_step_size_array would be identical and the fill would
            # produce the same values again. Skip directly to a smaller
            # max_columns.
            if max_columns == last_used_max_columns:
                max_columns -= 1
                if max_columns < 1:
                    break

            last_used_max_columns = max_columns

            # h_start = h_auto * R^(1/(n+acc)) * 2^max_columns
            start_step_size_array = (
                base_step_size_array * safety_factor * (2.0 ** max_columns)
            )

            first_non_finite_column_index = _fill_first_extrapolation_row(
                start_step_size_array=start_step_size_array,
                last_column=max_columns
            )
            # Fill completed with all columns being finite
            if first_non_finite_column_index == _ALL_COLUMNS_NON_FINITE:
                break

            # If a non-finite column was found, max_columns is capped to the
            # last finite column. It can happen because step is already too
            # large at column 0, or stencil left the domain.
            max_columns = first_non_finite_column_index - 1

            # Fewer than two usable columns means that no extrapolation is
            # possible with current max_columns
            if max_columns < 1:
                # If the first iteration yielded a non-finite result in the
                # first column, the evaluation point might have escaped
                # function domain, so, repeat the richardson extrapolation with
                # one richardson equation less than the previous iteration
                max_columns = maximum_richardson_equations - 2 - iteration
                if max_columns > 0:
                    continue

                else:
                    break


        # If the previous loop was broken when the maximum columns equals to 0
        # or -1 then
        if max_columns < 1:
            warn(f"Richardson extrapolation only found non-finite "
                 f"values for all maximum_richardson_equations "
                 f"({maximum_richardson_equations}) provided "
                 f"down to two (the minimum), returning the derivative "
                 f"value evaluated using the stencil only.",
                 RuntimeWarning, stacklevel=2)

            stencil_evaluation = _evaluate_component_central_difference(
                user_function=user_function,
                point=point,
                indexes_counter=indexes_counter,
                point_number=point_number,
                step_size_array=base_step_size_array,
            )

            return stencil_evaluation, float('nan')


    # Fixed step sizes (non 'auto')
    else:
        # It is safe to use same array, it is already copied in
        # _evaluate_component_dispatcher.
        start_step_size_array = base_step_size_array
        first_non_finite_column_index = _fill_first_extrapolation_row(
            start_step_size_array=start_step_size_array,
            last_column=max_columns
        )

        # As non 'auto' step sizes do not depend on max_columns, when a
        # non-finite value is encountered, the already-filled columns stay
        # valid and only max_columns needs to be capped.
        if first_non_finite_column_index != _ALL_COLUMNS_NON_FINITE:
            max_columns = first_non_finite_column_index - 1

            if max_columns < 1:
                # No extrapolation is possible
                return float('nan'), float('inf')



    # current_extrapolation_table_row[0] = F_{0, 0}
    best_diagonal_value = float(current_extrapolation_table_row[0])
    best_truncation_error = float('inf')
    previous_truncation_error = float('inf')

    for row_j in range(1, max_columns+1):
        accuracy_constant = 2.0 ** (accuracy + 2 * (row_j - 1))

        # Iterate each column of the table from right-to-left so a 1-D array
        # can be used so the update can be run in-place:
        # current_extrapolation_table_row[column_k-1] still holds its
        # previous-pass value when current_extrapolation_table_row[column_k] is
        # overwritten.
        for column_k in range(max_columns, row_j-1, -1):
            current_extrapolation_table_row[column_k] = (
                accuracy_constant * current_extrapolation_table_row[column_k] -
                current_extrapolation_table_row[column_k-1]
            ) / (accuracy_constant - 1.0)

            if not np.isfinite(current_extrapolation_table_row[column_k]):
                # Previous diagonal was the best
                return best_diagonal_value, best_truncation_error


        # Romberg error estimation: |F_{j,j} - D| <= 2 * |F_{j,j} - F_{j-1,j-1}|
        current_diagonal_value = float(current_extrapolation_table_row[row_j])
        previous_diagonal_value = float(
            current_extrapolation_table_row[row_j-1]
        )
        current_truncation_error = 2 * abs(
            current_diagonal_value - previous_diagonal_value
        )



        # Error has reached machine precision relative to the value
        machine_precision_floor = FLOAT_EPSILON * max(
            abs(current_diagonal_value), 1.0
        )
        if current_truncation_error <= machine_precision_floor:
            return current_diagonal_value, current_truncation_error

        # Truncation error estimation has stopped decreasing, past this point
        # further passes only add noise, and the previous diagonal entry is the
        # most accurate found value
        if current_truncation_error > previous_truncation_error:
            return best_diagonal_value, best_truncation_error


        best_diagonal_value = current_diagonal_value
        best_truncation_error = current_truncation_error

        previous_truncation_error = current_truncation_error


    # Reached max_columns without an early exit, return the deepest estimate.
    return best_diagonal_value, best_truncation_error



def _evaluate_component_complex_step(
    user_function: Callable[[np.ndarray], complex],
    point: np.ndarray,
    indexes_counter: Counter[int],
) -> Tuple[float, float]:
    """
    Computes one first derivative component at `point` using the complex-step
    method.

    Parameters
    ----------
    user_function : Callable[[np.ndarray], complex]
        Function to differentiate. Must accept a 1-D array of shape (D,)
        of complex dtype and return a (possibly complex) scalar. It must
        be composed only of operations that are analytic over the
        complex numbers (basic arithmetic, np.sin, np.exp, np.cos,
        np.tan, np.sinh, etc.). Non-analytic operations such as abs(),
        np.real(), np.imag(), np.conj(), np.maximum, or branch-cut
        functions evaluated on or across their cut **will silently
        produce wrong results**.
    point : np.ndarray
        Real evaluation point, shape (D,).
    indexes_counter : Counter[int]
        A Counter that encodes each active coordinate index to its
        per-coordinate differentiation order. It must contain exactly one entry
        k with value 1 (Counter({k: 1})), meaning df(x)/dx_k is computed.

        It is assumed that the coordinate indexes are valid.

    Returns
    -------
    Tuple[float, float]
        First, the numerical approximation of df(x)/dx_k at `point`.

        Second, float('nan'), kept for signature symmetry with
        ***_evaluate_component_central_difference*** and
        ***_evaluate_component_richardson***.

    Mathematical Description
    ------------------------
    Letting:

    - e_k be the k-th standard basis vector, with k being the active coordinate
      index encoded by `indexes_counter`.
    - eps = FLOAT_EPSILON, the machine epsilon, ~2.2e-16 for a 64 bit float.
    - h be the constant imaginary step size, h = sqrt(eps).
    - i = sqrt(-1), the imaginary unit.

    ---

    The complex-step method allows to compute approximate derivatives with near
    machine precision by applying an imaginary step size to a function. Unlike
    finite-differences, using this method never subtracts two nearly-equal
    floats making the results independent of the chosen step-size for small
    enough steps [1]:

    Given a function f: R^D -> R that admits an analytic extension to
    C^D, that is, it is a complex-differentiable function when evaluated on
    C^D. Then the Taylor expansion of f along coordinate k at the complex point
    x + i*h*e_k is:

        f(x + i*h*e_k) = f(x) +
                         (i*h) * df(x)/dx_k +
                         (i*h)^2 / 2! * d^2f(x)/dx_k^2 +
                         (i*h)^3 / 3! * d^3f(x)/dx_k^3 +
                         (i*h)^4 / 4! * d^4f(x)/dx_k^4 +
                         (i*h)^5 / 5! * d^5f(x)/dx_k^5 +
                         ...

    as i^2 = -1, i^3 = -i, i^4 = +1, i^5 = +i, ...:

        f(x + i*h*e_k) = f(x) +
                         i*h * df(x)/dx_k +
                         -h^2 / 2! * d^2f(x)/dx_k^2 +
                         -i*h^3 / 3! * d^3f(x)/dx_k^3 +
                         h^4 / 4! * d^4f(x)/dx_k^4 +
                         i*h^5 / 5! * d^5f(x)/dx_k^5 +
                         ...

    separating both the real part and the imaginary part gives:

        Re(f(x + i*h*e_k))
            = f(x) - h^2/2! * d^2f(x)/dx_k^2 + h^4/4! * d^4f(x)/dx_k^4 - ...

        Im(f(x + i*h*e_k))
            = h * df(x)/dx_k - h^3/3! * d^3f(x)/dx_k^3 + ...

    dividing the imaginary part by h:

        Im(f(x + i*h*e_k)) / h = df(x)/dx_k + O(h^2)

    which is the complex-step formula for the first derivative with a
    truncation error of O(h^2):

        |E_t(h)| ≈ h^2 / 6 * |d^3f(x)/dx_k^3| + O(h^4).

    ---

    Now that the truncation error is known, it is interesting to look at how
    roundoff affects the complex-step formula. Let fl(.) denote the
    floating-point value of an expression as actually computed. The
    derivative estimate is:

        fl(Im(f(x + i*h*e_k))) / h

    so its absolute roundoff error is approximately:

        E_r ≈ |fl(Im(f(x + i*h*e_k))) - Im(f(x + i*h*e_k))| / h

    this can be bounded to the roundoff error of the computed imaginary part
    itself. This is done by showing that when f is a composition of basic
    complex arithmetic operations or elementary analytic functions and is
    evaluated at x + i*h*e_k and every intermediate complex value has the form:

        O(1) + i * O(h)

    with absolute floating-point error O(eps) on the real part and O(eps * h)
    on the imaginary part:

    - Complex addition / subtraction:

        (a + i*b) +- (c + i*d) = (a +- c) + i*(b +- d)

    the real part is O(1) +- O(1) = O(1) with absolute error O(eps), and the
    imaginary part is the sum or difference of two O(h) quantities, so it is
    O(h) with absolute error O(eps * h).

    - Complex multiplication:

        (a + i*b) * (c + i*d) = (a*c - b*d) + i*(a*d + b*c)

    the real part a*c - b*d = O(1) + O(h^2) is dominated by a*c, so the result
    is O(1) with absolute error O(eps). The imaginary part is a sum of two O(h)
    terms; each individual product is computed with relative error eps, so the
    absolute error is O(eps * h).

    - Complex division:

        (a + i*b) / (c + i*d) = ((a*c + b*d) + i*(b*c - a*d)) / (c^2 + d^2)

    the denominator c^2 + d^2 = O(1) + O(h^2) is dominated by c^2 = O(1). The
    numerator's real part a*c + b*d = O(1) + O(h^2) is dominated by a*c, while
    its imaginary part b*c - a*d is a difference of two O(h) terms. So the
    result is O(1) real, O(h) imaginary, with absolute errors O(eps) and
    O(eps * h) respectively.

    - Analytic elementary functions (sin, cos, exp, tan, sinh, cosh, ...).
    For any g analytic at a, expanding around the real point gives:

        g(a + i*b) = [g(a) - b^2/2 * g''(a) + O(b^4)]
                   + i * [b * g'(a) - b^3/6 * g'''(a) + O(b^5)]

    so for b = O(h) the output has real part g(a) + O(h^2) = O(1) and imaginary
    part b * g'(a) + O(h^3) = O(h). Assuming that the implementation of g has a
    relative error of order eps on each component, the absolute floating-point
    errors are O(eps) on the real part and O(eps * h) on the imaginary part.


    Each elementary operation therefore preserves the format
    "O(1) real, O(h) imaginary, with O(eps) and O(eps * h) absolute errors",
    so the property is preserved under composition.

    In particular, the absolute floating-point error in the imaginary part of f
    stays proportional to its own magnitude (O(h)) rather than to the
    O(1) magnitude of the real part.


    It follows that:

        |fl(Im(f(x + i*h*e_k))) - Im(f(x + i*h*e_k))| ≈ eps * h * |df(x)/dx_k|

    and dividing by h to form the derivative estimate gives:

        E_r ≈ eps * h * |df(x)/dx_k| / h = eps * |df(x)/dx_k|

    which is machine precision on the value of the derivative itself,
    independent of h.


    This conclusion assumes that f is a composition of elementary analytic
    operations. If a function performs a cancellation between intermediate
    values (in either the real or the imaginary part), or uses non-analytic
    operations, this error estimate no longer applies.

    ---

    Summarizing, both errors are:

    - Truncation error: E_t(h) ≈ h^2 / 6 * |d^3f(x)/dx_k^3|
    - Roundoff error: E_r ≈ eps * |df(x)/dx_k|

    Because E_r does not depend on h, making h smaller improves the total
    error as E_t decreases while E_r stays the same (in contrast with finite
    differences).

    Since no roundoff/truncation balance is needed, h can be chosen by knowing
    that:

    - h must be orders of magnitude greater than the tiniest representable
      float, that is h >> ~2.2e-308 (for 64 bits float).
    - Any product of the form h * a arising inside the evaluation of f,
      where a is bounded by max_k(|x_k|) must be:

        h * max(1, max_k |x_k|) >> ~2.2e-308    for 64 bits float

    Given how easy these are to satisfy, the standard choice is
    [1, Section 2.2]:

        h = sqrt(eps)   (~1.49e-8 for 64 bits float)

    so the truncation error is given by:

        E_t = h^2 / 6 * |d^3f(x)/dx_k^3| = eps / 6 * |d^3f(x)/dx_k^3|

    which is already below machine precision on the derivative for any function
    whose third derivative is O(1).

    Smaller values (down to ~1e-100) can be used, making E_t even smaller but
    offering no benefit since E_t already has the same magnitude as E_r.

    References
    ----------
    [1] Martins, J. R. R. A., Sturdza, P., & Alonso, J. J. (2003).
        **The complex-step derivative approximation.**
        ACM Transactions on Mathematical Software, 29(3), 245-262.
        https://doi.org/10.1145/838250.838251
    """

    complex_step_size = FLOAT_EPSILON ** 0.5

    active_coordinate_index = next(iter(indexes_counter))

    # Create new array with type np.complex128, with two np.float64 inside
    point_complex = point.astype(np.complex128)
    point_complex[active_coordinate_index] += 1j * complex_step_size

    return (
        float(np.imag(user_function(point_complex))) / complex_step_size,
        float('nan')
    )



def _evaluate_component_dispatcher(
    user_function: Callable[[np.ndarray], float],
    point: np.ndarray,
    step_size: Union[Literal['auto', 'complex'], np.ndarray],
    point_number: int,
    indexes_tuple: Tuple[int, ...],
    richardson_extrapolation: bool,
    maximum_richardson_equations: int
) -> Tuple[float, float]:
    """
    Evaluates one component of the derivative tensor at `point`, dispatching
    to the complex step, the plain central differences, or the Richardson
    extrapolation path.

    Parameters
    ----------
    user_function : Callable[[np.ndarray], float]
        Function to differentiate. Must accept a 1-D NumPy array of shape
        (D,) and return a scalar. When `step_size` = 'complex', the
        function must additionally accept arrays of complex dtype and be
        analytic in a neighborhood of `point` (see
        ***_evaluate_component_complex_step***).
    point : np.ndarray
        Single evaluation point, shape (D,).
    step_size : 'auto', 'complex' or np.ndarray
        Finite differences step size. See ***_evaluate_component_complex_step***
        for the complex path and ***_auto_step_size*** for the 'auto' path.
        If an np.ndarray is provided, it is assumed that dimensions match with
        `point`.
    point_number : int
        Base stencil point count for the central differences and Richardson
        paths. Ignored when `step_size` = 'complex'.
    indexes_tuple : tuple of int
        Indexes tuple encoding the component to evaluate. For example:

            (0,)           ->  df / dx_0
            (0, 1)         ->  d^{2}f / dx_0 dx_1
            (1, 1)         ->  d^{2}f / dx_1^{2}
            (0, 1, 1)      ->  d^{3}f / dx_0 dx_1^{2}
            (2, 0, 1, 1)   ->  d^{4}f / dx_2 dx_0 dx_1^{2}

        Its length defines the total derivative order. It is assumed that
        every coordinate index is valid for `point`.
    richardson_extrapolation : bool
        When True, the central differences result is refined by Richardson
        extrapolation through ***_evaluate_component_richardson***.
        Ignored when `step_size` = 'complex'.
    maximum_richardson_equations : int
        Maximum number of step size equations used in the Richardson
        extrapolation. Must be >= 2 when `richardson_extrapolation` is
        True; ignored otherwise.

    Returns
    -------
    (component_value, error_estimate) : tuple of float
        - For the complex step and plain central difference paths,
          `error_estimate` is nan as no estimate is produced on these paths.

        - For the Richardson path, `error_estimate` is the Romberg style
          upper bound of the truncation error when at least one extrapolation
          pass succeeded, or nan when no pass succeeded (the base stencil is
          used then).

    References
    ----------
    [1] William H. Press, Saul A. Teukolsky, William T. Vetterling &
        Brian P. Flannery (2007).
        **Numerical Recipes: The Art of Scientific Computing, 3rd edition.**
        Cambridge University Press.
        Section 5.7.
        ISBN: 978-0-521-88068-8.
    """

    indexes_counter = Counter(indexes_tuple)

    if isinstance(step_size, str) and step_size == 'complex':
        return _evaluate_component_complex_step(
            user_function=user_function,
            point=point,
            indexes_counter=indexes_counter,
        )

    elif isinstance(step_size, str) and step_size == 'auto':
        # As counter preserves insertion order, two indexes_tuple with the
        # same indexes: for example (0, 1, 1) and (1, 0, 1) produces the same
        # step_size_array but two different cache entries. It must be sorted
        # to prevent this cache duplication.
        indexes_counter_tuple = tuple(sorted(indexes_counter.items()))

        step_size_array = _auto_step_size(
            variable_number=point.size,
            indexes_counter_tuple=indexes_counter_tuple,
            point_number=point_number,
        )

        # After numerous test, the factor
        # (|f(x)| / |f^{n+acc}(x)|) ^ (1 / (n+acc)) which was not
        # computed in _auto_step_size has been seen to be too
        # expensive to estimate for the accuracy it provides, so it is
        # assumed (|f(x)| / |f^{n+acc}(x)|) ~ 1, which is reasonable [1] and
        # in some extreme cases using richardson extrapolation, complex-step
        # or tweaking the step size can be done at its expense

    else:
        # step_size is an array of floats
        step_size_array = step_size

    # All step_size_array paths are read-only at this point



    if not richardson_extrapolation:
        component_value = _evaluate_component_central_difference(
            user_function=user_function,
            point=point,
            indexes_counter=indexes_counter,
            point_number=point_number,
            step_size_array=step_size_array
        )

        return component_value, float('nan')

    else:
        # As when computing mixed partial derivatives, each component might
        # have different accuracy
        accuracies_list = []
        for per_dimension_derivative_order in indexes_counter.values():
            effective_point_number = _effective_point_number(
                per_dimension_derivative_order, point_number
            )

            current_dimension_accuracy = (
                effective_point_number - per_dimension_derivative_order + 1
            )
            accuracies_list.append(current_dimension_accuracy)

        # The minimum accuracy across coordinate directions determines the
        # leading error term of the composed stencil, which is the quantity
        # needed in Richardson Extrapolation
        component_accuracy = min(accuracies_list)


        return _evaluate_component_richardson(
            user_function=user_function,
            point=point,
            indexes_counter=indexes_counter,
            point_number=point_number,
            step_size_is_auto=(step_size == 'auto'),
            base_step_size_array=step_size_array,
            accuracy=component_accuracy,
            maximum_richardson_equations=maximum_richardson_equations,
        )



def nth_numerical_derivative(
    function_to_differentiate: Callable[[np.ndarray], float],
    derivative_order: int = 1,
    step_size: Union[Literal['auto', 'complex'], float, Tuple[float, ...]] =
               'auto',
    point_number: Union[Literal['auto'], int] = 'auto',
    single_component: Optional[Union[int, Tuple[int, ...]]] = None,
    richardson_extrapolation: bool = False,
    maximum_richardson_equations: Union[Literal['auto'], int] = 'auto'
) -> Callable[[np.ndarray], DifferentiationResult]:
    """
    Builds a callable that evaluates the `derivative_order`-th partial
    derivative of `function_to_differentiate` using central finite-difference or
    complex-step differentiation.


    - For `derivative_order` = 1 the returned callable evaluates the gradient
    vector df/dx_i.
    - For `derivative_order` = 2 the Hessian matrix d^{2}f(x)/(dx_i dx_j).
    - For `derivative_order` >= 3 the full `derivative_order`-th order tensor
    of mixed partial derivatives.

    `function_to_differentiate` is assumed to be `derivative_order`-times
    continuously differentiable in a neighbourhood of every evaluation point.
    The radius of that neighbourhood is (when `step_size` != 'complex'):

    - For odd `derivative_order`: `step_size` * p/2
    - For even `derivative_order`: `step_size` * (p-1)/2

    where p is the effective point number: `point_number` or
    `point_number` + 1, the one that has opposite parity with
    `derivative_order`.


    Three numerical strategies are available:

    - Plain central finite-difference (the default). Truncation error is
    O(h^(p + 1 - n)), where n = `derivative_order` and p is the effective
    stencil point count along that coordinate. [1]

    - Central finite-difference with Richardson extrapolation
    (`richardson_extrapolation` = True). Each extrapolation equation
    eliminates the leading truncation term, raising accuracy to
    O(h^(p + 1 - n + 2*(`maximum_richardson_equations`-1))) at the cost of
    `maximum_richardson_equations`-1 extra stencil evaluations per tensor
    component. [2][4]
    It provides more accuracy than the plain central stencil and an upper bound
    of the error made by the truncation but also can be more expensive.

    - Complex-step (`step_size` = 'complex'). The most accurate method,
    reaching almost machine-precision and also the cheapest one with just one
    function evaluation. It is only available for the first derivative, the
    function must accept complex inputs and it must be analytical over the
    complex plane in the neighbourhood of radius sqrt(eps)≈1.5e-8, where eps is
    the machine precision of a 64 bit float (~2.2e-16). [3]

    Parameters
    ----------
    function_to_differentiate : Callable[[np.ndarray], float]
        Function to differentiate. Must accept a scalar for one variable
        functions or 1-D NumPy array of shape (D,) for D-variables functions,
        it must return a scalar.

        When `step_size` = 'complex', the function must additionally accept
        arrays of complex dtype, and be analytic in a neighbourhood of every
        evaluation point: it must be composed of operations that are
        differentiable over the complex numbers (basic arithmetic, np.sin,
        np.cos, np.exp, np.tan, np.sinh, np.cosh, np.log, polynomials, ...).
        Non-analytic operations like abs(), np.real(), np.imag(), np.conj(),
        np.maximum, np.minimum, np.where on the imaginary part, or any
        non-differentiable function evaluated on or across cuts **will
        silently produce wrong results**.
    derivative_order : int
        Order n of the derivative (1 = gradient, 2 = Hessian, ...). Must be a
        positive integer.

        Default is 1.
    step_size : 'auto', 'complex', float, or tuple of floats, optional
        Finite-difference step.

        - 'auto': selects a per-coordinate optimal step that balances
        truncation error O(h^acc) against floating-point roundoff
        O(eps / h^n). Yields h_k ~ eps^(1/(n + acc)) along each active
        coordinate k, where acc = effective_point_number - n + 1 is itself
        computed on a per-coordinate basis to satisfy the parity constraint
        of the central-difference stencil (see Notes). This automatic step
        size selection assumes that (|f(x)| / |f^{n+acc}(x)|) ~ 1 at the
        evaluated point, if it is not the case then accuracy will be lost,
        manual step size might be provided to address this.

        - 'complex': uses the complex-step method along a single coordinate
        to compute df/dx_k with O(h^2) truncation error giving effectively
        machine-precision accuracy. Only available for `derivative_order` = 1;
        any other order raises ValueError. Requires `function_to_differentiate`
        to be analytic over the complex numbers (see above).

        - float: the same absolute step h_k = `step_size` is used for every
        active coordinate.

        - tuple of length D: h_k is taken directly from entry k for coordinate
        k.

        When `richardson_extrapolation` = True and `step_size` is not
        'complex', the base step is internally scaled by a safety factor
        proportional to 2^(`maximum_richardson_equations` - 1) so the finest
        halved step remains in the truncation-dominated regime.

        Default is 'auto'.
    point_number : 'auto' or int, optional
        Base number of stencil points per coordinate direction.

        Must satisfy `point_number` >= max(2, `derivative_order` + 1). When
        `point_number` and `derivative_order` share the same parity, the
        effective point count along that coordinate is automatically set to
        `point_number` + 1 so the shown constraint is satisfied. The achieved
        accuracy is then acc = effective_point_number - `derivative_order` + 1,
        always a positive even integer.

        When 'auto':
        - With `richardson_extrapolation` = False, `point_number` =
        `derivative_order` + 7 is selected, a generally good balance between
        accuracy and stencil-coefficient roundoff.
        - With `richardson_extrapolation` = True, `point_number` =
        `derivative_order` + 1 is selected, since Richardson refines best
        from a low-accuracy base stencil with high truncation error.

        Ignored when `step_size` = 'complex'.

        Default is 'auto'.
    single_component : int, tuple of int, or None, optional
        Restrict computation to a single tensor component instead of the full
        derivative tensor.

        - For `derivative_order` = 1, an integer j selects df/dx_j. A 1-tuple
        (j,) is also accepted.
        - For `derivative_order` >= 1, a tuple of `derivative_order`
        coordinate indices (i_1, ..., i_n) selects the mixed partial
        d^{n}f(x)/(dx_{i_1} ... dx_{i_n}). As smoothness is assumed, the
        Schwarz's theorem can be applied, that is, the order of the indices
        does not matter mathematically; numerically, different results may be
        given due to roundoff errors.

        Examples:

            (0,) or 0    ->  df / dx_0
            (0, 1)       ->  d^{2}f / dx_0 dx_1
            (1, 1)       ->  d^{2}f / dx_1^{2}
            (0, 1, 1)    ->  d^{3}f / dx_0 dx_1^{2}
            (2, 0, 1, 1) ->  d^{4}f / dx_2 dx_0 dx_1^{2}

        When None, the full tensor of shape (D,) * n is computed at each input
        point. Computing the full tensor evaluates every index combination, the
        symmetry is taken into account for the mixed partials, so for higher
        orders many components computation are saved. In the case of roundoff
        errors being amplified, a big difference may be obtained when comparing
        a component extracted from the full tensor with a component computed
        directly using this parameter.

        Default is None.
    richardson_extrapolation : bool, optional
        When True, refines the central-difference result by Richardson
        extrapolation. The component is evaluated at successively halved step
        sizes h, h/2, h/4, ... and the values are recombined to cancel
        successive truncation-error terms. The resulting
        ***DifferentiationResult*** carries a non-None error_estimate property
        with a Romberg-style upper bound of the truncation error.

        Ignored when `step_size` = 'complex'.

        Default is False.
    maximum_richardson_equations : 'auto' or int, optional
        Maximum number of step-size equations used in the Richardson
        extrapolation. Must be >= 2, and is only consulted when
        `richardson_extrapolation` = True, ignored otherwise.

        - `maximum_richardson_equations` = 2: eliminates the leading
        O(h^acc) truncation term, raising accuracy to O(h^(acc + 2)).
        - `maximum_richardson_equations` = K: eliminates up to the first K - 1
        truncation terms, reaching O(h^(acc + 2*(K - 1))).

        Each additional equation requires one additional stencil evaluation
        per tensor component. The extrapolation terminates early when the
        Romberg error estimate stops decreasing across two consecutive passes
        or drops below machine precision relative to the current value, so this
        variable acts as a worst-case bound.

        Default cap ('auto') is set to 5, for very smooth functions on a
        extensive well-conditioned domain a higher value can be used, while for
        functions with a small radius of analyticity values above 3 rarely help.

        Default is 5.

    Returns
    -------
    nth_derivative_function : Callable[[np.ndarray], DifferentiationResult]
        A function that accepts:

        - a scalar (0-D array) for a single evaluation point and a
        `function_to_differentiate` that only has one variable, returning
        a ***DifferentiationResult*** whose `derivative` is also a float.

        - a 1-D array of shape (D,) for a single evaluation point, returning a
        ***DifferentiationResult*** whose `derivative` is either a float
        (when `single_component` is set or D = 1) or a tensor of shape
        (D,) * `derivative_order`.

        - a 2-D array of shape (M, D) for M evaluation points, returning a
        ***DifferentiationResult*** whose `derivative` has shape (M,) (when
        `single_component` is set or D = 1) or (M, D, ..., D).


        The returned ***DifferentiationResult*** also has the following
        attributes and methods:

        - `error_estimate`: an array (or scalar) with the same shape as
        `derivative` when `richardson_extrapolation` = True, None
        otherwise. Also None when the complex-step method was used,
        which does not produce a Romberg-style estimate.

        - `differentiation_method`: 'central_difference',
        'richardson_extrapolation', or 'complex_step'.

        - `derivative_order`, `step_size`, `point_number`, `evaluation_points`,
        `single_component`, `richardson_extrapolation`,
        `maximum_richardson_equations`: the configuration that produced the
        result.

        - Convenience properties: `has_error_estimate`,
        `relative_error_estimate`, `is_full_tensor`, `is_scalar`,
        `is_array`, `shape`, `ndim`, `size`, `dtype`.

        - Conversion methods: `as_array()` (writable copy), `as_float()`
        (scalar-only), `to_list()`, `to_dict()`.

        - Comparison methods: `==` (strict, field-by-field, NaN-aware) and
        `allclose(other, rtol=..., atol=..., equal_nan=...)` (numerical
        equivalence on `derivative` and `error_estimate`).

        - `summary(style='compact' | 'full')`: a multi-line summary;
        `'compact'` shows the method, the derivative and the error information,
        `'full'` adds every configuration field and the `evaluation_points`.
        str(result) returns the compact summary.

        The object also forwards `np.asarray(result)`, `result[idx]`,
        `len(result)`, iteration, and `value in result` to the underlying
        `derivative`, so it can be used as a drop-in replacement for a
        standard numpy.ndarray in arithmetic expressions.

    Raises
    ------
    ValueError
        At construction time when:

        - `derivative_order` is not a positive integer.
        - `richardson_extrapolation` is not a bool.
        - `point_number` is a string that is not 'auto', or is an integer
        smaller than max(2, `derivative_order` + 1).
        - `step_size` is a string that is not 'auto' or 'complex'.
        - `step_size` = 'complex' is combined with `derivative_order` != 1.
        - `step_size` cannot be interpreted as a positive finite scalar or
        tuple of floats.
        - `single_component` is an integer combined with `derivative_order`
        != 1, or a tuple whose length does not equal `derivative_order`, or
        cannot be converted to a tuple of integers.
        - `maximum_richardson_equations` is provided as an integer < 2 with
        `richardson_extrapolation` = True' or a string different to 'auto' was
        provided.

        At evaluation time when:

        - The evaluation array is more than 2-dimensional.
        - `single_component` contains indices outside [0, D).
        - `step_size` has a length that does not match D, or contains
        non-positive / non-finite values.

    Notes
    -----
    - As stated in the description of `single_component`, the symmetry is taken
    into account for the mixed partials, so in the case of roundoff errors
    being amplified, a big difference may be obtained when comparing a
    component extracted from the full tensor with a component computed directly
    from using said parameter.

    - The internal evaluator always passes a new allocated array to
    `function_to_differentiate`, so in-place changes are safe. If
    `function_to_differentiate` caches input arrays for later use, those
    caches will hold perturbed evaluation points, that is the user-provided
    point plus a perturbation characteristic of finite-differences.



    The following is a brief guideline on how to choose the different
    parameters of this function:


    **Choosing `step_size`**

    - 'complex' is the most accurate choice when applicable: the function
    must be analytic and `derivative_order` must be 1. It achieves near
    machine precision (relative error of order eps) at the cost of one
    complex-valued function evaluation per component, completely free of
    subtractive cancellation. Use it whenever the function is composed of basic
    arithmetic and elementary transcendentals (np.sin, np.exp, np.cos, ...).

    - 'auto' with the default `point_number` is the right choice everywhere
    else: for higher-order derivatives, for non-analytic functions, and for
    functions whose complex extension is unavailable or expensive.

    - A numeric scalar or per-coordinate tuple is useful when the user has
    prior knowledge of the function's natural scale, or when coordinates
    have very different magnitudes (for example mixing very small and very large
    variables in the same vector).


    **Choosing `point_number` (central-difference path only)**

    - `point_number` = `derivative_order` + 1 and `derivative_order` + 3 are
    fine for low derivative orders (n <= ~4), and are the preferred bases
    when `richardson_extrapolation` = True because Richardson then refines
    them strongly.
    - `point_number` = `derivative_order` + 5 and `derivative_order` + 7
    ('auto') are good general-purpose choices and usually give the best
    accuracy without Richardson.
    - Going beyond `point_number` = `derivative_order` + 11 generally
    degrades accuracy in practice, because the stencil coefficients grow
    fast enough that the roundoff in the weighted sum overcomes the
    additional truncation accuracy.


    **Using `richardson_extrapolation`**

    Richardson extrapolation gives its largest accuracy relative gain on top of
    a low-accuracy base stencil: that is why `point_number` = 'auto' selects
    `derivative_order` + 1 in that case. This doesn't mean a higher
    `point_number` will produce more accurate results, whether the
    extrapolation is worth it depends on the cost of each function evaluation.

    When the function has a small radius of analyticity around the evaluation
    point - for example log(x) close to 0, sqrt(x) close to 0, or rapidly
    growing combinations like exp(x * y) for mixed partials of moderate order
    the evaluated points can land outside the well-behaved region, producing
    nan / inf at some of them. This Richardson extrapolation has been
    implemented to detect this and cap `maximum_richardson_equations`
    automatically, but the returned accuracy is then bounded by what was
    achievable inside the analyticity radius.

    Examples
    --------
    >>> import numpy as np
    >>> from ripples import nth_numerical_derivative

    Build the gradient of a 3-variable function and evaluate it at a single
    point. The result behaves like an ndarray of the derivative and carries
    the configuration that produced it:


    >>> def f(x):
    ...     return x[0] ** 2 + 3.0 * x[1] + np.sin(x[2])
    >>> gradient_f = nth_numerical_derivative(f, derivative_order=1)
    >>> result = gradient_f(np.array([1.0, 2.0, 0.5]))
    >>> result.derivative
    array([2.        , 3.        , 0.87758256])
    >>> result.differentiation_method
    'central_difference'

    Differentiate a single-variable function. The 1-D input has length 1, and
    the derivative comes back as a 0-D array (`is_scalar` = True). Use
    `as_float()` to extract a float:

    >>> def h(x):
    ...     return np.exp(x[0]) * np.cos(x[0])
    >>> third_derivative_h = nth_numerical_derivative(h, derivative_order=3)
    >>> result = third_derivative_h(np.array([1.5]))
    >>> result.derivative
    array(-9.57496905)
    >>> result.is_scalar
    True
    >>> result.as_float()
    -9.574969045195797

    Compute the Hessian of a 2-variable function. The derivative is a square
    symmetric tensor of shape (D, D):

    >>> def g(x):
    ...     return x[0] ** 2 * x[1] + x[1] ** 3
    >>> hessian_g = nth_numerical_derivative(g, derivative_order=2)
    >>> hessian_g(np.array([1.0, 2.0])).derivative
    array([[ 4.,  2.],
           [ 2., 12.]])

    For `derivative_order` >= 3, the result is the full mixed-partials tensor
    of shape (D,) * `derivative_order`. Below, every entry of the third-order
    tensor of (x_0 + x_1)^3 equals 6 because every third partial of that
    polynomial equals 6:

    >>> def cubed_sum(x):
    ...     return (x[0] + x[1]) ** 3
    >>> third_order_cs = nth_numerical_derivative(cubed_sum, derivative_order=3)
    >>> result = third_order_cs(np.array([0.5, -0.25]))
    >>> result.derivative.shape
    (2, 2, 2)
    >>> result.derivative
    array([[[6., 6.],
            [6., 6.]],

           [[6., 6.],
            [6., 6.]]])

    Compute a single component of the derivative tensor directly, skipping
    the rest. For `derivative_order` = 1 a plain integer selects df/dx_j:

    >>> partial_x0 = nth_numerical_derivative(
    ...     f, derivative_order=1, single_component=0,
    ... )
    >>> partial_x0(np.array([1.0, 2.0, 0.5])).derivative
    array(2.)

    For `derivative_order` >= 2, a tuple of length `derivative_order` selects
    a mixed partial. Extract the off-diagonal Hessian entry of g without
    computing the full Hessian:

    >>> cross_partial = nth_numerical_derivative(
    ...     g, derivative_order=2, single_component=(0, 1),
    ... )
    >>> cross_partial(np.array([1.0, 2.0])).derivative
    array(2.)

    Higher-order mixed partials follow the same pattern. Below, (1, 1, 1)
    selects d^3 g / dx_1^3, which equals 6 (third derivative of x_1^3):

    >>> mixed_partial = nth_numerical_derivative(
    ...     g, derivative_order=3, single_component=(1, 1, 1),
    ... )
    >>> mixed_partial(np.array([1.0, 2.0])).derivative
    array(6.)

    Use the complex-step method for an analytic function. It is restricted to
    `derivative_order` = 1, but it gives near machine precision (relative
    error of order eps) at the cost of a single complex-valued function
    evaluation per component. Because the method is free of subtractive
    cancellation, no truncation-error estimate is produced
    (`error_estimate` is None):

    >>> def analytic_f(x):
    ...     return np.sin(x[0]) * np.exp(x[1])
    >>> gradient_complex = nth_numerical_derivative(
    ...     analytic_f, derivative_order=1, step_size='complex',
    ... )
    >>> result = gradient_complex(np.array([1.0, 0.5]))
    >>> result.differentiation_method
    'complex_step'
    >>> result.derivative
    array([0.8908079 , 1.38735111])
    >>> result.error_estimate is None
    True

    Provide a single absolute step size, applied uniformly to every
    coordinate. Useful when the user has prior knowledge of the natural
    scale of the function:

    >>> gradient_fixed = nth_numerical_derivative(
    ...     f, derivative_order=1, step_size=1e-5,
    ... )
    >>> result = gradient_fixed(np.array([1.0, 2.0, 0.5]))
    >>> result.derivative
    array([2.        , 3.        , 0.87758256])
    >>> result.step_size
    1e-05

    For variables whose magnitudes differ by many orders of magnitude, pass
    a tuple of per-coordinate step sizes so each direction is sampled
    individually. The function below has a small coordinate (x_1 ~ 1e-6)
    with a very large prefactor, which would lose accuracy under a single
    shared step:

    >>> def f_multi_scale(x):
    ...     return x[0] ** 2 + 1.0e10 * x[1] ** 2
    >>> gradient_per_coord = nth_numerical_derivative(
    ...     f_multi_scale, derivative_order=1, step_size=(1e-6, 1e-12),
    ... )
    >>> gradient_per_coord(np.array([1.0, 1.0e-6])).derivative
    array([2.00000000e+00, 2.00000002e+04])

    Enable Richardson extrapolation to refine the central-difference result.
    The returned ***DifferentiationResult*** then carries a per-component
    truncation-error upper bound with the same shape as `derivative`:

    >>> hessian_richardson = nth_numerical_derivative(
    ...     g, derivative_order=2, richardson_extrapolation=True,
    ... )
    >>> result = hessian_richardson(np.array([1.0, 2.0]))
    >>> result.differentiation_method
    'richardson_extrapolation'
    >>> result.has_error_estimate
    True
    >>> result.error_estimate.shape
    (2, 2)
    >>> bool(np.max(result.relative_error_estimate) < 1e-9)
    True

    Cap the Richardson depth explicitly to bound the per-component cost in
    the worst case. The cap is then exposed on the result:

    >>> hessian_bounded = nth_numerical_derivative(
    ...     g,
    ...     derivative_order=2,
    ...     richardson_extrapolation=True,
    ...     maximum_richardson_equations=3,
    ... )
    >>> hessian_bounded(np.array([1.0, 2.0])).maximum_richardson_equations
    3

    Evaluate the derivative at a batch of M points by stacking them as an
    (M, D) array. The returned ***DifferentiationResult*** carries an array
    with a leading M-axis:

    >>> points = np.array([[1.0, 2.0], [0.5, 1.0], [2.0, 3.0]])
    >>> batch_result = hessian_g(points)
    >>> batch_result.derivative.shape
    (3, 2, 2)
    >>> batch_result.derivative[0]
    array([[ 4.,  2.],
           [ 2., 12.]])
    >>> batch_result.derivative[2]
    array([[ 6.,  4.],
           [ 4., 18.]])

    Override `point_number` to demand a wider base stencil. Useful when the
    function is very smooth and a wider stencil pays off, or when the user
    wants a specific accuracy order:

    >>> gradient_wide = nth_numerical_derivative(
    ...     f, derivative_order=1, point_number=10,
    ... )
    >>> result = gradient_wide(np.array([1.0, 2.0, 0.5]))
    >>> result.point_number
    10
    >>> result.derivative
    array([2.        , 3.        , 0.87758256])

    The returned ***DifferentiationResult*** exposes the usual NumPy-style
    shape and dtype properties of the underlying derivative array:

    >>> result = hessian_g(np.array([1.0, 2.0]))
    >>> result.shape
    (2, 2)
    >>> result.ndim
    2
    >>> result.size
    4
    >>> result.dtype
    dtype('float64')
    >>> result.is_scalar
    False
    >>> result.is_array
    True
    >>> result.is_full_tensor
    True

    Convert the result back to plain Python or NumPy structures. `as_array()`
    returns a writable copy that is safe to mutate; `as_float()` returns a
    Python float when the derivative is scalar (raises otherwise); `to_list()`
    returns a nested list; and `to_dict()` returns a fully-serialisable
    dictionary that contains the derivative, the error estimate, and every
    configuration field:

    >>> writable_copy = result.as_array()
    >>> writable_copy.flags.writeable
    True
    >>> writable_copy[0, 0] = 99.0
    >>> bool(result.derivative[0, 0] != 99.0) # the original is untouched
    True
    >>> dict_form = result.to_dict()
    >>> len(dict_form)
    10
    >>> list(dict_form)[:3]
    ['derivative', 'error_estimate', 'derivative_order']
    >>> scalar_result = cross_partial(np.array([1.0, 2.0]))
    >>> round(scalar_result.as_float(), 12)
    2.0

    Two results compare strictly equal only when every configuration field
    and every numerical value matches. Use `allclose()` to compare results
    obtained with different settings (different stencil sizes, different
    step sizes, ...) up to a tolerance:

    >>> first = hessian_g(np.array([1.0, 2.0]))
    >>> second = hessian_g(np.array([1.0, 2.0]))
    >>> first == second
    True
    >>> alternative = nth_numerical_derivative(
    ...     g, derivative_order=2, point_number=5,
    ... )(np.array([1.0, 2.0]))
    >>> first == alternative # different point_number
    False
    >>> first.allclose(alternative, relative_tolerance=1e-6)
    True

    The result also behaves like a NumPy array of the underlying derivative:
    `np.asarray(result)`, indexing, `len()`, and iteration all forward to
    the derivative, so the result can be plugged directly into NumPy
    expressions:

    >>> result = hessian_g(np.array([1.0, 2.0]))
    >>> np.asarray(result).shape
    (2, 2)
    >>> len(result)
    2
    >>> for row in result:
    ...     print(row)
    [4. 2.]
    [ 2. 12.]
    >>> 2 * np.asarray(result)
    array([[ 8.,  4.],
           [ 4., 24.]])

    Print a multi-line summary. `'compact'` (the default, also used by
    `str(result)`) shows only the method, the derivative, and the error
    information:

    >>> result = hessian_richardson(np.array([1.0, 2.0]))
    >>> print(result.summary())
    DifferentiationResult (compact)
    ===============================
      differentiation_method  : richardson_extrapolation
      derivative              :
        array([[ 4.,  2.],
               [ 2., 12.]])
      error_estimate          :
        array([[1.7923e-11, 1.8576e-12],
               [1.8576e-12, 1.7923e-11]])
      max  |error_estimate|   : 1.79234e-11
      mean |error_estimate|   : 9.89053e-12
      relative_error_estimate :
        array([[4.4809e-12, 9.2881e-13],
               [9.2881e-13, 1.4936e-12]])
      max  |relative_error|   : 4.48086e-12
      mean |relative_error|   : 1.95803e-12

    `'full'` also adds every configuration field and the array of
    evaluation points at which the derivative was computed:

    >>> print(result.summary('full'))  # doctest: +ELLIPSIS
    DifferentiationResult (full)
    ============================
      derivative_order             : 2
      differentiation_method       : richardson_extrapolation
      shape                        : (2, 2)
      ndim                         : 2
      size                         : 4
      dtype                        : float64
      step_size                    : 'auto'
      point_number                 : 3
      single_component             : None
      richardson_extrapolation     : True
      maximum_richardson_equations : 5
      evaluation_points.shape      : (1, 2)
      evaluation_points            : array([[1., 2.]])
      derivative                   :
        array([[ 4.,  2.],
               [ 2., 12.]])
      ...


    References
    ----------
    [1] Randall J. LeVeque (2007).
        **Finite Difference Methods for Ordinary and Partial Differential
        Equations: Steady-State and Time-Dependent Problems.**
        SIAM (Society for Industrial and Applied Mathematics).
        Chapter 1.
        https://doi.org/10.1137/1.9780898717839

    [2] William H. Press, Saul A. Teukolsky, William T.
        Vetterling & Brian P. Flannery (2007).
        **Numerical Recipes: The Art of Scientific Computing, 3rd edition.**
        Cambridge University Press.
        Section 5.7.
        ISBN: 978-0-521-88068-8.

    [3] Martins, J. R. R. A., Sturdza, P., & Alonso, J. J. (2003).
        **The complex-step derivative approximation.**
        ACM Transactions on Mathematical Software, 29(3), 245-262.
        https://doi.org/10.1145/838250.838251

    [4] Richard L. Burden, Douglas J. Faires & Annette M. Burden (2016).
        **Numerical Analysis, 10th edition.**
        Cengage Learning.
        Sections 4.2 & 4.5.
        ISBN: 978-1-305-25366-7.

    [5] Wikipedia contributors.
        **Finite Differences.**
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences
    """

    validated_parameters = _validate_nth_numerical_derivative_parameters(
        derivative_order=derivative_order,
        step_size=step_size,
        point_number=point_number,
        single_component=single_component,
        richardson_extrapolation=richardson_extrapolation,
        maximum_richardson_equations=maximum_richardson_equations,
    )
    (
        _step_size,
        _point_number,
        _single_component_tuple,
        _richardson_extrapolation,
        _maximum_richardson_equations,
        differentiation_method
    ) = validated_parameters



    def nth_derivative_function(
        evaluated_points: np.ndarray
    ) -> DifferentiationResult:
        """
        Evaluate the pre-configured derivative at point(s) `evaluated_points`.
        """

        _evaluated_points = np.asarray(evaluated_points, dtype=float)
        original_input_dimensions = _evaluated_points.ndim

        if original_input_dimensions > 2:
            raise ValueError(
                f"evaluated_points must be 0-D, 1-D, or 2-D. Got "
                f"{original_input_dimensions}-D array of shape "
                f"{_evaluated_points.shape}."
            )

        # Normalise to a 2-D (M, D) view for the inner loops.
        if original_input_dimensions == 0:
            # For example: 5.0  ->  [[5.0]]
            _evaluated_points = _evaluated_points.reshape(1, 1)

        elif original_input_dimensions == 1:
            # For example: [2.0, 3.0, 4.0]  ->  [[2.0, 3.0, 4.0]]
            _evaluated_points = _evaluated_points.reshape(1, -1)

        num_points, variable_number = _evaluated_points.shape


        # Validate single_component indices, they must be
        # 0 <= index < variable_number
        if _single_component_tuple is not None:
            out_of_range = [
                index for index in _single_component_tuple
                if not (0 <= index < variable_number)
            ]
            if out_of_range:
                raise ValueError(
                    f"single_component indices {out_of_range} out of range for "
                    f"D={variable_number} (valid range: 0 to "
                    f"{variable_number - 1})."
                )


        resolved_step_size = _resolve_step_size(_step_size, variable_number)


        single_point_input = original_input_dimensions <= 1



        # Single-component or function of 1 variable, one scalar per input point
        if _single_component_tuple is not None or variable_number == 1:
            if _single_component_tuple is not None:
                indexes_tuple = _single_component_tuple
            else:
                # variable_number == 1
                indexes_tuple = (0,) * derivative_order

            values_per_point = np.zeros(num_points, dtype=float)
            errors_per_point = (
                np.zeros(num_points, dtype=float)
                if _richardson_extrapolation else None
            )

            for current_point_index in range(num_points):
                component_value, component_error = (
                    _evaluate_component_dispatcher(
                        user_function=function_to_differentiate,
                        point=_evaluated_points[current_point_index],
                        step_size=resolved_step_size,
                        point_number=_point_number,
                        indexes_tuple=indexes_tuple,
                        richardson_extrapolation=_richardson_extrapolation,
                        maximum_richardson_equations=(
                            _maximum_richardson_equations
                        )
                    )
                )

                values_per_point[current_point_index] = component_value
                if errors_per_point is not None:
                    errors_per_point[current_point_index] = component_error

            # return a float for a single evaluated
            if single_point_input:
                derivative_output = float(values_per_point[0])
                error_output = (
                    float(errors_per_point[0])
                    if errors_per_point is not None else None
                )

            else:
                derivative_output = values_per_point
                error_output = errors_per_point


        # Full-tensor, compute every index combination
        else:
            all_component_indexes = list(
                product(range(variable_number), repeat=derivative_order)
            )

            tensor_shape = (num_points,) + (variable_number,) * derivative_order

            values_tensor = np.zeros(tensor_shape, dtype=float)
            errors_tensor = (
                np.zeros(tensor_shape, dtype=float)
                if _richardson_extrapolation else None
            )

            for current_point_index in range(num_points):
                component_cache = {}

                for current_component_indexes in all_component_indexes:
                    canonical_key = tuple(
                        sorted(Counter(current_component_indexes).items())
                    )

                    cached = component_cache.get(canonical_key)

                    if cached is None:
                        component_value, component_error = (
                            _evaluate_component_dispatcher(
                                user_function=function_to_differentiate,
                                point=_evaluated_points[current_point_index],
                                step_size=resolved_step_size,
                                point_number=_point_number,
                                indexes_tuple=current_component_indexes,
                                richardson_extrapolation=(
                                    _richardson_extrapolation
                                ),
                                maximum_richardson_equations=(
                                    _maximum_richardson_equations
                                ),
                            )
                        )

                        component_cache[canonical_key] = (
                            component_value, component_error
                        )

                    else:
                        component_value, component_error = cached

                    current_tensor_index = (
                        (current_point_index, * current_component_indexes)
                    )

                    values_tensor[current_tensor_index] = component_value
                    if errors_tensor is not None:
                        errors_tensor[current_tensor_index] = component_error


            # Drop the leading M-axis for a single input point
            if single_point_input:
                derivative_output = values_tensor[0]
                error_output = (
                    errors_tensor[0] if errors_tensor is not None else None
                )

            else:
                derivative_output = values_tensor
                error_output = errors_tensor


        return DifferentiationResult(
            derivative=derivative_output,
            error_estimate=error_output,
            derivative_order=derivative_order,
            differentiation_method=differentiation_method,
            step_size=step_size,
            point_number=(
                None if differentiation_method == 'complex_step'
                else _point_number
            ),
            evaluation_points=_evaluated_points,
            single_component=_single_component_tuple,
            richardson_extrapolation=_richardson_extrapolation,
            maximum_richardson_equations=(
                _maximum_richardson_equations
                if _richardson_extrapolation else None
            ),
        )

    return nth_derivative_function



def _hessian_vector_product_central_difference(
    gradient_function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    unit_direction: np.ndarray,
    point_number: int,
    step_size_scalar: float,
) -> np.ndarray:
    """
    Computes H(point) @ unit_direction using a 1-D central-difference
    stencil applied to `gradient_function` along `unit_direction`.

    Parameters
    ----------
    gradient_function : Callable[[np.ndarray], np.ndarray]
        Gradient of the user's scalar function. Must accept a 1-D float
        array of shape (D,) and return a 1-D array-like of shape (D,).
    point : np.ndarray
        Real evaluation point, shape (D,).
    unit_direction : np.ndarray
        Direction along which the Hessian is applied, shape (D,). Must have
        unit Euclidean norm.
    point_number : int
        Base stencil point count. The effective point count is
        ***_effective_point_number***(1, `point_number`); the achieved
        accuracy along the directional axis is therefore
        acc = effective_point_number.
    step_size_scalar : float
        Step size h_t in t-space along `unit_direction`. The displacement in
        coordinate space at stencil offset k is k * h_t * unit_direction.

    Returns
    -------
    np.ndarray
        Shape (D,). Numerical approximation of H(point) @ unit_direction.

    Mathematical Description
    ------------------------
    The Hessian-vector product along a unit direction u at point x is the
    directional derivative of the gradient along u:

        (H(x) @ u)_i = sum_j  d^{2}f / dx_i dx_j  *  u_j
                     = d/dt [grad(x + t * u)_i] |_{t=0}

    Letting phi: R -> R^D:

        phi(t) := grad(x + t * u)

    a central-difference stencil of accuracy O(h_t^{acc}) for the first
    derivative gives:

        phi'(0) ≈ (1/h_t) *
                  sum_{(offset, c_offset)}  c_offset * phi(offset * h_t)
                = (1/h_t) *
                  sum_{(offset, c_offset)}  c_offset *
                                            grad(x + offset * h_t * u)

    which equals H(x) @ u up to a truncation error of order h_t^{acc}.
    """

    stencil = _central_difference_stencil(
        derivative_order=1, point_number=point_number
    )

    hvp_accumulator = np.zeros_like(point, dtype=float)
    for offset, coefficient in stencil:
        # New arrey per call so the user gradient is free to store its
        # input.
        shifted_point = (
            point + (offset * step_size_scalar) * unit_direction
        )
        gradient_value = np.asarray(
            gradient_function(shifted_point), dtype=float
        )
        hvp_accumulator += coefficient * gradient_value


    return hvp_accumulator / step_size_scalar



def numerical_hessian_vector_product(
    gradient_function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    vector: np.ndarray,
    step_size: Union[Literal['auto'], float] = 'auto',
    point_number: Union[Literal['auto'], int] = 'auto',
) -> DifferentiationResult:
    """
    Computes the Hessian-vector product H(`point`) @ `vector` of a scalar
    function f, given its gradient `gradient_function`, using only the
    central finite-difference path, without ever forming the Hessian.

    The Hessian-vector product is recovered as the directional derivative
    of the gradient:

        H(x) @ v = d/dt [grad(x + t * v)] |_{t=0}.

    The directional derivative is approximated by a central-difference
    stencil applied to `gradient_function` along the direction `vector`.
    Truncation error is O(h^p), where p is the effective stencil point
    count along the directional axis. Costs p calls to `gradient_function`,
    independent of the dimension D of `point`.

    Parameters
    ----------
    gradient_function : Callable[[np.ndarray], np.ndarray]
        Gradient of the scalar function f. Must accept a 1-D NumPy array of
        shape (D,) and return a 1-D array-like of shape (D,). It must be once
        continuously differentiable in a neighbourhood of `point` along
        `vector` (so that f is twice continuously differentiable there).
    point : np.ndarray
        Evaluation point at which the Hessian-vector product is computed,
        shape (D,).
    vector : np.ndarray
        Vector to multiply the Hessian by, shape (D,).
    step_size : 'auto' or float, optional
        Step size in t-space along the unit direction `vector` /
        ||`vector`||.

        - 'auto': computed by ***_auto_step_size*** with `variable_number`
          = 1 and per-coordinate derivative order 1, balancing truncation
          error O(h^acc) against roundoff O(eps / h).
        - float: a strictly positive finite scalar used directly.

        Default is 'auto'.
    point_number : 'auto' or int, optional
        Base number of stencil points used for the directional first
        derivative. Must satisfy `point_number` >= 2. When `point_number`
        is odd the effective point count is automatically promoted to
        `point_number` + 1 to satisfy the central-difference parity
        constraint (see ***_central_difference_coefficients***).

        When 'auto', `point_number` = 4 (= 1 + 3).

        Default is 'auto'.

    Returns
    -------
    DifferentiationResult
        A ***DifferentiationResult*** carrying:

        - `derivative`: np.ndarray of shape (D,), the HVP H(`point`) @
          `vector`.
        - `error_estimate`: always None.
        - `derivative_order`: 2 (the HVP is a second-order quantity).
        - `differentiation_method`: 'central_difference'.
        - `step_size`, `point_number`, `evaluation_points`,
          `single_component` (always None for HVP),
          `richardson_extrapolation` (always False),
          `maximum_richardson_equations` (always None): configuration data.

    Raises
    ------
    ValueError
        - `gradient_function` is not callable.
        - `point` cannot be interpreted as a 1-D float array.
        - `vector` does not have the same shape as `point`.
        - `step_size` is a string other than 'auto', or numeric but
          non-positive or non-finite.
        - `point_number` is not 'auto' and is not an integer >= 2.

    Notes
    -----
    - The central-difference stencil costs p_eff calls to `gradient_function`,
    where p_eff is the effective point count (`point_number` or
    `point_number` + 1), independently of D.

    - A scaling is performed dividing by ||`vector`||, it is mathematically
    equivalent to using `vector` directly as the perturbation axis and dividing
    by an effective step h_t * ||`vector`||. Doing it explicitly avoids
    step-size numerical problems when ||`vector`|| is very small or very large
    compared to 1.

    - Compared to forming the full Hessian. The full Hessian via
    ***nth_numerical_derivative***(f, 2) evaluates O(D^2) tensor
    components, each costing one full second-derivative stencil. The HVP
    here costs p_eff gradient calls regardless of D, which is the standard
    win whenever a callable for grad f is available and only the action of
    H on one (or a few) vectors is required (e.g. truncated Newton, CG
    inside trust-region steps, Lanczos for spectral estimates).

    See ***nth_numerical_derivative*** Notes for more information about some
    numerical considerations.

    Examples
    --------
    Hessian-vector product of a quadratic. Since the gradient of
    f(x) = 0.5 * x^T A x is grad(x) = A @ x, the HVP equals A @ v at every
    point:

    >>> import numpy as np
    >>> from ripples import numerical_hessian_vector_product
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> def gradient_quadratic(x):
    ...     return A @ x
    >>> result = numerical_hessian_vector_product(
    ...     gradient_quadratic,
    ...     point=np.array([1.0, -2.0]),
    ...     vector=np.array([1.0, 1.0]),
    ... )
    >>> result.derivative
    array([5., 4.])
    >>> result.differentiation_method
    'central_difference'
    >>> result.error_estimate is None
    True
    >>> result.derivative_order
    2
    >>> result.shape
    (2,)
    >>> result.has_error_estimate
    False

    Provide an explicit step size. Useful when the user has prior
    knowledge of the natural t-scale of the gradient along `vector`:

    >>> result = numerical_hessian_vector_product(
    ...     gradient_quadratic,
    ...     point=np.array([1.0, -2.0]),
    ...     vector=np.array([1.0, 1.0]),
    ...     step_size=1e-5,
    ... )
    >>> result.derivative
    array([5., 4.])
    >>> result.step_size
    1e-05

    Override `point_number` to demand a wider directional stencil. The
    accuracy along the directional axis is then O(h^p_eff), where p_eff
    is the effective point count:

    >>> result = numerical_hessian_vector_product(
    ...     gradient_quadratic,
    ...     point=np.array([1.0, -2.0]),
    ...     vector=np.array([1.0, 1.0]),
    ...     point_number=4,
    ... )
    >>> result.point_number
    4
    >>> result.derivative
    array([5., 4.])

    Pass a non-trivial gradient. Below, f(x) = sin(x_0) + x_0 * x_1^{2},
    so grad f = (cos(x_0) + x_1^{2}, 2 * x_0 * x_1) and the Hessian is
    [[-sin(x_0), 2 * x_1], [2 * x_1, 2 * x_0]]. At x = (1, 2) and
    v = (1, 0), H @ v = (-sin(1), 4):

    >>> def gradient_nonlinear(x):
    ...     return np.array([np.cos(x[0]) + x[1] ** 2, 2.0 * x[0] * x[1]])
    >>> result = numerical_hessian_vector_product(
    ...     gradient_nonlinear,
    ...     point=np.array([1.0, 2.0]),
    ...     vector=np.array([1.0, 0.0]),
    ... )
    >>> bool(np.allclose(result.derivative, [-np.sin(1.0), 4.0]))
    True

    The zero vector short-circuits to H @ 0 = 0, with no call to
    `gradient_function`:

    >>> def noisy_gradient(x):
    ...     raise AssertionError("must not be called for the zero vector")
    >>> result = numerical_hessian_vector_product(
    ...     noisy_gradient,
    ...     point=np.array([1.0, -2.0]),
    ...     vector=np.zeros(2),
    ... )
    >>> result.derivative
    array([0., 0.])

    References
    ----------
    [1] Randall J. LeVeque (2007).
        **Finite Difference Methods for Ordinary and Partial Differential
        Equations: Steady-State and Time-Dependent Problems.**
        SIAM (Society for Industrial and Applied Mathematics).
        Chapter 1.
        https://doi.org/10.1137/1.9780898717839

    [2] William H. Press, Saul A. Teukolsky, William T. Vetterling &
        Brian P. Flannery (2007).
        **Numerical Recipes: The Art of Scientific Computing, 3rd edition.**
        Cambridge University Press.
        Section 5.7.
        ISBN: 978-0-521-88068-8.
    """

    if not callable(gradient_function):
        raise ValueError(
            f"gradient_function must be callable. Got "
            f"{type(gradient_function).__name__}."
        )


    try:
        point_array = np.asarray(point, dtype=float)

    except (TypeError, ValueError) as conversion_error:
        raise ValueError(
            f"Provided point must be convertible to a 1-D float array. Got "
            f"{point!r}."
        ) from conversion_error

    if point_array.ndim != 1:
        raise ValueError(
            f"Provided point must be 1-D. Got {point_array.ndim}-D array of "
            f"shape {point_array.shape}."
        )

    variable_number = point_array.size


    try:
        vector_array = np.asarray(vector, dtype=float)

    except (TypeError, ValueError) as conversion_error:
        raise ValueError(
            f"Provided vector must be convertible to a 1-D float array. Got "
            f"{vector!r}."
        ) from conversion_error

    if vector_array.shape != point_array.shape:
        raise ValueError(
            f"Provided vector must have the same shape as point. Got vector "
            f"shape {vector_array.shape} and point shape "
            f"{point_array.shape}."
        )


    if isinstance(step_size, bool):
        raise ValueError(
            f"step_size must be 'auto' or a positive finite number. "
            f"Got {step_size!r}."
        )

    if isinstance(step_size, str):
        if step_size != 'auto':
            raise ValueError(
                f"Provided step_size is a string but is not 'auto'. "
                f"Got {step_size!r}."
            )

    elif isinstance(step_size, (int, float)):
        if step_size <= 0 or not np.isfinite(step_size):
            raise ValueError(
                f"Provided step_size must be a strictly positive finite number "
                f"when numeric. Got {step_size}."
            )

    else:
        raise ValueError(
            f"Provided step_size must be 'auto' or a positive finite number. "
            f"Got {step_size!r}."
        )


    if isinstance(point_number, str):
        if point_number != 'auto':
            raise ValueError(
                f"point_number string must be 'auto'. Got "
                f"{point_number!r}."
            )

        _point_number = 5

    elif isinstance(point_number, int) and not isinstance(
        point_number, bool
    ):
        if point_number < 2:
            raise ValueError(
                f"point_number must be >= 2 (the directional first "
                f"derivative requires at least two stencil points). "
                f"Got {point_number}."
            )
        _point_number = point_number

    else:
        raise ValueError(
            f"point_number must be an integer >= 2 or 'auto'. Got "
            f"{point_number!r}."
        )



    # H @ 0 = 0
    vector_norm = float(np.linalg.norm(vector_array))
    if vector_norm == 0.0:
        zero_hvp = np.zeros(variable_number, dtype=float)

        return DifferentiationResult(
            derivative=zero_hvp,
            error_estimate=None,
            derivative_order=2,
            differentiation_method='central_difference',
            step_size=step_size,
            point_number=_point_number,
            evaluation_points=point_array,
            single_component=None,
            richardson_extrapolation=False,
            maximum_richardson_equations=None,
        )


    # Working along the unit vector u = v / ||v|| makes the t-space step
    # h_t directly comparable to the standard 1-D step. The result is
    # scaled back by ||v|| at the end via H @ v = ||v|| * (H @ u).
    unit_direction = vector_array / vector_norm



    # Treat the t-axis as a single coordinate of derivative order 1. This
    # reuses _auto_step_size unchanged on the 'auto' path.
    if isinstance(step_size, str): # 'auto' is the only remaining case
        step_size_array_aux = _auto_step_size(
            variable_number=1,
            indexes_counter_tuple=((0, 1),),
            point_number=_point_number,
        )
        base_step_size_scalar = float(step_size_array_aux[0])

    else:
        base_step_size_scalar = float(step_size)



    # compute H @ u along the directional axis
    hvp_estimate_in_unit_frame = (
        _hessian_vector_product_central_difference(
            gradient_function=gradient_function,
            point=point_array,
            unit_direction=unit_direction,
            point_number=_point_number,
            step_size_scalar=base_step_size_scalar,
        )
    )


    # scale H @ u back to H @ v
    # H @ v = ||v|| * (H @ u), so the value scales linearly in ||v||.
    hvp_value = hvp_estimate_in_unit_frame * vector_norm



    return DifferentiationResult(
        derivative=hvp_value,
        error_estimate=None,
        derivative_order=2,
        differentiation_method='central_difference',
        step_size=step_size,
        point_number=_point_number,
        evaluation_points=point_array,
        single_component=None,
        richardson_extrapolation=False,
        maximum_richardson_equations=None,
    )
