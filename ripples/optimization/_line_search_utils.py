"""
Utilities for the line-search algorithms.

Contains
--------
_cubic_polynomial_minimizer
    Finds the minimizer of cubic polynomial interpolating three points
    alongside the derivative of the first: (a, fa, fpa), (b, fb), (c, fc).

_quadratic_polynomial_minimizer
    Finds the minimizer of quadratic polynomial interpolating two points
    alongside the derivative of the first: (a, fa, fpa), (b, fb).

_validate_line_search_parameters
    Validates the step-size and Wolfe-condition parameters shared by
    every optimizer in the _line_search_optimization.py module.

_print_line_search_information
    Prints a single iteration's status (cost, gradient norm, step size)
    to stdout.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Optional



def _cubic_polynomial_minimizer(
    a: float, fa: float, fpa: float,
    b: float, fb: float,
    c: float, fc: float
) -> Optional[float]:
    """
    Finds the minimizer of cubic polynomial interpolating three points
    alongside the derivative of the first: (a, fa, fpa), (b, fb), (c, fc).

    Mathematical Description
    ------------------------
    Starting with the third-order polynomial equation:

        p(x) = D + C*(x) + A*(x)^2 + B*(x)^3

    First, make it relative to the point a because it will be the starting
    point to obtain the polynomial coefficients:

        p(x) = D + C*(x-a) + A*(x-a)^2 + B*(x-a)^3

    Second, impose that p(a) = fa and p'(a) = fpa:

        p(x) = fa + fpa*(x-a) + A*(x-a)^2 + B*(x-a)^3

    From p(b) = fb, p(c) = fc and making db = b - a, dc = c - a:

        fb - fa - fpa*db = A*db^2 + B*db^3
        fc - fa - fpa*dc = A*dc^2 + B*dc^3

    Using the last 2 equations we can form a 2x2 linear system:

        [db^2 db^3] * [A] = [rhs_b]
        [dc^2 dc^3]   [B] = [rhs_c]

    where:
    - rhs_b = fb - fa - fpa*db
    - rhs_c = fc - fa - fpa*dc

    Then using Cramer's rule the system can be solved:

        A = det([[rhs_b, db^3],[rhs_c, dc^3]])/lhs_determinant
        B = det([[db^2, rhs_b],[dc^2, rhs_c]])/lhs_determinant

    with lhs_determinant being the determinant of the matrix on the left-hand
    side of the 2x2 system:

        db^2*dc^3 - dc^2*db^3 = (db*dc)^2*(dc-db)

    Lastly, as the polynomial is defined, it is needed to find minimum using the
    derivative of p(x):

        p'(x) = fpa + 2*A(x - a) + 3*B*(x - a)^2

    with s = x - a and imposing p'(x) = 0:

        3*B*s^2 + 2*A*s + fpa = 0

    Applying the quadratic formula to the last equation:

        s = (-2*A +- sqrt(4*A^2 - 12*B*fpa))/6*B

    Factoring gives:

        s = (-A +- sqrt(A^2 - 3*B*fpa))/3*B

    which is the result and output of this algorithm (using the positive square
    root which leads to descent)

    References
    ----------
    [1] Pages 57-59 from Nocedal, J., & Wright, S. J. (2006).
        *Numerical Optimization* (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] SciPy contributors (2026).
        SciPy.optimize._linesearch.py source code (Version 1.17.1).
        GitHub
        https://github.com/scipy/scipy/blob/bb6b9da396f15355efdb2e28bdfa1aead105ce92/scipy/optimize/_linesearch.py
    """

    # Suppress all numerical warnings, which are really common in this
    # function due to its nature
    with np.errstate(all='ignore'):
        try:
            db = b - a
            dc = c - a

            # Determinant check (points do not coincide or are collinear)
            left_side_determinant = (db * dc) ** 2 * (dc - db)

            if left_side_determinant == 0.0:
                return None

            rhs_b = fb - fa - fpa * db
            rhs_c = fc - fa - fpa * dc

            A = (dc**3 * rhs_b - db**3 * rhs_c) / left_side_determinant
            B = (-dc**2 * rhs_b + db**2 * rhs_c) / left_side_determinant

            if B == 0.:
                return None

            radical = A*A - 3.0*B*fpa

            if radical < 0.:
                return None

            function_mininum = a + (-A + np.sqrt(radical)) / (3.0 * B)

            if np.isfinite(function_mininum):
                return function_mininum
            else:
                return None

        except (ZeroDivisionError, ValueError, OverflowError,
                FloatingPointError):
            return None



def _quadratic_polynomial_minimizer(
    a: float, fa: float, fpa: float,
    b: float, fb: float
) -> Optional[float]:
    """
    Finds the minimizer of quadratic polynomial interpolating two points
    alongside the derivative of the first: (a, fa, fpa), (b, fb).

    Mathematical Description
    ------------------------
    Beginning with the quadratic polynomial:

        q(x) = A + B*(x) + C*(x)^2

    First, make it relative to the point a, because it will be the
    starting point to obtain the polynomial coefficients:

        q(x) = A + B*(x - a) + C*(x - a)^2

    Second, impose that q(a) = fa and q'(a) = fpa:

        q(x) = fa + fpa*(x - a) + C*(x - a)^2

    Now let D = b - a, obtaining:

        q(b) = fa + fpa*D + C*D^2 = f(b)

    Solving for C gives:

        C = (fb - fa - fpa*D)/D^2

    Now that the polynomial is defined, find the minimum by setting
    the derivative to zero:

        q'(x) = fpa + 2*C*(x-a) = 0

    obtaining:

        x = a - fpa/(2*C)

    References
    ----------
    [1] Pages 57-59 from Nocedal, J., & Wright, S. J. (2006).
        *Numerical Optimization* (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] SciPy contributors (2026).
        SciPy.optimize._linesearch.py source code (Version 1.17.1).
        GitHub
        https://github.com/scipy/scipy/blob/bb6b9da396f15355efdb2e28bdfa1aead105ce92/scipy/optimize/_linesearch.py
    """

    # Suppress all numerical warnings, which are really common in this
    # function due to its nature
    with np.errstate(all='ignore'):
        try:
            D = b - a
            if D == 0.0:
                return None

            C = (fb - fa - fpa * D) / (D * D)
            if C <= 0.0: # Ensure positive curvature (minimum exists)
                return None

            function_minimum = a - fpa / (2.0 * C)

            if np.isfinite(function_minimum):
                return function_minimum
            else:
                return None

        except (ZeroDivisionError, ValueError, OverflowError,
                FloatingPointError):
            return None



def _validate_line_search_parameters(
    initial_alpha,
    max_alpha,
    bracketing_max_iter,
    wolfe_c1,
    wolfe_c2,
    zoom_max_iter,
    zoom_epsilon
) -> None:
    """
    Validate line search parameters for Wolfe line search algorithms.

    Parameters
    ----------
    initial_alpha : float
        Initial trial step length.
    max_alpha : float
        Maximum allowable step length.
    bracketing_max_iter : int
        Maximum iteration number for bracketing.
    wolfe_c1 : float
        Armijo (sufficient decrease) constant.
    wolfe_c2 : float
        Curvature constant.
    zoom_max_iter : int
        Maximum iteration number for zoom.
    zoom_epsilon : float
        Tolerance used in zoom phase (interval convergence threshold).

    Raises
    ------
    TypeError
        If any parameter has an incorrect type.
    ValueError
        If any parameter has an out-of-range or inconsistent value.
    """

    for var_name, var_value in {
        'initial_alpha': initial_alpha,
        'max_alpha': max_alpha,
        'wolfe_c1': wolfe_c1,
        'wolfe_c2': wolfe_c2,
        'zoom_epsilon': zoom_epsilon,
    }.items():

        if not isinstance(var_value, (int, float)):
            raise TypeError(f"'{var_name}' must be a real scalar.")

        if not np.isfinite(var_value):
            if var_name != 'max_alpha' or not (var_value == np.inf):
                raise ValueError(f"'{var_name}' must be finite "
                                 f"(not -inf or NaN).")

    for var_name, var_value in {
        'bracketing_max_iter': bracketing_max_iter,
        'zoom_max_iter': zoom_max_iter,
    }.items():

        # As bool is a subclass of int it must be excluded explicitly
        if not isinstance(var_value, int) or isinstance(var_value, bool):
            raise TypeError(f"'{var_name}' must be an integer.")

        if var_value <= 0:
            raise ValueError(f"'{var_name}' must be strictly positive.")


    # Step length checks
    if initial_alpha <= 0:
        raise ValueError("'initial_alpha' must be strictly positive.")

    if max_alpha <= 0:
        raise ValueError("max_alpha must be strictly positive.")

    if initial_alpha > max_alpha:
        raise ValueError("'initial_alpha' cannot be greater than 'max_alpha'.")

    # Wolfe parameters
    if not (0 < wolfe_c1 < wolfe_c2 < 1):
        raise ValueError("'wolfe_c1' and 'wolfe_c2' must satisfy "
                         "0 < c1 < c2 < 1.")

    if zoom_epsilon < 0:
        raise ValueError("'zoom_epsilon' must be non-negative.")

    # Prevent infinite zoom loops
    if zoom_epsilon >= max_alpha:
        raise ValueError("'zoom_epsilon' must be smaller than 'max_alpha'.")



def _print_line_search_information(
    current_iter, current_cost, grad_norm, alpha, verbose_freq, max_iters,
    converged: bool = False,
) -> None:
    """
    Prints information about the current iteration status (cost, gradient
    norm, and step size) to stdout.

    At the first iteration, a column header is printed before the row.
    Subsequent iterations are printed when any of the following holds:
    `converged` is True, the iteration index is a multiple of
    `verbose_freq`, or the iteration index equals `max_iters`. The
    `converged` flag guarantees that the iteration at which the optimizer
    declared convergence is shown to the user.

    Parameters
    ----------
    current_iter : int
        The current iteration number.
    current_cost : float
        The current cost value.
    grad_norm : float
        The current gradient norm.
    alpha : Optional[float]
        The step size taken at the current iteration, or None if no step
        size is available (it will be rendered as '-').
    verbose_freq : int
        Print every `verbose_freq` iterations.
    max_iters : int
        Maximum iterations allowed; the final iteration is always printed.
    converged : bool, default=False
        If True, force the row to be printed at the convergent iteration
        regardless of `verbose_freq`.
    """

    # Format the step-size column once. None is rendered as a right-aligned
    # '-' so the column still aligns with the formatted numeric rows.
    alpha_str = '-' if alpha is None else f'{alpha:.2e}'

    if current_iter == 1:
        print(f"{'Iter':>5s} | {'Cost':>12s} | {'Grad Norm':>12s} "
              f"| {'Step Size':>12s}")
        print("-" * 50)
        print(f"{1:5d} | {current_cost:12.6f} | {grad_norm:12.2e} "
              f"| {alpha_str:>12s}")

    elif (converged
          or current_iter % verbose_freq == 0
          or current_iter == max_iters):
        print(f"{current_iter:5d} | {current_cost:12.6f} | "
              f"{grad_norm:12.2e} "
              f"| {alpha_str:>12s}")
