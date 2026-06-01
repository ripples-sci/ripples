"""
Utilities for the trust-region algorithms.

Contains
--------
_validate_trust_region_parameters
    Validates the trust-region radius update parameters, tolerances and
    inner-solver iteration limits shared by every optimizer in the
    _trust_region_optimization.py module.

_compute_step_to_boundary
    Computes the scalar tau such that the step p + tau * d lies exactly on the
    trust-region boundary ||p + tau * d|| = delta.

_solve_secular_equation_newton
    Solves the secular equation associated with the trust-region subproblem
    using Newton's method in the Lanczos / GLTR reduced space.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np

from ._utils import FLOAT_EPSILON



def _validate_trust_region_parameters(
    grad_rel_tol,
    abs_tol,
    eta,
    rho_make_smaller_threshold,
    rho_make_bigger_threshold,
    make_trust_radius_smaller_multiplier,
    make_trust_radius_bigger_multiplier,
    initial_trust_radius,
    max_trust_radius,
    inner_max_iter,
) -> None:
    """
    Validates parameters for the Trust-Region optimization algorithm.

    Parameters
    ----------
    grad_rel_tol : float
        Relative gradient norm tolerance for convergence.
    abs_tol : float
        Absolute tolerance for convergence.
    eta : float
        Acceptance threshold for trial step (minimum ratio rho to accept step).
    rho_make_smaller_threshold : float
        Threshold below which trust radius is reduced.
    rho_make_bigger_threshold : float
        Threshold above which trust radius is increased.
    make_trust_radius_smaller_multiplier : float
        Multiplicative factor (<1) to shrink trust radius.
    make_trust_radius_bigger_multiplier : float
        Multiplicative factor (>1) to expand trust radius.
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum allowable trust-region radius.
    inner_max_iter : int
        Maximum number of inner subproblem iterations.

    Raises
    ------
    TypeError
        If any parameter has an incorrect type.
    ValueError
        If any parameter has an out-of-range or inconsistent value.
    """

    # Floats validation
    for var_name, var_value in {
        'grad_rel_tol': grad_rel_tol,
        'abs_tol': abs_tol,
        'eta': eta,
        'rho_make_smaller_threshold': rho_make_smaller_threshold,
        'rho_make_bigger_threshold': rho_make_bigger_threshold,
        'make_trust_radius_smaller_multiplier':
            make_trust_radius_smaller_multiplier,
        'make_trust_radius_bigger_multiplier':
            make_trust_radius_bigger_multiplier,
        'initial_trust_radius': initial_trust_radius,
        'max_trust_radius': max_trust_radius,
    }.items():

        if not isinstance(var_value, (int, float)):
            raise TypeError(f"'{var_name}' must be a real scalar.")

        if not np.isfinite(var_value):
            if not var_name == 'max_trust_radius':
                raise ValueError(f"'{var_name}' must be finite "
                                 f"(not inf or NaN).")

    # Integer validation
    if not isinstance(inner_max_iter, int) and inner_max_iter != 'auto':
        raise TypeError("'inner_max_iter' must be an integer or 'auto.")
    if isinstance(inner_max_iter, int) and inner_max_iter <= 0:
        raise ValueError("'inner_max_iter' must be strictly positive.")

    # Tolerances checks
    if grad_rel_tol < 0:
        raise ValueError("'grad_rel_tol' must be non-negative.")
    if abs_tol < 0:
        raise ValueError("'abs_tol' must be non-negative.")
    if grad_rel_tol == 0 and abs_tol == 0:
        raise ValueError("At least one of 'grad_rel_tol' or 'abs_tol' "
                         "must be positive.")

    # Acceptance parameter eta
    if not (0 <= eta < rho_make_smaller_threshold):
        raise ValueError(f"'eta' must satisfy "
            f"0 <= eta < {rho_make_smaller_threshold} "
            f"('rho_make_smaller_threshold).")

    # Rho thresholds
    if not (0 < rho_make_smaller_threshold <
            rho_make_bigger_threshold < 1):
        raise ValueError(
            "'rho_make_smaller_threshold' and "
            "'rho_make_bigger_threshold' must satisfy "
            "0 < rho_small < rho_big < 1."
        )

    # Radius update multipliers
    if not (0 < make_trust_radius_smaller_multiplier < 1):
        raise ValueError(
            "'make_trust_radius_smaller_multiplier' "
            "must satisfy 0 < multiplier < 1."
        )
    if not (make_trust_radius_bigger_multiplier > 1):
        raise ValueError(
            "'make_trust_radius_bigger_multiplier' "
            "must be strictly greater than 1."
        )

    # Trust-region radius checks
    if initial_trust_radius <= 0:
        raise ValueError("'initial_trust_radius' must be strictly positive.")
    if max_trust_radius <= 0:
        raise ValueError("'max_trust_radius' must be strictly positive.")
    if initial_trust_radius > max_trust_radius:
        raise ValueError("'initial_trust_radius' cannot exceed "
                         "'max_trust_radius'.")



def _compute_step_to_boundary(
    step_dot_searchdir: float,
    searchdir_dot_searchdir: float,
    step_norm_sq: float,
    trust_radius_sq: float,
) -> float:
    """
    Computes tau > 0 such that:

        ||step_direction + tau * search_direction|| = trust_radius.

    Solves the second-order polynomial (expanding the above equation) for tau:

        a * tau^2 + b * tau + c = 0

    where:

        a = ||search_direction||^2 > 0
        b = 2 * (step_direction^T * search_direction)
        c = ||step_direction||^2 - trust_radius^2  <= 0

    as a is the square of a vector norm_2 it is always positive, and given that
    the condition to call this function is that the step_direction is greater
    than the radius (c<0) or exactly equal (c=0).

    Because a > 0 and c <= 0 the discriminant (b^2 - 4.0 * a * c) is
    non-negative and the two roots have opposite signs, the unique non-negative
    root is returned.

    Parameters
    ----------
    step_dot_searchdir : float
        The dot product `step_direction`^T * `search_direction` at the
        current CG iterate.
    searchdir_dot_searchdir : float
        The squared norm of `search_direction` at the current CG
        iterate.
    step_norm_sq : float
        The squared norm of the current CG iterate `step_direction`.
    trust_radius_sq : float
        Squared trust-region radius.

    Returns
    -------
    float
        The non-negative `tau` that places the new iterate exactly on
        the trust-region boundary along `search_direction`.
    """

    # a = ||search_direction||^2
    a = searchdir_dot_searchdir
    # b = 2 * (step_direction^T * search_direction)
    b = 2.0 * step_dot_searchdir
    # c = ||step_direction||^2 - radius^2
    c = step_norm_sq - trust_radius_sq

    discriminant = b * b - 4.0 * a * c
    # Numerical safety although mathematically guaranteed non negative
    if discriminant < 0:
        return 0.0

    return (-b + np.sqrt(discriminant)) / (2.0 * a)



def _solve_secular_equation_newton(
    eigvals: np.ndarray,
    g_norm__Eigvecs_first_component: np.ndarray,
    delta: float
) -> float:
    """
    Solves root of phi(lambda) = 1/||y_sol(lambda)|| - 1/delta = 0.

    Mathematical Description
    ------------------------
    As the goal is to find lambda >= 0 such that phi(lambda) = 0, and it has
    the following properties:

    - phi(lambda) is strictly increasing for all valid lambda,
    - phi(lambda) is smooth and convex,
    - phi(lambda) -> -inf as lambda approaches a pole,
    - phi(lambda) -> 1/delta as lambda -> +inf.

    then phi(lambda) is particularly well-suited for Newton's method.

    ---

    As shown in above[2, Theorem 4.1], the matrix T_k + lambda * I must be
    positive definite in order for y(lambda) to be well-defined, this requires:

        lambda >= max(0, -min_i eig_i)

    where eig_i are the eigenvalues of T_k; this value defines a strict lower
    bound for lambda.

    ---

    At each Newton iteration, compute the squared norm of y_sol(lambda), which
    expression is derived in the function that solves the subproblem
    (_solve_trust_region_subproblem_lanczos):

        ||y_sol(lambda)||^2 =
            sum_{i=1}^k  (||g|| * Eigvecs[1,i]/(eig_i + lambda))^2

    and:

        phi(lambda) = 1 / ||y_sol(lambda)|| - 1 / delta

    which derivative is:

        phi'(lambda) = -1/2 * (||y_sol(lambda)||^2^(-3/2) *
                                d/dlambda (||y_sol(lambda)||^2)

    where

        d/dlambda (||y_sol(lambda)||^2) =
            sum_{i=1}^k  ( -2 * (||g|| * Eigvecs[1,i])^2 / (eig_i + lambda)^3 )

    The Newton's is then:

    start with lambda_0 = max(0, -min_i eig_i) and

    for iteration i = 1, 2, ...

        lambda_{i+1} = lambda_i - phi(lambda_i) / phi'(lambda_i)

    which iterates until phi is below a certain tolerance.

    References
    ----------
    [1] Gould, N. I. M., Lucidi, S., Roma, M., & Toint, P. L. (1999).
        **Solving the trust-region subproblem using the Lanczos method**.
        SIAM Journal on Optimization, 9(2), 504-525.
        https://doi.org/10.1137/S1052623497322735

    [2] Pages 175-176 & Theorem 4.1 & Chapter 4.3 from
        Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5
    """

    min_eig = eigvals[0]

    # Ensure T_k + lambda * I is positive definite [2, Theorem 4.1].
    lower_bound = max(0.0, -min_eig)

    lmbda = lower_bound + FLOAT_EPSILON

    for _ in range(30):

        denom = np.maximum(eigvals + lmbda, FLOAT_EPSILON)

        # ||y_sol(lambda)||^2
        term_sq = (g_norm__Eigvecs_first_component / denom)**2
        norm_y_sq = float(np.sum(term_sq))

        # If g is orthogonal to every eigenvector of T_k then every numerator
        # is zero, norm_y_sq is zero, and the Newton step is undefined. In that
        # degenerate situation lambda is already as small as it can be.
        if norm_y_sq <= 0.0:
            break

        norm_y = np.sqrt(norm_y_sq)

        # phi(lambda)
        phi = 1.0 / norm_y - 1.0 / delta

        if abs(phi) < FLOAT_EPSILON:
            break

        # phi'(lambda)
        norm_y_primer_sq = -2.0 * np.sum(term_sq / denom)
        phi_prime = -0.5 * (norm_y_sq ** -1.5) * norm_y_primer_sq

        # phi is strictly increasing and convex on (lower_bound, +inf),
        # so phi_prime > 0 there. A non-positive value indicates that
        # norm_y_sq is slipping through roundoff noise; so lambda is returned
        # rather than divide by ~0 and make result worse.
        if phi_prime <= 0.0:
            return lmbda

        # Newton Step
        lmbda -= phi / phi_prime
        if lmbda < lower_bound:
            lmbda = lower_bound + FLOAT_EPSILON

    return lmbda
