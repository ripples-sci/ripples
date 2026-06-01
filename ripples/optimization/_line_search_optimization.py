"""
Line-search algorithms for local optimization.

All algorithms implemented in this module share the same outer loop: at
each iteration k they compute a descent direction p_k from gradient (and
possibly curvature) information, then accept a step size alpha_k > 0
along that direction by enforcing the Wolfe conditions:

    Armijo:    f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * g_k^T * p_k
    Curvature: |g(x_k + alpha * p_k)^T * p_k| <= c2 * |g_k^T * p_k|

with 0 < c1 < c2 < 1, before updating x_{k+1} = x_k + alpha_k * p_k. The
algorithms differ in how p_k is constructed: from the gradient alone
(non-linear CG), from finite-difference Hessian-vector products solved
through an inner CG (Newton-CG), or from an approximation of the inverse
Hessian built up across iterations (BFGS, L-BFGS).

Contains
--------
_line_search_wolfe
    Bracketing phase of the strong-Wolfe line search. Expands the trial
    step until an alpha satisfying both Wolfe conditions is bracketed,
    then delegates the refinement to _zoom.

_zoom
    Zoom phase of the strong-Wolfe line search. Refines a bracket
    [alpha_lo, alpha_hi] using cubic / quadratic interpolation with a
    bisection fallback until a step size satisfying both Wolfe
    conditions is found.

_conjugate_gradient_optimizer
    Non-linear Conjugate Gradient (Polak-Ribière+) optimizer with Wolfe
    line search.

_solve_newton_system_cg
    Truncated Conjugate Gradient inner solver for the Newton system
    H * p = -g. Used by _newton_cg_optimizer.

_newton_cg_optimizer
    Truncated Newton optimizer that solves the Newton system via inner
    CG on Hessian-vector products and takes a Wolfe step along the
    result.

_bfgs_optimizer
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer. Maintains a dense
    inverse-Hessian approximation that is updated rank-2 every step.

_lbfgs_optimizer
    Limited-memory BFGS (L-BFGS) optimizer. Represents the
    inverse-Hessian approximation implicitly through the m most recent
    (s_k, y_k) pairs, bringing memory and per-step cost from O(N^2) down
    to O(m*N).
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Any, Optional, Union, Tuple, Dict, Callable, Literal
from collections import deque

from ._line_search_utils import (
    _cubic_polynomial_minimizer, _quadratic_polynomial_minimizer,
    _validate_line_search_parameters, _print_line_search_information
)
from ._utils import FLOAT_EPSILON, _initial_check, _check_termination

from ..differentiation import numerical_hessian_vector_product




def _line_search_wolfe(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    search_direction: np.ndarray,
    grad: np.ndarray,
    cost_k: float,
    c1: float,
    c2: float,
    initial_alpha: float,
    max_alpha: float,
    max_iter: int,
    zoom_max_iter: int,
    zoom_epsilon: float
) -> Tuple[float, float]:
    """
    Finds a step size alpha satisfying the Wolfe conditions using a safeguarded
    line search algorithm.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function f(x).
    grad_func : Callable[[np.ndarray], np.ndarray]
        The gradient function g(x) = f'(x).
    initial_params : np.ndarray
        The current parameter vector x_k.
    search_direction : np.ndarray
        The search direction vector p_k.
    grad : np.ndarray
        The current gradient vector g_k = g(x_k).
    cost_k : float
        The current cost f(x_k).
    c1 : float
        Parameter for the sufficient decrease condition (Armijo rule).
    c2 : float
        Parameter for the curvature condition (Wolfe).
    initial_alpha : float
        Initial guess for the step size.
    max_alpha : float
        Maximum allowable step size.
    max_iter : int
        Maximum number of iterations for the outer line search loop.
    zoom_max_iter : int
        Maximum number of iterations for the inner zoom phase.
    zoom_epsilon : float
        Tolerance for the zoom phase interval width.

    Returns
    -------
    alpha : float
        The valid step size satisfying Wolfe conditions.
    phi_alpha : float
        The function value at the new point f(x_k + alpha * p_k).

    Mathematical Description
    ------------------------
    The goal of this line search is to find a scalar step size alpha > 0
    such that moving from x_k along the direction p_k leads to a
    stable and sufficient reduction in the objective function f(x).

    To achieve this, the following function phi is used:

        phi(alpha) = f(x_k + alpha * p_k)

    The derivative of this scalar function with respect to alpha is:

        phi'(alpha) = gradient(f(x_k + alpha * p_k))^T * p_k

    ---

    Simply picking an alpha that reduces the function (phi(alpha) < phi(0)) is
    not enough to guarantee global convergence, as the steps might become
    vanishingly small. It is needed to satisfy the "Wolfe Conditions":


    1. Sufficient Decrease (Armijo Rule): The step must decrease the function
    value proportional to the step size and the initial slope.

        phi(alpha) <= phi(0) + c1 * alpha * phi'(0)

    where:
    - phi(0) is the current cost f(x_k).
    - phi'(0) is the directional derivative (slope) at alpha=0.
    - c1 is a small constant (typically 1e-4), defining the slope of the
    acceptance line.

    The function phi(alpha) must lie below the linear function:

        L(alpha) = phi(0) + c1 * alpha * phi'(0)


    2. Curvature Condition (Wolfe): To not stop the search too short, it is
    required that the slope at the new point is closer to zero (flatter) than
    the initial slope. This implies that a stationary point (minimum) is close
    along the line.

        |phi'(alpha)| <= c2 * phi'(0)

    where
    -c2 is a constant (e.g., 0.9 for Quasi-Newton, 0.4 for Conjugate Gradient).

    ---

    This implementation works in two phases in order to find alpha:

    Phase 1 (Bracketing): Iterate through trial step sizes (multiplying by 2.
    every iteration). If a trial step size violates the Armijo rule (function
    increased or didn't drop enough) or has a positive slope, then a valid
    point exists between the previous trial and the current one. This defines a
    "bracket" [lo, hi].

    Phase 2 (Zoom): Once a bracket is identified, the _zoom function
    refines the interval (using interpolation or bisection) to find an
    alpha that satisfies both Wolfe conditions.

    References
    ----------
    [1] Algorithm 3.5 from Nocedal, J., & Wright, S. J. (2006).
        *Numerical Optimization* (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] SciPy contributors (2026).
        SciPy.optimize._linesearch.py source code (Version 1.17.1).
        GitHub
        https://github.com/scipy/scipy/blob/bb6b9da396f15355efdb2e28bdfa1aead105ce92/scipy/optimize/_linesearch.py
    """

    # evaluated_point is always initial_params + alpha * search_direction,
    # so the only variable changing across this algorithm is alpha, the
    # following functions are computed this way in order to only create one new
    # evaluated_point per iteration
    def alpha_to_evaluated_point(alpha: float) -> np.ndarray:
        return initial_params + alpha * search_direction

    def phi(evaluated_point: np.ndarray) -> float:
        return function(evaluated_point)

    def phi_prime(evaluated_point: np.ndarray) -> float:
        return float(np.dot(grad_func(evaluated_point), search_direction))



    phi_0 = cost_k
    phi_prime_0 = float(np.dot(grad, search_direction))

    if not np.isfinite(phi_0) or not np.isfinite(phi_prime_0):
        return 0.0, phi_0

    if phi_prime_0 >= 0.0: # check if descent direction
        return 0.0, phi_0

    # Initial bracketing
    alpha_prev = 0.0
    phi_prev = phi_0
    phi_prime_prev = phi_prime_0

    alpha = float(max(FLOAT_EPSILON, initial_alpha))
    evaluated_point = alpha_to_evaluated_point(alpha)
    phi_alpha = phi(evaluated_point)

    for current_iteration in range(max_iter):

        # Armijo or non-monotone (function value increased after expanding
        # alpha) bracket condition
        if ((phi_alpha > phi_0 + c1 * alpha * phi_prime_0) or
            (current_iteration > 0 and phi_alpha >= phi_prev)):
            return _zoom(
                phi=phi, phi_prime=phi_prime,
                alpha_to_evaluated_point=alpha_to_evaluated_point,
                alpha_lo=alpha_prev, phi_lo=phi_prev,
                phi_prime_lo=phi_prime_prev,
                alpha_hi=alpha, phi_hi=phi_alpha,
                phi_0=phi_0, phi_prime_0=phi_prime_0,
                c1=c1, c2=c2, max_iter=zoom_max_iter, epsilon=zoom_epsilon
            )


        phi_prime_alpha = phi_prime(evaluated_point)
        # Curvature condition: |phi'(alpha)| is low enough relative to
        # |phi'(0)|.
        if abs(phi_prime_alpha) <= -c2 * phi_prime_0:
            return alpha, phi_alpha


        # If slope turned positive, the minimum lies between
        # (alpha_prev, alpha)
        if phi_prime_alpha >= 0:
            return _zoom(
                phi=phi, phi_prime=phi_prime,
                alpha_to_evaluated_point=alpha_to_evaluated_point,
                alpha_lo=alpha, phi_lo=phi_alpha, phi_prime_lo=phi_prime_alpha,
                alpha_hi=alpha_prev, phi_hi=phi_prev,
                phi_0=phi_0, phi_prime_0=phi_prime_0,
                c1=c1, c2=c2, max_iter=zoom_max_iter, epsilon=zoom_epsilon
            )


        # Expansion if none of the 3 conditions met
        alpha_prev, phi_prev = alpha, phi_alpha
        alpha = min(alpha * 2.0, max_alpha)

        if alpha == alpha_prev: # Hit max_alpha
            return alpha, phi_alpha

        evaluated_point = alpha_to_evaluated_point(alpha)
        phi_alpha = phi(evaluated_point)
        phi_prime_prev = phi_prime_alpha


    # If bracketing loop did not return, give the last tried point
    return alpha, phi_alpha



def _zoom(
    phi: Callable,
    phi_prime: Callable,
    alpha_to_evaluated_point: Callable,
    alpha_lo: float,
    phi_lo: float,
    phi_prime_lo: float,
    alpha_hi: float,
    phi_hi: float,
    phi_0: float,
    phi_prime_0: float,
    c1: float,
    c2: float,
    max_iter: int,
    epsilon: float
) -> Tuple[float, float]:
    """
    Wolfe conditions enforced zoom: maintains [alpha_lo, alpha_hi] as a valid
    bracket and uses cubic -> quadratic -> bisection fallback.

    Inputs require Armijo condition violated or non-monotonicity triggered
    when entering zoom.

    Mathematical Description
    ------------------------
    The Zoom phase is entered when a "bracket" [alpha_lo, alpha_hi] has
    been found. This bracket guarantees that a step size satisfying the
    Wolfe conditions exists strictly within this interval.

    This algorithm generates a trial step size alpha_j inside the interval
    and refines the bracket based on the value and slope at alpha_j.

    ---

    The logic used in this code is the following:

    First a model of the objective function phi(alpha) is used to achieve a
    minimum inside the bracket using polynomial interpolation:

    - Cubic Interpolation is used first (using 3 points and the
    derivative of the first).
    - If that fails or is too close to the boundaries, try Quadratic
    Interpolation (using 2 points and the derivative of the first).
    - If that fails or is too close to the boundaries, fall back to Bisection
    (alpha_j = midpoint).

    The mathematical derivations for the Cubic and Quadratic minimizers
    are detailed within the inner functions _cubic_polynomial_minimizer and
    _quadratic_polynomial_minimizer respectively [2].


    Second, once a new trial point alpha_j is chosen, phi(alpha_j) is
    evaluated, and then one gets the following cases:

    - Case A, The Armijo condition is violated (function value too high):

        phi(alpha_j) > phi(0) + c1 * alpha_j * phi'(0)
        OR
        phi(alpha_j) >= phi(alpha_lo)

    If the function is higher at alpha_j than at alpha_lo, or insufficiently
    decreased, the valid minimum must lie to the left of alpha_j then the new
    bracket becomes [alpha_lo, alpha_j].

    - Case B, Armijo condition is met:

        phi(alpha_j) <= phi(0) + c1 * alpha_j * phi'(0)
        OR
        phi(alpha_j) < phi(alpha_lo)

    so check the slope (derivative) phi'(alpha_j):

    if:

        |phi'(alpha_j)| <= -c2 * phi'(0)

    then Both Sufficient Decrease and Curvature conditions are satisfied.
    alpha_j is returned.

    - Case B.1, if:

        phi'(alpha_j) * (alpha_hi - alpha_lo) >= 0

    The slope at alpha_j (phi'(alpha_j)) has the same sign as the bracket
    direction implies (alpha_hi - alpha_lo), meaning that a minimum was
    overshot:

    - Case B.1.1, if:

        alpha_hi - alpha_lo > 0
        and
        phi'(alpha_j) > 0

    then between alpha_lo and alpha_j there is a minimum.

    - Case B.1.2, if:

        alpha_hi - alpha_lo < 0
        and
        phi'(alpha_j) < 0

    then between alpha_lo and alpha_j there is a minimum.

    So, in both cases the bracket becomes [alpha_j, alpha_lo]; note that no
    assumption is made about which of (alpha_lo, alpha_hi) is numerically
    larger: alpha_lo is the lower-phi endpoint by convention, and
    phi'(alpha_j) * (alpha_hi - alpha_lo) >= 0 handles both orderings correctly.

    - Case B.2, if otherwise:

        phi'(alpha_j) * (alpha_hi - alpha_lo) < 0

    The function decreased enough, but the slope is still negative (the minimum
    has not been reached yet). The minimum lies between alpha_j and alpha_hi.
    So, the bracket becomes [alpha_j, alpha_hi].

    References
    ----------
    [1] Algorithm 3.6 from Nocedal, J., & Wright, S. J. (2006).
        *Numerical Optimization* (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] SciPy contributors (2026).
        SciPy.optimize._linesearch.py source code (Version 1.17.1).
        GitHub
        https://github.com/scipy/scipy/blob/bb6b9da396f15355efdb2e28bdfa1aead105ce92/scipy/optimize/_linesearch.py
    """

    # The bracket convention used here follows [1]: alpha_lo is the lower-phi
    # endpoint (not necessarily the lower alpha). The slope sign check below
    # uses (alpha_hi - alpha_lo) so the algorithm is correct regardless of
    # which endpoint has the larger alpha.

    # Interpolator results must stay far away from the bounds
    delta_cubic = 0.2
    delta_quad = 0.1

    alpha_aux, phi_aux = 0., phi_0

    for _ in range(max_iter):

        interval = alpha_hi - alpha_lo
        if abs(interval) <= epsilon:
            return alpha_lo, phi_lo

        alpha_j = None

        # Cubic
        alpha_j = _cubic_polynomial_minimizer(
            a=alpha_lo, fa=phi_lo, fpa=phi_prime_lo,
            b=alpha_hi, fb=phi_hi,
            c=alpha_aux, fc=phi_aux
        )
        if alpha_j is not None:
            margin = delta_cubic * abs(interval)
            min_bound = min(alpha_lo, alpha_hi) + margin
            max_bound = max(alpha_lo, alpha_hi) - margin
            if not (min_bound < alpha_j < max_bound):
                alpha_j = None

        # Quadratic
        if alpha_j is None:
            alpha_j = _quadratic_polynomial_minimizer(
                a=alpha_lo, fa=phi_lo, fpa=phi_prime_lo,
                b=alpha_hi, fb=phi_hi
            )
            if alpha_j is not None:
                margin = delta_quad * abs(interval)
                min_bound = min(alpha_lo, alpha_hi) + margin
                max_bound = max(alpha_lo, alpha_hi) - margin
                if not (min_bound < alpha_j < max_bound):
                    alpha_j = None

        # Bisection
        if alpha_j is None:
            alpha_j = alpha_lo + 0.5 * interval



        evaluated_point = alpha_to_evaluated_point(alpha_j)
        phi_j = phi(evaluated_point)

        # Armijo / bracket tightening
        if (phi_j > phi_0 + c1 * alpha_j * phi_prime_0) or (phi_j >= phi_lo):
            # shrink upper to alpha_j
            alpha_aux, phi_aux = alpha_hi, phi_hi
            alpha_hi, phi_hi = alpha_j, phi_j

        else:
            # Check curvature
            phi_prime_alpha_j = phi_prime(evaluated_point)
            if abs(phi_prime_alpha_j) <= -c2 * phi_prime_0:
                return alpha_j, phi_j

            # Ensure slope changes sign across bracket,
            # keep a valid bracket
            if phi_prime_alpha_j * (alpha_hi - alpha_lo) >= 0.0:
                alpha_aux, phi_aux = alpha_hi, phi_hi
                alpha_hi, phi_hi = alpha_lo, phi_lo

            else:
                alpha_aux, phi_aux = alpha_lo, phi_lo

            alpha_lo, phi_lo = alpha_j, phi_j
            phi_prime_lo = phi_prime_alpha_j


    return alpha_lo, phi_lo



def _conjugate_gradient_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int,np.ndarray,float,np.ndarray,float,float,
                                 np.ndarray], None]],
    initial_alpha: float,
    max_alpha: float,
    bracketing_max_iter: int,
    wolfe_c1: float,
    wolfe_c2: float,
    zoom_max_iter: int,
    zoom_epsilon: float
) -> Dict[str, Any]:
    """
    Non-linear Conjugate Gradient (Polak-Ribière+ method) optimizer.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function to minimize.
    grad_func : Callable[[np.ndarray], np.ndarray]
        The gradient of the objective function.
    initial_params : np.ndarray
        Initial guess for the parameters.
    common_options : Dict[str, Any]
        Configuration dict. Must contain:
        - 'max_iters': int
        - 'verbose': bool
        - 'verbose_freq': int
        - 'tolerances': Dict[str, float]
    callback : Optional[Callable]
        Function called after each iteration with signature (iter, params,
        cost, gradient, gradient norm, step size alpha, step direction).
    initial_alpha : float
        Initial guess for the step size alpha.
    max_alpha : float
        Maximum allowable step size during line search.
    bracketing_max_iter : int
        Maximum iterations perfomed during the bracketing phase.
    wolfe_c1 : float
        Parameter for the sufficient decrease condition (Armijo).
    wolfe_c2 : float
        Parameter for the curvature condition.
    zoom_max_iter : int
        Maximum iterations perfomed during the zoom phase.
    zoom_epsilon : float
        The epsilon that indicates the minimun distance allowed to
        alpha_hi - alpha_lo, otherwhise ends the zoom algorithm.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'success': Boolean indicating convergence.
        - 'final_params': The optimized parameters.
        - 'final_cost': The value of the function at the optimized
        parameters.
        - 'iteration_number': Total iterations performed.
        - 'termination_reason': String description of why the optimizer stopped.

    Mathematical Description
    ------------------------
    Conjugate gradient is an iterative method for solving linear systems of
    equations based on constructing the quadratic function:

        quad_f(x_k) = 0.5 * x_k^T * A * x_k - b^T * x_k

    where:
    - x_k = current params at iteration k
    - A = symmetric positive definite matrix
    - b = constant vector

    which gradient is the linear system to be solved:

        quad_g_k := gradient(quad_f(x_k)) = A * x_k - b

    The ideal search directions, that is the unit norm vector
    (x_{k+1}-x_k) / ||x_{k+1}-x_k||, for this function satisfy A-conjugacy:

        d_i^T * A * d_j = 0 for i != j

    This property ensures that any step along d_j leaves the quadratic model
    unchanged in the d_i direction, so it does not undo progress made along
    previous directions.

    Taking into account that the first direction is the gradient at the first
    guess x_0 and every other direction used in the algorithm is conjugate to
    the gradient, the name of the algorithm "Conjugate gradient" comes
    immediately.

    In the nonlinear case, A is unknown and varies across iterations, so exact
    A-conjugacy cannot be enforced. Instead, beta_k is chosen to approximate
    this condition using gradient information alone, as shown below.

    ---

    From this theoretical knowledge, the Conjugate Gradient direction method is
    implemented as follows.

    First start with the search direction, which at iteration k is built as a
    linear combination of:

    - the current steepest descent direction -g_{k+1}
    - the previous search direction d_k

    giving the general update:

        d_{k+1} = -g_{k+1} + beta_k * d_k

    where beta_k is a scalar chosen to enforce approximate A-conjugacy which
    can be derived by taking the exact quadratic case and requirement
    A-conjugacy of successive directions:

        d_{k+1}^T * A * d_k = (-g_{k+1} + beta_k * d_k)^T * A * d_k = 0

    rearranging:

        -g_{k+1}^T * A * d_k + beta_k * d_k^T * A * d_k = 0

    and solving for beta_k:

        beta_k = g_{k+1}^T * A * d_k / (d_k^T * A * d_k)

    This still involves A explicitly. To eliminate A, express A * d_k
    through the gradient difference. Knowing that the parameter update is:

        x_{k+1} = x_k + alpha_k * d_k:

    where:
    - alpha_k = is the step size at iteration k, obtained by performing a line
    search (in this code Wolfe conditions).

    then:

        g_{k+1} = A * x_{k+1} - b
                = A * (x_k + alpha_k * d_k) - b
                = (A * x_k - b) + alpha_k * A * d_k
                = g_k + alpha_k * A * d_k

    solving for A * d_k:

        A * d_k = (g_{k+1} - g_k) / alpha_k

    allows to substitute into beta_k, and by cancelling both alpha_k:

        beta_k = g_{k+1}^T * A * d_k / (d_k^T * A * d_k)
               = g_{k+1}^T * (g_{k+1} - g_k) / (d_k^T * (g_{k+1} - g_k)
               = (g_{k+1}^T * g_{k+1} - g_{k+1}^T * g_k) /
                 (d_k^T * g_{k+1} - d_k^T * g_k)

    using the fact that when using an exact line search along d_k:

        phi(alpha) := f(x_k + alpha * d_k)

        d(phi)/d(alpha)|_{alpha=alpha_k} = d_k^T * grad(f(x_k + alpha_k * d_k))
                                         = d_k^T * g_{k+1}
                                         = 0

    so the expression for beta becomes:

        beta_k = (g_{k+1}^T * g_{k+1} - g_{k+1}^T * g_k) /
                 (d_k^T * g_{k+1} - d_k^T * g_k)
               = (g_{k+1}^T * g_{k+1} - g_{k+1}^T * g_k) / (-d_k^T * g_k)

    the denominator can be further simplified by the definition of the search
    direction d_k = -g_k + beta_{k-1} * d_{k-1}:

        d_k^T * g_k = -g_k^T * g_k + beta_{k-1} * d_{k-1}^T * g_k

    that by the proven property d_k^T * g_{k+1} = 0, is:

        d_k^T * g_k = -g_k^T * g_k

    so, finally obtaining:

        beta_k = (g_{k+1}^T * g_{k+1} - g_{k+1}^T * g_k) / (g_k^T * g_k)
               = (g_{k+1}^T * (g_{k+1} - g_k)) / (g_k^T * g_k)


    Which is the Polak-Ribière formula, derived from the A-conjugacy
    requirement without any explicit reference to A.

    This expression has the following interpretations:

    - If the gradient changes little in the direction of g_{k+1},
    then beta_k ≈ 0 and the method behaves like steepest descent.

    - If the gradient change aligns strongly with g_{k+1}, then beta_k > 0 and
    the previous direction is reused, accelerating convergence.

    ---

    As shown in the function _solve_newton_system_cg, where the update
    performed is the one given by Fletcher-Reeves:

        beta_k^FR = ||g_{k+1}||^2 / ||g_k||^2

    In the linear case, the exact line search enforces mutual orthogonality
    of all residuals [1, eq. 5.16]:

        g_{k+1}^T * g_k = 0

    expanding the PR numerator under this condition:

        g_{k+1}^T * (g_{k+1} - g_k) = g_{k+1}^T * g_{k+1} - g_{k+1}^T * g_k
                                    = ||g_{k+1}||^2 - 0
                                    = ||g_{k+1}||^2

    so the Polak-Ribière formula reduces exactly to Fletcher-Reeves.

    In the nonlinear case, gradient orthogonality no longer holds in general,
    so g_{k+1}^T * g_k != 0. The Polak-Ribière formula retains this cross
    term, meaning beta_k accounts for the actual change in the gradient
    between iterations rather than just the ratio of norms. This makes PR
    more adaptive to the local curvature of the nonlinear objective.

    ---

    In practice, the Polak-Ribière beta_k may become negative. To see why
    this is a problem, check the descent condition on d_{k+1}. Substituting
    into g_{k+1}^T * d_{k+1}:

        g_{k+1}^T * d_{k+1} = g_{k+1}^T * (-g_{k+1} + beta_k * d_k)
                            = -||g_{k+1}||^2 + beta_k * g_{k+1}^T * d_k

    for d_{k+1} to be a descent direction one needs:

        g_{k+1}^T * d_{k+1} < 0

    the first term -||g_{k+1}||^2 is always negative, which is favorable.
    However, when beta_k < 0 and d_k is a descent direction (g_{k+1}^T * d_k
    can have either sign), the second term beta_k * g_{k+1}^T * d_k may be
    positive and large enough to violate the descent condition.

    To prevent this, the PR+ variant implemented in this code does:

        beta_k = max(0, g_{k+1}^T * (g_{k+1} - g_k) / (g_k^T * g_k))

    when beta_k = 0, the update reduces to d_{k+1} = -g_{k+1}, i.e. an
    automatic reset to steepest descent. This occurs precisely when the PR
    numerator is negative, indicating that the gradient information from the
    previous iteration is no longer useful.

    Even with PR+, descent is not analytically guaranteed for all cases
    under an inexact line search, since the Wolfe conditions only bound
    d_k^T * g_{k+1} rather than forcing it to zero. For safety, an
    explicit check is performed after each direction update:

        if g_{k+1}^T * d_{k+1} >= 0 then reset d_{k+1} = -g_{k+1}

    This ensures a valid descent direction is always passed to the
    line search.

    References
    ----------
    [1] Algorithm 5.4 with update 5.45 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] Wikipedia contributors.
        **Conjugate Gradient Method.**
        https://en.wikipedia.org/wiki/Conjugate_gradient_method

    [3] Wikipedia contributors.
        **Nonlinear Conjugate Gradient Method.**
        https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    """

    _validate_line_search_parameters(initial_alpha, max_alpha,
        bracketing_max_iter, wolfe_c1, wolfe_c2, zoom_max_iter, zoom_epsilon)

    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    verbose_freq = common_options['verbose_freq']
    tolerances = common_options['tolerances']

    params = initial_params.copy()

    grad = grad_func(params)
    if grad.size != params.size:
        raise ValueError(f"Provided initial params (length = {params.size}) "
                         f"mismatch its dimensions with provided gradient "
                         f"(length = {grad.size}).")

    current_cost = function(params)
    grad_norm = float(np.linalg.norm(grad))

    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check

    search_direction = -grad


    for current_iter in range(1, max_iters + 1):
        alpha, new_cost = _line_search_wolfe(
            function, grad_func, params, search_direction, grad, current_cost,
            wolfe_c1, wolfe_c2, initial_alpha, max_alpha, bracketing_max_iter,
            zoom_max_iter, zoom_epsilon
        )

        if alpha is None or alpha <= 0.0:
            search_direction = -grad
            # Fallback to steepest descent when line search fails, with
            # FLOAT_EPSILON as initial step.
            alpha, new_cost = _line_search_wolfe(
                function, grad_func, params, search_direction, grad,
                current_cost, wolfe_c1, wolfe_c2, FLOAT_EPSILON, max_alpha,
                bracketing_max_iter, zoom_max_iter, zoom_epsilon
            )

            if alpha is None or alpha <= 0.0:
                converged = False
                termination_reason = (
                    "Line search failed to find a suitable step size. Gradient "
                    "probably is too high or too low making stepsize "
                    "numerically unstable. Consider changing to a trust region "
                    "algorithm."
                )
                break

        params += alpha * search_direction

        previous_cost = current_cost
        current_cost = new_cost

        previous_grad = grad
        grad = grad_func(params)
        grad_norm = float(np.linalg.norm(grad))


        terminate, converged, termination_reason = _check_termination(
            current_cost, previous_cost, grad_norm, **tolerances
        )

        if verbose:
            _print_line_search_information(
                current_iter=current_iter, current_cost=current_cost,
                grad_norm=grad_norm, alpha=alpha, verbose_freq=verbose_freq,
                max_iters=max_iters, converged=converged
            )
        if callback:
            callback(
                current_iter, params, current_cost, grad, grad_norm, alpha,
                search_direction
            )

        if terminate: break


        # Polak-Ribière +
        denominator = np.dot(previous_grad, previous_grad)
        # The denominator is ||g_k||^2, if it is below machine epsilon the
        # previous gradient information is not reliable, so a restart to
        # steepest descent by setting beta = 0 must be performed.
        if denominator < FLOAT_EPSILON:
            beta = 0.0
        else:
            numerator = np.dot(grad, grad - previous_grad)
            beta = max(0.0, numerator / denominator)

        search_direction = -grad + beta * search_direction

        # Reset if not a descent direction
        if np.dot(grad, search_direction) >= 0:
            search_direction = -grad

    else:
        termination_reason = (f"Maximum number of iterations ({max_iters}) "
                              f"reached.")
        converged = False


    return {
        'success': converged,
        'final_params': params,
        'final_cost': current_cost,
        'iteration_number': current_iter,
        'termination_reason': termination_reason
    }



def _solve_newton_system_cg(
    hvp_function: Callable[[np.ndarray], np.ndarray],
    grad: np.ndarray,
    max_iter: Union[str, int],
) -> Union[np.ndarray, Literal['hvp_evaluation_error']]:
    """
    Solves the Newton system H * p = -g using the Truncated Conjugate
    Gradient (Newton-CG) method.

    Parameters
    ----------
    hvp_function : Callable[[np.ndarray], np.ndarray]
        Function that computes the matrix-vector product H * v.
        Must accept a 1D array and return a 1D array.
    grad : np.ndarray
        The gradient vector at the current point
        (right hand side of the system of the system is -grad).
    max_iter : Union[str, int, None], optional
        Maximum number of CG iterations.
        - If 'auto' defaults to a heuristic based on the number of dimensions
        (N).
        - If int, uses that specific number.

    Returns
    -------
    np.ndarray, 'hvp_evaluation_error'
        The calculated search direction `p`. If the method fails or curvature
        is negative at step 0, returns steepest descent -grad. If the hvp
        function gives an invalid value (NaN or +-Inf) this function
        returns 'hvp_evaluation_error'.

    Mathematical Description
    ------------------------
    At outer iteration k, model the objective function f(x) taking a step p,
    which is what this functions aims to calculate, using a second-order Taylor
    expansion around the current params x_k:

        f(x_k + p) ≈ m_k(p) = f(x_k) + g^T * p + 0.5 * p^T * H * p

    where:
    - f(x_k) is the function value at x_k, outer iteration k
    - g = current gradient vector at x_k, outer iteration k
    - p = step vector to be calculated
    - H = current exact Hessian matrix at x_k, outer iteration k

    in order to find the optimal step p_optimal, differentiate m_k(p) and set
    to 0:

        gradient(m_k) = g + H * p_optimal = 0

    This yields the search direction (Newton condition), the equation that
    one aims to solve for p_optimal:

        H * p_optimal = - g

    Directly forming or factorizing H is often infeasible for large-scale
    problems, so the solution is obtained iteratively using the Conjugate
    Gradient method, which requires only matrix-vector products H*v and
    never constructs H explicitly.

    ---

    The Conjugate Gradient method minimizes the quadratic model:

        m_quad(p_i) = g^T * p_i + 0.5 * p_i^T * H * p_i

    where:
    - p_i = step vector at inner iteration i
    - g = current gradient vector at x_k, outer iteration k
    - H = current exact Hessian matrix at x_k, outer iteration k

    whose unconstrained minimum is the Newton condition exactly:

        gradient(m_quad(p_i)) = H * p_i + g = 0  ->  H * p_i = -g

    now, define the gradient residual at an inner iterate p_i as:

        r_i := gradient(m_quad(p_i)) = H * p_i + g

    This residual measures how far p_i is from solving the Newton condition.
    First, choose p_0 at 0 (without loss of generality), this implies that:

        r_0 = H * 0 + g = g

    The initial residual is therefore aligned with the steepest descent
    direction (the gradient) of the original objective.

    Note that r_i = H * p_i + g equals the negative of the standard
    linear-system residual (-g - H * p_i), but is more natural here since
    it is exactly the gradient of the quadratic model. The CG iteration
    drives ||r_i|| toward zero, equivalently solving the problem.


    Now, search for p_i conjugate directions in the Krylov subspace:

        K_i(H, r_0) = span{r_0, H * r_0, H^2 * r_0,
                           ..., H^(i-1) * r_0}

    The Krylov subspace is useful because minimizing m_quad over K_i(H, r_0)
    at each step naturally produces search directions that are H-conjugate:

        d_j^T * H * d_i = 0  for  j != i

    ensuring that progress along one direction is never undone by later steps,
    hence the name conjugate gradient. This property will be explicitly
    demonstrated below when beta_i is derived.

    ---

    This implementation is based on the following procedure:

    Start by initializing:

        p_0 = 0            (initial step estimate)
        r_0 = g            (initial gradient residual)
        d_0 = -r_0 = -g    (initial search direction: negative gradient
                            of m_quad at p_0, i.e. steepest descent of m_quad)

    At iteration i, compute curvature along the search direction:

        d_i^T * H * d_i


    - If the curvature is non-positive, the quadratic model is not convex along
    d_i, indicating negative curvature or zero curvature, so the Hessian is not
    positive definite.

    If this occurs at i = 0, the Newton direction is unreliable and a fall back
    to steepest descent is done:

        p_0 = - g

    If it occurs at i > 0, the method is truncated and returns the accumulated
    iterate p_i, which lies on the boundary of the implicit trust region.


    - If the curvature is positive, choose alpha_i (inner step size at
    iteration i) to minimize the quadratic model m(p_i + alpha_i * d_i) along a
    search direction d_i; so substitute p(alpha_i) = p_i + alpha_i * d_i into
    the model and find the root of its derivative with respect to alpha_i:

        m_quad(p_i + alpha_i * d_i)
            = g^T * (p_i + alpha_i * d_i) +
              0.5 * (p_i + alpha_i * d_i)^T * H * (p_i + alpha_i * d_i)

    differentiating with respect to alpha and setting to zero gives:

        d(m_quad(p_i + alpha_i * d_i))/d(alpha_i)
            = g^T * d_i + d_i^T * H * (p_i + alpha_i * d_i)
            = d_i^T * g + d_i^T * H * (p_i + alpha_i * d_i)
            = d_i^T * (H * p_i + g) + alpha * d_i^T * H * d_i
            = 0

    now, recall that the residual:

        r_i = H * p_i + g

    substituting this in, gives:

        d_i^T * r_i = - alpha_i * d_i^T * H * d_i

    achieving that the optimal inner step size is:

        alpha_i = - d_i^T * r_i / (d_i^T * H * d_i)


    Now that alpha is known, the step aproximation is:

        p_{i+1} = p_i + alpha_i * d_i


    And the residual:

        r_{i+1} = H * p_{i+1} + g
                = H * (p_i + alpha_i * d_i) + g
                = (H * p_i + g) + alpha_i * H * d_i
                = r_i + alpha_i * H * d_i

    so compute its norm and check if it is less than a pre-determined
    tolerance:

    If sqrt(r_{i+1}^T * r_{i+1}) <= tol, convergence has been reached so stop.

    Otherwise, compute the Fletcher-Reeves coefficient beta_i in order to
    construct a new search direction d_i. The requirement is that d_{i+1} must
    be H-conjugate to d_i:

        d_{i+1}^T * H * d_i = 0

    knowing that, all d_i for i >= 1 are constructed in this way:

        d_0 = -r_0
        d_i = -r_i + beta_{i-1} * d_{i-1},   i >= 1

    so:

        d_{i+1} = -r_{i+1} + beta_i * d_i

    and introducing it in the requirement to be met, gives:

        (-r_{i+1} + beta_i * d_i)^T * H * d_i = 0

    rearranging:

        -r_{i+1}^T * H * d_i + beta_i * d_i^T * H * d_i = 0

    solving for beta_i:

        beta_i = r_{i+1}^T * H * d_i / (d_i^T * H * d_i)

    this is the Fletcher-Reeves coefficient that force the H-conjugancy between
    d_{i+1} and d_i.

    ---

    Alpha can be calculated easier by taking into account the way d_i is
    calculated that the numerator of alpha_i is:

        d_i^T * r_i = (-r_i + beta_{i-1} * d_{i-1})^T * r_i
                    = -r_i^T * r_i + beta_{i-1} * d_{i-1}^T * r_i

    the second term can be shown to be zero by knowing that:

        r_i = r_{i-1} + alpha_{i-1} * H * d_{i-1}

    so:

        d_{i-1}^T * r_i = d_{i-1}^T * (r_{i-1} + alpha_{i-1} * H * d_{i-1})
                        = d_{i-1}^T * r_{i-1} +
                          alpha_{i-1} * d_{i-1}^T * H * d_{i-1})

    and now, by the defintion of alpha_{i-1}:

        alpha_{i-1} = - d_{i-1}^T * r_{i-1} / (d_{i-1}^T * H * d_{i-1})

    one obtains:

        d_{i-1}^T * r_i = d_{i-1}^T * r_{i-1} - d_{i-1}^T * r_{i-1} = 0

    meaning that:

        d_i^T * r_i = -r_i^T * r_i + beta_{i-1} * d_{i-1}^T * r_i = -r_i^T * r_i

    finally achieving the definition of alpha_i used in code:

        alpha_i = -d_i^T * r_i / (d_i^T * H * d_i)
                = r_i^T * r_i / (d_i^T * H * d_i)

    ---

    Beta can be calculated easier too:

        beta_i = r_{i+1}^T * H * d_i / (d_i^T * H * d_i)

    from the residual update:

        r_{i+1} = r_i + alpha_i * H * d_i

    solve for H * d_i:

        H * d_i = (r_{i+1} - r_i) / alpha_i

    substituting this into the numerator:

        r_{i+1}^T * H * d_i = r_{i+1}^T * (r_{i+1} - r_i) / alpha_i
                            = (r_{i+1}^T * r_{i+1} - r_{i+1}^T * r_i) / alpha_i

    and into the denominator:

        d_i^T * H * d_i = d_i^T * (r_{i+1} - r_i) / alpha_i
                        = (d_i^T * r_{i+1} - d_i^T * r_i) / alpha_i

    but as shown before:

        d_{i-1}^T * r_i = 0, so d_i^T * r_{i+1} = 0

        - d_i^T * r_i = - (-r_i^T * r_i) = r_i^T * r_i

    inserting these into the denominator gives:

        d_i^T * H * d_i = (0 + r_i^T * r_i) / alpha_i
                        = (r_i^T * r_i) / alpha_i

    so, in the end beta_is is, as both the numerator and the denominator have
    alpha_i they cancel:

        beta_i = (r_{i+1}^T * r_{i+1} - r_{i+1}^T * r_i) / (r_i^T * r_i)

    but by the definition of the residual:

        r_{i+1} = r_i + alpha_i * H * d_i

    the second term of the numerator can be proven zero by:

        r_{i+1}^T * r_i = (r_i + alpha_i * H * d_i)^T * r_i
                        = r_i^T * r_i + alpha_i * d_i^T * H * r_i

    knowing the definition of alpha_i:

        alpha_i = r_i^T * r_i / (d_i^T * H * d_i)

    and substituting:

        r_{i+1}^T * r_i
            = r_i^T * r_i + alpha_i * d_i^T * H * r_i

            = r_i^T * r_i +
              r_i^T * r_i * d_i^T * H * r_i / (d_i^T * H * d_i)

            = r_i^T * r_i * (1 + (d_i^T * H * r_i) / (d_i^T * H * d_i))

    focusing now on the second summand and knowing the way d_i is calculated:

    d_i = -r_i + beta_{i-1} * d_{i-1}

    introducing it in the denominator

        (d_i^T * H * r_i) / (d_i^T * H * d_i)
            = (d_i^T * H * r_i) / (d_i^T * H * (-r_i + beta_{i-1} * d_{i-1}))
            = (d_i^T * H * r_i) / (-d_i^T * H * r_i +
                                   d_i^T * H * beta_{i-1} * d_{i-1})

    beta_{i-1} * d_i^T * H * d_{i-1} = 0 by the H-conjugacy of d_i and
    d_{i-1}, which was enforced at the previous iteration when beta_{i-1} was
    chosen so that d_i^T * H * d_{i-1} = 0. Therefore:

        (d_i^T * H * r_i) / (d_i^T * H * d_i)
            = (d_i^T * H * r_i) / (-d_i^T * H * r_i + 0)
            = - d_i^T * H * r_i / d_i^T * H * r_i
            = -1

    so:

        r_{i+1}^T * r_i
            = r_i^T * r_i * (1 + (d_i^T * H * r_i) / (d_i^T * H * d_i))
            = r_i^T * r_i * (1 - 1)
            = 0

    at last achieving the beta implemented in the code:

        beta_i = (r_{i+1}^T * r_{i+1} - r_{i+1}^T * r_i) / (r_i^T * r_i)
               = r_{i+1}^T * r_{i+1} / (r_i^T * r_i)

    which is the Fletcher-Reeves beta only dependant on two consecutive
    residuals.

    References
    ----------
    [1] Algorithm 7.1 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5
    """

    n_dim = grad.size

    # max_iter heuristic
    if max_iter == 'auto':
        max_iter_val = max(int(n_dim*1.25),10)
    else:
        max_iter_val = int(max_iter)

    current_step = np.zeros_like(grad)
    residual = grad.copy()
    search_direction = -residual.copy()

    residual_dot_residual = np.dot(residual, residual)

    norm_rr_0 = np.sqrt(residual_dot_residual)
    tol = min(0.5, np.sqrt(norm_rr_0)) * norm_rr_0 # [1]


    for current_iteration in range(max_iter_val):
        Hs_dot_search_direction = hvp_function(search_direction)

        if not np.isfinite(Hs_dot_search_direction).all():
            return 'hvp_evaluation_error'

        sdir_Hs_sdir = np.dot(search_direction, Hs_dot_search_direction)

        # Negative Curvature / Non-Positive Definite Hessian
        if sdir_Hs_sdir <= FLOAT_EPSILON or not np.isfinite(sdir_Hs_sdir):
            if current_iteration == 0:
                # If negative curvature in the first direction (steepest
                # descent), the Hessian is not positive definite.
                # Fall back to steepest descent direction.
                return -grad
            else:
                # If negative curvature encountered later, it have reached
                # the boundary of the trust region (implicitly).
                # Return the current x accumulated so far.
                break

        alpha = residual_dot_residual / max(sdir_Hs_sdir, FLOAT_EPSILON)

        previous_step = current_step
        # A new current_step must be created at each iteration, so previous_step
        # actually stores the previously used step and its not changed,
        # += must not be used
        current_step = current_step + alpha * search_direction
        residual = residual + alpha * Hs_dot_search_direction

        # If NaN/Inf appears, abort and return last valid x
        # or steepest descent
        if (not np.isfinite(current_step).all() or
            not np.isfinite(residual).all()
        ):
            return -grad if current_iteration == 0 else previous_step

        resi_dot_resi_new = np.dot(residual, residual)

        if np.sqrt(resi_dot_resi_new) <= tol:
            break

        beta = resi_dot_resi_new / max(residual_dot_residual, FLOAT_EPSILON)
        search_direction = - residual + beta * search_direction

        residual_dot_residual = resi_dot_resi_new

    # Final check to ensure descent direction
    if np.dot(current_step, grad) >= 0:
        return -grad

    return current_step



def _newton_cg_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    hvp_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int,np.ndarray,float,np.ndarray,float,float,
                                 np.ndarray], None]],
    initial_alpha: float,
    max_alpha: float,
    bracketing_max_iter: int,
    wolfe_c1: float,
    wolfe_c2: float,
    zoom_max_iter: int,
    zoom_epsilon: float,
    hvp_h: Union[str, float],
    hvp_point_number: int,
    inner_max_iter: Union[str, int],
) -> Dict[str, Any]:
    """
    Newton-CG (Truncated Newton) optimizer.

    The Hessian-vector products are approximated via finite differences,
    avoiding the need for a full Hessian matrix.
    The search direction is computed using a truncated Conjugate Gradient
    inner loop, and the step size is determined via a Wolfe line search.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function to minimize.
    grad_func : Callable[[np.ndarray], np.ndarray]
        The gradient of the objective function.
    hvp_function : Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
        Function that given (current_params, search_direction) computes the
        Hessian product of a vector. All the others hvp parameters are ignored
        if this parameter is provided.
    initial_params : np.ndarray
        Initial guess for the parameters.
    common_options : Dict[str, Any]
        Configuration dict. Must contain:
        - 'max_iters': int
        - 'verbose': bool
        - 'verbose_freq': int
        - 'tolerances': Dict[str, float]
    callback : Optional[Callable]
        Function called after each iteration with signature (iter, params,
        cost, gradient, gradient norm, step size alpha, step direction).
    initial_alpha : float
        Initial guess for the step size alpha.
    max_alpha : float
        Maximum allowable step size during line search.
    bracketing_max_iter : int
        Maximum iterations perfomed during the bracketing phase.
    wolfe_c1 : float
        Parameter for the sufficient decrease condition (Armijo).
    wolfe_c2 : float
        Parameter for the curvature condition.
    zoom_max_iter : int
        Maximum iterations perfomed during the zoom phase.
    zoom_epsilon : float
        The epsilon that indicates the minimun distance allowed to
        alpha_hi - alpha_lo, otherwhise ends the zoom algorithm.
    hvp_h : Union[str, float]
        Step used in the finite differences.
    hvp_point_number : int
        Number of points used in the central finite differences formula. Must
        be even.
    inner_max_iter : Union[str, int]
        Maximum number of CG iterations.
        - If 'auto' defaults to the number of dimensions (N).
        - If int, uses that specific number.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'success': Boolean indicating convergence.
        - 'final_params': The optimized parameters.
        - 'final_cost': The value of the function at the optimized
        parameters.
        - 'iteration_number': Total iterations performed.
        - 'termination_reason': String description of why the optimizer stopped.

    References
    ----------
    [1] Algorithm 7.1 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5
    """

    _validate_line_search_parameters(initial_alpha, max_alpha,
        bracketing_max_iter, wolfe_c1, wolfe_c2, zoom_max_iter, zoom_epsilon)

    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    verbose_freq = common_options['verbose_freq']
    tolerances = common_options['tolerances']

    params = initial_params.copy()

    grad = grad_func(params)
    if grad.size != params.size:
        raise ValueError(f"Provided initial params (length = {params.size}) "
                         f"mismatch its dimensions with provided gradient "
                         f"(length = {grad.size}).")

    current_cost = function(params)
    grad_norm = float(np.linalg.norm(grad))

    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check


    # params change at each iteration, so the current params is passed to
    # the hvp function.
    if hvp_function:
        _hvp_func = lambda vector: hvp_function(params, vector)
    else:
        _hvp_func = lambda vector: np.atleast_1d(
            numerical_hessian_vector_product(
                gradient_function=grad_func,
                point=params,
                vector=vector,
                step_size=hvp_h,
                point_number=hvp_point_number,
            ).derivative
        )


    for current_iter in range(1, max_iters + 1):
        search_direction = _solve_newton_system_cg(_hvp_func, grad,
            inner_max_iter)

        if (
            not isinstance(search_direction, np.ndarray)
            and search_direction == 'hvp_evaluation_error'
        ):
            converged = False
            termination_reason = ("Invalid value (NaN or +-Inf) "
                                  "encountered while evaluating the hessian.")
            break

        previous_cost = current_cost
        _initial_alpha = 1.0 if current_iter > 1 else initial_alpha

        alpha, current_cost = _line_search_wolfe(
            function, grad_func, params, search_direction, grad, current_cost,
            wolfe_c1, wolfe_c2, _initial_alpha, max_alpha, bracketing_max_iter,
            zoom_max_iter, zoom_epsilon
        )

        if alpha is None or alpha <= 0.0:
            search_direction = -grad
            # Fallback to steepest descent when line search fails,
            # with FLOAT_EPSILON as initial step
            alpha, current_cost = _line_search_wolfe(
                function, grad_func, params, search_direction, grad,
                current_cost, wolfe_c1, wolfe_c2, FLOAT_EPSILON, max_alpha,
                bracketing_max_iter, zoom_max_iter, zoom_epsilon
            )

            if alpha is None or alpha <= 0.0:
                converged = False
                termination_reason = (
                    "Line search failed to find a suitable step size. Gradient "
                    "probably is too high or too low making stepsize "
                    "numerically unstable. Consider changing to a trust region "
                    "algorithm."
                )
                break

        params += alpha * search_direction
        grad = grad_func(params)
        grad_norm = float(np.linalg.norm(grad))

        # Check termination at end of iteration so the convergent state is
        # available to _print_line_search_information via the converged flag.
        terminate, converged, termination_reason = _check_termination(
            current_cost, previous_cost, grad_norm, **tolerances
        )

        if verbose:
            _print_line_search_information(
                current_iter=current_iter, current_cost=current_cost,
                grad_norm=grad_norm, alpha=alpha, verbose_freq=verbose_freq,
                max_iters=max_iters, converged=converged
            )
        if callback:
            callback(
                current_iter, params, current_cost, grad, grad_norm, alpha,
                search_direction
            )

        if terminate: break

    else:
        termination_reason = (f"Maximum number of iterations ({max_iters}) "
                              f"reached.")
        converged = False


    return {
        'success': converged,
        'final_params': params,
        'final_cost': current_cost,
        'iteration_number': current_iter,
        'termination_reason': termination_reason
    }



def _bfgs_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int,np.ndarray,float,np.ndarray,float,float,
                                 np.ndarray], None]],
    initial_alpha: float,
    max_alpha: float,
    bracketing_max_iter: int,
    wolfe_c1: float,
    wolfe_c2: float,
    zoom_max_iter: int,
    zoom_epsilon: float
) -> Dict[str, Any]:
    """
    Dense Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function to minimize.
    grad_func : Callable[[np.ndarray], np.ndarray]
        The gradient of the objective function.
    initial_params : np.ndarray
        Initial guess for the parameters.
    common_options : Dict[str, Any]
        Configuration dict. Must contain:
        - 'max_iters': int
        - 'verbose': bool
        - 'verbose_freq': int
        - 'tolerances': Dict[str, float]
    callback : Optional[Callable]
        Function called after each iteration with signature (iter, params,
        cost, gradient, gradient norm, step size alpha, step direction).
    initial_alpha : float
        Initial guess for the step size alpha.
    max_alpha : float
        Maximum allowable step size during line search.
    bracketing_max_iter : int
        Maximum iterations perfomed during the bracketing phase.
    wolfe_c1 : float
        Parameter for the sufficient decrease condition (Armijo).
    wolfe_c2 : float
        Parameter for the curvature condition.
    zoom_max_iter : int
        Maximum iterations perfomed during the zoom phase.
    zoom_epsilon : float
        The epsilon that indicates the minimun distance allowed to
        alpha_hi - alpha_lo, otherwhise ends the zoom algorithm.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'success': Boolean indicating convergence.
        - 'final_params': The optimized parameters.
        - 'final_cost': The value of the function at the optimized
        parameters.
        - 'iteration_number': Total iterations performed.
        - 'termination_reason': String description of why the optimizer stopped.

    Mathematical Description
    ------------------------
    At iteration k, model the objective function f(x) taking a step p_k using a
    second-order (as the curvature information is what one wants) Taylor
    expansion around the current position x_k:

        f(x_k + p_k) ≈ m_k(p_k) = f(x_k) + g_k^T * p_k + 0.5 * p_k^T * H_k * p_k

    where:
    - p_k = step vector at iteration k
    - g_k = gradient vector at x_k
    - H_k = an approximation of the Hessian matrix at x_k

    To find the optimal step p_k, differentiate m_k(p) and set to 0:

        gradient(m_k) = g_k + H_k * p_k = 0

    This yields the search direction (Quasi-Newton condition):

        p_k = - H_k^-1 * g_k

    ---

    Now, it is needed to update H_k to H_{k+1} such that it incorporates
    the new curvature information (Quasi-Newton) found during the step.

    First, apply a first-order Taylor expansion to the gradient around
    x_k taking a step defined as:

        s_k := x_{k+1} - x_k = alpha * p_k:

    so, the expansion is:

        g_{x_k + s_k} ≈ g_{x_k} + H_k * s_k

    which can be rearranged as:

        H_k * (x_{k+1} - x_k) = H_k * s_k = g_{k+1} - g_k

    last, define:

        y_k := g_{k+1} - g_k

    to finally achieve:

        H_k * s_k = g_{k+1} - g_k = y_k

    so:

        s_k = H_k^-1 * y_k

    which is the Secant condition (Quasi-Newton).


    This condition ensures that the updated approximation incorporates the
    newly observed curvature information along direction s_k.

    The curvature information is obtained knowing that the BFGS update force
    H_{k+1} to satisfy the Secant condition (as shown below). So H_{k+1} is
    required to be positive definite, equivalently for any non-zero vector z:

        z^T * H_{k+1} * z > 0.

    Knowing this and left-multiplying the Secant condition by s_k^T yields:

        s_k^T * H_{k+1} * s_k = s_k^T * y_k

    The left-hand side is positive by positive definiteness of H_k{k+1}
    (since s_k cant be zero as s_k = 0 an exit for the algorithm), so:

        s_k^T * y_k > 0

    is therefore a necessary condition for H_{k+1} to be Positive Definite.


    On the other hand, the weak Wolfe curvature condition on the line search
    step states (see _line_search_wolfe):

        g_{k+1}^T * p_k >= c2 * g_k^T * p_k          (with 0 < c1 < c2 < 1)

    now expand y_k^T * s_k using their definitions (y_k = g_{k+1} - g_k,
    s_k = alpha * p_k, alpha > 0):

        y_k^T * s_k = (g_{k+1} - g_k)^T * (alpha * p_k)
                    = alpha * (g_{k+1}^T * p_k - g_k^T * p_k)

    and from the Wolfe condition, subtract g_k^T * p_k from both sides

        g_{k+1}^T * p_k - g_k^T * p_k >= (c2 - 1) * g_k^T * p_k

    Since p_k is a descent direction: g_k^T * p_k < 0 and since c2 < 1:
    (c2 - 1) < 0. So the right-hand side is the product of two negatives, hence:

        (c2 - 1) * g_k^T * p_k > 0

    Therefore:

        g_{k+1}^T * p_k - g_k^T * p_k > 0

    And multiplying by alpha > 0:

        y_k^T * s_k = alpha * (g_{k+1}^T * p_k - g_k^T * p_k) > 0


    Taking all of these facts into account, one can conclude that:

    - Satisfying the Wolfe conditions during the line search is sufficient
    to guarantee the curvature condition at every iteration.
    - The curvature condition is necessary for H_{k+1} to be Positive Definite.

    For these two reasons this code only updates the inverse Hessian if the
    condition is met, if not it skips it.

    ---

    At this point, it is needed to update H_k accordingly so one can add two
    matrices of rank-one, U_k and V_k; whose sum is a rank-two update matrix
    [2]:

        H_{k+1} = H_k + U_k + V_k.

    In order to maintain the symmetry and positive definiteness of H_{k+1},
    the update form can be modified as:

        H_{k+1} = H_k + c1 * u * u^T + c2 * v * v^T

    where c1 and c2 are scalars, u and v are column vectors. All four are
    going to be determined in the following way:

    -First substitute the update into the Secant condition (H_{k+1} * s_k =
    y_k):

        (H_k + c1 * u * u^T + c2 * v * v^T) * s_k = y_k

    using the identity (a * b^T) * c = (b^T * c) * a, expand each term such
    that:

        H_k * s_k + c1 * (u^T * s_k) * u + c2 * (v^T * s_k) * v = y_k

    -Second, to make this last equation solvable, u and v are chosen to mirror
    the two quantities already present in the equation, so:

        u := y_k
        v := H_k * s_k

    these choices are natural because u must eventually produce the y_k term
    on the right-hand side, and v must cancel the H_k * s_k term on the left.

    -Third, substitute u and v into the update:

        H_k * s_k
        + c1 * (y_k^T * s_k) * y_k
        + c2 * (s_k^T * H_k * s_k) * H_k * s_k = y_k

    By grouping the terms, moving the y_k to the left side and grouping the
    H_k * s_k terms together:

        c1 * (y_k^T * s_k) * y_k - y_k
        + [1 + c2 * (s_k^T * H_k * s_k)] * H_k * s_k = 0

    which is equivalent to:

        [c1 * (y_k^T * s_k) - 1] * y_k
        + [1 + c2 * (s_k^T * H_k * s_k)] * H_k * s_k = 0

    -Fifth, for the last equation to hold in general (assuming y_k and H_k * s_k
    are linearly independent), each bracketed coefficient must be zero
    independently, meaning that:

    from the y_k term:

        c1 * (y_k^T * s_k) - 1 = 0 => c1 = 1 / (y_k^T * s_k)

    and from the H_k * s_k term:

        1 + c2 * (s_k^T * H_k * s_k) = 0 => c2 = -1 / (s_k^T * H_k * s_k)

    -Sixth, by substituting c1, c2, u, and v back into the update:

        H_{k+1} = H_k + y_k * y_k^T / (y_k^T * s_k) -
                  (H_k * s_k) * (H_k * s_k)^T / (s_k^T * H_k * s_k)

    Assuming now that H_k is symmetric (H_k = H_k^T), equivalent as assuming
    that all the second partial derivatives of the function are continous,
    note that (H_k * s_k)^T = s_k^T * H_k, so:

        H_{k+1} = H_k +
                  y_k * y_k^T / (y_k^T * s_k) -
                  H_k * s_k * s_k^T * H_k / (s_k^T * H_k * s_k)

    Notice that the symmetry of H_{k+1} is preserved because:
    - H_k is symmetric by assumption.
    - y_k * y_k^T is of the form w * w^T, hence symmetric.
    - H_k * s_k * s_k^T * H_k = (H_k * s_k) * (H_k * s_k)^T,
    is also of the form w * w^T, hence symmetric.


    The last update equation of H_{k+1} is the BFGS Hessian update equation,
    and it can be transformed by the Sherman-Morrison-Woodbury formula [2],
    giving:

        rho := 1 / (y_k^T * s_k)

        H_{k+1}^-1
            = (I - rho * s_k * y_k^T) * H_k^-1 * (I - rho * y_k * s_k^T) +
              rho * s_k * s_k^T

    This update guarantees:
    - H_{k+1}^-1 is symmetric
    - H_{k+1}^-1 * y_k = s_k
    - H_{k+1}^-1 is positive definite if H_k^-1 is too and the curvature
    condition (y_k^T * s_k > 0) is met.


    This is exactly what one wants as it allows to calculate H_{k+1}^-1 at
    every iteration so the Secant condition (s_k = H_k^-1 * y_k) can be applied
    directly in order to calculate the next iteration's params
    (s_k = x_{k+1} - x_k).

    At first the first iteration, k = 0, the inverse Hessian it given a
    provisional value of [1, eq. 6.20]:

        H_0^-1 = (y_k^T * s_k / (y_k^T * y_k)) * I

    ---

    If one looks at the update formula for H_k^-1, it involves matrix-matrix
    multiplications, resulting in O(n^3) complexity per iteration; so, it can
    be expanded algebraically to avoid matrix-matrix products by letting:

        rho_k = 1 / (y_k^T * s_k)   (scalar, already known)
        Hy_k = H_k^-1 * y_k         (matrix-vector product, O(n^2))
        yHy = y_k^T * H_k^-1 * y_k  (scalar = y_k^T * Hy_k, O(n))

    and defining for compactness:

        B_k := H_k^-1


    -First, expand the right factor of B_k gives:

        B_k * (I - rho * y_k * s_k^T) = B_k - rho * B_k * y_k * s_k^T
                                      = B_k - rho * Hy_k * s_k^T

    -Second, applying then the left factor (I - rho * s_k * y_k^T) to the
    result:

        (I - rho * s_k * y_k^T) * (B_k - rho * Hy_k * s_k^T)

    analyzing term by term:

        I * B_k = B_k

        I * (-rho * Hy_k * s_k^T) = - rho * Hy_k * s_k^T

        (-rho*s_k*y_k^T) * B_k = - rho * s_k * (y_k^T * B_k)
                               = - rho * s_k * Hy_k^T
        [since (B_k*y_k)^T = y_k^T*B_k^T and B_k symmetric => B_k^T=B_k]

        rho^2 * s_k * (y_k^T * Hy_k) * s_k^T = rho^2 * yHy * s_k * s_k^T

    -Third, combine with the remaining rho * s_k * s_k^T from the BFGS update:

        B_{k+1} = B_k - rho * Hy_k * s_k^T - rho * s_k * Hy_k^T +
                  rho^2 * yHy * s_k * s_k^T + rho * s_k * s_k^T

    -Fourth, collect the two s_k * s_k^T terms and simplify them so they
    can be computed cheaply:

        rho^2 * yHy * s_k * s_k^T + rho * s_k * s_k^T
            = rho * (rho * yHy + 1) * s_k * s_k^T
            = rho * (1 + rho * yHy) * s_k * s_k^T

    -Fifth, final result unroll B_k := H_k^-1 obtaining the inverse Hessian
    update used in this code:

        H_{k+1}^-1 = H_k^-1 +
                     rho * (1 + rho * yHy) * s_k * s_k^T -
                     rho * (Hy_k * s_k^T + s_k * Hy_k^T)

    which requires only:
    - matrix-vector products  O(n^2)
    - vector inner products   O(n)
    - vector outer products   O(n^2)

    Achieves the same mathematical result as the standard BFGS formula,
    while reducing per-iteration complexity from O(n^3) to O(n^2).

    References
    ----------
    [1] Algorithm 6.1 & eqs. 6.19, 6.20 from Nocedal, J., & Wright, S. J.
        (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] Hessian update from Wikipedia contributors.
        **BFGS Method**.
        https://en.wikipedia.org/wiki/BFGS#Algorithm
    """

    _validate_line_search_parameters(initial_alpha, max_alpha,
        bracketing_max_iter, wolfe_c1, wolfe_c2, zoom_max_iter, zoom_epsilon)

    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    verbose_freq = common_options['verbose_freq']
    tolerances = common_options['tolerances']

    params = initial_params.copy()
    dim_num = params.size

    grad = grad_func(params)
    if grad.size != dim_num:
        raise ValueError(f"Provided initial params (length = {dim_num}) "
                         f"mismatch its dimensions with provided gradient "
                         f"(length = {grad.size}).")

    grad_norm = float(np.linalg.norm(grad))

    current_cost = function(params)
    previous_cost = np.inf

    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check

    # Initialize inverse Hessian approximation to Identity
    inverse_Hessian = np.eye(dim_num, dtype=float)
    inv_Hessian_is_identity = True


    for current_iter in range(1, max_iters + 1):

        # Search Direction, p = -H_k^-1 * g_k
        search_direction = -inverse_Hessian.dot(grad)

        # Ensure descent direction if H_k^-1 becomes ill-conditioned,
        # p might not be descent.
        if np.dot(search_direction, grad) >= 0:
            # Reset inverse Hessian to Identity and use steepest descent
            inverse_Hessian = np.eye(dim_num, dtype=float)
            inv_Hessian_is_identity = True
            search_direction = -grad

        previous_cost = current_cost
        _initial_alpha = 1.0 if current_iter > 1 else initial_alpha

        alpha, current_cost = _line_search_wolfe(
            function, grad_func, params, search_direction, grad,
            current_cost, wolfe_c1, wolfe_c2, _initial_alpha, max_alpha,
            bracketing_max_iter, zoom_max_iter, zoom_epsilon
        )

        if alpha is None or alpha <= 0.0:
            # Retry Line Search with steepest descent and FLOAT_EPSILON as
            # inital step
            search_direction = -grad

            alpha, current_cost = _line_search_wolfe(
                function, grad_func, params, search_direction, grad,
                current_cost, wolfe_c1, wolfe_c2, FLOAT_EPSILON, max_alpha,
                bracketing_max_iter, zoom_max_iter, zoom_epsilon
            )

            if alpha is None or alpha <= 0.0:
                converged = False
                termination_reason = (
                    "Line search failed to find a suitable step size. Gradient "
                    "probably is too high or too low making stepsize "
                    "numerically unstable. Consider changing to a trust region "
                    "algorithm."
                )
                break

            if not inv_Hessian_is_identity:
                inverse_Hessian = np.eye(dim_num, dtype=float)
                inv_Hessian_is_identity = True

        step_vector = alpha * search_direction # s_k
        params += step_vector

        grad_new = grad_func(params)
        grad_norm = float(np.linalg.norm(grad_new))

        grad_delta_k = grad_new - grad # y_k
        step_dot_grad_delta = np.dot(step_vector, grad_delta_k)


        # Anything below sqrt(eps) * ||s|| * ||y|| is roundoff noise, not real
        # curvature.
        curvature_tolerance = float(
            np.sqrt(FLOAT_EPSILON) * np.linalg.norm(step_vector) *
            np.linalg.norm(grad_delta_k)
        )

        # Curvature condition: y_k^T * s_k > 0 in exact arithmetic, a
        # tolerance is added to prevent numerical issues, see _bfgs_optimizer.
        if step_dot_grad_delta > curvature_tolerance:

            rho = 1.0 / step_dot_grad_delta

            # Initial scaling for the inverse Hessian after line search
            # and before update
            if current_iter == 1:

                grad_delta_sq = np.dot(grad_delta_k,grad_delta_k)

                # grad_delta_sq is > 0 by the curvature_tolerance
                scale_factor = step_dot_grad_delta / grad_delta_sq

                inverse_Hessian = (
                    scale_factor * np.eye(dim_num, dtype=float)
                )

            # Update inverse Hessian
            inv_Hs_dot_grad_delta = np.dot(inverse_Hessian, grad_delta_k)
            grad_delta_inv_Hs_grad_delta = np.dot(
                grad_delta_k, inv_Hs_dot_grad_delta
            )
            term1_scalar = rho * (1.0 + rho * grad_delta_inv_Hs_grad_delta)

            inverse_Hessian += term1_scalar * np.outer(step_vector, step_vector)

            inv_Hs_grad_delta_outer_step = np.outer(
                inv_Hs_dot_grad_delta, step_vector
            )

            inverse_Hessian -= rho * (inv_Hs_grad_delta_outer_step +
                                      inv_Hs_grad_delta_outer_step.T)

            inv_Hessian_is_identity = False

        if not np.all(np.isfinite(inverse_Hessian)):
            inverse_Hessian = np.eye(dim_num, dtype=float)
            inv_Hessian_is_identity = True

        grad = grad_new


        # Check termination at end of iteration so the convergent state is
        # available to _print_line_search_information via the converged flag.
        terminate, converged, termination_reason = _check_termination(
            current_cost, previous_cost, grad_norm, **tolerances
        )

        if verbose:
            _print_line_search_information(
                current_iter=current_iter, current_cost=current_cost,
                grad_norm=grad_norm, alpha=alpha, verbose_freq=verbose_freq,
                max_iters=max_iters, converged=converged
            )
        if callback:
            callback(
                current_iter, params, current_cost, grad, grad_norm, alpha,
                search_direction
            )

        if terminate: break

    else:
        termination_reason = (f"Maximum number of iterations ({max_iters}) "
                              f"reached.")
        converged = False


    return {
        'success': converged,
        'final_params': params,
        'final_cost': current_cost,
        'iteration_number': current_iter,
        'termination_reason': termination_reason
    }



def _lbfgs_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int,np.ndarray,float,np.ndarray,float,float,
                                 np.ndarray], None]],
    memory_size: int,
    initial_alpha: float,
    max_alpha: float,
    bracketing_max_iter: int,
    wolfe_c1: float,
    wolfe_c2: float,
    zoom_max_iter: int,
    zoom_epsilon: float
) -> Dict[str, Any]:
    """
    Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function to minimize.
    grad_func : Callable[[np.ndarray], np.ndarray]
        The gradient of the objective function.
    initial_params : np.ndarray
        Initial guess for the parameters.
    common_options : Dict[str, Any]
        Configuration dict. Must contain:
        - 'max_iters': int
        - 'verbose': bool
        - 'verbose_freq': int
        - 'tolerances': Dict[str, float]
    callback : Optional[Callable]
        Function called after each iteration with signature (iter, params,
        cost, gradient, gradient norm, step size alpha, step direction).
    memory_size: int
        Number of vectors (m) to store in memory.
    initial_alpha : float
        Initial guess for the step size alpha.
    max_alpha : float
        Maximum allowable step size during line search.
    bracketing_max_iter : int
        Maximum iterations perfomed during the bracketing phase.
    wolfe_c1 : float
        Parameter for the sufficient decrease condition (Armijo).
    wolfe_c2 : float
        Parameter for the curvature condition.
    zoom_max_iter : int
        Maximum iterations perfomed during the zoom phase.
    zoom_epsilon : float
        The epsilon that indicates the minimun distance allowed to
        alpha_hi - alpha_lo, otherwhise ends the zoom algorithm.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'success': Boolean indicating convergence.
        - 'final_params': The optimized parameters.
        - 'final_cost': The value of the function at the optimized
        parameters.
        - 'iteration_number': Total iterations performed.
        - 'termination_reason': String description of why the optimizer stopped.

    Mathematical Description
    ------------------------
    At iteration k, model the objective function f(x) taking a step p_k using a
    second-order (as the curvature information is what one wants) Taylor
    expansion around the current position x_k:

        f(x_k + p_k) ≈ m_k(p_k) = f(x_k) + g_k^T * p_k + 0.5 * p_k^T * H_k * p_k

    where:
    - p_k = step vector at iteration k
    - g_k = gradient vector at x_k
    - H_k = an approximation of the Hessian matrix at x_k

    The optimal step p_k (taking the gradient of the quadratic model with
    respect to p and setting it to zero) satisfies the Newton equation:

        H_k * p_k = -g_k  =>  p_k = H_k^-1 * -g_k

    Limited-memory BFGS avoids storing the dense inverse Hessian approximation
    H_k^-1 (which requires O(N^2) memory), making it suitable for large-scale
    optimization problems where N is large.

    Instead of a full matrix, L-BFGS stores a history of the last 'm'
    updates, consisting of the pair of vectors:

        s_i = x_{i+1} - x_i    (step vector)
        y_i = g_{i+1} - g_i    (gradient difference)

    for i = k-m, ..., k-1. Where k is the current iteration index.

    ---

    The standard BFGS update formula for the inverse Hessian is (see
    _bfgs_optimizer):

        H_{k+1}^-1
            = (I - rho * s_k * y_k^T) * H_k^-1 * (I - rho * y_k * s_k^T) +
              rho * s_k * s_k^T

    where rho_k = 1 / (y_k^T * s_k).

    Letting V_k = (I - rho_k * y_k * s_k^T), this can be written as:

        H_{k+1}^-1 = V_k^T * H_k^-1 * V_k + rho_k * s_k * s_k^T

    L-BFGS conceptually applies this update recursively m times
    starting from a sparse initial approximation H_k^0 (a scaled
    identity matrix gamma_k * I).

    ---

    The last update can be expanded recurvively for the last m updates:

    Starting with B_0 as the initial approximation chosen as [1, eq. 7.20]:

        B_0 = gamma_k * I

    where:

        gamma_k = s_{k-1}^T * y_{k-1} / (y_{k-1}^T * y_{k-1})

    and by letting B_j := H_j^-1 for compactness, then knowing that the oldest
    pair indexed is k-m and that the newest indexed is k-1, one obtains:

        B_k = (V_{k-1}^T * ... * V_{k-m}^T) *
              B_0 *
              (V_{k-m} * ... * V_{k-1}) +

              rho_{k-m} * (V_{k-1}^T * ... * V_{k-m+1}^T) *
              s_{k-m}* s_{k-m}^T *
              (V_{k-m+1} * ... * V_{k-1}) +

              rho_{k-m+1} * (V_{k-1}^T * ... * V_{k-m+2}^T) *
              s_{k-m+1} * s_{k-m+1}^T *
              (V_{k-m+2} * ... * V_{k-1}) +

              ... +

              rho_{k-1} * s_{k-1} * s_{k-1}^T

    From now on the step p_k (from Newton condition p_k = H_k^-1 * -g_k) is
    treated as a search direction (named q from now) that later will be passed
    to a line search algorithm which will find the minimum along said search
    direction and return the step size alpha from which finally the new
    iterations paramateres are computed.

    In order to achieve this, right multiplying the expanded update for the
    last m iterations by * -g_k gives the matrix-vector product that calculates
    the search direction q, so:

        q = B_k * -g_k = (V_{k-1}^T * ... * V_{k-m}^T) *
                         B_0 *
                         (V_{k-m} * ... * V_{k-1}) * -g_k +

                         rho_{k-m} * (V_{k-1}^T * ... * V_{k-m+1}^T) *
                         s_{k-m}* s_{k-m}^T *
                         (V_{k-m+1} * ... * V_{k-1}) * -g_k +

                         rho_{k-m+1} * (V_{k-1}^T * ... * V_{k-m+2}^T) *
                         s_{k-m+1} * s_{k-m+1}^T *
                         (V_{k-m+2} * ... * V_{k-1}) * -g_k +

                         ... +

                         rho_{k-1} * s_{k-1} * s_{k-1}^T * -g_k

    ---

    The last equation that calculates the search direction for the current
    iteration k is the core part of this algorithm, it is calculated and
    implemented in this code in the following way:

    -First, define a vector q1 (search direction in the code) that will be
    initialized to -g_k, that is q1_{i=k-1} = -g_k, from the first term
    of the last equation apply V_i to q1_i by constructing a first loop
    (backward pass) for i = k-1 down to k-m:

        q1_{i+1} <- V_i * q1_i = (I - rho_i * y_i * s_i^T) * q1_i
                            = q1_i - rho_i * (s_i^T * q1_i) * y_i
                            = q1_i - scalar_1_i * y_i

    where:

        scalar_1_i = rho_i * (s_i^T * q1_i)

    scalar_1 is stored for the second loop; scalar_1 is commonly called
    alpha across literature [1, Algorithm 7.4], renamed here as scalar_1 for
    clarity with the step size alpha.

    -Second, update the initial approximation of the scaled search direction
    with the last q1, that is q1_{k-m+1}:

        q_scaled <- B_0 * q1_{k-m} = gamma_k * I * q1_{k-m} = gamma_k * q1_{k-m}

    with:

        gamma_k = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})

    This scaling matches the magnitude of the inverse Hessian along the most
    recent update direction.


    Now, notice that at each iteration the rank-one terms (second term in the
    update formula: B_{k+1} = V_k^T * B_k * V_k + rho_k * s_k * s_k^T)
    left-multiplied by q1_i are:

        rho_i * s_i * s_i^T * q1_i = rho_i * (s_i^T * q1_i) * s_i
                                   = scalar_1_i * s_i

    So an efficient way of computing the search direction is including them in
    the second loop as scalar_1_i was stored from the first loop.

    -Third, define another vector q2 (still search direction in code, but
    here made different from q1 for clarity). Then, perform a second loop
    (forward pass), with q2_{i=k-m} = q_scaled, for i = k-m up to k-1:

        q2_{i+1} <- V_i^T * q2_i + rho_i * s_i * s_i^T * q1_i =
            = (I - rho_i * s_i * y_i^T) * q2_i + rho_i * s_i * s_i^T * q1_i
            = q2_i - rho_i * (y_i^T * q2_i) * s_i + rho_i * (s_i^T * q1_i) * s_i
            = q2_i - scalar_2_i * s_i + scalar_1_i * s_i
            = q2_i + (scalar_1_i - scalar_2_i) * s_i

    where:

        scalar_2_i = rho_i * y_i^T * q2_i

    scalar_2 is commonly called beta across literature [1, Algorithm 7.4],
    renamed here as scalar_2 for consistency with scalar_1.


    This resulting final q2, that is q2_k, is the exact BFGS search
    direction for the last m iterations, as shown above in the expansion
    * -g_k; reintroducing the curvature information in the correct order and
    completing the application of B_k = H_k^-1 to -g_k in order to finally
    achieve the final search direction q_k = H_k^-1 * -g_k.

    So one can calculate the next iteration params as one would have q_k and
    alpha (from line search), so x_{k+1} - x_k = s_k = alpha * q_k.


    This reduces:

    - Memory: O(N^2) → O(m*N)
    - Cost: O(N^2) → O(m*N)

    while preserving the essential curvature information needed for fast
    convergence.

    References
    ----------
    [1] Algorithms 7.4 and 7.5 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] Wikipedia contributors.
        **Limited-memory BFGS (L-BFGS)**.
        https://en.wikipedia.org/wiki/LBFGS
    """

    _validate_line_search_parameters(initial_alpha, max_alpha,
        bracketing_max_iter, wolfe_c1, wolfe_c2, zoom_max_iter, zoom_epsilon)

    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    verbose_freq = common_options['verbose_freq']
    tolerances = common_options['tolerances']

    # Store tuples of (s_k, y_k, rho_k)
    # s_k = x_{k+1} - x_k = step_vector
    # y_k = g_{k+1} - g_k = grad_delta
    # rho_k = 1 / (y_k^T * s_k)
    history = deque(maxlen=memory_size)

    params = initial_params.copy()

    grad = grad_func(params)
    if grad.size != params.size:
        raise ValueError(f"Provided initial params (length = {params.size}) "
                         f"mismatch its dimensions with provided gradient "
                         f"(length = {grad.size}).")

    current_cost = function(params)
    grad_norm = float(np.linalg.norm(grad))

    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check


    for current_iter in range(1, max_iters + 1):

        # Two-Loop recursion
        search_direction = -grad # q
        scalar_1_list = []

        # First loop (backward pass, newest -> oldest)
        # reversed(history) because deque appends to the right (newest at end)
        # Applies the right chain (V_{k-m} * ... * V_{k-1}) to -g_k,
        # starting with V_{k-1} (innermost) and going outwards.
        for step_vector, grad_delta, rho in reversed(history):

            # step_vector is s_k, grad_delta is y_k
            scalar_1_i = rho * np.dot(step_vector, search_direction)
            scalar_1_list.append(scalar_1_i)

            search_direction -= scalar_1_i * grad_delta

        # Scaling step: apply B_0 = gamma_k * I.
        # gamma_k is updated at every iteration using the most recent stored
        # pair, contrary to BFGS which only scales at the first iteration.
        if history:
            # Newest element. rho_last = 1 / (s_last^T * y_last), so
            # s_last^T * y_last can be recovered directly instead of
            # recomputing the dot product.
            _, grad_delta_last, rho_last = history[-1]
            step_dot_grad_delta_last = 1 / rho_last
            grad_delta_sq = np.dot(grad_delta_last, grad_delta_last)

            gamma = (
                step_dot_grad_delta_last / grad_delta_sq
                if grad_delta_sq > FLOAT_EPSILON else 1.0
            )
        else:
            gamma = 1.0

        search_direction *= gamma

        # Second loop (fordward pass, oldest -> newest)
        # scalar_1 in reverse order to make it consistent (oldest with oldest,
        # newest with newest)
        # Applies the left chain (V_{k-m}^T * ... * V_{k-1}^T) and adds the
        # rank-one contribution scalar_1_i * s_i at each step.
        for (step_vector, grad_delta, rho), scalar_1_i in zip(history,
            reversed(scalar_1_list), strict=True):

            scalar_2 = rho * np.dot(grad_delta, search_direction)
            search_direction += step_vector * (scalar_1_i - scalar_2)

        # If the Two-Loop recursion produced a bad direction, reset history and
        # use steepest descent
        if np.dot(search_direction, grad) >= 0:
            history.clear()
            search_direction = -grad



        _initial_alpha = 1.0 if current_iter > 1 else initial_alpha
        previous_cost = current_cost

        alpha, current_cost = _line_search_wolfe(
            function, grad_func, params, search_direction, grad, current_cost,
            wolfe_c1, wolfe_c2, _initial_alpha, max_alpha, bracketing_max_iter,
            zoom_max_iter, zoom_epsilon
        )

        if alpha is None or alpha <= 0.0:
            # Reset history and retry Line Search with steepest descent and
            # FLOAT_EPSILON as inital step
            history.clear()
            search_direction = -grad

            alpha, current_cost = _line_search_wolfe(
                function, grad_func, params, search_direction, grad,
                current_cost, wolfe_c1, wolfe_c2, FLOAT_EPSILON, max_alpha,
                bracketing_max_iter, zoom_max_iter, zoom_epsilon
            )

            if alpha is None or alpha <= 0.0:
                converged = False
                termination_reason = (
                    "Line search failed to find a suitable step size. Gradient "
                    "probably is too high or too low making stepsize "
                    "numerically unstable. Consider changing to a trust region "
                    "algorithm."
                )
                break

        step_vector = alpha * search_direction # s_k
        params += step_vector

        grad_new = grad_func(params)
        grad_norm = float(np.linalg.norm(grad_new))

        grad_delta = grad_new - grad # y_k
        step_dot_grad_delta = np.dot(step_vector, grad_delta)

        # Anything below sqrt(eps) * ||s|| * ||y|| is roundoff noise, not real
        # curvature. See _bfgs_optimizer.
        curvature_tolerance = float(
            np.sqrt(FLOAT_EPSILON) * np.linalg.norm(step_vector) *
            np.linalg.norm(grad_delta)
        )
        # Curvature condition: y_k^T * s_k > 0 in exact arithmetic, here a
        # tolerance is added to prevent numerical issues if not met, skip
        # update completely (keep H_k^-1).
        if step_dot_grad_delta > curvature_tolerance:
            rho = 1.0 / step_dot_grad_delta

            # Deque droppes oldest automatically
            history.append((step_vector, grad_delta, rho))



        # Check termination at end of iteration so the convergent state is
        # available to _print_line_search_information via the converged flag.
        terminate, converged, termination_reason = _check_termination(
            current_cost, previous_cost, grad_norm, **tolerances
        )

        if verbose:
            _print_line_search_information(
                current_iter=current_iter, current_cost=current_cost,
                grad_norm=grad_norm, alpha=alpha, verbose_freq=verbose_freq,
                max_iters=max_iters, converged=converged
            )
        if callback:
            callback(
                current_iter, params, current_cost, grad, grad_norm, alpha,
                search_direction
            )

        if terminate: break

        grad = grad_new

    else:
        termination_reason = (f"Maximum number of iterations ({max_iters}) "
                              f"reached.")
        converged = False


    return {
        'success': converged,
        'final_params': params,
        'final_cost': current_cost,
        'iteration_number': current_iter,
        'termination_reason': termination_reason
    }
