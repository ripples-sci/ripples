"""
Trust-region algorithms for local optimization.

The trust-region algorithm is based in building a local quadratic approximation
of the objective function around the current point x_k:

    m_k(p) = f(x_k) + g_k^T * p + 0.5 * p^T * B_k * p

and approximately minimize this model subject to the trust-region
constraint:

    ||p|| <= delta_k

where g_k is the gradient, B_k is the Hessian (or a Hessian
approximation) and delta_k is the current trust-region radius.

The algorithms differ in how the trust-region subproblem is solved:
through a truncated Conjugate Gradient method that detects negative
curvature directions (Steihaug-Toint CG), or through a Krylov-subspace
projection using the Lanczos process and a reduced-dimensional secular
equation solve (GLTR / Lanczos trust-region method).

Contains
--------
_solve_trust_region_subproblem_cg
    Steihaug-Toint truncated Conjugate Gradient solver for the
    trust-region quadratic subproblem.

_solve_trust_region_subproblem_lanczos
    Generalized Lanczos Trust-Region (GLTR) solver that projects the
    subproblem onto a Krylov subspace and solves the reduced problem.

_trust_region_optimizer
    Generic trust-region optimization outer loop that updates the
    trust-region radius using the agreement between predicted and actual
    reduction.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Any, Optional, Union, Dict, Callable

from ._trust_region_utils import (
    _validate_trust_region_parameters, _compute_step_to_boundary,
    _solve_secular_equation_newton
)
from ._utils import FLOAT_EPSILON, _initial_check, _check_termination

from ..differentiation import numerical_hessian_vector_product



def _solve_trust_region_subproblem_cg(
    grad: np.ndarray,
    hvp_func: Callable[[np.ndarray], np.ndarray],
    trust_radius: float,
    max_iter: Union[int, str],
    grad_rel_tol: float,
    abs_tol: float
) -> Union[np.ndarray, str]:
    """
    Steihaug-Toint Conjugate Gradient solver for the Trust Region Subproblem.

    Minimizes:
        m(p) = g^T * p + 0.5 p^T * B * p
    Subject to:
        ||p|| <= delta

    Parameters
    ----------
    grad : np.ndarray
        The gradient vector (g).
    hvp_func : Callable
        Function that returns the Hessian-vector product (B * v).
    trust_radius : float
        The trust region radius (delta).
    max_iter : Union[int, str]
        Maximum iterations. If 'auto', defaults to dimension of grad.
    grad_rel_tol : float
        Tolerance multiplied by the gradient norm for the residual.
    abs_tol : float
        Absolute tolerance for the residual.

    Returns
    -------
    np.ndarray
        The approximate solution vector p. If the hvp function gives an invalid
        value (NaN or +-Inf) this function returns 'hvp_evaluation_error'.

    Mathematical Description
    ------------------------
    The goal is to approximately solve the Trust Region Subproblem for the step
    direction p. So build the following model as a second-order Taylor
    expansion of the current around the current params x_k taking a step p:

        minimize  f(x_k + p) ≈ m(p)
            = f(x_k) + g(x_k)^T * p + 0.5 * p^T * B(x_k) * p

        subject to  ||p|| <= delta

    where:
    - g(x_k)) is the gradient at the current point, x_k, corresponding to the
    current outer iteration k.
    - B(x_k) is the Hessian (or a symmetric approximation) at the current point,
    (x_k), corresponding to the current outer iteration k.
    - delta is the trust-region radius.


    Notice that the gradient of the model is, which is defined in the code(more
    details below) as a residual in order to know how far one is from the
    solution:

        gradient(m(p)) = g + B * p

    If B were positive definite and the unconstrained minimizer:

        p_sol = -B^-1 * g

    and satisfied ||p_sol|| <= delta, then p_sol would solve the subproblem.

    However, B may be indefinite, B^-1 is never formed explicitly and the
    constraint ||p|| <= delta must be enforced; the Steihaug-Toint method
    addresses these by applying a truncated Conjugate Gradient (CG),
    truncated because sometimes it terminates earlier than a standard CG
    method, directly to the quadratic model.

    ---

    The algorithm starts from an initial step direction:

        p_0 = 0

    then defines the residual:

        r := gradient(m(p)) = g + B * p

    which at the first iteration, i=0, is:

        r_0 = gradient(m(0)) = g

    This residual plays the same role as the gradient in standard CG.
    The initial search direction is chosen as steepest descent:

        d_0 = -r_0 = -g

    ---

    At each CG iteration i, the updates of the step have the form of:

        p_{i+1} = p_i + alpha_i * d_i

    where alpha_i minimizes the quadratic model along d_i.

    For the quadratic model m(p), the exact line minimizer is (derived in the
    Mathematical Description of the function _solve_newton_system_cg in
    _line_search_optimization.py):

        alpha_i = (r_i^T * r_i) / (d_i^T * B * d_i)

    and the curvature is defined as:

        d_i^T * B * d_i

    If the curvature is <= 0, then the model is non-convex along d_i, and m(p)
    decreases indefinitely in that direction.

    In this case, the optimal step lies on the trust-region boundary.
    The algorithm therefore computes:

        p = p_i + tau * d_i

    where tau > 0 is chosen such that:

        ||p_i + tau * d_i|| = delta

    Even if d_i^T * B * d_i > 0, the unconstrained step p_i + alpha_i * d_i may
    violate the trust-region constraint, in order to address this situacion,
    compute tau in ||p_i + tau * d_i|| = delta as just described in order to
    yield a step that lies exactly on the boundary.

    ---

    After a successful iteration, update step direction as:

        p_{i+1} = p_i + alpha_i * d_i

    and the residual as:

        r_{i+1} = r_i + B * p_{i+1} = r_i + alpha_i * B * d_i

    This avoids recomputing B * p, to accelerate convergence, the next search
    direction is formed as:

        d_{i+1} = -r_{i+1} + beta_i * d_i

    where beta_i is chosen using the Fletcher-Reeves formula (derived in the
    Mathematical Description of the function _solve_newton_system_cg in
    _line_search_optimization.py), as the residual is the gradient of the
    model, one can substitute directly:

        beta_i = r_{i+1}^T * r_{i+1} / (r_i^T * r_i).

    References
    ----------
    [1] Algorithm 7.2 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5
    """

    dim_num = grad.size

    step_direction = np.zeros_like(grad)

    residual = grad.copy()
    search_direction = -residual
    residual_dot_residual = np.dot(residual, residual)

    grad_norm = np.sqrt(residual_dot_residual)
    if grad_norm == 0.:
        return step_direction

    trust_radius_sq = trust_radius * trust_radius


    if max_iter == 'auto':
        # Heuristic to assure a minimum of iters
        # _max_iters = max(300, int(dim_num*1.25))
        _max_iters = max(20, 2*dim_num)
    else:
        _max_iters = int(max_iter)

    stop_tol = max(grad_rel_tol * grad_norm, abs_tol)
    stop_tol_sq = stop_tol * stop_tol

    step_norm_sq = 0.


    for _ in range(_max_iters):

        B_searchdir = hvp_func(search_direction)
        if not np.isfinite(B_searchdir).all():
            return 'hvp_evaluation_error'

        # Curvature along the search direction, d^T * B * d.
        searchdir_B_searchdir = float(np.dot(search_direction, B_searchdir))
        searchdir_dot_searchdir = float(
            np.dot(search_direction, search_direction)
        )

        step_dot_searchdir = float(np.dot(step_direction, search_direction))


        # if curvature is negative, then the model m(p) is unbounded along this
        # direction then go to boundary along search_direction
        if searchdir_B_searchdir <= 0.0:
            tau = _compute_step_to_boundary(
                step_dot_searchdir=step_dot_searchdir,
                searchdir_dot_searchdir=searchdir_dot_searchdir,
                step_norm_sq=step_norm_sq,
                trust_radius_sq=trust_radius_sq
            )

            return step_direction + tau * search_direction



        # Check if this step_direction goes outside trust region

        # if searchdir_B_searchdir is really small then alpha would be very
        # large leading to making the step to the boundary.
        alpha = residual_dot_residual / searchdir_B_searchdir
        # ||p + alpha*d||^2 = ||p||^2 + 2*alpha*p^T*d + alpha^2 * ||d||^2
        step_norm_sq_new = (
            step_norm_sq +
            2.0 * alpha * step_dot_searchdir +
            (alpha * alpha) * searchdir_dot_searchdir
        )
        if step_norm_sq_new >= trust_radius_sq:
            # Go to boundary along search_direction
            tau = _compute_step_to_boundary(
                step_dot_searchdir=step_dot_searchdir,
                searchdir_dot_searchdir=searchdir_dot_searchdir,
                step_norm_sq=step_norm_sq,
                trust_radius_sq=trust_radius_sq
            )

            return step_direction + tau * search_direction



        # If step_direction is inside the trust region the step is accepted:

        step_direction += alpha * search_direction
        step_norm_sq = step_norm_sq_new

        # Since residual = B * step_direction + g, and step_direction
        # increased, update residual: r_new = r_old + alpha * B * d
        residual_update = alpha * B_searchdir
        residual += residual_update

        residual_dot_residual_new = np.dot(residual, residual)
        if residual_dot_residual_new <= stop_tol_sq:
            return step_direction


        # Fleecher-Reeves beta
        beta = residual_dot_residual_new / residual_dot_residual

        search_direction = -residual + beta * search_direction

        residual_dot_residual = residual_dot_residual_new


    # None of the three Steihaug exit conditions (negative curvature, boundary
    # hit, residual small enough) has been met, so the best step_direction is
    # returned, it is strictly inside the trust region and improves on the zero
    # step.
    return step_direction



def _solve_trust_region_subproblem_lanczos(
    grad: np.ndarray,
    hvp_func: Callable[[np.ndarray], np.ndarray],
    trust_radius: float,
    max_iter: Union[int, str],
    grad_rel_tol: float,
    abs_tol: float,
) -> Union[np.ndarray,str]:
    """
    Solves the trust-region subproblem using a Generalized Lanczos Trust-Region
    (GLTR) approach.

    Minimizes the quadratic model:
        m(p) = f + g^T p + 1/2 p^T H p
    Subject to:
        ||p|| <= delta

    Parameters
    ----------
    grad : np.ndarray
        The gradient vector (g) of the objective function.
    hvp_func : Callable[[np.ndarray], np.ndarray]
        A function that returns the Hessian-vector product (H*v) for a
        given vector v.
    trust_radius : float
        The radius of the trust region (delta).
    max_iter : int, str
        The maximum dimension of the Krylov subspace (maximum Lanczos
        iterations).
    tol : float, optional
        Convergence tolerance for the residual.

    Returns
    -------
    np.ndarray, str
        The approximate solution vector p in the full parameter space. If the
        hvp function gives an invalid value (NaN or +-Inf) this function
        returns 'hvp_evaluation_error'.

    Mathematical Description
    ------------------------
    The trust-region subproblem consist of the minization of a quadratic
    approximation of the objective function within a ball of radius delta. The
    approximation is a second-order Taylor expansion of the current around the
    current params x_k taking a step p:

        min_p  f(x_k + p) ≈ m(p)
            = f(x_k) + g(x_k)^T * p + 1/2 * p^T * H(x_k) * p
        s.t. ||p|| <= delta

    where:
    - g(x_k)) is the gradient at the current point, x_k, corresponding to the
      current outer iteration.
    - H(x_k) is the Hessian (or a symmetric approximation) at the current point,
      (x_k), corresponding to the current outer iteration.
    - delta is the trust-region radius.

    For large-scale problems, forming H explicitly is too expensive.
    This algorithm projects the problem onto a lower-dimensional Krylov
    subspace generated by the gradient and the Hessian in order to restrict
    the solution p at ireration k, in the following way:

        K_k(H, g) = span{ g, H * g, H^2 * g, ..., H^(k-1) * g}

    This subspace captures increasingly accurate curvature information,
    because it tends to contain the directions where the function curves the
    most (the dominant eigenvectors of H) as k grows, while keeping
    computations efficient (only matrix-vector products with H are required):

    ---

    In order to create a basis in this subspace:

    - First, consider the Gram-Schmidt ortogonalization that, given the
    linearly independent vectors w1, w2, ... of K_k(H, g) described above:

        w_1 = g, w_2 = H * g, w_3 = H^2 * g, ...

    builds a orthonormal basis starting with q_1 = g / ||g|| and continues to
    construct q_2, q_3, ... via:

        v_k = ||v_k|| * q_k = w_k - sum_{j=1}^(k-1) ((q_j^T * w_k) * q_j)

    this gives q_k by normalizing v_k, this ensures:

        q_i^T * q_j = 0, when i != j and q_i^T * q_j = 1, when i = j.



    - Second, as expressed by the Gram-Schmidt ortogonalization combined with
    the Krylov subspace K_j(H, g) described above, the j-th vector of the
    orthornormal basis q_j can be expressed as a matrix polynomial p(H^(j-1),
    H^(j-2),...) of degree j-1, which is only a function of H, multiplied by
    the first vector of the basis q_1 = g / ||g||, that is:

        q_j = p(H^(j-1), H^(j-2),...) * q_1
            = p(H^(j-1), H^(j-2),...) * g / ||g||

    now, left-multiplying by H:

        H * q_j = H * pol(H^(j-1), H^(j-2),...) * g / ||g||

    at this point, H * pol is a matrix polynomial of degree j and once one
    multiplies by the gradient g, it is clear that it is contained in the
    Krylov subspace K_{j+1}(H, g), so:

        H * pol * g = H * q_j is contained in K_{j+1}(H,g)
                    = span{ g, H * g, ..., H^j * g}

    now, taking into account that the Gram-Schmidt ortogonalization projects
    the Krylov subspace K_{j+1}(H,g) into a orthogonal one made of q_1,
    q_2, ..., q_{j+1}, it can be written that:

        H * q_j is contained in K_{j+1}(H,g) = span{ g, H * g, ..., H^j * g}
                                             = span{ q_1, q_2, ..., q_{j+1}}

    which is intuitive if the reader thinks of a 1 by 1 relation between the
    two spans being projected by the Gram-Schmidt ortogonalization, so both
    span{ g, H * g, ..., H^j * g} and span{ q_1, q_2, ..., q_{j+1}} describe
    the same subspace.



    - Third, now keep in mind that a vector basis q_k is orthogonal to all
    other basis vectors (by the Gram-Schmidt ortogonalization), in other
    words:

        q_k is not contained in span{ q_1, q_2, ..., q_{k-1}}

    in addition, from to last expression of the second step:

        H * q_j is contained in span{ q_1, q_2, ..., q_{j+1}}

    one can modify it, by imposing j + 1 <= k - 1, so the Krylov subspace
    K_{j+1} is contained in K_{k-1}:

        H * q_j is contained in span{ q_1, q_2, ..., q_{k-1}}

    given these facts about H * q_j and g_k, one can multiply them, as just
    showed the first is contained in the span{ q_1, q_2, ..., q_{k-1}} and
    the second is not, so:

        (H * q_j)^T * q_k = 0, for j + 1 <= k - 1

    by manipulating this expression, given that the Hessian matrix H is
    symmetric, and that for two random vectors from the orthonormal basis q_a1,
    q_a2:

        q_a1^T * H * q_a2 = (H * q_a1)^T * q_a2,

    one gets:

        q_j^T * H * q_k = 0, for j <= k - 2

    concluding:

        H * q_k is contained in span{ q_{k-1}, q_k, q_{k+1}}



    - Fourth, given the last result, there exist scalar coefficients c1, c2, c3
    such that:

        H * q_k = c1 * q_{k-1} + c2 * q_{k} + c3 * q_{k+1}

    to obtain their value exploit q_i^T * q_j = 0 when i != j, so
    left-multiply by q_k^T:

        q_k^T * H * q_k = c1 * q_k^T * q_{k-1} + c2 * q_k^T * q_{k} +
                          c3 * q_k^T * q_{k+1}
                        = c2

    so, as the first and third term are 0 given that span{ q_1, q_2, ...,
    q_{k+1}} is a orthonormal basis:

        c2 = q_k^T * H * q_k = alpha_k,

    which is commonly called alpha in literature; continue left-multiplying
    by q_{k-1}^T:

        q_{k-1}^T * H * q_k = c1 * q_{k-1}^T * q_{k-1} +
                              alpha_k * q_{k-1}^T * q_{k} +
                              c3 * q_{k-1}^T * q_{k+1}
                            = c1

    so, by exactly the same reasoning:

        c1 = q_{k-1}^T * H * q_k = beta_{k-1}

    which is commonly called beta in literature; and last, left-multiplying
    by q_{k+1}^T:

        q_{k+1}^T * H * q_k = c1 * q_{k+1}^T * q_{k-1} +
                              alpha_k * q_{k+1}^T * q_{k} +
                              c3 * q_{k+1}^T * q_{k+1}
                            = c3

    so, by exactly the same reasoning:

        c3 = q_{k+1}^T * H * q_k = q_k^T * H * q_{k+1} = beta_k

    knowing all these results, the expression for H * q_k becomes:

        H * q_k = beta_{k-1} * q_{k-1} + alpha_k * q_k + beta_k * q_{k+1}

    where:

        alpha_k = q_k^T * H * q_k
        beta_k = q_k^T * H * q_{k+1}



    - Fifth, rewrite the last result such as:

        beta_k * q_{k+1} = H * q_k - alpha_k * q_k - beta_{k-1} * q_{k-1}

    expanding the right hand side gives:

        beta_k * q_{k+1} = H * q_k -
                           (q_{k-1}^T * H * q_k) * q_{k-1} -
                           (q_k^T * H * q_k) * q_k

    looking at the right hand side of this last equation one can notice that
    it is exactly a Gram-Schmidt ortogonalization for the vector H * q_k in
    the orthonormal basis span{q_1, q_2, ..., q_k}, where beta_k can be
    redefined in terms of previously calculated variables:

        beta_k = ||H * q_k - alpha_k * q_k - beta_{k-1} * q_{k-1}||

    Meaning that q_{k+1} can be computed efficiently by Hessian vector products
    (H * q_k) starting with H * q_1 = H * (g / ||g||) obtaining the whole basis
    as a result.



    This last result is the Lanczos recurrence that this code implements in
    order to create a basis in the Krylov subspace K_k(H, g) = span{ g, H * g,
    H^2 * g, ..., H^(k-1) * g}:

        (1) Initialize: q_1 = g / ||g||, beta_0 = 0

    Iteration k = 1, 2, 3, ..., until convergence or until a desired dimension
    is reached:

        (2) Hq_k = H * q_k

        (3) alpha_k = q_k^T * Hq_k

        (4) beta_k__q_k_plus_1 = Hq_k - alpha_k * q_k - beta_{k-1} * q_{k-1}

        (5) beta_k = ||beta_k__q_k_plus_1||

        (6) q_{k+1} = beta_k__q_k_plus_1 / beta_k

    ---

    From the orthonormal basis defined by the Lanczos recurrence one can
    create a matrix for changing H to the basis span{q_1, q_2, ..., q_k}:

        Q_k = [q_1, ..., q_k]

    in order to perform the change:

        Q_k^T * H * Q_k,

    at row i and column j the resulting matrix is:

        [Q_k^T * H * Q_k]_{i,j} = q_i^T * H * q_j,

    expanding now H * q_j with the Lanczos recurrence, we get that at row i and
    column j:

        [Q_k^T * H * Q_k]_{i,j} = q_i^T * H * q_j
                                = q_i^T * (beta_{j-1} * q_{j-1} +
                                           alpha_j * q_j +
                                           beta_j * q_{j+1})
                                =  beta_{j-1} * (q_i^T * q_{j-1}) +
                                   alpha_j * (q_i^T * q_j) +
                                   beta_j * (q_i^T * q_{j+1})

    using the orthonormality q_i^T * q_j = 0 when i != j, and q_i^T * q_j = 1
    when i = j:

    - for the diagonal entries:

        [Q_k^T * H * Q_k]_{i,i} = alpha_i

    - for the superdiagonal (one position above diagonal):

        [Q_k^T * H * Q_k]_{i,i+1} = beta_i

    - for the subdiagonal (one position below diagonal):

        [Q_k^T * H * Q_k]_{i,i-1} = beta_{i-1}

    - for all others, when |i - j| >= 2 all are 0.

    Concluding that:

        [ alpha_1  beta_1                        0         ]
        [ beta_1   alpha_2  beta_2                         ]
        [          beta_2   alpha_3  ...                   ]
        [                   ...      ...         beta_{k-1}]
        [ 0                          beta_{k-1}  alpha_k   ]

    = Q_k^T * H * Q_k

    which is a symmetric tridiagonal matrix, named T_k, by the change of basis
    to the parameters of the basis q_1, ..., q_k:

        T_k := Q_k^T * H * Q_k

    ---

    The quadratic model is transformed into the Krylov subspace, starting with
    the step p:

        p = Q_k * y

    Substituing into the quadratic model yields a reduced problem:

        min_y  m(y) = g^T * (Q_k * y) + 1/2 * (Q_k * y)^T * H * (Q_k * y)
        s.t. ||Q_k * y|| <= delta

    by expanding m(y), one obtains:

    -The first term:

        g^T * Q_k * y = ||g|| * q_1^T * Q_k * y = ||g|| * e_1^T * y,
        with e_1 = (1, 0, ..., 0)^T

    -The second term:

        (Q_k * y)^T * H * (Q_k * y) = y^T * Q_k^T * H * Q_k * y = y^T * T_k * y

    -The constraint term:

        ||p|| = ||Q_k * y|| = ||y||

    because Q_k is orthonormal, Q_k^T * Q_k = I, so the norm is preserved and
    the trust-region constraint remains a simple Euclidean ball in the reduced
    space.

    inserting these results into the quadratic model, gives the reduced model
    implemented:

        min_y  m(y) = ||g|| * e_1^T * y + 1/2 * y^T * T_k * y
        s.t.   ||y|| <= delta

    ---

    In order to solve this last reduced problem, min_y  m(y), it can be shown
    [2, Theorem 4.1] that if and only if y_sol is a feasible global solution
    of m(y) then there exist a scalar lambda >= 0 (which can be thought as a
    Lagrange multiplier) that satisfies:

        (T_k + lambda * I) * y_sol = -||g|| * e_1
        lambda * (delta - ||y_sol||) = 0
        T_k + lambda * I is positive semidefinite

    by rearranging the first equation:

        lambda * I * y_sol = lambda * y_sol
                           = -||g|| * e_1 - T_k * y_sol
                           = -grad(m(y_sol)) = 0

    which is exactly the negative gradient of m(y) at y = y_sol, so lambda will
    be exactly zero, and positive otherwise as:

        lambda * y = -grad(m(y)).

    ---

    Now, an eigendecomposition of the tridiagonal matrix T_k is performed as it
    is really unexpensive:

        T_k = Eigvecs * Eigvals * Eigvecs^T

    where:
    - Eigvals is a diagonal matrix that contains the eigenvalues eig_i in
    ascending order.
    - Eigvecs is a mtrix that contains at each column i the eigenvectors
    corresponding to the eig_i from eigvals, that is
    [eigvec_1 | eigvec_2 | ... | eigvec_k].

    Note that:

        T_k + lambda * I = Eigvecs * Eigvals * Eigvecs^T +
                           lambda * Eigvecs * I * Eigvecs ^T
                        = Eigvecs * (Eigvals + lambda * I) * Eigvecs^T

    Therefore:

        (T_k + lambda * I)^-1 = Eigvecs * (Eigvals + lambda * I)^-1 * Eigvecs^T

    and Eigvals + lambda * I^-1 is diagonal and trivial to invert:

        (Eigvals + lambda * I)^-1
            = diag(1/(eig_1 + lambda), 1/(eig_2 + lambda),
                   ..., 1/(eig_k + lambda))


    Taking into account the first equation of the last section:

        (T_k + lambda * I) * y_sol = -||g|| * e_1

    and rearranging, one can substitute:

        y_sol = - (T_k + lambda * I)^-1 * ||g|| * e_1
              = - Eigvecs * (Eigvals + lambda * I)^-1 * Eigvecs^T *
                  (||g|| * e_1)

    the 2 last products can be expressed as:

        Eigvecs^T * (||g|| * e_1) = ||g|| * Eigvecs[1,:]

    so:

        y_sol = - Eigvecs * (Eigvals + lambda * I)^-1 * Eigvecs^T *
                  (||g|| * e_1)
              = - Eigvecs * (Eigvals + lambda * I)^-1 * ||g|| * Eigvecs[1,:]
              = - Eigvecs * diag(1/(eig_1 + lambda), 1/(eig_2 + lambda), ...,
                                 1/(eig_k + lambda)) * ||g|| * Eigvecs[1,:]

    where Eigvecs[1,:] is the first row of the Eigvecs matrix, containing the
    first component of every eigenvector, in the code computed as Eigvecs[0, :]
    as 0 is the first. This last equation is the one implemented in code to
    compute y, the process is detailed in the next part below for each case.


    Lastly, the norm of the solution, ||y_sol||, can be computed by:

    from the 2 first products of the y_sol definition:

        Eigvecs * (Eigvals + lambda * I)^-1
            = [eigvec_1 | eigvec_2 | ... | eigvec_k] *
               diag(1/(eig_1 + lambda), 1/(eig_2 + lambda), ...,
                    1/(eig_k + lambda))

            = (eigvec_1 * 1/(eig_1 + lambda),
               eigvec_2 * 1/(eig_2 + lambda),
               ...,
               eigvec_k * 1/(eig_k + lambda))

    the expression for y_sol changes for:

        y_sol = - Eigvecs * (Eigvals + lambda * I)^-1 * ||g|| * Eigvecs[1,:]
              = [eigvec_1 * 1/(eig_1 + lambda),
                 eigvec_2 * 1/(eig_2 + lambda),
                 ...,
                 eigvec_k * 1/(eig_k + lambda)] * ||g|| * Eigvecs[1,:]

    since Eigvecs = [eigvec_1 | eigvec_2 | ... | eigvec_k]:

        y_sol = - ||g|| * (eigvec_1 * (1/(eig_1 + lambda)) * Eigvecs[1,1] +
                           eigvec_2 * (1/(eig_2 + lambda)) * Eigvecs[1,2] +
                           ... +
                           eigvec_k * (1/(eig_k + lambda)) * Eigvecs[1,k])^T

    so, the equation for y_sol is:

        y_sol = sum_{i=1}^k  (-||g|| * Eigvecs[1,i]/(eig_i + lambda)) * eigvec_i

    And finally the norm implemented in the code is:

        ||y_sol||^2 = sum_{i=1}^k  (||g|| * Eigvecs[1,i]/(eig_i + lambda))^2

    ---

    Now, two cases arise when solving the reduced problem:

    - First Case: Interior Solution (lambda = 0)

    If T_k is positive definite (the minimum eigenvalue of T_k: eig_1 > 0) and
    the unconstrained Newton step satisfies ||y|| <= delta, then knowing the
    last result for the soluton y_sol, it is computed in the code by:

        y_sol = - Eigvecs * diag(1/(eig_1 + lambda), 1/(eig_2 + lambda), ...,
                                 1/(eig_k + lambda)) * ||g|| * Eigvecs[1,:]

    so (first row in python is row 0, so Eigvecs[1,:] in this Mathematical
    Description means Eigvecs[0, :] in the code):

        (1) Eigvecs_first_components = -||g|| * Eigvecs[0, :]

        (2) y_coeffs_unconstrained = Eigvecs_first_components / eigvals

        (3) y_sol = y_unconstrained = Eigvecs @ y_coeffs_unconstrained


    -Second Case: Boundary Solution (lambda > 0), if T_k is indefinite or the
    unconstrained step violates the trust-region constraint, the
    solution lies on the boundary:

        ||y_sol(lambda)|| = delta

    The value of lambda is then found by solving the secular equation:

        1 / ||y_sol(lambda)|| - 1 / delta = 0

    which is monotonic and strictly convex for valid lambda. As described in
    its dedicated function (_solve_secular_equation_newton) the Newton's method
    is used to solve this equation efficiently.


    Once y is computed (interior or boundary), the full-space step is:

        p_sol = Q_k * y_sol

    which is the solution of the subproblem.

    ---

    The quality of the current solution is measured using the residual of the
    optimality conditions in the full space:

        r_k = H * p + g

    which can be computed effiently using already known information, by
    introducing p = Q_k * y:

        r_k = H * Q_k * y + g


    Now, notice that when k is the current iteration:

        T_k * e_k = [0, ..., 0, beta_{k-1}, alpha_k]^T,

    where e_k = [0, ..., 1]^T which has k dimensions, meaning that T_k * e_k is
    nonzero only at positions k-1 and k; left-multiplying by Q_k gives:

        Q_k * T_k * e_k = beta_{k-1} * q_{k-1} + alpha_k * q_k

    recalling from step fourth in the derivation of the Lanczos basis:

        H * q_k = beta_{k-1} * q_{k-1} + alpha_k * q_k + beta_k * q_{k+1}

    allows to substitute directly obtaining:

        H * q_k = Q_k * T_k * e_k + beta_k * q_{k+1}

    now, this last equation can be expanded to all iterations, not only the
    current one:

    - First Term:

    For 1 <= i <= k:

        H * q_i is the i-th column of H * Q_k.

    - Second Term, given that:

    For 1 <= i <= k-1:

        T_i * e_i = [0, ..., beta_{i-1}, alpha_i, beta_i, ..., 0]^T
        (nonzero only at positions i-1, i, i+1)

    so:

        Q_i * T_i * e_i
            = beta_{i-1} * q_{i-1} + alpha_i * q_i + beta_k * q_{i+1}

    And, for i = k:

        T_k * e_k = [0, ..., 0, beta_{k-1}, alpha_k]^T,

    so:

        Q_i * T_i * e_i = beta_{i-1} * q_{i-1} + alpha_i * q_i

    so one can conclude that:

        Q_i * T_i * e_i is the i-th column of Q_k * T_k, for 1 <= i <= k

    - Third Term, notice how in order to complete the equation there is a term
    missing with q_k{k+1} that will be needed to be applied to the last column
    of Q_k * T_k in order to be equal to H * Q_k, this term is obtained by
    expanding the third term (left-multiplying by * e_k^T) so it is a matrix
    with all zeros except for the k-th column:

        beta_k * q_{k+1} * e_k^T

    The expansion is completed, it can be expressed as:

        H * Q_k = Q_k * T_k + beta_k * q_{k+1} * e_k^T


    Inserting this result into the residual definition:

        r_k = H * Q_k * y + g = (Q_k * T_k + beta_k * q_{k+1} * e_k^T) * y + g

    expanding:

        r_k = Q_k * T_k * y + beta_k * q_{k+1} * e_k^T * y + g

    in this last expression and using the result when obtained from
    [2, Theorem 4.1] (making y_sol a trial possible solution y):

        (T_k + lambda * I) * y = -||g|| * e_1

    rearranged:

        T_k * y = -||g|| * e_1 - lambda * y

    left-multiplying by Q_k gives

        Q_k * (T_k * y) = Q_k * (-||g|| * e_1 - lambda * y)
                        = -||g|| * (Q_k * e_1) - lambda * (Q_k * y)
                        = -||g|| * q_1 - lambda * p

    introducing this result in the residual equation and noting that e_k^T * y
    is the last component of y:

        r_k = -||g|| * q_1 - lambda * p + beta_k * (e_k^T * y) * q_{k+1} + g

    but g = ||g|| * q_1, so:

        r_k = -||g|| * q_1 - lambda * p +
              beta_k * (e_k^T * y) * q_{k+1} + ||g|| * q_1
            = - lambda * p + beta_k * (e_k^T * y) * q_{k+1}

    Finally, by making the norm of both sides:

        ||r_k|| = ||lambda * p|| + ||beta_k * (e_k^T * y) * q_{k+1}||
                = lambda * ||p|| + |beta_k * (e_k^T * y)|

    where:
    - beta_k is the current Lanczos coefficient,
    - (e_k^T * y) is the last component of y in the reduced basis.

    which provides, when sufficiently small, a stopping condition to the
    algorithm for the uncontrained case, lambda = 0:

        ||r_k|| = |beta_k * (e_k^T * y)|

    and the constrained case, as ||p|| is really close to the trust radius
    delta:

        ||r_k|| = lambda * delta + |beta_k * (e_k^T * y)|

    finally making the current accepted step the solution.

    If beta_k ≈ 0, the Krylov subspace becomes invariant under H.
    In this case, the reduced problem is exact and the algorithm
    terminates immediately.

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

    dim_num = grad.shape[0]
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm == 0.0:
        return np.zeros_like(grad)

    if max_iter == 'auto':
        # Up to 30 Lanczos vectors, for low-dimensional problems dim_num + 1
        # is enough to span the full Krylov basis, and for high-dimensional
        # more than 30 vectors is not worth it provided that each iteration
        # needs one hvp evaluation and one np.linalg.eighh.
        _max_iters = min(dim_num + 1, 30)
    else:
        _max_iters = int(max_iter)

    # Lanczos basis Q first vector: q_1 = g / ||g||
    Lanczos_basis = [grad / grad_norm]

    alphas = []
    betas = []

    trust_radius_sq = trust_radius**2
    tolerance_val = max(grad_rel_tol * grad_norm, abs_tol)

    y_solution = np.zeros(1)
    basis_size_for_y_solution = 1

    PRECISION_FLOOR = np.sqrt(FLOAT_EPSILON)


    # Lanczos Loop
    for current_iter in range(_max_iters):

        # Build the Lanczos basis, using the Lanczos recurrence

        Lanczos_vec_k = Lanczos_basis[-1] # q_k
        H_dot_Lanczos_vec_k = hvp_func(Lanczos_vec_k) # Hq_k

        if not np.isfinite(H_dot_Lanczos_vec_k).all():
            return 'hvp_evaluation_error'

        alpha_k = float(np.dot(Lanczos_vec_k, H_dot_Lanczos_vec_k))
        alphas.append(alpha_k)

        # Orthogonalization:
        beta_k__q_k_plus_1 = H_dot_Lanczos_vec_k - alpha_k * Lanczos_vec_k
        if current_iter > 0:
            beta_k__q_k_plus_1 -= betas[-1] * Lanczos_basis[-2]

        beta_k = float(np.linalg.norm(beta_k__q_k_plus_1))
        betas.append(beta_k)

        # Check for Lanczos Breakdown (invariant subspace found)
        # If beta is near zero, the Krylov subspace has captured the exact
        # solution.
        numerical_breakdown = beta_k < PRECISION_FLOOR

        if not numerical_breakdown and current_iter < _max_iters - 1:
            # Column k+1 of the basis Q: q_{k+1}
            Lanczos_basis.append(beta_k__q_k_plus_1 / beta_k)



        # T_k, size: (current_iter+1,current_iter+1)
        Tridiagonal = (np.diag(alphas[:current_iter + 1]) +
                       np.diag(betas[:current_iter], 1) +
                       np.diag(betas[:current_iter], -1))

        # Eigen-decomposition of T_k, as it is real symmetric, use eigh
        try:
            eigvals, Eigvecs = np.linalg.eigh(Tridiagonal)
        except np.linalg.LinAlgError:
            break
        # eigvals = [eig_1, eig_2, ..., eig_k] is a vector of eigenvalues with
        # eig_1 <= eig_2 <= ... <= eig_k
        # Eigvecs = [eigvec_1 | eigvec_2 | ... | eigvec_k] contains the
        # corresponding orthonormal eigenvectors as columns


        # Solve the reduced subproblem


        # In Mathematical Description: -||g|| * Eigvecs[1,:], that is the first
        # component of Eigvecs, which is 0 in Python
        g_norm__Eigvecs_first_component = - grad_norm * Eigvecs[0, :]

        step_is_interior = False

        # If T_k is positive definite, then the solution is the newton
        # unconstrained solution.
        # As the eigenvalues are given in ascending order, the first
        # is the smallest, so min eig_i = eig[0] must be > 0
        if eigvals[0] > 0.:

            # As lambda = 0 in the unconstrained solution:
            y_coeffs_unconstrained = (
                g_norm__Eigvecs_first_component  /
                np.maximum(eigvals, FLOAT_EPSILON)
            )

            # Check if the Newton step is inside the trust region
            norm_y_sq = np.sum(y_coeffs_unconstrained**2)
            if norm_y_sq <= trust_radius_sq:

                y_solution = Eigvecs @ y_coeffs_unconstrained
                basis_size_for_y_solution = current_iter + 1

                residual = abs(beta_k * y_solution[-1])
                if residual <= tolerance_val:
                    break

                step_is_interior = True


        # Boundary Solution if T_k is indefinite or boundary condition is not
        # met (norm(y) > trust_radius)
        if not step_is_interior:

            lambda_optimal = _solve_secular_equation_newton(
                eigvals, g_norm__Eigvecs_first_component , trust_radius
            )

            # Reconstruct y with optimal lambda
            inv_diag = np.maximum(eigvals + lambda_optimal, FLOAT_EPSILON)
            y_coeffs_boundary = g_norm__Eigvecs_first_component  / inv_diag

            y_solution = Eigvecs @ y_coeffs_boundary
            basis_size_for_y_solution = current_iter + 1

            residual_bounded = (
                lambda_optimal * trust_radius +
                abs(beta_k * y_solution[-1])
            )

            if residual_bounded <= tolerance_val:
                break


        if numerical_breakdown:
            break

    # Conversion from Krylov to canonial basis
    Lanczos_basis_array = np.column_stack(
        Lanczos_basis[:basis_size_for_y_solution]
    )
    return Lanczos_basis_array @ y_solution



def _trust_region_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    hvp_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int,np.ndarray,float,np.ndarray,float,
                                 np.ndarray,float], None]],
    inner_max_iter: int,
    grad_rel_tol: float,
    abs_tol: float,
    hvp_h: Union[str, int],
    hvp_point_number: int,
    initial_trust_radius: float,
    max_trust_radius: float,
    eta: float,
    rho_make_smaller_threshold: float,
    rho_make_bigger_threshold: float,
    make_trust_radius_smaller_multiplier: float,
    make_trust_radius_bigger_multiplier: float
) -> Dict[str, Any]:
    """
    Trust Region optimization algorithm.

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
        - 'inner_method': str ('ncg' or 'lanczos')
    callback : Optional[Callable]
        Function called after each iteration with signature
        (iter, params, cost, gradient, gradient norm, step_direction,
        trust radius).
    inner_max_iter : int
        Maximum number of iterations for the inner subproblem solver.
    grad_rel_tol : float
        Tolerance multipled by the gradient norm at current point for the inner
        subproblem solver.
    abs_tol : float
        Absolute tolerance for the inner subproblem solver.
    hvp_h : Union[str, int]
        Step size for finite-difference Hessian-vector products.
    hvp_point_number : int
        Number of points used in the for finite-difference Hessian Vector
        product function. Must be even.
    initial_trust_radius : float
        Initial size of the trust region (delta).
    max_trust_radius : float
        Maximum allowable size for the trust region.
    eta : float
        Minimum rho (actual/predicted reduction) to accept a step.
    rho_make_smaller_threshold : float
        Threshold for rho below which the trust radius is shrunk.
    rho_make_bigger_threshold : float
        Threshold for rho above which the trust radius is expanded.
    make_trust_radius_smaller_multiplier : float
        Factor to shrink the trust radius.
    make_trust_radius_bigger_multiplier : float
        Factor to grow the trust radius.

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
    The Trust Region method iterates by defining a region around the
    current point x_k within which a quadratic model m_k(p) is trusted
    to accurately approximate the objective function f(x).

    The quadratic model is defined as the second-order Taylor expansion of f(x)
    around x_k taking a step p:

        f(x_k + p) ≈ m_k(p) = f(x_k) + g_k^T * p + 0.5 * p^T * B_k * p

    where:
    - g_k is the gradient at the current point, x_k, corresponding to the
    current iteration.
    - B_k is the Hessian (or an approximation) at the current point,
    x_k, corresponding to the current iteration.

    At each iteration, the Trust Region Subproblem needs to be solved:

        min  m_k(p)
        s.t. ||p|| <= delta_k

    where delta_k is the current trust radius.

    ---

    One can solve the Subproblem without the need to compute the full
    Hessian B_k is not needed. Instead, one can use iterative methods
    that only require Hessian-Vector Products (HVPs): B_k * v.

    - 'ncg': Uses the Steihaug-Toint Conjugate Gradient algorithm. It
    approximates the solution to the subproblem by traversing the
    CG path until the trust region boundary is hit or negative
    curvature is encountered.

    - 'lanczos': Uses the Generalized Lanczos Trust Region (GLTR) method.
    It solves the subproblem restricted to a Krylov subspace, often
    yielding a more accurate solution on the boundary than NCG.

    ---

    Once a candidate step p_k is computed, evaluate the quality of the
    quadratic model by comparing the Actual Reduction in f(x) to the
    Predicted Reduction in m_k(p).

    Actual Reduction:

        f(x_k) - f(x_k + p_k)

    Predicted Reduction:

        m_k(0) - m_k(p_k) = -(g_k^T * p_k + 0.5 * p_k^T * B_k * p_k)

    The ratio rho_k is defined as:

        rho_k := Actual Reduction / Predicted Reduction

    ---

    So, the Trust Radius Update Logic is the following:

    - rho_k > eta: The step is accepted. x_{k+1} = x_k + p_k.
    - rho_k < eta: The step is rejected. x_{k+1} = x_k.

    At each iteration the Trust Radius is adjusted:

    - If rho_k < rho_make_smaller_threshold: The model is a poor
    approximation. Shrink the trust region.

    - If rho_k > rho_make_bigger_threshold AND ||p_k|| approx delta_k:
    The model is accurate and it were constrained by the radius.
    Expand the trust region.

    - Otherwise: Keep the trust radius unchanged.

    This approach allows the algorithm to take large steps when the
    model is good (Newton-like convergence) and restrict steps to a
    small descent region when the model is poor (Gradient Descent-like
    behavior), ensuring global convergence and robustness.

    References
    ----------
    [1] Algorithm 4.1 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5
    """

    _validate_trust_region_parameters(
        grad_rel_tol, abs_tol, eta, rho_make_smaller_threshold,
        rho_make_bigger_threshold, make_trust_radius_smaller_multiplier,
        make_trust_radius_bigger_multiplier,initial_trust_radius,
        max_trust_radius, inner_max_iter
    )

    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    verbose_freq = common_options['verbose_freq']
    inner_method = common_options['inner_method']
    tolerances = common_options['tolerances']

    params = initial_params.copy()

    grad = grad_func(params)
    if grad.size != params.size:
        raise ValueError(f"Provided initial params (length = {params.size}) "
                         f"mismatch its dimensions with provisded gradient "
                         f"(length = {grad.size}).")

    grad_norm = float(np.linalg.norm(grad))
    current_cost = function(params)

    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check


    if inner_method == 'lanczos':
        subproblem_solver = _solve_trust_region_subproblem_lanczos
    else: # inner_method == 'ncg':
        subproblem_solver = _solve_trust_region_subproblem_cg


    trust_radius = initial_trust_radius


    # params change at each iteration, so the current params are passed to
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


    if verbose:
        print(f"{'Iter':>5s} | {'Cost':>12s} | {'Grad Norm':>12s} "
              f"| {'Trust Radius':>14s} | {'Rho':>7s}")
        print("-" * 60)
        print(f"{0:5d} | {current_cost:12.6f} | {grad_norm:12.2e} "
              f"| {initial_trust_radius:14.2e} | -")


    BOUNDARY_PROXIMITY_FACTOR = 0.99
    terminate = False
    converged = False

    for current_iter in range(1, max_iters + 1):

        step_direction = subproblem_solver(grad, _hvp_func, trust_radius,
            inner_max_iter, grad_rel_tol, abs_tol)
        if not isinstance(step_direction,np.ndarray):
            # usually because step_direction is 'hvp_evaluation_error'
            converged = False
            termination_reason = ("Subproblem solver failed: the "
                                  "Hessian-vector product returned an "
                                  "invalid value (NaN or +-Inf).")
            break

        hvp_step_direction = _hvp_func(step_direction)
        if not isinstance(hvp_step_direction,np.ndarray):
            converged = False
            termination_reason = ("Hessian-vector product evaluation at "
                                  "the trial step returned an invalid "
                                  "value (NaN or +-Inf).")
            break


        # Model using second order Taylor series for reduction
        predicted_reduction = -(
            np.dot(grad, step_direction) +
            0.5 * np.dot(step_direction, hvp_step_direction)
        )

        # If the model predicts no reduction, or an increase, it's a very bad
        # step.
        if predicted_reduction <= 0.:
            rho = -1.0
            new_cost = current_cost
            step_accepted = False

        else:
            trial_params = params + step_direction
            new_cost = function(trial_params)

            if not np.isfinite(new_cost):
                rho = -1.0
                step_accepted = False
            else:
                actual_reduction = current_cost - new_cost

                rho = actual_reduction / max(predicted_reduction, FLOAT_EPSILON)

                # Only accept the step if the agreement is good enough
                step_accepted = rho > eta

        if rho < rho_make_smaller_threshold:
            trust_radius *= make_trust_radius_smaller_multiplier
            if trust_radius < FLOAT_EPSILON:
                converged = True
                termination_reason = (
                    "Trust radius collapsed below machine epsilon (~2.2e-16). "
                    "Point is likely a critical point but the quadratic model "
                    "can`t be trusted anymore."
                )
                break

        elif rho > rho_make_bigger_threshold:

            if (
                float(np.linalg.norm(step_direction)) >=
                trust_radius * BOUNDARY_PROXIMITY_FACTOR
            ):

                trust_radius = min(
                    make_trust_radius_bigger_multiplier * trust_radius,
                    max_trust_radius
                )

        if step_accepted:
            params = trial_params

            grad = grad_func(params)
            grad_norm = float(np.linalg.norm(grad))

            terminate, converged, termination_reason = _check_termination(
                new_cost, current_cost, grad_norm, **tolerances
            )

            current_cost = new_cost


        if verbose and (current_iter % verbose_freq == 0 or terminate):
            print(f"{current_iter:5d} | {current_cost:12.6f} "
                  f"| {grad_norm:12.2e} | {trust_radius:14.2e} | {rho:7.3f}")
        if callback:
            callback(
                current_iter, params, current_cost, grad, grad_norm,
                step_direction, trust_radius
            )


        if step_accepted:
            if terminate: break


        if not np.isfinite(new_cost) or not np.isfinite(params).all():
            converged = False
            termination_reason = ("Invalid value (NaN or +-Inf) "
                                  "encountered while evaluating the "
                                  "objective function or its gradient.")
            break

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
