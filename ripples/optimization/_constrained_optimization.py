"""
Augmented Lagrangian algorithm for constrained optimization.

This module wraps any unconstrained inner solver into a constrained
optimizer for the general smooth problem

    minimize f(x)
    subject to  c_i(x)  = 0   (equality constraints, i in E)
                c_j(x) >= 0   (inequality constraints, j in I)
                lb <= x <= ub (box bounds)

Equality and active inequality constraints contribute both a Lagrange
multiplier term and a quadratic penalty so the augmented Lagrangian
stays bounded below even at infeasible iterates. Box bounds are enforced
through a pure quadratic penalty (no multiplier) with an independent scale
factor.

Contains
--------
_augmented_lagrangian_optimizer
    Augmented Lagrangian outer loop that wraps an unconstrained inner
    solver and handles equality constraints, inequality constraints
    and box bounds simultaneously. Updates multipliers and the penalty
    parameter using the adaptive feasibility-tolerance rule of
    Nocedal & Wright (Algorithm 17.4), and supports an optional
    unconstrained warm start as well as noise-perturbed restarts that
    keep the best-feasible iterate across runs.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Any, Optional, Tuple, Dict, Callable

from ._utils import OptimizationResult

from ..differentiation import nth_numerical_derivative



def _augmented_lagrangian_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    inner_solver: Callable[[Callable,np.ndarray,Callable], OptimizationResult],
    constraints: Optional[Tuple[dict, ...]],
    bounds: Tuple,
    verbose_outer: bool,
    verbose_freq_outer: int,
    callback: Optional[Callable[[int, int, np.ndarray, float, float, float,
                                 np.ndarray, float],None]],
    max_outer_iter: int,
    mu_init: float,
    mu_factor: float,
    tol_constraints: float,
    tol_grad: float,
    const_gradient_point_number: int,
    bounds_penalty_scale: float,
    make_initial_param_unconstrained: bool,
    maximum_restarts: int,
    satisfaction_threshold: float,
    restart_noise_amplitude: float,
    seed: int
) -> Dict[str, Any]:
    """
    Augmented Lagrangian Constrained Optimizer.

    This function transforms a constrained optimization problem into a
    sequence of unconstrained sub-problems using the Augmented Lagrangian
    method. It wraps an internal solver to find the minima of the penalized
    objective function.

    The method handles equality constraints, inequality constraints, and
    bounds by incorporating them into the Lagrangian with both Lagrange
    multipliers and quadratic penalties.

    Parameters
    ----------
    function : Callable
        The objective function to be minimized.
    grad_func : Callable
        The gradient of the objective function.
    initial_params : np.ndarray
        The starting point for the optimization. Shape `(n_params,)`.
    inner_solver : Callable
        The solver for the unconstrained sub-problems. It must accept
        the signature (func, x0, grad_func) and return an
        OptimizationResult whose attributes are read here:

        - final_params: np.ndarray, the solution found by the
        inner solver.
        - final_cost: float, the objective value at
        final_params.
        - success: bool, whether the inner solver reported
        convergence.
        - termination_reason: str, propagated when the outer loop
        stops on the inner solver's verdict.
        - func_evals_number: int, count of objective evaluations.
        - grad_evals_number: int, count of gradient evaluations.

    constraints : Tuple[dict], optional
        A Tuple of constraint dictionaries. Each dictionary must contain:
        - 'type': str, either 'eq' (equality) or 'ineq' (inequality).
        - 'fun': Callable, the constraint function `c(x)`.
        - 'grad': Callable (optional), the gradient of the constraint.
            If None, finite differences are used.

        Convention:
        - Equality: `c(x) = 0`.
        - Inequality: `c(x) >= 0` is satisfied (feasible).
    bounds : Tuple
        A tuple `(lower_bounds, upper_bounds)` defining the box
        constraints. Each element is an array of shape `(n_params,)`.
    verbose_outer : bool
        If True, prints progress details for the outer (Augmented
        Lagrangian) loop.
    verbose_freq_outer : int
        The frequency (in iterations) at which to print verbose messages.
    callback : Callable, optional
        A function called after every outer iteration. Signature:
        `(run_num, iter_num, params, cost, infeasibility, mu, multipliers,
        grad_norm)`.
    max_outer_iter : int
        The maximum number of iterations for the Augmented Lagrangian loop
        (updates to multipliers and penalty parameters).
    mu_init : float
        The initial penalty parameter (mu) for the quadratic term.
    mu_factor : float
        The factor by which `mu` is multiplied if the constraint violation
        does not decrease sufficiently.
    tol_constraints : float
        The tolerance for maximum constraint violation. A solution is
        feasible if violation <= tol_constraints.
    tol_grad : float
        The tolerance for the norm of the gradient of the Lagrangian. Used
        as a convergence criterion.
    const_gradient_point_number : int
        The number of points used in the central difference formula if a
        gradient for some constraint is not provided. Must be an even
        integer.
    bounds_penalty_scale : float
        A scaling factor for the quadratic penalty applied to box bound
        violations.
    make_initial_param_unconstrained : bool
        If True, attempts to run the `inner_solver` on the unconstrained
        objective first to find a better starting point (bypassing
        constraints initially).
    maximum_restarts : int
        The number of additional times to restart the optimizer with
        perturbed parameters if convergence is not reached.
    satisfaction_threshold : float
        If a feasible solution with cost lower than this threshold is
        found, the optimizer terminates early.
    restart_noise_amplitude : float
        The magnitude of the noise added to the initial parameters during
        a restart.
    seed : int
        Seed for the random number generator used in restarts.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the optimization results:
        - 'success': bool, True if convergence criteria were met.
        - 'final_params': np.ndarray, the best parameters found.
        - 'final_cost': float, the function value at the best parameters.
        - 'termination_reason': str, a description of why the optimizer
        stopped.
        - 'func_evals_number': int, total internal function evaluations.
        - 'grad_evals_number': int, total internal gradient evaluations.

    Mathematical Description
    ------------------------
    Starting with the general constrained optimization problem:

        minimize f(x)

        s.t.   c_i(x) = 0   for i in E   (equality constraints)
               c_j(x) >= 0  for j in I   (inequality constraints)
               lb <= x <= ub             (bounds)

    where:
    - f(x) is a smooth objective function,
    - c_i(x) are smooth constraint functions,
    - E and I index equality and inequality constraints respectively,
    - lb and ub are lower and upper bound vectors.

    ---

    A classical approach is to form the (standard) Lagrangian:

        L(x, lambda) = f(x) -
                       sum_i(i in E)  lambda_i * c_i(x) -
                       sum_i(j in I)  lambda_j * c_j(x)

    where lambda_i and lambda_j are Lagrange multipliers for equality and
    inequality constraints respectively; so one needs to search for saddle
    points satisfying the Karush-Kuhn-Tucker (KKT) suficient conditions for a
    point x* for being optimal:

        1. Stationarity: grad_x(L(x*, lambda*)) = 0
        2. Primal feasibility: c_i(x*) = 0 for i in E, c_j(x*) >= 0 for j in I
        3. Dual feasibility: lambda_j* >= 0 for j in I (inequality
        multipliers)
        4. Complementarity: lambda_j* * c_j(x*) = 0 for j in I

    However, this formulation alone is numerically unstable when
    constraints are not satisfied exactly, as the Lagrangian is
    unbounded when constraints are violated.

    For that reason the Augmented Lagrangian is used instead as it stabilizes
    the problem by adding quadratic penalty terms to the standard Lagrangian:

        L_A(x, lambda, mu) = f(x) +
            sum_{i in E}  (- lambda_i * c_i(x) + (mu/2) * c_i(x)^2) +
            sum_{j in I}  Phi_j(x, lambda_j, mu)

    where:
    - lambda = (lambda_1, ..., lambda_m) are Lagrange multipliers.
    - mu > 0 is a penalty parameter that controls the strength of
    the constraint enforcement.
    - the piecewise term Phi_j handles inequality complementarity by introducing
    a slack s_j >= 0 such that c_j(x) - s_j = 0 and minimising L_A over s_j
    analytically yields [1]:

        Phi_j = - lambda_j * c_j(x) + (mu/2) * c_j(x)^2
            if lambda_j - mu * c_j(x) > 0  (constraint active or violated)

        Phi_j = - lambda_j^2 / (2 * mu)
            if lambda_j - mu * c_j(x) <= 0  (constraint inactive, c_j large)

    In the inactive case, Phi_j = -lambda_j^2 / (2 * mu) is a constant with
    respect to x, so it does not influence the minimiser but it does appear in
    the value of L_A.


    The corresponding gradient contributions are:

    For equality constraints:

        grad_x(L_A) = -(lambda_i - mu * c_i(x)) * grad_x(c_i(x))

    For active inequality constraints:

        grad_x(L_A) = -(lambda_j - mu * c_j(x)) * grad_x(c_j(x))

    For inactive inequality constraints:

        grad_x(L_A) = 0


    Box bounds lb <= x <= ub are enforced via a pure quadratic penalty
    (no Lagrange multiplier), scaled by bounds_penalty_scale:

        P(x) = (bounds_penalty_scale * mu / 2) *
                (||max(lb - x, 0)||^2 + ||max(x - ub, 0)||^2)

        grad(P(x)) = bounds_penalty_scale * mu *(max(x-ub, 0) - max(lb-x, 0))

    P(x) is zero inside the feasible box and grows quadratically outside.

    ---

    The optimization strategy used in this code alternates between two
    loops:

    - Inner loop that minimises the augmented Lagrangian L_A(x; lambda^k, mu_k)
    via a unconstrained optimizer `inner_solver`, starting from x^k.

    - A Outer loop that updates multipliers and penalty parameters by comparing
    the resulting infeasibility ||c(x^{k+1})|| to an adaptive feasibility
    tolerance eta_k (initialised to 1/mu_0^0.1) [1, Algorithm 17.4]:

    If infeasibility <= eta_k (sufficient progress):

    - Update multipliers:

        lambda_i  <- lambda_i - mu * c_i(x)           (equality)
        lambda_j  <- max(0, lambda_j - mu * c_j(x))   (inequality)

    - Tighten tolerances:

        eta_{k+1} = eta_k / mu^0.9

    Else (insufficient progress):

    - Hold multipliers fixed.
    - Increase penalty: mu <- mu_factor * mu
    - Reset tolerances: eta_{k+1} = 1 / mu^0.1

    Convergence is declared when both:

        ||grad_x L_A(x, lambda, mu)|| <= tol_grad
        max constraint violation <= tol_constraints


    Constraint violation is determined by:

        max_i |c_i(x)| <= tol_constraints (for equality constraints)
        max_j max(-c_j(x), 0) <= tol_constraints (for inequality constraints)
        max(||max(lb - x, 0)||_inf, ||max(x - ub, 0)||_inf) <=
            tol_constraints (for bounds)

    This ensures all constraints are satisfied within tolerance and
    approximate the KKT conditions for the original constrained problem,
    sufficient for determining that x is a locally optimal point.

    References
    ----------
<<<<<<< HEAD
    [1] Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
        Sections 17.3 & 17.4.
=======
    [1] Chapter 17.3 & 17.4 from Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial Engineering.
        Springer, New York.
>>>>>>> 0b31c1bcc1ea81f3b19c9862cfe05a99490d7882
        https://doi.org/10.1007/978-0-387-40065-5
    """

    _constraints = constraints if constraints is not None else ()

    # Validate constraints
    const_funs = []
    const_grads = []
    const_types = []

    for current_constraint in _constraints:
        if not isinstance(current_constraint, dict):
            raise TypeError(
                f"Each constraint must be a dict, got "
                f"{type(current_constraint).__name__}."
            )

        if 'type' not in current_constraint or 'fun' not in current_constraint:
            raise ValueError(
                "Each constraint must be a dict with keys 'type' and 'fun'."
            )

        if not callable(current_constraint['fun']):
            raise TypeError("Constraint 'fun' must be callable.")

        if (
            'grad' in current_constraint
            and current_constraint['grad'] is not None
            and not callable(current_constraint['grad'])
        ):
            raise TypeError("Constraint 'grad' must be callable or None.")

        constraint_type = current_constraint['type']
        if constraint_type not in ('eq', 'ineq'):
            raise ValueError("Constraint 'type' must be 'eq' or 'ineq'.")


        constraint_type = current_constraint['type']

        if constraint_type not in ('eq', 'ineq'):
            raise ValueError("Constraint 'type' must be 'eq' or 'ineq'.")

        const_funs.append(current_constraint['fun'])
        const_grads.append(current_constraint.get('grad', None))
        const_types.append(constraint_type)

    const_count = len(const_funs)



    try:
        lb, ub = bounds
    except (TypeError, ValueError):
        raise ValueError("Bounds must be a tuple (lb, ub) or None.")



    inner_solver_func_calls = 0
    inner_solver_grad_calls = 0



    UNCONSTRAINED_START_GRAD_TOL = 1e-1

    # Find local unconstrained minimum as starting point
    if make_initial_param_unconstrained:
        if verbose_outer:
            print('Attempting to find unconstrained starting point...')

        warm_res = inner_solver(function, initial_params.copy(), grad_func)
        inner_solver_func_calls += warm_res.func_evals_number
        inner_solver_grad_calls += warm_res.grad_evals_number

        unconstrained_min = warm_res.final_params

        if unconstrained_min is not None:
            cost_unconstrained = function(unconstrained_min)
            cost_initial = function(initial_params)

            grad_norm_unc = float(np.linalg.norm(grad_func(unconstrained_min)))
            inner_solver_func_calls += 2
            inner_solver_grad_calls += 1

            suitable = (
                np.isfinite(cost_unconstrained)
                and grad_norm_unc < UNCONSTRAINED_START_GRAD_TOL
                and cost_unconstrained < cost_initial
            )

            if suitable:
                _initial_params = unconstrained_min
                if verbose_outer:
                    print(f'Using unconstrained minimum: {_initial_params}')

            else:
                _initial_params = initial_params.copy()
                if verbose_outer:
                    print('Unconstrained minimum not suitable, using provided '
                          'initial point.')

        else:
            _initial_params = initial_params.copy()
            if verbose_outer:
                print('Unconstrained minimum not suitable, using provided '
                      'initial point.')

    else:
        _initial_params = initial_params.copy()

    dim_num = _initial_params.size

    if dim_num != bounds[0].size:
        raise ValueError(f"Provided initial params (length = {dim_num}) "
                         f"mismatch their dimensions with provided bounds "
                         f"(shape = ({bounds[0].size}, {bounds[1].size})).")



    rng = np.random.default_rng(seed)



    _numerical_const_grads = [
        None if grad_fn is not None
        else nth_numerical_derivative(
            function_to_differentiate=fun,
            derivative_order=1,
            point_number=const_gradient_point_number,
        )
        for fun, grad_fn in zip(const_funs, const_grads, strict=True)
    ]



    _cache_x_vals = None
    _cache_x_grads = None
    _cache_vals = None
    _cache_grads = None

    def _eval_constraint_values(x: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluate all user-provided constraints and at point `x`.

        Parameters
        ----------
        x : np.ndarray
            The point at which to evaluate constraints.

        Returns
        -------
        Optional[np.ndarray]
            - A 1-D array of shape (n_constraints,) containing the constraint
            values `c(x)`.

            Returns None if any evaluation results in NaN or Inf.
        """

        nonlocal _cache_x_vals, _cache_vals

        if _cache_x_vals is not None and np.array_equal(x, _cache_x_vals):
            return _cache_vals


        if const_count == 0:
            _cache_x_vals = x.copy()
            _cache_vals = np.array([])
            return _cache_vals


        values = np.empty(const_count)

        for constraint_index, constraint_function in enumerate(const_funs):
            constraint_value = float(constraint_function(x))

            if not np.isfinite(constraint_value):
                return None

            values[constraint_index] = constraint_value


        _cache_x_vals = x.copy()
        _cache_vals = values

        return values


    def _eval_constraint_grads(x: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluate all user-provided constraint's gradients at point `x`.

        If a gradient function is not provided for a constraint, it is
        approximated via finite differences.

        Parameters
        ----------
        x : np.ndarray
            The point at which to evaluate constraints.

        Returns
        -------
        Optional[np.ndarray]
            - A 2-D array of shape (n_constraints, n_dims) containing the
            gradients `grad(c(x))`.

            Returns None if any evaluation results in NaN or Inf.
        """

        nonlocal _cache_x_grads, _cache_grads

        if _cache_x_grads is not None and np.array_equal(x, _cache_x_grads):
            return _cache_grads


        if const_count == 0:
            _cache_x_grads = x.copy()
            _cache_grads = np.zeros((0, dim_num))
            return _cache_grads


        grads = np.empty((const_count, dim_num))

        for i in range(const_count):
            analytic = const_grads[i]

            if analytic is not None:
                # Gradient provided by user
                grads[i, :] = np.atleast_1d(analytic(x)).astype(float).ravel()

            else:
                # Finite differences via the pre-configured callable.
                grads[i, :] = np.asarray(
                    _numerical_const_grads[i](x).derivative, dtype=float
                )

            if not np.all(np.isfinite(grads[i, :])):
                return None


        _cache_x_grads = x.copy()
        _cache_grads = grads

        return grads


    # Augmented objective and gradient builders
    def _make_augmented_function_and_grad(
        lambda_vec: np.ndarray, mu: float
    ) -> Tuple[Callable, Callable]:
        """
        Construct L_A(x) and grad_x(L_A(x)) for fixed lambda and mu.

        The formulation includes:
        1. The original objective f(x).
        2. The Lagrangian terms for constraints: `- lambda * c(x)`.
        3. The quadratic penalty terms: `+ (mu/2) * c(x)^2`.
        4. Quadratic penalties for box bound violations.

        Parameters
        ----------
        lambda_vec : np.ndarray
            Current estimates of the Lagrange multipliers.
        mu : float
            Current penalty parameter.

        Returns
        -------
        Tuple[Callable, Callable]
            - lagrangian_func(x): Evaluates the scalar Augmented Lagrangian.
            - lagrangian_grad(x): Evaluates the gradient of the Augmented
            Lagrangian.
        """

        def lagrangian_func(x: np.ndarray) -> float:
            """
            Compute the value of the Augmented Lagrangian at x.

            Returns `np.inf` if the objective or constraints are undefined.
            """

            func_val = function(x)
            if not np.isfinite(func_val): return np.inf

            Lagrangian = func_val

            if const_count > 0:
                const_vals = _eval_constraint_values(x)
                if const_vals is None: return np.inf

            for i, ctype in enumerate(const_types):

                const_i = const_vals[i]
                lambda_i = lambda_vec[i]

                if ctype == 'eq':
                    Lagrangian -= lambda_i * const_i
                    Lagrangian += 0.5 * mu * (const_i * const_i)

                else: # inequalities, c(x) < 0 means constraint violated
                    if lambda_i - mu * const_i > 0:
                        Lagrangian -= lambda_i * const_i
                        Lagrangian += 0.5 * mu * (const_i * const_i)
                    else:
                        Lagrangian -= (lambda_i * lambda_i) / (2.0 * mu)

            # Bounds penalty
            lb_viol = np.maximum(lb - x, 0.0)
            ub_viol = np.maximum(x - ub, 0.0)
            Lagrangian += 0.5 * (bounds_penalty_scale * mu) * (
                np.dot(lb_viol, lb_viol) + np.dot(ub_viol, ub_viol)
            )

            return float(Lagrangian) if np.isfinite(Lagrangian) else np.inf


        def lagrangian_grad(x: np.ndarray) -> np.ndarray:
            """
            Compute the gradient of the Augmented Lagrangian at x.

            Includes gradients from the objective, active constraints, and
            bound penalties.
            """

            grad = np.atleast_1d(grad_func(x)).astype(float).ravel()

            if const_count > 0:
                const_vals = _eval_constraint_values(x)
                grads = _eval_constraint_grads(x)

                if const_vals is None:
                    return np.full_like(grad, np.nan)


                for i, const_type in enumerate(const_types):
                    const_i = const_vals[i]
                    lambda_i = lambda_vec[i]
                    grad_i = grads[i, :]
                    shared_term = lambda_i - mu * const_i

                    if const_type == 'eq':
                        # Gradient is 0 when inactive
                        grad -= shared_term  * grad_i

                    else: # inequalities
                        if shared_term  > 0.:
                            grad -= shared_term * grad_i

            # Bounds penalty gradient
            grad += bounds_penalty_scale * mu * (
                np.maximum(x - ub, 0.0) - np.maximum(lb - x, 0.0)
            )

            return grad


        return lagrangian_func, lagrangian_grad


    def _compute_infeasibility(
        current_params: np.ndarray,
        lambda_vec: np.ndarray,
        mu: float
    ) -> float:
        """
        Compute the maximum active constraint violation at `current_params`.

        - Equality constraints always contribute |c_i(x)|.
        - Inequality constraints contribute max(-c_j(x), 0) only when the
        constraint is active (lambda_j - mu*c_j(x) > 0). When inactive,
        c_j(x) >= lambda_j/mu >= 0, so there is no violation.
        - Bound violations contribute max(lb - x, 0) and max(x - ub, 0).

        Parameters
        ----------
        current_params : np.ndarray
            The point at which to evaluate infeasibility.
        lambda_vec : np.ndarray
            Current Lagrange multiplier estimates.
        mu : float
            Current penalty parameter.

        Returns
        -------
        float
            The infinity norm (maximum value) of all active constraint
            violations.
            Returns 0.0 if the point satisfies all active constraints.
        """

        const_vals = _eval_constraint_values(current_params)
        if const_vals is None: return np.inf

        violations = []

        for i, ctype in enumerate(const_types):
            const_i = const_vals[i]
            lambda_i = lambda_vec[i]

            if ctype == 'eq':
                violations.append(np.abs(const_i))

            else: # inequalities
                if lambda_i - mu * const_i > 0:
                    violations.append(np.maximum(-const_i, 0.0))

        # Bounds
        lb_viol = np.maximum(lb - current_params, 0.0)
        ub_viol = np.maximum(current_params - ub, 0.0)
        violations.extend(lb_viol[lb_viol > 0.])
        violations.extend(ub_viol[ub_viol > 0.])

        return float(np.max(violations)) if violations else 0.0



    overall_best_x = None
    overall_best_cost = np.inf
    overall_best_feas = np.inf
    overall_best_is_feasible = False
    overall_success = False

    overall_best_termination_reason = ""

    total_outer_iterations = 0

    # Exponents of mu used by the adaptive feasibility tolerance schedule [1]:
    # eta_0 = 1/mu^ETA_INIT_EXPONENT,
    # eta_{k+1} = eta_k / mu^ETA_TIGHTEN_EXPONENT after sufficient progress.
    ETA_INIT_EXPONENT = 0.1
    ETA_TIGHTEN_EXPONENT = 0.9

    for run in range(1, max(1, maximum_restarts + 1) + 1):

        outer_best_is_feasible = False
        outer_success = False
        outer_best_x = None
        outer_best_cost = np.inf
        outer_best_feas = np.inf


        if run == 1:
            current_params = _initial_params.copy()

        else:
            # small random perturbation to escape basins
            perturb = (
                (1.0 + np.abs(_initial_params)) * restart_noise_amplitude
                * rng.standard_normal(dim_num)
            )
            current_params = _initial_params + perturb
            # Project back into bounds so a noisy restart cannot start outside
            # the feasible domain.
            current_params = np.clip(current_params, lb, ub)


        lambda_vec = np.zeros(const_count, dtype=float)
        mu = float(mu_init)

        eta_k = 1.0 / (mu ** ETA_INIT_EXPONENT)

        # Outer loop (augmenting multipliers & penalties), it is called here
        # "Outer" cause the inner iters are considered the ones inside the
        # inner solver.
        # At least one outer_iter must be done
        for outer_iter in range(1, max(1, max_outer_iter + 1)):

            # Build augmented objective for current lambda and mu
            aug_fun, aug_grad = (
                _make_augmented_function_and_grad(lambda_vec, mu)
            )

            # Inner solver
            inner_res = inner_solver(aug_fun, current_params, aug_grad)
            if (
                inner_res.final_params is None
                or not np.all(np.isfinite(inner_res.final_params))
            ):
                termination_reason = (
                    f"Inner solver produced an invalid result at run {run}, "
                    f"outer iteration {outer_iter}: "
                    f"{inner_res.termination_reason}"
                )

                break

            new_params = inner_res.final_params
            inner_solver_func_calls += inner_res.func_evals_number
            inner_solver_grad_calls += inner_res.grad_evals_number

            current_cost = function(new_params)
            current_infeas = _compute_infeasibility(new_params, lambda_vec, mu)
            grad_norm = float(np.linalg.norm(aug_grad(new_params)))

            if verbose_outer and (
                outer_iter == 1 or outer_iter % max(1,verbose_freq_outer) == 0
                or outer_iter == max_outer_iter
            ):
                print(f"[Run {run} Iter {outer_iter}] x={new_params} "
                      f"Cost={current_cost:.6g} Infeas={current_infeas:.3e} "
                      f"Mu={mu:.3e}, Grad_norm={grad_norm:.3e}")

            if callback:
                callback(
                    run, outer_iter, new_params.copy(), current_cost,
                    current_infeas, mu, lambda_vec.copy(), grad_norm
                )

            if not np.isfinite(current_cost):
                termination_reason = ("Objective function returned an invalid "
                                      "value (NaN or +-Inf).")
                break
            if not np.isfinite(grad_norm):
                termination_reason = ("Invalid value (NaN or +-Inf) "
                                      "encountered while evaluating the "
                                      "gradient.")
                break


            outer_is_feasible = current_infeas <= tol_constraints
            outer_best_is_feasible = outer_best_feas <= tol_constraints

            outer_update_best = (
                # Current is feasible and previous best was not
                (outer_is_feasible and not outer_best_is_feasible)
                # Both feasible, current is cheaper
                or (outer_is_feasible and outer_best_is_feasible
                    and current_cost < outer_best_cost)
                # Both infeasible, current is less infeasible
                or (not outer_is_feasible and not outer_best_is_feasible
                    and current_infeas < outer_best_feas)
            )

            if outer_update_best:
                outer_best_x = new_params.copy()
                outer_best_cost = current_cost
                outer_best_feas = current_infeas


            # check for convergence
            if outer_is_feasible and grad_norm <= tol_grad:
                outer_success = True
                termination_reason = (
                    f"Converged with an infeasibility of "
                    f"({current_infeas:.2e}) <= tolerance for constraints "
                    f"({tol_constraints:.2e}), and a gradient norm "
                    f"({grad_norm:.2e}) <= tolerance for the gradient "
                    f"({tol_grad:.2e})"
                )
                current_params = new_params.copy()
                break


            if current_infeas <= eta_k:
                if const_count > 0:
                    const_vals = _eval_constraint_values(new_params)
                    if const_vals is None:
                        termination_reason = ("Invalid value encountered "
                                              "while evaluating constraints")
                        current_params = new_params.copy()
                        break

                    for i, ctype in enumerate(const_types):
                        if ctype == 'eq':
                            lambda_vec[i] -= mu * const_vals[i]

                        else:
                            lambda_vec[i] = max(
                                0.0, lambda_vec[i] - mu * const_vals[i]
                            )

                eta_k /= mu ** ETA_TIGHTEN_EXPONENT

            else:
                mu *= mu_factor
                eta_k = 1.0 / (mu ** ETA_INIT_EXPONENT)


            current_params = new_params.copy()

        # End of Outer Loop
        else:
            termination_reason = (
                f"Max outer iterations ({max_outer_iter}) reached on run "
                f"{run}."
            )

        total_outer_iterations += outer_iter


        # Update best every "run loop" iter
        update_best = (
            (outer_best_is_feasible and not overall_best_is_feasible)
            or (outer_best_is_feasible and overall_best_is_feasible
                and outer_best_cost < overall_best_cost)
            or (not outer_best_is_feasible and not overall_best_is_feasible
                and outer_best_feas < overall_best_feas)
        )

        if update_best:
            overall_success = outer_success
            overall_best_x = (outer_best_x.copy() if outer_best_x is not None
                else None)
            overall_best_cost = outer_best_cost
            overall_best_feas = outer_best_feas

            overall_best_is_feasible = overall_best_feas <= tol_constraints
            overall_best_termination_reason = termination_reason

        # Early exit if a good feasible solution is found
        if outer_best_is_feasible and outer_best_cost < satisfaction_threshold:
            overall_best_x = (
                outer_best_x.copy() if outer_best_x is not None else None
            )
            overall_best_cost = outer_best_cost
            overall_best_feas = outer_best_feas
            overall_best_is_feasible = True
            overall_success = True
            overall_best_termination_reason = (
                f"Final cost ({outer_best_cost:.6g}) is less than the "
                f"satisfaction threshold ({satisfaction_threshold:.6g}) and "
                f"the point is feasible (infeasibility = "
                f"{outer_best_feas:.2e})."
            )

            break

    # End of run loop

    overall_best_termination_reason = termination_reason

    return {
        'success': overall_success,
        'final_params': overall_best_x,
        'final_cost': overall_best_cost,
        'termination_reason': overall_best_termination_reason,
        'iteration_number': total_outer_iterations,
        'func_evals_number': inner_solver_func_calls,
        'grad_evals_number': inner_solver_grad_calls
    }
