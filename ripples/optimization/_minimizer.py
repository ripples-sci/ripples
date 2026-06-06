"""
Public entry point for the optimization module.

***minimizer*** is the single user-facing function of `ripples.optimization`.
The numerical strategies themselves are in the per-algorithm modules
(_steepest_descent_optimization.py, _line_search_optimization.py,
_trust_region_optimization.py, _global_optimization.py,
_constrained_optimization.py).

This file is the dispatch that normalises and validates user input, builds the
gradient / Hessian / Hessian-vector-product wrappers, picks one of three
dispatch branches, and outsputs via ***OptimizationResult***.

The three dispatch branches are:

- Unconstrained local minimization - a direct call to the chosen local
optimizer (`nesterov`, `adam`, `conjugate_gradient`,
`newton_conjugate_gradient`, `bfgs`, `lbfgs`, `trust_ncg`, `trust_lanczos`).

- Unconstrained global minimization - a direct call to `direct` or
`annealing`. The `annealing` path additionally has a recursive ***minimizer***
call for a local-refinement step.

- Constrained or bounds-only local minimization - the augmented-Lagrangian
outer loop in _constrained_optimization.py, driven by a recursive
***minimizer*** call for each inner sub-problem.

The output of every branch is collected together with the complete
configuration that produced it (method, resolved tolerances, resolved
method parameters, bounds, constraint count, ...) and returned as an
***OptimizationResult*** (see optimization/_utils.py), which exposes
both the optimized parameters and the full diagnostic record of the
run.

Contains
--------
_merge_user_dict_with_defaults
    Build a complete settings dict from a defaults table overridden
    with a user-provided dict.

_print_run_banner
    Render the opening header of a verbose ***minimizer*** run.

minimizer
    A unified interface to the optimization algorithms, with optional support
    for analytical gradients and Hessians, box bounds, and general
    equality / inequality constraints. Returns the run as an
    ***OptimizationResult*** object.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
import time
from warnings import warn
from typing import Optional, Union, Tuple, Literal, Dict, Callable, Any

from ._steepest_descent_optimization import _nesterov_optimizer, _adam_optimizer
from ._line_search_optimization import (
    _conjugate_gradient_optimizer, _newton_cg_optimizer, _bfgs_optimizer,
    _lbfgs_optimizer
)
from ._trust_region_optimization import _trust_region_optimizer
from ._global_optimization import _direct_optimizer, _annealing_optimizer
from ._constrained_optimization import _augmented_lagrangian_optimizer
from ._utils import (
    OptimizationResult, AVAILABLE_METHODS, LOCAL_OPTIMIZATION_METHODS,
    GLOBAL_OPTIMIZATION_METHODS, METHODS_THAT_REQUIRE_GRADIENT,
    METHODS_THAT_REQUIRE_HESSIAN, METHODS_WITH_INNER_MINIMIZER,
    METHODS_THAT_REQUIRE_BOUNDS
)

from ..differentiation import nth_numerical_derivative



def _merge_user_dict_with_defaults(
    user_dict: Dict[str, Any],
    defaults:  Dict[str, Any],
    dict_label: str,
) -> Dict[str, Any]:
    """
    Build a complete settings dict from `defaults` overridden with `user_dict`.

    Unknown keys in `user_dict` are rejected with `ValueError`.

    Parameters
    ----------
    user_dict : dict
        User-provided overrides. Empty dict is accepted.
    defaults : dict
        Canonical key set with default values.
    dict_label : str
        Currently evaluated dict name, for example: 'method_params',
        'constrained_params' or 'tolerances'.

    Returns
    -------
    dict
        New dict with every key in `defaults`, the value taken from
        `user_dict` when present and from `defaults` otherwise.

    Raises
    ------
    ValueError
        When `user_dict` contains a key not in `defaults`.
    """

    unknown_keys = set(user_dict) - set(defaults)

    if unknown_keys:
        raise ValueError(
            f"Unknown key(s) in {dict_label}: {sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(defaults)}."
        )

    return {
        key: user_dict[key] if key in user_dict else defaults[key]
        for key in defaults
    }



def _print_run_banner(
    *,
    title: str,
    method: str,
    initial_params: Optional[np.ndarray],
    bounds: Optional[np.ndarray],
    max_iters: int,
    verbose_freq: int,
    tolerances: Dict[str, float],
    method_params: Dict[str, Any],
    constrained_params: Optional[Dict[str, Any]],
) -> None:
    """
    Print the opening banner of a verbose minimizer run.
    """

    print("=" * 60)
    print(
        f"Starting {title} with: "
        f"{method.replace('_', ' ').title()}"
    )
    if bounds is not None:
        print(f"Bounds: {bounds}")
    if initial_params is not None:
        print(f"Initial parameters: {initial_params}")
    print(f"Max Iters: {max_iters:,d}, Verbose_freq: {verbose_freq:,d}")
    print(f"Tolerances: {tolerances}")
    print(f"Method parameters: {method_params}")
    if constrained_params is not None:
        print(f"Constrained params: {constrained_params}")
    print("=" * 60)



def minimizer(
    function: Callable[[np.ndarray], float],
    method: Literal[
        'nesterov', 'adam', 'conjugate_gradient', 'newton_conjugate_gradient',
        'bfgs', 'lbfgs', 'trust_ncg', 'trust_lanczos', 'direct', 'annealing'
    ] = 'trust_ncg',
    initial_params: Optional[Union[float, np.ndarray]] = None,
    method_params: Optional[Dict[str, Any]] = None,
    gradient_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    gradient_point_number: int = 2,
    hessian_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hessian_vector_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None,
    bounds: Optional[Tuple[Any, Any]] = None,
    constraints: Optional[Tuple[Dict[str, Any]]] = None,
    constrained_params: Optional[Dict[str, Any]] = None,
    max_iters: Union[Literal['auto'], int] = 'auto',
    tolerances: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    verbose_frequency: Union[Literal['auto'], int] = 'auto',
    callback: Optional[Callable] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar `function` of any variable number.

    This function provides a comprehensive optimization framework supporting
    both local and global optimization methods, with optional support for
    constraints and bounds.

    See the `method` parameter for the algorithm complete list and a selection
    guide.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective to minimize. Must accept a 1-D numpy.ndarray of
        parameters and return a real scalar.

    method : str, default='trust_ncg'
        The optimization algorithm to use. One of:

        Steepest-descent methods:
        - `'nesterov'` : Nesterov Accelerated Gradient with momentum and
        learning rate decay. [2, Chapter 5.4]
        - `'adam'` : Adaptive Moment Estimation with adaptive per coordinate
        learning rates. [2, Chapter 5.8]

        Line-search methods (strong Wolfe enforced):
        - `'conjugate_gradient'` : Non-linear conjugate gradient using the
        Polak-Ribière+ update. [1, Algorithm 5.4 with eq. 5.45]
        - `'newton_conjugate_gradient'` : Truncated Newton using conjugate
        gradient on Hessian-vector products for the inner solve.
        [1, Algorithm 7.1]
        - `'bfgs'` : Dense Broyden-Fletcher-Goldfarb-Shanno.
        [1, Algorithm 6.18 with eq. 6.19]
        - `'lbfgs'` : Limited memory BFGS. Memory efficient variant that stores
        only a few of the curvature pairs. [1, Algorithms 7.4, 7.5]

        Trust-region methods:
        - `'trust_ncg'` : Trust-region Newton-CG using the Steihaug-Toint CG
        subproblem solver. [1, Algorithm 7.2]
        - `'trust_lanczos'` : Trust-region with the Generalized Lanczos
        (GLTR / Krylov) subproblem solver.
        [8][1, pp. 175-176 & Theorem 4.1 & Chapter 4.3]

        Global optimization methods:
        - `'direct'` : DIviding RECTangles. Deterministic derivative-free
        global solver that systematically subdivides a provided box with the
        specified finite `bounds` which are required. [2, Chapter 7.6]
        - `'annealing'` : Dual annealing combining generalized simulated
        annealing with a local-search refinement step. Stochastic
        derivative-free global solver. Requires finite bounds.
        [9][10][2, Chapter 8.3][11]

        **Method selection guide:**

        - For a general-purpose first try, or when the user is unsure of
        how the function behaves, use `'trust_ncg'` or `'lbfgs'`.

        - For smooth problems with fewer than ~500 parameters, the quasi-Newton
        `'bfgs'` and `'conjugate_gradient'` and the trust-region `'trust_ncg'`
        and `'trust_lanczos'` are appropriate.

        - For smooth problems with more than ~500 parameters, `'trust_ncg'`,
        `'lbfgs'` and `'newton_conjugate_gradient'` scale best.

        - For ill-conditioned or poorly scaled problems, prefer trust-region
        methods (in particular `'trust_lanczos'`, which is the least
        sensitive to ill-conditioning) since they suffer less from numerical
        instability than line-search methods.

        - Steepest-descent methods (`'nesterov'`, `'adam'`) perform poorly as
        general-purpose solvers, but they remain useful when the starting point
        is close to a local optimum, when the function is noisy or stochastic,
        or when it is only piecewise-smooth.

        - `'direct'` is a brute-force grid-style search. Past ~4 dimensions it
        is not practical, but for tight boxes and very few variables it can be
        effective.

        - When the objective presents many local minima, or when a bounded
        problem must be solved without a reliable gradient, prefer a global
        method such as `'annealing'`.

    initial_params : Optional[Union[float, np.ndarray]], default=None
        Starting point for the iteration. Required for every local
        method. May be:

        - a scalar (single-variable problem),
        - a 1-D array-like of any length, one entry per parameter,
        - None, accepted only for the global methods `'direct'` and
        `'annealing'` (which use bounds).

    method_params : Optional[Dict[str, Any]], default=None
        Algorithm-specific parameters. Unknown keys are rejected with
        ValueError. The accepted keys per method are listed below; every key is
        optional and defaults to the value shown.

        `'nesterov'`:
        - `'learning_rate'` : float, default=1e-4 - initial step size.
        - `'momentum'` : float, default=1e-1 - momentum coefficient.
        - `'decay_rate'` : float, default=0.0 - learning-rate decay factor; the
        rate at iteration k is `lr_0 / (1 + decay_rate * k)`.

        `'adam'`:
        - `'learning_rate'` : float, default=1e-3.
        - `'beta1'` : float, default=0.9 - first-moment decay rate.
        - `'beta2'` : float, default=0.999 - second-moment decay rate.
        - `'decay_rate'` : float, default=0.0 - learning-rate decay factor,
        same convention as `'nesterov'`.

        `'conjugate_gradient'`:
        - `'initial_alpha'` : float, default=1.0 - initial line-search step.
        - 'max_alpha' : float, default=np.inf - maximum allowed step.
        - `'bracketing_max_iter'` : int, default=300 - line-search
        bracketing-phase iteration cap.
        - `'wolfe_c1'` : float, default=1e-4 - Armijo (sufficient decrease)
        constant.
        - `'wolfe_c2'` : float, default=0.4 - curvature condition constant.
        - `'zoom_max_iter'` : int, default=500 - line-search zoom-phase
        iteration cap.
        - `'zoom_epsilon'` : float, default=0.0 - minimum bracket width
        before the zoom phase gives up.

        Every line-search method (`'conjugate_gradient'`,
        `'newton_conjugate_gradient'`, `'bfgs'`, `'lbfgs'`)
        requires `0 < wolfe_c1 < wolfe_c2 < 1`.

        `'newton_conjugate_gradient'`: every key accepted by
        `'conjugate_gradient'` (with `wolfe_c2` default 0.9), plus:
        - `'inner_max_iter'` : Union[str, int], default='auto' - CG iteration
        cap for the inner Newton solve. `'auto'` resolves to
        `int(n_params * 1.2)`.
        - `'hvp_h'` : Union[str, float], default='auto' - finite difference
        step for the Hessian-vector product when neither `hessian_function`
        nor `hessian_vector_function` is supplied.
        - `'hvp_point_number'` : int, default=2 - central-difference stencil
        width for the same HVP. Must be even.

        `'bfgs'`: every key accepted by `'conjugate_gradient'` with
        `wolfe_c2` default 0.9.

        `'lbfgs'`: every key accepted by `'bfgs'`, plus:
        - `'memory_size'` : int, default=20 - number of curvature
          pairs (s_k, y_k) kept in memory.

        `'trust_ncg'` and `'trust_lanczos'`:
        - `'inner_max_iter'` : Union[str, int], default='auto' -
        subproblem-solver iteration cap.
        - `'grad_rel_tol'` : float, default=1e-3 - relative toleranc
        on the subproblem residual (multiplied by the current gradient norm).
        - `'abs_tol'` : float, default=1e-8 - absolute tolerance on the
        subproblem residual.
        - `'hvp_h'` : Union[str, float], default='auto' - finite difference
        step for the HVP when no analytical `hessian_function` or
        `hessian_vector_function` is supplied.
        - `'hvp_point_number'` : int, default=2 - central-difference stencil
        width for the same HVP. Must be even.
        - `'initial_trust_radius'` : float, default=1.0.
        - `'max_trust_radius'` : float, default=np.inf.
        - `'eta'` : float, default=0.15 - minimum ratio of actual to
        predicted reduction required to accept a step.
        - `'rho_make_smaller_threshold'` : float, default=0.25 - shrink the
        trust radius when the reduction ratio falls below this value.
        - `'rho_make_bigger_threshold'` : float, default=0.75 - enlarge the
        trust radius when the reduction ratio exceeds this value and the step
        hit the boundary.
        - `'make_trust_radius_smaller_multiplier'` : float, default=0.25 -
        factor by which the radius shrinks.
        - `'make_trust_radius_bigger_multiplier'` : float, default=2 - factor
        by which the radius grows.

        `'direct'`:
        - `'max_fun_evals'` : Union[str, int], default='auto' - total
        function-evaluation budget.
        - `'max_iters_with_no_progress'` : int, default=100 - terminate after
        this many iterations without an improvement.
        - `'max_rectangles'` : Union[str, int], default='auto' - cap on the
        number of hyperrectangles kept in memory.
        - `'max_diameter'` : float, default=0.05 - force any rectangle with
        normalised (maximum is 1) norm-2 diameter above this value to be split.
        Lower values push the search closer to a brute-force sweep.
        - `'f_min'` : float, default=-np.inf - early termination target value.
        - `'len_tol'` : Union[str, float], default='auto' - tolerance on the
        best rectangle's side length.
        - `'vol_tol'` : Union[str, float], default='auto' - tolerance on the
        best rectangle's volume.

        `'annealing'`:
        - `'inner_minimizer_params'` : Dict, default={} - keyword arguments
        forwarded to the nested ***minimizer*** call that performs the local
        refinement step. Common keys are `'method'` (default `'trust_ncg'`),
        `'method_params'`, `'gradient_function'`, and so on.
        - `'x0'` : Optional[np.ndarray], default=None - starting point for
        the annealing chain. None samples uniformly inside the bounds.
        - `'seed'` : Optional[int], default=None - random seed for
          reproducibility.
        - `'max_fun_evals'` : Union[str, int], default='auto' - total
        function evaluation budget.
        - `'f_min'` : float, default=-np.inf - early termination target value.
        - `'temp_init'` : float, default=1e-2 - initial temperature
        of the annealing schedule.
        - `'no_local_search'` : bool, default=False - when True, the local
        refinement step is skipped (pure simulated annealing).
        - `'reanneal_interval'` : Union[str, int], default='auto' -
        iterations between scheduled local-search calls and the stagnation
        window.
        - `'visit_const'` : float, default=2.62 - Tsallis visiting index
        `q_v`. Must satisfy `1 < q_v < 3`. Larger values give
        heavier-tailed jumps.
        - `'accept_const'` : float, default=1.5 - Tsallis acceptance index
        `q_a`. Must satisfy `1 < q_a < 3`. Larger values accept more uphill
        moves.
        - `'cooling_power'` : float, default=1.0 - exponent of the power-law
        cooling schedule:

            T_k = T_init *
                  exp(-cooling_power * (k_eff / max_iters) ** (1 / n))

        where:
        - k_eff is the iteration count since the last reanneal.
        - n is the problem dimension.

    gradient_function : Optional[Callable[[np.ndarray], np.ndarray]], \
        default=None
        Analytical gradient of `function`. Must accept a 1-D numpy.ndarray and
        return a 1-D numpy.ndarray of the same shape. If None, the gradient is
        built numerically by central finite differences with
        `gradient_point_number` stencil points.

        Used by every gradient-based method: `'nesterov'`, `'adam'`,
        `'conjugate_gradient'`, `'newton_conjugate_gradient'`,
        `'bfgs'`, `'lbfgs'`, `'trust_ncg'`, `'trust_lanczos'`.
        It is ignored for a derivative-free method (`'direct'`, `'annealing'`
        without constraints / bounds-only annealing).

    gradient_point_number : int, default=2
        Stencil width of the central-difference numerical gradient used when
        `gradient_function` is None. Must be even. Higher values (4, 6, ...)
        trade extra function evaluations for higher accuracy.

    hessian_function : Optional[Callable[[np.ndarray], np.ndarray]], \
        default=None
        Analytical Hessian. Must accept a 1-D numpy.ndarray and return
        the 2-D `(n, n)` symmetric matrix. When supplied, the
        Hessian-vector product is formed internally as `H @ p`.

        Used only by the second-order methods
        `'newton_conjugate_gradient'`, `'trust_ncg'`, `'trust_lanczos'`.
        If provided elsewhere will be ignored. If both `hessian_function` and
        `hessian_vector_function` are supplied, the explicit
        `hessian_function` is ignored in favour of the vector-product form.

    hessian_vector_function : Optional[Callable[[np.ndarray, np.ndarray], \
        np.ndarray]], default=None
        Analytical Hessian-vector product. Must accept the current
        parameter vector `x` and a direction `p` and return
        `H(x) @ p` as a 1-D numpy.ndarray of size `n`.

        Preferred over `hessian_function` for large-scale problems:
        constructing the full Hessian costs O(n^2) function calls and
        O(n^2) memory, whereas a single HVP costs O(n) gradient calls
        and O(n) memory.

        Same applicability rules as `hessian_function`.

    bounds : Optional[Tuple[array_like, array_like]], default=None
        Box constraints as a 2-tuple `(lower, upper)`. Each of `lower` and
        `upper` may be:

        - a scalar (broadcast to every coordinate).
        - a 1-D array-like of length `n_params`, one entry per coordinate.
        - `None`, treated as -np.inf for `lower` and +np.inf for `upper` on
        every coordinate.

        Individual coordinates can be left unbounded by placing -np.inf in
        `lower` or +np.inf in `upper` for the coordinate in question.

        Each lower bound must be strictly less than the matching upper
        bound, equal bounds are rejected.

        - Required for `'direct'` and `'annealing'`, in which case both `lower`
        and `upper` must be finite array-likes of the same size as the problem
        dimension.
        - Optional for every other method. Supplying bounds for a local method
        engages the augmented-Lagrangian outer loop, which converges the
        objective subject to the box.

        Example: bounds=([-1, -3, -10], [2, 5, -6]) on a 3-D problem means
        first parameter in [-1, 2], second in [-3, 5], third in [-10, -6].

    constraints : Optional[Tuple[Dict[str, Any]]], default=None
        General equality / inequality constraints. Each entry is a dict
        with:

        - `'type'`: str - either `'eq'` (equality) or `'ineq'` (inequality).
        - `'fun'`: Callable[[np.ndarray], float] - the constrain function
        `c(x)`. The convention is `c(x) = 0` for equalities and `c(x) >= 0` for
        inequalities (i.e. the feasible solution set is {x : c(x) >= 0}).
        - `'grad'`: Callable[[np.ndarray], np.ndarray], optional - analytical
        gradient of the constraint. If absent it is approximated by central
        differences with a stencil width
        `constrained_params['const_gradient_point_number']`.

        Whenever `constraints` or `bounds` is supplied to a non bounds-only
        method, the augmented-Lagrangian outer loop is used [1, Algorithm 17.4].

    constrained_params : Optional[Dict[str, Any]], default=None
        Parameters for the augmented-Lagrangian outer loop. Used whenever
        `constraints` is supplied or whenever `bounds` is supplied to a method
        that does not natively handle bounds (every method except 'direct'` and
        `'annealing'`).
        Accepted keys (all optional):

        - `'max_outer_iter'` : int. Default 100 for local methods, 10 for
        global methods.
        - `'mu_init'` : float. Initial penalty parameter. Default 10.0 for
        local methods, 1e5 for global methods.
        - `'mu_factor'` : float, default=10.0 - factor by which `mu` is
        multiplied when the constraint violation has not decreased enough
        between outer iterations.
        - `'tol_constraints'` : float, default=1e-7 - maximum allowed
        constraint violation at convergence.
        - `'tol_grad'` : float, default=1e-4 - convergence tolerance on the
        gradient norm of the Lagrangian.
        - `'const_gradient_point_number'` : int, default=2 - stencil width of
        the central-difference numerical gradient for any constraint without an
        analytical `'grad'`.
        - `'verbose_inner'` : bool, default=False - whether the inner
        unconstrained solver should print its own progress.
        - `'verbose_freq_inner'` : int, default=1 - print frequency for the
        inner solver.
        - `'callback_inner'` : Optional[Callable], default=None -
        per-iteration callback for the inner solver, signature as described
        under `callback` below.
        - `'bounds_penalty_scale'` : float, default=1e1 - scale of the
        quadratic penalty applied to bound violations.
        - `'make_initial_param_unconstrained'` : bool, default=False - when
        True, the outer loop first solves the unconstrained problem to obtain a
        (possibly better) starting point.
        - `'satisfaction_threshold'` : float, default=-np.inf - early
        termination when a feasible point with cost below this value is found.
        - `'restart_noise_amplitude'` : float, default=1e-1 - relative noise
        amplitude added to `initial_params` on each restart.
        - `'maximum_restarts'` : int, default=0 - number of perturbed
        restarts attempted if the first run does not satisfy the convergence
        criteria.
        - `'seed'` : Optional[int], default=None - random seed for the
        restart perturbations.

    max_iters : Union[Literal['auto'], int], default='auto'
        Maximum number of outer iterations the solver is allowed to
        perform.

        - `'auto'` resolves to `200_000 * n_params` for local methods,
        `1_000 * n_dim` for `'direct'`, and `200_000 * n_dim` for
        `'annealing'`.
        - any positive integer sets the cap directly.


    tolerances : Optional[Dict[str, float]], default=None
        Convergence tolerances. Accepted keys (all optional, both defaults
        chosen close to machine epsilon):

        - `'gradient'` : float, default=1e-8 - convergence is declared when the
        gradient norm falls below this value.
        - `'relative'` : float, default=1e-16 - convergence is declared when
        `|f_new - f_old| / max(|f_new|, |f_old|, 1) <` this value.

        The solver stops as soon as one of them is satisfied.

    verbose : bool, default=False
        When True, prints optimization progress to stdout, from a starting
        header indicating all the parameters provided to the optimization
        algorithm to a per-iteration progress controlled by `verbose_frequency`,
        until eventually reaching the last iteration and the result is showed
        alongside the final cost, the iteration number and the elapsed time.

    verbose_frequency : Union[Literal['auto'], int], default='auto'
        Frequency at which the progress is being printed to stdout every
        `verbose_frequency` iterations.

        - `'auto'` resolves to 1 for constrained / bounded local runs (where
        each outer iteration is informative), and to max(1, max_iters // 10000)
        otherwise.
        - any positive integer sets the frequency directly.

    callback : Optional[Callable], default=None
        User callback called at the end of every iteration. The signature
        depends on the chosen method, because each family has access to
        different per-iteration quantities:

        - first-order methods (`'nesterov'`, `'adam'`)
        `callback(iter, params, cost, gradient, gradient_norm, learning_rate)`

        - line-search methods (`'conjugate_gradient'`,
        `'newton_conjugate_gradient'`, `'bfgs'`, `'lbfgs'`):
        `callback(iter, params, cost, gradient, gradient_norm, alpha,
        direction)`

        - trust-region methods (`'trust_ncg'`, `'trust_lanczos'`):
        `callback(iter, params, cost, gradient, gradient_norm, direction,
        trust_radius)`

        - `'direct'`:
        `callback(iter, best_params, best_cost)`

        - `'annealing'`:
        `callback(iter, current_cost, current_params, best_cost, best_params)`
        where `candidate_cost` and `candidate_x` describe the trial
    point proposed this iteration (whether or not it was accepted),
    and `best_cost` / `best_x` describe the global best so far.

        - augmented-Lagrangian outer loop (entered whenever `constraints` or
        `bounds` are supplied to a method that does not natively handle bounds):
        `callback(run_num, iter_num, params, cost, infeasibility, mu,
        multipliers, grad_norm)`.

        Where `run_num` is indexes restarts,  `infeasibility` is the maximum
        active constraint violation, `mu` is the current quadratic-penalty
        coefficient, and `multipliers` is the vector of Lagrange multiplier
        estimates.

    Returns
    -------
    OptimizationResult
        Output object carrying the optimized parameters, the final
        cost, and the complete record of the run.

        The returned ***OptimizationResult*** has the following
        attributes and methods:

        - Core output: `success`, `final_params`, `final_cost`,
        `termination_reason`, `func_evals_number`, `grad_evals_number`,
        `elapsed_time`, `iteration_number`. `final_params` and
        `final_cost` are None when the run failed before producing them
        (typically when the starting point is rejected by the initial-state
        check).

        - Configuration that produced the result: `method`,
        `initial_params`, `bounds`, `max_iters`, `tolerances`,
        `method_params`, `gradient_point_number`,
        `analytical_gradient_provided`,
        `analytical_hessian_provided`,
        `hessian_vector_product_provided`, `constraint_count`,
        `constrained_params`. `method_params` and `constrained_params`
        are the resolved dicts after being merged with their respective
        defaults, with callable entries replaced by `'<callable>'`.

        - Convenience properties: `is_constrained`, `is_bounded`,
        `is_global_method`, `is_local_method`, `total_evaluations`,
        `average_iteration_time`, `func_evals_per_iteration`,
        `grad_evals_per_iteration`, `evaluations_per_second`, plus the
        numpy-style `shape`, `ndim`, `size`, `dtype`,
        `variable_number`, `is_scalar`, `is_array` of `final_params`.
        The per-iteration and per-second properties return None when the
        denominator is zero.

        - Conversion methods: `as_array()` (writable copy of
        `final_params`), `as_float()` (single-variable problems only),
        `to_list()`, `to_dict()`.

        - Comparison methods: `==` (strict, field-by-field, NaNs compared as
        equal, with `elapsed_time` excluded so two byte-identical runs on
        different hardware still compare equal) and
        `allclose(other, rtol=..., atol=..., equal_nan=...)`
        (numerical equivalence on `final_params`, and `final_cost`).

        - `summary(style='compact' | 'full')` : a multi-line summary.
        `'compact'` shows the method, success flag, final cost, full
        `final_params`, termination reason, and the basic diagnostic
        counters. `'full'` adds every configuration field and the derived
        throughput statistics. `str(result)` returns the compact summary.

        The object also forwards `np.asarray(result)`, `result[idx]`,
        `len(result)`, iteration, and `value in result` to the underlying
        `final_params`, so it can be used as a drop-in replacement for a
        standard numpy.ndarray in arithmetic expressions.

    Raises
    ------
    ValueError
        - When `method` is not in the catalogue listed above.
        - When `initial_params` is `None` and `method` is not a global method.
        - When `bounds` is supplied but cannot be unpacked into
        `(lower, upper)`.
        - When `method` is `'direct'` or `'annealing'` and the bounds contain
        non-finite values.
        - When `method` is `'direct'` or `'annealing'` and the lower or upper
        bound is `None`.
        - When `lower` and `upper` differ in shape, or when any lower bound is
        greater than or equal to its upper bound.
        - When `max_iters` is a string other than `'auto'` or an integer less
        than 1.
        - When any tolerance in `tolerances` is negative.
        - When `method_params`, `constrained_params`, or `tolerances` contains
        a key not accepted by the chosen method.
        - When `method` is `'annealing'` and
        `method_params['inner_minimizer_params']['method']` is itself a global
        method.

    Warns
    -----
    RuntimeWarning
        - When `initial_params` is supplied for a global method (`'direct'`,
        `'annealing'`): the value is ignored.
        - When `gradient_function` is supplied for a method that does not use a
        gradient (unconstrained `'direct'`): the value is ignored.
        - When `hessian_function` or `hessian_vector_function` is supplied for
        a method that does not use second-order information: the value is
        ignored.
        - When both `hessian_function` and `hessian_vector_function` are
        supplied: `hessian_function` is ignored in favour of the
        vector-product form.

    Notes
    -----
    **Performance tips:**

    - Providing an analytical `gradient_function` typically yields a 10x to
    100x speedup over the numerical default and substantially improves
    robustness, especially for second-order and trust-region methods. Adding an
    analytical `hessian_function` or `hessian_vector_function` on top can give
    a further 2x to 5x. Providing them is strongly recommended where possible.

    - A well-chosen and educated guess of `initial_params` (from physics, a
    heuristic, or a better model model) dramatically improves convergence speed
    and reliability, particularly for non-convex problems.

    - Persistent flat gradients or vanishing step sizes are nearly always a
    symptom of bad scaling, overly tight tolerances, or numerical inaccuracies
    in the objective `function`. It is recommended to rescale the parameters so
    typical variations are O(1).

    - Global optimizers are orders of magnitude slower than local ones. Since
    there is no general way to certify a candidate minimum as global,
    `'direct'` and `'annealing'` terminate on iteration / evaluation number or
    on the `f_min` objective `function` target. Use them only when the objective
    `function` present many local minima or when no reasonable initial guess is
    available.

    - When tolerances are set extremely tight, some methods will fail to attain
    a numerically sensible solution. The defaults (`gradient=1e-8`,
    `relative=1e-16`) are chosen close to double precision machine epsilon
    and are appropriate for well-scaled problems, the gradient represent a
    little less than the square root of the machine epsilon for a 64 bit float
    (sqrt(eps) ≈ 1.5e-8) and the relative represent a little less than said
    epsilon (~2.2e-16).

    Examples
    --------
    >>> import numpy as np
    >>> from ripples.optimization import minimizer

    **Example 1: minimize a 2-D quadratic with the default method.** When the
    objective is smooth and the dimension is moderate, calling `minimizer` with
    only `function` and `initial_params` already gives a high-quality answer.

    >>> def quadratic(x):
    ...     return (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
    >>>
    >>> result = minimizer(
    ...     function=quadratic,
    ...     initial_params=[0.0, 0.0],
    ... )
    >>> result.success
    True
    >>> result.final_params
    [ 3. -1.]
    >>> result.final_cost
    9.192785343449083e-19 # 0.0
    >>> print(result)
    OptimizationResult (compact)
    ============================
    method                 : trust_ncg
    success                : True
    final_cost             : 9.19279e-19
    final_params           : array([ 3., -1.])
    termination_reason     : Gradient norm (1.92e-09) is below tolerance (1.00e-08).
    iteration_number       : 3
    func_evals_number      : 68
    grad_evals_number      : 16
    elapsed_time           : 1.666 ms

    **Example 2 - Rosenbrock with the default method.** The Rosenbrock "banana"
    valley is the canonical benchmark for unconstrained optimization: a narrow
    curved valley that line-search methods can struggle with but trust-region
    methods navigate smoothly.

    >>> def rosenbrock(x):
    ...     return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    >>>
    >>> result = minimizer(
    ...     function=rosenbrock,
    ...     initial_params=[-1.2, 1.0],
    ...     method='trust_ncg',
    ... )
    >>> result.final_params
    [0.99999998 0.99999997]
    >>> result.iteration_number
    24
    >>> result.func_evals_number
    689

    **Example 3 - Speed up and make more accurate the same run with an
    analytical gradient.** Numerical gradients are convenient but expensive, an
    analytical gradient typically cuts time and function evaluations by an
    order of magnitude.

    >>> def rosenbrock_grad(x):
    ...     dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    ...     dx1 = 200 * (x[1] - x[0] ** 2)
    ...     return np.array([dx0, dx1])
    >>>
    >>> result = minimizer(
    ...     function=rosenbrock,
    ...     initial_params=[-1.2, 1.0],
    ...     method='trust_ncg',
    ...     gradient_function=rosenbrock_grad,
    ... )
    >>> result.analytical_gradient_provided
    True
    >>> result.final_params
    [1. 1.]   # More accurate than example 2.
    >>> result.iteration_number
    24        # Same as example 2.
    >>> result.func_evals_number
    25        # A lot less evaluations than example 2.
    >>> result.grad_evals_number
    166

    **Example 4 - Provide a Hessian-vector product for a trust-region
    method.** For quadratic-like objectives in moderate-to-large
    dimensions, exposing `H(x) @ p` lets `'trust_ncg'` solve the
    subproblem with O(n) memory.

    >>> rng = np.random.default_rng(seed=0)
    >>> A = rng.standard_normal((5, 5))
    >>> A = A @ A.T + np.eye(5)                 # SPD
    >>>
    >>> def quadratic_5d(x):
    ...     return 0.5 * x @ A @ x
    >>>
    >>> def quadratic_5d_grad(x):
    ...     return A @ x
    >>>
    >>> def quadratic_5d_hvp(x, p):
    ...     return A @ p                        # constant Hessian
    >>>
    >>> result = minimizer(
    ...     function=quadratic_5d,
    ...     initial_params=rng.standard_normal(5),
    ...     method='trust_ncg',
    ...     gradient_function=quadratic_5d_grad,
    ...     hessian_vector_function=quadratic_5d_hvp,
    ... )
    >>> result.hessian_vector_product_provided
    True
    >>> np.allclose(result.final_params, 0.0)
    True
    >>> result.final_params
    [ 2.77555756e-17 -5.55111512e-17 -6.93889390e-18 -1.11022302e-16
      2.77555756e-17]

    **Example 5 - Box-constrained optimization where the optimum is
    on a bound.** L-BFGS handles bounds through the augmented-Lagrangian
    outer loop. The example pushes the unconstrained optimum (the
    origin) outside the feasible region so the solver has to settle at
    a corner.

    >>> def shifted_sphere(x):
    ...     return np.sum((x - 3.0) ** 2)       # unconstrained min at [3, 3]
    >>>
    >>> result = minimizer(
    ...     function=shifted_sphere,
    ...     initial_params=[0.0, 0.0],
    ...     method='lbfgs',
    ...     bounds=([-1.0, -1.0], [1.0, 1.0]),  # feasible only in [-1, 1]^2
    ... )
    >>> result.is_bounded
    True
    >>> result.final_params
    [1.00000004 1.00000004]

    **Example 6 - Equality constraint.** Find the point on the line
    x + y = 3 closest to (3, 2). The augmented-Lagrangian outer
    loop drives the constraint violation to zero.

    >>> def objective_eq(x):
    ...     return (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2
    >>>
    >>> constraints = (
    ...     {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 3.0},
    ... )
    >>>
    >>> result = minimizer(
    ...     function=objective_eq,
    ...     initial_params=[0.0, 0.0],
    ...     method='trust_lanczos',
    ...     constraints=constraints,
    ... )
    >>> result.final_params
    [2. 1.]
    >>> result.is_constrained
    True
    >>> result.constraint_count
    1

    **Example 7 - Inequality constraints + bounds.** Inequalities are
    written in the form c(x) >= 0 (i.e. the feasible set is where
    every constraint is non-negative). Here, a 2-D parabola is minimized
    subject to a half-plane constraint and a bounding box.

    >>> def objective_ineq(x):
    ...     return x[0] ** 2 + x[1] ** 2
    >>>
    >>> constraints = (
    ...     # x[0] + x[1] >= 1
    ...     {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1.0},
    ... )
    >>>
    >>> result = minimizer(
    ...     function=objective_ineq,
    ...     initial_params=[2.0, 2.0],
    ...     method='bfgs',
    ...     constraints=constraints,
    ...     bounds=([-5.0, -5.0], [5.0, 5.0]),
    ... )
    >>> result.final_params
    [0.49999997 0.49999997]
    >>> result.constraint_count
    1
    >>> result.is_bounded
    True

    **Example 8 - Global optimization with DIRECT.** Rastrigin's
    function has a regular grid of local minima around the global
    minimum at the origin. DIRECT is deterministic and derivative-free,
    so it is well suited to this kind of problem in low dimensions.

    >>> def rastrigin(x):
    ...     return 10.0 * len(x) + np.sum(
    ...         x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)
    ...     )
    >>>
    >>> result = minimizer(
    ...     function=rastrigin,
    ...     method='direct',
    ...     bounds=([-3.12, -3.12], [6.12, 6.12]),
    ...     method_params={'f_min': 1e-2}
    ... )
    >>> result.is_global_method
    True
    >>> np.allclose(result.final_params, 0.0, atol=1e-2)
    True
    >>> result.final_cost
    0.001548170275121663

    **Example 9 - Global optimization with dual annealing and a
    custom local refiner.** Dual annealing combines a stochastic global
    explorer with a local solver called periodically for refinement.
    For smooth pieces, refining with `'lbfgs'` rather than the
    default speeds things up a lot.

    >>> result = minimizer(
    ...     function=rastrigin,
    ...     method='annealing',
    ...     bounds=([-1e7] * 5, [1e7] * 5),
    ...     method_params={
    ...         'seed': 42,
    ...         'f_min': 1e-16, # The algorithm stop when it reaches this value
    ...         'inner_minimizer_params': {
    ...             'method': 'lbfgs',
    ...             'max_iters': 100,
    ...         },
    ...     },
    ... )
    >>> result.is_global_method
    True
    >>> result.final_cost
    0.0   # < f_min = 1e-16
    >>> result.final_params
    [-5.32683077e-14 -1.59122346e-13  5.01454115e-12  2.98022279e-13
     2.75254025e-13]


    **Example 10 - Tracking the trajectory with a callback.** Useful
    for plotting convergence curves or for early stopping based on
    something the solver does not check natively.

    >>> trajectory = []
    >>> def record_iteration(
    ...     iter_num, params, cost, gradient, gradient_norm,
    ...     alpha, direction,
    ... ):
    ...     trajectory.append((iter_num, cost, gradient_norm))
    >>>
    >>> result = minimizer(
    ...     function=rosenbrock,
    ...     initial_params=[-1.2, 1.0],
    ...     method='lbfgs',
    ...     gradient_function=rosenbrock_grad,
    ...     callback=record_iteration,
    ... )
    >>> trajectory[0]
    (1, 4.750587318071407, 35.38321878189292)
    >>> trajectory[-1]
    (39, 2.8992563920715254e-22, 7.190284830757114e-11)
    >>> trajectory[-1][1] < trajectory[0][1]    # cost decreased
    True

    **Example 11 - Read the OptimizationResult.** The result object is
    the same regardless of method, so a single inspection idiom works
    across the board. It also behaves like a numpy.ndarray of the final
    parameters.

    >>> result = minimizer(
    ...     function=rosenbrock,
    ...     initial_params=[0.0, 0.0],
    ...     method='trust_ncg',
    ... )
    >>>
    >>> # Most-used attributes
    >>> result.success
    True
    >>> result.final_params   # read-only ndarray
    [0.99999998 0.99999997]
    >>> result.final_cost
    2.3274002517737543e-16   # numerically 0
    >>> result.termination_reason
    Gradient norm (8.39e-12) is below tolerance (1.00e-08).
    >>>
    >>> # Diagnostic counters
    >>> result.iteration_number
    18
    >>> result.func_evals_number
    491
    >>> result.grad_evals_number
    118
    >>> result.elapsed_time
    0.02986431121826172   # in seconds
    >>>
    >>> # NumPy-style access (forwarded to final_params)
    >>> result.shape, result.ndim, result.size
    (2,) 1 2
    >>> result[0]
    0.9999999847441806
    >>> np.asarray(result)
    [0.99999998 0.99999997]
    >>>
    >>> # Conversion helpers
    >>> writable_copy = result.as_array()    # writable copy
    >>> result.to_list()   # plain Python list
    [0.9999999847441806, 0.9999999694883385]
    >>> result.to_dict()   # JSON-friendly dict
    [Omitted here due to its length]
    >>>
    >>> # Two complementary string forms
    >>> print(result)   # compact summary
    OptimizationResult (compact)
    ============================
    method                 : trust_ncg
    success                : True
    final_cost             : 2.3274e-16
    final_params           : array([0.99999998, 0.99999997])
    termination_reason     : Gradient norm (8.39e-12) is below tolerance (1.00e-08).
    iteration_number       : 18
    func_evals_number      : 491
    grad_evals_number      : 118
    elapsed_time           : 29.864 ms   # will vary
    >>> print(result.summary('full'))   # full configuration
    [Omitted here due to its length]
    >>>
    >>> # Configuration that produced the result
    >>> result.method
    trust_ncg
    >>> result.tolerances
    {'gradient': 1e-08, 'relative': 1e-16}
    >>> result.method_params   # user-provided merged with the defaults
    [Omitted here due to its length]
    >>> result.analytical_gradient_provided
    False
    >>>
    >>> # Compare against a rerun with a different method
    >>> other = minimizer(
    ...     function=rosenbrock,
    ...     initial_params=[0.0, 0.0],
    ...     method='lbfgs',
    ... )
    >>> result.allclose(other)   # Only final_params and final_cost compared
    True
    >>> abs(result.final_cost - other.final_cost) < 1e-10
    True

    References
    ----------
    [1] Nocedal, J., & Wright, S. J. (2006).
        **Numerical Optimization** (2nd ed.).
        Springer Series in Operations Research and Financial
        Engineering. Springer, New York.
        https://doi.org/10.1007/978-0-387-40065-5

    [2] Kochenderfer, M. J., & Wheeler, T. A. (2019).
        **Algorithms for Optimization**.
        MIT Press.
        ISBN: 978-0-262-03942-0

    [3] SciPy contributors (2026).
        SciPy.optimize._linesearch.py source code (Version 1.17.1).
        GitHub
        https://github.com/scipy/scipy/blob/bb6b9da396f15355efdb2e28bdfa1aead105ce92/scipy/optimize/_linesearch.py

    [4] Wikipedia contributors.
        **Conjugate Gradient Method**.
        https://en.wikipedia.org/wiki/Conjugate_gradient_method

    [5] Wikipedia contributors.
        **Nonlinear Conjugate Gradient Method**.
        https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method

    [6] Wikipedia contributors.
        **BFGS Method**.
        https://en.wikipedia.org/wiki/BFGS

    [7] Wikipedia contributors.
        **Limited-memory BFGS (L-BFGS)**.
        https://en.wikipedia.org/wiki/LBFGS

    [8] Gould, N. I. M., Lucidi, S., Roma, M., & Toint, P. L. (1999).
        **Solving the trust-region subproblem using the Lanczos
        method**. SIAM Journal on Optimization, 9(2), 504-525.
        https://doi.org/10.1137/S1052623497322735

    [9] Tsallis, C. & Stariolo, D. A. (1996).
        **Generalized simulated annealing**.
        Physica A, 233(1-2), 395-406.
        https://doi.org/10.1016/S0378-4371(96)00271-3

    [10] Xiang, Y., Gubian, S. & Martin, F. (2017).
         **Generalized Simulated Annealing**.
         Computational Optimization in Engineering - Paradigms and
         applications.
         http://dx.doi.org/10.5772/66071

    [11] Wikipedia contributors.
         **q-Gaussian distribution**.
         https://en.wikipedia.org/wiki/Q-Gaussian_distribution#Related_distributions
    """

    minimizer_start_time = time.time()


    # NON-METHOD PARAMETERS VALIDATION


    # The following function is called by every algorithm and, in addition, by
    # every nested minimizer call
    function_call_count = 0
    def _function(params: np.ndarray) -> float:
        nonlocal function_call_count
        function_call_count += 1
        return float(function(params))



    if method not in AVAILABLE_METHODS:
        raise ValueError(f"Specified method ({method}) must be in "
                         f"{AVAILABLE_METHODS}.")



    if initial_params is None and method not in METHODS_THAT_REQUIRE_BOUNDS:
        raise ValueError(f"Initial params must be provided if method is "
                         f"in {LOCAL_OPTIMIZATION_METHODS}.")

    if initial_params is not None and method in METHODS_THAT_REQUIRE_BOUNDS:
        warn(f"Provided method ({method}) does not need initial params. "
             f"They will be ignored.", RuntimeWarning, stacklevel=2)

    _initial_params = (
        None if initial_params is None
        else np.atleast_1d(np.asarray(initial_params, dtype=float)).copy()
    )



    DEFAULT_PARAMS = {
        'nesterov': {'learning_rate': 1e-4, 'momentum': 1e-1,
            'decay_rate': 0.0},

        'adam': {'learning_rate': 1e-3, 'beta1': 0.9, 'beta2': 0.999,
            'decay_rate': 0.0},

        'conjugate_gradient': {'initial_alpha': 1.0, 'max_alpha': np.inf,
            'bracketing_max_iter': 300, 'wolfe_c1': 1e-4, 'wolfe_c2': 0.4,
            'zoom_max_iter': 500, 'zoom_epsilon': 0.},

        'newton_conjugate_gradient': {'initial_alpha': 1.0,
            'max_alpha': np.inf, 'bracketing_max_iter': 300, 'wolfe_c1': 1e-4,
            'wolfe_c2': 0.9, 'zoom_max_iter': 500, 'zoom_epsilon': 0.,
            'hvp_h': 'auto', 'hvp_point_number': 2, 'inner_max_iter': 'auto'},

        'bfgs': {'initial_alpha': 1.0, 'max_alpha': np.inf,
            'bracketing_max_iter': 300, 'wolfe_c1': 1e-4, 'wolfe_c2': 0.9,
            'zoom_max_iter': 500, 'zoom_epsilon': 0.},

        'lbfgs': {'memory_size': 20, 'initial_alpha': 1.0,
            'max_alpha': np.inf, 'bracketing_max_iter': 300, 'wolfe_c1': 1e-4,
            'wolfe_c2': 0.9, 'zoom_max_iter': 500, 'zoom_epsilon': 0.},

        'trust_ncg': {'inner_max_iter': 'auto', 'grad_rel_tol': 1e-3,
            'abs_tol': 1e-8, 'hvp_h': 'auto', 'hvp_point_number': 2,
            'initial_trust_radius': 1.0, 'max_trust_radius': np.inf,
            'eta': 0.15, 'rho_make_smaller_threshold': 0.25,
            'rho_make_bigger_threshold': 0.75,
            'make_trust_radius_smaller_multiplier': 0.25,
            'make_trust_radius_bigger_multiplier': 2},

        'trust_lanczos': {'inner_max_iter': 'auto', 'grad_rel_tol': 1e-3,
            'abs_tol': 1e-8, 'hvp_h': 'auto', 'hvp_point_number': 2,
            'initial_trust_radius': 1.0, 'max_trust_radius': np.inf,
            'eta': 0.15, 'rho_make_smaller_threshold': 0.25,
            'rho_make_bigger_threshold': 0.75,
            'make_trust_radius_smaller_multiplier': 0.25,
            'make_trust_radius_bigger_multiplier': 2},

        'direct': {'max_fun_evals': 'auto', 'max_iters_with_no_progress': 100,
            'max_rectangles': 'auto', 'max_diameter': 0.05,
            'f_min': -np.inf, 'len_tol': 'auto', 'vol_tol': 'auto'},

        'annealing': {'inner_minimizer_params': {}, 'x0': None, 'seed': None,
            'max_fun_evals': 'auto', 'f_min': -np.inf, 'temp_init': 1e-2,
            'no_local_search': False, 'reanneal_interval': 'auto',
            'visit_const': 2.62, 'accept_const': 1.5, 'cooling_power': 1.}
    }


    method_params = {} if method_params is None else method_params
    final_method_params = _merge_user_dict_with_defaults(
        user_dict=method_params,
        defaults=DEFAULT_PARAMS[method],
        dict_label='method_params'
    )



    if bounds is not None:
        try:
            lb_raw, ub_raw = bounds
        except (TypeError, ValueError):
            if method not in METHODS_THAT_REQUIRE_BOUNDS:
                raise ValueError("Bounds must be an array-like (lb, ub), "
                                 "both array-like or None.")
            else:
                raise ValueError("Bounds must be an array-like (lb, ub), "
                                 "both array-like.")


        lb = np.array(lb_raw) if lb_raw is not None else None
        ub = np.array(ub_raw) if ub_raw is not None else None


        if method in METHODS_THAT_REQUIRE_BOUNDS:
            if lb is None or ub is None:
                raise ValueError(f"Lower bound or Upper bound can't be None "
                                 f"when method is in "
                                 f"{METHODS_THAT_REQUIRE_BOUNDS}.")

            if (
                (lb.size == 1 and ub.size != 1)
                or (lb.size != 1 and ub.size == 1)
            ):
                raise ValueError(
                    f"Lower and upper bound must be either both scalar "
                    f"or both array-like of the same length when method "
                    f"is in {METHODS_THAT_REQUIRE_BOUNDS}. "
                    f"Got lb.size={lb.size}, ub.size={ub.size}."
                )

            if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
                raise ValueError(f"Lower bound and Upper bound must be "
                                 f"finite entirely when method is in "
                                 f"{METHODS_THAT_REQUIRE_BOUNDS}.")

            dim_num = lb.size

        else:
            dim_num = len(_initial_params)

            if lb is None: lb = -np.inf * np.ones(dim_num)
            if ub is None: ub = np.inf * np.ones(dim_num)

            if lb.size == 1: lb = np.ones(dim_num) * lb.item()
            if ub.size == 1: ub = np.ones(dim_num) * ub.item()


        if lb.size != ub.size:
            raise ValueError(
                "Bounds must be a sequence of 2 array-like, "
                "both of the same size."
            )


        if np.any(lb >= ub):
            raise ValueError(
                "Each lower bound must be strictly less than its "
                "corresponding upper bound."
            )

        _bounds = np.asarray([lb, ub], dtype=float)

    else:
        if method in METHODS_THAT_REQUIRE_BOUNDS:
            raise ValueError(f"Bounds must be provided if method is in "
                             f"{METHODS_THAT_REQUIRE_BOUNDS}.")

        else:
            dim_num = len(_initial_params)
            # if bounds is None and method does not require bounds, effectively
            # disable bounds
            _bounds = np.array([-np.inf * np.ones(dim_num),
                                np.inf * np.ones(dim_num)])



    if constraints is not None and not isinstance(constraints, tuple):
        if isinstance(constraints, dict):
            constraints = (constraints,)

        else:
            raise ValueError(
                "Provided constraints must be None or a Tuple of Dictionaries "
                "(all with keys 'type', 'fun' and optionally 'grad')."
            )



    UNCONSTRAINED_LOCAL_OPTIMIZATION = False
    GLOBAL_OPTIMIZATION_WITHOUT_CONSTRAINTS = False
    CONSTRAINED_OPTIMIZATION = False

    if (
        method not in METHODS_THAT_REQUIRE_BOUNDS
        and constraints is None
        and bounds is None
    ):
        UNCONSTRAINED_LOCAL_OPTIMIZATION = True

    elif method in GLOBAL_OPTIMIZATION_METHODS and constraints is None:
        GLOBAL_OPTIMIZATION_WITHOUT_CONSTRAINTS = True

    else:
        CONSTRAINED_OPTIMIZATION = True



    if (
        gradient_function
        and method not in METHODS_THAT_REQUIRE_GRADIENT
        and constraints is None
    ):
        warn_msg = (
            f"Provided method ({method}) does not need gradient "
            f"information. It will be ignored."
        )
        if method in METHODS_WITH_INNER_MINIMIZER:
            warn_msg += (
                " If the user wants to pass a gradient function to a inner "
                "minimizer, method_params={'inner_minimizer_params': "
                "{'gradient_function': <the function>}} must be used."
            )

        warn(warn_msg, RuntimeWarning, stacklevel=2)

    gradient_call_count = 0
    if gradient_function:
        def _grad_func(point: np.ndarray) -> np.ndarray:
            nonlocal gradient_call_count
            gradient_call_count += 1
            return np.atleast_1d(
                np.squeeze(np.asarray(gradient_function(point), dtype=float))
            )

    else:
        _numerical_gradient = nth_numerical_derivative(
            function_to_differentiate=_function,
            derivative_order=1,
            point_number=gradient_point_number,
        )
        def _grad_func(point: np.ndarray) -> np.ndarray:
            nonlocal gradient_call_count
            gradient_call_count += 1
            return np.atleast_1d(_numerical_gradient(point).derivative)



    if hessian_function:
        if method not in METHODS_THAT_REQUIRE_HESSIAN:
            warn_msg = (
                f"Provided method ({method}) does not need hessian "
                f"information. It will be ignored."
            )
            if method in METHODS_WITH_INNER_MINIMIZER:
                warn_msg += (
                    " If the user wants to pass a hessian function to a "
                    "inner minimizer, method_params={'inner_minimizer_params': "
                    "{'hessian_function': <the function>}} must be used."
                )

            warn(warn_msg, RuntimeWarning, stacklevel=2)

        def _hess_vector_product_func(
            point: np.ndarray, vector: np.ndarray
        ) -> np.ndarray:
            hessian = np.asarray(hessian_function(point), dtype=float)

            if hessian.ndim != 2:
                raise ValueError("Hessian must be 2-dimensional.")

            rows_n, columns_n = hessian.shape
            if rows_n != columns_n:
                raise ValueError("Hessian must be square.")

            if rows_n != point.size:
                raise ValueError("Hessian dimension must match x dimension.")

            return hessian @ vector



    if CONSTRAINED_OPTIMIZATION:
        if hessian_function:
            warn("hessian_function is ignored when performing constrained "
                "optimization.", RuntimeWarning, stacklevel=2)

        if hessian_vector_function:
            warn("hessian_vector_function is ignored when performing "
                 "constrained optimization.", RuntimeWarning, stacklevel=2)



    if hessian_vector_function:
        if method not in METHODS_THAT_REQUIRE_HESSIAN:
            warn_msg = (
                f"Provided method ({method}) does not need hessian "
                f"information. It will be ignored."
            )
            if method in METHODS_WITH_INNER_MINIMIZER:
                warn_msg += (
                    " If the user wants to pass a hessian vector product "
                    "function to a inner minimizer,"
                    "method_params={'inner_minimizer_params': "
                    "{'hessian_vector_function': <the function>}} must be used."
                )

            warn(warn_msg, RuntimeWarning, stacklevel=2)

        if hessian_function and method not in METHODS_WITH_INNER_MINIMIZER:
            warn("Both 'hessian_function' and 'hessian_vector_function' were "
                 "provided, 'hessian_function' is ignored.", RuntimeWarning,
                 stacklevel=2)

        def _hess_vector_product_func(
            point: np.ndarray, vector: np.ndarray
        ) -> np.ndarray:
            hvp = np.asarray(
                hessian_vector_function(point, vector), dtype=float
            ).ravel()

            if hvp.size != point.size:
                raise ValueError("Hessian dimension must match x dimension.")

            return hvp



    if CONSTRAINED_OPTIMIZATION:
        # The Augmented Lagrangian defaults differ between local and global
        # methods in only two keys: max_outer_iter, mu_init; everything
        # else is shared.
        SHARED_CONSTRAINED_PARAMS = {
            'mu_factor': 10.0,
            'tol_constraints': 1e-7,
            'tol_grad': 1e-4,
            'const_gradient_point_number': 2,
            'verbose_inner': False,
            'verbose_freq_inner': 1,
            'callback_inner': None,
            'bounds_penalty_scale': 1e1,
            'make_initial_param_unconstrained': False,
            'satisfaction_threshold': -np.inf,
            'restart_noise_amplitude': 1e-1,
            'maximum_restarts': 0,
            'seed': None,
        }

        if method in GLOBAL_OPTIMIZATION_METHODS:
            DEFAULT_CONSTRAINED_PARAMS = {
                'max_outer_iter': 10,
                'mu_init': 1e5,
                **SHARED_CONSTRAINED_PARAMS,
            }

        else:
            DEFAULT_CONSTRAINED_PARAMS = {
                'max_outer_iter': 100,
                'mu_init': 10.0,
                **SHARED_CONSTRAINED_PARAMS,
            }

        constrained_params = (
            {} if constrained_params is None else constrained_params
        )
        final_constrained_params = _merge_user_dict_with_defaults(
            user_dict=constrained_params,
            defaults=DEFAULT_CONSTRAINED_PARAMS,
            dict_label='constrained_params'
        )



    if isinstance(max_iters, str):
        if max_iters != 'auto':
            raise ValueError(
                f"max_iters must be 'auto' or a positive integer. "
                f"Got {max_iters!r}."
            )

        if method == 'annealing':
            _max_iters = _bounds.shape[1] * 200000
        elif method == 'direct':
            _max_iters = _bounds.shape[1] * 1000
        else:
            _max_iters = len(_initial_params) * 200000

    else:
        if (
            not isinstance(max_iters, int)
            or isinstance(max_iters, bool)
        ):
            raise TypeError(
                f"max_iters must be 'auto' or a positive integer. "
                f"Got {type(max_iters).__name__}."
            )

        if max_iters < 1:
            raise ValueError(
                "Specified max_iters must be greater than zero."
            )

        _max_iters = int(max_iters)



    # Relative tolerance a little less than np.finfo(float).eps (~2.2e-16)
    default_tolerances = {'gradient':1e-8, 'relative': 1e-16}

    tolerances = {} if tolerances is None else tolerances
    final_tolerances = _merge_user_dict_with_defaults(
        user_dict=tolerances,
        defaults=default_tolerances,
        dict_label='tolerances',
    )

    for tolerance_name, tolerance_value in final_tolerances.items():
        if float(tolerance_value) < 0:
            raise ValueError(
                f"{tolerance_name.title()} tolerance cannot be negative."
            )

        final_tolerances[tolerance_name] = float(tolerance_value)



    _verbose = bool(verbose)

    if verbose_frequency == 'auto':
        if CONSTRAINED_OPTIMIZATION:
            # This refers to the outer loop of Augmented Lagrangian
            _verbose_frequency = 1

        else:
            # This refers to the usual iteration number
            _verbose_frequency = max(1, _max_iters // 10000)

    else:
        if (
            not isinstance(verbose_frequency, int)
            or isinstance(verbose_frequency, bool)
        ):
            raise TypeError(
                f"verbose_frequency must be 'auto' or a positive integer. "
                f"Got {type(verbose_frequency).__name__}."
            )

        if verbose_frequency < 1:
            raise ValueError(
                f"verbose_frequency must be 'auto' or a positive integer. "
                f"Got {verbose_frequency}."
            )

        _verbose_frequency = int(verbose_frequency)

    _TRUST_REGION_INNER_METHOD_MAP = {
        'trust_ncg': 'ncg',
        'trust_lanczos': 'lanczos',
    }
    inner_method = _TRUST_REGION_INNER_METHOD_MAP.get(method)

    common_options = {
        'max_iters': _max_iters,
        'verbose': _verbose,
        'verbose_freq': _verbose_frequency,
        'tolerances': {
            'gradient_tolerance': final_tolerances['gradient'],
            'relative_tolerance': final_tolerances['relative']
        },
        'inner_method': inner_method
    }



    if callback is not None:
        if not isinstance(callback, Callable):
            raise ValueError(
                "Provided callback must be a Callable when it is not None."
            )



    # OPTIMIZER CALLS



    optimizer_map = {
        'nesterov': _nesterov_optimizer,
        'adam': _adam_optimizer,
        'conjugate_gradient': _conjugate_gradient_optimizer,
        'newton_conjugate_gradient': _newton_cg_optimizer,
        'bfgs': _bfgs_optimizer,
        'lbfgs': _lbfgs_optimizer,
        'trust_ncg': _trust_region_optimizer,
        'trust_lanczos': _trust_region_optimizer,
        'direct': _direct_optimizer,
        'annealing': _annealing_optimizer
    }
    optimizer_func = optimizer_map[method]


    if UNCONSTRAINED_LOCAL_OPTIMIZATION:

        if _verbose:
            _print_run_banner(
                title='Local Minimization',
                method=method,
                initial_params=_initial_params,
                bounds=None,
                max_iters=_max_iters,
                verbose_freq=common_options['verbose_freq'],
                tolerances=final_tolerances,
                method_params=final_method_params,
                constrained_params=None,
            )

        if method in METHODS_THAT_REQUIRE_HESSIAN:
            results = optimizer_func(
                function=_function,
                grad_func=_grad_func,
                hvp_function=(
                    None if (
                        hessian_function is None
                        and hessian_vector_function is None
                    )
                    else _hess_vector_product_func
                ),
                initial_params=_initial_params,
                common_options=common_options,
                callback=callback,
                **final_method_params
            )

        elif method in METHODS_THAT_REQUIRE_GRADIENT:
            results = optimizer_func(
                function=_function,
                grad_func=_grad_func,
                initial_params=_initial_params,
                common_options=common_options,
                callback=callback,
                **final_method_params
            )


    elif GLOBAL_OPTIMIZATION_WITHOUT_CONSTRAINTS:
        if _verbose:
            _print_run_banner(
                title='Global Minimization',
                method=method,
                initial_params=None,
                bounds=_bounds,
                max_iters=_max_iters,
                verbose_freq=common_options['verbose_freq'],
                tolerances=final_tolerances,
                method_params=final_method_params,
                constrained_params=None,
            )

        if method in METHODS_WITH_INNER_MINIMIZER:
            inner_minimizer_params = (
                final_method_params['inner_minimizer_params']
            )

            if (
                inner_minimizer_params.get('method') in
                METHODS_THAT_REQUIRE_BOUNDS
            ):
                error_msg = (
                    set(AVAILABLE_METHODS) - set(METHODS_THAT_REQUIRE_BOUNDS)
                )

                raise ValueError(f"Inner method must be in {error_msg}.")

            def inner_solver(function, parameters, bounds):
                return minimizer(
                    function=function,
                    initial_params=np.atleast_1d(
                        np.asarray(parameters, dtype=float)
                    ).copy(),
                    bounds=bounds,
                    method=inner_minimizer_params.get('method', 'trust_ncg'),
                    method_params=inner_minimizer_params.get(
                        'method_params'
                    ),
                    gradient_function=inner_minimizer_params.get(
                        'gradient_function'
                    ),
                    gradient_point_number=inner_minimizer_params.get(
                        'gradient_point_number', 2
                    ),
                    hessian_function=inner_minimizer_params.get(
                        'hessian_function'
                    ),
                    hessian_vector_function=inner_minimizer_params.get(
                        'hessian_vector_function'
                    ),
                    max_iters=inner_minimizer_params.get('max_iters', 'auto'),
                    tolerances=inner_minimizer_params.get('tolerances'),
                    constraints=inner_minimizer_params.get('constraints'),
                    constrained_params=inner_minimizer_params.get(
                        'constrained_params'
                    ),
                    verbose=inner_minimizer_params.get('verbose', False),
                    verbose_frequency=inner_minimizer_params.get(
                        'verbose_frequency', 'auto'
                    ),
                    callback=inner_minimizer_params.get('callback'),
                )

            optimizer_kwargs = {
                key: value
                for key, value in final_method_params.items()
                if key != 'inner_minimizer_params'
            }
            optimizer_kwargs['inner_solver'] = inner_solver

        else:
            optimizer_kwargs = final_method_params

        results = optimizer_func(
            function=_function,
            bounds=_bounds,
            common_options=common_options,
            callback=callback,
            **optimizer_kwargs,
        )


    # Constrained minimization
    else:
        if _verbose:
            _print_run_banner(
                title='Constrained Minimization',
                method=method,
                initial_params=(
                    None if method in METHODS_THAT_REQUIRE_BOUNDS
                    else _initial_params
                ),
                bounds=(
                    _bounds if method in METHODS_THAT_REQUIRE_BOUNDS
                    else None
                ),
                max_iters=_max_iters,
                verbose_freq=common_options['verbose_freq'],
                tolerances=final_tolerances,
                method_params=final_method_params,
                constrained_params=final_constrained_params,
            )

        # Distinguishing inner and outer verbosity / callbacks makes the
        # call sites below substantially easier to read.
        verbose_inner = final_constrained_params.pop('verbose_inner')
        verbose_frequency_inner = final_constrained_params.pop(
            'verbose_freq_inner'
        )
        callback_inner = final_constrained_params.pop('callback_inner')

        def inner_solver(function, parameters, gradient):
            return minimizer(
                function=function,
                initial_params=(
                    np.atleast_1d(np.asarray(parameters, dtype=float)).copy()
                    if method not in METHODS_THAT_REQUIRE_BOUNDS
                    else None
                ),
                method=method,
                method_params=final_method_params,
                gradient_function=(
                    gradient if method in METHODS_THAT_REQUIRE_GRADIENT
                    else None
                ),
                gradient_point_number=gradient_point_number,
                # The inner solver minimises the augmented Lagrangian, not
                # the user's original function. So, let the inner minimizer
                # build numerical Hessians of the Lagrangian function directly.
                hessian_function=None,
                hessian_vector_function=None,
                max_iters=_max_iters,
                tolerances=tolerances,
                verbose=verbose_inner,
                verbose_frequency=verbose_frequency_inner,
                callback=callback_inner,
                constraints=None,
                bounds=(
                    _bounds if method in METHODS_THAT_REQUIRE_BOUNDS else None
                ),
            )

        results = _augmented_lagrangian_optimizer(
            function=_function,
            grad_func=_grad_func,
            initial_params=(
                _initial_params
                if method not in METHODS_THAT_REQUIRE_BOUNDS
                else (ub + lb) / 2 # midpoint
            ),
            inner_solver=inner_solver,
            constraints=constraints,
            bounds=_bounds,
            verbose_outer=_verbose,
            verbose_freq_outer=common_options.get('verbose_freq', 1),
            callback=callback,
            **final_constrained_params,
        )



    # RESULTS OUTPUT



    minimizer_elapsed_time = time.time()-minimizer_start_time

    if method in METHODS_WITH_INNER_MINIMIZER:
        gradient_call_count += results.get("grad_evals_number", 0)

    if _verbose:
        print("-" * 60)
        if results.get('success'):
            print(f"Optimization Finished Successfully in "
                  f"{minimizer_elapsed_time:.3f} s.")
        else:
            print(f"Optimization Finished Unsuccessfully in "
                  f"{minimizer_elapsed_time:.3f} s.")
        print(f"Termination Reason: {results.get('termination_reason')}")
        print(f"Total Iters: {results.get('iteration_number')}")
        print(f"Final Cost: {results.get('final_cost')}")
        print(f"Final Parameters: {results.get('final_params')}")
        print("="*60)


    return OptimizationResult(
        # Core output
        success=results.get('success', False),
        final_params=results.get('final_params'),
        final_cost=results.get('final_cost'),
        termination_reason=results.get('termination_reason'),
        func_evals_number=function_call_count,
        grad_evals_number=gradient_call_count,
        elapsed_time=minimizer_elapsed_time,
        iteration_number=results.get('iteration_number', 0),
        # Configuration
        method=method,
        initial_params=(
            _initial_params if method not in METHODS_THAT_REQUIRE_BOUNDS
            else None
        ),
        bounds=_bounds if (bounds is not None) else None,
        max_iters=_max_iters,
        tolerances=final_tolerances,
        method_params=final_method_params,
        gradient_point_number=gradient_point_number,
        analytical_gradient_provided=(gradient_function is not None),
        analytical_hessian_provided=(hessian_function is not None),
        hessian_vector_product_provided=(hessian_vector_function is not None),
        constraint_count=(0 if constraints is None else len(constraints)),
        constrained_params=(
            None if not CONSTRAINED_OPTIMIZATION
            else final_constrained_params
        ),
    )
