"""
Utilities for the global optimization algorithms.

Contains
--------
_validate_direct_parameters
    Validates the function-evaluation budget, memory-safety limit,
    forced-split diameter, target value and tolerances that govern the
    DIRECT optimizer in the _global_optimization.py module.

_validate_annealing_parameters
    Validates the box constraints, initial guess, RNG seed,
    function-evaluation budget, target value, initial temperature,
    local-search flag, reannealing interval, Tsallis entropic indices
    and cooling exponent that govern the Dual Annealing optimizer in
    the _global_optimization.py module.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np



def _validate_direct_parameters(
    maxfun,
    max_iters_with_no_progress,
    max_rectangles,
    max_diameter,
    f_min,
    len_tol,
    vol_tol,
) -> None:
    """
    Validate DIRECT algorithm parameters.

    Parameters
    ----------
    maxfun : int or 'auto'
        Maximum number of function evaluations.
    max_iters_with_no_progress : int
        Maximum iterations without improvement before stopping.
    max_rectangles : int or 'auto'
        Maximum number of rectangles (memory safety limit).
    max_diameter : float
        Maximum diameter for forced rectangle splitting.
    f_min : float
        Target function value (stopping criterion).
    len_tol : float or 'auto'
        Tolerance for best rectangle's diameter.
    vol_tol : float or 'auto'
        Tolerance for best rectangle's volume.

    Raises
    ------
    TypeError
        If any parameter has an incorrect type.
    ValueError
        If any parameter has an out-of-range or inconsistent value.
    """

    # Validate maxfun (can be 'auto' or int)
    if maxfun != 'auto':
        if not isinstance(maxfun, int):
            raise TypeError("'maxfun' must be an integer or 'auto'.")
        if maxfun <= 0:
            raise ValueError("'maxfun' must be strictly positive.")

    # Validate max_rectangles (can be 'auto' or int)
    if max_rectangles != 'auto':
        if not isinstance(max_rectangles, int):
            raise TypeError("'max_rectangles' must be an integer or 'auto'.")
        if max_rectangles <= 0:
            raise ValueError("'max_rectangles' must be strictly positive.")

    # Validate max_iters_with_no_progress
    if not isinstance(max_iters_with_no_progress, int):
        raise TypeError("'max_iters_with_no_progress' must be an integer.")
    if max_iters_with_no_progress <= 0:
        raise ValueError("'max_iters_with_no_progress' must be "
                         "strictly positive.")

    # Validate float parameters that can be 'auto'
    for var_name, var_value in {
        'len_tol': len_tol,
        'vol_tol': vol_tol,
    }.items():
        if var_value != 'auto':
            if not isinstance(var_value, (int, float)):
                raise TypeError(f"'{var_name}' must be a real scalar "
                                f"or 'auto'.")
            if not np.isfinite(var_value):
                raise ValueError(f"'{var_name}' must be finite (not inf "
                                 f"or NaN).")
            if var_value <= 0:
                raise ValueError(f"'{var_name}' must be strictly positive.")

    # Validate float parameters that cannot be 'auto'
    for var_name, var_value in {
        'max_diameter': max_diameter,
        'f_min': f_min,
    }.items():
        if not isinstance(var_value, (int, float)):
            raise TypeError(f"'{var_name}' must be a real scalar.")

    # max_diameter must be positive
    if not 0 < max_diameter <= 1:
        raise ValueError("'max_diameter' must be 0 < max_diameter <= 1.")



def _validate_annealing_parameters(
    bounds,
    x0,
    seed,
    max_fun_evals,
    f_min,
    temp_init,
    no_local_search,
    reanneal_interval,
    visit_const,
    accept_const,
    cooling_power,
) -> None:
    """
    Validate Dual Annealing optimizer parameters.

    Parameters
    ----------
    bounds : array of shape (2, n_dimensions)
        Box constraints as (lower_bounds, upper_bounds). Each row must be
        finite and lower_bounds[i] < upper_bounds[i] for all i.
    x0 : array-like or None
        Initial guess. If provided, must be a 1-D array-like of finite
        values with length matching the number of dimensions in bounds.
    seed : int or None
        Seed for the random number generator. If provided, must be a
        non-negative integer.
    max_fun_evals : int or 'auto'
        Maximum number of function evaluations. If not 'auto', must be a
        strictly positive integer.
    f_min : float
        Target global minimum value (stopping criterion). Must be a finite
        real scalar.
    temp_init : float
        Initial temperature. Must be a strictly positive finite real scalar.
    no_local_search : bool
        Whether to disable the local search step.
    reanneal_interval : int or 'auto'
        Iterations between scheduled local search calls and stagnation
        window. If not 'auto', must be a strictly positive integer.
    visit_const : float
        Tsallis entropic index for the visiting distribution (q_v).
        Must be a real scalar in the open interval (1, 3).
    accept_const : float
        Tsallis entropic index for the acceptance criterion (q_a).
        Must be a real scalar in the open interval (1, 3).
    cooling_power : float
        Exponent for the power-law cooling schedule. Must be a strictly
        positive finite real scalar.

    Raises
    ------
    TypeError
        If any parameter has an incorrect type.
    ValueError
        If any parameter has an out-of-range or inconsistent value.
    """

    dim_num = bounds.shape[1]

    # x0
    if x0 is not None:
        try:
            _x0 = np.asarray(x0, dtype=float).ravel()
        except (TypeError, ValueError):
            raise TypeError("'x0' must be convertible to a 1-D numeric array.")
        if _x0.ndim != 1:
            raise ValueError("'x0' must be a 1-D array-like after flattening.")
        if _x0.size != dim_num:
            raise ValueError(
                f"'x0' has {_x0.size} element(s) but 'bounds' defines "
                f"{dim_num} dimension(s). They must match."
            )
        if not np.all(np.isfinite(_x0)):
            raise ValueError("'x0' must contain only finite values (no inf "
                             "or NaN).")

    # seed
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError("'seed' must be a non-negative integer or None.")
        if seed < 0:
            raise ValueError("'seed' must be non-negative.")

    # max_fun_evals and reanneal_interval
    for var_name, var_value in {
        'max_fun_evals': max_fun_evals,
        'reanneal_interval': reanneal_interval,
    }.items():
        if var_value != 'auto':
            if not isinstance(var_value, int):
                raise TypeError(f"'{var_name}' must be a strictly positive "
                                f"integer or 'auto'.")
            if var_value <= 0:
                raise ValueError(f"'{var_name}' must be strictly positive.")

    # no_local_search
    if not isinstance(no_local_search, bool):
        raise TypeError("'no_local_search' must be a boolean.")

    # Finite real scalars: f_min, temp_init, cooling_power
    for var_name, var_value in {
        'f_min':        f_min,
        'temp_init':    temp_init,
        'cooling_power': cooling_power,
    }.items():
        if not isinstance(var_value, (int, float)):
            raise TypeError(f"'{var_name}' must be a real scalar.")
        if not np.isfinite(var_value):
            if not (var_name == 'f_min' and var_value == -np.inf):
                raise ValueError(f"'{var_name}' must be finite "
                                 f"(not inf or NaN).")

    if temp_init <= 0.0:
        raise ValueError("'temp_init' must be strictly positive.")

    if cooling_power <= 0.0:
        raise ValueError("'cooling_power' must be strictly positive.")

    # Tsallis entropic indices: visit_const (q_v), accept_const (q_a)
    for var_name, var_value in {
        'visit_const':  visit_const,
        'accept_const': accept_const,
    }.items():
        if not isinstance(var_value, (int, float)):
            raise TypeError(f"'{var_name}' must be a real scalar.")
        if not np.isfinite(var_value):
            raise ValueError(f"'{var_name}' must be finite (not inf or NaN).")
        if not (1.0 < var_value < 3.0):
            raise ValueError(
                f"'{var_name}' must be in the open interval (1, 3). "
                f"Got {var_value}. "
                f"Values <= 1 make the q-Gaussian non-normalizable; "
                f"values >= 3 make the Student-t degrees of freedom "
                f"non-positive."
            )
