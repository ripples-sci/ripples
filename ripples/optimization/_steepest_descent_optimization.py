"""
Steepest-descent algorithms for local optimization.

All algorithms implemented in this module are based on the classic and
well-known gradient-descent algorithm which at each iteration takes steps
in the opposite direction of the gradient (maximum/steepest descent).

Contains
--------
_nesterov_optimizer
    Nesterov Accelerated Gradient (NAG) optimizer with momentum and
    learning rate decay.

_adam_optimizer
    Adaptive Moment Estimation (ADAM) optimizer with learning rate decay.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Any, Optional, Dict, Callable

from ._utils import _initial_check, _check_termination



def _nesterov_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int, np.ndarray, float, np.ndarray, float,
                                 float], None]],
    learning_rate: float,
    momentum: float,
    decay_rate: float
) -> Dict[str, Any]:
    """
    Nesterov Accelerated Gradient (NAG) optimizer with momentum and
    learning rate decay.

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
        - 'max_iters': int - Maximum number of iterations
        - 'verbose': bool - Whether to print progress
        - 'verbose_freq': int - Frequency of verbose output
        - 'tolerances': Dict[str, float] - Convergence tolerances
    callback : Optional[Callable]
        Function called after each iteration with signature:
        (iter, params, cost, gradient, gradient_norm, learning_rate).
    learning_rate : float
        Initial learning rate (step size).
    momentum : float
        Momentum coefficient (typically between 0.9 and 0.99).
    decay_rate : float
        Learning rate decay factor. Learning rate is updated as:
        lr_k = lr_0 / (1 + decay_rate * k)

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'success': Boolean indicating convergence.
        - 'final_params': The optimized parameters.
        - 'final_cost': The value of the function at the optimized parameters.
        - 'iteration_number': Total iterations performed.
        - 'termination_reason': String description of why the optimizer stopped.

    Mathematical Description
    ------------------------
    Nesterov Accelerated Gradient (NAG), also called Nesterov Momentum,
    is an improvement over classical momentum methods that provides
    better convergence guarantees, particularly for convex functions.

    ---

    Standard gradient descent with momentum updates parameters using:

        v_{k+1} = mu * v_k - eta * grad(f(x_k))
        x_{k+1} = x_k + v_{k+1}

    where:
    - x_k is the current parameter vector
    - v_k is the velocity (accumulated gradient)
    - mu is the momentum coefficient (0 <= mu <= 1)
    - eta is the learning rate

    The momentum term mu * v_k accumulates past gradients, allowing the
    optimizer to:

    - Build velocity in consistent descent directions.
    - Dampen oscillations in directions of high curvature.
    - Accelerate convergence in shallow directions.

    ---

    The key of Nesterov's method is: instead of computing the gradient at the
    current position x_k, compute it at an approximate future position
    (lookahead point). This lookahead position is obtained by taking a step in
    the direction of the current momentum:

        x_lookahead = x_k + mu * v_k

    The gradient is then evaluated at this lookahead point, giving the
    update rules:

        x_lookahead = x_k + mu * v_k
        v_{k+1} = mu * v_k - eta * grad(f_lookahead)
        x_{k+1} = x_k + v_{k+1}


    The lookahead gradient grad(f(x_k + mu * v_k)) provides a "corrective" view:

    - If the momentum is towards a minimum, the lookahead gradient will be
    small, allowing the momentum to continue.

    - If the momentum is overshooting or heading in a poor direction, the
    lookahead gradient will be large in a corrective direction, slowing or
    redirecting the momentum.

    ---

    This implementation also includes a learning rate decay:

        eta_k = eta_0 / (1 + decay_rate * k)

    As iterations progress, the learning rate decreases, which allows
    large initial steps for fast progress, enables fine-tuning near
    convergence and improves stability in later iterations.

    References
    ----------
    [1] Kochenderfer, M. J., & Wheeler, T. A. (2019).
        **Algorithms for Optimization**.
        MIT Press.
        Section 5.4.
        ISBN: 978-0-262-03942-0
    """

    for var_name, value in [('learning_rate', learning_rate),
                        ('momentum', momentum),
                        ('decay_rate', decay_rate)]:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{var_name} must be of type int or float")
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{var_name} must be non-negative and finite")

        # values >= 1 give a non-decaying geometric accumulation of past
        # gradients and the velocity diverges.
        if var_name == 'momentum' and not value < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")


    params = initial_params.copy()
    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    tolerances = common_options['tolerances']
    verbose_freq = common_options['verbose_freq']


    _learning_rate = learning_rate
    velocity = np.zeros_like(params)

    grad = grad_func(params)
    if grad.size != params.size:
        raise ValueError(f"Provided initial params (length = {params.size}) "
                         f"mismatch its dimensions with provided gradient "
                         f"(length = {grad.size}).")

    current_cost = function(params)
    previous_cost = np.inf

    grad_norm = float(np.linalg.norm(grad))
    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check


    if verbose:
        print(f"{'Iter':>5s} | {'Cost':>12s} | {'Grad Norm':>12s} | "
              f"{'Learning Rate':>15s}")
        print("-" * 53)
        print(f"{0:5d} | {current_cost:12.6f} | {grad_norm:12.2e} | "
              f"{_learning_rate:15.6f}")


    for current_iter in range(1, max_iters + 1):
        # Nesterov lookahead
        lookahead_params = params + momentum * velocity
        grad_lookahead = grad_func(lookahead_params)

        # Update velocity with Nesterov momentum
        velocity = momentum * velocity - _learning_rate * grad_lookahead
        params += velocity

        previous_cost = current_cost
        current_cost = function(params)

        # Gradient at actual position
        grad = grad_func(params)
        grad_norm = float(np.linalg.norm(grad))

        terminate, converged, termination_reason = _check_termination(
            current_cost, previous_cost, grad_norm, **tolerances
        )

        if verbose and (current_iter % verbose_freq == 0 or converged):
            print(f"{current_iter:5d} | {current_cost:12.6f} | "
                  f"{grad_norm:12.2e} | {_learning_rate:15.6f}")
        if callback:
            callback(current_iter, params, current_cost, grad, grad_norm,
                     _learning_rate)

        if terminate: break


        _learning_rate = learning_rate / (1.0 + decay_rate * current_iter)

    else:
        termination_reason = "Maximum number of iterations reached."
        converged = False


    return {
        'success': converged,
        'final_params': params,
        'final_cost': current_cost,
        'iteration_number': current_iter,
        'termination_reason': termination_reason}



def _adam_optimizer(
    function: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    common_options: Dict[str, Any],
    callback: Optional[Callable[[int, np.ndarray, float, np.ndarray, float,
                                 float], None]],
    learning_rate: float,
    beta1: float,
    beta2: float,
    decay_rate: float
) -> Dict[str, Any]:
    """
    Adaptive Moment Estimation (ADAM) optimizer with learning rate decay.

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
        - 'max_iters': int - Maximum number of iterations
        - 'verbose': bool - Whether to print progress
        - 'verbose_freq': int - Frequency of verbose output
        - 'tolerances': Dict[str, float] - Convergence tolerances
    callback : Optional[Callable]
        Function called after each iteration with signature:
        (iter, params, cost, gradient, gradient_norm, learning_rate).
    learning_rate : float
        Initial learning rate (step size).
    beta1 : float
        Exponential decay rate for the first moment estimates
        (typically 0.9).
    beta2 : float
        Exponential decay rate for the second moment estimates
        (typically 0.999).
    decay_rate : float
        Learning rate decay factor. Learning rate is updated as:
        lr_t = lr_0 / (1 + decay_rate * t)

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'success': Boolean indicating convergence.
        - 'final_params': The optimized parameters.
        - 'final_cost': The value of the function at the optimized parameters.
        - 'iteration_number': Total iterations performed.
        - 'termination_reason': String description of why the optimizer stopped.

    Mathematical Description
    ------------------------
    ADAM (Adaptive Moment Estimation) is an adaptive learning rate
    optimization algorithm. It computes adaptive learning rates for each
    parameter by maintaining running averages of both the gradient and
    its squared values.

    ---

    Traditional gradient descent uses a single fixed learning rate for
    all parameters:

        x_{k+1} = x_k - eta * grad(f(x_k))

    This has the limitation that a single learning rate may be too large
    for some parameters and too small for others, especially when
    parameters have different scales.

    In addition the gradient magnitude can vary significantly across
    iterations, making it hard to choose an appropriate fixed learning rate.
    And, the convergence can be slow in regions with small gradients.

    In order to address this, ADAM maintains two types of moment
    estimates:

    - m_k: First moment estimate, acts like momentum, accumulates decaying
    average of past gradients, helps accelerate convergence in consistent
    directions. beta_1 controls how much history to retain.

    - v_k: Second moment estimate, accumulates a decaying average of past square
    gradients, allows adaptive scaling (parameters with large gradients get
    smaller) for effective learning rates. beta_2 controls the decay.

    In early iterations, m_k and v_k are biased toward zero because they
    are initialized at zero. The bias correction terms (1 - beta_1^k) and
    (1 - beta_2^k) compensate for this initialization bias. As k grows,
    these correction factors approach 1, effectively removing the
    correction.


    This implementation works in the following way (at each iteration k):

    Updates biased moments:

        m_k = beta_1 * m_{k-1} + (1 - beta_1) * g_k
        v_k = beta_2 * v_{k-1} + (1 - beta_2) * g_k^2

    Compute bias-corrected moments:

        m_hat_k = m_k / (1 - beta_1^k)
        v_hat_k = v_k / (1 - beta_2^k)

    And update parameters:

        x_{k+1} = x_k - eta * m_hat_k / (sqrt(v_hat_k))

    so, the step size adapts for each parameter with eta / sqrt(v_hat_k),
    parameters with consistently large gradients get smaller steps and vice
    versa.


    This implementation also includes a learning rate decay:

        eta_k = eta_0 / (1 + decay_rate * k)

    As iterations progress, the learning rate decreases, which allows
    large initial steps for fast progress, enables fine-tuning near
    convergence and improves stability in later iterations.

    References
    ----------
    [1] Kochenderfer, M. J., & Wheeler, T. A. (2019).
        **Algorithms for Optimization**.
        MIT Press.
        Section 5.8.
        ISBN: 978-0-262-03942-0
    """

    for var_name, var_value in {'learning_rate': learning_rate,
            'beta1': beta1, 'beta2': beta2, 'decay_rate': decay_rate}.items():

        if not isinstance(var_value, (int, float)):
            raise TypeError(f"{var_name} must be of type int or float")
        if var_value < 0 or not np.isfinite(var_value):
            raise ValueError(f"{var_name} must be non-negative and finite")

        if var_name in ('beta1','beta2') and not 0 < var_value < 1:
            raise ValueError(f"{var_name} must be 0 < {var_name} < 1")


    params = initial_params.copy()
    max_iters = common_options['max_iters']
    verbose = common_options['verbose']
    tolerances = common_options['tolerances']
    verbose_freq = common_options['verbose_freq']

    # ADAM state
    _learning_rate = learning_rate
    first_moment = np.zeros_like(params)
    second_moment = np.zeros_like(params)

    grad = grad_func(params)
    if grad.size != params.size:
        raise ValueError(f"Provided initial params (length = {params.size}) "
                         f"mismatch its dimensions with provided gradient "
                         f"(length = {grad.size}).")

    grad_norm = float(np.linalg.norm(grad))

    current_cost = function(params)
    previous_cost = np.inf

    init_check = _initial_check(
        grad_norm, tolerances['gradient_tolerance'], params, current_cost
    )
    if not init_check.get('success'): return init_check

    EPSILON = 1e-8


    if verbose:
        print(f"{'Iter':>5s} | {'Cost':>12s} | {'Grad Norm':>12s} | "
              f"{'Learning Rate':>15s}")
        print("-" * 53)
        print(f"{0:5d} | {current_cost:12.6f} | {grad_norm:12.2e} | "
              f"{_learning_rate:15.6f}")


    for current_iter in range(1, max_iters + 1):
        grad = grad_func(params)
        grad_norm = float(np.linalg.norm(grad))


        terminate, converged, termination_reason = _check_termination(
            current_cost, previous_cost, grad_norm, **tolerances
        )
        if terminate: break


        # Update biased moments estimates
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * (grad ** 2)

        # Moments estimates
        first_moment_hat = first_moment / (1.0 - beta1 ** current_iter)
        second_moment_hat = second_moment / (1.0 - beta2 ** current_iter)

        update_step = (
            -_learning_rate * first_moment_hat /
            np.maximum(np.sqrt(second_moment_hat), EPSILON)
        )
        params += update_step

        previous_cost = current_cost
        current_cost = function(params)


        _learning_rate = learning_rate / (1.0 + decay_rate * current_iter)


        if verbose and (current_iter % verbose_freq == 0 or converged):
            print(f"{current_iter:5d} | {current_cost:12.6f} | "
                  f"{grad_norm:12.2e} | {_learning_rate:15.6f}")
        if callback:
            callback(current_iter, params, current_cost, grad, grad_norm,
                     _learning_rate)

    else:
        termination_reason = "Maximum number of iterations reached."
        converged = False


    return {
        'success': converged,
        'final_params': params,
        'final_cost': current_cost,
        'iteration_number': current_iter,
        'termination_reason': termination_reason,
    }
