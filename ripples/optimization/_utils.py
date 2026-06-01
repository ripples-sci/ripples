"""
Utilities shared by ***minimizer*** and the optimizer modules
(_steepest_descent_optimization.py, _line_search_optimization.py,
_trust_region_optimization.py, _global_optimization.py,
_constrained_optimization.py).

Any supporting code shared between several inner-optimizer modules its here, so
each numerical strategy file keeps its focus on the strategy itself.

Contains
--------
FLOAT_EPSILON
    Machine epsilon for IEEE-754 double precision (~2.22e-16), the
    smallest positive float `e` such that `1.0 + e != 1.0`. Used as
    a safe lower bound throughout the module wherever a denominator
    must be prevented from collapsing to zero.

AVAILABLE_METHODS, METHODS_THAT_REQUIRE_GRADIENT, METHODS_THAT_REQUIRE_HESSIAN,
GLOBAL_OPTIMIZATION_METHODS, LOCAL_OPTIMIZATION_METHODS,
METHODS_WITH_INNER_MINIMIZER, METHODS_THAT_REQUIRE_BOUNDS
    Method-classification constants. Centralised here so both
    ***minimizer*** and ***OptimizationResult*** classify the chosen
    method through a single source of truth.

_initial_check
    Pre-algorithm validation of the starting point. It indicates when the
    gradient is non-finite, the cost is non-finite, or the gradient norm is
    already below tolerance.

_check_termination
    Standard convergence check that runs at each iteration.

_normalize_settings_dict
    Recursively replaces callables and ndarrays in a dictionary with safe
    placeholders, so the dictionary can be repr'd, compared, and
    converted to a dict through `OptimizationResult.to_dict()`.

OptimizationResult
    Output class for the optimization public functions. Carries the optimized
    parameters and the final cost together with the complete record of the run
    (function and gradient evaluation counts, iteration number, elapsed time,
    termination reason) and every configuration setting that produced the
    result. Behaves like a read-only numpy.ndarray of the final parameters and
    has conversion helpers (`as_array`, `as_float`, `to_list`, `to_dict`),
    equality (`==`, `allclose`), and a `summary`.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple, Union



FLOAT_EPSILON = float(np.finfo(float).eps)



AVAILABLE_METHODS = (
    'nesterov', 'adam', 'conjugate_gradient', 'newton_conjugate_gradient',
    'bfgs', 'lbfgs', 'trust_ncg', 'trust_lanczos', 'direct', 'annealing'
)
LOCAL_OPTIMIZATION_METHODS = (
    'nesterov', 'adam', 'conjugate_gradient', 'newton_conjugate_gradient',
    'bfgs', 'lbfgs', 'trust_ncg', 'trust_lanczos'
)
GLOBAL_OPTIMIZATION_METHODS = ('direct', 'annealing')

METHODS_THAT_REQUIRE_GRADIENT = (
    'nesterov', 'adam', 'conjugate_gradient', 'newton_conjugate_gradient',
    'bfgs', 'lbfgs', 'trust_ncg', 'trust_lanczos'
)
METHODS_THAT_REQUIRE_HESSIAN = (
    'newton_conjugate_gradient', 'trust_ncg', 'trust_lanczos'
)

METHODS_WITH_INNER_MINIMIZER = ('annealing',)

METHODS_THAT_REQUIRE_BOUNDS = ('direct','annealing')



def _initial_check(grad_norm, tolerance, params, current_cost) -> Dict:
    """
    Evaluates the initial optimization state before iterations begin.

    Checks whether:
    - the gradient norm is finite,
    - the current cost is finite,
    - the gradient norm is below the tolerance.

    Parameters
    ----------
        grad_norm : float
            The computed gradient norm.
        tolerance : float
            Gradient tolerance value.
        params : dict
            Parameters at the current iteration.
        current_cost : float
            The current cost value.

    Returns
    -------
        dict
            A dictionary containing the success flag, final parameters,
            final cost, iteration number, and termination reason.
    """

    if (
        not np.isfinite(grad_norm)
        or not np.isfinite(current_cost)
        or grad_norm < tolerance
    ):

        if not np.isfinite(grad_norm):
            termination_reason = (
                "Invalid value (NaN or +-Inf) encountered while "
                "evaluating the gradient."
            )

        elif not np.isfinite(current_cost):
            termination_reason = (
                "Objective function returned an invalid value (NaN or "
                "+-Inf) at the starting point."
            )

        elif grad_norm < tolerance:
            termination_reason = (
                "Gradient norm below tolerance at starting point. Point may "
                "already be a local minimum or results may be inaccurate. "
                "Consider changing the starting point."
            )

        return {
            'success': False,
            'final_params': params,
            'final_cost': current_cost,
            'iteration_number': 0,
            'termination_reason': termination_reason,
        }

    return {'success': True}



def _check_termination(
    cost_new: float,
    cost_old: float,
    grad_norm: float,
    gradient_tolerance: float,
    relative_tolerance: float,
) -> Tuple[bool, bool, str]:
    """
    Checks for convergence against standard termination criteria in
    optimization algorithms.

    This function evaluates whether an optimization process should terminate
    based on specified tolerances for gradient norm and relative cost change.

    The relative cost change is measured against the combined scale
    `max(|cost_old|, |cost_new|, 1.0)`, so the test behaves as a relative
    tolerance when the costs are >> 1 and as an absolute tolerance when the
    costs are << 1. The lower bound of 1 prevents over-tightening when the
    optimum is near zero.

    Parameters
    ----------
    cost_new : float
        The current cost value after an optimization step.
    cost_old : float
        The previous cost value before the optimization step. If +inf (can
        happen in first iteration when no previous cost has been recorded) the
        relative cost change test is skipped.
    grad_norm : float
        The norm of the gradient vector, measuring the steepness
        of the loss function at the current point.
    gradient_tolerance : float
        The threshold below which the gradient norm indicates
        convergence.
    relative_tolerance : float
        The threshold below which the relative change in cost indicates
        convergence, expressed as a fraction of the previous cost.


    Returns
    -------
    Tuple[bool, bool, str]
        terminate : bool
            True if any termination criterion is met.
        converged : bool
            True only when `terminate` is True and the run is considered
            successful. False when `terminate` is False.
        reason : str
            Explanation of why the loop should stop, or an empty string
            when no criterion is met.
    """

    if not np.isfinite(cost_new):
        return True, False, ("Objective function returned an invalid value "
                             "(NaN or +-Inf).")


    if not np.isfinite(grad_norm):
        return True, False, ("Invalid value (NaN or +-Inf) encountered "
                             "while evaluating the gradient.")
    if grad_norm < gradient_tolerance:
        return True, True, (f"Gradient norm ({grad_norm:.2e}) is below "
                            f"tolerance ({gradient_tolerance:.2e}).")


    if not np.isfinite(cost_old):
        return False, False, ""


    cost_delta = cost_old - cost_new
    cost_change = abs(cost_delta)
    cost_scale = max(abs(cost_old), abs(cost_new), 1.0)

    if cost_change / cost_scale >= relative_tolerance:
        return False, False, ""

    msg = (f"Relative cost change ({cost_change / cost_scale:.2e}) "
           f"is below tolerance ({relative_tolerance:.2e}).")


    GRADIENT_FAR_FROM_TOLERANCE_FACTOR = 5e2

    PRECISION_FLOOR_FACTOR = 1e2

    # When the cost change falls below
    # ~PRECISION_FLOOR_FACTOR * eps * cost_scale, the difference is at
    # the floating-point rounding level, no further progress is numerically
    # possible regardless of the gradient.
    precision_floor = PRECISION_FLOOR_FACTOR * FLOAT_EPSILON * cost_scale
    at_precision_floor = cost_change < precision_floor

    # cost_change has stopped meaningfully decreasing, but the gradient is
    # still far from zero. Either the algorithm is stuck at the floating-point
    # precision floor or progress has slowed because a saddle point or
    # ill-conditioning.
    if grad_norm > GRADIENT_FAR_FROM_TOLERANCE_FACTOR * gradient_tolerance:
        if at_precision_floor:
            msg += (f" Gradient norm ({grad_norm:.2e}) is greatly above "
                    f"tolerance ({gradient_tolerance:.2e}) but cost change "
                    f"({cost_change}) is at the floating-point precision floor "
                    f"(~{1e2 * FLOAT_EPSILON * cost_scale:.2e}). "
                    f"Likely precision loss or severe ill-conditioning.")

        else:
            msg += (f" Gradient norm ({grad_norm:.2e}) remains greatly above "
                    f"tolerance ({gradient_tolerance:.2e}). "
                    f"Possible slow convergence or saddle point.")

        return True, False, msg


    # The sign of cost_delta is only trustworthy above the precision
    # floor. Below it, the difference is dominated by rounding noise.
    cost_increased = (cost_delta < 0.0) and not at_precision_floor

    # Cost change is below tolerance and gradient is moderate.
    if cost_increased:
        msg += (f" However, the cost increased by {-cost_delta:.2e} "
                f"(from {cost_old:.6g} to {cost_new:.6g}): the optimizer "
                f"is no longer descending. Likely oscillation near a "
                f"minimum (for methods without a descent guarantee) or a "
                f"failed step.")
        return True, False, msg


    # Cost change is below tolerance and gradient is within a moderate
    # multiple of its tolerance, this is considered as a success notifing the
    # user
    if grad_norm >= gradient_tolerance:
        msg += (f" Gradient norm ({grad_norm:.2e}) is still above tolerance "
                f"({gradient_tolerance:.2e}) but within "
                f"{GRADIENT_FAR_FROM_TOLERANCE_FACTOR:g} times of it.")

    return True, True, msg



def _normalize_settings_dict(
    settings_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Recursively convert a settings dictionary into a form safe for
    representation and comparison.

    The conversion rules are:
        - callables (functions, methods, lambdas, callable objects) are
        replaced by the placeholder string '<callable>.
        - numpy.ndarray values are converted to nested Python lists via
        .tolist().
        - nested dictionaries are sanitised recursively.
        - tuples and lists are sanitised element-wise, preserving the
        container type.
        - every other value is passed through unchanged.

    Parameters
    ----------
    settings_dict : dict or None
        Dictionary to sanitise. None is returned unchanged.

    Returns
    -------
    dict or None
        A new dictionary with the conversions described above. The input
        is never modified in-place.
    """

    if settings_dict is None:
        return None

    sanitised_dict: Dict[str, Any] = {}

    for key, value in settings_dict.items():

        if callable(value):
            sanitised_dict[key] = '<callable>'

        elif isinstance(value, dict):
            sanitised_dict[key] = _normalize_settings_dict(value)

        elif isinstance(value, np.ndarray):
            sanitised_dict[key] = value.tolist()

        elif isinstance(value, (list, tuple)):
            container_type = type(value)
            sanitised_elements = [
                _normalize_settings_dict({'_': element})['_']
                for element in value
            ]
            sanitised_dict[key] = container_type(sanitised_elements)

        else:
            sanitised_dict[key] = value

    return sanitised_dict



class OptimizationResult:
    """
    Output class for the optimization public functions.

    Holds the optimized parameters and final cost together with the full
    diagnostic record of the run (function and gradient evaluation
    counts, iteration number, elapsed time, termination reason) and
    every configuration setting that produced the result.

    The instance behaves like a read-only numpy.ndarray of the final
    parameters: np.asarray(result), result[idx], len(result),
    iteration over result, and value in result are all supported
    and forward to the underlying final_params array, so the result
    can be used wherever an array is expected.

    The final parameters, initial parameters, and bounds are stored as
    read-only arrays, making the result effectively immutable from the
    user's perspective.

    Attributes
    ----------
    - Settings that produced the result:

    method, initial_params, bounds, max_iters, tolerances,
    method_params, gradient_point_number,
    analytical_gradient_provided, analytical_hessian_provided,
    hessian_vector_product_provided, constraint_count,
    constrained_params.

        See the ***minimizer*** docstring for their precise meaning.

    - Core output attributes :

    success : bool
        True if the run converged successfully.
    final_params : numpy.ndarray or None
        The optimized parameter vector. None when the run failed before
        producing one (e.g. _initial_check rejected the starting
        point).
    final_cost : float or None
        Objective value at final_params. None when final_params
        is None.
    termination_reason : str
        Explanation of why the loop stopped.
    func_evals_number, grad_evals_number, iteration_number : int
        grad_evals_number is 0 for global methods that never call the
        gradient.
    elapsed_time : float
        Elapsed time of the run, in seconds.

    - Convenience properties:

    is_scalar, is_array, shape, ndim, size, dtype, variable_number,
    is_constrained, is_bounded, is_global_method, is_local_method,
    total_evaluations, average_iteration_time,
    func_evals_per_iteration, grad_evals_per_iteration,
    evaluations_per_second

        Read-only properties derived from the output fields. The
        per-iteration and per-second properties return None when the
        denominator is zero.

    - Methods:

    as_array(), as_float(), to_list(), to_dict()
        Conversion helpers to plain Python and NumPy structures.
    summary(style='compact' | 'full')
        Multi-line summary which is also called when str(result).
    __eq__, allclose()
        Field-by-field exact and floating-point equality, respectively.

    Notes
    -----
    - Instances are unhashable: equality is value-based and the underlying
    parameter array, although read-only, is not a truly immutable Python value.

    - elapsed_time is intentionally excluded from __eq__: two
    identical runs on different hardware (or in different CPU states) would
    otherwise never compare equal.
    """

    # Prevents creation of a per-instance dict (__dict__), it also
    # disallows attributes not listed.
    __slots__ = (
        # Core output attributes
        '_success',
        '_final_params',
        '_final_cost',
        '_termination_reason',
        '_func_evals_number',
        '_grad_evals_number',
        '_elapsed_time',
        '_iteration_number',
        # Configuration that produced the result
        '_method',
        '_initial_params',
        '_bounds',
        '_max_iters',
        '_tolerances',
        '_method_params',
        '_gradient_point_number',
        '_analytical_gradient_provided',
        '_analytical_hessian_provided',
        '_hessian_vector_product_provided',
        '_constraint_count',
        '_constrained_params',
    )

    # Setting __hash__ to None explicitly makes instances of this class
    # unhashable, this is necessary because:
    #
    # As this class defines a custom __eq__ method (value-based equality,
    # comparing all relevant fields), Python automatically sets __hash__
    # to None as a safety measure. So it is clearer to explicitly set it
    # here.
    #
    # In addition, objects that compare equal must have the same hash
    # value. Since this class compares instances based on their values
    # (final_params, final_cost, ...), and these are mutable (numpy
    # arrays, even if read-only), the hash value could change if someone
    # modifies the outputted data. Allowing hashing would violate this and
    # cause bugs in dict/set operations.
    #
    # In practice, this means that the user cannot do:
    #   my_set = {result1, result2}
    #   my_dict = {result: value}
    #
    # both would raise TypeError: unhashable type
    __hash__ = None # type: ignore[assignment]



    def __init__(
        self,
        # Core output
        success: bool,
        final_params: Optional[Union[float, np.ndarray]],
        final_cost: Optional[float],
        termination_reason: str,
        func_evals_number: int,
        grad_evals_number: int,
        elapsed_time: float,
        iteration_number: int,
        # Configuration that produced the result
        method: str,
        initial_params: Optional[Union[float, np.ndarray]],
        bounds: Optional[np.ndarray],
        max_iters: int,
        tolerances: Dict[str, float],
        method_params: Optional[Dict[str, Any]],
        gradient_point_number: int,
        analytical_gradient_provided: bool,
        analytical_hessian_provided: bool,
        hessian_vector_product_provided: bool,
        constraint_count: int,
        constrained_params: Optional[Dict[str, Any]],
    ) -> None:

        # Make ndarray attributes read-only so the result is immutable
        # from the user's perspective.
        if final_params is None:
            self._final_params = None
        else:
            self._final_params = np.atleast_1d(
                np.asarray(final_params, dtype=float)
            ).copy()
            self._final_params.setflags(write=False)

        if initial_params is None:
            self._initial_params = None
        else:
            self._initial_params = np.atleast_1d(
                np.asarray(initial_params, dtype=float)
            ).copy()
            self._initial_params.setflags(write=False)

        if bounds is None:
            self._bounds = None
        else:
            self._bounds = np.asarray(bounds, dtype=float).copy()
            self._bounds.setflags(write=False)

        # Scalar / counter / status fields. final_cost may be None,
        # NaN, or +-inf, it is kept as-is.
        self._success = bool(success)
        self._final_cost = (
            None if final_cost is None else float(final_cost)
        )
        self._termination_reason = str(termination_reason)
        self._func_evals_number = int(func_evals_number)
        self._grad_evals_number = int(grad_evals_number)
        self._elapsed_time = float(elapsed_time)
        self._iteration_number = int(iteration_number)

        # Configuration. Dicts are normalized once at construction time
        # rather than at each call site so callers don't have to
        # remember the rule. Original dicts are not mutated.
        self._method = str(method)
        self._max_iters = int(max_iters)
        self._tolerances = (
            {} if tolerances is None
            else {key: float(value) for key, value in tolerances.items()}
        )
        self._method_params = _normalize_settings_dict(method_params) or {}
        self._gradient_point_number = int(gradient_point_number)
        self._analytical_gradient_provided = (
            bool(analytical_gradient_provided)
        )
        self._analytical_hessian_provided = (
            bool(analytical_hessian_provided)
        )
        self._hessian_vector_product_provided = (
            bool(hessian_vector_product_provided)
        )
        self._constraint_count = int(constraint_count)
        self._constrained_params = _normalize_settings_dict(constrained_params)



    # Information about the parameters that built the result


    @property
    def method(self) -> str:
        """Optimization algorithm requested from ***minimizer***."""
        return self._method
    @property
    def initial_params(self) -> Optional[np.ndarray]:
        """Read-only copy of the starting point. None when the method
        does not require one ('direct', 'annealing' without
        x0)."""
        return self._initial_params
    @property
    def bounds(self) -> Optional[np.ndarray]:
        """Read-only copy of the (lower, upper) bound array of shape
        (2, n). None when the run was unbounded."""
        return self._bounds
    @property
    def max_iters(self) -> int:
        """Effective iteration cap actually used by the inner loop, after
        'auto' has been resolved."""
        return self._max_iters
    @property
    def tolerances(self) -> Dict[str, float]:
        """Effective convergence tolerances actually used by the inner
        loop, after the user-provided dict has been merged with the
        defaults."""
        return dict(self._tolerances)
    @property
    def method_params(self) -> Dict[str, Any]:
        """Effective per-method parameter dict, after the user-provided
        dict has been merged with the defaults and normalized (callables
        replaced with '<callable>')."""
        return dict(self._method_params)
    @property
    def gradient_point_number(self) -> int:
        """Stencil width used by the numerical gradient when no
        analytical gradient was supplied."""
        return self._gradient_point_number
    @property
    def analytical_gradient_provided(self) -> bool:
        """True when the user provided an analytical gradient function;
        the numerical gradient was therefore not built."""
        return self._analytical_gradient_provided
    @property
    def analytical_hessian_provided(self) -> bool:
        """True when the user provided an analytical Hessian function."""
        return self._analytical_hessian_provided
    @property
    def hessian_vector_product_provided(self) -> bool:
        """True when the user provided an analytical Hessian-vector
        product. Takes precedence over the analytical Hessian when both
        are given."""
        return self._hessian_vector_product_provided
    @property
    def constraint_count(self) -> int:
        """Number of equality + inequality constraints supplied. Zero
        for unconstrained runs."""
        return self._constraint_count
    @property
    def constrained_params(self) -> Optional[Dict[str, Any]]:
        """Effective augmented-Lagrangian outer-loop parameter dict.
        None when the run was unconstrained and unbounded."""
        return (
            None if self._constrained_params is None
            else dict(self._constrained_params)
        )


    # Core output information


    @property
    def success(self) -> bool:
        """True if the optimizer reported successful convergence."""
        return self._success
    @property
    def final_params(self) -> Optional[np.ndarray]:
        """The optimized parameter vector (read-only). None when the
        optimizer failed before producing one."""
        return self._final_params
    @property
    def final_cost(self) -> Optional[float]:
        """Objective value at final_params. None when final_params is None."""
        return self._final_cost
    @property
    def termination_reason(self) -> str:
        """Readable explanation of why the loop stopped."""
        return self._termination_reason
    @property
    def func_evals_number(self) -> int:
        """Total number of objective function evaluations."""
        return self._func_evals_number
    @property
    def grad_evals_number(self) -> int:
        """Total number of gradient evaluations. Zero for global methods
        that never call the gradient."""
        return self._grad_evals_number
    @property
    def elapsed_time(self) -> float:
        """Elapsed time of the run, in seconds."""
        return self._elapsed_time
    @property
    def iteration_number(self) -> int:
        """Number of outer iterations completed."""
        return self._iteration_number


    # Convenience properties derived from the output fields


    @property
    def is_constrained(self) -> bool:
        """True when one or more user constraints were supplied."""
        return self._constraint_count > 0
    @property
    def is_bounded(self) -> bool:
        """True when finite bounds were supplied on at least one
        coordinate."""
        if self._bounds is None:
            return False
        # The minimizer normalises absent bounds to +-inf, so a fully
        # unbounded run reaches this branch with _bounds set but
        # every entry infinite.
        return bool(np.isfinite(self._bounds).any())
    @property
    def is_global_method(self) -> bool:
        """True when the chosen method performs a global search."""
        return self._method in GLOBAL_OPTIMIZATION_METHODS
    @property
    def is_local_method(self) -> bool:
        """True when the chosen method performs a local search."""
        return self._method in LOCAL_OPTIMIZATION_METHODS
    @property
    def total_evaluations(self) -> int:
        """Sum of func_evals_number and grad_evals_number."""
        return self._func_evals_number + self._grad_evals_number
    @property
    def average_iteration_time(self) -> Optional[float]:
        """Elapsed time per iteration, in seconds. None when no
        iteration completed."""
        if self._iteration_number == 0:
            return None
        return self._elapsed_time / self._iteration_number
    @property
    def func_evals_per_iteration(self) -> Optional[float]:
        """Average number of objective evaluations per iteration. None
        when no iteration completed."""
        if self._iteration_number == 0:
            return None
        return self._func_evals_number / self._iteration_number
    @property
    def grad_evals_per_iteration(self) -> Optional[float]:
        """Average number of gradient evaluations per iteration. None
        when no iteration completed."""
        if self._iteration_number == 0:
            return None
        return self._grad_evals_number / self._iteration_number
    @property
    def evaluations_per_second(self) -> Optional[float]:
        """Objective function + gradient function evaluations per second.
        None when the elapsed time fell below machine epsilon."""
        if self._elapsed_time <= FLOAT_EPSILON:
            return None
        return self.total_evaluations / self._elapsed_time


    # Shape / dtype information from final_params


    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of final_params. None when final_params is None."""
        if self._final_params is None:
            return None
        return self._final_params.shape
    @property
    def ndim(self) -> Optional[int]:
        """Number of dimensions of final_params. None when final_params is
        None."""
        if self._final_params is None:
            return None
        return self._final_params.ndim
    @property
    def size(self) -> int:
        """Total number of elements in final_params. Zero when final_params is
        None, so the user can safely sum or compare without a None check."""
        if self._final_params is None:
            return 0
        return self._final_params.size
    @property
    def variable_number(self) -> int:
        """Problem dimensionality. Alias of size."""
        return self.size
    @property
    def dtype(self) -> Optional[np.dtype]:
        """NumPy dtype of final_params. None when final_params is None."""
        if self._final_params is None:
            return None
        return self._final_params.dtype
    @property
    def is_scalar(self) -> bool:
        """True when the problem has exactly one variable. A 1-element
        final_params array still counts as scalar."""
        return self.size == 1
    @property
    def is_array(self) -> bool:
        """True when the problem has more than one variable."""
        return self.size > 1


    # Array-like behaviour methods


    # This class has been thought so it can be treated as an array of
    # final_params, the following special methods give this array-like
    # behaviour:
    def __array__(self, dtype=None, copy=False) -> np.ndarray:
        """Return final_params as a numpy.ndarray of `dtype`.

        Raises
        ------
        ValueError :
            When final_params is None and so cannot be converted.
        """
        if self._final_params is None:
            raise ValueError(
                "final_params is None; cannot convert to ndarray."
            )
        return np.asarray(self._final_params, dtype=dtype, copy=copy)

    def __getitem__(self, key):
        """Index into final_params."""
        if self._final_params is None:
            raise TypeError("final_params is None; cannot index into it.")
        return self._final_params[key]

    def __len__(self) -> int:
        """
        Length of final_params.

        Returns 0 when final_params is None so len(result).
        """
        if self._final_params is None:
            return 0
        return len(self._final_params)

    def __iter__(self):
        """Iterate over the components of final_params. Iteration
        over a None final_params yields nothing."""
        if self._final_params is None:
            return iter(())
        return iter(self._final_params)

    def __contains__(self, item) -> bool:
        """Element-wise membership test in final_params."""
        if self._final_params is None:
            return False
        return item in self._final_params


    # Conversions


    def as_array(self) -> np.ndarray:
        """
        Return a writable copy of final_params safe to mutate without affecting
        the result.

        Raises
        ------
        ValueError :
            When final_params is None.
        """
        if self._final_params is None:
            raise ValueError(
                "final_params is None, nothing to convert."
            )
        return np.array(self._final_params, dtype=float, copy=True)

    def as_float(self) -> float:
        """
        Return final_params as a Python float when the problem has exactly one
        variable.

        Raises
        ------
        ValueError :
            When final_params is None, or the problem has more than
            one variable. Use as_array() instead in that case.
        """
        if self._final_params is None:
            raise ValueError(
                "final_params is None, nothing to convert."
            )
        if self.size != 1:
            raise ValueError(
                f"as_float() is only defined for single-variable "
                f"problems (size == 1). Got size={self.size}. "
                f"Use as_array() instead."
            )
        return float(self._final_params.item())

    def to_list(self):
        """Return final_params converted to a Python list. None when
        final_params is None."""
        if self._final_params is None:
            return None
        return self._final_params.tolist()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary of Python objects.

        The dictionary contains the types: float, int, str, bool, list, tuple,
        dict, None. Callable values inside method_params and constrained_params
        are already represented as <callable>
        (see ***_normalize_settings_dict***).
        """
        final_params = (
            None if self._final_params is None
            else self._final_params.tolist()
        )
        initial_params = (
            None if self._initial_params is None
            else self._initial_params.tolist()
        )
        bounds = (
            None if self._bounds is None
            else self._bounds.tolist()
        )

        return {
            # Core output
            'success': self._success,
            'final_params': final_params,
            'final_cost': self._final_cost,
            'termination_reason': self._termination_reason,
            'func_evals_number': self._func_evals_number,
            'grad_evals_number': self._grad_evals_number,
            'elapsed_time': self._elapsed_time,
            'iteration_number': self._iteration_number,
            # Configuration
            'method': self._method,
            'initial_params': initial_params,
            'bounds': bounds,
            'max_iters': self._max_iters,
            'tolerances': dict(self._tolerances),
            'method_params': dict(self._method_params),
            'gradient_point_number': self._gradient_point_number,
            'analytical_gradient_provided': (
                self._analytical_gradient_provided
            ),
            'analytical_hessian_provided': (
                self._analytical_hessian_provided
            ),
            'hessian_vector_product_provided': (
                self._hessian_vector_product_provided
            ),
            'constraint_count': self._constraint_count,
            'constrained_params': (
                None if self._constrained_params is None
                else dict(self._constrained_params)
            ),
        }


    # Comparisons


    @staticmethod
    def _arrays_strictly_equal(
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
    ) -> bool:
        """
        Element-wise equality of two optional arrays, NaN's are compared as
        equal.
        """
        if (first is None) != (second is None):
            return False
        if first is None:
            return True
        return bool(np.array_equal(first, second, equal_nan=True))

    @staticmethod
    def _floats_strictly_equal(
        first: Optional[float],
        second: Optional[float],
    ) -> bool:
        """Equality for optional floats, NaN's are compared as
        equal."""
        if (first is None) != (second is None):
            return False
        if first is None:
            return True
        if np.isnan(first) and np.isnan(second):
            return True
        return first == second

    def __eq__(self, other) -> bool:
        """
        Strict, attribute-to-attribute equality.

        Two results are equal if every output and configuration attribute
        matches exactly, with the deliberate exception of elapsed_time
        (non-deterministic across machines / CPU states, see Notes on the
        class). NaN entries in final_params, initial_params, bounds, and
        final_cost compare equal to themselves.
        """
        if not isinstance(other, OptimizationResult):
            return NotImplemented

        # Output arrays / scalars (elapsed_time intentionally excluded).
        if not self._arrays_strictly_equal(
            self._final_params, other._final_params,
        ):
            return False
        if not self._arrays_strictly_equal(
            self._initial_params, other._initial_params,
        ):
            return False
        if not self._arrays_strictly_equal(
            self._bounds, other._bounds,
        ):
            return False
        if not self._floats_strictly_equal(
            self._final_cost, other._final_cost,
        ):
            return False

        return (
            self._success == other._success
            and self._termination_reason == other._termination_reason
            and self._func_evals_number == other._func_evals_number
            and self._grad_evals_number == other._grad_evals_number
            and self._iteration_number == other._iteration_number
            and self._method == other._method
            and self._max_iters == other._max_iters
            and self._tolerances == other._tolerances
            and self._method_params == other._method_params
            and (
                self._gradient_point_number
                == other._gradient_point_number
            )
            and (
                self._analytical_gradient_provided
                == other._analytical_gradient_provided
            )
            and (
                self._analytical_hessian_provided
                == other._analytical_hessian_provided
            )
            and (
                self._hessian_vector_product_provided
                == other._hessian_vector_product_provided
            )
            and self._constraint_count == other._constraint_count
            and self._constrained_params == other._constrained_params
        )

    def allclose(
        self,
        other: 'OptimizationResult',
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-12,
        equal_nan: bool = True,
    ) -> bool:
        """
        Floating-point equality.

        Compares final_params and final_cost. Any other configuration
        attribute (method, max_iters, tolerances, method params, ...) and
        tracker (evaluation counts, iteration number, elapsed time) is ignored,
        since two results obtained with different settings can still agree
        numerically on the optimum.

        Parameters
        ----------
        other : OptimizationResult
            Result to compare against.
        relative_tolerance, absolute_tolerance : float, optional
            Relative and absolute tolerances forwarded to numpy.allclose.
        equal_nan : bool, optional
            Whether NaN should compare equal to NaN.

        Raises
        ------
        TypeError
            When `other` is not an OptimizationResult instance.
        """
        if not isinstance(other, OptimizationResult):
            raise TypeError(
                "other must be an OptimizationResult instance."
            )

        # final_params comparison
        if (self._final_params is None) != (other._final_params is None):
            return False
        if self._final_params is not None and not np.allclose(
            self._final_params,
            other._final_params,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=equal_nan,
        ):
            return False

        # final_cost comparison
        if (self._final_cost is None) != (other._final_cost is None):
            return False
        if self._final_cost is not None and not np.isclose(
            self._final_cost,
            other._final_cost,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=equal_nan,
        ):
            return False

        return True


    # Representations


    def __repr__(self) -> str:
        """Compact one-line developer-oriented representation."""
        if self._final_params is None:
            shape_descriptor = "no_params"
        elif self._final_params.size == 1:
            shape_descriptor = "scalar"
        else:
            shape_descriptor = f"shape={self._final_params.shape}"

        cost_descriptor = (
            "None" if self._final_cost is None
            else f"{self._final_cost:.6g}"
        )

        return (
            f"OptimizationResult("
            f"method='{self._method}', "
            f"success={self._success}, "
            f"final_cost={cost_descriptor}, "
            f"{shape_descriptor})"
        )

    def __str__(self) -> str:
        """String representation is summary('compact')."""
        return self.summary('compact')


    @staticmethod
    def _format_block(
        label: str, string_value: str, label_width: int,
    ) -> str:
        """
        Render a single summary entry.

        Single-line values are emitted as "  label : value";
        multi-line values are emitted as "  label :" followed by
        the value indented by four spaces.
        """
        if '\n' in string_value:
            indented = '\n'.join(
                f"    {line}" for line in string_value.splitlines()
            )
            return f"  {label:<{label_width}} :\n{indented}"
        return f"  {label:<{label_width}} : {string_value}"

    @staticmethod
    def _render_array(
        array: Optional[np.ndarray], precision: int,
    ) -> str:
        """Render an ndarray to a numpy-style multi-line string. Returns
        'None' when the array is None."""
        if array is None:
            return 'None'
        with np.printoptions(precision=precision, suppress=False):
            return repr(array)

    @staticmethod
    def _format_elapsed_time(seconds: float) -> str:
        """Pick a unit for the elapsed time."""
        if seconds < 1e-3:
            return f"{seconds * 1e6:.3f} us"
        if seconds < 1.0:
            return f"{seconds * 1e3:.3f} ms"
        return f"{seconds:.3f} s"

    @staticmethod
    def _format_optional_float(
        value: Optional[float], precision: int = 6,
    ) -> str:
        """Format an optional float, handling None / NaN /
        +-inf."""
        if value is None:
            return 'None'
        if not np.isfinite(value):
            return f"{value}"
        return f"{value:.{precision}g}"

    def summary(
        self, style: Literal['compact', 'full'] = 'compact',
    ) -> str:
        """
        Return a multi-line summary of the result.

        Parameters
        ----------
        style : 'compact', 'full', optional
            - 'compact' (default): a header, the method used, the
              success flag, the final_cost, the full
              final_params vector, the termination_reason, and
              the basic diagnostic counters (iteration_number,
              func_evals_number, grad_evals_number,
              elapsed_time).
            - 'full': everything in 'compact' plus the full
              configuration (variable_number,
              is_global_method, is_bounded, is_constrained,
              the three *_provided flags, max_iters,
              tolerances, gradient_point_number,
              method_params, constraint_count,
              constrained_params, initial_params, bounds)
              and the derived throughput stats
              (func_evals_per_iteration,
              grad_evals_per_iteration, evaluations_per_second).

        Returns
        -------
        str
            Multi-line summary suitable for print(result.summary()).

        Raises
        ------
        ValueError
            When `style` is not 'compact' or 'full'.
        """
        if style not in ('compact', 'full'):
            raise ValueError(
                f"style must be 'compact' or 'full'. Got {style!r}."
            )

        # Label-column width chosen to fit the longest label in each
        # style.
        label_width = 22 if style == 'compact' else 34

        header = f"OptimizationResult ({style})"
        lines = [header, "=" * len(header)]


        # Configuration
        if style == 'full':
            lines.append(self._format_block(
                'method', self._method, label_width,
            ))
            lines.append(self._format_block(
                'variable_number', f"{self.variable_number}", label_width,
            ))
            lines.append(self._format_block(
                'is_global_method', f"{self.is_global_method}", label_width,
            ))
            lines.append(self._format_block(
                'is_bounded', f"{self.is_bounded}", label_width,
            ))
            lines.append(self._format_block(
                'is_constrained', f"{self.is_constrained}", label_width,
            ))
            lines.append(self._format_block(
                'analytical_gradient_provided',
                f"{self._analytical_gradient_provided}",
                label_width,
            ))
            lines.append(self._format_block(
                'analytical_hessian_provided',
                f"{self._analytical_hessian_provided}",
                label_width,
            ))
            lines.append(self._format_block(
                'hessian_vector_product_provided',
                f"{self._hessian_vector_product_provided}",
                label_width,
            ))
            lines.append(self._format_block(
                'max_iters', f"{self._max_iters}", label_width,
            ))
            lines.append(self._format_block(
                'tolerances', f"{self._tolerances}", label_width,
            ))
            lines.append(self._format_block(
                'gradient_point_number',
                f"{self._gradient_point_number}",
                label_width,
            ))
            lines.append(self._format_block(
                'method_params',
                f"{self._method_params}",
                label_width,
            ))
            lines.append(self._format_block(
                'constraint_count', f"{self._constraint_count}", label_width,
            ))
            lines.append(self._format_block(
                'constrained_params',
                f"{self._constrained_params}",
                label_width,
            ))
            lines.append(self._format_block(
                'initial_params',
                self._render_array(self._initial_params, precision=6),
                label_width,
            ))
            lines.append(self._format_block(
                'bounds',
                self._render_array(self._bounds, precision=6),
                label_width,
            ))

        else:
            # Compact: only the chosen method as configuration context.
            lines.append(self._format_block(
                'method', self._method, label_width,
            ))

        # Core output.
        lines.append(self._format_block(
            'success', f"{self._success}", label_width,
        ))
        lines.append(self._format_block(
            'final_cost',
            self._format_optional_float(self._final_cost),
            label_width,
        ))
        lines.append(self._format_block(
            'final_params',
            self._render_array(self._final_params, precision=8),
            label_width,
        ))
        lines.append(self._format_block(
            'termination_reason', self._termination_reason, label_width,
        ))
        lines.append(self._format_block(
            'iteration_number', f"{self._iteration_number}", label_width,
        ))
        lines.append(self._format_block(
            'func_evals_number', f"{self._func_evals_number}", label_width,
        ))
        lines.append(self._format_block(
            'grad_evals_number', f"{self._grad_evals_number}", label_width,
        ))
        lines.append(self._format_block(
            'elapsed_time',
            self._format_elapsed_time(self._elapsed_time),
            label_width,
        ))

        # Derived throughput stats
        if style == 'full':
            lines.append(self._format_block(
                'total_evaluations',
                f"{self.total_evaluations}",
                label_width,
            ))
            lines.append(self._format_block(
                'average_iteration_time',
                (
                    'None' if self.average_iteration_time is None
                    else self._format_elapsed_time(
                        self.average_iteration_time
                    )
                ),
                label_width,
            ))
            lines.append(self._format_block(
                'func_evals_per_iteration',
                self._format_optional_float(
                    self.func_evals_per_iteration, precision=4,
                ),
                label_width,
            ))
            lines.append(self._format_block(
                'grad_evals_per_iteration',
                self._format_optional_float(
                    self.grad_evals_per_iteration, precision=4,
                ),
                label_width,
            ))
            lines.append(self._format_block(
                'evaluations_per_second',
                self._format_optional_float(
                    self.evaluations_per_second, precision=4,
                ),
                label_width,
            ))

        return '\n'.join(lines)
