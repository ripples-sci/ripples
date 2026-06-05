"""
Utilities for _numerical_differentiation.py.

Any supporting code is here, while _numerical_differentiation.py keeps its
focus on the numerical strategies themselves.

Contains
--------
FLOAT_EPSILON
    Machine epsilon for IEEE-754 double precision (~2.22e-16), the
    smallest positive float e such that 1.0 + e != 1.0.

_solve_gauss_jordan_fraction
    Solves the square linear system matrix * x = rhs using Gauss-Jordan
    elimination with exact Fraction arithmetic. Used to obtain
    rational central-difference coefficients with no floating-point
    contamination.

_effective_point_number
    Returns the effective point number, given a derivative_order and a
    point_number. Enforces the parity rule that the effective stencil
    count must have opposite parity to the derivative order; the input
    point_number is promoted by one when needed.

_validate_nth_numerical_derivative_parameters
    Validates and normalises the parameters supplied to
    ***nth_numerical_derivative*** at construction time. Centralising
    this work here keeps the public function's body focused on what
    happens once the inputs are known to be valid.

_resolve_step_size
    Resolves an already validated step_size into the concrete form
    expected by ***_evaluate_component_dispatcher***. Translates the
    'auto', scalar, and per-coordinate-tuple inputs into a single
    per-coordinate float array.

DifferentiationResult
    Output class for the numerical-differentiation public functions.
    Carries the computed derivative, an optional truncation-error
    estimate, the evaluation points, and every configuration setting
    that produced the result. Behaves like a read-only numpy.ndarray of
    the derivative for indexing, iteration, and arithmetic, and exposes
    conversion helpers (as_array, as_float, to_list, to_dict),
    equality (==, allclose), and a summary.

_PRECOMPUTED_CENTRAL_DIFFERENCES_COEFFICIENTS
    Lookup table of central-difference stencil coefficients for the most
    common (derivative_order, point_number, return_fractions)
    combinations, in both float and exact Fraction form. Makes the
    construction of the most-used stencils O(1) rather than requiring a
    fresh linear-system solve.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from fractions import Fraction
from functools import lru_cache
from warnings import warn
from numbers import Real
from typing import Literal, Union, Dict, Optional, Tuple, Any



FLOAT_EPSILON = float(np.finfo(float).eps)



def _solve_gauss_jordan_fraction(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction]:
    """
    Solve the square linear system `matrix` * x = `rhs` using
    Gauss-Jordan elimination with exact `Fraction` arithmetic.

    Parameters
    ----------
    matrix : list[list[Fraction]]
        Square coefficient matrix (dim_num x dim_num), modified in-place
        during elimination.
    rhs : list[Fraction]
        Right-hand side vector of length dim_num, modified in-place during
        elimination.

    Returns
    -------
    list[Fraction]
        Exact solution vector x of length dim_num.

    Raises
    ------
    ValueError
        When `matrix` is singular (a zero pivot is encountered with no
        non-zero row available to swap).

    Mathematical Description
    ------------------------
    Letting n be the dimension number of the problem (`dim_num`).

    The goal is to find a vector x = [x_1, ..., x_n] satisfying the linear
    system:

    A * x = b

    where A is an (n x n) matrix (`matrix`) and b is a vector of length n
    (`rhs`).

    The algorithm operates on the augmented matrix [A | b]

        [ a_{0,0}    a_{0,1}    ...  a_{0,n-1}    | b_0     ]
        [ a_{1,0}    a_{1,1}    ...  a_{1,n-1}    | b_1     ]
        [ ...        ...        ...  ...          | ...     ]
        [ a_{n-1,0}  a_{n-1,1}  ...  a_{n-1,n-1}  | b_{n-1} ]

    reducing it column by column to the identity on the left-hand side, at
    which point the right-hand side holds the solution directly.

    ---

    ### Column elimination (outer loop, col = 0, 1, ..., n-1)

    At each step col, the algorithm zeroes every entry in column col except
    the diagonal one. There are three sub-steps:

    - Step 1, when a pivot row is located as the first row at index >= col
    whose entry in column col is non-zero. If no such row exists, the
    matrix is singular and the algorithms exits. The located pivot row is
    then swapped with row col so that the pivot element lands on the
    diagonal:

    matrix[col] <-> matrix[pivot_row]
    rhs[col] <-> rhs[pivot_row]

    After the swap the pivot element is:

    pivot = matrix[col][col]

    - Step 2, an inner loop (row = 0, 1, ..., n-1, row != col) is performed
    so the entry in column col is eliminated by subtracting a suitable
    multiple of the pivot row. Defining:

    factor = matrix[row][col] / pivot

    the update applied to each element j in row is:

    matrix[row][j] <- matrix[row][j] - factor * matrix[col][j]

    and the corresponding right-hand side entry is updated as:

    rhs[row] <- rhs[row] - factor * rhs[col]

    After this step, every entry in column col outside row col is exactly
    zero.

    - Step 3, lastly the pivot row is scaled so that the diagonal entry
    matrix[col][col] becomes exactly 1:

    matrix[col][j] <- matrix[col][j] / pivot        (for all j)
    rhs[col] <- rhs[col] / pivot



    After all n columns have been processed the left-hand side equals the
    identity matrix I_n, and the augmented system reads:

        [ 1   0   ...  0   | x_0     ]
        [ 0   1   ...  0   | x_1     ]
        [ ... ... ...  ... | ...     ]
        [ 0   0   ...  1   | x_{n-1} ]

    so the solution x is read directly from `rhs`, which is returned.

    ---

    - Because every arithmetic operation is performed on `Fraction`
    objects, all intermediate values are kept as exact rationals
    throughout. No rounding or cancellation error can accumulate regardless
    of the size or condition number of the matrix.

    - The algorithm applies full (not just forward) elimination at every
    column, so the result is obtained without a separate back-substitution
    pass.

    - Partial pivoting (swapping rows to place the first non-zero entry on
    the diagonal) is necessary for correctness, not merely for numerical
    stability: without it a zero pivot would force a division by zero even
    when the system is non-singular.

    References
    ----------
    [1] Wikipedia contributors.
        **Gaussian_elimination.**
        https://en.wikipedia.org/wiki/Gaussian_elimination
    """

    dim_num = len(rhs)

    for col in range(dim_num):
        # Locate a non-zero pivot in the current column at or below the
        # diagonal, and swap it into the diagonal position
        pivot_row = next(
            (row for row in range(col, dim_num) if matrix[row][col] != 0),
            None
        )

        if pivot_row is None:
            raise ValueError(
                f"Matrix is singular: no non-zero pivot found in column "
                f"{col}."
            )

        # Swap current iteration row (col) with the located pivot_row
        matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
        rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        pivot = matrix[col][col]

        # Eliminate every other row in the current column
        for row in range(dim_num):
            if row == col or matrix[row][col] == 0:
                continue

            factor = matrix[row][col] / pivot

            matrix[row] = [
                matrix[row][j] - factor * matrix[col][j]
                for j in range(dim_num)
            ]
            rhs[row] = rhs[row] - factor * rhs[col]

        # Normalise the pivot row so the diagonal entry becomes exactly 1
        rhs[col] = rhs[col] / pivot
        matrix[col] = [matrix[col][j] / pivot for j in range(dim_num)]

    return rhs



@lru_cache(maxsize=8192)
def _effective_point_number(
    derivative_order: int,
    point_number: int
) -> int:
    """
    Computes the effective point number given a `derivative_order` and a
    `point_number`.

    See ***_central_difference_coefficients*** for more information.

    Parameters
    ----------
    derivative_order : int
        The order of the derivative to be calculated.
    point_number : int
        The specified number of stencil points to be used in the
        central-difference formula.

    Returns
    -------
    int
        The effective point number:
        - `point_number` if `derivative_order` and `point_number` have the
        opposite parity, that is, one is odd and the other is even.
        - `point_number` + 1 otherwise.
    """

    return (
        point_number if (derivative_order % 2) != (point_number % 2)
                     else point_number + 1
    )



def _validate_nth_numerical_derivative_parameters(
    derivative_order: int,
    step_size: Union[Literal['auto', 'complex'], float, Tuple[float, ...]],
    point_number: Union[Literal['auto'], int],
    single_component: Optional[Union[int, Tuple[int, ...]]],
    richardson_extrapolation: bool,
    maximum_richardson_equations: Union[Literal['auto'], int],
) -> Tuple[Union[Literal['auto', 'complex'], float, Tuple[float, ...]], int,
           Optional[Tuple[int, ...]], bool, int,
           Literal['complex_step', 'richardson_extrapolation',
                   'central_difference']
    ]:
    """
    Validates and normalises every parameter inputted to
    ***nth_numerical_derivative*** at construction time.

    Two kinds of issues are reported:

    - Errors that raise ValueError:
        - `derivative_order` not a positive integer.
        - `richardson_extrapolation` not a bool.
        - `step_size` neither 'auto', 'complex', a positive finite number,
          nor an array-like of positive finite numbers.
        - `step_size` = 'complex' combined with `derivative_order` != 1.
        - `point_number` neither 'auto' nor an integer satisfying
          `point_number` >= max(2, `derivative_order` + 1).
        - `single_component` not None, not an integer (only allowed when
          `derivative_order` = 1), and not convertible to a tuple of
          exactly `derivative_order` integers.
        - `maximum_richardson_equations` neither 'auto' nor an integer
          >= 2 when `richardson_extrapolation` = True.

    - Errors that are normalised silently but signaled to the user with a
    RuntimeWarning:
        - `richardson_extrapolation` = True with `step_size` = 'complex'
          (extrapolation is disabled, complex-step is already exact to
          machine precision).
        - `point_number` != 'auto' with `step_size` = 'complex' (the
          complex-step method uses no stencil).
        - `maximum_richardson_equations` != 'auto' with
          `richardson_extrapolation` = False (the value is ignored).
        - `maximum_richardson_equations` > 50 (capped at 50 to prevent
          wasted computation).

    Parameters
    ----------
    derivative_order : int
        Order n of the derivative requested by the user. Must be a
        positive integer, booleans are rejected explicitly.
    step_size : 'auto', 'complex', float, or tuple of floats
        Finite-difference step specification. See
        ***nth_numerical_derivative***.
    point_number : 'auto' or int
        Base stencil point count per coordinate.
    single_component : int, tuple of int, or None
        Tensor component selector. A bare integer is accepted only when
        `derivative_order` = 1; for higher orders a tuple of length
        `derivative_order` is required.
    richardson_extrapolation : bool
        Whether Richardson extrapolation was requested.
    maximum_richardson_equations : 'auto' or int
        Cap on the number of Richardson equations.

    Returns
    -------
    tuple
        Tuple with the normalised values:

        - `_step_size` : the unmodified string 'auto' / 'complex',
          a single positive float (cast from any positive real
          scalar), or a tuple of positive floats (cast from any
          array-like).

        - `_point_number` : a positive integer satisfying `_point_number`
          >= max(2, `derivative_order` + 1). When the input was
          'auto', this is `derivative_order` + 1 when
          `richardson_extrapolation` = True (the best base for
          Richardson, see ***_evaluate_component_richardson***), and
          `derivative_order` + 7 otherwise (the best balance between
          truncation error and stencil-coefficient roundoff).

        - `_single_component` : either None or a tuple of `derivative_order`
        integers. A int input is converted to tuple as (int,).

        - `_richardson_extrapolation` : `richardson_extrapolation`
          unless overridden to False by `step_size` = 'complex'.

        - `_maximum_richardson_equations` : an integer >= 2. Set to 5
          when 'auto' was passed alongside
          `richardson_extrapolation` = True, and to 2 (the minimum
          accepted value, ignored downstream) when
          `richardson_extrapolation` = False.

        - `differentiation_method` : one of 'complex_step',
          'richardson_extrapolation', or 'central_difference',
          identifying the path the dispatcher will take.

    Raises
    ------
    ValueError :
        On any of the hard errors enumerated above.

    Notes
    -----
    - Warnings are emitted with stacklevel=3 so they surface at the
    user's call location: the user calls ***nth_numerical_derivative***,
    which calls this validator, which emits the warning.

    - The function never mutates its inputs. Array-like step sizes are
    converted through np.asarray(..., dtype=float).flatten(), which
    is a copy, before being repacked as a tuple; the caller's original
    container is left untouched.

    - Booleans are rejected explicitly wherever an integer is expected
    (`derivative_order`, `point_number`, `single_component`,
    `maximum_richardson_equations`).
    """

    if (
        not isinstance(derivative_order, int)
        or isinstance(derivative_order, bool)
        or derivative_order < 1
    ):
        # repr() (!r) is used so booleans and non-standard numeric types show
        # their exact representation in the error message.
        raise ValueError(
            f"derivative_order must be a positive integer. "
            f"Got {derivative_order!r}."
        )



    if not isinstance(richardson_extrapolation, bool):
        raise ValueError(
            f"richardson_extrapolation must be a bool. "
            f"Got {richardson_extrapolation!r}."
        )



    if isinstance(step_size, str):
        if step_size == 'complex':
            if derivative_order != 1:
                raise ValueError(
                    f"step_size='complex' is only available for "
                    f"derivative_order=1. For higher-order derivatives use "
                    f"step_size='auto' or a numeric step. Got "
                    f"derivative_order={derivative_order}."
                )

            if richardson_extrapolation:
                warn("richardson_extrapolation=True is ignored because "
                     "step_size was set to 'complex'.",
                     RuntimeWarning, stacklevel=3)

                richardson_extrapolation = False

            if not isinstance(point_number, str) or point_number != 'auto':
                warn("point_number is ignored when step_size='complex', "
                     "the complex-step method uses no stencil.",
                     RuntimeWarning, stacklevel=3)


        elif step_size != 'auto':
            raise ValueError(
                f"Provided step_size is a string but is not 'auto' or "
                f"'complex'. Got '{step_size!r}'."
            )

        _step_size = step_size

    elif isinstance(step_size, bool):
        raise ValueError(
            f"step_size must be 'auto', 'complex', a positive finite "
            f"number, or a tuple of positive finite numbers. "
            f"Got {step_size!r}."
        )

    elif isinstance(step_size, Real):
        if (not isinstance(step_size, (int, float)) or
            step_size <= 0 or
            not np.isfinite(step_size)
        ):
            raise ValueError(
                f"step_size must be a strictly positive finite number. "
                f"Got {step_size}."
            )

        _step_size = float(step_size)

    else:
        # tuple of floats
        try:
            # .flatten() creates a copy, the user's step is not modified
            _step_size = np.asarray(step_size, dtype=float).flatten()

        except (TypeError, ValueError) as conversion_error:
            raise ValueError(
                f"step_size must be 'auto', 'complex', a positive finite "
                f"number, or a tuple of positive finite numbers. "
                f"Got {step_size!r}."
            ) from conversion_error

        if np.any(~np.isfinite(_step_size)) or np.any(_step_size <= 0):
            raise ValueError("Provided step size contains non-finite or "
                             "negative values.")

        _step_size = tuple(_step_size)



    if isinstance(point_number, str):
        if point_number != 'auto':
            raise ValueError(
                f"Provided point_number is a string but it is not 'auto'. Got "
                f"{point_number!r} "
            )

        if richardson_extrapolation:
            # When extrapolation is performed, minimal point number must be
            # used in order to get an accuracy of 2, the least achievable so
            # truncation error dominates roundoff. Richardson then provides its
            # full theoretical accuracy improvement (O(h^2) -> O(h^4) per
            # equation).
            _point_number = derivative_order + 1

        else:
            _point_number = derivative_order + 7

    elif isinstance(point_number, int) and not isinstance(point_number, bool):
        minimum_allowed_point_number = max(2, derivative_order + 1)
        if point_number < minimum_allowed_point_number:
            raise ValueError(
                f"point_number must be >= max(2, derivative_order+1) = "
                f"{minimum_allowed_point_number}. Got {point_number!r}."
            )

        _point_number = point_number

    else:
        raise ValueError(
            f"point_number must be an integer >= max(2, derivative_order + 1) "
            f"or 'auto'. Got {point_number!r}."
        )



    if single_component is None:
        _single_component = None

    elif isinstance(single_component, bool):
        raise ValueError(
            f"single_component must be None, an integer (only when "
            f"derivative_order=1), or a tuple of integers. "
            f"Got {single_component!r}."
        )

    elif isinstance(single_component, int):
        if derivative_order != 1:
            raise ValueError(
                f"single_component may only be given as an int when "
                f"derivative_order=1. For derivative_order={derivative_order} "
                f"single_component must be a tuple of {derivative_order} "
                f"integers."
            )

        _single_component = (single_component,)

    else:
        try:
            _single_component = tuple(
                int(coordinate_index) for coordinate_index in
                np.asarray(single_component, dtype=int).flatten()
            )

        except (TypeError, ValueError) as conversion_error:
            raise ValueError(
                f"single_component could not be converted to a tuple of "
                f"integers. Got {single_component!r}."
            ) from conversion_error

        if len(_single_component) != derivative_order:
            raise ValueError(
                f"single_component tuple must have length derivative_order="
                f"{derivative_order}. Got length "
                f"{len(_single_component)}."
            )



    if richardson_extrapolation:
        if maximum_richardson_equations == 'auto':

            _maximum_richardson_equations = 5

        elif (
            isinstance(maximum_richardson_equations, int)
            and not isinstance(maximum_richardson_equations, bool)
        ):
            if maximum_richardson_equations < 2:
                raise ValueError(
                    f"maximum_richardson_equations must be >= 2 when "
                    f"richardson_extrapolation=True. Got "
                    f"{maximum_richardson_equations}."
                )

            elif maximum_richardson_equations > 50:
                warn(
                    f"maximum_richardson_equations="
                    f"{maximum_richardson_equations} is excessive; capping "
                    f"to 50 to prevent wasted computation.",
                    RuntimeWarning, stacklevel=3,
                )

                _maximum_richardson_equations = 50

            else:
                _maximum_richardson_equations = maximum_richardson_equations

        else:
            raise ValueError(
                f"maximum_richardson_equations must be an integer >= 2 or "
                f"'auto' when richardson_extrapolation=True. Got "
                f"{maximum_richardson_equations!r}."
            )

        _richardson_extrapolation = True

    else:
        if maximum_richardson_equations != 'auto':
            warn(
                "maximum_richardson_equations is ignored when "
                "richardson_extrapolation=False.",
                RuntimeWarning, stacklevel=3,
            )

        # Minimum allowed value. The dispatcher ignores this argument
        # when richardson_extrapolation=False.
        _maximum_richardson_equations = 2

        _richardson_extrapolation = False



    if isinstance(step_size, str) and step_size == 'complex':
        differentiation_method = 'complex_step'

    elif richardson_extrapolation:
        differentiation_method = 'richardson_extrapolation'

    else:
        differentiation_method = 'central_difference'


    return (
        _step_size,
        _point_number,
        _single_component,
        _richardson_extrapolation,
        _maximum_richardson_equations,
        differentiation_method,
    )



@lru_cache(maxsize=8192)
def _resolve_step_size(
    step_size: Union[Literal['auto', 'complex'], float, Tuple[float, ...]],
    variable_number: int,
) -> Union[Literal['auto', 'complex'], np.ndarray]:
    """
    Resolves an already validated `step_size` into the concrete form
    expected by ***_evaluate_component_dispatcher***.

    'auto' and 'complex' are passed unchanged, float and tuples are
    casted into a read-only array of length `variable_number`.

    Parameters
    ----------
    step_size : 'auto', 'complex', float, or tuple of floats
        Step-size specification previously normalised by
        ***_validate_nth_numerical_derivative_parameters***. A
        float is therefore guaranteed to be positive and finite, and
        a tuple to contain only positive finite floats.
    variable_number : int
        Dimensionality D of the evaluation point, that is, the number
        of scalar coordinates the function-to-differentiate consumes.

    Returns
    -------
    'auto', 'complex', or np.ndarray
        - The input string itself when it is 'auto' or
          'complex': the dispatcher branches on the string value
          directly.

        - A read-only numpy.ndarray of shape (`variable_number`,) and
          float dtype otherwise. A scalar input is broadcast to that
          shape via np.full; a tuple input is copied into a fresh
          array via np.array. The write flag is cleared in both
          cases (see Notes).

    Raises
    ------
    ValueError :
        When `step_size` is a tuple whose length differs from
        `variable_number`. This is the one check that cannot be
        performed by
        ***_validate_nth_numerical_derivative_parameters***: that
        function runs at construction time, before any evaluation point
        (and hence any D) is known.

    Notes
    -----
    - Marking the returned array read-only is essential: a single cached array
    is reused across every evaluation of the configured derivative, so a
    mutation through one call would silently corrupt every subsequent one.
    """

    if isinstance(step_size, str):
        # 'auto' or 'complex', directly given to
        # _evaluate_component_dispatcher
        return step_size

    elif isinstance(step_size, (int, float)):
        step_size_array = np.full(
            variable_number, float(step_size), dtype=float
        )

        step_size_array.setflags(write=False)

        return step_size_array

    # np.ndarray
    else:
        if len(step_size) != variable_number:
            raise ValueError(
                f"Provided step size array-like must have the same "
                f"dimensions as the evaluated point: {variable_number}, "
                f"currently it has {len(step_size)}."
            )

        step_size_array = np.array(step_size, dtype=float)

        step_size_array.setflags(write=False)

        return step_size_array



class DifferentiationResult:
    """
    Output class for the numerical-differentiation public functions.

    Holds the computed derivative together with the optional truncation error
    estimate and every configuration setting that produced the result.

    The instance behaves like a read-only numpy.ndarray of the derivative:
    np.asarray(result), result[idx], len(result), iteration over
    result, and value in result are all supported and forward to the
    underlying derivative, so the result can be used wherever an array is
    expected.

    The derivative, error estimate, and evaluation points are stored as
    read-only arrays, making the result effectively immutable from the user's
    perspective.

    Attributes
    ----------
    - Settings that produced the result:

    derivative_order, differentiation_method, step_size, point_number,
    single_component, richardson_extrapolation, maximum_richardson_equations.

        See the nth_numerical_derivative docstring for their precise
        meaning.

    - Core output attributes:

    derivative : float or numpy.ndarray
        The computed derivative.
    error_estimate : float, numpy.ndarray or None
        Romberg-style upper bound on the truncation error, with the same shape
        as derivative. None when no Richardson extrapolation was applied or
        when the complex-step method was used.
    evaluation_points : numpy.ndarray
        Read-only copy of the points at which the derivative was evaluated.

    - Convenience properties:

    has_error_estimate, relative_error_estimate, is_full_tensor, is_scalar,
    is_array, shape, ndim, size, dtype

        Read-only properties derived from the output fields.

    - Methods:

    as_array(), as_float(), to_list(), to_dict()
        Conversion helpers to plain Python and NumPy structures.
    summary(style='compact' | 'full')
        Multi-line summary which is also called when str(result).
    __eq__, allclose()
        Field-by-field exact and floating-point equality, respectively.

    Notes
    -----
    Instances are unhashable: equality is value-based and the underlying
    derivative array, although flagged read-only, is not a truly immutable
    Python value.
    """

    # Prevents creation of a per-instance dict (__dict__), it also disallows
    # attributes not listed.
    __slots__ = (
        '_derivative',
        '_error_estimate',
        '_derivative_order',
        '_differentiation_method',
        '_step_size',
        '_point_number',
        '_evaluation_points',
        '_single_component',
        '_richardson_extrapolation',
        '_maximum_richardson_equations',
    )

    # Setting __hash__ to None explicitly makes instances of this class
    # unhashable, this is necessary because:
    #
    # As this class defines a custom __eq__ method (value-based equality,
    # comparing all fields), Python automatically sets __hash__ to None as a
    # safety measure. So it is clearer to explicitly set it here.
    #
    # In addition, objects that compare equal must have the same hash value.
    # Since this class compares instances based on their values (the
    # derivative, error estimate, ...), and these are mutable (numpy arrays,
    # even if flagged read-only), the hash value could change if someone
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
        derivative: Union[float, np.ndarray],
        error_estimate: Optional[Union[float, np.ndarray]],
        derivative_order: int,
        differentiation_method: Literal[
            'complex_step', 'richardson_extrapolation', 'central_difference'
        ],
        step_size: Union[Literal['auto', 'complex'], float, Tuple[float, ...]],
        point_number: Optional[int],
        evaluation_points: np.ndarray,
        single_component: Optional[Tuple[int, ...]],
        richardson_extrapolation: bool,
        maximum_richardson_equations: Optional[int],
    ) -> None:

        # Make ndarrays attributes read-only so the result is immutable from
        # the user's perspective:
        self._derivative = np.asarray(derivative, dtype=float)
        self._derivative.setflags(write=False)
        if error_estimate is None:
            self._error_estimate = None
        else:
            self._error_estimate = np.asarray(error_estimate, dtype=float)
            self._error_estimate.setflags(write=False)

        self._derivative_order = derivative_order
        self._differentiation_method = differentiation_method
        self._step_size = step_size
        self._point_number = point_number

        # As evaluation_points is at the user-provided input array, it must be
        # copied to prevent modifiying the user's array.
        if isinstance(evaluation_points, np.ndarray):
            evaluation_points = evaluation_points.copy()
            evaluation_points.setflags(write=False)
        self._evaluation_points = evaluation_points
        self._single_component = single_component
        self._richardson_extrapolation = richardson_extrapolation
        self._maximum_richardson_equations = maximum_richardson_equations



    # Information about the parameters that built the derivative function


    @property
    def derivative_order(self) -> int:
        """Order (n) of the computed derivative."""
        return self._derivative_order
    @property
    def differentiation_method(self) -> str:
        """Numerical strategy used to compute the derivative."""
        return self._differentiation_method
    @property
    def step_size(self) -> Union[
        Literal['auto', 'complex'], float, Tuple[float, ...]
    ]:
        """Step-size originally requested by the user."""
        return self._step_size
    @property
    def point_number(self) -> Optional[int]:
        """Base stencil point count. None for complex-step."""
        return self._point_number
    @property
    def evaluation_points(self) -> np.ndarray:
        """Read-only copy of the points at which the derivative was computed."""
        return self._evaluation_points
    @property
    def single_component(self) -> Optional[Tuple[int, ...]]:
        """Tensor component requested, or None for the full tensor."""
        return self._single_component
    @property
    def richardson_extrapolation(self) -> bool:
        """True when Richardson extrapolation was requested, regardless of
        whether the procedure ultimately produced an estimate."""
        return self._richardson_extrapolation
    @property
    def maximum_richardson_equations(self) -> Optional[int]:
        """Cap on the number of Richardson equations. None when unused."""
        return self._maximum_richardson_equations


    @property
    def is_full_tensor(self) -> bool:
        """True when the full derivative tensor was computed."""
        return self._single_component is None


    # Core output information


    @property
    def derivative(self) -> Union[float, np.ndarray]:
        """The computed derivative (read-only when an ndarray)."""
        if self._derivative.size == 1:
            return np.squeeze(self._derivative)
        return self._derivative
    @property
    def error_estimate(self) -> Optional[Union[float, np.ndarray]]:
        """Truncation-error upper bound when Richardson extrapolation
        produced one. None when Richardson extrapolation was not applied
        or when the complex-step method was used."""
        return self._error_estimate
    @property
    def has_error_estimate(self) -> bool:
        """True if a non-None error_estimate was produced."""
        return self._error_estimate is not None
    @property
    def relative_error_estimate(self) -> Optional[Union[float, np.ndarray]]:
        """
        Element-wise relative error:

            |error_estimate| / max(|derivative|, machine_epsilon)

        The return type matches derivative: a float for a scalar
        derivative, a read-only numpy.ndarray (same shape as derivative)
        otherwise.

        Returns None when error_estimate is None.
        """
        if self._error_estimate is None:
            return None

        denominator = np.maximum(np.abs(self._derivative), FLOAT_EPSILON)
        relative_error = np.abs(self._error_estimate) / denominator

        if self._derivative.size == 1:
            return float(relative_error)

        relative_error.setflags(write=False)
        return relative_error


    # Shape / dtype information


    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the derivative."""
        return self._derivative.shape
    @property
    def ndim(self) -> int:
        """Number of dimensions of the derivative."""
        return self._derivative.ndim
    @property
    def size(self) -> int:
        """Total number of elements in the underlying derivative."""
        return self._derivative.size
    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the underlying derivative."""
        return self._derivative.dtype
    @property
    def is_scalar(self) -> bool:
        """True when the derivative is a 0-D value (Python float or 0-D
        ndarray)."""
        return self._derivative.ndim == 0
    @property
    def is_array(self) -> bool:
        """True when the derivative has at least one dimension."""
        return not self.is_scalar


    # Array-like behaviour methods


    # This class has been thought so it can be treated as an array, the
    # following special methods give this array-like behaviour:
    def __array__(self, dtype=None, copy=False) -> np.ndarray:
        """Return the derivative as a numpy.ndarray of `dtype`."""
        return np.asarray(self._derivative, dtype=dtype, copy=copy)
    def __getitem__(self, key):
        """Index into the derivative."""
        if self.is_scalar:
            return np.squeeze(self._derivative)
        return self._derivative[key]
    def __len__(self) -> int:
        """
        Length of the first axis of the derivative.

        Scalar derivatives are treated as length 1 so that len(result)
        and iteration over a scalar result both behave consistently
        (yielding the scalar exactly once).
        """
        return len(np.atleast_1d(self._derivative))
    def __iter__(self):
        """Iterate along the first axis of the derivative."""
        return iter(np.atleast_1d(self._derivative))
    def __contains__(self, item) -> bool:
        """Element-wise membership test in the derivative."""
        return bool(np.any(self._derivative == item))


    # Conversions


    def as_array(self) -> np.ndarray:
        """
        Return the derivative as a new writable numpy.ndarray.

        The returned array is a copy and does not share memory with the
        result's read-only buffer, so it is safe to mutate.
        """
        return np.array(self._derivative, copy=True)

    def as_float(self) -> float:
        """
        Return the derivative as a Python float.

        Convenience helper for the common case in which the caller knows
        the derivative is scalar (for example, a univariate first
        derivative at a single point, or any call made with
        single_component set).

        Raises
        ------
        ValueError
            When the derivative is not a scalar (size > 1). For
            non-scalar derivatives, use as_array() instead.
        """
        if self._derivative.size != 1:
            raise ValueError(
                f"as_float() requires a scalar derivative; got an array "
                f"of shape {self._derivative.shape}. Use as_array() instead."
            )
        return float(self._derivative)

    def to_list(self):
        """
        Return the derivative converted to a Python list / scalar.
        """
        return self._derivative.tolist()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary of Python objects.

        The dictionary contains only the following types float, int, str, bool,
        list, tuple, None.
        """
        derivative = (
            self._derivative.tolist()
            if isinstance(self._derivative, np.ndarray)
            else self._derivative
        )

        error_estimate = (
            self._error_estimate.tolist()
            if isinstance(self._error_estimate, np.ndarray)
            else self._error_estimate
        )

        evaluation_points = self._evaluation_points.tolist()

        return {
            'derivative': derivative,
            'error_estimate': error_estimate,
            'derivative_order': self._derivative_order,
            'differentiation_method': self._differentiation_method,
            'step_size': self._step_size,
            'point_number': self._point_number,
            'evaluation_points': evaluation_points,
            'single_component': self._single_component,
            'richardson_extrapolation': self._richardson_extrapolation,
            'maximum_richardson_equations': (
                self._maximum_richardson_equations
            ),
        }


    # Comparisons


    def __eq__(self, other) -> bool:
        """
        Strict, attribute-to-attribute equality.

        Two results are equal if every attribute matches exactly. np.nan
        entries in derivative, error_estimate, and evaluation_points compare
        equal to themselves.
        """
        if not isinstance(other, DifferentiationResult):
            return NotImplemented

        if not np.array_equal(
            self._derivative,
            other._derivative,
            equal_nan=True,
        ):
            return False

        # Handle None explicitly, np.array_equal on Python None is not
        # well defined across NumPy versions.
        self_error = self._error_estimate
        other_error = other._error_estimate
        if (self_error is None) != (other_error is None):
            return False
        if self_error is not None and not np.array_equal(
            self_error,
            other_error,
            equal_nan=True,
        ):
            return False

        if not np.array_equal(
            self._evaluation_points,
            other._evaluation_points,
            equal_nan=True,
        ):
            return False

        return (
            self._derivative_order == other._derivative_order
            and self._differentiation_method == other._differentiation_method
            and self._step_size == other._step_size
            and self._point_number == other._point_number
            and self._single_component == other._single_component
            and (
                self._richardson_extrapolation
                == other._richardson_extrapolation
            )
            and (
                self._maximum_richardson_equations
                == other._maximum_richardson_equations
            )
        )

    def allclose(
        self,
        other: 'DifferentiationResult',
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-8,
        equal_nan: bool = True,
    ) -> bool:
        """
        Floating-point equality.

        Compares derivative_order, differentiation_method,
        derivative, and error_estimate. Any other configuration
        attribute (step size, point number, ...) is ignored, since two
        results obtained with different settings can still agree
        numerically.

        Parameters
        ----------
        other : DifferentiationResult
            Result to compare against.
        relative_tolerance, absolute_tolerance : float, optional
            Relative and absolute tolerances forwarded to numpy.allclose.
        equal_nan : bool, optional
            Whether NaN should compare equal to NaN.

        Raises
        ------
        TypeError
            When `other` is not a DifferentiationResult instance.
        """
        if not isinstance(other, DifferentiationResult):
            raise TypeError(
                "other must be a DifferentiationResult instance."
            )

        if self._derivative_order != other._derivative_order:
            return False
        if self._differentiation_method != other._differentiation_method:
            return False

        if not np.allclose(
            self._derivative,
            other._derivative,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=equal_nan,
        ):
            return False

        self_error = self._error_estimate
        other_error = other._error_estimate
        if (self_error is None) != (other_error is None):
            return False

        if self_error is not None and not np.allclose(
            self_error,
            other_error,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=equal_nan,
        ):
            return False

        return True


    # Representations


    def __repr__(self) -> str:
        """Compact one-line developer-oriented representation."""
        derivative_array = self._derivative

        shape_descriptor = (
            "scalar" if derivative_array.ndim == 0
            else f"shape={derivative_array.shape}"
        )

        error_descriptor = (
            "" if self._error_estimate is None
            else ", error_estimate=available"
        )

        return (
            f"DifferentiationResult("
            f"order={self._derivative_order}, "
            f"differentiation_method='{self._differentiation_method}', "
            f"{shape_descriptor}"
            f"{error_descriptor})"
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
    def _render_array(array: np.ndarray, precision: int) -> str:
        """Render an ndarray to a numpy-style multi-line string."""
        with np.printoptions(precision=precision, suppress=False):
            return repr(array)

    def summary(
        self, style: Literal['compact', 'full'] = 'compact',
    ) -> str:
        """
        Return a multi-line summary of the result.

        Parameters
        ----------
        style : 'compact', 'full', optional
            - 'compact' (default): a header, the differentiation
              differentiation_method used, the complete derivative
              value, and the error information (error_estimate and
              relative_error_estimate, with max / mean statistics for
              arrays).
            - 'full': everything in 'compact' plus the full
              configuration (derivative_order,
              shape / ndim / size / dtype, step_size,
              point_number, single_component,
              richardson_extrapolation,
              maximum_richardson_equations) and the
              evaluation_points at which the derivative was
              computed.

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

        # Label-column width chosen to fit the longest label in each style.
        label_width = 23 if style == 'compact' else 28

        header = f"DifferentiationResult ({style})"
        lines = [header, "=" * len(header)]

        derivative_array = np.asarray(self._derivative)
        derivative_is_scalar = derivative_array.ndim == 0
        has_error = self._error_estimate is not None

        if style == 'full':
            # Configuration attributes
            lines.append(self._format_block(
                'derivative_order',
                f"{self._derivative_order}",
                label_width,
            ))
            lines.append(self._format_block(
                'differentiation_method', f"{self._differentiation_method}",
                label_width,
            ))
            lines.append(self._format_block(
                'shape', f"{derivative_array.shape}", label_width,
            ))
            lines.append(self._format_block(
                'ndim', f"{derivative_array.ndim}", label_width,
            ))
            lines.append(self._format_block(
                'size', f"{derivative_array.size}", label_width,
            ))
            lines.append(self._format_block(
                'dtype', f"{derivative_array.dtype}", label_width,
            ))
            lines.append(self._format_block(
                'step_size', f"{self._step_size!r}", label_width,
            ))
            lines.append(self._format_block(
                'point_number',
                f"{self._point_number}",
                label_width,
            ))
            lines.append(self._format_block(
                'single_component',
                f"{self._single_component}",
                label_width,
            ))
            lines.append(self._format_block(
                'richardson_extrapolation',
                f"{self._richardson_extrapolation}",
                label_width,
            ))
            lines.append(self._format_block(
                'maximum_richardson_equations',
                f"{self._maximum_richardson_equations}",
                label_width,
            ))


            # Evaluation points
            lines.append(self._format_block(
                'evaluation_points.shape',
                f"{self._evaluation_points.shape}",
                label_width,
            ))
            lines.append(self._format_block(
                'evaluation_points',
                self._render_array(
                    self._evaluation_points, precision=6,
                ),
                label_width,
            ))

        else:
            # Compact: header + differentiation_method only as configuration
            lines.append(self._format_block(
                'differentiation_method', f"{self._differentiation_method}",
                label_width,
            ))


        # Derivative is always shown in full form
        if derivative_is_scalar:
            derivative_str = f"{float(derivative_array):.10g}"
        else:
            derivative_str = self._render_array(
                derivative_array, precision=8,
            )
        lines.append(self._format_block(
            'derivative', derivative_str, label_width,
        ))


        # Error information
        if not has_error:
            # The following is an explanation of why there is no error
            # estimate, choosing the message that is actually applicable.
            if self._differentiation_method == 'complex_step':
                no_error_message = (
                    'not computed (complex_step provides no estimate)'
                )
            elif not self._richardson_extrapolation:
                no_error_message = (
                    'not computed (richardson_extrapolation=False)'
                )
            else:
                no_error_message = 'not computed'
            lines.append(self._format_block(
                'error_estimate', no_error_message, label_width,
            ))

        else:
            error_array = np.asarray(self._error_estimate, dtype=float)
            finite_mask = np.isfinite(error_array)

            if not finite_mask.any():
                lines.append(self._format_block(
                    'error_estimate', 'all non-finite', label_width,
                ))

            elif error_array.ndim == 0:
                # Scalar error
                lines.append(self._format_block(
                    'error_estimate',
                    f"{float(error_array):.6g}",
                    label_width,
                ))
                relative_value = self.relative_error_estimate
                lines.append(self._format_block(
                    'relative_error_estimate',
                    f"{float(relative_value):.6g}",
                    label_width,
                ))

            else:
                # Array error: full array plus max / mean.
                lines.append(self._format_block(
                    'error_estimate',
                    self._render_array(error_array, precision=4),
                    label_width,
                ))
                absolute_errors = np.abs(error_array)
                lines.append(self._format_block(
                    'max  |error_estimate|',
                    f"{np.nanmax(absolute_errors):.6g}",
                    label_width,
                ))
                lines.append(self._format_block(
                    'mean |error_estimate|',
                    f"{np.nanmean(absolute_errors):.6g}",
                    label_width,
                ))

                relative_array = np.asarray(
                    self.relative_error_estimate, dtype=float,
                )
                lines.append(self._format_block(
                    'relative_error_estimate',
                    self._render_array(relative_array, precision=4),
                    label_width,
                ))
                absolute_relative_errors = np.abs(relative_array)
                lines.append(self._format_block(
                    'max  |relative_error|',
                    f"{np.nanmax(absolute_relative_errors):.6g}",
                    label_width,
                ))
                lines.append(self._format_block(
                    'mean |relative_error|',
                    f"{np.nanmean(absolute_relative_errors):.6g}",
                    label_width,
                ))

        return "\n".join(lines)



# The following Dictionary is a pre-computed ouput of the
# _central_difference_coefficients function in _numerical_differentiation.py
# for all the combinations where
# derivative_order <= 20 and
# derivative_order + 1 <= point_number <= derivative_order + 11,
# in order to make O(1) the computation of the most common central-difference
# stencils
_PRECOMPUTED_CENTRAL_DIFFERENCES_COEFFICIENTS: Dict[tuple[int, int, bool],
    Union[Tuple[float, ...], Tuple[Fraction, ...]]] = {
 (1, 2, False): (0.5,),
 (1, 2, True): (Fraction(1, 2),),
 (1, 4, False): (0.6666666666666666, -0.08333333333333333),
 (1, 4, True): (Fraction(2, 3), Fraction(-1, 12)),
 (1, 6, False): (0.75, -0.15, 0.016666666666666666),
 (1, 6, True): (Fraction(3, 4), Fraction(-3, 20), Fraction(1, 60)),
 (1, 8, False): (0.8, -0.2, 0.0380952380952381, -0.0035714285714285713),
 (1, 8, True): (Fraction(4, 5), Fraction(-1, 5), Fraction(4, 105),
                Fraction(-1, 280)),
 (1, 10, False): (0.8333333333333334, -0.23809523809523808, 0.05952380952380952,
                  -0.00992063492063492, 0.0007936507936507937),
 (1, 10, True): (Fraction(5, 6), Fraction(-5, 21), Fraction(5, 84),
                 Fraction(-5, 504), Fraction(1, 1260)),
 (1, 12, False): (0.8571428571428571, -0.26785714285714285, 0.07936507936507936,
                  -0.017857142857142856, 0.0025974025974025974,
                  -0.00018037518037518038),
 (1, 12, True): (Fraction(6, 7), Fraction(-15, 56), Fraction(5, 63),
                 Fraction(-1, 56), Fraction(1, 385), Fraction(-1, 5544)),
 (2, 3, False): (-2.0, 1.0),
 (2, 3, True): (Fraction(-2, 1), Fraction(1, 1)),
 (2, 5, False): (-2.5, 1.3333333333333333, -0.08333333333333333),
 (2, 5, True): (Fraction(-5, 2), Fraction(4, 3), Fraction(-1, 12)),
 (2, 7, False): (-2.7222222222222223, 1.5, -0.15, 0.011111111111111112),
 (2, 7, True): (Fraction(-49, 18), Fraction(3, 2), Fraction(-3, 20),
                Fraction(1, 90)),
 (2, 9, False): (-2.8472222222222223, 1.6, -0.2, 0.025396825396825397,
                 -0.0017857142857142857),
 (2, 9, True): (Fraction(-205, 72), Fraction(8, 5), Fraction(-1, 5),
                Fraction(8, 315), Fraction(-1, 560)),
 (2, 11, False): (-2.9272222222222224, 1.6666666666666667, -0.23809523809523808,
                  0.03968253968253968, -0.00496031746031746,
                  0.00031746031746031746),
 (2, 11, True): (Fraction(-5269, 1800), Fraction(5, 3), Fraction(-5, 21),
                 Fraction(5, 126), Fraction(-5, 1008), Fraction(1, 3150)),
 (2, 13, False): (-2.9827777777777778, 1.7142857142857142, -0.26785714285714285,
                  0.05291005291005291, -0.008928571428571428,
                  0.001038961038961039, -6.012506012506013e-05),
 (2, 13, True): (Fraction(-5369, 1800), Fraction(12, 7), Fraction(-15, 56),
                 Fraction(10, 189), Fraction(-1, 112), Fraction(2, 1925),
                 Fraction(-1, 16632)),
 (3, 4, False): (-1.0, 0.5),
 (3, 4, True): (Fraction(-1, 1), Fraction(1, 2)),
 (3, 6, False): (-1.625, 1.0, -0.125),
 (3, 6, True): (Fraction(-13, 8), Fraction(1, 1), Fraction(-1, 8)),
 (3, 8, False): (-2.033333333333333, 1.4083333333333334, -0.3,
                 0.029166666666666667),
 (3, 8, True): (Fraction(-61, 30), Fraction(169, 120), Fraction(-3, 10),
                Fraction(7, 240)),
 (3, 10, False): (-2.3180555555555555, 1.7337301587301588, -0.4830357142857143,
                  0.0833994708994709, -0.006779100529100529),
 (3, 10, True): (Fraction(-1669, 720), Fraction(4369, 2520),
                 Fraction(-541, 1120), Fraction(1261, 15120),
                 Fraction(-41, 6048)),
 (3, 12, False): (-2.527142857142857, 1.9950892857142857, -0.6572751322751322,
                  0.1530952380952381, -0.02261904761904762,
                  0.0015839947089947089),
 (3, 12, True): (Fraction(-1769, 700), Fraction(4469, 2240),
                 Fraction(-4969, 7560), Fraction(643, 4200), Fraction(-19, 840),
                 Fraction(479, 302400)),
 (3, 14, False): (-2.6869345238095237, 2.2081448412698412, -0.8170667989417989,
                  0.23056998556998556, -0.04682990620490621,
                  0.006053691678691678, -0.0003724747474747475),
 (3, 14, True): (Fraction(-90281, 33600), Fraction(222581, 100800),
                 Fraction(-247081, 302400), Fraction(31957, 138600),
                 Fraction(-2077, 44352), Fraction(20137, 3326400),
                 Fraction(-59, 158400)),
 (4, 5, False): (6.0, -4.0, 1.0),
 (4, 5, True): (Fraction(6, 1), Fraction(-4, 1), Fraction(1, 1)),
 (4, 7, False): (9.333333333333334, -6.5, 2.0, -0.16666666666666666),
 (4, 7, True): (Fraction(28, 3), Fraction(-13, 2), Fraction(2, 1),
                Fraction(-1, 6)),
 (4, 9, False): (11.375, -8.133333333333333, 2.816666666666667, -0.4,
                 0.029166666666666667),
 (4, 9, True): (Fraction(91, 8), Fraction(-122, 15), Fraction(169, 60),
                Fraction(-2, 5), Fraction(7, 240)),
 (4, 11, False): (12.741666666666667, -9.272222222222222, 3.4674603174603176,
                  -0.6440476190476191, 0.0833994708994709,
                  -0.005423280423280424),
 (4, 11, True): (Fraction(1529, 120), Fraction(-1669, 180),
                 Fraction(4369, 1260), Fraction(-541, 840),
                 Fraction(1261, 15120), Fraction(-41, 7560)),
 (4, 13, False): (13.717407407407407, -10.108571428571429, 3.9901785714285714,
                  -0.8763668430335096, 0.1530952380952381,
                  -0.018095238095238095, 0.0010559964726631393),
 (4, 13, True): (Fraction(37037, 2700), Fraction(-1769, 175),
                 Fraction(4469, 1120), Fraction(-4969, 5670),
                 Fraction(643, 4200), Fraction(-19, 1050),
                 Fraction(479, 453600)),
 (4, 15, False): (14.447883597883598, -10.747738095238095, 4.4162896825396825,
                  -1.0894223985890652, 0.23056998556998556,
                  -0.03746392496392496, 0.0040357944524611195,
                  -0.00021284271284271284),
 (4, 15, True): (Fraction(54613, 3780), Fraction(-90281, 8400),
                 Fraction(222581, 50400), Fraction(-247081, 226800),
                 Fraction(31957, 138600), Fraction(-2077, 55440),
                 Fraction(20137, 4989600), Fraction(-59, 277200)),
 (5, 6, False): (2.5, -2.0, 0.5),
 (5, 6, True): (Fraction(5, 2), Fraction(-2, 1), Fraction(1, 2)),
 (5, 8, False): (4.833333333333333, -4.333333333333333, 1.5,
                 -0.16666666666666666),
 (5, 8, True): (Fraction(29, 6), Fraction(-13, 3), Fraction(3, 2),
                Fraction(-1, 6)),
 (5, 10, False): (6.729166666666667, -6.5, 2.71875, -0.5277777777777778,
                  0.04513888888888889),
 (5, 10, True): (Fraction(323, 48), Fraction(-13, 2), Fraction(87, 32),
                 Fraction(-19, 36), Fraction(13, 288)),
 (5, 12, False): (8.246031746031745, -8.39608134920635, 3.982804232804233,
                  -1.033399470899471, 0.16005291005291006,
                  -0.011491402116402117),
 (5, 12, True): (Fraction(1039, 126), Fraction(-33853, 4032),
                 Fraction(3011, 756), Fraction(-3125, 3024), Fraction(121, 756),
                 Fraction(-139, 12096)),
 (5, 14, False): (9.470800264550265, -10.029106040564374, 5.207572751322751,
                  -1.6272266313932982, 0.34562389770723106,
                  -0.045750661375661375, 0.002854938271604938),
 (5, 14, True): (Fraction(286397, 30240), Fraction(-1819681, 181440),
                 Fraction(157477, 30240), Fraction(-73811, 45360),
                 Fraction(6271, 18144), Fraction(-2767, 60480),
                 Fraction(37, 12960)),
 (5, 16, False): (10.474125514403292, -11.433761390358612, 6.356836219336219,
                  -2.2657063358452247, 0.5911930148041259, -0.10889700577200577,
                  0.012677702955480733, -0.000701626048848271),
 (5, 16, True): (Fraction(203617, 19440), Fraction(-1244725, 108864),
                 Fraction(352423, 55440), Fraction(-6782981, 2993760),
                 Fraction(176989, 299376), Fraction(-24149, 221760),
                 Fraction(2711, 213840), Fraction(-4201, 5987520)),
 (6, 7, False): (-20.0, 15.0, -6.0, 1.0),
 (6, 7, True): (Fraction(-20, 1), Fraction(15, 1), Fraction(-6, 1),
                Fraction(1, 1)),
 (6, 9, False): (-37.5, 29.0, -13.0, 3.0, -0.25),
 (6, 9, True): (Fraction(-75, 2), Fraction(29, 1), Fraction(-13, 1),
                Fraction(3, 1), Fraction(-1, 4)),
 (6, 11, False): (-51.15, 40.375, -19.5, 5.4375, -0.7916666666666666,
                  0.05416666666666667),
 (6, 11, True): (Fraction(-1023, 20), Fraction(323, 8), Fraction(-39, 2),
                 Fraction(87, 16), Fraction(-19, 24), Fraction(13, 240)),
 (6, 13, False): (-61.768055555555556, 49.476190476190474, -25.188244047619047,
                  7.965608465608466, -1.5500992063492063, 0.19206349206349208,
                  -0.011491402116402117),
 (6, 13, True): (Fraction(-44473, 720), Fraction(1039, 21),
                 Fraction(-33853, 1344), Fraction(3011, 378),
                 Fraction(-3125, 2016), Fraction(121, 630),
                 Fraction(-139, 12096)),
 (6, 15, False): (-70.16646825396825, 56.824801587301586, -30.087318121693123,
                  10.415145502645503, -2.440839947089947, 0.41474867724867726,
                  -0.045750661375661375, 0.002447089947089947),
 (6, 15, True): (Fraction(-353639, 5040), Fraction(286397, 5040),
                 Fraction(-1819681, 60480), Fraction(157477, 15120),
                 Fraction(-73811, 30240), Fraction(6271, 15120),
                 Fraction(-2767, 60480), Fraction(37, 15120)),
 (6, 17, False): (-76.93891369047618, 62.84475308641975, -34.301284171075835,
                  12.713672438672438, -3.398559503767837, 0.7094316177649511,
                  -0.10889700577200577, 0.0108666025332692,
                  -0.0005262195366362033),
 (6, 17, True): (Fraction(-1034059, 13440), Fraction(203617, 3240),
                 Fraction(-1244725, 36288), Fraction(352423, 27720),
                 Fraction(-6782981, 1995840), Fraction(176989, 249480),
                 Fraction(-24149, 221760), Fraction(2711, 249480),
                 Fraction(-4201, 7983360)),
 (7, 8, False): (-7.0, 7.0, -3.0, 0.5),
 (7, 8, True): (Fraction(-7, 1), Fraction(7, 1), Fraction(-3, 1),
                Fraction(1, 2)),
 (7, 10, False): (-15.75, 17.0, -8.625, 2.1666666666666665,
                  -0.20833333333333334),
 (7, 10, True): (Fraction(-63, 4), Fraction(17, 1), Fraction(-69, 8),
                 Fraction(13, 6), Fraction(-5, 24)),
 (7, 12, False): (-24.275, 27.65625, -15.729166666666666, 5.008333333333334,
                  -0.8541666666666666, 0.06458333333333334),
 (7, 12, True): (Fraction(-971, 40), Fraction(885, 32), Fraction(-755, 48),
                 Fraction(601, 120), Fraction(-41, 48), Fraction(31, 480)),
 (7, 14, False): (-31.996006944444446, 37.95092592592592, -23.45017361111111,
                  8.751851851851852, -2.0240162037037037, 0.28055555555555556,
                  -0.017997685185185186),
 (7, 14, True): (Fraction(-184297, 5760), Fraction(40987, 1080),
                 Fraction(-135073, 5760), Fraction(2363, 270),
                 Fraction(-6995, 3456), Fraction(101, 360),
                 Fraction(-311, 17280)),
 (7, 16, False): (-38.817746913580244, 47.50136188271605, -31.264166666666668,
                  13.092959104938272, -3.693672839506173, 0.7098958333333333,
                  -0.08478395061728396, 0.004770447530864198),
 (7, 16, True): (Fraction(-251539, 6480), Fraction(12312353, 259200),
                 Fraction(-37517, 1200), Fraction(678739, 51840),
                 Fraction(-4787, 1296), Fraction(1363, 1920),
                 Fraction(-2747, 32400), Fraction(2473, 518400)),
 (7, 18, False): (-44.801884645061726, 56.205562219416386, -38.88034196127946,
                  17.779836209315377, -5.786028689674523, 1.3794497053872055,
                  -0.23124886012906845, 0.02446320847362514,
                  -0.001230797558922559),
 (7, 18, True): (Fraction(-23225297, 518400), Fraction(160253299, 2851200),
                 Fraction(-36951877, 950400), Fraction(50693869, 2851200),
                 Fraction(-659885, 114048), Fraction(1311029, 950400),
                 Fraction(-2637347, 11404800), Fraction(139499, 5702400),
                 Fraction(-4679, 3801600)),
 (8, 9, False): (70.0, -56.0, 28.0, -8.0, 1.0),
 (8, 9, True): (Fraction(70, 1), Fraction(-56, 1), Fraction(28, 1),
                Fraction(-8, 1), Fraction(1, 1)),
 (8, 11, False): (154.0, -126.0, 68.0, -23.0, 4.333333333333333,
                  -0.3333333333333333),
 (8, 11, True): (Fraction(154, 1), Fraction(-126, 1), Fraction(68, 1),
                 Fraction(-23, 1), Fraction(13, 3), Fraction(-1, 3)),
 (8, 13, False): (233.56666666666666, -194.2, 110.625, -41.94444444444444,
                  10.016666666666667, -1.3666666666666667,
                  0.08611111111111111),
 (8, 13, True): (Fraction(7007, 30), Fraction(-971, 5), Fraction(885, 8),
                 Fraction(-755, 18), Fraction(601, 60), Fraction(-41, 30),
                 Fraction(31, 360)),
 (8, 15, False): (304.1587301587302, -255.96805555555557, 151.8037037037037,
                  -62.533796296296295, 17.503703703703703, -3.238425925925926,
                  0.37407407407407406, -0.02056878306878307),
 (8, 15, True): (Fraction(19162, 63), Fraction(-184297, 720),
                 Fraction(40987, 270), Fraction(-135073, 2160),
                 Fraction(2363, 135), Fraction(-1399, 432), Fraction(101, 270),
                 Fraction(-311, 15120)),
 (8, 17, False): (365.5543898809524, -310.54197530864195, 190.0054475308642,
                  -83.3711111111111, 26.185918209876544, -5.909876543209877,
                  0.9465277777777777, -0.09689594356261023,
                  0.004770447530864198),
 (8, 17, True): (Fraction(4913051, 13440), Fraction(-251539, 810),
                 Fraction(12312353, 64800), Fraction(-37517, 450),
                 Fraction(678739, 25920), Fraction(-4787, 810),
                 Fraction(1363, 1440), Fraction(-2747, 28350),
                 Fraction(2473, 518400)),
 (8, 19, False): (418.74672527189887, -358.4150771604938, 224.82224887766554,
                  -103.68091189674523, 35.559672418630754, -9.257645903479236,
                  1.8392662738496073, -0.26428441157607824, 0.02446320847362514,
                  -0.00109404227459783),
 (8, 19, True): (Fraction(91172887, 217728), Fraction(-23225297, 64800),
                 Fraction(160253299, 712800), Fraction(-36951877, 356400),
                 Fraction(50693869, 1425600), Fraction(-131977, 14256),
                 Fraction(1311029, 712800), Fraction(-2637347, 9979200),
                 Fraction(139499, 5702400), Fraction(-4679, 4276800)),
 (9, 10, False): (21.0, -24.0, 13.5, -4.0, 0.5),
 (9, 10, True): (Fraction(21, 1), Fraction(-24, 1), Fraction(27, 2),
                 Fraction(-4, 1), Fraction(1, 2)),
 (9, 12, False): (54.0, -65.25, 41.0, -15.0, 3.0, -0.25),
 (9, 12, True): (Fraction(54, 1), Fraction(-261, 4), Fraction(41, 1),
                 Fraction(-15, 1), Fraction(3, 1), Fraction(-1, 4)),
 (9, 14, False): (91.5375, -115.3, 78.5375, -33.2, 8.6875, -1.3, 0.0875),
 (9, 14, True): (Fraction(7323, 80), Fraction(-1153, 10), Fraction(6283, 80),
                 Fraction(-166, 5), Fraction(139, 16), Fraction(-13, 10),
                 Fraction(7, 80)),
 (9, 16, False): (129.55734126984126, -168.52777777777777, 122.0875,
                  -57.394444444444446, 17.993055555555557, -3.692857142857143,
                  0.4597222222222222, -0.026587301587301587),
 (9, 16, True): (Fraction(652969, 5040), Fraction(-6067, 36),
                 Fraction(9767, 80), Fraction(-10331, 180), Fraction(2591, 144),
                 Fraction(-517, 140), Fraction(331, 720), Fraction(-67, 2520)),
 (9, 18, False): (166.11278025793652, -221.6993253968254, 168.61260416666667,
                  -86.02527777777777, 30.77467757936508, -7.78297619047619,
                  1.3544357638888889, -0.14688492063492065,
                  0.00751860119047619),
 (9, 18, True): (Fraction(66976673, 403200), Fraction(-5586823, 25200),
                 Fraction(1618681, 9600), Fraction(-309691, 3600),
                 Fraction(248167, 8064), Fraction(-65377, 8400),
                 Fraction(156031, 115200), Fraction(-7403, 50400),
                 Fraction(2021, 268800)),
 (9, 20, False): (200.37387596200097, -273.0909689529221, 216.05104437229437,
                  -117.65090458152959, 46.58749098124098, -13.712781216179653,
                  2.982225378787879, -0.4569400853775854, 0.04423566017316017,
                  -0.0020398366101491102),
 (9, 20, True): (Fraction(666523661, 3326400), Fraction(-269158459, 985600),
                 Fraction(39926233, 184800), Fraction(-130451323, 1108800),
                 Fraction(5165621, 110880), Fraction(-81091903, 5913600),
                 Fraction(314923, 105600), Fraction(-3039931, 6652800),
                 Fraction(32699, 739200), Fraction(-21713, 10644480)),
 (10, 11, False): (-252.0, 210.0, -120.0, 45.0, -10.0, 1.0),
 (10, 11, True): (Fraction(-252, 1), Fraction(210, 1), Fraction(-120, 1),
                  Fraction(45, 1), Fraction(-10, 1), Fraction(1, 1)),
 (10, 13, False): (-637.0, 540.0, -326.25, 136.66666666666666, -37.5, 6.0,
                   -0.4166666666666667),
 (10, 13, True): (Fraction(-637, 1), Fraction(540, 1), Fraction(-1305, 4),
                  Fraction(410, 3), Fraction(-75, 2), Fraction(6, 1),
                  Fraction(-5, 12)),
 (10, 15, False): (-1066.0, 915.375, -576.5, 261.7916666666667, -83.0, 17.375,
                   -2.1666666666666665, 0.125),
 (10, 15, True): (Fraction(-1066, 1), Fraction(7323, 8), Fraction(-1153, 2),
                  Fraction(6283, 24), Fraction(-83, 1), Fraction(139, 8),
                  Fraction(-13, 6), Fraction(1, 8)),
 (10, 17, False): (-1493.7232142857142, 1295.5734126984128, -842.6388888888889,
                   406.9583333333333, -143.48611111111111, 35.986111111111114,
                   -6.154761904761905, 0.6567460317460317,
                   -0.033234126984126984),
 (10, 17, True): (Fraction(-167297, 112), Fraction(652969, 504),
                  Fraction(-30335, 36), Fraction(9767, 24),
                  Fraction(-10331, 72), Fraction(2591, 72), Fraction(-517, 84),
                  Fraction(331, 504), Fraction(-67, 2016)),
 (10, 19, False): (-1899.8947585978835, 1661.1278025793652, -1108.496626984127,
                   562.0420138888888, -215.06319444444443, 61.54935515873016,
                   -12.971626984126985, 1.934908234126984, -0.1836061507936508,
                   0.008354001322751322),
 (10, 19, True): (Fraction(-22981127, 12096), Fraction(66976673, 40320),
                  Fraction(-5586823, 5040), Fraction(1618681, 2880),
                  Fraction(-309691, 1440), Fraction(248167, 4032),
                  Fraction(-65377, 5040), Fraction(156031, 80640),
                  Fraction(-7403, 40320), Fraction(2021, 241920)),
 (10, 21, False): (-2276.7668113425925, 2003.7387596200097, -1365.4548447646105,
                   720.1701479076479, -294.12726145382396, 93.17498196248197,
                   -22.854635360299422, 4.26032196969697, -0.5711751067219817,
                   0.04915073352573353, -0.0020398366101491102),
 (10, 21, True): (Fraction(-78685061, 34560), Fraction(666523661, 332640),
                  Fraction(-269158459, 197120), Fraction(39926233, 55440),
                  Fraction(-130451323, 443520), Fraction(5165621, 55440),
                  Fraction(-81091903, 3548160), Fraction(44989, 10560),
                  Fraction(-3039931, 5322240), Fraction(32699, 665280),
                  Fraction(-21713, 10644480)),
 (11, 12, False): (-66.0, 82.5, -55.0, 22.0, -5.0, 0.5),
 (11, 12, True): (Fraction(-66, 1), Fraction(165, 2), Fraction(-55, 1),
                  Fraction(22, 1), Fraction(-5, 1), Fraction(1, 2)),
 (11, 14, False): (-191.125, 249.33333333333334, -180.125, 82.66666666666667,
                   -23.958333333333332, 4.0, -0.2916666666666667),
 (11, 14, True): (Fraction(-1529, 8), Fraction(748, 3), Fraction(-1441, 8),
                  Fraction(248, 3), Fraction(-575, 24), Fraction(4, 1),
                  Fraction(-7, 24)),
 (11, 16, False): (-353.9861111111111, 477.3388888888889, -366.675,
                   186.30555555555554, -63.81944444444444, 14.25,
                   -1.886111111111111, 0.11388888888888889),
 (11, 16, True): (Fraction(-25487, 72), Fraction(85921, 180),
                  Fraction(-14667, 40), Fraction(6707, 36), Fraction(-4595, 72),
                  Fraction(57, 4), Fraction(-679, 360), Fraction(41, 360)),
 (11, 18, False): (-536.5522817460318, 742.8896825396826, -599.0319444444444,
                   329.2944444444444, -127.65376984126983, 34.676984126984124,
                   -6.354513888888889, 0.7146825396825397,
                   -0.03754960317460317),
 (11, 18, True): (Fraction(-5408447, 10080), Fraction(936041, 1260),
                  Fraction(-431303, 720), Fraction(59273, 180),
                  Fraction(-128675, 1008), Fraction(43693, 1260),
                  Fraction(-18301, 2880), Fraction(1801, 2520),
                  Fraction(-757, 20160)),
 (11, 20, False): (-726.5417576058201, 1027.8738963293652, -862.0942956349206,
                   504.6693452380952, -215.34122023809525, 67.55977802579365,
                   -15.381163194444444, 2.4340443121693123,
                   -0.24115823412698412, 0.011311590608465608),
 (11, 20, True): (Fraction(-87882491, 120960), Fraction(82887751, 80640),
                  Fraction(-17379821, 20160), Fraction(1695689, 3360),
                  Fraction(-1447093, 6720), Fraction(10896041, 161280),
                  Fraction(-177191, 11520), Fraction(147211, 60480),
                  Fraction(-19447, 80640), Fraction(5473, 483840)),
 (11, 22, False): (-916.2723252177028, 1319.7670772707231, -1143.5627201140874,
                   704.8246693121694, -324.8011630911045, 113.91928323412698,
                   -30.405076919367286, 6.048820546737214, -0.8511517237103174,
                   0.07586116622574957, -0.0032274787808641977),
 (11, 22, True): (Fraction(-2659975211, 2903040), Fraction(478917077, 362880),
                  Fraction(-368867591, 322560), Fraction(10656949, 15120),
                  Fraction(-628607179, 1935360), Fraction(9186451, 80640),
                  Fraction(-25219187, 829440), Fraction(548749, 90720),
                  Fraction(-109819, 129024), Fraction(55057, 725760),
                  Fraction(-2677, 829440)),
 (12, 13, False): (924.0, -792.0, 495.0, -220.0, 66.0, -12.0, 1.0),
 (12, 13, True): (Fraction(924, 1), Fraction(-792, 1), Fraction(495, 1),
                  Fraction(-220, 1), Fraction(66, 1), Fraction(-12, 1),
                  Fraction(1, 1)),
 (12, 15, False): (2640.0, -2293.5, 1496.0, -720.5, 248.0, -57.5, 8.0, -0.5),
 (12, 15, True): (Fraction(2640, 1), Fraction(-4587, 2), Fraction(1496, 1),
                  Fraction(-1441, 2), Fraction(248, 1), Fraction(-115, 2),
                  Fraction(8, 1), Fraction(-1, 2)),
 (12, 17, False): (4838.625, -4247.833333333333, 2864.0333333333333, -1466.7,
                   558.9166666666666, -153.16666666666666, 28.5,
                   -3.2333333333333334, 0.17083333333333334),
 (12, 17, True): (Fraction(38709, 8), Fraction(-25487, 6), Fraction(85921, 30),
                  Fraction(-14667, 10), Fraction(6707, 12), Fraction(-919, 6),
                  Fraction(57, 2), Fraction(-97, 30), Fraction(41, 240)),
 (12, 19, False): (7272.840608465608, -6438.627380952381, 4457.338095238095,
                   -2396.1277777777777, 987.8833333333333, -306.3690476190476,
                   69.35396825396825, -10.89345238095238, 1.0720238095238095,
                   -0.050066137566137564),
 (12, 19, True): (Fraction(10996535, 1512), Fraction(-5408447, 840),
                  Fraction(936041, 210), Fraction(-431303, 180),
                  Fraction(59273, 60), Fraction(-25735, 84),
                  Fraction(43693, 630), Fraction(-18301, 1680),
                  Fraction(1801, 1680), Fraction(-757, 15120)),
 (12, 21, False): (9780.701689814814, -8718.501091269842, 6167.2433779761905,
                   -3448.3771825396825, 1514.0080357142858, -516.8189285714286,
                   135.1195560515873, -26.367708333333333, 3.651066468253968,
                   -0.3215443121693122, 0.01357390873015873),
 (12, 21, True): (Fraction(422526313, 43200), Fraction(-87882491, 10080),
                  Fraction(82887751, 13440), Fraction(-17379821, 5040),
                  Fraction(1695689, 1120), Fraction(-1447093, 2800),
                  Fraction(10896041, 80640), Fraction(-25313, 960),
                  Fraction(147211, 40320), Fraction(-19447, 60480),
                  Fraction(5473, 403200)),
 (12, 23, False): (12264.447302188551, -10995.267902612433, 7918.602463624338,
                   -4574.2508804563495, 2114.474007936508, -779.5227914186507,
                   227.83856646825396, -52.12298900462963, 9.07323082010582,
                   -1.13486896494709, 0.09103339947089947,
                   -0.0035208859427609427),
 (12, 23, True): (Fraction(2914032679, 237600), Fraction(-2659975211, 241920),
                  Fraction(478917077, 60480), Fraction(-368867591, 80640),
                  Fraction(10656949, 5040), Fraction(-628607179, 806400),
                  Fraction(9186451, 40320), Fraction(-3602741, 69120),
                  Fraction(548749, 60480), Fraction(-109819, 96768),
                  Fraction(55057, 604800), Fraction(-2677, 760320)),
 (13, 14, False): (214.5, -286.0, 214.5, -104.0, 32.5, -6.0, 0.5),
 (13, 14, True): (Fraction(429, 2), Fraction(-286, 1), Fraction(429, 2),
                  Fraction(-104, 1), Fraction(65, 2), Fraction(-6, 1),
                  Fraction(1, 2)),
 (13, 16, False): (691.1666666666666, -953.3333333333334, 760.5,
                   -407.3333333333333, 149.16666666666666, -36.0,
                   5.166666666666667, -0.3333333333333333),
 (13, 16, True): (Fraction(4147, 6), Fraction(-2860, 3), Fraction(1521, 2),
                  Fraction(-1222, 3), Fraction(895, 6), Fraction(-36, 1),
                  Fraction(31, 6), Fraction(-1, 3)),
 (13, 18, False): (1390.0791666666667, -1969.9333333333334, 1650.025,
                   -954.7333333333333, 393.5416666666667, -114.2,
                   22.272916666666667, -2.6333333333333333, 0.14375),
 (13, 18, True): (Fraction(333619, 240), Fraction(-29549, 15),
                  Fraction(66001, 40), Fraction(-14321, 15), Fraction(9445, 24),
                  Fraction(-571, 5), Fraction(10691, 480), Fraction(-79, 30),
                  Fraction(23, 160)),
 (13, 20, False): (2249.596693121693, -3259.209623015873, 2840.1261904761905,
                   -1748.134126984127, 790.2420634920635, -262.9626488095238,
                   63.109722222222224, -10.411772486772486, 1.0648809523809524,
                   -0.0511739417989418),
 (13, 20, True): (Fraction(17006951, 7560), Fraction(-32852833, 10080),
                  Fraction(1192853, 420), Fraction(-2202649, 1260),
                  Fraction(199141, 252), Fraction(-1767109, 6720),
                  Fraction(45439, 720), Fraction(-78713, 7560),
                  Fraction(1789, 1680), Fraction(-619, 12096)),
 (13, 22, False): (3212.8476171186066, -4741.1341214726635, 4269.124813988095,
                   -2764.310925925926, 1345.9637504133598, -498.32712797619047,
                   139.38524787808643, -28.763778659611994, 4.161781994047619,
                   -0.3788883377425044, 0.01638571979717813),
 (13, 22, True): (Fraction(11658781433, 3628800), Fraction(-172046275, 36288),
                  Fraction(22950815, 5376), Fraction(-14927279, 5400),
                  Fraction(651231101, 483840), Fraction(-33487583, 67200),
                  Fraction(5780585, 41472), Fraction(-260945, 9072),
                  Fraction(1118687, 268800), Fraction(-137491, 362880),
                  Fraction(118921, 7257600)),
 (13, 24, False): (4234.884892300986, -6347.192696759259, 5875.183389274692,
                   -3968.854857390873, 2054.51900421627, -828.9862464175485,
                   261.2070283564815, -63.57000165343916, 11.620258349867726,
                   -1.508960512866763, 0.12447958002645503,
                   -0.00491335728314895),
 (13, 24, True): (Fraction(56347684423, 13305600), Fraction(-548397449, 86400),
                  Fraction(3045695069, 518400), Fraction(-3200484557, 806400),
                  Fraction(66270565, 32256), Fraction(-3008225291, 3628800),
                  Fraction(90273149, 345600), Fraction(-38447137, 604800),
                  Fraction(28111729, 2419200), Fraction(-803105, 532224),
                  Fraction(301141, 2419200), Fraction(-392251, 79833600)),
 (14, 15, False): (-3432.0, 3003.0, -2002.0, 1001.0, -364.0, 91.0, -14.0, 1.0),
 (14, 15, True): (Fraction(-3432, 1), Fraction(3003, 1), Fraction(-2002, 1),
                  Fraction(1001, 1), Fraction(-364, 1), Fraction(91, 1),
                  Fraction(-14, 1), Fraction(1, 1)),
 (14, 17, False): (-10939.5, 9676.333333333334, -6673.333333333333, 3549.0,
                   -1425.6666666666667, 417.6666666666667, -84.0,
                   10.333333333333334, -0.5833333333333334),
 (14, 17, True): (Fraction(-21879, 2), Fraction(29029, 3), Fraction(-20020, 3),
                  Fraction(3549, 1), Fraction(-4277, 3), Fraction(1253, 3),
                  Fraction(-84, 1), Fraction(31, 3), Fraction(-7, 12)),
 (14, 19, False): (-21811.472222222223, 19461.108333333334, -13789.533333333333,
                   7700.116666666667, -3341.5666666666666, 1101.9166666666667,
                   -266.46666666666664, 44.545833333333334, -4.608333333333333,
                   0.22361111111111112),
 (14, 19, True): (Fraction(-785213, 36), Fraction(2335333, 120),
                  Fraction(-206843, 15), Fraction(462007, 60),
                  Fraction(-100247, 30), Fraction(13223, 12),
                  Fraction(-3997, 15), Fraction(10691, 240),
                  Fraction(-553, 120), Fraction(161, 720)),
 (14, 21, False): (-35048.042129629626, 31494.353703703702, -22814.46736111111,
                   13253.922222222222, -6118.469444444445, 2212.677777777778,
                   -613.5795138888889, 126.21944444444445, -18.220601851851853,
                   1.6564814814814814, -0.07164351851851852),
 (14, 21, True): (Fraction(-75703771, 2160), Fraction(17006951, 540),
                  Fraction(-32852833, 1440), Fraction(1192853, 90),
                  Fraction(-2202649, 360), Fraction(199141, 90),
                  Fraction(-1767109, 2880), Fraction(45439, 360),
                  Fraction(-78713, 4320), Fraction(1789, 1080),
                  Fraction(-619, 8640)),
 (14, 23, False): (-49759.510787037034, 44979.8666396605, -33187.938850308645,
                   19922.582465277777, -9675.08824074074, 3768.6985011574075,
                   -1162.7632986111112, 278.77049575617286, -50.33661265432099,
                   6.473883101851852, -0.5304436728395062, 0.0208545524691358),
 (14, 23, True): (Fraction(-1074805433, 21600), Fraction(11658781433, 259200),
                  Fraction(-172046275, 5184), Fraction(22950815, 1152),
                  Fraction(-104490953, 10800), Fraction(651231101, 172800),
                  Fraction(-33487583, 28800), Fraction(5780585, 20736),
                  Fraction(-260945, 5184), Fraction(1118687, 172800),
                  Fraction(-137491, 259200), Fraction(10811, 518400)),
 (14, 25, False): (-65260.40946063646, 59288.3884922138, -44430.34887731481,
                   27417.522483281893, -13890.992000868055, 5752.653211805556,
                   -1934.3012416409465, 522.414056712963, -111.24750289351852,
                   18.075957433127574, -2.112544718013468, 0.1584285563973064,
                   -0.0057322501636737746),
 (14, 25, True): (Fraction(-44656915069, 684288), Fraction(56347684423, 950400),
                  Fraction(-3838782143, 86400), Fraction(21319865483, 777600),
                  Fraction(-3200484557, 230400), Fraction(13254113, 2304),
                  Fraction(-3008225291, 1555200), Fraction(90273149, 172800),
                  Fraction(-38447137, 345600), Fraction(28111729, 1555200),
                  Fraction(-160621, 76032), Fraction(301141, 1900800),
                  Fraction(-392251, 68428800)),
 (15, 16, False): (-715.0, 1001.0, -819.0, 455.0, -175.0, 45.0, -7.0, 0.5),
 (15, 16, True): (Fraction(-715, 1), Fraction(1001, 1), Fraction(-819, 1),
                  Fraction(455, 1), Fraction(-175, 1), Fraction(45, 1),
                  Fraction(-7, 1), Fraction(1, 2)),
 (15, 18, False): (-2538.25, 3653.0, -3139.5, 1883.0, -812.5, 249.0, -51.625,
                   6.5, -0.375),
 (15, 18, True): (Fraction(-10153, 4), Fraction(3653, 1), Fraction(-6279, 2),
                  Fraction(1883, 1), Fraction(-1625, 2), Fraction(249, 1),
                  Fraction(-413, 8), Fraction(13, 2), Fraction(-3, 8)),
 (15, 20, False): (-5512.541666666667, 8114.4375, -7257.75, 4628.5, -2185.25,
                   763.78125, -192.9375, 33.416666666666664, -3.5625,
                   0.17708333333333334),
 (15, 20, True): (Fraction(-132301, 24), Fraction(129831, 16),
                  Fraction(-29031, 4), Fraction(9257, 2), Fraction(-8741, 4),
                  Fraction(24441, 32), Fraction(-3087, 16), Fraction(401, 12),
                  Fraction(-57, 16), Fraction(17, 96)),
 (15, 22, False): (-9495.273726851852, 14241.717592592593, -13166.198660714286,
                   8830.063492063493, -4482.980034722223, 1736.9375,
                   -508.31221064814815, 109.29629629629629, -16.3671875,
                   1.5320767195767195, -0.06774966931216932),
 (15, 22, True): (Fraction(-16407833, 1728), Fraction(3076211, 216),
                  Fraction(-5898457, 448), Fraction(556294, 63),
                  Fraction(-5164393, 1152), Fraction(27791, 16),
                  Fraction(-1756727, 3456), Fraction(2951, 27),
                  Fraction(-2095, 128), Fraction(4633, 3024),
                  Fraction(-1639, 24192)),
 (15, 24, False): (-14279.84207175926, 21760.324991732803, -20684.806059854498,
                   14469.01904141865, -7800.012710813492, 3284.886082175926,
                   -1078.609056712963, 272.2382523148148, -51.28332093253968,
                   6.822399966931217, -0.5737805886243387,
                   0.023001405423280424),
 (15, 24, True): (Fraction(-246755671, 17280), Fraction(2632128911, 120960),
                  Fraction(-2502034141, 120960), Fraction(2333563391, 161280),
                  Fraction(-125798605, 16128), Fraction(113525663, 34560),
                  Fraction(-37276729, 34560), Fraction(4704277, 17280),
                  Fraction(-4135487, 80640), Fraction(330095, 48384),
                  Fraction(-138809, 241920), Fraction(11129, 483840)),
 (15, 26, False): (-19658.44724708644, 30366.093272256294, -29559.504599144344,
                   21429.566915371473, -12150.355132034007, 5482.953831845238,
                   -1976.1533878279322, 565.3139522707231, -126.21744876217532,
                   21.302424668310085, -2.564783985063933, 0.19676170183982683,
                   -0.007240012350689434),
 (15, 26, True): (Fraction(-313880922829, 15966720),
                  Fraction(121211727193, 3991680),
                  Fraction(-6356475869, 215040), Fraction(31105444969, 1451520),
                  Fraction(-14109186785, 1161216), Fraction(147381799, 26880),
                  Fraction(-819550333, 414720), Fraction(205141127, 362880),
                  Fraction(-49759967, 394240), Fraction(34012985, 1596672),
                  Fraction(-14891341, 5806080), Fraction(116357, 591360),
                  Fraction(-462397, 63866880)),
 (16, 17, False): (12870.0, -11440.0, 8008.0, -4368.0, 1820.0, -560.0, 120.0,
                   -16.0, 1.0),
 (16, 17, True): (Fraction(12870, 1), Fraction(-11440, 1), Fraction(8008, 1),
                  Fraction(-4368, 1), Fraction(1820, 1), Fraction(-560, 1),
                  Fraction(120, 1), Fraction(-16, 1), Fraction(1, 1)),
 (16, 19, False): (45283.333333333336, -40612.0, 29224.0, -16744.0, 7532.0,
                   -2600.0, 664.0, -118.0, 13.0, -0.6666666666666666),
 (16, 19, True): (Fraction(135850, 3), Fraction(-40612, 1), Fraction(29224, 1),
                  Fraction(-16744, 1), Fraction(7532, 1), Fraction(-2600, 1),
                  Fraction(664, 1), Fraction(-118, 1), Fraction(13, 1),
                  Fraction(-2, 3)),
 (16, 21, False): (97630.86666666667, -88200.66666666667, 64915.5, -38708.0,
                   18514.0, -6992.8, 2036.75, -441.0, 66.83333333333333,
                   -6.333333333333333, 0.2833333333333333),
 (16, 21, True): (Fraction(1464463, 15), Fraction(-264602, 3),
                  Fraction(129831, 2), Fraction(-38708, 1), Fraction(18514, 1),
                  Fraction(-34964, 5), Fraction(8147, 4), Fraction(-441, 1),
                  Fraction(401, 6), Fraction(-19, 3), Fraction(17, 60)),
 (16, 23, False): (167147.64444444445, -151924.37962962964, 113933.74074074074,
                   -70219.72619047618, 35320.25396825397, -14345.53611111111,
                   4631.833333333333, -1161.8564814814815, 218.59259259259258,
                   -29.09722222222222, 2.4513227513227513,
                   -0.09854497354497355),
 (16, 23, True): (Fraction(7521644, 45), Fraction(-16407833, 108),
                  Fraction(3076211, 27), Fraction(-5898457, 84),
                  Fraction(2225176, 63), Fraction(-5164393, 360),
                  Fraction(27791, 6), Fraction(-250961, 216),
                  Fraction(5902, 27), Fraction(-2095, 72), Fraction(4633, 1890),
                  Fraction(-149, 1512)),
 (16, 25, False): (250080.1624228395, -228477.47314814816, 174082.59993386242,
                   -110318.96565255732, 57876.0761656746, -24960.040674603173,
                   8759.696219135802, -2465.39212962963, 544.4765046296296,
                   -91.17034832451499, 10.915839947089948, -0.834589947089947,
                   0.0306685405643739),
 (16, 25, True): (Fraction(648207781, 2592), Fraction(-246755671, 1080),
                  Fraction(2632128911, 15120), Fraction(-2502034141, 22680),
                  Fraction(2333563391, 40320), Fraction(-25159721, 1008),
                  Fraction(113525663, 12960), Fraction(-5325247, 2160),
                  Fraction(4704277, 8640), Fraction(-4135487, 45360),
                  Fraction(66019, 6048), Fraction(-12619, 15120),
                  Fraction(11129, 362880)),
 (16, 27, False): (342757.6669823232, -314535.155953383, 242928.74617805035,
                   -157650.69119543652, 85718.2676614859, -38881.13642250882,
                   14621.210218253967, -4516.922029320987, 1130.6279045414462,
                   -224.38657557720057, 34.083879469296136, -3.730594887365721,
                   0.2623489357864358, -0.008910784431617766),
 (16, 27, True): (Fraction(1085856289, 3168), Fraction(-313880922829, 997920),
                  Fraction(121211727193, 498960), Fraction(-6356475869, 40320),
                  Fraction(31105444969, 362880), Fraction(-2821837357, 72576),
                  Fraction(147381799, 10080), Fraction(-117078619, 25920),
                  Fraction(205141127, 181440), Fraction(-49759967, 221760),
                  Fraction(6802597, 199584), Fraction(-14891341, 3991680),
                  Fraction(116357, 443520), Fraction(-35569, 3991680)),
 (17, 18, False): (2431.0, -3536.0, 3094.0, -1904.0, 850.0, -272.0, 59.5, -8.0,
                   0.5),
 (17, 18, True): (Fraction(2431, 1), Fraction(-3536, 1), Fraction(3094, 1),
                  Fraction(-1904, 1), Fraction(850, 1), Fraction(-272, 1),
                  Fraction(119, 2), Fraction(-8, 1), Fraction(1, 2)),
 (17, 20, False): (9429.333333333334, -14033.5, 12784.0, -8364.0, 4080.0,
                   -1483.25, 392.0, -71.33333333333333, 8.0,
                   -0.4166666666666667),
 (17, 20, True): (Fraction(28288, 3), Fraction(-28067, 2), Fraction(12784, 1),
                  Fraction(-8364, 1), Fraction(4080, 1), Fraction(-5933, 4),
                  Fraction(392, 1), Fraction(-214, 3), Fraction(8, 1),
                  Fraction(-5, 12)),
 (17, 22, False): (22003.005555555555, -33377.61111111111, 31437.25,
                   -21628.533333333333, 11334.041666666666, -4555.55,
                   1387.6527777777778, -310.8888888888889, 48.425,
                   -4.694444444444445, 0.21388888888888888),
 (17, 22, True): (Fraction(3960541, 180), Fraction(-600797, 18),
                  Fraction(125749, 4), Fraction(-324428, 15),
                  Fraction(272017, 24), Fraction(-91111, 20),
                  Fraction(99911, 72), Fraction(-2798, 9), Fraction(1937, 40),
                  Fraction(-169, 36), Fraction(77, 360)),
 (17, 24, False): (40217.81296296296, -62000.879894179896, 60060.51878306879,
                   -43095.98492063492, 23961.954365079364, -10448.575925925927,
                   3558.767592592593, -931.2074074074075, 181.3503968253968,
                   -24.834656084656086, 2.1403439153439154,
                   -0.08756613756613757),
 (17, 24, True): (Fraction(21717619, 540), Fraction(-117181663, 1890),
                  Fraction(227028761, 3780), Fraction(-54300941, 1260),
                  Fraction(12076825, 504), Fraction(-5642231, 540),
                  Fraction(3843469, 1080), Fraction(-125713, 135),
                  Fraction(457003, 2520), Fraction(-18775, 756),
                  Fraction(16181, 7560), Fraction(-331, 3780)),
 (17, 26, False): (63576.94901344797, -99375.49757495591, 98603.09326636905,
                   -73325.45510361552, 42855.37322944224, -19994.724404761906,
                   7456.778221450617, -2204.027204585538, 506.7872767857143,
                   -87.72100970017637, 10.787217537477954, -0.842202380952381,
                   0.03144317680776014),
 (17, 26, True): (Fraction(11535401629, 181440), Fraction(-450767257, 4536),
                  Fraction(2650451147, 26880), Fraction(-6652085287, 90720),
                  Fraction(6220543135, 145152), Fraction(-33591137, 1680),
                  Fraction(386559383, 51840), Fraction(-49987337, 22680),
                  Fraction(2270407, 4480), Fraction(-795805, 9072),
                  Fraction(7828931, 725760), Fraction(-14149, 16800),
                  Fraction(114101, 3628800)),
 (17, 28, False): (91323.99824535033, -144464.45257679725, 146344.3397389069,
                   -112225.73000716491, 68447.65935019842, -33814.55890997024,
                   13598.926890432098, -4437.5358114878945, 1162.2734983766234,
                   -239.45393136473865, 37.492211750440916, -4.203670183982684,
                   0.3011905931003153, -0.010374900626636738),
 (17, 28, True): (Fraction(91134044329, 997920),
                  Fraction(-2306623464247, 15966720),
                  Fraction(21635547187, 147840), Fraction(-8144894581, 72576),
                  Fraction(551961925, 8064), Fraction(-1817870687, 53760),
                  Fraction(70496837, 5184), Fraction(-4428305737, 997920),
                  Fraction(28638419, 24640), Fraction(-764658775, 3193344),
                  Fraction(68025869, 1814400), Fraction(-3107353, 739200),
                  Fraction(18033847, 59875200), Fraction(-354971, 34214400)),
 (18, 19, False): (-48620.0, 43758.0, -31824.0, 18564.0, -8568.0, 3060.0,
                   -816.0, 153.0, -18.0, 1.0),
 (18, 19, True): (Fraction(-48620, 1), Fraction(43758, 1), Fraction(-31824, 1),
                  Fraction(18564, 1), Fraction(-8568, 1), Fraction(3060, 1),
                  Fraction(-816, 1), Fraction(153, 1), Fraction(-18, 1),
                  Fraction(1, 1)),
 (18, 21, False): (-187187.0, 169728.0, -126301.5, 76704.0, -37638.0, 14688.0,
                   -4449.75, 1008.0, -160.5, 16.0, -0.75),
 (18, 21, True): (Fraction(-187187, 1), Fraction(169728, 1),
                  Fraction(-252603, 2), Fraction(76704, 1), Fraction(-37638, 1),
                  Fraction(14688, 1), Fraction(-17799, 4), Fraction(1008, 1),
                  Fraction(-321, 2), Fraction(16, 1), Fraction(-3, 4)),
 (18, 23, False): (-434088.2, 396054.1, -300398.5, 188623.5, -97328.4, 40802.55,
                   -13666.65, 3568.25, -699.5, 96.85, -8.45, 0.35),
 (18, 23, True): (Fraction(-2170441, 5), Fraction(3960541, 10),
                  Fraction(-600797, 2), Fraction(377247, 2),
                  Fraction(-486642, 5), Fraction(816051, 20),
                  Fraction(-273333, 20), Fraction(14273, 4), Fraction(-1399, 2),
                  Fraction(1937, 20), Fraction(-169, 20), Fraction(7, 20)),
 (18, 25, False): (-789276.9444444445, 723920.6333333333, -558007.9190476191,
                   360363.1126984127, -193931.93214285714, 86263.03571428571,
                   -31345.727777777778, 9151.116666666667, -2095.2166666666667,
                   362.7007936507936, -44.70238095238095, 3.5023809523809524,
                   -0.13134920634920635),
 (18, 25, True): (Fraction(-14206985, 18), Fraction(21717619, 30),
                  Fraction(-117181663, 210), Fraction(227028761, 630),
                  Fraction(-54300941, 280), Fraction(2415365, 28),
                  Fraction(-5642231, 180), Fraction(549067, 60),
                  Fraction(-125713, 60), Fraction(457003, 1260),
                  Fraction(-3755, 84), Fraction(1471, 420),
                  Fraction(-331, 2520)),
 (18, 27, False): (-1242084.8125, 1144385.0822420635, -894379.4781746032,
                   591618.5595982143, -329964.54796626983, 154279.34362599207,
                   -59984.173214285714, 19174.572569444445, -4959.06121031746,
                   1013.5745535714286, -157.89781746031747, 17.651810515873017,
                   -1.2633035714285714, 0.043536706349206346),
 (18, 27, True): (Fraction(-19873357, 16), Fraction(11535401629, 10080),
                  Fraction(-450767257, 504), Fraction(2650451147, 4480),
                  Fraction(-6652085287, 20160), Fraction(1244108627, 8064),
                  Fraction(-33591137, 560), Fraction(55222769, 2880),
                  Fraction(-49987337, 10080), Fraction(2270407, 2240),
                  Fraction(-159161, 1008), Fraction(711721, 40320),
                  Fraction(-14149, 11200), Fraction(8777, 201600)),
 (18, 29, False): (-1777206.4762581168, 1643831.968416306, -1300180.0731911752,
                   878066.0384334415, -505015.78503224207, 246411.5736607143,
                   -101443.67672991072, 34968.6691468254, -9984.455575847764,
                   2324.5469967532467, -431.0170764565296, 61.35089195526695,
                   -6.305505275974026, 0.4170331289081289,
                   -0.01333915794853295),
 (18, 29, True): (Fraction(-8758073515, 4928), Fraction(91134044329, 55440),
                  Fraction(-2306623464247, 1774080),
                  Fraction(21635547187, 24640), Fraction(-8144894581, 16128),
                  Fraction(110392385, 448), Fraction(-1817870687, 17920),
                  Fraction(70496837, 2016), Fraction(-4428305737, 443520),
                  Fraction(28638419, 12320), Fraction(-152931755, 354816),
                  Fraction(68025869, 1108800), Fraction(-3107353, 492800),
                  Fraction(1387219, 3326400), Fraction(-354971, 26611200)),
 (19, 20, False): (-8398.0, 12597.0, -11628.0, 7752.0, -3876.0, 1453.5, -399.0,
                   76.0, -9.0, 0.5),
 (19, 20, True): (Fraction(-8398, 1), Fraction(12597, 1), Fraction(-11628, 1),
                  Fraction(7752, 1), Fraction(-3876, 1), Fraction(2907, 2),
                  Fraction(-399, 1), Fraction(76, 1), Fraction(-9, 1),
                  Fraction(1, 2)),
 (19, 22, False): (-35341.583333333336, 54048.666666666664, -51599.25, 36176.0,
                   -19420.375, 8037.0, -2532.5416666666665, 589.3333333333334,
                   -95.625, 9.666666666666666, -0.4583333333333333),
 (19, 22, True): (Fraction(-424099, 12), Fraction(162146, 3),
                  Fraction(-206397, 4), Fraction(36176, 1),
                  Fraction(-155363, 8), Fraction(8037, 1), Fraction(-60781, 24),
                  Fraction(1768, 3), Fraction(-765, 8), Fraction(29, 3),
                  Fraction(-11, 24)),
 (19, 24, False): (-88211.3, 137129.65, -134680.23333333334, 98486.7375,
                   -56073.75, 25141.908333333333, -8834.35, 2389.85, -481.45,
                   68.125, -6.05, 0.25416666666666665),
 (19, 24, True): (Fraction(-882113, 10), Fraction(2742593, 20),
                  Fraction(-4040407, 30), Fraction(7878939, 80),
                  Fraction(-224295, 4), Fraction(3017029, 120),
                  Fraction(-176687, 20), Fraction(47797, 20),
                  Fraction(-9629, 20), Fraction(545, 8), Fraction(-121, 20),
                  Fraction(61, 240)),
 (19, 26, False): (-170608.34365079366, 268964.91984126985, -270635.35535714286,
                   205118.20575396824, -122718.41765873016, 58815.00357142857,
                   -22584.19722222222, 6879.596031746032, -1629.3964285714285,
                   289.9503968253968, -36.55099206349206, 2.9160714285714286,
                   -0.11091269841269841),
 (19, 26, True): (Fraction(-214966513, 1260), Fraction(338895799, 1260),
                  Fraction(-151555799, 560), Fraction(1033795757, 5040),
                  Fraction(-123700165, 1008), Fraction(16468201, 280),
                  Fraction(-8130311, 360), Fraction(8668291, 1260),
                  Fraction(-456231, 280), Fraction(146135, 504),
                  Fraction(-184217, 5040), Fraction(1633, 560),
                  Fraction(-559, 5040)),
 (19, 28, False): (-282987.44573412696, 451580.9607266865, -463993.5162946429,
                   362669.2998511905, -226370.45324900793, 114787.10279017857,
                   -47460.68576388889, 15925.591865079365, -4284.199553571429,
                   904.4881572420635, -144.7096378968254, 16.53044642857143,
                   -1.2034242724867725, 0.042019675925925926),
 (19, 28, True): (Fraction(-2852513453, 10080), Fraction(36415488673, 80640),
                  Fraction(-2078690953, 4480), Fraction(487427539, 1344),
                  Fraction(-1825451335, 8064), Fraction(1028492441, 8960),
                  Fraction(-27337355, 576), Fraction(80264983, 5040),
                  Fraction(-9596607, 2240), Fraction(14587585, 16128),
                  Fraction(-29173463, 201600), Fraction(185141, 11200),
                  Fraction(-727831, 604800), Fraction(7261, 172800)),
 (19, 30, False): (-423682.9584378946, 683314.7463564214, -715038.4507268557,
                   574075.5604256854, -371712.25739397324, 197839.5623015873,
                   -87099.3596216067, 31682.58023088023, -9454.461361099839,
                   2283.224639249639, -436.36543216765875, 63.66673641173641,
                   -6.674600788389851, 0.44836700336700336,
                   -0.014512404551467052),
 (19, 30, True): (Fraction(-214756417973, 506880), Fraction(18941484769, 27720),
                  Fraction(-2537070829331, 3548160), Fraction(3182674907, 5544),
                  Fraction(-2664433461, 7168), Fraction(498555697, 2520),
                  Fraction(-8829784681, 101376), Fraction(219560281, 6930),
                  Fraction(-3727326847, 394240), Fraction(63290987, 27720),
                  Fraction(-703770169, 1612800), Fraction(26472629, 415800),
                  Fraction(-355238273, 53222400), Fraction(26633, 59400),
                  Fraction(-154477, 10644480)),
 (20, 21, False): (184756.0, -167960.0, 125970.0, -77520.0, 38760.0, -15504.0,
                   4845.0, -1140.0, 190.0, -20.0, 1.0),
 (20, 21, True): (Fraction(184756, 1), Fraction(-167960, 1),
                  Fraction(125970, 1), Fraction(-77520, 1), Fraction(38760, 1),
                  Fraction(-15504, 1), Fraction(4845, 1), Fraction(-1140, 1),
                  Fraction(190, 1), Fraction(-20, 1), Fraction(1, 1)),
 (20, 23, False): (772616.0, -706831.6666666666, 540486.6666666666, -343995.0,
                   180880.0, -77681.5, 26790.0, -7235.833333333333,
                   1473.3333333333333, -212.5, 19.333333333333332,
                   -0.8333333333333334),
 (20, 23, True): (Fraction(772616, 1), Fraction(-2120495, 3),
                  Fraction(1621460, 3), Fraction(-343995, 1),
                  Fraction(180880, 1), Fraction(-155363, 2), Fraction(26790, 1),
                  Fraction(-43415, 6), Fraction(4420, 3), Fraction(-425, 2),
                  Fraction(58, 3), Fraction(-5, 6)),
 (20, 25, False): (1918126.5277777778, -1764226.0, 1371296.5,
                   -897868.2222222222, 492433.6875, -224295.0,
                   83806.36111111111, -25241.0, 5974.625, -1069.888888888889,
                   136.25, -11.0, 0.4236111111111111),
 (20, 25, True): (Fraction(69052555, 36), Fraction(-1764226, 1),
                  Fraction(2742593, 2), Fraction(-8080814, 9),
                  Fraction(7878939, 16), Fraction(-224295, 1),
                  Fraction(3017029, 36), Fraction(-25241, 1),
                  Fraction(47797, 8), Fraction(-9629, 9), Fraction(545, 4),
                  Fraction(-11, 1), Fraction(61, 144)),
 (20, 27, False): (3692832.0833333335, -3412166.873015873, 2689649.198412698,
                   -1804235.7023809524, 1025591.0287698413, -490873.67063492065,
                   196050.0119047619, -64526.27777777778, 17198.990079365078,
                   -3620.8809523809523, 579.9007936507936, -66.4563492063492,
                   4.8601190476190474, -0.17063492063492064),
 (20, 27, True): (Fraction(44313985, 12), Fraction(-214966513, 63),
                  Fraction(338895799, 126), Fraction(-151555799, 84),
                  Fraction(1033795757, 1008), Fraction(-123700165, 252),
                  Fraction(16468201, 84), Fraction(-1161473, 18),
                  Fraction(8668291, 504), Fraction(-152077, 42),
                  Fraction(146135, 252), Fraction(-16747, 252),
                  Fraction(1633, 336), Fraction(-43, 252)),
 (20, 29, False): (6100955.699404762, -5659748.91468254, 4515809.607266865,
                   -3093290.1086309524, 1813346.4992559524, -905481.8129960317,
                   382623.6759672619, -135601.95932539683, 39813.97966269841,
                   -9520.443452380952, 1808.976314484127, -263.10843253968255,
                   27.550744047619048, -1.8514219576719577,
                   0.060028108465608465),
 (20, 29, True): (Fraction(2049921115, 336), Fraction(-2852513453, 504),
                  Fraction(36415488673, 8064), Fraction(-2078690953, 672),
                  Fraction(2437137695, 1344), Fraction(-1825451335, 2016),
                  Fraction(1028492441, 2688), Fraction(-136686775, 1008),
                  Fraction(80264983, 2016), Fraction(-3198869, 336),
                  Fraction(14587585, 8064), Fraction(-2652133, 10080),
                  Fraction(185141, 6720), Fraction(-55987, 30240),
                  Fraction(7261, 120960)),
 (20, 31, False): (9102459.97041847, -8473659.168757891, 6833147.463564213,
                   -4766923.004845704, 2870377.802128427, -1486849.029575893,
                   659465.2076719577, -248855.31320459055, 79206.45057720058,
                   -21009.914135777417, 4566.449278499278, -793.3916948502886,
                   106.11122735289402, -10.268616597522847, 0.6405242905242905,
                   -0.019349872735289403),
 (20, 31, True): (Fraction(12616009519, 1386), Fraction(-214756417973, 25344),
                  Fraction(18941484769, 2772), Fraction(-2537070829331, 532224),
                  Fraction(15913374535, 5544), Fraction(-2664433461, 1792),
                  Fraction(498555697, 756), Fraction(-44148923405, 177408),
                  Fraction(219560281, 2772), Fraction(-3727326847, 177408),
                  Fraction(63290987, 13860), Fraction(-703770169, 887040),
                  Fraction(26472629, 249480), Fraction(-27326021, 2661120),
                  Fraction(26633, 41580), Fraction(-154477, 7983360)),
}
