"""
Tests and benchmarks for `ripples.differentiation`.

This script exercises every public surface of the differentiation module
(***nth_numerical_derivative***, ***numerical_hessian_vector_product***,
and the ***DifferentiationResult*** they produce) along:

- Numerical correctness against analytical formulas, with tolerances
chosen per method (machine epsilon for complex-step, a few machine epsilons for
the Richardson path, and a relaxed bound for plain central differences).

- API: input/output shapes, parameter validation, the warning channel, and the
properties every method sufaces of the ***DifferentiationResult*** wrapper.

- Performance: elapsed time of the three numerical strategies on the same
problem, the Hessian-vector product against forming the full Hessian, the
impact of the internal stencil cache, and scaling with the dimension and the
derivative order.


Output convention
-----------------
For each test the script prints, on its own block:

- a header line of the form:
    "Section <n>, Test <m>: s<n>.<name> - <description>", where the description
    states in one sentence what the test verifies and why it matters, with the
    checking procedure folded into the same sentence.
- RESULT     : the measured value(s) or timings.
- VERDICT    : one of PASS / FAIL / INFO / SKIP, followed by an
               interpretation of what the result means in practice.

Pure-benchmark blocks omit the VERDICT line since they only collect
timings and report them; they are flagged with INFO and never cause the
suite to fail.


How to run
----------
Through the library's unified entry point:

    >>> import ripples
    >>> ripples.test("differentiation")

from the command line:

    python -m ripples.differentiation._test
    # append --no-benchmarks to skip the timing-only sections

or when a caller wants the tallies back:

    >>> from ripples.differentiation._test import run
    >>> reporter = run()
    >>> reporter.failed
    0

Under continuous integration the suite is reached through pytest, which
collects the bridge test at the foot of this file. However it is launched,
every path funnels through the same `run()`, so the output and the verdicts are
identical.

The exit code of the command-line form is 0 when every test passes and the
number of failures otherwise. The summary block at the end lists every
failure together with its test name, the expected value, and the actual
value.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np

import sys
import time
import math
import warnings
from typing import Callable, List, Tuple, Optional, Any

from ._numerical_differentiation import (
    nth_numerical_derivative, numerical_hessian_vector_product
)



FLOAT_EPSILON = float(np.finfo(float).eps)

SEPARATOR_WIDTH = 79

_CURRENT_SECTION_TEST_NUMBER = 0



# REPORTING UTILITIES
#
# Every test routes through check_* helpers that record a pass / fail entry on
# the module-level REPORT object. The helpers print the verdict line in place,
# so that the caller can read the output top-to-bottom as the suite progresses,
# and the summary at the end of `main()` re-prints every failure.



class Reporter:
    """
    Aggregator for pass / fail / skip / info events.

    A single instance is kept at module level (`REPORT`) so every test
    can record into the same accumulator without explicit wiring.

    Attributes
    ----------
    passed : int
        Number of `check_*` calls that returned a passing verdict.
    failed : int
        Number of `check_*` calls that returned a failing verdict.
    skipped : int
        Number of tests that decided to short-circuit (for example,
        because the platform lacks an optional dependency).
    info : int
        Number of pure-information blocks (benchmarks).
    failure_records : list of tuple
        Tuples (test_name, message, expected, actual) recorded for
        every failure, replayed verbatim in the final summary.
    """

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.info: int = 0
        self.failure_records: List[Tuple[str, str, Any, Any]] = []
        self.verbose: bool = True
        self._writer: Optional[_StdoutProxy] = None

    def reset(self) -> None:
        """
        Clear every counter and the recorded failure list.

        Called at the top of `run()` so the same module-level reporter
        can be reused across repeated invocations - for instance when
        `ripples.test()` drives this suite on its own and then again as
        part of an "everything" run - without the second run inheriting
        the tallies of the first.
        """
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.info = 0
        self.failure_records.clear()

    def record_pass(self, test_name: str, message: str) -> None:
        self.passed += 1
        print(f"  VERDICT     : PASS - {message}")

    def record_fail(
        self, test_name: str, message: str,
        expected: Any = None, actual: Any = None,
    ) -> None:
        self.failed += 1
        self.failure_records.append((test_name, message, expected, actual))
        print(f"  VERDICT     : FAIL - {message}")
        if expected is not None or actual is not None:
            print(f"                expected = {expected!r}")
            print(f"                actual   = {actual!r}")
        if not self.verbose and self._writer is not None:
            self._writer.mark_failed()

    def record_skip(self, test_name: str, reason: str) -> None:
        self.skipped += 1
        print(f"  VERDICT     : SKIP - {reason}")

    def record_info(self, message: str) -> None:
        self.info += 1
        print(f"  VERDICT     : INFO - {message}")


    def print_summary(self) -> None:
        """Print the end-of-run accounting and replay each failure."""

        total_tests = self.passed + self.failed + self.skipped
        print()
        print("=" * SEPARATOR_WIDTH)
        print(" FINAL SUMMARY ".center(SEPARATOR_WIDTH, "="))
        print("=" * SEPARATOR_WIDTH)
        print(f"  Total tests      : {total_tests}")
        print(f"  Passed           : {self.passed}")
        print(f"  Failed           : {self.failed}")
        print(f"  Skipped          : {self.skipped}")
        print(f"  Info / benchmark : {self.info}")

        if self.failure_records:
            print()
            print(" FAILURES ".center(SEPARATOR_WIDTH, "-"))
            for test_name, message, expected, actual in self.failure_records:
                print()
                print(f"  [{test_name}]")
                print(f"    {message}")
                if expected is not None or actual is not None:
                    print(f"    expected = {expected!r}")
                    print(f"    actual   = {actual!r}")
        print("=" * SEPARATOR_WIDTH)



class _StdoutProxy:
    """
    stdout proxy used in quiet mode.

    It buffers everything written for the current test and, at each boundary
    (the next `test_block`, or the end of `run()`), either flushes that buffer
    to the real stream - when the test recorded a failure - or discards it. The
    final summary is written straight through, because `run()` restores the
    real stream before printing it.
    """

    def __init__(
        self, real_stream: Any, label: Optional[str] = None
    ) -> None:
        self._real_stream = real_stream
        self._buffer: List[str] = []
        self._current_test_failed = False
        self._label = label
        self._label_emitted = False

    def write(self, text: str) -> int:
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        self._real_stream.flush()

    def mark_failed(self) -> None:
        """Flag the in-progress test so its buffered output is kept."""
        self._current_test_failed = True

    def commit(self) -> None:
        """Flush the current test's buffer if it failed, else discard it."""
        if self._current_test_failed:
            if self._label is not None and not self._label_emitted:
                self._real_stream.write(
                    "\n" + f" {self._label} ".center(SEPARATOR_WIDTH, "=")
                    + "\n"
                )
                self._label_emitted = True
            self._real_stream.write("".join(self._buffer))
            self._real_stream.flush()

        self._buffer.clear()
        self._current_test_failed = False



REPORT = Reporter()



def section_banner(section_title: str) -> None:
    """
    Open a new test section.

    Prints a blank line, the section title as a plain comment line, and a
    full-width rule of "#" beneath it, then resets the per-section test
    counter so the next `test_block` starts again at "Test 1".
    """
    global _CURRENT_SECTION_TEST_NUMBER
    _CURRENT_SECTION_TEST_NUMBER = 0

    if not REPORT.verbose:
        return

    print()
    print(f"# {section_title} ")
    print("#" * SEPARATOR_WIDTH)



def test_block(test_name: str, description: str) -> None:
    """
    Print the one-line header that opens a single test.

    The header reads:

        Section <n>, Test <m>: s<n>.<rest> - <description>

    where <n> is the integer section number prefixing `test_name`, <m> is
    the position of this test within the current section (reset by
    `section_banner`), and <rest> is whatever follows that number. The
    VERDICT line is appended
    afterwards by whichever `check_*` helper the test calls.

    Parameters
    ----------
    test_name : str
        The dotted identifier, beginning with its section number, e.g.
        "16.edge.high_dimensional_gradient". The same string is handed to the
        `check_*` helpers, so a failure in the summary traces back here.
    description : str
        A single sentence stating what the test verifies and why it
        matters, with the checking procedure folded in.
    """
    global _CURRENT_SECTION_TEST_NUMBER

    if not REPORT.verbose and REPORT._writer is not None:
        REPORT._writer.commit()

    section_number_text, _, name_remainder = test_name.partition(".")
    section_number = int(section_number_text)

    _CURRENT_SECTION_TEST_NUMBER += 1

    numbered_name = f"s{section_number}.{name_remainder}"

    print()
    print("-" * SEPARATOR_WIDTH)
    print(
        f"Section {section_number}, Test {_CURRENT_SECTION_TEST_NUMBER}: "
        f"{numbered_name} - {description}"
    )



def _format_for_result_line(value: Any) -> str:
    """Compact textual representation tailored for the RESULT line."""
    if isinstance(value, np.ndarray):
        with np.printoptions(precision=6, suppress=False, linewidth=200):
            return np.array2string(value)

    if isinstance(value, float):
        return f"{value:.6e}"

    return repr(value)



def check_allclose(
    test_name: str,
    actual_value: Any,
    expected_value: Any,
    relative_tolerance: float,
    absolute_tolerance: float = 0.0,
    interpretation: str = "",
) -> bool:
    """
    Floating-point equality check.

    `relative_tolerance` and `absolute_tolerance` are the same parameters
    that numpy.allclose accepts. The pair is reported on the RESULT line
    so the caller can re-run the test with a different tolerance band
    without inspecting the source.

    Returns True on pass, False on fail (so the caller may use the
    return value to gate follow-up assertions).
    """

    actual_array = np.asarray(actual_value, dtype=float)
    expected_array = np.asarray(expected_value, dtype=float)

    # Element-wise absolute difference normalised by max(|expected|, eps),
    # mirroring the relative-error formula of DifferentiationResult.
    denominator = np.maximum(np.abs(expected_array), FLOAT_EPSILON)
    relative_error_per_entry = (
        np.abs(actual_array - expected_array) / denominator
    )

    max_relative_error = float(np.max(relative_error_per_entry))

    print(f"  RESULT      : actual   = {_format_for_result_line(actual_value)}")
    print(f"                expected = "
          f"{_format_for_result_line(expected_value)}")
    print(f"                max relative error = {max_relative_error:.6e}")
    print(f"                tolerance band     "
          f"= rtol={relative_tolerance:.1e}, "
          f"atol={absolute_tolerance:.1e}")

    is_close = bool(np.allclose(
        actual_array, expected_array,
        rtol=relative_tolerance, atol=absolute_tolerance,
        equal_nan=False,
    ))

    if is_close:
        message = interpretation or (
            "actual value matches the expected one within the tolerance band."
        )
        REPORT.record_pass(test_name, message)

    else:
        REPORT.record_fail(
            test_name,
            interpretation or "value disagrees beyond the tolerance band.",
            expected=expected_value, actual=actual_value,
        )

    return is_close



def check_truth(
    test_name: str,
    condition: bool,
    message_on_pass: str,
    message_on_fail: str,
) -> bool:
    """Plain boolean assertion. Used for shape / type / flag checks."""

    print(f"  RESULT      : condition evaluated to {bool(condition)}")
    if condition:
        REPORT.record_pass(test_name, message_on_pass)
        return True

    REPORT.record_fail(
        test_name, message_on_fail, expected=True, actual=bool(condition)
    )
    return False



def check_raises(
    test_name: str,
    callable_object: Callable[[], Any],
    expected_exception_type: type,
    interpretation: str = "",
) -> bool:
    """
    Assertion that calling `callable_object()` raises the given
    exception type. Both "no exception" and "wrong exception type" count
    as failures.
    """

    try:
        callable_object()

    except expected_exception_type as raised_exception:
        print(f"  RESULT      : raised {type(raised_exception).__name__}: "
              f"{raised_exception}")
        REPORT.record_pass(
            test_name,
            interpretation
            or f"call raised {expected_exception_type.__name__} as expected.",
        )
        return True

    except BaseException as wrong_exception:
        REPORT.record_fail(
            test_name,
            f"call raised {type(wrong_exception).__name__} instead of "
            f"{expected_exception_type.__name__}.",
            expected=expected_exception_type.__name__,
            actual=type(wrong_exception).__name__,
        )

        return False

    REPORT.record_fail(
        test_name,
        f"call returned normally; expected {expected_exception_type.__name__}.",
        expected=expected_exception_type.__name__,
        actual="<no exception>",
    )

    return False



def check_warns(
    test_name: str,
    callable_object: Callable[[], Any],
    expected_warning_category: type,
    interpretation: str = "",
) -> bool:
    """
    Assertion that calling `callable_object()` emits at least one warning of
    the expected category. The function's return value is captured but ignored:
    only the warning channel is inspected.
    """

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        callable_object()

    category_matches = [
        warning_record for warning_record in captured_warnings
        if issubclass(warning_record.category, expected_warning_category)
    ]

    print(f"  RESULT      : {len(captured_warnings)} warning(s) captured; "
          f"{len(category_matches)} of category "
          f"{expected_warning_category.__name__}.")

    if category_matches:
        REPORT.record_pass(
            test_name,
            interpretation
            or f"{expected_warning_category.__name__} emitted as expected.",
        )

        return True

    REPORT.record_fail(
        test_name,
        f"no {expected_warning_category.__name__} captured.",
        expected=expected_warning_category.__name__,
        actual=[w.category.__name__ for w in captured_warnings] or None,
    )

    return False



# CALL COUNTING WRAPPER



class CallCountingFunction:
    """
    Wraps a callable and counts how many times it is invoked.

    Used by the benchmarking and the short-circuit tests. The wrapper
    is transparent: it forwards every argument to the inner function
    and returns the inner result unchanged.

    Attributes
    ----------
    inner_function : Callable
        The wrapped function.
    call_count : int
        Number of invocations since the last `reset_count()` (or since
        construction).
    """

    __slots__ = ("inner_function", "call_count")

    def __init__(self, inner_function: Callable) -> None:
        self.inner_function = inner_function
        self.call_count = 0

    def __call__(self, *positional_arguments, **keyword_arguments):
        self.call_count += 1
        return self.inner_function(*positional_arguments, **keyword_arguments)

    def reset_count(self) -> None:
        self.call_count = 0



# SECTION 1 - KNOWN-VALUE CORRECTNESS



# The fields are: a name, the function under test, the analytical value, the
# evaluation point, the derivative order, and the `single_component` selector
# (None means "compute the full tensor and then index it").
KNOWN_VALUE_TEST_CASES: List[Tuple[
    str, Callable[[np.ndarray], float], float, np.ndarray, int,
    Optional[Tuple[int, ...]],
]] = [
    # one-variable, first derivative
    ("d/dx exp(x) at x=1",
        lambda x: np.exp(x[0]), float(np.e),
        np.array([1.0]), 1, (0,)),

    ("d/dx exp(x) at x=5",
        lambda x: np.exp(x[0]), float(np.exp(5.0)),
        np.array([5.0]), 1, (0,)),

    ("d/dx sin(x) at x=1",
        lambda x: np.sin(x[0]), float(np.cos(1.0)),
        np.array([1.0]), 1, (0,)),

    ("d/dx sin(10x) at x=0.1 (high-frequency: stresses step selection)",
        lambda x: np.sin(10.0 * x[0]), 10.0 * float(np.cos(10.0 * 0.1)),
        np.array([0.1]), 1, (0,)),

    ("d/dx log(x) at x=2",
        lambda x: np.log(x[0]), 0.5,
        np.array([2.0]), 1, (0,)),

    ("d/dx tanh(x) at x=0.5",
        lambda x: np.tanh(x[0]), 1.0 - float(np.tanh(0.5)) ** 2,
        np.array([0.5]), 1, (0,)),


    # one-variable, higher-order derivatives
    ("d^2/dx^2 exp(x) at x=1 (second derivative)",
        lambda x: np.exp(x[0]), float(np.e),
        np.array([1.0]), 2, (0, 0)),

    ("d^2/dx^2 x^5 at x=1.5 (= 20 * 1.5^3 = 67.5)",
        lambda x: x[0] ** 5, 20.0 * 1.5 ** 3,
        np.array([1.5]), 2, (0, 0)),

    ("d^4/dx^4 exp(x) at x=0 (= 1, higher-order stress)",
        lambda x: np.exp(x[0]), 1.0,
        np.array([0.0]), 4, (0, 0, 0, 0)),

    ("d^6/dx^6 sin(x) at x=0 (= 0, alternating signs)",
        lambda x: np.sin(x[0]), 0.0,
        np.array([0.0]), 6, (0, 0, 0, 0, 0, 0)),

    # multivariable, single component
    ("df/dx_1 of x_0^4 + x_1 + x_2 at (2, 3, 4) = 1",
        lambda x: x[0] ** 4 + x[1] + x[2], 1.0,
        np.array([2.0, 3.0, 4.0]), 1, (1,)),

    ("df/dx_0 of x_0^4 + x_1 + x_2 at (2, 3, 4) = 32",
        lambda x: x[0] ** 4 + x[1] + x[2], 32.0,
        np.array([2.0, 3.0, 4.0]), 1, (0,)),

    # multivariable, mixed partials
    ("d^2/dx_0 dx_1 of x_0 * x_1 at (2, 3) = 1 (canonical cross-term)",
        lambda x: x[0] * x[1], 1.0,
        np.array([2.0, 3.0]), 2, (0, 1)),

    ("d^2/dx_0 dx_1 of sin(x_0) * cos(x_1) at (1, 0.5)",
        lambda x: np.sin(x[0]) * np.cos(x[1]),
        -float(np.cos(1.0)) * float(np.sin(0.5)),
        np.array([1.0, 0.5]), 2, (0, 1)),

    ("d^3/dx_0 dx_0 dx_1 of x_0^2 * x_1 at (3, 7) = 2 (verify mixed third)",
        lambda x: x[0] ** 2 * x[1], 2.0,
        np.array([3.0, 7.0]), 3, (0, 0, 1)),
]



def section_01_known_values() -> None:
    """
    Run every entry in ***KNOWN_VALUE_TEST_CASES*** through every
    available numerical strategy that the case admits:

    - The complex-step method is invoked only for first-order
    derivatives, because it is mathematically restricted to that order.
    - The plain central-difference path is invoked for every case at
    the default `point_number` ('auto').
    - The Richardson path is invoked for every case at its 'auto'
    base (`derivative_order` + 1).

    Each strategy is granted a tolerance band matched to its
    theoretical accuracy:

    - complex-step      :  rtol = 1e-13   (single subtractive-cancellation-free
                                          step away from machine precision).
    - richardson        :  rtol = 1e-9 for derivative order <= 3, widened
                          to 1e-7 for order >= 4 (a high-order derivative
                          carries a higher truncation+roundoff floor).
    - central-difference:  rtol = 1e-6    (default 'auto' stencil; a
                                          comfortable margin above the
                                          truncation+roundoff floor).
    """

    section_banner("SECTION 1 - Known-value correctness across all methods")

    for (case_name, function_to_differentiate, analytical_value,
            evaluation_point, derivative_order, single_component
        ) in KNOWN_VALUE_TEST_CASES:

        # Plain central differences
        test_block(
            test_name=f"1.central : {case_name}",
            description=(
                f"verify the order-{derivative_order} derivative from the "
                f"default central-difference path matches the closed-form "
                f"value, evaluated at {evaluation_point.tolist()} with "
                f"single_component={single_component}; this is the broadest "
                f"entry point every user starts from, so a regression here "
                f"would surface at once on ordinary textbook problems."
            ),
        )

        central_difference_result = nth_numerical_derivative(
            function_to_differentiate,
            derivative_order=derivative_order,
            single_component=single_component,
        )(evaluation_point)

        check_allclose(
            test_name=f"1.central : {case_name}",
            actual_value=float(central_difference_result.as_float()),
            expected_value=analytical_value,
            relative_tolerance=1e-6, absolute_tolerance=1e-9,
            interpretation=(
                "central-difference value within the expected accuracy band "
                "for the 'auto' stencil."
            ),
        )


        # Richardson extrapolation
        test_block(
            test_name=f"1.richardson : {case_name}",
            description=(
                "verify the same derivative refined by Richardson extrapolation"
                " gains accuracy and still lands on the right value, repeating "
                "the call with richardson_extrapolation=True and comparing at "
                "a tighter tolerance; Richardson is the middle cost/accuracy "
                "tier, the one users reach for when they want extra digits "
                "without paying for a wider stencil."
            ),
        )

        richardson_result = nth_numerical_derivative(
            function_to_differentiate,
            derivative_order=derivative_order,
            single_component=single_component,
            richardson_extrapolation=True,
        )(evaluation_point)

        # Richardson's achievable accuracy degrades with derivative
        # order: a fourth or higher derivative carries a higher
        # truncation+roundoff floor than a first or second, so the band
        # is widened for the high-order cases.
        if derivative_order <= 3:
            richardson_rtol, richardson_atol = 1e-9, 1e-10
        else:
            richardson_rtol, richardson_atol = 1e-7, 1e-8

        check_allclose(
            test_name=f"1.richardson : {case_name}",
            actual_value=float(richardson_result.as_float()),
            expected_value=analytical_value,
            relative_tolerance=richardson_rtol,
            absolute_tolerance=richardson_atol,
            interpretation=(
                "Richardson value lies inside the tighter band, as expected "
                "after at least one extrapolation pass."
            ),
        )


        # Complex-step (first derivatives only)
        if derivative_order == 1:
            test_block(
                test_name=f"1.complex : {case_name}",
                description=(
                    "verify the same first derivative from the "
                    "complex-step method reaches essentially machine "
                    "precision, repeating the call with "
                    "step_size='complex' and comparing at a "
                    "machine-precision tolerance; with no "
                    "subtractive cancellation, complex-step is the "
                    "gold-standard reference wherever it applies "
                    "(analytic, first order)."
                ),
            )

            complex_step_result = nth_numerical_derivative(
                function_to_differentiate,
                derivative_order=1,
                step_size='complex',
                single_component=single_component,
            )(evaluation_point)

            check_allclose(
                test_name=f"1.complex : {case_name}",
                actual_value=float(complex_step_result.as_float()),
                expected_value=analytical_value,
                relative_tolerance=1e-13, absolute_tolerance=1e-14,
                interpretation=(
                    "complex-step value matches the analytical answer to "
                    "essentially machine precision."
                ),
            )



# SECTION 2 - POLYNOMIAL CANCELLATION
#
# For a polynomial p(x) of degree d, the (d+1)-th derivative is exactly zero.
# Any non-zero value produced by a numerical scheme is roundoff and must lie
# within a small multiple of FLOAT_EPSILON.
#
# This is a qualitative test that exposes catastrophic cancellation more than
# any tolerance, because the expected value is exactly zero and the only thing
# the scheme can produce is its own noise floor.



def section_02_polynomial_cancellation() -> None:
    """
    For each polynomial degree from 1 to 5, check that the (degree + 1)-th
    derivative at a non-trivial point is zero up to a small absolute
    tolerance.

    Independent property of the symbolic class of polynomials: it is the
    one place in the API where the "expected error" is not a tolerance
    band around a non-zero value but a direct measurement of the
    scheme's noise floor.
    """

    section_banner("SECTION 2 - Polynomial cancellation")

    for polynomial_degree in range(1, 6):
        polynomial_coefficients = np.array(
            [(-1) ** k * (k + 1) for k in range(polynomial_degree + 1)],
            dtype=float,
        )

        def polynomial_function(
            x: np.ndarray,
            coefficients: np.ndarray = polynomial_coefficients,
        ) -> float:
            return float(
                sum(
                    coefficient * x[0] ** power
                    for power, coefficient in enumerate(coefficients)
                )
            )

        test_block(
            test_name=f"2.degree_{polynomial_degree}",
            description=(
                f"verify the order-{polynomial_degree + 1} derivative of a "
                f"degree-{polynomial_degree} polynomial comes out numerically "
                f"zero, evaluated at x = 1.7 with the default 'auto' "
                f"configuration and compared to zero on an absolute-only "
                f"tolerance; polynomial cancellation is the cleanest probe of "
                f"the subtractive-cancellation noise floor, since the exact "
                f"answer is zero and anything non-zero is pure roundoff."
            ),
        )

        derivative_above_degree = nth_numerical_derivative(
            polynomial_function,
            derivative_order=polynomial_degree + 1,
            single_component=(0,) * (polynomial_degree + 1),
        )(np.array([1.7]))


        # An absolute tolerance is the right shape of bound here: the true
        # value is zero, so a relative tolerance would be undefined.
        check_truth(
            test_name=f"2.degree_{polynomial_degree}",
            condition=abs(float(derivative_above_degree.as_float())) < 1e-4,
            message_on_pass=(
                f"|d^{polynomial_degree + 1} p / dx^{polynomial_degree + 1}|"
                f" = {abs(float(derivative_above_degree.as_float())):.3e} "
                f"is within the noise floor as expected."
            ),
            message_on_fail=(
                f"|d^{polynomial_degree + 1} p / dx^{polynomial_degree + 1}|"
                f" = {abs(float(derivative_above_degree.as_float())):.3e} "
                f"exceeds the noise floor; subtractive cancellation has "
                f"contaminated the result more than acceptable."
            ),
        )



# SECTION 3 - SCHWARZ SYMMETRY OF MIXED PARTIALS
#
# The Schwarz's theorem states that for any function whose mixed partials are
# continuous, the order of differentiation can be permuted without changing the
# value. This implementation inherently applies it for the full tensor and the
# symmetric-stencil central-difference formula it builds; so check that the
# implementation preserves it across permutations of the index tuple.



def section_03_schwarz_symmetry() -> None:
    """
    For three multivariate functions of growing nonlinearity, verify
    that every permutation of the `single_component` index tuple
    produces the same value (up to a small relative tolerance).

    Stress-tests the index-permutation handling inside the dispatcher:
    the implementation uses Counter() to canonicalise tuples for
    caching, so a bug in the canonical-key construction would surface
    here as an asymmetry that has no mathematical justification.
    """

    section_banner("SECTION 3 - Schwarz symmetry of mixed partials")

    symmetric_test_functions = [
        ("f(x, y, z) = x*y + y*z + x*z (bilinear)",
            lambda x: x[0] * x[1] + x[1] * x[2] + x[0] * x[2],
            np.array([1.7, -0.4, 2.1]),
            (0, 1, 2),
        ),
        ("f(x, y, z) = sin(x) * cos(y) * exp(z)",
            lambda x: np.sin(x[0]) * np.cos(x[1]) * np.exp(x[2]),
            np.array([0.3, 1.1, 0.5]),
            (0, 1, 2),
        ),
        ("f(x, y) = (x^2 + y^2)^(3/2)",
            lambda x: (x[0] ** 2 + x[1] ** 2) ** 1.5,
            np.array([1.3, 0.7]),
            (0, 0, 1, 1),
        ),
    ]

    from itertools import permutations

    for case_name, function_to_differentiate, evaluation_point, index_tuple in (
        symmetric_test_functions
    ):
        derivative_order = len(index_tuple)
        unique_permutations = sorted(set(permutations(index_tuple)))

        test_block(
            test_name=f"3.symmetry : {case_name}",
            description=(
                f"verify every permutation of the {derivative_order}-index "
                f"tuple {index_tuple} yields the same numerical value, "
                f"computing the partial at {evaluation_point.tolist()} for "
                f"each of the {len(unique_permutations)} unique permutations "
                f"and reporting the max pairwise difference; Schwarz's theorem "
                f"is what justifies the Counter()-based canonical cache key "
                f"the dispatcher uses, so an asymmetry here would mean that "
                f"key is wrong."
            ),
        )

        permutation_values = []
        for permuted_tuple in unique_permutations:
            value = float(
                nth_numerical_derivative(
                    function_to_differentiate,
                    derivative_order=derivative_order,
                    single_component=permuted_tuple,
                )(evaluation_point).as_float()
            )

            permutation_values.append(value)

        permutation_array = np.array(permutation_values, dtype=float)
        max_pairwise_spread = float(
            np.max(permutation_array) - np.min(permutation_array)
        )
        reference_magnitude = max(np.abs(permutation_array).max(), 1.0)

        relative_spread = max_pairwise_spread / reference_magnitude

        check_truth(
            test_name=f"3.symmetry : {case_name}",
            condition=relative_spread < 1e-6,
            message_on_pass=(
                f"every permutation agrees within "
                f"{relative_spread:.2e} relative; Schwarz preserved."
            ),
            message_on_fail=(
                f"permutations disagree by {relative_spread:.2e} relative, "
                f"more than the 1e-6 tolerance for symmetric stencils."
            ),
        )



# SECTION 4 - ORDER OF ACCURACY VIA SLOPE FITTING
#
# A central-difference stencil of effective point count p has truncation
# error O(h^(p - n + 1)) for an n-th derivative. Verify the observed
# exponent by computing the derivative at a geometric sequence of step
# sizes and fitting a line through log(|error|) vs log(h). The slope must
# match the theoretical accuracy within a small tolerance.
#
# This is how one would prove on a regression that a particular stencil really
# delivers its theoretical accuracy.



def section_04_order_of_accuracy() -> None:
    """
    For each (`derivative_order`, `point_number`) pair, evaluate the
    derivative at a sequence of geometrically shrinking step sizes,
    linearly fit log(error) ~ slope * log(h) + intercept, and check the
    measured slope against the theoretical truncation order.

    Catches subtle stencil-construction bugs that pure point-tests miss:
    a wrong coefficient sign that happens to produce a numerically close
    answer at a specific point still ruins the convergence slope.
    """

    section_banner("SECTION 4 - Order of accuracy via slope fitting")

    # target_function and its analytical fourth derivative.
    def target_function(x: np.ndarray) -> float:
        return float(np.exp(2.0 * x[0]))

    analytical_derivative_table = {
        1: 2.0 * float(np.exp(2.0 * 0.3)),
        2: 4.0 * float(np.exp(2.0 * 0.3)),
        3: 8.0 * float(np.exp(2.0 * 0.3)),
    }

    convergence_test_configurations = [
        # (derivative_order, point_number, theoretical_slope, tolerance)
        (1, 2, 2.0, 0.25),    # central 2-point first derivative: O(h^2)
        (1, 4, 4.0, 0.30),    # central 4-point first derivative: O(h^4)
        (2, 3, 2.0, 0.25),    # central 3-point second derivative: O(h^2)
        (2, 5, 4.0, 0.30),    # central 5-point second derivative: O(h^4)
        (3, 4, 2.0, 0.30),    # central 4-point third derivative: O(h^2)
    ]

    for (derivative_order, point_number, theoretical_slope, slope_tolerance
        ) in convergence_test_configurations:

        # Geometric space, well inside the truncation-dominated regime: too
        # small and roundoff dominates; too large and the leading-order term
        # is no longer the only relevant one.
        step_size_grid = np.geomspace(1e0, 1e-2, num=7)

        observed_errors = []
        for current_step_size in step_size_grid:
            numerical_result = nth_numerical_derivative(
                target_function,
                derivative_order=derivative_order,
                point_number=point_number,
                step_size=float(current_step_size),
                single_component=(0,) * derivative_order,
            )(np.array([0.3]))

            observed_errors.append(abs(
                float(numerical_result.as_float())
                - analytical_derivative_table[derivative_order]
            ))

        observed_errors = np.array(observed_errors, dtype=float)

        # If any reading is exactly zero or denormal it would break the log,
        # so a floor is applied.
        observed_errors = np.maximum(observed_errors, 1e-300)

        # Linear fit on log-log axes: slope is what we are after.
        fit_slope, _ = np.polyfit(
            np.log(step_size_grid), np.log(observed_errors), 1,
        )


        test_block(
            test_name=f"4.order_n{derivative_order}_p{point_number}",
            description=(
                f"verify the observed convergence slope of the order-"
                f"{derivative_order} derivative on a {point_number}-point "
                f"stencil matches the theoretical exponent "
                f"{theoretical_slope:g}, fitting log(|error|) against log(h) "
                f"over 7 geometrically shrinking steps between 1. and 1e-2; "
                f"the log-log slope is the standard published criterion, since "
                f"a correct stencil must deliver the advertised asymptotic "
                f"order, not merely a low error at a single step size."
            ),
        )

        print(f"  RESULT      : measured slope = {fit_slope:.4f} "
              f"(theoretical = {theoretical_slope:g}, "
              f"tolerance band = +/- {slope_tolerance:g})")

        slope_difference = abs(fit_slope - theoretical_slope)

        check_truth(
            test_name=(
                f"4.order_n{derivative_order}_p{point_number}"
            ),
            condition=slope_difference < slope_tolerance,
            message_on_pass=(
                f"slope deviates by {slope_difference:.3f}, inside the band."
            ),
            message_on_fail=(
                f"slope deviates by {slope_difference:.3f}, outside the "
                f"+/- {slope_tolerance} tolerance - likely stencil bug."
            ),
        )



# SECTION 5 - THREE-WAY CROSS-VALIDATION
#
# The three numerical strategies should agree on a problem they all support
# (first derivative of an analytic function).



def section_05_three_way_cross_validation() -> None:
    """
    On a panel of analytic single-variable functions, compute the first
    derivative via central-difference, Richardson extrapolation, and
    complex-step, and check pairwise agreement to a tight tolerance.

    Failure on one of these pairings localises the bug to one of the
    three implementations, since the third leg of the triangle keeps
    the reference honest.
    """

    section_banner("SECTION 5 - Three-way cross-validation (first derivative)")

    analytic_single_variable_panel = [
        ("exp(x) at x = 0.7", lambda x: np.exp(x[0]), 0.7),
        ("sin(2*x + 1) at x = 0.4",
            lambda x: np.sin(2.0 * x[0] + 1.0), 0.4),
        ("x^3 / (1 + x^2) at x = 1.3",
            lambda x: x[0] ** 3 / (1.0 + x[0] ** 2), 1.3),
        ("tanh(0.5 * x) at x = -0.6",
            lambda x: np.tanh(0.5 * x[0]), -0.6),
    ]

    for case_name, function_under_test, evaluation_coordinate in (
        analytic_single_variable_panel
    ):
        evaluation_point = np.array([evaluation_coordinate])

        central_value = float(
            nth_numerical_derivative(
                function_under_test, derivative_order=1,
            )(evaluation_point).as_float()
        )

        richardson_value = float(
            nth_numerical_derivative(
                function_under_test, derivative_order=1,
                richardson_extrapolation=True,
            )(evaluation_point).as_float()
        )

        complex_step_value = float(
                nth_numerical_derivative(
                function_under_test, derivative_order=1, step_size='complex',
            )(evaluation_point).as_float()
        )


        test_block(
            test_name=f"5.cross : {case_name}",
            description=(
                "verify the three numerical strategies agree on the first "
                "derivative of an analytic function, computing it by each and "
                "reporting the max pairwise difference; this mutual cross- "
                "validation pins any bug to a single path, with complex-step "
                "as the de-facto ground truth, Richardson as the high-accuracy "
                "reference, and central-difference as the baseline."
            ),
        )

        derivative_estimates = np.array(
            [central_value, richardson_value, complex_step_value],
            dtype=float,
        )
        max_pairwise_difference = float(
            np.max(derivative_estimates) - np.min(derivative_estimates)
        )
        reference_magnitude = max(np.abs(derivative_estimates).max(), 1.0)

        max_pairwise_relative = max_pairwise_difference / reference_magnitude

        print(f"  RESULT      : central    = {central_value:.12e}")
        print(f"                richardson = {richardson_value:.12e}")
        print(f"                complex    = {complex_step_value:.12e}")
        print(f"                max pairwise relative = "
              f"{max_pairwise_relative:.3e}")

        check_truth(
            test_name=f"5.cross : {case_name}",
            condition=max_pairwise_relative < 1e-7,
            message_on_pass=(
                f"three strategies agree within {max_pairwise_relative:.2e} "
                f"relative."
            ),
            message_on_fail=(
                f"three strategies disagree by {max_pairwise_relative:.2e} "
                f"relative, more than 1e-7; one path is suspect."
            ),
        )



# SECTION 6 - RICHARDSON IMPROVES ACCURACY



def section_06_richardson_improves() -> None:
    """
    For a smooth analytic test function, check that Richardson
    extrapolation on the minimal base stencil produces an answer at
    least one decimal digit closer to the analytical value than the
    base stencil alone.

    The chosen base stencil is `point_number` = `derivative_order` + 1
    (Richardson's preferred base, per the module's 'auto' policy).
    """

    section_banner("SECTION 6 - Richardson narrows the error")

    def analytic_test_function(x: np.ndarray) -> float:
        return float(np.exp(x[0]) * np.sin(x[1]))

    evaluation_point = np.array([0.4, 1.1])

    # df/dx of exp(x)*sin(y) = exp(x)*sin(y); evaluated at the point above.
    analytical_first_partial = (
        float(np.exp(0.4)) * float(np.sin(1.1))
    )

    base_central_value = float(
            nth_numerical_derivative(
            analytic_test_function,
            derivative_order=1, point_number=2, single_component=(0,),
        )(evaluation_point).as_float()
    )

    richardson_value_result = nth_numerical_derivative(
        analytic_test_function,
        derivative_order=1, point_number=2,
        richardson_extrapolation=True,
        single_component=(0,),
    )(evaluation_point)

    richardson_value = float(richardson_value_result.as_float())

    base_error = abs(base_central_value - analytical_first_partial)
    richardson_error = abs(richardson_value - analytical_first_partial)


    test_block(
        test_name="6.improvement",
        description=(
            "verify Richardson extrapolation cuts the base stencil's error by "
            "at least a factor of ten, computing df/dx of exp(x)*sin(y) at "
            "(0.4, 1.1) with the bare 2-point stencil and again with the "
            "2-point-plus-Richardson refinement and reporting the ratio; the "
            "reduction factor is the most direct evidence the extrapolation "
            "table is doing real work rather than being short-circuited or "
            "built on wrong coefficients."
        ),
    )

    print(f"  RESULT      : base error       = {base_error:.6e}")
    print(f"                richardson error = {richardson_error:.6e}")
    print(f"                reported error_estimate = "
          f"{richardson_value_result.error_estimate}")
    print(f"                reduction factor = "
          f"{base_error / max(richardson_error, 1e-300):.1f}x")

    check_truth(
        test_name="6.improvement",
        condition=richardson_error * 10.0 < base_error,
        message_on_pass=(
            f"richardson reduces the error by "
            f"{base_error / max(richardson_error, 1e-300):.1f}x, well over "
            f"the required 10x."
        ),
        message_on_fail=(
            f"richardson reduced the error by only "
            f"{base_error / max(richardson_error, 1e-300):.1f}x, less than "
            f"the expected 10x improvement."
        ),
    )



# SECTION 7 - ERROR ESTIMATE COVERS TRUE ERROR
#
# Richardson extrapolation returns a Romberg-style upper bound. For it to be
# trustworthy, the bound must in fact bound the observed error against the
# analytical value.



def section_07_error_estimate_reliability() -> None:
    """
    On a small panel of functions with known derivatives, compare the
    `error_estimate` reported by Richardson against the actual error
    against the analytical answer.

    The reported bound is required to either cover the true error
    outright, or at most be off by a small multiplicative factor (the
    module documents this as a Romberg-style upper bound, not a
    rigorous interval).
    """

    section_banner("SECTION 7 - Reported error estimate covers true error")

    error_estimate_panel = [
        ("d/dx exp(x) at 1.2",
            lambda x: np.exp(x[0]), float(np.exp(1.2)),
            np.array([1.2]), 1, (0,)),
        ("d^2/dx^2 sin(x) at 0.7",
            lambda x: np.sin(x[0]), -float(np.sin(0.7)),
            np.array([0.7]), 2, (0, 0)),
        ("d/dx_0 of x_0^2 + x_1^3 at (1, 1)",
            lambda x: x[0] ** 2 + x[1] ** 3, 2.0,
            np.array([1.0, 1.0]), 1, (0,)),
    ]

    # The reported bound is Romberg-style, so allow a safety multiplier; even
    # so a wildly under-estimating bound (e.g. 1e-20 when the true error is
    # 1e-9) would still fail.
    safety_multiplier = 50.0

    for (case_name, function_under_test, analytical_value,
            evaluation_point, derivative_order, single_component
        ) in error_estimate_panel:

        richardson_result = nth_numerical_derivative(
            function_under_test,
            derivative_order=derivative_order,
            single_component=single_component,
            point_number=derivative_order+3,
            richardson_extrapolation=True,
        )(evaluation_point)

        observed_true_error = abs(
            float(richardson_result.as_float()) - analytical_value
        )
        reported_error_estimate = float(richardson_result.error_estimate)


        test_block(
            test_name=f"7.estimate : {case_name}",
            description=(
                "verify the error_estimate Richardson reports is at least as "
                "large as the true error, up to a small safety factor, by "
                "reading error_estimate off a Richardson-enabled result and "
                "comparing it to the observed error against the analytical "
                "value; a caller reading that bound expects it to be honest, "
                "and under-reporting would mislead anything that treats it as "
                "a confidence interval, such as an adaptive optimizer gating "
                "on accuracy."
            ),
        )

        print(f"  RESULT      : true error       = "
              f"{observed_true_error:.3e}")
        print(f"                error_estimate   = "
              f"{reported_error_estimate:.3e}")
        print(f"                safety multiplier allowed = "
              f"{safety_multiplier:.0f}")

        # The bound is allowed to be slightly tight (off by safety_multiplier)
        # or substantially loose, but not wildly optimistic.
        check_truth(
            test_name=f"7.estimate : {case_name}",
            condition=(
                reported_error_estimate * safety_multiplier
                >= observed_true_error
            ),
            message_on_pass=(
                "reported error_estimate is consistent with the true error."
            ),
            message_on_fail=(
                f"reported error_estimate of {reported_error_estimate:.3e} "
                f"is more than {safety_multiplier}x smaller than the true "
                f"error of {observed_true_error:.3e}; the bound is not "
                f"honest."
            ),
        )



# SECTION 8 - INPUT/OUTPUT SHAPE VALIDATION
#
# As indicated in ***nth_numerical_derivative*** the input dimension must be
# 0-D, 1-D or 2-D and `single_component`'s length must match it.



def section_08_shape_validation() -> None:
    """
    Walk through the seven input/output shape combinations documented
    in the public docstring and verify each one. The configurations
    cover scalar (0-D) inputs, 1-D inputs at several dimensions, 2-D
    batches, and the `single_component` selector.
    """

    section_banner("SECTION 8 - Input/output shape")

    # Scalar input for a one-variable function. According to the docstring,
    # a 0-D input is normalised to a 1-D point of length 1; the derivative
    # comes back as a 0-D scalar.
    one_variable_function = lambda x: x[0] ** 2

    test_block(
        test_name="8.shape.0d_input",
        description=(
            "verify a 0-D input gives back a scalar derivative, evaluating "
            "d/dx of f(x) = x^2 at the scalar input 3.0 and checking the "
            "result reports is_scalar=True; someone differentiating a "
            "one-variable function with a plain float expects a plain float "
            "back, not a shape-(1,) array."
        ),
    )

    zero_dim_result = nth_numerical_derivative(
        one_variable_function, derivative_order=1,
    )(np.array(3.0))

    check_truth(
        test_name="8.shape.0d_input",
        condition=zero_dim_result.is_scalar and zero_dim_result.shape == (),
        message_on_pass="0-D input yields is_scalar=True and shape=().",
        message_on_fail=(
            f"got is_scalar={zero_dim_result.is_scalar}, "
            f"shape={zero_dim_result.shape}"
        ),
    )


    # 1-D input, D = 3, full tensor. The first derivative tensor has shape (3,).
    three_variable_function = lambda x: x[0] + x[1] ** 2 + x[2] ** 3

    test_block(
        test_name="8.shape.1d_input_full_gradient",
        description=(
            "verify a 1-D input of length D yields a derivative of shape (D,), "
            "computing the gradient of x_0 + x_1^2 + x_2^3 at (1, 2, 3); this "
            "is the everyday gradient call, and downstream optimizers expect a "
            "1-D vector of partials."
        ),
    )
    one_dim_full_gradient = nth_numerical_derivative(
        three_variable_function, derivative_order=1,
    )(np.array([1.0, 2.0, 3.0]))
    check_truth(
        test_name="8.shape.1d_input_full_gradient",
        condition=one_dim_full_gradient.shape == (3,),
        message_on_pass="gradient shape is (3,) as documented.",
        message_on_fail=(
            f"expected (3,), got {one_dim_full_gradient.shape}"
        ),
    )


    # 1-D input, D = 3, full Hessian. The second derivative tensor is (3, 3).
    test_block(
        test_name="8.shape.1d_input_full_hessian",
        description=(
            "verify the full Hessian comes back with shape (D, D), computing "
            "it for the same function at the same point; Newton-style "
            "optimizers consume the Hessian as a square matrix, and the API "
            "guarantees exactly that shape."
        ),
    )
    one_dim_full_hessian = nth_numerical_derivative(
        three_variable_function, derivative_order=2,
    )(np.array([1.0, 2.0, 3.0]))
    check_truth(
        test_name="8.shape.1d_input_full_hessian",
        condition=one_dim_full_hessian.shape == (3, 3),
        message_on_pass="Hessian shape is (3, 3) as documented.",
        message_on_fail=(
            f"expected (3, 3), got {one_dim_full_hessian.shape}"
        ),
    )


    # 1-D input + single_component scalar. Result is a scalar.
    test_block(
        test_name="8.shape.1d_input_single_component",
        description=(
            "verify single_component collapses the output to a scalar, "
            "computing df/dx_1 of the same function with "
            "single_component=(1,); when the caller needs only one tensor "
            "entry, the API must not pay for or return the full tensor."
        ),
    )
    one_dim_component_result = nth_numerical_derivative(
        three_variable_function, derivative_order=1, single_component=(1,),
    )(np.array([1.0, 2.0, 3.0]))
    check_truth(
        test_name="8.shape.1d_input_single_component",
        condition=one_dim_component_result.is_scalar,
        message_on_pass="single_component result is scalar.",
        message_on_fail=(
            f"expected scalar, got shape={one_dim_component_result.shape}"
        ),
    )


    # 2-D batch input, full tensor. (M, D) -> (M, D, D, ...).
    batch_evaluation_points = np.array(
        [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [-1.0, 1.0, 0.0]]
    )

    test_block(
        test_name="8.shape.2d_batch_full",
        description=(
            "verify a batch of M points adds a leading M axis to the tensor, "
            "computing the Hessian at three points stacked into an (M=3, D=3) "
            "array; vectorized evaluation is the standard way to evaluate a "
            "model at many candidate parameter vectors at once."
        ),
    )

    batch_hessian = nth_numerical_derivative(
        three_variable_function, derivative_order=2,
    )(batch_evaluation_points)

    check_truth(
        test_name="8.shape.2d_batch_full",
        condition=batch_hessian.shape == (3, 3, 3),
        message_on_pass="batch Hessian shape is (M, D, D) = (3, 3, 3).",
        message_on_fail=(
            f"expected (3, 3, 3), got {batch_hessian.shape}"
        ),
    )


    # 2-D batch input + single_component. (M, D) -> (M,).
    test_block(
        test_name="8.shape.2d_batch_single_component",
        description=(
            "verify a batch combined with single_component yields a 1-D array "
            "of length M, taking the single-component derivative across the "
            "same three-point batch; this is the common pattern of tracking "
            "one specific partial along a trajectory of points, for monitoring "
            "or plotting."
        ),
    )
    batch_component_result = nth_numerical_derivative(
        three_variable_function, derivative_order=1, single_component=(2,),
    )(batch_evaluation_points)
    check_truth(
        test_name="8.shape.2d_batch_single_component",
        condition=batch_component_result.shape == (3,),
        message_on_pass="batch single-component shape is (M,) = (3,).",
        message_on_fail=(
            f"expected (3,), got {batch_component_result.shape}"
        ),
    )


    # 3-D input is documented as an error.
    test_block(
        test_name="8.shape.invalid_3d_input",
        description=(
            "verify a 3-D input is rejected with a ValueError at eval time, "
            "evaluating the gradient on a (2, 2, 3) array and catching the "
            "error; the inputs shapes covered are only 0-D, 1-D, and 2-D, "
            "so anything beyond that must fail loudly."
        ),
    )
    invalid_three_dim_callable = nth_numerical_derivative(
        three_variable_function, derivative_order=1,
    )
    check_raises(
        test_name="8.shape.invalid_3d_input",
        callable_object=lambda: invalid_three_dim_callable(np.zeros((2, 2, 3))),
        expected_exception_type=ValueError,
        interpretation="3-D input correctly rejected.",
    )



# SECTION 9 - BATCH IS LOOP
#
# Evaluating the derivative at M points in one batched call must produce
# values that are pointwise identical to M individual one-point calls.



def section_09_batch_equals_loop() -> None:
    """
    Pick a non-trivial multivariable function, evaluate its full Hessian
    on a batch of M = 5 points in one call, then again on the same M
    points one-at-a-time. Compare the two tensors entry by entry.
    """

    section_banner("SECTION 9 - Batch evaluation matches the per-point loop")

    def quartic_test_function(x: np.ndarray) -> float:
        return float(
            x[0] ** 4 + x[1] ** 2 + x[0] * x[1] + np.exp(x[2])
        )

    batch_points = np.array([
        [0.5, 1.0, 0.0],
        [1.0, -1.0, 0.5],
        [-0.5, 0.5, -0.3],
        [2.0, 0.1, 1.0],
        [0.0, 2.0, -0.2],
    ])

    hessian_callable = nth_numerical_derivative(
        quartic_test_function, derivative_order=2,
    )

    batched_hessians = np.asarray(hessian_callable(batch_points))
    looped_hessians = np.stack(
        [np.asarray(hessian_callable(point)) for point in batch_points],
    )

    test_block(
        test_name="9.batch_equals_loop",
        description=(
            "verify batch evaluation at M points returns the same tensor as "
            "stacking M individual evaluations, computing Hessians of a "
            "quartic at 5 points both ways and comparing element-wise; hidden "
            "mutable state in the inner loop would show up here as an "
            "off-by-one or accumulated drift, far easier to catch now than to "
            "chase later in a long pipeline."
        ),
    )

    max_absolute_difference = float(
        np.max(np.abs(batched_hessians - looped_hessians))
    )

    print(f"  RESULT      : batch shape    = {batched_hessians.shape}")
    print(f"                loop  shape    = {looped_hessians.shape}")
    print(f"                max |diff|     = {max_absolute_difference:.3e}")

    check_truth(
        test_name="9.batch_equals_loop",
        # The two paths should be byte-identical in practice (both go through
        # the same dispatcher with the same step sizes).
        condition=max_absolute_difference == 0.0,
        message_on_pass="batch and loop produce bitwise identical tensors.",
        message_on_fail=(
            f"batch and loop disagree by up to "
            f"{max_absolute_difference:.3e} - state leak suspected."
        ),
    )



# SECTION 10 - SINGLE-COMPONENT MATCHES FULL-TENSOR INDEXING
#
# The single_component path skips the full tensor and computes only the
# requested component. It must produce the same value as extracting that
# component from the full tensor, within roundoff.



def section_10_single_component_matches_full() -> None:
    """
    For a multivariable function, compute the full Hessian and then
    extract the (0, 1) entry; separately compute the (0, 1) component
    directly via `single_component=(0, 1)`. The two values must agree.
    """

    section_banner("SECTION 10 - Single-component matches full-tensor entry")

    def coupled_test_function(x: np.ndarray) -> float:
        return float(
            x[0] ** 3 * np.sin(x[1]) + x[1] ** 2 * np.cos(x[0])
        )

    evaluation_point = np.array([0.7, 1.1])

    full_hessian = np.asarray(nth_numerical_derivative(
        coupled_test_function, derivative_order=2,
    )(evaluation_point))

    direct_off_diagonal = float(nth_numerical_derivative(
        coupled_test_function, derivative_order=2, single_component=(0, 1),
    )(evaluation_point).as_float())

    full_off_diagonal = float(full_hessian[0, 1])

    test_block(
        test_name="10.single_vs_full",
        description=(
            "verify that pulling H[0, 1] out of the full Hessian agrees with "
            "single_component=(0, 1) up to roundoff, computing both at the "
            "same point and comparing; this is the documented behaviour in the "
            "public Notes, and a discrepancy beyond roundoff means the two "
            "paths disagree on step sizes or coefficient bookkeeping."
        ),
    )

    print(f"  RESULT      : full[0, 1]       = {full_off_diagonal:.12e}")
    print(f"                single_component = {direct_off_diagonal:.12e}")

    check_allclose(
        test_name="10.single_vs_full",
        actual_value=direct_off_diagonal,
        expected_value=full_off_diagonal,
        relative_tolerance=1e-8, absolute_tolerance=1e-10,
        interpretation=(
            "the two computational paths converge on the same numerical "
            "answer (small differences from independent roundoff are "
            "permitted, as documented)."
        ),
    )



# SECTION 11 - DifferentiationResult API SURFACE



def section_11_result_api_surface() -> None:
    """
    Walk through every documented property, method, and operator on
    DifferentiationResult, asserting that each one exists and returns
    a value of the documented type.
    """

    section_banner("SECTION 11 - DifferentiationResult API surface")

    def simple_quadratic(x: np.ndarray) -> float:
        return float(x[0] ** 2 + 2.0 * x[1] ** 2)

    evaluation_point = np.array([1.0, 2.0])

    richardson_result = nth_numerical_derivative(
        simple_quadratic, derivative_order=2,
        richardson_extrapolation=True,
    )(evaluation_point)


    # shape / dtype properties
    test_block(
        test_name="11.shape_dtype",
        description=(
            "verify .shape, .ndim, .size, .dtype, .is_scalar, .is_array, and "
            ".is_full_tensor agree with one another and with the underlying "
            "array, read off the Hessian of a 2-variable quadratic and checked "
            "pairwise; these are advertised as drop-in replacements for the "
            "NumPy attributes, so any inconsistency makes the wrapper fragile "
            "in mixed-array code."
        ),
    )

    derived_array = np.asarray(richardson_result.derivative)

    all_shape_properties_consistent = (
        richardson_result.shape == derived_array.shape == (2, 2)
        and richardson_result.ndim == derived_array.ndim == 2
        and richardson_result.size == derived_array.size == 4
        and richardson_result.dtype == derived_array.dtype
        and richardson_result.is_array
        and not richardson_result.is_scalar
        and richardson_result.is_full_tensor
    )

    check_truth(
        test_name="11.shape_dtype",
        condition=all_shape_properties_consistent,
        message_on_pass="every shape/dtype property is internally consistent.",
        message_on_fail="some shape/dtype property disagrees with the array.",
    )


    # error_estimate / relative_error_estimate / has_error_estimate
    test_block(
        test_name="11.error_properties",
        description=(
            "verify has_error_estimate and relative_error_estimate stay "
            "consistent with error_estimate, checking that has_error_estimate "
            "is True now that Richardson is enabled and that the relative "
            "variant matches the derivative's shape; callers branch on "
            "has_error_estimate to know whether a Romberg bound exists, and "
            "the relative variant is its user-facing form."
        ),
    )

    error_property_consistency = (
        richardson_result.has_error_estimate
        and richardson_result.relative_error_estimate is not None
        and np.asarray(richardson_result.relative_error_estimate).shape
            == richardson_result.shape
    )

    check_truth(
        test_name="11.error_properties",
        condition=error_property_consistency,
        message_on_pass=(
            "has_error_estimate=True; relative_error_estimate has the "
            "right shape."
        ),
        message_on_fail="error-related properties are inconsistent.",
    )


    # as_array / as_float / to_list / to_dict
    test_block(
        test_name="11.conversions",
        description=(
            "verify as_array returns a writable copy, as_float works on a "
            "scalar result, and to_dict carries the documented keys, by "
            "calling as_array and asserting it is writable, calling as_float "
            "on a scalar, and checking the to_dict key set; these conversions "
            "are the canonical bridge between the wrapper and the plain Python "
            "and NumPy structures used downstream."
        ),
    )

    writable_copy = richardson_result.as_array()
    writable_flag_correct = writable_copy.flags.writeable
    writable_copy[0, 0] = 999.0  # mutating the copy must not touch the result

    original_unmodified = float(richardson_result.derivative[0, 0]) != 999.0

    scalar_partial_result = nth_numerical_derivative(
        simple_quadratic, derivative_order=2, single_component=(0, 0),
    )(evaluation_point)

    as_float_works = np.isfinite(scalar_partial_result.as_float())

    dict_form = richardson_result.to_dict()
    expected_dict_keys = {
        'derivative', 'error_estimate', 'derivative_order',
        'differentiation_method', 'step_size', 'point_number',
        'evaluation_points', 'single_component', 'richardson_extrapolation',
        'maximum_richardson_equations',
    }
    dict_keys_match = set(dict_form.keys()) == expected_dict_keys

    list_form = richardson_result.to_list()
    list_form_is_nested_list = (
        isinstance(list_form, list) and len(list_form) == 2
        and isinstance(list_form[0], list)
    )

    check_truth(
        test_name="11.conversions",
        condition=(
            writable_flag_correct and original_unmodified
            and as_float_works and dict_keys_match and list_form_is_nested_list
        ),
        message_on_pass=(
            "every conversion behaves exactly as the docstring promises."
        ),
        message_on_fail=(
            f"writable={writable_flag_correct}, "
            f"original_unmodified={original_unmodified}, "
            f"as_float={as_float_works}, "
            f"dict_keys_match={dict_keys_match}, "
            f"list_form_is_nested_list={list_form_is_nested_list}"
        ),
    )


    # as_float on a non-scalar must raise
    test_block(
        test_name="11.as_float_non_scalar_raises",
        description=(
            "verify as_float on an array-valued result raises ValueError, "
            "calling it on the full Hessian and catching; otherwise as_float "
            "would silently truncate and hide bugs downstream."
        ),
    )

    check_raises(
        test_name="11.as_float_non_scalar_raises",
        callable_object=richardson_result.as_float,
        expected_exception_type=ValueError,
    )


    # __array__ / __getitem__ / __len__ / __iter__ / __contains__
    test_block(
        test_name="11.array_like_behaviour",
        description=(
            "verify np.asarray(result), result[idx], len(result), "
            "iter(result), and `value in result` all forward to the "
            "underlying array, exercising each and asserting the wrapper "
            "behaves as the inner array would; these operators are the "
            "contract that lets the wrapper drop into any expression "
            "expecting a NumPy array."
        ),
    )

    converts_via_np_asarray = (
        np.asarray(richardson_result).shape == (2, 2)
    )

    indexing_yields_row = (
        np.asarray(richardson_result[0]).shape == (2,)
    )

    length_matches_first_axis = len(richardson_result) == 2
    iteration_yields_two_rows = (
        sum(1 for _ in richardson_result) == 2
    )

    array_like_all_pass = (
        converts_via_np_asarray and indexing_yields_row
        and length_matches_first_axis and iteration_yields_two_rows
    )

    check_truth(
        test_name="11.array_like_behaviour",
        condition=array_like_all_pass,
        message_on_pass="every array-like operator behaves as documented.",
        message_on_fail=(
            "at least one of the array-like operators misbehaves."
        ),
    )


    # summary() compact and full
    test_block(
        test_name="11.summary_compact_and_full",
        description=(
            "verify summary('compact') and summary('full') each return non- "
            "empty multi-line strings while an unknown style raises "
            "ValueError, calling each style and then style='non-existant' "
            "to catch the error; summary() is the user-facing inspection tool "
            "and has to be robust against typos."
        ),
    )

    compact_summary_text = richardson_result.summary('compact')
    full_summary_text = richardson_result.summary('full')

    str_calls_compact = str(richardson_result) == compact_summary_text

    both_summaries_multiline = (
        "\n" in compact_summary_text and "\n" in full_summary_text
        and len(full_summary_text) > len(compact_summary_text)
    )

    check_truth(
        test_name="11.summary_compact_and_full",
        condition=str_calls_compact and both_summaries_multiline,
        message_on_pass=(
            "both summary styles produce non-empty multi-line strings; "
            "str(result) routes to the compact one."
        ),
        message_on_fail="summary outputs are inconsistent.",
    )

    test_block(
        test_name="11.summary_invalid_style_raises",
        description=(
            "verify summary('non-existant') raises ValueError, invoking it and "
            "catching; this prevents a silent fallthrough on a mistyped style."
        ),
    )

    check_raises(
        test_name="11.summary_invalid_style_raises",
        callable_object=lambda: richardson_result.summary('non-existant'),
        expected_exception_type=ValueError,
    )


    # __eq__ vs allclose
    test_block(
        test_name="11.equality_semantics",
        description=(
            "verify that == is strict, requiring matching configuration, while "
            "allclose compares values only, by computing the same Hessian "
            "twice (expecting ==) and then with a different point_number "
            "(expecting not == but allclose); the two operators serve "
            "different audiences, byte equality for memoisation and allclose "
            "for cross-config sanity."
        ),
    )

    same_settings_again = nth_numerical_derivative(
        simple_quadratic, derivative_order=2,
        richardson_extrapolation=True,
    )(evaluation_point)

    wider_point_number = nth_numerical_derivative(
        simple_quadratic, derivative_order=2, point_number=5,
        richardson_extrapolation=True
    )(evaluation_point)

    equality_holds = richardson_result == same_settings_again

    inequality_via_config = not (richardson_result == wider_point_number)

    allclose_succeeds = richardson_result.allclose(
        wider_point_number, absolute_tolerance=1e-10
    )

    check_truth(
        test_name="11.equality_semantics",
        condition=(
            equality_holds and inequality_via_config and allclose_succeeds
        ),
        message_on_pass=(
            "== and allclose follow the documented split semantics."
        ),
        message_on_fail="equality semantics violated."
    )


    # allclose with a wrong type raises
    test_block(
        test_name="11.allclose_wrong_type",
        description=(
            "verify allclose() with a non-result argument raises TypeError, "
            "passing a bare ndarray and catching; otherwise the comparison "
            "would silently degrade into a NumPy comparison and bypass the "
            "configuration check."
        ),
    )

    check_raises(
        test_name="11.allclose_wrong_type",
        callable_object=lambda: richardson_result.allclose(np.zeros((2, 2))),
        expected_exception_type=TypeError,
    )


    # repr()
    test_block(
        test_name="11.repr_non_empty",
        description=(
            "verify repr(result) returns a non-empty string, calling repr() "
            "and checking it is non-empty; a usable repr is what makes "
            "tracebacks, debuggers, and logs readable."
        ),
    )

    repr_text = repr(richardson_result)

    check_truth(
        test_name="11.repr_non_empty",
        condition=isinstance(repr_text, str) and len(repr_text) > 0,
        message_on_pass=f"repr is {repr_text!r}",
        message_on_fail="repr is empty or non-string.",
    )


    # unhashability
    test_block(
        test_name="11.unhashable",
        description=(
            "verify results are explicitly unhashable, calling hash(result) "
            "and catching the TypeError; as documented in the Notes, hashing a "
            "value-equal object whose array is only soft-immutable would "
            "quietly break dict and set invariants."
        ),
    )

    check_raises(
        test_name="11.unhashable",
        callable_object=lambda: hash(richardson_result),
        expected_exception_type=TypeError,
    )



# SECTION 12 - IMMUTABILITY ENFORCEMENT
#
# The derivative array exposed on .derivative is documented to be read-only.
# Mutating it would silently corrupt any cached or shared reference. A
# straightforward assignment must therefore raise.



def section_12_immutability() -> None:
    """
    Try to mutate result.derivative directly; expect ValueError or
    similar NumPy write-protect exception. Then call as_array() and
    confirm that the copy can be mutated.
    """

    section_banner("SECTION 12 - Result immutability")

    immutability_result = nth_numerical_derivative(
        lambda x: x[0] ** 3, derivative_order=2,
    )(np.array([1.0, 2.0]))


    test_block(
        test_name="12.derivative_is_readonly",
        description=(
            "verify result.derivative cannot be mutated in place, attempting "
            "to assign into result.derivative[0, 0] and expecting NumPy's "
            "write- protect ValueError; an in-place mutation would silently "
            "propagate to every other handle on the same result and undermine "
            "any reasoning about correctness."
        ),
    )

    def _attempt_mutation_in_place():
        immutability_result.derivative[0, 0] = 999.0

    check_raises(
        test_name="12.derivative_is_readonly",
        callable_object=_attempt_mutation_in_place,
        expected_exception_type=ValueError,
    )

    test_block(
        test_name="12.as_array_is_writable",
        description=(
            "verify as_array() hands back a writable, independent copy, "
            "confirming flags.writeable is True, mutating the copy, and "
            "checking the original is untouched; users who legitimately need "
            "to write into the values must have an escape hatch."
        ),
    )
    writable_local_copy = immutability_result.as_array()
    is_writable = writable_local_copy.flags.writeable

    writable_local_copy[0, 0] = -42.0

    original_intact = float(immutability_result.derivative[0, 0]) != -42.0

    check_truth(
        test_name="12.as_array_is_writable",
        condition=is_writable and original_intact,
        message_on_pass="as_array() yields an independent writable copy.",
        message_on_fail=(
            f"writable={is_writable}, original_intact={original_intact}"
        ),
    )



# SECTION 13 - HESSIAN-VECTOR PRODUCT



def section_13_hessian_vector_product() -> None:
    """
    Four checks on numerical_hessian_vector_product:

    1. On a quadratic, HVP(v) = A @ v exactly (within roundoff).

    2. On a general nonlinear function, HVP matches the full Hessian @ v
       computed via nth_numerical_derivative.

    3. Linearity: HVP(point, alpha * v) = alpha * HVP(point, v).

    4. Symmetry: u · HVP(point, v) ≈ v · HVP(point, u). This is the
       Hessian's symmetry restated through the HVP.
    """

    section_banner("SECTION 13 - Hessian-vector product")

    # Quadratic case
    quadratic_matrix = np.array([[4.0, 1.0], [1.0, 3.0]])

    def gradient_of_quadratic(x: np.ndarray) -> np.ndarray:
        return quadratic_matrix @ x

    quadratic_test_vector = np.array([1.0, 1.0])
    quadratic_evaluation_point = np.array([1.0, -2.0])


    test_block(
        test_name="13.quadratic",
        description=(
            "verify the Hessian-vector product of a quadratic equals A @ v "
            "exactly at any point, building the gradient of 0.5 * x^T A x, "
            "taking the HVP at (1, -2) along (1, 1), and comparing to A @ (1, "
            "1); if this basic linear-algebra identity fails, the directional- "
            "difference stencil is wrong."
        ),
    )
    hvp_quadratic_result = numerical_hessian_vector_product(
        gradient_of_quadratic,
        point=quadratic_evaluation_point, vector=quadratic_test_vector,
    )
    expected_hvp_quadratic = quadratic_matrix @ quadratic_test_vector
    check_allclose(
        test_name="13.quadratic",
        actual_value=np.asarray(hvp_quadratic_result),
        expected_value=expected_hvp_quadratic,
        relative_tolerance=1e-8, absolute_tolerance=1e-10,
        interpretation="HVP matches A @ v to better than 1e-8 relative.",
    )


    # General nonlinear function: HVP vs full Hessian @ v
    def nonlinear_scalar_function(x: np.ndarray) -> float:
        return float(np.sin(x[0]) * np.exp(x[1]) + x[0] ** 2 * x[1])

    def nonlinear_gradient_function(x: np.ndarray) -> np.ndarray:
        return np.array([
            float(np.cos(x[0]) * np.exp(x[1]) + 2.0 * x[0] * x[1]),
            float(np.sin(x[0]) * np.exp(x[1]) + x[0] ** 2),
        ])

    general_evaluation_point = np.array([0.7, 0.5])
    general_test_vector = np.array([0.3, -0.6])

    test_block(
        test_name="13.general_vs_full",
        description=(
            "verify the HVP matches the full Hessian times v built from "
            "nth_numerical_derivative, computing both at the same point and "
            "vector; this cross-checks the HVP against the standalone Hessian "
            "path, whose dispatcher logic is independent."
        ),
    )

    hvp_general = np.asarray(numerical_hessian_vector_product(
        nonlinear_gradient_function,
        point=general_evaluation_point, vector=general_test_vector,
    ))

    full_hessian_general = np.asarray(nth_numerical_derivative(
        nonlinear_scalar_function, derivative_order=2,
    )(general_evaluation_point))

    full_hessian_times_v = full_hessian_general @ general_test_vector

    check_allclose(
        test_name="13.general_vs_full",
        actual_value=hvp_general, expected_value=full_hessian_times_v,
        relative_tolerance=1e-5, absolute_tolerance=1e-8,
        interpretation=(
            "HVP and (full Hessian @ v) agree to a few digits; differences "
            "come from independent step-size selection in the two paths."
        ),
    )


    # Linearity in v
    scaling_alpha = 2.7
    test_block(
        test_name="13.linearity_in_vector",
        description=(
            "verify HVP(point, alpha * v) equals alpha * HVP(point, v), "
            "computing the product at v and at alpha*v and comparing; the HVP "
            "is linear in v, and the implementation's internal scaling and "
            "unit-direction logic must preserve that."
        ),
    )

    hvp_at_scaled_vector = np.asarray(numerical_hessian_vector_product(
        nonlinear_gradient_function,
        point=general_evaluation_point,
        vector=scaling_alpha * general_test_vector,
    ))

    check_allclose(
        test_name="13.linearity_in_vector",
        actual_value=hvp_at_scaled_vector,
        expected_value=scaling_alpha * hvp_general,
        relative_tolerance=1e-6, absolute_tolerance=1e-9,
        interpretation="linearity in v upheld.",
    )


    # Hessian symmetry: u^T H v == v^T H u
    test_block(
        test_name="13.hessian_symmetry",
        description=(
            "verify u . HVP(v) equals v . HVP(u), the symmetry identity, "
            "picking two unrelated vectors and comparing the two inner "
            "products; the Hessian of a C^2 function is symmetric, and the HVP "
            "inherits that as this bilinear identity."
        ),
    )

    direction_u = np.array([0.4, -0.9])
    direction_v = np.array([1.1, 0.2])

    inner_product_u_with_Hv = float(direction_u @ np.asarray(
        numerical_hessian_vector_product(
            nonlinear_gradient_function,
            point=general_evaluation_point, vector=direction_v,
        )
    ))

    inner_product_v_with_Hu = float(direction_v @ np.asarray(
        numerical_hessian_vector_product(
            nonlinear_gradient_function,
            point=general_evaluation_point, vector=direction_u,
        )
    ))

    check_allclose(
        test_name="13.hessian_symmetry",
        actual_value=inner_product_u_with_Hv,
        expected_value=inner_product_v_with_Hu,
        relative_tolerance=1e-6, absolute_tolerance=1e-9,
        interpretation="bilinear symmetry holds.",
    )


    # Zero-vector short-circuit
    sentinel_call_counter = CallCountingFunction(nonlinear_gradient_function)

    test_block(
        test_name="13.zero_vector_short_circuit",
        description=(
            "verify the HVP with a zero vector returns zero without ever "
            "calling the gradient, wrapping the gradient in a counting proxy, "
            "calling HVP with zeros, and asserting the counter never moved; "
            "this short-circuit is documented because gradients can be "
            "expensive or undefined at the point and must not run when the "
            "answer is mathematically zero."
        ),
    )

    hvp_zero_vector = numerical_hessian_vector_product(
        sentinel_call_counter,
        point=general_evaluation_point, vector=np.zeros(2),
    )

    short_circuit_correct = (
        sentinel_call_counter.call_count == 0
        and np.allclose(np.asarray(hvp_zero_vector), np.zeros(2))
    )

    check_truth(
        test_name="13.zero_vector_short_circuit",
        condition=short_circuit_correct,
        message_on_pass=(
            f"gradient_function never called "
            f"(call_count={sentinel_call_counter.call_count}); "
            f"output is the zero vector."
        ),
        message_on_fail=(
            f"call_count={sentinel_call_counter.call_count} (expected 0); "
            f"output={np.asarray(hvp_zero_vector)}"
        ),
    )



# SECTION 14 - PARAMETER VALIDATION
#
# Every documented ValueError path is exercised once, to guarantee that
# malformed user input fails at construction time.



def section_14_parameter_validation() -> None:
    """
    Enumerate every documented ValueError raised by
    `_validate_nth_numerical_derivative_parameters` indirectly via
    `nth_numerical_derivative`, plus the eval-time ValueErrors raised
    by the returned callable on a bad evaluation array.
    """

    section_banner("SECTION 14 - Parameter validation (ValueError paths)")

    placeholder_function = lambda x: x[0] ** 2

    invalid_construction_calls = [
        # (test_name, description, lambda raising ValueError)
        (
            "14.invalid.derivative_order_zero",
            "reject derivative_order=0 at construction, since the n-th "
            "derivative is only defined for n >= 1.",
            lambda: nth_numerical_derivative(
                placeholder_function, derivative_order=0),
        ),
        (
            "14.invalid.derivative_order_negative",
            "reject a negative derivative_order at construction, since the "
            "n-th derivative is only defined for n >= 1.",
            lambda: nth_numerical_derivative(
                placeholder_function, derivative_order=-1),
        ),
        (
            "14.invalid.derivative_order_bool",
            "reject a boolean derivative_order at construction; booleans "
            "are special-cased so True is not quietly accepted as 1.",
            lambda: nth_numerical_derivative(
                placeholder_function, derivative_order=True),
        ),
        (
            "14.invalid.richardson_not_bool",
            "reject a non-bool richardson_extrapolation at construction; "
            "the flag is documented as strictly boolean.",
            lambda: nth_numerical_derivative(
                placeholder_function, richardson_extrapolation=1),
        ),
        (
            "14.invalid.step_size_unknown_string",
            "reject step_size='banana' at construction; the only accepted "
            "strings are 'auto' and 'complex'.",
            lambda: nth_numerical_derivative(
                placeholder_function, step_size='banana'),
        ),
        (
            "14.invalid.step_size_negative",
            "reject step_size <= 0 at construction; the step size must be a "
            "strictly positive real number.",
            lambda: nth_numerical_derivative(
                placeholder_function, step_size=-1e-3),
        ),
        (
            "14.invalid.step_size_nonfinite",
            "reject step_size=inf at construction; non-finite step sizes have "
            "no defined truncation behaviour.",
            lambda: nth_numerical_derivative(
                placeholder_function, step_size=float('inf')),
        ),
        (
            "14.invalid.complex_with_higher_order",
            "reject step_size='complex' together with derivative_order=2; the "
            "complex-step method is mathematically restricted to first "
            "derivatives.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                derivative_order=2, step_size='complex',
            ),
        ),
        (
            "14.invalid.point_number_too_small",
            "reject a point_number below derivative_order + 1 (here "
            "derivative_order=2, point_number=2); the stencil needs at least "
            "derivative_order + 1 points to encode the derivative.",
            lambda: nth_numerical_derivative(
                placeholder_function, derivative_order=2, point_number=2),
        ),
        (
            "14.invalid.point_number_string",
            "reject a non-'auto' string point_number; the only allowed string "
            "is 'auto'.",
            lambda: nth_numerical_derivative(
                placeholder_function, point_number='foo'),
        ),
        (
            "14.invalid.single_component_int_with_high_order",
            "reject a scalar single_component for derivative_order=2; it is "
            "only allowed for first derivatives, since otherwise which index "
            "repeats is ambiguous.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                derivative_order=2, single_component=0,
            ),
        ),
        (
            "14.invalid.single_component_wrong_length",
            "reject a single_component of the wrong length (here (0, 1, 2) for "
            "derivative_order=2); the tuple length must match "
            "derivative_order.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                derivative_order=2, single_component=(0, 1, 2),
            ),
        ),
        (
            "14.invalid.max_richardson_too_small",
            "reject maximum_richardson_equations < 2; the table needs at least "
            "one extrapolation pass, which means two equations.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                richardson_extrapolation=True,
                maximum_richardson_equations=1,
            ),
        ),
        (
            "14.invalid.max_richardson_string",
            "reject a non-'auto' string maximum_richardson_equations; the only "
            "allowed string is 'auto'.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                richardson_extrapolation=True,
                maximum_richardson_equations='loads',
            ),
        ),
    ]

    for test_name, description, raising_call in invalid_construction_calls:
        test_block(test_name=test_name, description=description)
        check_raises(
            test_name=test_name,
            callable_object=raising_call,
            expected_exception_type=ValueError,
        )

    # Eval-time errors

    eval_callable = nth_numerical_derivative(
        lambda x: x[0] + x[1], derivative_order=1, single_component=(5,),
    )


    test_block(
        test_name="14.invalid.single_component_index_out_of_range",
        description=(
            "verify single_component=(5,) against a 2-D evaluation point fails "
            "at eval time, calling the eval function with a 2-D point and "
            "expecting ValueError; the validator runs at construction time "
            "before D is known, so this check has to fire at the first call "
            "site instead."
        ),
    )

    check_raises(
        test_name="14.invalid.single_component_index_out_of_range",
        callable_object=lambda: eval_callable(np.array([1.0, 2.0])),
        expected_exception_type=ValueError,
    )

    bad_step_size_callable = nth_numerical_derivative(
        lambda x: x[0] + x[1], derivative_order=1, step_size=(1e-3, 1e-3, 1e-3),
    )
    test_block(
        test_name="14.invalid.step_size_tuple_length_mismatch",
        description=(
            "verify a step_size tuple whose length differs from D fails at "
            "eval time, configuring a 3-entry tuple but evaluating at a 2-D "
            "point; tuple-form step_size must carry exactly one entry per "
            "coordinate."
        ),
    )

    check_raises(
        test_name="14.invalid.step_size_tuple_length_mismatch",
        callable_object=lambda: bad_step_size_callable(np.array([1.0, 2.0])),
        expected_exception_type=ValueError,
    )


    # HVP validation
    test_block(
        test_name="14.invalid.hvp_non_callable_gradient",
        description=(
            "verify a non-callable gradient_function is rejected by passing a "
            "list as the gradient; the signature is documented to require a "
            "callable."
        ),
    )

    check_raises(
        test_name="14.invalid.hvp_non_callable_gradient",
        callable_object=lambda: numerical_hessian_vector_product(
            [1, 2], point=np.zeros(2), vector=np.zeros(2),
        ),
        expected_exception_type=ValueError,
    )


    test_block(
        test_name="14.invalid.hvp_vector_shape_mismatch",
        description=(
            "verify a vector whose shape differs from the point's is rejected "
            "by passing mismatched shapes; the HVP is defined coordinatewise."
        ),
    )

    check_raises(
        test_name="14.invalid.hvp_vector_shape_mismatch",
        callable_object=lambda: numerical_hessian_vector_product(
            lambda x: x, point=np.zeros(3), vector=np.zeros(2),
        ),
        expected_exception_type=ValueError,
    )



# SECTION 15 - WARNING EMISSION



def section_15_warning_emission() -> None:
    """
    Exercise every documented RuntimeWarning path. Each call must
    actually emit the warning (otherwise the user has no way to know
    they are using the API outside its sweet spot).
    """

    section_banner("SECTION 15 - RuntimeWarning emission")

    placeholder_function = lambda x: np.sin(x[0])

    warning_cases = [
        (
            "15.warn.complex_with_richardson",
            "warn that richardson_extrapolation=True is ignored when "
            "step_size='complex', setting both flags and expecting a "
            "RuntimeWarning; complex-step already reaches machine "
            "precision, so Richardson on top gains nothing.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                step_size='complex', richardson_extrapolation=True,
            ),
        ),
        (
            "15.warn.complex_with_point_number",
            "warn that point_number is ignored when step_size='complex', "
            "setting both and expecting a RuntimeWarning; the "
            "complex-step method uses no stencil.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                step_size='complex', point_number=5,
            ),
        ),
        (
            "15.warn.max_richardson_without_extrapolation",
            "warn that maximum_richardson_equations is ignored when "
            "richardson_extrapolation=False, setting the flag off with "
            "the value at 10; it would otherwise have no effect, and "
            "the warning is the only signal that the configuration is "
            "contradictory.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                richardson_extrapolation=False,
                maximum_richardson_equations=10,
            ),
        ),
        (
            "15.warn.max_richardson_above_50",
            "warn that maximum_richardson_equations above 50 is capped, "
            "setting it to 100 with Richardson on; beyond 50 levels the "
            "table burns computation with no measurable gain.",
            lambda: nth_numerical_derivative(
                placeholder_function,
                richardson_extrapolation=True,
                maximum_richardson_equations=100,
            ),
        ),
    ]


    for test_name, description, warning_call in warning_cases:
        test_block(test_name=test_name, description=description)
        check_warns(
            test_name=test_name,
            callable_object=warning_call,
            expected_warning_category=RuntimeWarning,
        )



# SECTION 16 - EDGE CASES
#
# log near zero, functions with very large return values, mixed-scale
# coordinates, high dimensions.



def section_16_edge_cases() -> None:
    """
    Five edge cases that historically trip implementations:

    1. Near-singularity step selection: log(x) at x close to its domain
       boundary should still produce a finite, near-correct answer
       (and Richardson should cap its table length if necessary).

    2. Constant function: gradient must be exactly zero.

    3. Very large function values: adding a large constant must not
       change the derivative (subtractive cancellation regression test).

    4. Mixed scales: x = [1e8, 1e-8] - the auto step size should
       handle each coordinate independently.

    5. High-dimensional input: D = 30 gradient must complete and
       agree with the analytical answer entry-by-entry.
    """

    section_banner("SECTION 16 - Edge cases")


    # log near singularity
    log_function = lambda x: np.log(x[0])
    test_block(
        test_name="16.edge.log_near_singularity",
        description=(
            "verify d/dx log(x) at x = 0.05 stays accurate despite the small "
            "radius of analyticity changing default point number to 2, where "
            "the analytical value is 1/0.05 = 20."
        ),
    )

    log_derivative = float(nth_numerical_derivative(
        log_function, derivative_order=1, point_number=2
    )(np.array([0.05])).as_float())

    check_allclose(
        test_name="16.edge.log_near_singularity",
        actual_value=log_derivative, expected_value=20.0,
        relative_tolerance=1e-4, absolute_tolerance=1e-6,
        interpretation=(
            "log near singularity computed within engineering accuracy; the "
            "step selector keeps perturbations away from x <= 0."
        ),
    )


    # Constant function
    constant_function = lambda x: 42.0
    test_block(
        test_name="16.edge.constant_function",
        description=(
            "verify the gradient of a constant is the zero vector, computing "
            "it at an arbitrary point and checking its max absolute value sits "
            "below the noise floor; this is a bug-catching invariant, since "
            "every subtraction in the stencil must cancel exactly for a "
            "constant signal."
        ),
    )

    constant_gradient = np.asarray(nth_numerical_derivative(
        constant_function, derivative_order=1,
    )(np.array([1.7, -0.5, 3.3])))

    check_truth(
        test_name="16.edge.constant_function",
        condition=float(np.max(np.abs(constant_gradient))) < 1e-10,
        message_on_pass=(
            f"max|gradient| = {float(np.max(np.abs(constant_gradient))):.2e} "
            f"is within the noise floor."
        ),
        message_on_fail=(
            "gradient of a constant came out non-zero beyond roundoff."
        ),
    )


    # Large additive offset
    large_offset = 1e10
    biased_function = lambda x: x[0] ** 2 + large_offset
    test_block(
        test_name="16.edge.large_additive_offset",
        description=(
            "verify adding a huge constant to f(x) leaves d^2 f / dx^2 "
            "unchanged, computing d^2/dx^2 of x^2 + 1e10 at x = 1 and "
            "expecting 2.0; the scheme should be invariant to additive shifts, "
            "and a regression here means cancellation is amplifying the offset."
        ),
    )

    biased_second = float(nth_numerical_derivative(
        biased_function, derivative_order=2,
    )(np.array([1.0])).as_float())

    check_allclose(
        test_name="16.edge.large_additive_offset",
        actual_value=biased_second, expected_value=2.0,
        relative_tolerance=1e-3, absolute_tolerance=1e-3,
        interpretation=(
            "result is close to the offset-free value; subtractive "
            "cancellation handled within reasonable tolerance."
        ),
    )


    # Mixed-scale coordinates
    def mixed_scale_function(x: np.ndarray) -> float:
        return float(x[0] ** 2 + 1e16 * x[1] ** 2)

    mixed_scale_point = np.array([1e8, 1e-8])

    test_block(
        test_name="16.edge.mixed_scales",
        description=(
            "verify the gradient stays accurate even when coordinates differ "
            "by sixteen orders of magnitude, taking the gradient of x[0]^2 + "
            "1e16*x[1]^2 at (1e8, 1e-8) against the analytical (2e8, 2e8); "
            "per- coordinate 'auto' step sizing is the only thing keeping this "
            "well behaved, since a uniform step would either lose precision on "
            "x[0] or underflow on x[1]."
        ),
    )

    mixed_gradient = np.asarray(nth_numerical_derivative(
        mixed_scale_function, derivative_order=1,
    )(mixed_scale_point))

    check_allclose(
        test_name="16.edge.mixed_scales",
        actual_value=mixed_gradient, expected_value=np.array([2e8, 2e8]),
        relative_tolerance=1e-4, absolute_tolerance=1.0,
        interpretation="per-coordinate step sizing handles the scale ratio.",
    )


    # High-D gradient
    high_dimension_size = 30

    def high_dimensional_quadratic(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    high_dimensional_point = np.linspace(-1.0, 1.0, high_dimension_size)

    test_block(
        test_name="16.edge.high_dimensional_gradient",
        description=(
            f"verify the gradient of sum(x^2) in D = "
            f"{high_dimension_size} completes and equals 2x, computing it and "
            f"checking against 2x; this is where large-D performance and "
            f"correctness have to hold together for optimizers."
        ),
    )

    high_dimensional_gradient = np.asarray(nth_numerical_derivative(
        high_dimensional_quadratic, derivative_order=1,
    )(high_dimensional_point))

    check_allclose(
        test_name="16.edge.high_dimensional_gradient",
        actual_value=high_dimensional_gradient,
        expected_value=2.0 * high_dimensional_point,
        relative_tolerance=1e-6, absolute_tolerance=1e-9,
        interpretation=(
            f"gradient is correct entry-by-entry in D = {high_dimension_size}."
        ),
    )



# SECTION 20 - CACHING EFFECTIVENESS



def section_20_cache_effectiveness() -> None:
    """
    Time a single call against the average of 100 subsequent calls of
    the same configured derivative. The ratio is reported as an
    information metric.
    """

    section_banner("SECTION 20 - Stencil cache effectiveness")

    def lightweight_test_function(x: np.ndarray) -> float:
        return float(x[0] ** 4 + x[1] ** 4 + x[0] * x[1])

    evaluation_point = np.array([1.0, 1.0])
    cache_test_callable = nth_numerical_derivative(
        lightweight_test_function, derivative_order=2,
    )


    # Flush any Python imports / just in time compilation effects.
    cache_test_callable(evaluation_point)

    first_call_t0 = time.perf_counter()
    cache_test_callable(evaluation_point)
    first_call_duration = time.perf_counter() - first_call_t0

    repeated_calls_count = 100
    repeated_t0 = time.perf_counter()
    for _ in range(repeated_calls_count):
        cache_test_callable(evaluation_point)
    average_subsequent_duration = (
        (time.perf_counter() - repeated_t0) / repeated_calls_count
    )


    test_block(
        test_name="20.cache",
        description=(
            f"verify that after one warm call the later calls reuse the cached "
            f"stencil coefficients, timing one call after warm-up against the "
            f"average of {repeated_calls_count} more at the same point and "
            f"reporting the ratio; optimizers and Monte-Carlo evaluators call "
            f"the configured derivative thousands of times with the same "
            f"stencil, so recomputing coefficients here would be an "
            f"orders-of-magnitude regression."
        ),
    )

    print(f"  RESULT      : first-call duration       = "
          f"{first_call_duration * 1e6:.1f} us")
    print(f"                avg subsequent duration   = "
          f"{average_subsequent_duration * 1e6:.1f} us")
    print(f"                ratio (subsequent / first) = "
          f"{average_subsequent_duration / max(first_call_duration, 1e-12):.2f}"
    )

    # if subsequent calls are 5x slower than the first the cache is clearly
    # broken.
    cache_appears_healthy = (
        average_subsequent_duration < 5.0 * first_call_duration
    )
    if cache_appears_healthy:
        REPORT.record_info(
            "cache behaves as expected (subsequent calls do not regress)."
        )

    else:
        REPORT.record_fail(
            "20.cache",
            "subsequent calls are >5x slower than the first; cache "
            "appears broken.",
            expected="ratio < 5",
            actual=(
                average_subsequent_duration
                / max(first_call_duration, 1e-12)
            ),
        )



# SECTION 21 - METHOD-SPEED BENCHMARK



def section_21_method_speed_benchmark() -> None:
    """
    Time complex-step, central-difference, and Richardson extrapolation
    on the same first-derivative problem and report per-call latency.
    """

    section_banner("SECTION 21 - Method-speed benchmark")

    benchmark_function = lambda x: np.exp(np.sin(x[0]) + x[0] ** 2)
    benchmark_point = np.array([0.7])
    benchmark_repetitions = 200

    benchmarked_methods = {
        'complex_step': nth_numerical_derivative(
            benchmark_function, derivative_order=1, step_size='complex',
        ),
        'central_difference (auto)': nth_numerical_derivative(
            benchmark_function, derivative_order=1,
        ),
        'richardson_extrapolation': nth_numerical_derivative(
            benchmark_function, derivative_order=1,
            richardson_extrapolation=True,
        ),
    }

    # Flush any Python imports / just in time compilation effects.
    for callable_object in benchmarked_methods.values():
        callable_object(benchmark_point)


    test_block(
        test_name="21.method_speed",
        description=(
            f"measure the per-call latency of the three numerical strategies, "
            f"invoking each {benchmark_repetitions} times on the same point "
            f"and reporting the average elapsed time duration; latency drives "
            f"the practical choice of method inside tight optimizer loops, and "
            f"knowing the order of magnitude is the first step in tuning."
        ),
    )


    benchmark_table = []
    for method_label, callable_object in benchmarked_methods.items():
        loop_t0 = time.perf_counter()

        for _ in range(benchmark_repetitions):
            callable_object(benchmark_point)

        per_call_duration_seconds = (
            (time.perf_counter() - loop_t0) / benchmark_repetitions
        )

        benchmark_table.append((method_label, per_call_duration_seconds))

    print("  RESULT      :")

    for method_label, per_call_duration_seconds in benchmark_table:
        print(f"                {method_label:<32} = "
              f"{per_call_duration_seconds * 1e6:8.1f} us / call")

    REPORT.record_info("benchmark recorded; no hard threshold imposed.")



# SECTION 22 - HVP VS FULL HESSIAN BENCHMARK



def section_22_hvp_vs_full_benchmark() -> None:
    """
    On a D = 25 quadratic with an analytical gradient, compare:

      - one HVP call,
      - one full-Hessian build followed by one matrix-vector product.

    Report both wall-clock times and the speedup factor. This is purely
    informational.
    """

    section_banner("SECTION 22 - HVP vs full-Hessian benchmark")

    dimension_size = 25

    random_state = np.random.default_rng(seed=42)

    spd_matrix = random_state.standard_normal(
        (dimension_size, dimension_size)
    )

    spd_matrix = spd_matrix.T @ spd_matrix + np.eye(dimension_size)

    def quadratic_function(x: np.ndarray) -> float:
        return float(0.5 * x @ spd_matrix @ x)

    def quadratic_gradient(x: np.ndarray) -> np.ndarray:
        return spd_matrix @ x

    benchmark_point = random_state.standard_normal(dimension_size)
    direction_vector = random_state.standard_normal(dimension_size)

    hessian_callable = nth_numerical_derivative(
        quadratic_function, derivative_order=2,
    )

    # Flush any Python imports / just in time compilation effects.
    numerical_hessian_vector_product(
        quadratic_gradient, point=benchmark_point, vector=direction_vector,
    )
    hessian_callable(benchmark_point)

    timing_repetitions = 5

    hvp_t0 = time.perf_counter()
    for _ in range(timing_repetitions):
        numerical_hessian_vector_product(
            quadratic_gradient,
            point=benchmark_point, vector=direction_vector,
        )

    hvp_per_call_duration = (
        (time.perf_counter() - hvp_t0) / timing_repetitions
    )

    full_t0 = time.perf_counter()
    for _ in range(timing_repetitions):
        full_hessian = np.asarray(hessian_callable(benchmark_point))
        _full_h_times_v = full_hessian @ direction_vector

    full_per_call_duration = (
        (time.perf_counter() - full_t0) / timing_repetitions
    )


    test_block(
        test_name="22.hvp_speedup",
        description=(
            f"measure the per-call cost of one HVP against one full-Hessian "
            f"build plus matrix-vector product in D = {dimension_size}, "
            f"invoking each path {timing_repetitions} times on the same SPD "
            f"quadratic and reporting per-call wall-clock; the HVP advertises "
            f"O(p) gradient calls versus O(D^2) function calls for the full "
            f"Hessian, and that crossover is what justifies the more elaborate "
            f"HVP path."
        ),
    )

    print(f"  RESULT      : HVP per call          = "
          f"{hvp_per_call_duration * 1e3:.2f} ms")
    print(f"                full Hessian @ v per call = "
          f"{full_per_call_duration * 1e3:.2f} ms")
    print(f"                speedup factor        = "
          f"{full_per_call_duration / max(hvp_per_call_duration, 1e-12):.1f}x")

    REPORT.record_info(
        "benchmark recorded; HVP is the recommended path for one-direction "
        "queries in moderate-to-high D."
    )



# SECTION 17 - STRESS: HIGH-ORDER DERIVATIVE



def section_17_high_order_stress() -> None:
    """
    Compute d^8 exp(x) / dx^8 at x = 0.5. The analytical answer is
    exp(0.5) by the eigenvalue property of exp. A wide enough stencil
    (default 'auto' = order + 7 = 15 points) should still recover it
    to a few digits.
    """

    section_banner("SECTION 17 - Stress: 8th derivative of exp(x)")

    high_order_value = float(nth_numerical_derivative(
        lambda x: np.exp(x[0]), derivative_order=8,
        richardson_extrapolation=True, single_component=(0,) * 8,
    )(np.array([0.5])).as_float())


    test_block(
        test_name="17.high_order_stress",
        description=(
            "verify d^8 exp(x) / dx^8 at x = 0.5 equals exp(0.5), evaluating "
            "the 8th derivative with Richardson and comparing; high-order "
            "stencils are where coefficient roundoff bites hardest, so a "
            "regression in the stencil builder shows up here first."
        ),
    )

    check_allclose(
        test_name="17.high_order_stress",
        actual_value=high_order_value,
        expected_value=float(np.exp(0.5)),
        relative_tolerance=1e-3, absolute_tolerance=1e-4,
        interpretation=(
            "the high-order stencil + Richardson combination recovers "
            "the eigenvalue identity within engineering accuracy."
        ),
    )



# SECTION 18 - RESULT WITHOUT AN ERROR ESTIMATE



def section_18_result_without_error_estimate() -> None:
    """
    Exercise the result object's no-error-estimate contract: a
    central-difference result (no Richardson) reports error_estimate=None,
    has_error_estimate=False, relative_error_estimate=None and
    maximum_richardson_equations=None and still serialises; a complex-step
    first derivative likewise carries no error estimate; and a scalar
    Richardson result returns relative_error_estimate as a plain float.
    """

    section_banner("SECTION 18 - Result without an error estimate")

    def quadratic(x: np.ndarray) -> float:
        return float(x[0] ** 2 + 2.0 * x[1] ** 2)

    evaluation_point = np.array([1.0, 2.0])


    # Central-difference result, no-error-estimate.
    test_block(
        test_name="18.no_error_estimate_contract",
        description=(
            "verify a plain central-difference result honours its documented "
            "no-error-estimate contract, computing a Hessian without "
            "Richardson and checking error_estimate is None, "
            "has_error_estimate is False, relative_error_estimate is None and "
            "maximum_richardson_equations is None, while to_dict still carries "
            "the None and summary still renders; callers branch on "
            "has_error_estimate to decide whether a Romberg bound exists, so "
            "the None side has to be exactly as advertised."
        ),
    )

    central_result = nth_numerical_derivative(
        quadratic, derivative_order=2,
    )(evaluation_point)

    no_estimate_contract = (
        central_result.error_estimate is None
        and central_result.has_error_estimate is False
        and central_result.relative_error_estimate is None
        and central_result.maximum_richardson_equations is None
        and central_result.to_dict()["error_estimate"] is None
        and bool(central_result.summary("full"))
        and bool(central_result.summary("compact"))
    )

    check_truth(
        test_name="18.no_error_estimate_contract",
        condition=no_estimate_contract,
        message_on_pass=(
            "the central-difference result reports no error estimate exactly "
            "as documented."
        ),
        message_on_fail="a no-error-estimate property broke its contract.",
    )


    # Complex-step first derivative, no-error-estimate.
    test_block(
        test_name="18.complex_step_has_no_estimate",
        description=(
            "verify a complex-step first derivative reports no error estimate, "
            "computing df/dx with step_size='complex' and checking "
            "has_error_estimate is False and error_estimate is None; the "
            "docstring lists complex-step alongside plain differences as a "
            "method that produces no Romberg bound, so it must agree with the "
            "central-difference case above."
        ),
    )

    # Complex-step evaluates it at a complex argument, and float() would
    # discard the imaginary part the method depends on (and emit a
    # ComplexWarning).
    complex_result = nth_numerical_derivative(
        lambda x: np.exp(x[0]), derivative_order=1,
        step_size="complex", single_component=(0,),
    )(np.array([1.0]))

    complex_no_estimate = (
        complex_result.has_error_estimate is False
        and complex_result.error_estimate is None
        and complex_result.relative_error_estimate is None
        and abs(float(complex_result.as_float()) - float(np.e)) < 1e-12
    )

    check_truth(
        test_name="18.complex_step_has_no_estimate",
        condition=complex_no_estimate,
        message_on_pass=(
            "complex-step carries no error estimate, as documented."
        ),
        message_on_fail="complex-step reported an unexpected error estimate.",
    )


    # Scalar Richardson result: relative_error_estimate is a plain float.
    test_block(
        test_name="18.scalar_relative_error_is_float",
        description=(
            "verify a single-component Richardson result returns "
            "relative_error_estimate as a plain float rather than an array, "
            "extracting one second partial with Richardson on and checking the "
            "scalar branch; the property is documented to mirror derivative's "
            "type, so a scalar derivative must yield a scalar relative error."
        ),
    )

    scalar_richardson = nth_numerical_derivative(
        quadratic, derivative_order=2, single_component=(0, 0),
        richardson_extrapolation=True,
    )(evaluation_point)

    scalar_relative_error = scalar_richardson.relative_error_estimate

    scalar_branch_ok = (
        scalar_richardson.is_scalar is True
        and scalar_richardson.has_error_estimate is True
        and isinstance(scalar_relative_error, float)
        and math.isfinite(scalar_relative_error)
        and scalar_richardson.is_full_tensor is False
    )

    print(f"  RESULT      : is_scalar={scalar_richardson.is_scalar}, "
          f"relative_error_estimate={scalar_relative_error!r} "
          f"({type(scalar_relative_error).__name__})")

    check_truth(
        test_name="18.scalar_relative_error_is_float",
        condition=scalar_branch_ok,
        message_on_pass=(
            "the scalar Richardson result returns a float relative error."
        ),
        message_on_fail=(
            f"is_scalar={scalar_richardson.is_scalar}, relative type="
            f"{type(scalar_relative_error).__name__}."
        ),
    )



# SECTION 19 - VALIDATION GAPS AND ACCEPTED INPUT FORMS



def section_19_validation_and_input_forms() -> None:
    """
    Close the parameter-validation gaps Section 14 leaves open - a boolean
    step_size, a step-size tuple with a bad entry, a non-integer point_number,
    a boolean single_component, and a non-integer single_component tuple - and
    confirm two accepted forms the suite never otherwise takes: a bare-int
    single_component at first order, and an explicit in-range
    maximum_richardson_equations.
    """

    section_banner("SECTION 19 - Validation gaps and accepted input forms")

    quadratic = lambda x: float(x[0] ** 2 + 2.0 * x[1] ** 2)
    evaluation_point = np.array([1.0, 2.0])

    # Documented rejections Section 14 does not reach.
    rejected_calls = [
        ("19.invalid.step_size_bool",
         "a boolean step_size is rejected as not a valid step",
         lambda: nth_numerical_derivative(
             quadratic, derivative_order=1,
             step_size=True)(evaluation_point)),
        ("19.invalid.step_size_tuple_bad_entry",
         "a step-size tuple holding a non-finite entry is rejected",
         lambda: nth_numerical_derivative(
             quadratic, derivative_order=1,
             step_size=(1e-3, float("inf")))(evaluation_point)),
        ("19.invalid.point_number_not_integer",
         "a non-integer, non-'auto' point_number is rejected",
         lambda: nth_numerical_derivative(
             quadratic, derivative_order=1,
             point_number=3.5)(evaluation_point)),
        ("19.invalid.single_component_bool",
         "a boolean single_component is rejected",
         lambda: nth_numerical_derivative(
             quadratic, derivative_order=1,
             single_component=True)(evaluation_point)),
        ("19.invalid.single_component_not_integers",
         "a single_component tuple that is not integer-convertible is rejected",
         lambda: nth_numerical_derivative(
             quadratic, derivative_order=2,
             single_component=("x", "y"))(evaluation_point)),
    ]

    for case_name, what_it_checks, offending_call in rejected_calls:
        test_block(
            test_name=case_name,
            description=(
                f"verify {what_it_checks}, calling nth_numerical_derivative "
                f"with the offending argument and requiring a ValueError; "
                f"these are documented rejections the existing validation "
                f"section does not cover, and silent acceptance would let a "
                f"malformed request reach the numerical core."
            ),
        )

        check_raises(
            test_name=case_name,
            callable_object=offending_call,
            expected_exception_type=ValueError,
            interpretation=(
                "the malformed argument is rejected with ValueError."
            ),
        )


    # Accepted input forms the suite never otherwise takes.
    test_block(
        test_name="19.int_single_component_matches_tuple",
        description=(
            "verify a bare-int single_component at first order is accepted and "
            "equals the one-tuple form, differentiating once with "
            "single_component=0 and again with single_component=(0,) and "
            "comparing; the signature documents the int shorthand for "
            "first-order requests, so it must resolve to the same partial."
        ),
    )

    int_form_result = nth_numerical_derivative(
        quadratic, derivative_order=1, single_component=0,
    )(evaluation_point)

    tuple_form_result = nth_numerical_derivative(
        quadratic, derivative_order=1, single_component=(0,),
    )(evaluation_point)

    int_matches_tuple = (
        float(int_form_result.as_float())
        == float(tuple_form_result.as_float())
    )

    check_truth(
        test_name="19.int_single_component_matches_tuple",
        condition=int_matches_tuple,
        message_on_pass="the int shorthand equals the one-tuple form.",
        message_on_fail="the int shorthand disagreed with the one-tuple form.",
    )


    test_block(
        test_name="19.explicit_maximum_richardson_equations",
        description=(
            "verify an explicit in-range maximum_richardson_equations is "
            "accepted, recorded, and produces the right Hessian, running a "
            "Richardson Hessian with the cap set to 4 and checking the result "
            "reports the cap it was given and lands on the analytical value; "
            "the cap is a documented knob, so a valid value has to be honoured "
            "and surfaced on the result."
        ),
    )

    capped_result = nth_numerical_derivative(
        quadratic, derivative_order=2, richardson_extrapolation=True,
        maximum_richardson_equations=4,
    )(evaluation_point)

    cap_honoured = (
        capped_result.maximum_richardson_equations == 4
        and np.allclose(
            np.asarray(capped_result.derivative),
            np.array([[2.0, 0.0], [0.0, 4.0]]),
            rtol=1e-6, atol=1e-6,
        )
    )

    print(f"  RESULT      : maximum_richardson_equations="
          f"{capped_result.maximum_richardson_equations}")

    check_truth(
        test_name="19.explicit_maximum_richardson_equations",
        condition=cap_honoured,
        message_on_pass="the explicit cap is honoured and recorded.",
        message_on_fail=(
            f"cap={capped_result.maximum_richardson_equations}, or the "
            f"Hessian missed the analytical value."
        ),
    )



# MAIN ENTRY POINT
#
# `run()` is the single point shared by every way of launching the suite: the
# `ripples.test()` dispatcher, the `python -m ripples.differentiation._test`
# command line, and the pytest bridge below all come through here, so the
# output and the verdicts are identical no matter how the suite is started.



def run(
    include_benchmarks: bool = True, verbose: bool = True,
    show_summary: bool = True, quiet_label: Optional[str] = None,
) -> Reporter:
    """
    Execute the differentiation test and benchmark suite end to end.

    Parameters
    ----------
    include_benchmarks : bool, default True
        When True the three timing-only sections (the stencil cache, the
        per-method speed comparison, and the Hessian-vector product against
        the full Hessian) run as well. They never move the failure count -
        they only report INFO - but they are the slow part of the suite, so
        a caller that just wants a correctness gate (continuous integration,
        a quick local check) can pass False to skip them.

    Returns
    -------
    Reporter
        The reporter holding the run's tallies (`passed`, `failed`,
        `skipped`, `info`) and its failure records. It is returned rather
        than a bare count so a caller aggregating several submodules - as
        `ripples.test()` does when asked to test everything - can fold these
        numbers into one combined summary.
    """

    # A fresh on every call: the module-level reporter is reused, so
    # without this a second run would keep adding to the first run's counts.
    REPORT.reset()
    REPORT.verbose = verbose

    _real_stdout = sys.stdout
    REPORT._writer = (
        _StdoutProxy(_real_stdout, label=quiet_label)
        if not verbose else None
    )
    if REPORT._writer is not None:
        sys.stdout = REPORT._writer

    try:

        # The sections in reading order. The three benchmark sections are tagged
        # so they can be filtered out when the caller does not want timings.
        benchmark_sections = {
            section_20_cache_effectiveness,
            section_21_method_speed_benchmark,
            section_22_hvp_vs_full_benchmark,
        }
        ordered_sections = [
            section_01_known_values,
            section_02_polynomial_cancellation,
            section_03_schwarz_symmetry,
            section_04_order_of_accuracy,
            section_05_three_way_cross_validation,
            section_06_richardson_improves,
            section_07_error_estimate_reliability,
            section_08_shape_validation,
            section_09_batch_equals_loop,
            section_10_single_component_matches_full,
            section_11_result_api_surface,
            section_12_immutability,
            section_13_hessian_vector_product,
            section_14_parameter_validation,
            section_15_warning_emission,
            section_16_edge_cases,
            section_17_high_order_stress,
            section_18_result_without_error_estimate,
            section_19_validation_and_input_forms,
            section_20_cache_effectiveness,
            section_21_method_speed_benchmark,
            section_22_hvp_vs_full_benchmark,
        ]
        sections_to_run = [
            section for section in ordered_sections
            if include_benchmarks or section not in benchmark_sections
        ]

        print("=" * 78)
        print(" ripples.differentiation - test and benchmark "
              "suite ".center(78, "="))
        print("=" * 78)

        if not include_benchmarks:
            print("Benchmarks skipped (include_benchmarks=False); "
                  "running correctness sections only.")

        for section_runner in sections_to_run:
            try:
                section_runner()

            except Exception as section_exception:
                # An exception escaping a section is itself a failure; record
                # it, print the traceback, and carry on with the next section
                # so a single bug cannot abort the whole suite.
                import traceback

                REPORT.record_fail(
                    test_name=section_runner.__name__,
                    message=(
                        f"section raised an unhandled "
                        f"{type(section_exception).__name__}: "
                        f"{section_exception}"
                    ),
                )

                print()
                print(" UNCAUGHT EXCEPTION ".center(78, "!"))

                traceback.print_exc()

                print("!" * 78)

    finally:
        if REPORT._writer is not None:
            REPORT._writer.commit()
            sys.stdout = _real_stdout
            REPORT._writer = None


    if show_summary:
        REPORT.print_summary()

    return REPORT



def main(argv: Optional[List[str]] = None) -> int:
    """
    Command-line entry point for `python -m ripples.differentiation._test`.

    Recognises one optional flag, ``--no-benchmarks``, forwarded to
    `run(include_benchmarks=False)`. Returns the number of failures so the
    process exit code is 0 on a clean run and non-zero otherwise - the
    convention continuous-integration systems read.
    """
    arguments = sys.argv[1:] if argv is None else argv

    include_benchmarks = "--no-benchmarks" not in arguments

    reporter = run(include_benchmarks=include_benchmarks)

    return reporter.failed



# PYTEST BRIDGE
#
# A single function so that pytest - which continuous integration runs on
# every push - actually gates on this suite. It drives the same `run()` as
# every other entry point (benchmarks off, since pytest is a correctness gate,
# not a stopwatch) and turns a non-zero failure count into a failed test. The
# per-test report is printed by `run()`; pytest shows it with `-s` or on
# failure.



def test_differentiation_suite() -> None:
    """Fail under pytest if any correctness check in the suite fails."""
    reporter = run(include_benchmarks=False)

    assert reporter.failed == 0, (
        f"{reporter.failed} differentiation check(s) failed; "
        f"see the printed report of each."
    )
