"""
Tests and benchmarks for `ripples.optimization`.

This module exercises every public surface of the optimization module
(***minimizer*** and the ***OptimizationResult*** it returns) along:

- Numerical correctness against problems with a known minimum: every
local method is driven to the minimizer of a well-conditioned
quadratic, the strong methods are taken through the classic hard
landscapes (Rosenbrock, Booth), the second-order methods are fed an
analytical Hessian and a Hessian-vector product, and the global
methods are asked to locate the optimum of a bounded and of a
multimodal objective.

- Feature coverage: analytical versus finite-difference gradients, box
bounds and equality / inequality constraints through the
augmented-Lagrangian wrapper, the iteration callback, custom
tolerances and per-method parameters, and the every-property surface
of the ***OptimizationResult*** wrapper.

- API contract: parameter validation (every documented ValueError and
TypeError path), the warning channel, and the conversion / comparison
/ summary methods of the result object.

- Performance: elapsed time of several methods on the same problem, the
evaluation-count cost of a numerical gradient against an analytical one, and
the Hessian-vector path against forming the full Hessian.


Output convention
-----------------
For each test the script prints, on its own block:

- a header line of the form
               "Section <n>, Test <m>: s<n>.<name> - <description>", where
               the description states in one sentence what the test
               verifies and why it matters, with the checking procedure
               folded into the same sentence.
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
    >>> ripples.test("optimization")

from the command line:

    python -m ripples.optimization._test
    # append --no-benchmarks to skip the timing-only sections

or programmatically, when a caller wants the tallies back:

    >>> from ripples.optimization._test import run
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


A note on tolerances
--------------------
Optimization, unlike differentiation, returns an approximate minimizer, not
a closed-form quantity, so the bands used here are looser and chosen per
method: the quasi-Newton, conjugate-gradient and trust-region paths are held
to a few digits, the first-order methods (which the module itself documents as
weak general-purpose solvers) are started near the optimum and judged on a
relaxed band, and the global methods are judged on whether they reached the
right basin rather than on the last few digits.
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

from ._minimizer import minimizer
from ._utils import OptimizationResult



FLOAT_EPSILON = float(np.finfo(np.float64).eps)

SEPARATOR_WIDTH = 79

_CURRENT_SECTION_TEST_NUMBER = 0



# REPORTING UTILITIES
#
# Every test routes through `check_*` helpers that record a pass / fail entry
# on the module-level `REPORT` object. The helpers print the verdict line in
# place, so that the caller can read the output top-to-bottom as the suite
# progresses, and the summary at the end of `main()` re-prints every failure
# for triage convenience.



class Reporter:
    """
    Aggregator for pass / fail / skip / info events.

    A single instance is kept at module level (`REPORT`) so every test
    can record into the same accumulator.

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
    stdout is proxied in quiet mode.

    All output for the current test is buffered. At each boundary (the next
    test_block or when run() finishes) the buffer is either flushed to the real
    stream if the test failed, or discarded if the test passed. The final
    summary is written directly because run() restores the real stream before
    printing it.
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
        "12.invalid.unknown_method". The same string is handed to the
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

    # Element-wise absolute difference normalised by max(|expected|, eps)
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
    Assertion that calling `callable_object()` emits at least one
    warning of the expected category. The function's return value is
    captured but ignored: only the warning channel is inspected.
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



def check_warns_returning(
    test_name: str,
    callable_object: Callable[[], Any],
    expected_warning_category: type,
    interpretation: str = "",
) -> Any:
    """
    Like `check_warns`, but returns the callable's result so the caller can
    go on to make correctness assertions on it.

    Asserts that calling `callable_object()` emits at least one warning of
    `expected_warning_category`, recording a pass/fail on the warning channel
    exactly as `check_warns` does, while capturing the warning locally so it
    never propagates to an outer `filterwarnings='error'` handler.
    """

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        call_result = callable_object()

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

    else:
        REPORT.record_fail(
            test_name,
            f"no {expected_warning_category.__name__} captured.",
            expected=expected_warning_category.__name__,
            actual=[w.category.__name__ for w in captured_warnings] or None,
        )

    return call_result



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



# TEST PROBLEMS



# A perfectly conditioned quadratic: f(x) = 0.5 * sum((x - target)^2). Its
# Hessian is the identity, so every method - even the weak first-order ones -
# can reach the minimizer quickly.
QUADRATIC_TARGET = np.array([1.0, -2.0, 0.5])


def quadratic_bowl(params: np.ndarray) -> float:
    """f(x) = 0.5 * sum((x - target)^2); minimizer at `QUADRATIC_TARGET`."""
    return float(0.5 * np.sum((np.asarray(params) - QUADRATIC_TARGET) ** 2))


def quadratic_bowl_gradient(params: np.ndarray) -> np.ndarray:
    """Gradient of `quadratic_bowl`: (x - target)."""
    return np.asarray(params, dtype=float) - QUADRATIC_TARGET


def rosenbrock(params: np.ndarray) -> float:
    """
    The Rosenbrock banana, the canonical hard smooth test: a narrow curved
    valley whose unique minimizer is the all-ones vector with f = 0.
    """
    x = np.asarray(params, dtype=float)

    return float(np.sum(
        100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
    ))


def rosenbrock_gradient(params: np.ndarray) -> np.ndarray:
    """Analytical gradient of `rosenbrock`."""
    x = np.asarray(params, dtype=float)
    gradient = np.zeros_like(x)

    gradient[:-1] = (
        -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    )
    gradient[1:] += 200.0 * (x[1:] - x[:-1] ** 2)

    return gradient


def booth(params: np.ndarray) -> float:
    """Booth's function; smooth, convex, minimizer at (1, 3) with f = 0."""
    x = np.asarray(params, dtype=float)

    return float((x[0] + 2.0 * x[1] - 7.0) ** 2
                 + (2.0 * x[0] + x[1] - 5.0) ** 2)


def rastrigin(params: np.ndarray) -> float:
    """
    Rastrigin's function (A = 10); a smooth bowl studded with a regular grid
    of local minima, with the single global minimizer at the origin (f = 0).
    Used to check that a global method escapes the surrounding traps.
    """
    x = np.asarray(params, dtype=float)

    return float(
        10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))
    )


# A symmetric positive-definite quadratic for the second-order sections, built
# once with a fixed seed so the Hessian is reproducible from run to run.
_SPD_RNG = np.random.default_rng(seed=1234)
_SPD_FACTOR = _SPD_RNG.standard_normal((4, 4))
SPD_MATRIX = _SPD_FACTOR @ _SPD_FACTOR.T + 4.0 * np.eye(4)


def spd_quadratic(params: np.ndarray) -> float:
    """f(x) = 0.5 * x^T A x with A symmetric positive-definite; min at 0."""
    x = np.asarray(params, dtype=float)

    return float(0.5 * x @ SPD_MATRIX @ x)


def spd_quadratic_gradient(params: np.ndarray) -> np.ndarray:
    """Gradient of `spd_quadratic`: A x."""
    return SPD_MATRIX @ np.asarray(params, dtype=float)


def spd_quadratic_hessian(params: np.ndarray) -> np.ndarray:
    """Hessian of `spd_quadratic`: the constant matrix A."""
    return SPD_MATRIX


def spd_quadratic_hvp(params: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Hessian-vector product of `spd_quadratic`: A p, the Hessian being A."""
    return SPD_MATRIX @ np.asarray(vector, dtype=float)



# SECTION 1 - EVERY LOCAL METHOD REACHES THE MINIMUM
#
# The two first-order methods, which the module documents as weak
# general-purpose solvers, are started near the optimum and judged on a relaxed
# band, exactly the regime in which they are meant to be used.



def section_01_local_method_correctness() -> None:
    """
    Run all eight local methods on the perfectly conditioned quadratic bowl
    with the analytical gradient supplied, and check that each one lands on
    `QUADRATIC_TARGET`.

    Each record is (method, starting point, method_params override,
    tolerances override, max_iters override, rtol, atol). The first-order
    methods get a small learning-rate / iteration budget and a near start;
    everything else starts from the origin with default settings.
    """

    section_banner("SECTION 1 - Every local method reaches the minimum")

    far_start = np.zeros(3)
    near_start = QUADRATIC_TARGET + 0.1

    local_method_cases = [
        # (method, start, method_params, tolerances, max_iters, rtol, atol)
        ("conjugate_gradient", far_start, None, None, None, 1e-4, 1e-5),
        ("bfgs", far_start, None, None, None, 1e-4, 1e-5),
        ("lbfgs", far_start, None, None, None, 1e-4, 1e-5),
        ("newton_conjugate_gradient", far_start, None, None, None, 1e-5, 1e-6),
        ("trust_ncg", far_start, None, None, None, 1e-5, 1e-6),
        ("trust_lanczos", far_start, None, None, None, 1e-5, 1e-6),
        (
            "nesterov", near_start,
            {"learning_rate": 0.9, "momentum": 0.2}, {"gradient": 1e-7},
            20000, 1e-3, 1e-4,
        ),
        (
            "adam", near_start,
            {"learning_rate": 0.02}, {"gradient": 1e-5},
            100000, 1e-2, 3e-2,
        ),
    ]

    for (
        method, start, method_params, tolerances, max_iters, rtol, atol
    ) in local_method_cases:

        test_block(
            test_name=f"1.reaches_minimum : {method}",
            description=(
                f"verify {method} drives the quadratic bowl to its known "
                f"minimizer {QUADRATIC_TARGET.tolist()}, running it from "
                f"{np.asarray(start).tolist()} with the analytical gradient "
                f"and comparing the returned parameters; this is the headline "
                f"correctness guarantee, that the method actually finds the "
                f"point it was asked to find."
            ),
        )

        extra_keyword_arguments = {}
        if method_params is not None:
            extra_keyword_arguments["method_params"] = method_params

        if tolerances is not None:
            extra_keyword_arguments["tolerances"] = tolerances

        if max_iters is not None:
            extra_keyword_arguments["max_iters"] = max_iters


        result = minimizer(
            function=quadratic_bowl,
            method=method,
            initial_params=start,
            gradient_function=quadratic_bowl_gradient,
            **extra_keyword_arguments,
        )

        print(f"  RESULT      : final_cost = "
              f"{_format_for_result_line(result.final_cost)}, "
              f"iterations = {result.iteration_number}, "
              f"success = {result.success}")

        check_allclose(
            test_name=f"1.reaches_minimum : {method}",
            actual_value=np.asarray(result.final_params),
            expected_value=QUADRATIC_TARGET,
            relative_tolerance=rtol, absolute_tolerance=atol,
            interpretation=f"{method} located the minimizer.",
        )



# SECTION 2 - CLASSIC HARD TEST FUNCTIONS



def section_02_classic_test_functions() -> None:
    """
    Take the line-search, quasi-Newton and trust-region methods through the
    2-D Rosenbrock valley from the classic hard start (-1.2, 1.0) with the
    analytical gradient, requiring both the minimizer (1, 1) and a success
    verdict. Then solve Booth from the origin with a numerical gradient.
    """

    section_banner("SECTION 2 - Classic hard test functions")

    rosenbrock_minimizer = np.array([1.0, 1.0])
    rosenbrock_start = np.array([-1.2, 1.0])

    strong_methods = [
        "conjugate_gradient", "bfgs", "lbfgs", "trust_ncg", "trust_lanczos",
    ]

    for method in strong_methods:
        test_block(
            test_name=f"2.rosenbrock : {method}",
            description=(
                f"verify {method} converges to the Rosenbrock minimizer "
                f"(1, 1) from the hard start (-1.2, 1.0) and reports success, "
                f"running with the analytical gradient and checking both the "
                f"parameters and the success flag; the narrow curved valley is "
                f"where weak optimisers stall, so clearing it is the real "
                f"proof the method works."
            ),
        )

        result = minimizer(
            function=rosenbrock,
            method=method,
            initial_params=rosenbrock_start,
            gradient_function=rosenbrock_gradient,
        )

        print(f"  RESULT      : final_cost = "
              f"{_format_for_result_line(result.final_cost)}, "
              f"iterations = {result.iteration_number}, "
              f"success = {result.success}")

        reached_minimizer = bool(np.allclose(
            np.asarray(result.final_params), rosenbrock_minimizer,
            rtol=1e-3, atol=1e-4,
        ))

        check_truth(
            test_name=f"2.rosenbrock : {method}",
            condition=reached_minimizer and result.success,
            message_on_pass=(
                f"converged to {np.asarray(result.final_params)} and reported "
                f"success."
            ),
            message_on_fail=(
                f"final = {np.asarray(result.final_params)}, "
                f"success = {result.success}; expected (1, 1) with success."
            ),
        )


    # Booth, solved with a finite-difference gradient (no gradient_function).
    test_block(
        test_name="2.booth_numerical_gradient",
        description=(
            "verify lbfgs solves Booth's function to its minimizer (1, 3) "
            "using a finite-difference gradient, running from the origin with "
            "no gradient_function supplied and comparing the result; this is "
            "the everyday case in which the user has no analytical derivative "
            "and leans on the module's numerical fallback."
        ),
    )

    booth_result = minimizer(
        function=booth,
        method="lbfgs",
        initial_params=np.array([0.0, 0.0]),
    )

    print(f"  RESULT      : final_cost = "
          f"{_format_for_result_line(booth_result.final_cost)}, "
          f"analytical_gradient_provided = "
          f"{booth_result.analytical_gradient_provided}")

    check_allclose(
        test_name="2.booth_numerical_gradient",
        actual_value=np.asarray(booth_result.final_params),
        expected_value=np.array([1.0, 3.0]),
        relative_tolerance=1e-3, absolute_tolerance=1e-4,
        interpretation="the numerical-gradient path reaches Booth's minimizer.",
    )



# SECTION 3 - ANALYTICAL VS NUMERICAL GRADIENT



def section_03_analytical_vs_numerical_gradient() -> None:
    """
    Minimise Rosenbrock twice with bfgs, once with the analytical gradient and
    once with the finite-difference fallback, and check the two minimizers
    agree and that `analytical_gradient_provided` distinguishes them.
    """

    section_banner("SECTION 3 - Analytical vs numerical gradient")

    start = np.array([-1.2, 1.0])

    analytical_result = minimizer(
        function=rosenbrock, method="bfgs", initial_params=start,
        gradient_function=rosenbrock_gradient,
    )
    numerical_result = minimizer(
        function=rosenbrock, method="bfgs", initial_params=start,
    )

    test_block(
        test_name="3.same_optimum",
        description=(
            "verify the analytical-gradient and numerical-gradient runs reach "
            "the same Rosenbrock minimizer, solving once each and comparing "
            "the two parameter vectors with allclose; the gradient source "
            "should change the path and the evaluation count, never the answer."
        ),
    )
    check_allclose(
        test_name="3.same_optimum",
        actual_value=np.asarray(numerical_result.final_params),
        expected_value=np.asarray(analytical_result.final_params),
        relative_tolerance=1e-3, absolute_tolerance=1e-5,
        interpretation="both gradient sources converge to the same point.",
    )

    test_block(
        test_name="3.flag_reflects_source",
        description=(
            "verify analytical_gradient_provided is True for the run that was "
            "given a gradient and False for the one that was not, reading the "
            "flag on each result; callers and the summary rely on this flag to "
            "know whether the reported counts include the cost of differencing."
        ),
    )
    check_truth(
        test_name="3.flag_reflects_source",
        condition=(
            analytical_result.analytical_gradient_provided is True
            and numerical_result.analytical_gradient_provided is False
        ),
        message_on_pass=(
            "the flag is True with a supplied gradient and False without one."
        ),
        message_on_fail=(
            f"analytical={analytical_result.analytical_gradient_provided}, "
            f"numerical={numerical_result.analytical_gradient_provided}; "
            f"expected True / False."
        ),
    )



# SECTION 4 - SECOND-ORDER INFORMATION
#
# The trust-region and Newton-CG methods can take curvature from a full
# Hessian, from a Hessian-vector product, or build it numerically. Feed an
# analytical Hessian and an analytical HVP to an SPD quadratic - solvable in
# essentially one Newton step - and check both the answer and the flags,
# including the documented rule that an HVP takes precedence over a Hessian.



def section_04_second_order_information() -> None:
    """
    Solve the SPD quadratic (unique minimizer at the origin) with trust_ncg
    using, in turn, an analytical Hessian and an analytical Hessian-vector
    product, and verify the answer plus the curvature-source flags.
    """

    section_banner("SECTION 4 - Second-order information (Hessian and HVP)")

    start = _SPD_RNG.standard_normal(4)


    # Hessian path.
    test_block(
        test_name="4.analytical_hessian",
        description=(
            "verify trust_ncg with an analytical Hessian drives the SPD "
            "quadratic to the origin and sets analytical_hessian_provided, "
            "solving from a random start and checking the minimizer and the "
            "flag; an exact Hessian on a quadratic is the easiest possible "
            "Newton step, so anything but the origin signals a wiring bug."
        ),
    )

    hessian_result = minimizer(
        function=spd_quadratic, method="trust_ncg", initial_params=start,
        gradient_function=spd_quadratic_gradient,
        hessian_function=spd_quadratic_hessian,
    )
    print(f"  RESULT      : analytical_hessian_provided = "
          f"{hessian_result.analytical_hessian_provided}")

    hessian_ok = (
        bool(np.allclose(np.asarray(hessian_result.final_params), 0.0,
                         atol=1e-6))
        and hessian_result.analytical_hessian_provided
    )

    check_truth(
        test_name="4.analytical_hessian",
        condition=hessian_ok,
        message_on_pass="reached the origin with the Hessian flag set.",
        message_on_fail=(
            f"final = {np.asarray(hessian_result.final_params)}, "
            f"flag = {hessian_result.analytical_hessian_provided}."
        ),
    )


    # Hessian-vector-product path.
    test_block(
        test_name="4.hessian_vector_product",
        description=(
            "verify trust_ncg with an analytical Hessian-vector product "
            "reaches the origin and sets hessian_vector_product_provided, "
            "solving from the same start and checking the minimizer and the "
            "flag; the HVP path is the O(n)-memory route that large problems "
            "are meant to take, so it has to be exercised on its own."
        ),
    )

    hvp_result = minimizer(
        function=spd_quadratic, method="trust_ncg", initial_params=start,
        gradient_function=spd_quadratic_gradient,
        hessian_vector_function=spd_quadratic_hvp,
    )
    print(f"  RESULT      : hessian_vector_product_provided = "
          f"{hvp_result.hessian_vector_product_provided}")

    hvp_ok = (
        bool(np.allclose(np.asarray(hvp_result.final_params), 0.0, atol=1e-6))
        and hvp_result.hessian_vector_product_provided
    )

    check_truth(
        test_name="4.hessian_vector_product",
        condition=hvp_ok,
        message_on_pass="reached the origin with the HVP flag set.",
        message_on_fail=(
            f"final = {np.asarray(hvp_result.final_params)}, "
            f"flag = {hvp_result.hessian_vector_product_provided}."
        ),
    )


    # Precedence: HVP wins when both are supplied.
    test_block(
        test_name="4.hvp_takes_precedence",
        description=(
            "verify that when both a Hessian and a Hessian-vector product are "
            "supplied the HVP wins, passing both and checking that "
            "hessian_vector_product_provided is True, as the docstring "
            "promises; the precedence rule matters because the HVP is the "
            "cheaper of the two and should not be silently shadowed."
        ),
    )

    both_result = check_warns_returning(
        test_name="4.both_hessian_and_hvp_emits_warning",
        callable_object=lambda: minimizer(
            function=spd_quadratic, method="trust_ncg", initial_params=start,
            gradient_function=spd_quadratic_gradient,
            hessian_function=spd_quadratic_hessian,
            hessian_vector_function=spd_quadratic_hvp,
        ),
        expected_warning_category=RuntimeWarning,
        interpretation=(
            "passing both hessian_function and hessian_vector_function "
            "must emit a RuntimeWarning that hessian_function is ignored in "
            "favour of the vector-product form; the run itself must still "
            "converge."
        ),
    )

    check_truth(
        test_name="4.hvp_takes_precedence",
        condition=both_result.hessian_vector_product_provided is True,
        message_on_pass=(
            "the HVP flag is set even though a full Hessian was also given."
        ),
        message_on_fail=(
            "the HVP did not take precedence over the supplied Hessian."
        ),
    )



# SECTION 5 - STARTING-POINT INDEPENDENCE
#
# A convex problem has a single minimizer, so a correct local method must
# reach it regardless of where it starts. Disagreement between starts would
# point to a path-dependent bug.



def section_05_starting_point_independence() -> None:
    """
    Solve the convex quadratic bowl from several very different starting
    points with lbfgs and check every run lands on the same minimizer.
    """

    section_banner("SECTION 5 - Starting-point independence")

    starting_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([-7.0, 4.0, -3.0]),
        np.array([100.0, -100.0, 50.0]),
    ]

    for start in starting_points:
        test_block(
            test_name=f"5.convex_start : {start.tolist()}",
            description=(
                f"verify lbfgs reaches the quadratic bowl minimizer "
                f"{QUADRATIC_TARGET.tolist()} from {start.tolist()}, solving "
                f"with the analytical gradient and comparing; on a convex "
                f"problem the start cannot change the answer, so a mismatch "
                f"here exposes a path-dependent defect."
            ),
        )

        result = minimizer(
            function=quadratic_bowl, method="lbfgs", initial_params=start,
            gradient_function=quadratic_bowl_gradient,
        )

        check_allclose(
            test_name=f"5.convex_start : {start.tolist()}",
            actual_value=np.asarray(result.final_params),
            expected_value=QUADRATIC_TARGET,
            relative_tolerance=1e-4, absolute_tolerance=1e-5,
            interpretation="the start did not change the minimizer.",
        )



# SECTION 6 - GLOBAL SEARCH: DIRECT
#
# DIRECT is the deterministic, derivative-free global solver: it subdivides a
# finite box and never calls the gradient. We give it a bounded bowl whose
# minimizer sits in the interior and check it is located, that no gradient was
# ever evaluated, and that the bounded / global flags are set.



def section_06_global_direct() -> None:
    """
    Minimise a bounded bowl with the minimizer in the box interior using
    DIRECT, and check the located point, the zero gradient count, and the
    is_bounded / is_global_method flags.
    """

    section_banner("SECTION 6 - Global search: DIRECT")

    interior_minimizer = np.array([0.3, -0.4])

    def bounded_bowl(params: np.ndarray) -> float:
        return float(np.sum((np.asarray(params) - interior_minimizer) ** 2))

    result = minimizer(
        function=bounded_bowl, method="direct",
        bounds=([-1.0, -1.0], [1.0, 1.0]),
        method_params={"max_diameter": 0.01, 'f_min': 1e-4},
    )

    test_block(
        test_name="6.locates_interior_minimum",
        description=(
            "verify DIRECT locates the interior minimizer (0.3, -0.4) of a "
            "bounded bowl, running it on the unit box and comparing within a "
            "subdivision-sized band; DIRECT trades the last digits for "
            "determinism and a global guarantee, so the band is looser than "
            "for the local methods on purpose."
        ),
    )

    print(f"  RESULT      : final_cost = "
          f"{_format_for_result_line(result.final_cost)}, "
          f"grad_evals_number = {result.grad_evals_number}")

    check_allclose(
        test_name="6.locates_interior_minimum",
        actual_value=np.asarray(result.final_params),
        expected_value=interior_minimizer,
        relative_tolerance=5e-2, absolute_tolerance=5e-2,
        interpretation="DIRECT found the right basin to subdivision accuracy.",
    )


    test_block(
        test_name="6.derivative_free_and_flags",
        description=(
            "verify DIRECT never evaluates a gradient and reports itself as a "
            "bounded global method, reading grad_evals_number, "
            "is_global_method and is_bounded off the result; these are the "
            "properties a caller uses to reason about cost and about which "
            "guarantees apply."
        ),
    )

    check_truth(
        test_name="6.derivative_free_and_flags",
        condition=(
            result.grad_evals_number == 0
            and result.is_global_method is True
            and result.is_local_method is False
            and result.is_bounded is True
        ),
        message_on_pass=(
            "zero gradient calls; flagged as a bounded global method."
        ),
        message_on_fail=(
            f"grad_evals={result.grad_evals_number}, "
            f"global={result.is_global_method}, bounded={result.is_bounded}."
        ),
    )



# SECTION 7 - GLOBAL SEARCH: DUAL ANNEALING
#
# Dual annealing is the stochastic global solver, the one that has to climb out
# of local minima. Rastrigin's regular grid of traps is the standard probe.
# With a fixed seed the run is reproducible, so we both check it reaches the
# global basin and that two identical seeds give identical results.



def section_07_global_annealing() -> None:
    """
    Minimise 2-D Rastrigin with dual annealing under a fixed seed and a capped
    evaluation budget, check it reaches the global basin at the origin, and
    check that repeating the run with the same seed reproduces it exactly.
    """

    section_banner("SECTION 7 - Global search: dual annealing")

    bounds = ([-5.12, -5.12], [5.12, 5.12])
    annealing_params = {"seed": 0, "max_fun_evals": 20000}

    first_result = minimizer(
        function=rastrigin, method="annealing", bounds=bounds,
        method_params=annealing_params,
    )

    test_block(
        test_name="7.escapes_local_minima",
        description=(
            "verify dual annealing reaches the global minimum of 2-D Rastrigin "
            "at the origin, running it inside the standard box with a fixed "
            "seed and checking the final cost is well below the nearest local "
            "minimum's value; clearing that gap is the whole point of a global "
            "method, since a local solver would settle in the first trap it "
            "meets."
        ),
    )

    print(f"  RESULT      : final_cost = "
          f"{_format_for_result_line(first_result.final_cost)}, "
          f"final_params = {np.asarray(first_result.final_params)}")

    check_truth(
        test_name="7.escapes_local_minima",
        condition=(
            first_result.final_cost is not None
            and first_result.final_cost < 0.5
            and bool(np.allclose(np.asarray(first_result.final_params), 0.0,
                                 atol=0.1))
        ),
        message_on_pass=(
            f"reached the global basin (cost = {first_result.final_cost:.2e})."
        ),
        message_on_fail=(
            f"cost = {first_result.final_cost}, "
            f"params = {np.asarray(first_result.final_params)}; "
            f"the search did not reach the global basin."
        ),
    )


    test_block(
        test_name="7.seed_reproducibility",
        description=(
            "verify two annealing runs with the same seed produce the same "
            "result, repeating the run and comparing with allclose; a fixed "
            "seed is the only handle a user has on a stochastic method, so it "
            "must make the run deterministic."
        ),
    )

    second_result = minimizer(
        function=rastrigin, method="annealing", bounds=bounds,
        method_params=annealing_params,
    )

    check_truth(
        test_name="7.seed_reproducibility",
        condition=first_result.allclose(second_result,
                                        relative_tolerance=1e-9,
                                        absolute_tolerance=1e-12),
        message_on_pass="the same seed reproduced the run exactly.",
        message_on_fail="the same seed gave a different result.",
    )



# SECTION 8 - BOX BOUNDS



def section_08_box_bounds() -> None:
    """
    Minimise a sphere centred at (3, 3) - outside the feasible box - subject
    to the box [-1, 1]^2, and check the solution sits at the nearest corner
    (1, 1) and that the run is reported as bounded but not constrained.
    """

    section_banner("SECTION 8 - Box bounds via augmented Lagrangian")

    def shifted_sphere(params: np.ndarray) -> float:
        return float(np.sum((np.asarray(params) - 3.0) ** 2))

    result = minimizer(
        function=shifted_sphere, method="lbfgs",
        initial_params=np.array([0.0, 0.0]),
        bounds=([-1.0, -1.0], [1.0, 1.0]),
    )

    test_block(
        test_name="8.optimum_on_the_box",
        description=(
            "verify a bounded lbfgs run settles at the box corner (1, 1) when "
            "the free optimum (3, 3) lies outside the feasible region, solving "
            "on [-1, 1]^2 and comparing; this is the defining behaviour of a "
            "bound-constrained solve and the case where a missing projection "
            "would show up immediately."
        ),
    )

    print(f"  RESULT      : final_cost = "
          f"{_format_for_result_line(result.final_cost)}, "
          f"is_bounded = {result.is_bounded}")

    check_allclose(
        test_name="8.optimum_on_the_box",
        actual_value=np.asarray(result.final_params),
        expected_value=np.array([1.0, 1.0]),
        relative_tolerance=1e-3, absolute_tolerance=1e-3,
        interpretation="the solution rests on the active corner.",
    )


    test_block(
        test_name="8.bounded_not_constrained",
        description=(
            "verify the run reports is_bounded True and is_constrained False "
            "with a zero constraint count, reading the flags off the result; "
            "bounds and general constraints are tracked separately, and a "
            "caller branches on the difference."
        ),
    )

    check_truth(
        test_name="8.bounded_not_constrained",
        condition=(
            result.is_bounded is True
            and result.is_constrained is False
            and result.constraint_count == 0
        ),
        message_on_pass="flagged as bounded with no general constraints.",
        message_on_fail=(
            f"is_bounded={result.is_bounded}, "
            f"is_constrained={result.is_constrained}, "
            f"constraint_count={result.constraint_count}."
        ),
    )



# SECTION 9 - EQUALITY CONSTRAINT



def section_09_equality_constraint() -> None:
    """
    Minimise the squared distance to (3, 2) subject to x + y = 3 and check the
    solution is the known foot of the perpendicular (2, 1), that the constraint
    holds there, and that the constraint metadata is recorded.
    """

    section_banner("SECTION 9 - Equality constraint")

    def distance_to_point(params: np.ndarray) -> float:
        x = np.asarray(params)
        return float((x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2)

    equality_constraints = (
        {"type": "eq", "fun": lambda x: x[0] + x[1] - 3.0},
    )

    result = minimizer(
        function=distance_to_point, method="trust_lanczos",
        initial_params=np.array([0.0, 0.0]),
        constraints=equality_constraints,
    )


    test_block(
        test_name="9.foot_of_perpendicular",
        description=(
            "verify the equality-constrained solve lands on the known closest "
            "feasible point (2, 1), minimising the distance to (3, 2) on the "
            "line x + y = 3 and comparing; the answer is fixed by geometry, so "
            "the augmented-Lagrangian loop has exactly one number to hit."
        ),
    )

    print(f"  RESULT      : final = {np.asarray(result.final_params)}, "
          f"constraint value = "
          f"{result.final_params[0] + result.final_params[1] - 3.0:.2e}")

    check_allclose(
        test_name="9.foot_of_perpendicular",
        actual_value=np.asarray(result.final_params),
        expected_value=np.array([2.0, 1.0]),
        relative_tolerance=1e-3, absolute_tolerance=5e-3,
        interpretation="the constrained optimum matches the geometric answer.",
    )


    test_block(
        test_name="9.constraint_satisfied_and_counted",
        description=(
            "verify the equality holds at the solution and that the result "
            "records one constraint, checking |x + y - 3| against the "
            "augmented-Lagrangian tolerance and reading constraint_count; a "
            "right-looking point that quietly violates the constraint would be "
            "the worst kind of silent failure."
        ),
    )

    constraint_residual = abs(
        float(result.final_params[0] + result.final_params[1] - 3.0)
    )

    check_truth(
        test_name="9.constraint_satisfied_and_counted",
        condition=(
            constraint_residual < 1e-3
            and result.is_constrained is True
            and result.constraint_count == 1
        ),
        message_on_pass=(
            f"constraint residual {constraint_residual:.2e} is within "
            f"tolerance; one constraint recorded."
        ),
        message_on_fail=(
            f"residual = {constraint_residual:.2e}, "
            f"constraint_count = {result.constraint_count}."
        ),
    )



# SECTION 10 - INEQUALITY CONSTRAINT
#
# Inequalities are written c(x) >= 0. Place the unconstrained optimum (the
# origin) outside the feasible half-plane x + y >= 2, so the constrained
# optimum is forced onto the active boundary at the hand-computable point
# (1, 1).



def section_10_inequality_constraint() -> None:
    """
    Minimise the sphere subject to x + y - 2 >= 0 and check the solution is the
    active-boundary point (1, 1), that it is feasible there, and that the
    constraint is counted.
    """

    section_banner("SECTION 10 - Inequality constraint")

    def sphere(params: np.ndarray) -> float:
        return float(np.sum(np.asarray(params) ** 2))

    inequality_constraints = (
        {"type": "ineq", "fun": lambda x: x[0] + x[1] - 2.0},
    )

    result = minimizer(
        function=sphere, method="lbfgs",
        initial_params=np.array([0.5, 0.5]),
        constraints=inequality_constraints,
    )


    test_block(
        test_name="10.optimum_on_active_boundary",
        description=(
            "verify the inequality-constrained solve lands on the active "
            "boundary point (1, 1), minimising the sphere over x + y >= 2 from "
            "an interior infeasible start and comparing; with the free optimum "
            "pushed out of the feasible set, the closest feasible point is "
            "fixed and known."
        ),
    )

    boundary_value = float(
        result.final_params[0] + result.final_params[1] - 2.0
    )

    print(f"  RESULT      : final = {np.asarray(result.final_params)}, "
          f"c(x) = {boundary_value:.2e}")

    check_allclose(
        test_name="10.optimum_on_active_boundary",
        actual_value=np.asarray(result.final_params),
        expected_value=np.array([1.0, 1.0]),
        relative_tolerance=1e-2, absolute_tolerance=5e-3,
        interpretation="the optimum rests on the active inequality boundary.",
    )


    test_block(
        test_name="10.feasible_at_solution",
        description=(
            "verify the inequality is satisfied at the solution, checking "
            "c(x) = x + y - 2 is non-negative up to the augmented-Lagrangian "
            "tolerance; the c(x) >= 0 convention is the module's contract, "
            "so a feasible-looking point with a small negative residual would "
            "mean the sign handling is off."
        ),
    )

    check_truth(
        test_name="10.feasible_at_solution",
        condition=boundary_value > -1e-3 and result.constraint_count == 1,
        message_on_pass=(
            f"c(x) = {boundary_value:.2e} is feasible; one constraint counted."
        ),
        message_on_fail=(
            f"c(x) = {boundary_value:.2e} violates the >= 0 convention, "
            f"or constraint_count = {result.constraint_count}."
        ),
    )



# SECTION 11 - OPTIMIZATIONRESULT API SURFACE
#
# The result object is the only thing the caller actually holds, so its every
# property, conversion, comparison and string form is checked here against a
# single known run.



def section_11_result_api_surface() -> None:
    """
    Build one known result and exercise the shape / convenience properties,
    the conversion helpers, the summary and repr forms, the equality and
    allclose semantics, the array-like operators, and the unhashability and
    as_float guards.
    """

    section_banner("SECTION 11 - OptimizationResult API surface")

    api_result = minimizer(
        function=rosenbrock, method="trust_ncg",
        initial_params=np.array([0.0, 0.0]),
        gradient_function=rosenbrock_gradient,
    )


    # shape / dtype / scalar-vs-array properties
    test_block(
        test_name="11.shape_properties",
        description=(
            "verify shape, ndim, size, variable_number, dtype, is_array and "
            "is_scalar all agree with the underlying final_params, reading "
            "each off the result and cross-checking against the array; these "
            "are the numpy-style handles a caller reaches for first, and they "
            "must not drift from the data they describe."
        ),
    )

    final_array = np.asarray(api_result.final_params)
    shape_properties_consistent = (
        api_result.shape == final_array.shape == (2,)
        and api_result.ndim == final_array.ndim == 1
        and api_result.size == final_array.size == 2
        and api_result.variable_number == 2
        and api_result.dtype == final_array.dtype
        and api_result.is_array
        and not api_result.is_scalar
    )

    check_truth(
        test_name="11.shape_properties",
        condition=shape_properties_consistent,
        message_on_pass="every shape/dtype property matches final_params.",
        message_on_fail="a shape/dtype property disagrees with final_params.",
    )


    # convenience / classification properties
    test_block(
        test_name="11.convenience_properties",
        description=(
            "verify the classification and throughput properties are "
            "self-consistent, checking that the method is reported as local "
            "and not global or constrained, that total_evaluations is the sum "
            "of the two counters, and that the per-iteration and per-second "
            "figures are floats; the summary and any cost accounting are built "
            "directly on these."
        ),
    )

    throughput_well_formed = (
        api_result.is_local_method is True
        and api_result.is_global_method is False
        and api_result.is_constrained is False
        and api_result.total_evaluations
            == api_result.func_evals_number + api_result.grad_evals_number
        and isinstance(api_result.func_evals_per_iteration, float)
        and isinstance(api_result.average_iteration_time, float)
    )

    check_truth(
        test_name="11.convenience_properties",
        condition=throughput_well_formed,
        message_on_pass="classification and throughput properties cohere.",
        message_on_fail="a convenience property is inconsistent.",
    )


    # conversions
    test_block(
        test_name="11.conversions",
        description=(
            "verify as_array returns an independent writable copy, to_list "
            "yields a length-2 Python list, and to_dict carries the documented "
            "key set, mutating the copy to confirm it does not touch the "
            "result and checking the dict keys; these conversions are the "
            "bridge from the wrapper to plain Python and NumPy."
        ),
    )
    writable_copy = api_result.as_array()
    writable_copy[0] = 999.0
    original_untouched = float(api_result.final_params[0]) != 999.0

    list_form = api_result.to_list()
    list_form_ok = isinstance(list_form, list) and len(list_form) == 2

    dict_form = api_result.to_dict()
    expected_dict_keys = {
        "success", "final_params", "final_cost", "termination_reason",
        "func_evals_number", "grad_evals_number", "elapsed_time",
        "iteration_number", "method", "initial_params", "bounds",
        "max_iters", "tolerances", "method_params", "gradient_point_number",
        "analytical_gradient_provided", "analytical_hessian_provided",
        "hessian_vector_product_provided", "constraint_count",
        "constrained_params",
    }
    dict_keys_match = set(dict_form.keys()) == expected_dict_keys

    check_truth(
        test_name="11.conversions",
        condition=(
            writable_copy.flags.writeable and original_untouched
            and list_form_ok and dict_keys_match
        ),
        message_on_pass="every conversion behaves as documented.",
        message_on_fail=(
            f"writable={writable_copy.flags.writeable}, "
            f"original_untouched={original_untouched}, "
            f"list_ok={list_form_ok}, dict_keys_match={dict_keys_match}."
        ),
    )


    # as_float on a multi-variable result must raise
    test_block(
        test_name="11.as_float_non_scalar_raises",
        description=(
            "verify as_float raises ValueError on a multi-variable result, "
            "calling it on this 2-D run and catching; otherwise as_float would "
            "silently collapse a vector and hide the dimensionality from the "
            "caller."
        ),
    )

    check_raises(
        test_name="11.as_float_non_scalar_raises",
        callable_object=api_result.as_float,
        expected_exception_type=ValueError,
    )


    # as_float on a genuine single-variable result works
    test_block(
        test_name="11.as_float_scalar_ok",
        description=(
            "verify as_float returns a finite float on a single-variable "
            "problem, solving a 1-D quadratic and converting; the scalar case "
            "is the one place as_float is meant to be used, so it has to work "
            "there cleanly."
        ),
    )

    scalar_result = minimizer(
        function=lambda x: float((x[0] - 2.0) ** 2),
        method="bfgs", initial_params=np.array([0.0]),
    )
    scalar_value = scalar_result.as_float()

    check_truth(
        test_name="11.as_float_scalar_ok",
        condition=(
            isinstance(scalar_value, float)
            and math.isfinite(scalar_value)
        ),
        message_on_pass=f"as_float returned {scalar_value:.6f}.",
        message_on_fail="as_float did not return a finite float on a scalar.",
    )


    # summary compact and full
    test_block(
        test_name="11.summary_compact_and_full",
        description=(
            "verify summary('compact') and summary('full') each return "
            "multi-line strings with the full form the longer of the two, and "
            "that str(result) routes to the compact one, calling each and "
            "comparing; summary is the human-facing inspection path and str is "
            "what shows up in logs and tracebacks."
        ),
    )

    compact_summary = api_result.summary("compact")
    full_summary = api_result.summary("full")
    summaries_well_formed = (
        "\n" in compact_summary and "\n" in full_summary
        and len(full_summary) > len(compact_summary)
        and str(api_result) == compact_summary
    )

    check_truth(
        test_name="11.summary_compact_and_full",
        condition=summaries_well_formed,
        message_on_pass=(
            "both summaries are multi-line; str routes to the compact form."
        ),
        message_on_fail="summary outputs are inconsistent.",
    )


    # repr
    test_block(
        test_name="11.repr_non_empty",
        description=(
            "verify repr(result) returns a non-empty string, calling repr and "
            "checking it; a usable repr is what makes the object legible in a "
            "debugger or a list."
        ),
    )

    repr_text = repr(api_result)

    check_truth(
        test_name="11.repr_non_empty",
        condition=isinstance(repr_text, str) and len(repr_text) > 0,
        message_on_pass=f"repr is {repr_text!r}",
        message_on_fail="repr is empty or non-string.",
    )


    # == strict, allclose lenient
    test_block(
        test_name="11.equality_semantics",
        description=(
            "verify == is strict while allclose compares values only, solving "
            "Rosenbrock again with the same settings (expecting ==) and once "
            "more with lbfgs (expecting not == but allclose), since the "
            "deterministic trust_ncg run should reproduce bit-for-bit while a "
            "different method reaches the same optimum by a different path; "
            "the two operators serve memoisation and cross-method sanity "
            "respectively."
        ),
    )

    identical_run = minimizer(
        function=rosenbrock, method="trust_ncg",
        initial_params=np.array([0.0, 0.0]),
        gradient_function=rosenbrock_gradient,
    )

    different_method_run = minimizer(
        function=rosenbrock, method="lbfgs",
        initial_params=np.array([0.0, 0.0]),
        gradient_function=rosenbrock_gradient,
    )

    equality_strict = (api_result == identical_run)
    inequality_via_method = not (api_result == different_method_run)
    allclose_across_methods = api_result.allclose(
        different_method_run, relative_tolerance=1e-3, absolute_tolerance=1e-4,
    )

    check_truth(
        test_name="11.equality_semantics",
        condition=(
            equality_strict
            and inequality_via_method
            and allclose_across_methods
        ),
        message_on_pass="== and allclose follow the documented split.",
        message_on_fail=(
            f"strict_equal={equality_strict}, "
            f"differs_by_method={inequality_via_method}, "
            f"allclose={allclose_across_methods}."
        ),
    )


    # allclose with a wrong type raises
    test_block(
        test_name="11.allclose_wrong_type",
        description=(
            "verify allclose with a non-result argument raises TypeError, "
            "passing a bare ndarray and catching; otherwise the comparison "
            "would silently degrade into a NumPy operation and skip the type "
            "contract."
        ),
    )

    check_raises(
        test_name="11.allclose_wrong_type",
        callable_object=lambda: api_result.allclose(np.zeros(2)),
        expected_exception_type=TypeError,
    )


    # array-like behaviour
    test_block(
        test_name="11.array_like_behaviour",
        description=(
            "verify np.asarray(result), result[idx], len(result), iteration "
            "and `value in result` all forward to final_params, exercising "
            "each and checking the wrapper behaves as the inner array would; "
            "these operators are what let the result drop into any expression "
            "that expects a NumPy array."
        ),
    )

    converts_to_array = np.asarray(api_result).shape == (2,)
    indexing_returns_scalar = np.ndim(api_result[0]) == 0

    length_is_two = len(api_result) == 2
    iterates_twice = sum(1 for _ in api_result) == 2

    array_like_ok = (
        converts_to_array and indexing_returns_scalar
        and length_is_two and iterates_twice
    )

    check_truth(
        test_name="11.array_like_behaviour",
        condition=array_like_ok,
        message_on_pass="every array-like operator forwards correctly.",
        message_on_fail="an array-like operator misbehaves.",
    )


    # unhashability
    test_block(
        test_name="11.unhashable",
        description=(
            "verify results are explicitly unhashable, calling hash(result) 2 "
            "and catching the TypeError; a value-equal object backed by a "
            "soft-immutable array would quietly break dict and set invariants "
            "if it were hashable."
        ),
    )

    check_raises(
        test_name="11.unhashable",
        callable_object=lambda: hash(api_result),
        expected_exception_type=TypeError,
    )



# SECTION 12 - PARAMETER VALIDATION
#
# Every documented error path is exercised once, so that malformed input fails
# loudly at the call site rather than silently producing a wrong answer later.
# The expected type is recorded per case: most are ValueError, but a few
# wrong-type inputs are TypeError.



def section_12_parameter_validation() -> None:
    """
    Enumerate the documented ValueError and TypeError paths of `minimizer`:
    a bad method, a missing start, unknown / negative / malformed settings,
    an unusable bounds object, a required bound left as None, and the three
    malformed-constraint cases.
    """

    section_banner("SECTION 12 - Parameter validation")

    simple_objective = lambda x: float(np.sum(np.asarray(x) ** 2))

    invalid_calls = [
        # (test_name, description, callable, expected_exception_type)
        (
            "12.invalid.unknown_method",
            "reject a method that is not in the catalogue, since dispatching "
            "on an unknown name has no defined behaviour.",
            lambda: minimizer(simple_objective, method="non-existant",
                              initial_params=[0.0]),
            ValueError,
        ),
        (
            "12.invalid.missing_initial_params",
            "reject a local method called without a starting point, since "
            "every local method needs one to iterate from.",
            lambda: minimizer(simple_objective, method="bfgs"),
            ValueError,
        ),
        (
            "12.invalid.unknown_method_param_key",
            "reject an unknown key in method_params, since a silently ignored "
            "typo would leave the user thinking a setting took effect.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0],
                              method_params={"not_a_real_key": 1}),
            ValueError,
        ),
        (
            "12.invalid.unknown_tolerance_key",
            "reject an unknown key in tolerances, for the same reason a bad "
            "method_params key is rejected.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0],
                              tolerances={"not_a_tolerance": 1e-6}),
            ValueError,
        ),
        (
            "12.invalid.negative_tolerance",
            "reject a negative tolerance, since a convergence threshold below "
            "zero can never be met and signals a mistake.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0],
                              tolerances={"gradient": -1.0}),
            ValueError,
        ),
        (
            "12.invalid.max_iters_zero",
            "reject max_iters = 0, since an optimiser that may not take "
            "a single step is meaningless.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0], max_iters=0),
            ValueError,
        ),
        (
            "12.invalid.max_iters_bad_string",
            "reject a max_iters string other than 'auto', since no other "
            "string value has a meaning.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0], max_iters="lots"),
            ValueError,
        ),
        (
            "12.invalid.max_iters_float",
            "reject a non-integer max_iters, since an iteration count must be "
            "a whole number.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0], max_iters=1.5),
            TypeError,
        ),
        (
            "12.invalid.verbose_frequency_zero",
            "reject verbose_frequency = 0, since a printing cadence of zero "
            "iterations is undefined.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0], verbose_frequency=0),
            ValueError,
        ),
        (
            "12.invalid.verbose_frequency_bad_string",
            "reject a verbose_frequency string other than 'auto', for the same "
            "reason as the max_iters string.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0], verbose_frequency="often"),
            TypeError,
        ),
        (
            "12.invalid.bounds_not_unpackable",
            "reject a bounds object that cannot be split into (lower, upper), "
            "since the pair is the only shape the box constraint understands.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0], bounds=5),
            ValueError,
        ),
        (
            "12.invalid.required_bound_is_none",
            "reject a None lower or upper bound for a method that requires "
            "finite bounds, since DIRECT and annealing search inside the box "
            "and cannot proceed without both sides.",
            lambda: minimizer(simple_objective, method="direct",
                              bounds=(None, [1.0, 1.0])),
            ValueError,
        ),
        (
            "12.invalid.constraint_missing_keys",
            "reject a constraint dict missing 'type' or 'fun', since both are "
            "needed to know what the constraint is and how to evaluate it.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0, 0.0],
                              constraints=({"fun": lambda x: x[0]},)),
            ValueError,
        ),
        (
            "12.invalid.constraint_bad_type",
            "reject a constraint 'type' that is neither 'eq' nor 'ineq', since "
            "the sign convention depends entirely on which one it is.",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0, 0.0],
                              constraints=({"type": "non-existant",
                                           "fun": lambda x: x[0]},)),
            ValueError,
        ),
        (
            "12.invalid.constraint_fun_not_callable",
            "reject a constraint whose 'fun' is not callable, since the outer "
            "loop has to be able to evaluate c(x).",
            lambda: minimizer(simple_objective, method="bfgs",
                              initial_params=[0.0, 0.0],
                              constraints=({"type": "eq", "fun": 5},)),
            TypeError,
        ),
    ]

    for test_name, description, raising_call, exception_type in invalid_calls:
        test_block(test_name=test_name, description=description)
        check_raises(
            test_name=test_name,
            callable_object=raising_call,
            expected_exception_type=exception_type,
        )



# SECTION 13 - WARNING EMISSION



def section_13_warning_emission() -> None:
    """
    Pass a starting point to DIRECT (a global, bounds-only method) and confirm
    a RuntimeWarning is emitted telling the caller the start will be ignored.
    """

    section_banner("SECTION 13 - Warning emission")

    test_block(
        test_name="13.initial_params_for_global_method",
        description=(
            "verify supplying initial_params to a global method emits a "
            "RuntimeWarning, calling DIRECT with both bounds and an ignored "
            "start and inspecting the warning channel; the start has no effect "
            "on a box search, so the warning is the only signal that the "
            "caller's intent and the method disagree."
        ),
    )

    check_warns(
        test_name="13.initial_params_for_global_method",
        callable_object=lambda: minimizer(
            function=lambda x: float(np.sum(np.asarray(x) ** 2)),
            method="direct",
            initial_params=[0.0, 0.0],
            bounds=([-1.0, -1.0], [1.0, 1.0]),
            method_params={"max_fun_evals": 2000},
        ),
        expected_warning_category=RuntimeWarning,
    )



# SECTION 14 - CALLBACK AND TRAJECTORY
#
# The per-iteration callback is how a caller plots a convergence curve or
# stops early on a custom criterion. Record the cost at each iteration and
# check the callback fired and that the cost fell from start to finish.



def section_14_callback_and_trajectory() -> None:
    """
    Run lbfgs on Rosenbrock with a callback that records the iteration cost,
    and check the callback was invoked, that the recorded cost decreased, and
    that the final cost is well below the starting cost.
    """

    section_banner("SECTION 14 - Callback and trajectory")

    recorded_costs: List[float] = []

    def record_iteration(*callback_arguments, **callback_keywords) -> None:
        # The documented local-method signature is
        # (iter_num, params, cost, gradient, gradient_norm, alpha, direction);
        # reading positionally keeps this robust to method-family differences.
        if len(callback_arguments) >= 3:
            recorded_costs.append(float(callback_arguments[2]))

    start = np.array([-1.2, 1.0])
    starting_cost = rosenbrock(start)

    result = minimizer(
        function=rosenbrock, method="lbfgs", initial_params=start,
        gradient_function=rosenbrock_gradient,
        callback=record_iteration,
    )


    test_block(
        test_name="14.callback_fires_and_cost_falls",
        description=(
            "verify the callback is invoked each iteration and the cost it "
            "sees falls over the run, recording the per-iteration cost and "
            "comparing the first and last entries against the starting and "
            "final costs; a callback that never fires or sees a "
            "non-decreasing cost would make convergence monitoring useless."
        ),
    )

    callback_fired = len(recorded_costs) >= 2

    cost_decreased = (
        callback_fired
        and recorded_costs[-1] < recorded_costs[0]
        and result.final_cost is not None
        and result.final_cost < starting_cost
    )

    print(f"  RESULT      : callback calls = {len(recorded_costs)}, "
          f"first cost = "
          f"{recorded_costs[0] if recorded_costs else float('nan'):.4e}, "
          f"final cost = "
          f"{_format_for_result_line(result.final_cost)}")

    check_truth(
        test_name="14.callback_fires_and_cost_falls",
        condition=cost_decreased,
        message_on_pass=(
            f"callback fired {len(recorded_costs)} times; cost fell to "
            f"{result.final_cost:.2e}."
        ),
        message_on_fail=(
            f"callback_fired={callback_fired}; recorded costs did not decrease "
            f"as expected."
        ),
    )



# SECTION 15 - CUSTOM SETTINGS ARE HONOURED



def section_15_custom_settings() -> None:
    """
    Solve the quadratic bowl while overriding, in turn, the gradient
    tolerance, an lbfgs method parameter, the gradient stencil width, and the
    iteration cap, and check each override is recorded on the result and that
    the run still reaches the minimizer.
    """

    section_banner("SECTION 15 - Custom settings are honoured")


    # Custom tolerance.
    test_block(
        test_name="15.custom_tolerance",
        description=(
            "verify a custom gradient tolerance is stored on the result and "
            "the run still converges, solving with a tighter 1e-10 threshold "
            "and checking both the recorded value and the minimizer; the "
            "tolerances the user passes are the ones the loop should actually "
            "use."
        ),
    )

    tolerance_result = minimizer(
        function=quadratic_bowl, method="bfgs",
        initial_params=np.zeros(3),
        gradient_function=quadratic_bowl_gradient,
        tolerances={"gradient": 1e-10},
    )

    tolerance_ok = (
        abs(tolerance_result.tolerances.get("gradient", 0.0) - 1e-10)
        < 1e-20
        and bool(np.allclose(np.asarray(tolerance_result.final_params),
                             QUADRATIC_TARGET, atol=1e-6))
    )

    check_truth(
        test_name="15.custom_tolerance",
        condition=tolerance_ok,
        message_on_pass="the tighter tolerance was stored and met.",
        message_on_fail=(
            f"recorded tolerances = {tolerance_result.tolerances}; "
            f"final = {np.asarray(tolerance_result.final_params)}."
        ),
    )


    # Custom method parameter.
    test_block(
        test_name="15.custom_method_param",
        description=(
            "verify an lbfgs memory_size override is reflected in "
            "method_params and the run still converges, solving with a memory "
            "of five and checking the recorded value and the minimizer; the "
            "per-method dict the result reports has to be the merged, "
            "effective one."
        ),
    )

    method_param_result = minimizer(
        function=quadratic_bowl, method="lbfgs",
        initial_params=np.zeros(3),
        gradient_function=quadratic_bowl_gradient,
        method_params={"memory_size": 5},
    )

    method_param_ok = (
        method_param_result.method_params.get("memory_size") == 5
        and bool(np.allclose(np.asarray(method_param_result.final_params),
                             QUADRATIC_TARGET, atol=1e-5))
    )

    check_truth(
        test_name="15.custom_method_param",
        condition=method_param_ok,
        message_on_pass="memory_size = 5 was stored and the run converged.",
        message_on_fail=(
            f"method_params = {method_param_result.method_params}."
        ),
    )


    # Custom gradient stencil width.
    test_block(
        test_name="15.gradient_point_number",
        description=(
            "verify a wider numerical-gradient stencil is recorded and still "
            "converges, solving with gradient_point_number = 4 and no "
            "analytical gradient and checking the stored width, the "
            "analytical_gradient_provided flag and the minimizer; the wider "
            "stencil trades function calls for accuracy and the result should "
            "report exactly what was used."
        ),
    )

    stencil_result = minimizer(
        function=quadratic_bowl, method="bfgs",
        initial_params=np.zeros(3),
        gradient_point_number=4,
    )

    stencil_ok = (
        stencil_result.gradient_point_number == 4
        and stencil_result.analytical_gradient_provided is False
        and bool(np.allclose(np.asarray(stencil_result.final_params),
                             QUADRATIC_TARGET, atol=1e-5))
    )

    check_truth(
        test_name="15.gradient_point_number",
        condition=stencil_ok,
        message_on_pass="the 4-point stencil was recorded and converged.",
        message_on_fail=(
            f"gradient_point_number = "
            f"{stencil_result.gradient_point_number}, "
            f"analytical = {stencil_result.analytical_gradient_provided}."
        ),
    )


    # Explicit iteration cap.
    test_block(
        test_name="15.explicit_max_iters",
        description=(
            "verify an explicit integer max_iters is the value the result "
            "reports, solving with a cap of 5000 and reading max_iters back; "
            "the reported cap is what a caller checks to know whether a run "
            "stopped on convergence or on the budget."
        ),
    )
    capped_result = minimizer(
        function=quadratic_bowl, method="bfgs",
        initial_params=np.zeros(3),
        gradient_function=quadratic_bowl_gradient,
        max_iters=5000,
    )
    check_truth(
        test_name="15.explicit_max_iters",
        condition=capped_result.max_iters == 5000,
        message_on_pass="the explicit cap of 5000 is reported unchanged.",
        message_on_fail=f"max_iters reported as {capped_result.max_iters}.",
    )



# SECTION 16 - HIGH-DIMENSIONAL SCALING



def section_16_high_dimensional() -> None:
    """
    Solve a 5000-D well-conditioned quadratic with L-BFGS and the analytical
    gradient, checking the minimizer is recovered, the run reports success,
    and the dimensionality is reported as 5000.
    """

    section_banner("SECTION 16 - High-dimensional scaling")

    dimension = 5000
    target = np.linspace(-1.0, 1.0, dimension)

    def high_dimensional_quadratic(params: np.ndarray) -> float:
        return float(0.5 * np.sum((np.asarray(params) - target) ** 2))

    def high_dimensional_gradient(params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=float) - target

    result = minimizer(
        function=high_dimensional_quadratic, method="lbfgs",
        initial_params=np.zeros(dimension),
        gradient_function=high_dimensional_gradient,
    )


    test_block(
        test_name="16.five_thousand_variable_quadratic",
        description=(
            "verify L-BFGS recovers the minimizer of a 5000-variable quadratic "
            "entry by entry and reports the right dimensionality, solving from "
            "the origin and comparing the full vector; the limited-memory "
            "method is the one meant for large problems, so it has to stay "
            "accurate as the dimension grows."
        ),
    )

    print(f"  RESULT      : variable_number = {result.variable_number}, "
          f"success = {result.success}, "
          f"final_cost = {_format_for_result_line(result.final_cost)}")

    check_allclose(
        test_name="16.five_thousand_variable_quadratic",
        actual_value=np.asarray(result.final_params),
        expected_value=target,
        relative_tolerance=1e-4, absolute_tolerance=1e-5,
        interpretation="the 5000-D minimizer is recovered to a few digits.",
    )



# SECTION 17 - FAILURE AND DEGENERATE RESULTS
#
# The result object is the only thing the caller holds, so its behaviour when a
# run is rejected or converges in zero iterations is tested here. This section
# drives three degenerate cases: a directly constructed no-parameters result
# (the documented None contract), a start the objective cannot be evaluated at
# and a start already sitting on the minimizer.



def section_17_failure_and_degenerate_results() -> None:
    """
    Exercise the result object's failure surface: the None-final_params
    contract built directly, a `minimizer` run whose objective is non-finite at
    the start, and a run that begins already at the minimizer (zero
    iterations), checking the documented None / zero / empty returns and the
    conversions that must raise.
    """

    section_banner("SECTION 17 - Failure and degenerate results")


    test_block(
        test_name="17.none_params_contract",
        description=(
            "verify a result carrying no parameters honours its documented "
            "None / zero / empty contract, constructing an OptimizationResult "
            "with final_params=None and checking every shape property is None "
            "or zero, the array-like operators degrade safely, and as_array, "
            "as_float, indexing and np.asarray each raise; this is the state a "
            "failed run leaves behind and a caller must be able to inspect it "
            "without tripping over a None."
        ),
    )

    empty_result = OptimizationResult(
        success=False, final_params=None, final_cost=None,
        termination_reason="starting point rejected",
        func_evals_number=1, grad_evals_number=0, elapsed_time=0.0,
        iteration_number=0, method="bfgs",
        initial_params=np.array([1.0, 2.0]), bounds=None, max_iters=1000,
        tolerances={"gradient": 1e-6, "relative": 1e-9}, method_params={},
        gradient_point_number=2, analytical_gradient_provided=False,
        analytical_hessian_provided=False,
        hessian_vector_product_provided=False,
        constraint_count=0, constrained_params=None,
    )

    shape_properties_none = (
        empty_result.shape is None and empty_result.ndim is None
        and empty_result.size == 0 and empty_result.variable_number == 0
        and empty_result.dtype is None
    )

    array_like_degrades_safely = (
        len(empty_result) == 0
        and list(empty_result) == []
        and (5.0 in empty_result) is False
        and empty_result.to_list() is None
        and "no_params" in repr(empty_result)
    )

    dict_form = empty_result.to_dict()
    dict_carries_none = (
        dict_form["final_params"] is None and dict_form["final_cost"] is None
    )

    summaries_render = (
        bool(empty_result.summary("compact"))
        and bool(empty_result.summary("full"))
    )

    conversions_raise = True

    for failing_call, expected_exception in [
        (empty_result.as_array, ValueError),
        (empty_result.as_float, ValueError),
        (lambda: empty_result[0], TypeError),
        (lambda: np.asarray(empty_result), ValueError),
    ]:
        try:
            failing_call()
            conversions_raise = False
        except expected_exception:
            pass
        except BaseException:
            conversions_raise = False

    print(f"  RESULT      : shape_properties_none={shape_properties_none}, "
          f"array_like_degrades_safely={array_like_degrades_safely}, "
          f"dict_carries_none={dict_carries_none}, "
          f"summaries_render={summaries_render}, "
          f"conversions_raise={conversions_raise}")

    check_truth(
        test_name="17.none_params_contract",
        condition=(
            shape_properties_none and array_like_degrades_safely
            and dict_carries_none and summaries_render and conversions_raise
        ),
        message_on_pass=(
            "the no-parameters result honours its None / zero / empty contract."
        ),
        message_on_fail=(
            "a no-parameters property or conversion broke its documented "
            "contract."
        ),
    )


    # A start the objective cannot be evaluated at.
    test_block(
        test_name="17.rejected_nonfinite_start",
        description=(
            "verify a run whose objective is non-finite at the start fails "
            "loudly but returns a usable result, minimising a function that "
            "returns NaN and checking success is False, no iteration was "
            "taken, and the termination reason names the invalid value; a "
            "rejected run must report the rejection rather than crash or "
            "claim success."
        ),
    )

    rejected_result = minimizer(
        function=lambda x: float("nan"), method="bfgs",
        initial_params=[1.0, 2.0],
    )

    rejected_as_expected = (
        rejected_result.success is False
        and rejected_result.iteration_number == 0
        and "Invalid value" in rejected_result.termination_reason
        and rejected_result.final_cost is not None
        and not math.isfinite(rejected_result.final_cost)
    )

    print(f"  RESULT      : success={rejected_result.success}, "
          f"iterations={rejected_result.iteration_number}, "
          f"termination={rejected_result.termination_reason[:48]!r}")

    check_truth(
        test_name="17.rejected_nonfinite_start",
        condition=rejected_as_expected,
        message_on_pass=(
            "the non-finite start was rejected with a descriptive reason."
        ),
        message_on_fail=(
            "the non-finite start was not reported as a zero-iteration "
            "failure as expected."
        ),
    )


    # A start already on the minimizer: zero iterations, so the per-iteration
    # throughput properties must return None, not divide by 0.
    test_block(
        test_name="17.zero_iteration_at_optimum",
        description=(
            "verify a run that begins on the minimizer stops in zero "
            "iterations and reports it, starting bfgs exactly at the optimum "
            "of a sphere with the analytical gradient and checking the "
            "iteration count is zero, the termination names the "
            "below-tolerance gradient, and the three per-iteration throughput "
            "properties return None rather than dividing by zero."
        ),
    )

    at_optimum_result = minimizer(
        function=lambda x: float(np.sum(np.asarray(x) ** 2)),
        method="bfgs", initial_params=[0.0, 0.0],
        gradient_function=lambda x: 2.0 * np.asarray(x, dtype=float),
    )

    zero_iteration_as_expected = (
        at_optimum_result.iteration_number == 0
        and "tolerance" in at_optimum_result.termination_reason.lower()
        and at_optimum_result.average_iteration_time is None
        and at_optimum_result.func_evals_per_iteration is None
        and at_optimum_result.grad_evals_per_iteration is None
    )

    print(f"  RESULT      : iterations={at_optimum_result.iteration_number}, "
          f"average_iteration_time={at_optimum_result.average_iteration_time}, "
          f"func_evals_per_iteration="
          f"{at_optimum_result.func_evals_per_iteration}")

    check_truth(
        test_name="17.zero_iteration_at_optimum",
        condition=zero_iteration_as_expected,
        message_on_pass=(
            "a zero-iteration run reports it and the throughput properties "
            "return None."
        ),
        message_on_fail=(
            "the zero-iteration path mis-reported its counts or throughput."
        ),
    )



# SECTION 18 - COMBINED AND GLOBAL CONSTRAINTS



def section_18_combined_constraints() -> None:
    """
    Drive two constrained paths the single-feature sections miss: a local
    method (bfgs) under an inequality constraint and box bounds at once, and a
    global method (annealing) under an equality constraint through the
    augmented-Lagrangian wrapper, checking feasibility and the classification
    flags in each case.
    """

    section_banner("SECTION 18 - Combined and global constraints")

    objective = lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2)


    # Inequality + bounds together on a local method.
    test_block(
        test_name="18.inequality_with_bounds",
        description=(
            "verify a local method honours an inequality constraint and box "
            "bounds simultaneously, minimising a sphere centred at (2, 2) "
            "subject to x0 + x1 <= 1 inside a wide box and checking the "
            "constraint is satisfied and both is_constrained and is_bounded "
            "are set; constraints and bounds share the outer loop."
        ),
    )

    inequality_constraints = (
        {"type": "ineq", "fun": lambda x: 1.0 - (x[0] + x[1])},
    )

    combined_result = minimizer(
        function=objective, method="bfgs", initial_params=[0.0, 0.0],
        constraints=inequality_constraints,
        bounds=([-5.0, -5.0], [5.0, 5.0]),
    )

    constraint_slack = float(
        1.0 - (combined_result.final_params[0]
               + combined_result.final_params[1])
    )

    combined_as_expected = (
        constraint_slack > -1e-3
        and combined_result.is_constrained is True
        and combined_result.is_bounded is True
        and combined_result.constraint_count == 1
    )

    print(f"  RESULT      : final = "
          f"{np.asarray(combined_result.final_params)}, "
          f"1-(x0+x1) = {constraint_slack:.2e}, "
          f"is_constrained = {combined_result.is_constrained}, "
          f"is_bounded = {combined_result.is_bounded}")

    check_truth(
        test_name="18.inequality_with_bounds",
        condition=combined_as_expected,
        message_on_pass=(
            "the inequality and the box are honoured together and both flags "
            "are set."
        ),
        message_on_fail=(
            f"slack = {constraint_slack:.2e}, "
            f"is_constrained = {combined_result.is_constrained}, "
            f"is_bounded = {combined_result.is_bounded}."
        ),
    )


    # Equality constraint on a global method.
    test_block(
        test_name="18.global_with_equality_constraint",
        description=(
            "verify a global method drives an equality constraint to zero "
            "through the augmented-Lagrangian wrapper, running annealing on "
            "the same sphere subject to x0 + x1 = 1 inside a box and checking "
            "the equality holds to tolerance with is_global_method and "
            "is_constrained both set; a global-plus-constraint route, "
            "wrapping a stochastic inner solver."
        ),
    )

    equality_constraints = (
        {"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        global_constrained_result = minimizer(
            function=objective, method="annealing",
            bounds=([-5.0, -5.0], [5.0, 5.0]),
            constraints=equality_constraints,
            method_params={"seed": 1, "max_fun_evals": 1500},
        )

    equality_violation = abs(float(
        global_constrained_result.final_params[0]
        + global_constrained_result.final_params[1] - 1.0
    ))

    global_as_expected = (
        equality_violation < 1e-2
        and global_constrained_result.is_global_method is True
        and global_constrained_result.is_constrained is True
    )

    print(f"  RESULT      : final = "
          f"{np.asarray(global_constrained_result.final_params)}, "
          f"|x0+x1-1| = {equality_violation:.2e}, "
          f"is_global_method = {global_constrained_result.is_global_method}")

    check_truth(
        test_name="18.global_with_equality_constraint",
        condition=global_as_expected,
        message_on_pass=(
            "the global method satisfied the equality constraint to tolerance."
        ),
        message_on_fail=(
            f"|x0+x1-1| = {equality_violation:.2e}, "
            f"is_global_method = {global_constrained_result.is_global_method}, "
            f"is_constrained = {global_constrained_result.is_constrained}."
        ),
    )



# SECTION 19 - DETERMINISM AND INPUT FORMS



def section_19_determinism_and_input_forms() -> None:
    """
    Check that two annealing runs with the same seed compare strictly equal,
    that a scalar Python float is accepted as initial_params and resolved to a
    one-variable problem, and that a numpy array passed inside method_params is
    normalised to a list on the result.
    """

    section_banner("SECTION 19 - Determinism and input forms")

    rastrigin_2d = lambda x: float(
        10.0 * len(x)
        + np.sum(np.asarray(x) ** 2
                 - 10.0 * np.cos(2.0 * np.pi * np.asarray(x)))
    )


    # Seeded reproducibility of a stochastic method.
    test_block(
        test_name="19.annealing_reproducible_with_seed",
        description=(
            "verify a seeded stochastic run is reproducible, solving Rastrigin "
            "with annealing twice under the same seed and requiring the two "
            "results to compare strictly equal; without reproducibility a "
            "stochastic method cannot be debugged or regression-tested, so the "
            "seed has to pin the whole trajectory, not just its start."
        ),
    )

    annealing_keywords = {
        "function": rastrigin_2d,
        "method": "annealing",
        "bounds": ([-5.0, -5.0], [5.0, 5.0]),
        "method_params": {"seed": 7, "max_fun_evals": 1500},
    }

    first_seeded_run = minimizer(**annealing_keywords)
    second_seeded_run = minimizer(**annealing_keywords)

    check_truth(
        test_name="19.annealing_reproducible_with_seed",
        condition=(first_seeded_run == second_seeded_run),
        message_on_pass="the two seeded runs are identical.",
        message_on_fail=(
            "the seeded runs diverged; the seed is not pinning all randomness."
        ),
    )


    # A bare Python float as the starting point.
    test_block(
        test_name="19.scalar_float_initial_params",
        description=(
            "verify a bare Python float is accepted as initial_params, solving "
            "a one-dimensional quadratic from the float 0.0 and checking the "
            "run resolves to a single-variable problem and finds the "
            "minimizer; the signature admits a float, so the atleast-1d "
            "promotion has to work without a wrapping list or array."
        ),
    )

    scalar_started_result = minimizer(
        function=lambda x: float((x[0] - 3.0) ** 2),
        method="bfgs", initial_params=0.0,
    )

    scalar_start_as_expected = (
        scalar_started_result.size == 1
        and scalar_started_result.is_scalar is True
        and abs(float(scalar_started_result.final_params[0]) - 3.0) < 1e-5
    )

    print(f"  RESULT      : final = "
          f"{np.asarray(scalar_started_result.final_params)}, "
          f"size = {scalar_started_result.size}, "
          f"is_scalar = {scalar_started_result.is_scalar}")

    check_truth(
        test_name="19.scalar_float_initial_params",
        condition=scalar_start_as_expected,
        message_on_pass=(
            "the float start was promoted to a 1-D problem and solved."
        ),
        message_on_fail="a bare-float starting point was mishandled.",
    )


    # Normalisation of a numpy array buried in a settings dict.
    test_block(
        test_name="19.settings_array_normalised",
        description=(
            "verify a numpy array inside method_params is normalised to a "
            "plain list on the result, passing an array x0 to annealing and "
            "reading method_params back through to_dict; the result has to be "
            "representable and comparable, which it would not be with a live "
            "array buried in its recorded configuration."
        ),
    )

    normalised_result = minimizer(
        function=rastrigin_2d, method="annealing",
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        method_params={"x0": np.array([0.5, 0.5]), "seed": 3,
                       "max_fun_evals": 800},
    )

    stored_x0 = normalised_result.to_dict()["method_params"].get("x0")

    normalisation_as_expected = (
        isinstance(stored_x0, list) and stored_x0 == [0.5, 0.5]
    )

    print(f"  RESULT      : method_params['x0'] = {stored_x0!r} "
          f"({type(stored_x0).__name__})")

    check_truth(
        test_name="19.settings_array_normalised",
        condition=normalisation_as_expected,
        message_on_pass="the array setting was normalised to a list.",
        message_on_fail=(
            f"x0 was stored as {type(stored_x0).__name__}, not a list."
        ),
    )



# SECTION 20 - METHOD-SPEED BENCHMARK
#
# Pure performance information.



def section_20_method_speed_benchmark() -> None:
    """
    Time several methods on 2-D Rosenbrock with the analytical gradient and
    report per-run latency together with the function and gradient counts.
    """

    section_banner("SECTION 20 - Method-speed benchmark")

    start = np.array([-1.2, 1.0])
    benchmarked_methods = [
        "conjugate_gradient", "bfgs", "lbfgs", "trust_ncg", "trust_lanczos",
    ]

    test_block(
        test_name="20.method_speed",
        description=(
            "measure the elapsed time and evaluation counts of each strong "
            "method on Rosenbrock, solving once per method with the analytical "
            "gradient and reporting the figures side by side; this is the "
            "table a user consults when picking a method, so it is recorded as "
            "information rather than gated."
        ),
    )

    benchmark_table = []
    for method in benchmarked_methods:
        loop_t0 = time.perf_counter()
        result = minimizer(
            function=rosenbrock, method=method, initial_params=start,
            gradient_function=rosenbrock_gradient,
        )
        elapsed = time.perf_counter() - loop_t0
        benchmark_table.append((
            method, elapsed, result.func_evals_number,
            result.grad_evals_number, result.iteration_number,
        ))

    print("  RESULT      :")

    for method, elapsed, func_evals, grad_evals, iters in benchmark_table:
        print(f"                {method:<22} = {elapsed * 1e3:8.2f} ms, "
              f"f-evals = {func_evals:6d}, g-evals = {grad_evals:6d}, "
              f"iters = {iters:5d}")

    REPORT.record_info("benchmark recorded; no hard threshold imposed.")



# SECTION 21 - GRADIENT-COST BENCHMARK
#
# A numerical gradient costs extra function evaluations the analytical one does
# not. This block makes that cost visible by solving the same problem both ways
# and reporting the counts.



def section_21_gradient_cost_benchmark() -> None:
    """
    Solve Rosenbrock with bfgs using the analytical gradient and again using
    the finite-difference fallback, and report the function and gradient counts
    for each so the overhead of differencing is explicit.
    """

    section_banner("SECTION 21 - Gradient-cost benchmark")

    start = np.array([-1.2, 1.0])

    analytical_result = minimizer(
        function=rosenbrock, method="bfgs", initial_params=start,
        gradient_function=rosenbrock_gradient,
    )

    numerical_result = minimizer(
        function=rosenbrock, method="bfgs", initial_params=start,
    )


    test_block(
        test_name="21.gradient_cost",
        description=(
            "measure the extra function evaluations a numerical gradient "
            "costs, solving Rosenbrock once with the analytical gradient and "
            "once with the finite-difference fallback and reporting both "
            "counters; seeing the overhead is what justifies supplying a "
            "gradient when one is available."
        ),
    )

    print("  RESULT      :")
    print(f"                analytical : f-evals = "
          f"{analytical_result.func_evals_number:6d}, g-evals = "
          f"{analytical_result.grad_evals_number:6d}")
    print(f"                numerical  : f-evals = "
          f"{numerical_result.func_evals_number:6d}, g-evals = "
          f"{numerical_result.grad_evals_number:6d}")

    REPORT.record_info(
        "benchmark recorded; the numerical path trades function calls for not "
        "needing a derivative."
    )



# SECTION 22 - HVP VS FULL HESSIAN BENCHMARK
#
# A trust-region method can take curvature from a full Hessian or from a
# Hessian-vector product. The HVP avoids forming an n-by-n matrix; this block
# reports the time and evaluation counts of the two paths on the SPD quadratic.



def section_22_hvp_vs_hessian_benchmark() -> None:
    """
    Solve the SPD quadratic with trust_ncg twice, once given the full Hessian
    and once given the Hessian-vector product, and report the wall-clock time
    and evaluation counts of each path.
    """

    section_banner("SECTION 22 - HVP vs full-Hessian benchmark")

    start = np.array([3.0, -2.0, 1.5, -0.5])

    hessian_t0 = time.perf_counter()
    hessian_result = minimizer(
        function=spd_quadratic, method="trust_ncg", initial_params=start,
        gradient_function=spd_quadratic_gradient,
        hessian_function=spd_quadratic_hessian,
    )
    hessian_elapsed = time.perf_counter() - hessian_t0

    hvp_t0 = time.perf_counter()
    hvp_result = minimizer(
        function=spd_quadratic, method="trust_ncg", initial_params=start,
        gradient_function=spd_quadratic_gradient,
        hessian_vector_function=spd_quadratic_hvp,
    )
    hvp_elapsed = time.perf_counter() - hvp_t0

    test_block(
        test_name="22.hvp_vs_full_hessian",
        description=(
            "measure the cost of the full-Hessian path against the "
            "Hessian-vector path, solving the SPD quadratic with trust_ncg "
            "both ways and reporting time and counts; the HVP is the route "
            "large problems should take, and the figures here show the trade "
            "on a size where both are cheap."
        ),
    )

    print("  RESULT      :")
    print(f"                full Hessian : {hessian_elapsed * 1e3:8.2f} ms, "
          f"f-evals = {hessian_result.func_evals_number:6d}, g-evals = "
          f"{hessian_result.grad_evals_number:6d}")
    print(f"                HVP          : {hvp_elapsed * 1e3:8.2f} ms, "
          f"f-evals = {hvp_result.func_evals_number:6d}, g-evals = "
          f"{hvp_result.grad_evals_number:6d}")

    REPORT.record_info(
        "benchmark recorded; the HVP is the recommended path in "
        "moderate-to-high dimension."
    )



# END OF THE TESTS



# MAIN ENTRY POINT
#
# `run()` is the single point shared by every way of launching the suite: the
# `ripples.test()` dispatcher, the `python -m ripples.optimization._test`
# command line, and the pytest bridge below all come through here, so the
# output and the verdicts are identical no matter how the suite is started.



def run(
    include_benchmarks: bool = True, verbose: bool = True,
    show_summary: bool = True, quiet_label: Optional[str] = None,
) -> Reporter:
    """
    Execute the optimization test and benchmark suite end to end.

    Parameters
    ----------
    include_benchmarks : bool, default True
        When True the three timing-only sections (the method-speed table, the
        gradient-cost comparison, and the HVP-versus-Hessian comparison) run as
        well. They never move the failure count - they only report INFO - but
        they are extra work, so a caller that just wants a correctness gate
        (continuous integration, a quick local check) can pass False to skip
        them.

    Returns
    -------
    Reporter
        The reporter holding the run's tallies (`passed`, `failed`,
        `skipped`, `info`) and its failure records. It is returned rather than
        a bare count so a caller aggregating several submodules - as
        `ripples.test()` does when asked to test everything - can fold these
        numbers into one combined summary.
    """

    # Fresh on every call: the module-level reporter is reused, so
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
            section_20_method_speed_benchmark,
            section_21_gradient_cost_benchmark,
            section_22_hvp_vs_hessian_benchmark,
        }
        ordered_sections = [
            section_01_local_method_correctness,
            section_02_classic_test_functions,
            section_03_analytical_vs_numerical_gradient,
            section_04_second_order_information,
            section_05_starting_point_independence,
            section_06_global_direct,
            section_07_global_annealing,
            section_08_box_bounds,
            section_09_equality_constraint,
            section_10_inequality_constraint,
            section_11_result_api_surface,
            section_12_parameter_validation,
            section_13_warning_emission,
            section_14_callback_and_trajectory,
            section_15_custom_settings,
            section_16_high_dimensional,
            section_17_failure_and_degenerate_results,
            section_18_combined_constraints,
            section_19_determinism_and_input_forms,
            section_20_method_speed_benchmark,
            section_21_gradient_cost_benchmark,
            section_22_hvp_vs_hessian_benchmark,
        ]
        sections_to_run = [
            section for section in ordered_sections
            if include_benchmarks or section not in benchmark_sections
        ]

        print("=" * 78)
        print(" ripples.optimization - test and benchmark "
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
    Command-line entry point for `python -m ripples.optimization._test`.

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
# every other entry point (benchmarks off, since pytest is a correctness gate).



def test_optimization_suite() -> None:
    """Fail under pytest if any correctness check in the suite fails."""
    reporter = run(include_benchmarks=False)

    assert reporter.failed == 0, (
        f"{reporter.failed} optimization check(s) failed; "
        f"see the printed report of each."
    )
