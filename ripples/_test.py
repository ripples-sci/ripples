"""
Installation-integrity tests for `ripples`, and the unified test dispatcher.

This script does not re-test the numerical machinery - the differentiation and
optimization suites own that. Its job is the layer below: to prove that the
package the user actually installed is whole and wired correctly. A wheel can
import cleanly and still be broken - a re-export pointing at the wrong object, a
metadata field dropped during packaging, a NumPy too old for the code that
relies on it, a submodule whose test script never made it into the
distribution. Every section here checks one such property, so that a green run
means "what landed on this machine is the library, intact" before any feature
test is trusted.

It exercises, in reading order:

- Package import and the public metadata (`__version__`, author, licence).
- The runtime environment against the documented floors (Python, NumPy).
- The public API surface, the `__all__` declaration, and the curated `__dir__`.
- Submodule reachability and the identity of every top-level re-export.
- The bundled per-submodule test scripts and the unified `test()` entry point.
- A minimal end-to-end smoke call of each public function on the installed
  package - loose tolerances, since the point is "it runs and lands in the
  right place", not "it is accurate to the last digit".


The dispatcher
--------------
`test()` is the single front door named in the README and in both submodule
suites. It routes to the installation checks here, to one submodule suite, or
to everything at once, folding the per-suite tallies into one combined summary
in the last case.


Output convention
-----------------
For each test the script prints, on its own block:

- a header line of the form:
    "Section <n>, Test <m>: s<n>.<name> - <description>", where the description
    states in one sentence what the test verifies and why it matters, with the
    checking procedure folded into the same sentence.
- RESULT     : the measured value(s).
- VERDICT    : one of PASS / FAIL / INFO / SKIP, followed by an
               interpretation of what the result means in practice.


How to run
----------
Through the library's unified entry point::

    >>> import ripples
    >>> ripples.test("installation")   # just these checks
    >>> ripples.test("differentiation")
    >>> ripples.test("optimization")
    >>> ripples.test()                 # "all" - every suite, one summary

from the command line::

    python -m ripples._test
    # append --all to chain the submodule suites after the installation checks

or programmatically, when a caller wants the tallies back::

    >>> from ripples._test import run
    >>> reporter = run()
    >>> reporter.failed
    0

Under continuous integration the suite is reached through pytest, which
collects the bridge test at the foot of this file. However it is launched,
every path funnels through the same `run()`, so the output and the verdicts are
identical no matter how the suite is started.

The exit code of the command-line form is 0 when every test passes and the
number of failures otherwise. The summary block at the end lists every
failure together with its test name, the expected value, and the actual value.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np

import re
import sys
import importlib
from typing import Callable, Dict, List, Tuple, Optional, Any, Literal



FLOAT_EPSILON = float(np.finfo(float).eps)

SEPARATOR_WIDTH = 79

_CURRENT_SECTION_TEST_NUMBER = 0


# Kept here as the one place the environment checks read so a bump in the
# requirements is a one-line edit rather than a hunt through the assertions.
MINIMUM_PYTHON_VERSION = (3, 10)
MINIMUM_NUMPY_VERSION = (1, 24)


# The public surface. Every name here must be reachable from `import ripples`,
# and nothing private should leak alongside it. It mirrors the `__all__`
# assembled in ripples/__init__.py, on purpose - the test exists precisely to
# catch the day the two drift apart.
EXPECTED_SUBMODULES = (
    "differentiation",
    "optimization",
)

EXPECTED_PUBLIC_FUNCTIONS = (
    "nth_numerical_derivative",
    "numerical_hessian_vector_product",
    "DifferentiationResult",
    "minimizer",
    "OptimizationResult",
)

# The unified test entry point, re-exported at the top level from this module.
# It is part of the public surface - the README documents `ripples.test(...)` -
# so the import checks hold the package to its presence the same as any other
# advertised name.
EXPECTED_DISPATCHER = (
    "test",
)

EXPECTED_METADATA = (
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
)



# REPORTING UTILITIES
#
# Every test routes through `check_*` helpers that record a pass / fail entry
# on the module-level `REPORT` object. The helpers print the verdict line in
# place, so that the caller can read the output top-to-bottom as the suite
# progresses, and the summary at the end of `run()` re-prints every failure.



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
        because an optional part of the environment is absent).
    info : int
        Number of pure-information blocks.
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

    def reset(self) -> None:
        """
        Clear every counter and the recorded failure list.

        Called at the top of `run()` so the same module-level reporter
        can be reused across repeated invocations - for instance when
        `ripples.test()` drives this suite on its own and then again as
        part of an "all" run - without the second run inheriting the
        tallies of the first.
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
    print()
    print(f"# {section_title} ")
    print("#" * SEPARATOR_WIDTH)



def test_block(test_name: str, description: str) -> None:
    """
    Print the one-line header that opens a single test.

    The header reads::

        Section <n>, Test <m>: s<n>.<rest> - <description>

    where <n> is the section number (the section letter prefixing
    `test_name`, mapped A -> 1 ... F -> 6), <m> is the position of this
    test within the current section (reset by `section_banner`), and <rest>
    is whatever follows the section letter. The VERDICT line is appended
    afterwards by whichever `check_*` helper the test calls.

    Parameters
    ----------
    test_name : str
        The dotted identifier, beginning with its section letter, e.g.
        "C.surface.public_functions_present". The same string is handed to
        the `check_*` helpers, so a failure in the summary traces back here.
    description : str
        A single sentence stating what the test verifies and why it
        matters, with the checking procedure folded in.
    """
    global _CURRENT_SECTION_TEST_NUMBER

    section_letter, _, name_remainder = test_name.partition(".")
    section_number = ord(section_letter.upper()) - ord("A") + 1
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

    # Element-wise absolute difference normalised by max(|expected|, eps).
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
    """Plain boolean assertion. Used for presence / identity / flag checks."""

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



# ENVIRONMENT PROBES
#
# Two tiny parsers shared by the environment section. Both read the leading
# numeric release segment of a version - "1.24.3rc1" -> (1, 24, 3) - and
# ignore any pre-release or local suffix, since the requirement floors are
# expressed only in terms of the release numbers.



def _release_tuple(version_string: str) -> Tuple[int, ...]:
    """Parse the leading numeric release segment of a version into ints."""
    leading_match = re.match(r"\d+(?:\.\d+)*", version_string.strip())
    if leading_match is None:
        return ()
    return tuple(int(part) for part in leading_match.group(0).split("."))


def _format_version(version_tuple: Tuple[int, ...]) -> str:
    """Render a release tuple back into a dotted string for the RESULT line."""
    return ".".join(str(component) for component in version_tuple)



# SECTION A - PACKAGE IMPORT AND METADATA
#
# The first thing any install has to do is import. The second is identify
# itself: the version and the authorship/licence fields are what PyPI, pip, and
# any downstream tooling read, and a packaging slip that drops one of them is
# invisible until something asks for it. This section imports the package and
# confirms every metadata field is present, a non-empty string, and - for the
# version - shaped like a release number.



def section_a_import_and_metadata() -> None:
    """
    Import `ripples`, then confirm `__version__` parses as a dotted release
    number and that the author, e-mail, licence, and copyright fields are all
    present non-empty strings.
    """

    section_banner("SECTION A - Package import and metadata")

    # the import itself
    test_block(
        test_name="A.import.package",
        description=(
            "verify `import ripples` succeeds, importing the top-level package "
            "and checking the module object comes back; nothing else in the "
            "suite can run if the package will not load, so this is the gate "
            "every later test depends on."
        ),
    )
    import ripples
    check_truth(
        test_name="A.import.package",
        condition=ripples is not None,
        message_on_pass="the top-level package imported.",
        message_on_fail="the import produced no module object.",
    )

    # version is a release-shaped string
    test_block(
        test_name="A.metadata.version_is_release_shaped",
        description=(
            "verify `__version__` is a non-empty string whose leading segment "
            "parses as a dotted release number, reading the attribute and "
            "matching it against the release pattern; pip and PyPI reject or "
            "mis-sort a version that is not PEP 440 shaped, so a malformed one "
            "is a release blocker."
        ),
    )
    version_string = getattr(ripples, "__version__", None)
    version_is_release_shaped = (
        isinstance(version_string, str)
        and re.match(r"^\d+\.\d+", version_string.strip()) is not None
    )
    check_truth(
        test_name="A.metadata.version_is_release_shaped",
        condition=version_is_release_shaped,
        message_on_pass=f"__version__ is {version_string!r}.",
        message_on_fail=(
            f"__version__ is missing or not release-shaped: "
            f"{version_string!r}."
        ),
    )

    # the remaining metadata fields
    # Derived from EXPECTED_METADATA minus __version__ (handled separately
    # above with its own release-shape check), so adding a new metadata field
    # to EXPECTED_METADATA is enough; the loop below picks it up on its own.
    descriptive_metadata = tuple(
        metadata_name for metadata_name in EXPECTED_METADATA
        if metadata_name != "__version__"
    )
    for metadata_name in descriptive_metadata:
        test_block(
            test_name=f"A.metadata.{metadata_name.strip('_')}",
            description=(
                f"verify `{metadata_name}` is present as a non-empty string, "
                f"reading the attribute off the package and checking its type "
                f"and length; this field is surfaced on PyPI and in tooling, "
                f"and an empty or missing one is a packaging defect that "
                f"import alone would never reveal."
            ),
        )
        metadata_value = getattr(ripples, metadata_name, None)
        check_truth(
            test_name=f"A.metadata.{metadata_name.strip('_')}",
            condition=(
                isinstance(metadata_value, str)
                and len(metadata_value) > 0
            ),
            message_on_pass=f"{metadata_name} = {metadata_value!r}",
            message_on_fail=f"{metadata_name} is missing or empty.",
        )



# SECTION B - RUNTIME ENVIRONMENT
#
# The code is written against documented floors - Python 3.10 for the syntax
# it uses, NumPy 1.24 for the array API it relies on. pip enforces these at
# install time from the metadata, but an editable install, a hand-built
# environment, or a stale wheel can sidestep that. Checking the live
# interpreter and the imported NumPy here turns a confusing downstream failure
# ("some attribute does not exist") into a clear up-front verdict.



def section_b_runtime_environment() -> None:
    """
    Compare the running Python and the imported NumPy against the documented
    minimums, reporting both the found and the required version on each block.
    """

    section_banner("SECTION B - Runtime environment")

    # Python
    test_block(
        test_name="B.environment.python_version",
        description=(
            "verify the running interpreter meets the documented Python floor, "
            "comparing sys.version_info against the required minimum; the code "
            "uses syntax introduced in that release, so an older interpreter "
            "fails in ways that have nothing to do with the library's logic."
        ),
    )
    running_python_version = sys.version_info[: len(MINIMUM_PYTHON_VERSION)]
    print(f"  RESULT      : found Python "
          f"{_format_version(tuple(running_python_version))}, "
          f"require >= {_format_version(MINIMUM_PYTHON_VERSION)}")
    check_truth(
        test_name="B.environment.python_version",
        condition=tuple(running_python_version) >= MINIMUM_PYTHON_VERSION,
        message_on_pass="interpreter satisfies the documented floor.",
        message_on_fail="interpreter is older than the documented floor.",
    )

    # NumPy
    test_block(
        test_name="B.environment.numpy_version",
        description=(
            "verify the imported NumPy meets the documented floor, parsing "
            "numpy.__version__ and comparing against the required minimum; the "
            "whole library is built on NumPy, so a version below the floor is "
            "the most likely source of a subtle array-API mismatch."
        ),
    )
    installed_numpy_version = _release_tuple(np.__version__)
    comparable_numpy_version = installed_numpy_version[
        : len(MINIMUM_NUMPY_VERSION)
    ]
    print(f"  RESULT      : found NumPy {np.__version__}, "
          f"require >= {_format_version(MINIMUM_NUMPY_VERSION)}")
    check_truth(
        test_name="B.environment.numpy_version",
        condition=comparable_numpy_version >= MINIMUM_NUMPY_VERSION,
        message_on_pass="NumPy satisfies the documented floor.",
        message_on_fail="NumPy is older than the documented floor.",
    )



# SECTION C - PUBLIC API SURFACE
#
# The promise of the package is that `import ripples` is enough to reach every
# feature. That promise is held together by three things: the names being
# present, the `__all__` declaration agreeing with what is actually exported,
# and the curated `__dir__` showing the user the public surface and nothing
# else. A break in any of them is silent - the code still works, but the
# advertised interface no longer matches reality.



def section_c_public_api_surface() -> None:
    """
    Confirm every documented submodule, public function, and metadata name is
    reachable from the package; that `__all__` is the union of the three and
    contains no dangling name; and that `dir(ripples)` returns exactly the
    sorted `__all__`.
    """

    section_banner("SECTION C - Public API surface")

    import ripples

    # every advertised name is present
    expected_public_names = (
        EXPECTED_SUBMODULES + EXPECTED_PUBLIC_FUNCTIONS
        + EXPECTED_DISPATCHER + EXPECTED_METADATA
    )
    for public_name in expected_public_names:
        test_block(
            test_name=f"C.present.{public_name.strip('_')}",
            description=(
                f"verify `ripples.{public_name}` resolves, looking the name up "
                f"on the package; this is one entry of the advertised surface, "
                f"and a missing one means the documented one-import workflow "
                f"is broken for that feature."
            ),
        )
        check_truth(
            test_name=f"C.present.{public_name.strip('_')}",
            condition=hasattr(ripples, public_name),
            message_on_pass=f"ripples.{public_name} is reachable.",
            message_on_fail=f"ripples.{public_name} is absent.",
        )

    # the documented callables / classes are actually callable
    for callable_name in EXPECTED_PUBLIC_FUNCTIONS + EXPECTED_DISPATCHER:
        test_block(
            test_name=f"C.callable.{callable_name}",
            description=(
                f"verify `ripples.{callable_name}` is callable, fetching the "
                f"object and checking callable(); a name that resolves to a "
                f"non-callable (a stray constant, a half-finished re-export) "
                f"would pass the presence check yet fail the moment a user "
                f"tried to use it."
            ),
        )
        candidate = getattr(ripples, callable_name, None)
        check_truth(
            test_name=f"C.callable.{callable_name}",
            condition=callable(candidate),
            message_on_pass=f"{callable_name} is callable.",
            message_on_fail=f"{callable_name} resolved to a non-callable.",
        )

    # __all__ is well-formed and complete
    test_block(
        test_name="C.dunder_all.union",
        description=(
            "verify `__all__` is a list holding exactly the union of the "
            "documented submodules, public functions, and metadata, comparing "
            "the two as sets; `__all__` is what `from ripples import *` "
            "honours, so a name absent here is invisible to that idiom even "
            "when it exists on the package."
        ),
    )
    package_all = getattr(ripples, "__all__", None)
    all_matches_expected = (
        isinstance(package_all, list)
        and set(package_all) == set(expected_public_names)
    )
    print(f"  RESULT      : __all__ has {len(package_all or [])} entries; "
          f"expected {len(expected_public_names)}")
    check_truth(
        test_name="C.dunder_all.union",
        condition=all_matches_expected,
        message_on_pass="__all__ is exactly the documented public surface.",
        message_on_fail="__all__ diverges from the documented public surface.",
    )

    # no name in __all__ dangles
    test_block(
        test_name="C.dunder_all.no_dangling_name",
        description=(
            "verify every name listed in `__all__` actually resolves on the "
            "package, looking each one up; a name advertised but not bound "
            "raises ImportError under `from ripples import *`, the kind of "
            "fault that survives a clean import and only bites the star-import "
            "user."
        ),
    )
    dangling_names = [
        name for name in (package_all or []) if not hasattr(ripples, name)
    ]
    check_truth(
        test_name="C.dunder_all.no_dangling_name",
        condition=len(dangling_names) == 0,
        message_on_pass="every __all__ entry is bound on the package.",
        message_on_fail=f"__all__ lists unbound names: {dangling_names}.",
    )

    # __dir__ is the sorted __all__
    test_block(
        test_name="C.dunder_dir.equals_sorted_all",
        description=(
            "verify `dir(ripples)` equals the sorted `__all__`, calling dir() "
            "and comparing; the package defines a custom __dir__ so that tab-"
            "completion shows the public surface and nothing private, and a "
            "regression there quietly re-clutters the listing the user sees."
        ),
    )
    directory_listing = dir(ripples)
    check_truth(
        test_name="C.dunder_dir.equals_sorted_all",
        condition=directory_listing == sorted(package_all or []),
        message_on_pass="dir(ripples) is the curated, sorted public surface.",
        message_on_fail="dir(ripples) no longer matches sorted(__all__).",
    )



# SECTION D - SUBMODULE REACHABILITY AND RE-EXPORT IDENTITY
#
# Each public function is defined deep inside a submodule and re-exported at
# the top level. The re-export is plumbing, and plumbing leaks: a refactor can
# leave `ripples.minimizer` pointing at a stale object while
# `ripples.optimization.minimizer` points at the live one, and both still
# import. The identity check (`is`) is the only thing that catches that - two
# different objects with the same name behave identically right up until they
# don't.



def section_d_submodule_reexport_identity() -> None:
    """
    Import each submodule directly, confirm it carries its own `__all__`, and
    confirm every top-level public function is the very same object as the
    attribute it was re-exported from.
    """

    section_banner("SECTION D - Submodule reachability and re-export identity")

    import ripples

    # each submodule imports on its own-
    # The submodule objects are stashed in a dict here so the identity loop
    # below can iterate them without re-importing - one import per submodule
    # is enough.
    imported_submodule_objects: Dict[str, Any] = {}
    for submodule_name in EXPECTED_SUBMODULES:
        test_block(
            test_name=f"D.import.{submodule_name}",
            description=(
                f"verify `ripples.{submodule_name}` imports as a submodule, "
                f"importing it directly and checking it exposes its own "
                f"`__all__`; the top-level re-exports lean on the submodule "
                f"loading cleanly, so a fault here explains a whole column of "
                f"later failures at once."
            ),
        )
        imported_submodule_objects[submodule_name] = importlib.import_module(
            f"ripples.{submodule_name}"
        )
        check_truth(
            test_name=f"D.import.{submodule_name}",
            condition=hasattr(
                imported_submodule_objects[submodule_name], "__all__"
            ),
            message_on_pass=(
                f"ripples.{submodule_name} imported and declares __all__."
            ),
            message_on_fail=(
                f"ripples.{submodule_name} imported without an __all__."
            ),
        )

    # top-level re-exports are the same objects
    #
    # The library's stated invariant is that every public entry point reaches
    # the top level, so what is in a submodule's `__all__` must be re-exported
    # at `ripples.<name>` and must be the same object. The loop discovers the
    # names by reading each submodule's `__all__` rather than hardcoding a
    # function-to-submodule map; a new public name added to a submodule and
    # re-exported needs no edit here, and a name added to the submodule but
    # forgotten at the top level fails this check rather than slipping through.
    for submodule_name, submodule_object in imported_submodule_objects.items():
        submodule_public_names = getattr(submodule_object, "__all__", None)
        if not submodule_public_names:
            # Already reported as a failure by the import block above; nothing
            # useful to iterate on, so skip the identity checks for this one.
            continue

        for public_name in submodule_public_names:
            test_block(
                test_name=f"D.identity.{public_name}",
                description=(
                    f"verify `ripples.{public_name}` is the very object "
                    f"exported by ripples.{submodule_name}, comparing the two "
                    f"with `is`; a presence check passes even when the top "
                    f"level points at a stale copy, so identity is what "
                    f"actually proves the re-export wiring is live."
                ),
            )
            top_level_object = getattr(ripples, public_name, None)
            origin_object = getattr(submodule_object, public_name, None)
            check_truth(
                test_name=f"D.identity.{public_name}",
                condition=top_level_object is origin_object
                and top_level_object is not None,
                message_on_pass=(
                    f"ripples.{public_name} is ripples.{submodule_name}."
                    f"{public_name}."
                ),
                message_on_fail=(
                    f"ripples.{public_name} is a different object from "
                    f"ripples.{submodule_name}.{public_name} (or one of "
                    f"them is missing)."
                ),
            )

    # a phantom submodule does not resolve
    test_block(
        test_name="D.no_phantom_submodule",
        description=(
            "verify importing a submodule that does not exist raises "
            "ModuleNotFoundError, attempting `ripples.does_not_exist` and "
            "catching; this guards against an over-eager package hook that "
            "would answer for names the library never defined and so mask a "
            "caller's typo."
        ),
    )
    check_raises(
        test_name="D.no_phantom_submodule",
        callable_object=lambda: importlib.import_module(
            "ripples.does_not_exist"
        ),
        expected_exception_type=ModuleNotFoundError,
    )



# SECTION E - BUNDLED TEST SUITES AND THE UNIFIED ENTRY POINT
#
# The per-submodule test scripts are shipped inside the wheel (the MANIFEST and
# the setuptools config pull every .py under the package in), and the unified
# `test()` dispatcher in this module is the front door the README points the
# user at. This section proves both are reachable and correctly shaped, without
# running them - executing the full submodule suites is what `test("all")`
# does, and duplicating that here would only slow the installation gate down.



def section_e_test_suite_wiring() -> None:
    """
    Confirm each submodule ships a `_test` module exposing the `run` callable
    the dispatcher drives, that this module's own `test` dispatcher is callable,
    rejects an unknown target with ValueError and a non-string target with
    TypeError, and is the very object exposed as `ripples.test`.
    """

    section_banner(
        "SECTION E - Bundled test suites and the unified entry point"
    )

    import ripples

    # each submodule's _test ships and exposes run()
    for submodule_name in EXPECTED_SUBMODULES:
        test_block(
            test_name=f"E.suite.{submodule_name}_run_callable",
            description=(
                f"verify ripples.{submodule_name}._test ships and exposes a "
                f"callable `run`, importing the test module and checking the "
                f"attribute; the unified dispatcher drives each suite through "
                f"this exact entry point, so a suite missing from the wheel "
                f"breaks `test(\"all\")` even though the library itself works."
            ),
        )
        submodule_test = importlib.import_module(
            f"ripples.{submodule_name}._test"
        )
        check_truth(
            test_name=f"E.suite.{submodule_name}_run_callable",
            condition=callable(getattr(submodule_test, "run", None)),
            message_on_pass=(
                f"ripples.{submodule_name}._test.run is present and callable."
            ),
            message_on_fail=(
                f"ripples.{submodule_name}._test.run is missing or "
                f"not callable."
            ),
        )

    # the dispatcher in this module is callable
    test_block(
        test_name="E.dispatcher.callable",
        description=(
            "verify the `test` dispatcher defined in this module is callable, "
            "checking the local object; it is the single front door every "
            "other launch path funnels through, so it has to be a working "
            "callable before any of the documented `ripples.test(...)` forms "
            "can mean anything."
        ),
    )
    check_truth(
        test_name="E.dispatcher.callable",
        condition=callable(test),
        message_on_pass="the unified test() dispatcher is callable.",
        message_on_fail="the unified test() dispatcher is not callable.",
    )

    # the dispatcher rejects an unknown target
    test_block(
        test_name="E.dispatcher.rejects_unknown_target",
        description=(
            "verify `test(\"non-existant\")` raises ValueError, calling the "
            "dispatcher with a target it does not recognise and catching; a "
            "silent no-op on a mistyped target would let a caller believe a "
            "suite ran when nothing did, the worst failure mode a test runner "
            "can have."
        ),
    )
    check_raises(
        test_name="E.dispatcher.rejects_unknown_target",
        callable_object=lambda: test("non-existant"),
        expected_exception_type=ValueError,
    )

    # the dispatcher rejects a non-string target
    test_block(
        test_name="E.dispatcher.rejects_non_string_target",
        description=(
            "verify `test(None)` raises TypeError, calling the dispatcher with "
            "a non-string target and catching; the library raises TypeError "
            "for type errors elsewhere, and a clean TypeError here points at "
            "the real fault rather than the AttributeError that would surface "
            "if the dispatcher tried to `.strip()` a None."
        ),
    )
    check_raises(
        test_name="E.dispatcher.rejects_non_string_target",
        callable_object=lambda: test(None),  # type: ignore[arg-type]
        expected_exception_type=TypeError,
    )

    # test() is exposed at the top level and is this dispatcher
    test_block(
        test_name="E.dispatcher.exposed_at_top_level",
        description=(
            "verify `ripples.test` is exposed and is this module's dispatcher, "
            "comparing ripples.test against the local `test` by identity; the "
            "README documents `ripples.test(...)` as the entry point, so the "
            "re-export has to be present and has to resolve to the real "
            "dispatcher rather than to some object shadowing the name."
        ),
    )
    check_truth(
        test_name="E.dispatcher.exposed_at_top_level",
        condition=getattr(ripples, "test", None) is test,
        message_on_pass="ripples.test is wired and is the unified dispatcher.",
        message_on_fail=(
            "ripples.test is missing or is not this dispatcher; expose it with "
            "`from ._test import test` in __init__.py and list 'test' in the "
            "package's public surface."
        ),
    )



# SECTION F - END-TO-END SMOKE TEST
#
# Import succeeding and names resolving still does not prove the installed code
# runs. A broken NumPy build, a corrupted file, an ABI mismatch - these surface
# only when arithmetic actually happens. So each public entry point is called
# once on a trivial problem with a known answer, judged on a loose band. This
# is deliberately not an accuracy test - the submodule suites own that, with
# tolerances down to machine epsilon. Here the only question is whether the
# installed package computes and lands in the right place at all.



def _smoke_sine_of_first_coordinate(point: Any) -> float:
    """
    1-D sine, taking the point as a one-element array - the convention used
    throughout the differentiation suite - so the smoke call exercises the
    same input shape the rest of the tests do.
    """
    coordinates = np.asarray(point, dtype=float)
    return float(np.sin(coordinates[0]))


def _smoke_objective(point: Any) -> float:
    """A convex bowl with its minimum at (3, -1), for the optimizer smoke."""
    coordinates = np.asarray(point, dtype=float)
    return float((coordinates[0] - 3.0) ** 2 + (coordinates[1] + 1.0) ** 2)


def _smoke_objective_gradient(point: Any) -> np.ndarray:
    """Analytical gradient of `_smoke_objective`."""
    coordinates = np.asarray(point, dtype=float)
    return np.array([
        2.0 * (coordinates[0] - 3.0),
        2.0 * (coordinates[1] + 1.0),
    ])


def _smoke_identity_gradient(point: Any) -> np.ndarray:
    """
    The gradient of f(x) = 0.5 * x^T * x, which is simply x itself - the
    identity map.

    `numerical_hessian_vector_product` differentiates a gradient, not a scalar
    function, so the first argument it expects is grad_f, not f. This helper
    is that grad_f for the trivial bowl above; its Jacobian (the Hessian of f)
    is the identity, so for any direction v the product H @ v equals v.
    """
    return np.asarray(point, dtype=float)


def section_f_end_to_end_smoke() -> None:
    """
    Call each public entry point once on a trivial known-answer problem -
    the first derivative of sin at 0, the minimiser of a convex bowl, and a
    Hessian-vector product whose Hessian is the identity - and confirm each
    lands on the right value within a loose smoke band and returns the
    documented wrapper type.
    """

    section_banner("SECTION F - End-to-end smoke test")

    import ripples

    # differentiation: d/dx sin(x) at 0 is 1
    test_block(
        test_name="F.smoke.first_derivative",
        description=(
            "verify nth_numerical_derivative computes d/dx sin(x) at 0 as 1, "
            "passing the 1-D sine taking a one-element point array and "
            "evaluating the first-derivative callable; this is the smallest "
            "end-to-end run of the differentiation path, and a wrong or "
            "erroring answer means the installed code does not actually "
            "compute, however cleanly it imported."
        ),
    )
    first_derivative_callable = ripples.nth_numerical_derivative(
        _smoke_sine_of_first_coordinate, derivative_order=1,
    )
    first_derivative_result = first_derivative_callable(np.array([0.0]))
    check_allclose(
        test_name="F.smoke.first_derivative",
        actual_value=float(first_derivative_result.as_float()),
        expected_value=1.0,
        relative_tolerance=1e-4, absolute_tolerance=1e-6,
        interpretation="the differentiation path runs and is correct here.",
    )

    test_block(
        test_name="F.smoke.differentiation_result_type",
        description=(
            "verify the differentiation call returns a DifferentiationResult, "
            "checking the wrapper type of the value just produced; the "
            "documented return type is the contract downstream code is written "
            "against, so a bare ndarray slipping through would break callers "
            "that expect the wrapper's metadata."
        ),
    )
    check_truth(
        test_name="F.smoke.differentiation_result_type",
        condition=isinstance(
            first_derivative_result, ripples.DifferentiationResult
        ),
        message_on_pass="the result is a DifferentiationResult as documented.",
        message_on_fail="the result is not a DifferentiationResult.",
    )

    # optimization: minimiser of a convex bowl
    test_block(
        test_name="F.smoke.minimise_bowl",
        description=(
            "verify minimizer locates the minimum (3, -1) of a convex bowl "
            "from the origin with BFGS and the analytical gradient, then check "
            "the minimiser within a loose band; this is the smallest "
            "end-to-end run of the optimization path and confirms the "
            "installed solver converges rather than merely imports."
        ),
    )
    optimization_result = ripples.minimizer(
        function=_smoke_objective,
        method="bfgs",
        initial_params=[0.0, 0.0],
        gradient_function=_smoke_objective_gradient,
    )
    check_allclose(
        test_name="F.smoke.minimise_bowl",
        actual_value=np.asarray(optimization_result.final_params),
        expected_value=np.array([3.0, -1.0]),
        relative_tolerance=1e-3, absolute_tolerance=1e-3,
        interpretation="the optimization path runs and converges here.",
    )

    test_block(
        test_name="F.smoke.optimization_result_type",
        description=(
            "verify the minimiser call returns an OptimizationResult, checking "
            "the wrapper type of the value just produced; the documented "
            "return type carries the cost, counts, and termination reason "
            "callers rely on, so the contract has to hold on the installed "
            "package."
        ),
    )
    check_truth(
        test_name="F.smoke.optimization_result_type",
        condition=isinstance(
            optimization_result, ripples.OptimizationResult
        ),
        message_on_pass="the result is an OptimizationResult as documented.",
        message_on_fail="the result is not an OptimizationResult.",
    )

    # Hessian-vector product: the gradient is the identity, so H @ v = v
    test_block(
        test_name="F.smoke.hessian_vector_product",
        description=(
            "verify numerical_hessian_vector_product returns v unchanged when "
            "given the identity-map gradient (the gradient of "
            "f(x) = 0.5 * x^T * x, whose Hessian is therefore I), evaluating "
            "H @ v at a fixed point and direction; the entry point "
            "differentiates a gradient rather than a scalar function, and the "
            "identity Hessian gives an exact expected answer for a clean "
            "end-to-end check of the matrix-free path that touches no full "
            "Hessian."
        ),
    )
    probe_point = np.array([1.0, 2.0, 3.0])
    probe_direction = np.array([1.0, 0.0, -2.0])
    hessian_vector_result = ripples.numerical_hessian_vector_product(
        _smoke_identity_gradient,
        point=probe_point,
        vector=probe_direction,
    )
    check_allclose(
        test_name="F.smoke.hessian_vector_product",
        actual_value=np.asarray(hessian_vector_result),
        expected_value=probe_direction,
        relative_tolerance=1e-4, absolute_tolerance=1e-6,
        interpretation="the matrix-free HVP path runs and is correct here.",
    )



# MAIN ENTRY POINT
#
# `run()` is the single orchestration point for the installation suite, shared
# by every way of launching it: the `ripples.test("installation")` dispatcher,
# the `python -m ripples._test` command line, and the pytest bridge below all
# come through here, so the output and the verdicts are identical no matter how
# the suite is started.



def run(include_benchmarks: bool = True) -> Reporter:
    """
    Execute the installation-integrity suite end to end.

    Parameters
    ----------
    include_benchmarks : bool, default True
        Accepted for signature parity with the submodule suites so the unified
        dispatcher can drive every `run()` the same way. The installation suite
        has no timing-only sections, so the flag has no effect here.

    Returns
    -------
    Reporter
        The reporter holding the run's tallies (`passed`, `failed`,
        `skipped`, `info`) and its failure records. It is returned rather
        than a bare count so a caller aggregating several suites - as
        `test("all")` does - can fold these numbers into one combined summary.
    """

    # A fresh slate on every call: the module-level reporter is reused, so
    # without this a second run would keep adding to the first run's counts.
    REPORT.reset()

    ordered_sections = [
        section_a_import_and_metadata,
        section_b_runtime_environment,
        section_c_public_api_surface,
        section_d_submodule_reexport_identity,
        section_e_test_suite_wiring,
        section_f_end_to_end_smoke,
    ]

    print("=" * SEPARATOR_WIDTH)
    print(
        " ripples - installation-integrity suite ".center(SEPARATOR_WIDTH, "=")
    )
    print("=" * SEPARATOR_WIDTH)
    print(
        "Reading order: each block states what it checks and why it matters, "
        "then\nprints the RESULT and a VERDICT (pass / fail / info)."
    )

    for section_runner in ordered_sections:
        try:
            section_runner()
        except Exception as section_exception:
            # An exception escaping a section is itself a failure; record it,
            # print the traceback, and carry on with the next section so a
            # single fault cannot abort the whole suite.
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
            print(" UNCAUGHT EXCEPTION ".center(SEPARATOR_WIDTH, "!"))
            traceback.print_exc()
            print("!" * SEPARATOR_WIDTH)

    REPORT.print_summary()
    return REPORT



# UNIFIED DISPATCHER
#
# `test()` is the one front door named in the README and in both submodule
# suites. It routes to the installation checks above, to a single submodule
# suite, or to everything at once. In the "all" case it runs each suite in
# turn - each prints its own report - and then folds the per-suite tallies into
# one combined summary, so a caller sees both the detail and the grand total.
#
# It is re-exported at the top level from ripples/__init__.py (`from ._test
# import test`, with 'test' listed in the package's public surface), which is
# what makes `ripples.test(...)` resolve to the function defined here.



# The targets `test()` understands, each mapped to a short list of accepted
# spellings, so a caller can write the obvious word and have it resolve.
_TARGET_ALIASES = {
    "installation": ("installation", "install", "root", "integrity"),
    "differentiation": ("differentiation", "diff"),
    "optimization": ("optimization", "optimisation", "opt"),
    "all": ("all", "everything", "full"),
}


def _canonical_target(requested_target: str) -> str:
    """
    Map a requested target spelling to its canonical key.

    Raises
    ------
    TypeError
        If `requested_target` is not a string.
    ValueError
        If the string does not match any of the recognised target aliases.
    """
    if not isinstance(requested_target, str):
        raise TypeError(
            f"test target must be a string, got "
            f"{type(requested_target).__name__}: {requested_target!r}."
        )
    requested_lower = requested_target.strip().lower()
    for canonical_name, accepted_spellings in _TARGET_ALIASES.items():
        if requested_lower in accepted_spellings:
            return canonical_name
    raise ValueError(
        f"unknown test target {requested_target!r}; choose one of "
        f"'installation', 'differentiation', 'optimization', or 'all'."
    )


def _print_combined_summary(
    named_reporters: List[Tuple[str, Reporter]],
) -> None:
    """Print one grand-total table across several suites' reporters."""

    print()
    print("=" * SEPARATOR_WIDTH)
    print(" COMBINED SUMMARY (ALL SUITES) ".center(SEPARATOR_WIDTH, "="))
    print("=" * SEPARATOR_WIDTH)
    print(f"  {'suite':<20}{'passed':>9}{'failed':>9}"
          f"{'skipped':>9}{'info':>9}")
    print("  " + "-" * (SEPARATOR_WIDTH - 4))

    total_passed = total_failed = total_skipped = total_info = 0
    for suite_name, suite_reporter in named_reporters:
        total_passed += suite_reporter.passed
        total_failed += suite_reporter.failed
        total_skipped += suite_reporter.skipped
        total_info += suite_reporter.info
        print(f"  {suite_name:<20}{suite_reporter.passed:>9}"
              f"{suite_reporter.failed:>9}{suite_reporter.skipped:>9}"
              f"{suite_reporter.info:>9}")

    print("  " + "-" * (SEPARATOR_WIDTH - 4))
    print(f"  {'TOTAL':<20}{total_passed:>9}{total_failed:>9}"
          f"{total_skipped:>9}{total_info:>9}")
    print("=" * SEPARATOR_WIDTH)
    if total_failed == 0:
        print("  All suites passed.")
    else:
        print(f"  {total_failed} check(s) failed across all suites; "
              f"see each suite's own FAILURES block above.")
    print("=" * SEPARATOR_WIDTH)



def test(
    which: Literal[
        'installation', 'differentiation', 'optimization', 'all'
    ] = "all",
    include_benchmarks: bool = True
) -> int:
    """
    Run the Ripples test suites and return the number of failing checks.

    This is the unified entry point referenced throughout the documentation.
    It runs the installation-integrity checks defined in this module, a single
    submodule suite, or every suite at once, and returns the total failure
    count so the result reads as a truthiness gate: ``0`` means everything
    passed.

    Parameters
    ----------
    which : 'installation', 'differentiation', 'optimization' or 'all', \
    default 'all'
        Which suite to run. Accepted (case-insensitive) targets:

        - ``'installation'`` - the integrity checks in this module only.
        - ``'differentiation'`` - the differentiation suite only.
        - ``'optimization'`` - the optimization suite only.
        - ``'all'`` - the installation checks followed by both submodule
          suites, with a combined summary printed at the end.

        Common alternative spellings (``'install'``, ``'diff'``, ``'opt'``,
        ``'everything'``, ...) are accepted as well.
    include_benchmarks : bool, default True
        Forwarded unchanged to each submodule suite, where it toggles the
        timing-only sections. The installation suite has no benchmarks, so the
        flag does not affect that part of an ``'all'`` run.

    Returns
    -------
    int
        The number of failing checks across whatever was run; ``0`` on a fully
        passing run, which is also the convention continuous-integration
        systems read from the process exit code.

    Raises
    ------
    TypeError
        If `which` is not a string.
    ValueError
        If `which` is not one of the recognised targets.

    Examples
    --------
    >>> import ripples
    >>> ripples.test("installation")   # doctest: +SKIP
    >>> ripples.test()             # every suite, one summary  # doctest: +SKIP
    """

    canonical_target = _canonical_target(which)

    if canonical_target == "installation":
        return run(include_benchmarks=include_benchmarks).failed

    if canonical_target == "differentiation":
        differentiation_tests = importlib.import_module(
            "ripples.differentiation._test"
        )
        return differentiation_tests.run(
            include_benchmarks=include_benchmarks
        ).failed

    if canonical_target == "optimization":
        optimization_tests = importlib.import_module(
            "ripples.optimization._test"
        )
        return optimization_tests.run(
            include_benchmarks=include_benchmarks
        ).failed

    # canonical_target == "all": run each suite in turn, then combine.
    differentiation_tests = importlib.import_module(
        "ripples.differentiation._test"
    )
    optimization_tests = importlib.import_module(
        "ripples.optimization._test"
    )

    named_reporters: List[Tuple[str, Reporter]] = [
        ("installation", run(include_benchmarks=include_benchmarks)),
        (
            "differentiation",
            differentiation_tests.run(include_benchmarks=include_benchmarks),
        ),
        (
            "optimization",
            optimization_tests.run(include_benchmarks=include_benchmarks),
        ),
    ]

    _print_combined_summary(named_reporters)
    return sum(suite_reporter.failed for _, suite_reporter in named_reporters)



def main(argv: Optional[List[str]] = None) -> int:
    """
    Command-line entry point for `python -m ripples._test`.

    By default it runs the installation-integrity suite alone. With ``--all``
    it chains the two submodule suites after it and prints a combined summary;
    with ``--no-benchmarks`` it forwards `include_benchmarks=False` to whatever
    it runs. Returns the number of failures so the process exit code is 0 on a
    clean run and non-zero otherwise - the convention continuous-integration
    systems read.

    The CLI dispatches through `ripples.test` rather than the local `test`
    defined below, because `python -m ripples._test` loads this source file
    twice (once as `ripples._test` during the package import, once again as
    `__main__` for the CLI run) and the two copies hold separate function
    objects; going through the package attribute ensures the identity check
    in Section E resolves to the same `test` object regardless of how the
    suite was launched.
    """
    arguments = sys.argv[1:] if argv is None else argv
    include_benchmarks = "--no-benchmarks" not in arguments
    target = "all" if "--all" in arguments else "installation"
    import ripples
    return ripples.test(which=target, include_benchmarks=include_benchmarks)



# PYTEST BRIDGE
#
# A single function so that pytest - which continuous integration runs on every
# push - actually gates on this suite. It drives the same `run()` as every
# other entry point (benchmarks off, since pytest is a correctness gate, not a
# stopwatch) and turns a non-zero failure count into a failed test. The per-
# test report is printed by `run()`; pytest shows it with `-s` or on failure.



def test_installation_suite() -> None:
    """Fail under pytest if any installation-integrity check fails."""
    reporter = run(include_benchmarks=False)
    assert reporter.failed == 0, (
        f"{reporter.failed} installation check(s) failed; "
        f"see the printed report for the what / why of each."
    )



if __name__ == "__main__":
    sys.exit(main())
