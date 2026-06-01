"""
Algorithms for global optimization.

Unlike the local optimizers in this package, the methods here do not
require an initial guess and search the whole box-constrained domain
for the global minimum. Two algorithms are offered, covering the two
classical regimes: a deterministic, derivative-free space-partitioning
method (DIRECT) with no random component and fully reproducible runs;
and a stochastic, Tsallis-statistics-based annealing method (Dual
Annealing) that combines a heavy-tailed visiting distribution and a
power-law acceptance criterion with a periodic local search refinement.

Global optimizers are orders of magnitude slower than local methods,
and in general it is not possible to certify that a candidate is the
global minimum, so termination is governed by a maximum number of
iterations, a maximum number of function evaluations, a target
objective value, or geometry/stagnation tolerances specific to each
algorithm.

Contains
--------
_direct_optimizer
    DIRECT (DIviding RECTangles) deterministic global optimizer.
    Partitions the box-constrained search space into hyperrectangles
    and identifies the potentially optimal ones at each iteration
    through a lower convex-hull construction over the (diameter,
    function value) plane, then trisects them along their longest
    edges.

_annealing_optimizer
    Dual Annealing global optimizer. Combines Generalized Simulated
    Annealing (GSA) - a Tsallis q-Gaussian visiting distribution
    sampled via the Student-t equivalence, a q-exponential acceptance
    criterion and a power-law cooling schedule - with a deterministic
    local search method triggered on every significant improvement and
    on a fixed reannealing interval.
"""

# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

import numpy as np
from typing import Any, List, Optional, Union, Tuple, Dict, Callable, Iterable
from dataclasses import dataclass, field

from ._global_optimization_utils import (
    _validate_direct_parameters, _validate_annealing_parameters
)
from ._utils import OptimizationResult



def _direct_optimizer(
    function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    common_options: Dict[str, Any],
    callback: Callable[[int,np.ndarray,float], None],
    max_fun_evals: Optional[int],
    max_iters_with_no_progress: int,
    max_rectangles: int,
    max_diameter: float,
    f_min: float,
    len_tol: float,
    vol_tol: float
)-> Dict[str, Any]:
    """
    DIRECT (DIviding RECTangles) global optimizer.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function to be minimized. Must accept a 1D numpy
        array and return a scalar float.
    bounds : np.ndarray
        A array of (min, max) pairs for each dimension of the solution
        space. Must have shape (2, n_dimensions).
    common_options : Dict[str, Any]
        Configuration dictionary containing general options:
        - 'max_iters' (int): Maximum number of iterations.
        - 'verbose' (bool): Whether to print progress to stdout.
        - 'verbose_freq' (int): Frequency of verbose output.
    callback : Callable[[int, np.ndarray, float], None]
        A callback function executed at the end of every iteration.
        Signature: `(iteration_idx, current_best_params,
        current_best_cost)`.
    max_fun_evals : Optional[int]
        The maximum number of function evaluations allowed.
    max_iters_with_no_progress : int
        The optimizer will terminate if the best solution does not improve
        significantly for this many consecutive iterations.
    max_rectangles : int
        Hard limit on the number of hyperrectangles to store in memory.
    max_diameter : float
        If finite, any rectangle with a diameter larger than this value
        will be forced to split, regardless of its function value
        (encourages exploration).
    f_min : float
        Target global minimum value. If the optimizer finds a cost <=
        f_min, it terminates successfully.
    len_tol : float
        Termination tolerance based on the normalized size (half-diameter)
        of the hyperrectangle containing the best point.
    vol_tol : float
        Termination tolerance based on the volume of the hyperrectangle
        containing the best point.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the optimization results:
        - 'success': bool, indicating if convergence criteria were met.
        - 'final_params': np.ndarray, the best parameters found.
        - 'final_cost': float, the function value at 'final_params'.
        - 'iteration_number': int, total iterations.
        - 'termination_reason': str, why the loop stopped.

    References
    ----------
    [1] Chapter 7.6 from Kochenderfer, M. J., & Wheeler, T. A. (2019).
        **Algorithms for Optimization**.
        MIT Press.
        ISBN: 978-0-262-03942-0
    """

    # Automatically generate __init__, __repr__ and __eq__.
    @dataclass
    class Hyperrectangle:
        # center in unit cube coords, shape (n,)
        center: np.ndarray
        # half widths in unit cube coords, shape (n,)
        half_widths: np.ndarray
        # objective at center (original coords)
        function_value: float
        # unique id
        id: int

        # Cached private variables, these attributes are not initialized in
        # the constructor (__init__) and will be computed later
        _diameter: float = field(init=False)
        _volume: float = field(init=False)

        def __post_init__(self):
            # Initialize cached values after the class is created
            self._recompute_geometry()

        def _recompute_geometry(self) -> None:
            """Recompute cached variables."""
            widths = 2.0 * self.half_widths
            self._diameter = float(np.linalg.norm(widths))
            self._volume = float(np.prod(widths))

        def set_half_widths(self, new_half_widths: np.ndarray) -> None:
            """Safely update half_widths and refresh cached variables."""
            self.half_widths = new_half_widths
            self._recompute_geometry()

        def diameter(self) -> float:
            """Cached Euclidean diameter."""
            return self._diameter

        def volume(self) -> float:
            """Cached volume in unit cube coordinates."""
            return self._volume


    class Direct:
        def __init__(
            self,
            func: Callable[[np.ndarray], float],
            bounds: np.ndarray,
            maxfun: Optional[int],
            common_options: Dict[str, Any],
            max_iters_with_no_progress: int,
            max_rectangles: int,
            max_diameter: float,
            f_min: float,
            len_tol: float,
            vol_tol: float,
            callback: Callable[[int,np.ndarray,float], None]
        ):
            """
            Initializes the DIRECT (DIviding RECTangles) solver.
            """

            _validate_direct_parameters(
                maxfun, max_iters_with_no_progress, max_rectangles,
                max_diameter, f_min, len_tol, vol_tol
            )


            self.func = func
            self.bounds = bounds
            self.dim_num = bounds.shape[1]
            self.max_iters = common_options['max_iters']
            self.verbose = common_options['verbose']
            self.verbose_freq = common_options['verbose_freq']
            self.max_iters_with_no_progress = max_iters_with_no_progress
            self.max_diameter = max_diameter
            self.f_min_user = f_min
            self.callback = callback


            # Heuristics:
            self.maxfun = 1000000*self.dim_num if maxfun == 'auto' else maxfun

            self.max_rectangles = (
                1000000*self.dim_num if max_rectangles == 'auto'
                else max_rectangles)

            self.len_tol = (
                (1e-5)*(0.1/self.dim_num) if  len_tol == 'auto'
                else len_tol
            )

            self.vol_tol = (
                (self.len_tol / np.sqrt(self.dim_num ))**self.dim_num
                if vol_tol == 'auto'
                else vol_tol
            )


            # State of the algorithm:
            self.rectangles: List[Hyperrectangle] = []
            self._next_id = 1
            # In minimizer it is already being tracked, done here too for
            # simplicity
            self.function_evaluations = 0
            self.iteration_number = 0
            self.best_x: Optional[np.ndarray] = None
            self.prev_best_x: Optional[np.ndarray] = None
            self.best_f: float = np.inf
            self.prev_best_f: float = np.inf


            # Initialize with single rectangle covering unit cube
            center = np.full(self.dim_num, 0.5)
            half_widths = np.full(self.dim_num, 0.5)
            function_at_first = self._eval_at_unit(center)
            first_rectangle = Hyperrectangle(
                center=center,
                half_widths=half_widths,
                function_value=function_at_first,
                id=self._get_next_id()
            )
            self.rectangles.append(first_rectangle)


            self._update_best(first_rectangle)


        def _get_next_id(self) -> int:
            current_id = self._next_id
            self._next_id += 1
            return current_id

        def _original_to_unit(self, x_orig: np.ndarray) -> np.ndarray:
            """Scale a point from original bounds to unit cube [0,1]^n."""
            lows = self.bounds[0, :]
            highs = self.bounds[1, :]
            return (np.asarray(x_orig) - lows) / (highs - lows)

        def _unit_to_original(self, x_unit: np.ndarray) -> np.ndarray:
            """Scale a point from unit cube [0,1]^n to original bounds."""
            lows = self.bounds[0, :]
            highs = self.bounds[1, :]
            return lows + x_unit * (highs - lows)

        def _eval_at_unit(self, x_unit: np.ndarray) -> float:
            """Evaluate objective at a unit-cube point; update counter and
            return scalar."""
            x_orig = self._unit_to_original(x_unit)
            self.function_evaluations += 1
            return self.func(x_orig)

        def _update_best(self, rect: Hyperrectangle) -> None:
            """If this rectangle is the new best, update global best."""
            if rect.function_value < self.best_f:
                self.best_f = rect.function_value
                self.best_x = self._unit_to_original(rect.center).copy()



        def _split_rectangle(self, rect: Hyperrectangle
        ) -> List[Hyperrectangle]:
            """
            Trisect a rectangle along its longest side(s).

            When several dimensions tie for the longest side, the rule is:

            1. Test the objective function at parent `rect`
            center +/- delta * e_i for every longest dimension i, where
            delta = max_half / 3 in unit-cube coords.

            2. Then min_tested = min(f(c + delta * e_i), f(c - delta * e_i)).

            3. Sort the tied dimensions by min_tested in ascending order. The
            smallest min_tested is split first.

            4. Last, trisect along the sorted dimensions sequentially: the
            parent is replaced by three children along the first dimension; the
            central child is then trisected along the second dimension and so
            on.

            The result is exactly 2 * len(largest_dims) new children plus the
            shrunken parent, and they dont overlap the parent.

            Returns
            -------
            List[Hyperrectangle]
                The newly created rectangles. The parent `rect` is shrunk in
                place and remains valid in the rectangle set.
            """

            max_half = float(np.max(rect.half_widths))
            largest_dims = np.where(np.isclose(rect.half_widths, max_half))[0]
            delta = max_half / 3.0


            # 1-2: probe both sides of the parent center along each
            # longest dimension and keep the better of the two.
            tested_rectangles: List[Tuple[int, float, float, float]] = []

            for dim in largest_dims:
                plus_center = rect.center.copy()
                minus_center = rect.center.copy()

                plus_center[dim] = float(
                    np.clip(plus_center[dim] + delta, 0., 1.)
                )
                minus_center[dim] = float(
                    np.clip(minus_center[dim] - delta, 0., 1.)
                )

                function_at_plus = self._eval_at_unit(plus_center)
                function_at_minus = self._eval_at_unit(minus_center)
                minimum_tested = min(function_at_plus, function_at_minus)

                tested_rectangles.append(
                    (int(dim), minimum_tested, function_at_plus,
                     function_at_minus)
                )


            # 3: sort dimensions by minimum_tested ascending (smallest
            # minimum_tested splits first so the most promising side gets the
            # smallest children).
            tested_rectangles.sort(
                key=lambda tested_rentangle: tested_rentangle[1]
            )


            # 4: trisect along the sorted dimensions sequentially in the just
            # sorted order.
            new_rectangles: List[Hyperrectangle] = []
            # keep track off the central piece that is repeatedly subdivided
            central_rectangle_half_widths = rect.half_widths.copy()

            for (
                current_dimension, _, function_at_plus, function_at_minus
            ) in tested_rectangles:

                # Children inherit the current parent half-widths, with
                # only the split dimension reduced to one third. The side
                # childrens are as wide as the central piece in every dimension
                # except current_dimension.
                child_half_widths = central_rectangle_half_widths.copy()
                child_half_widths[current_dimension] = (
                    central_rectangle_half_widths[current_dimension] / 3.0
                )

                for sign, function_at_side in (
                    (+1.0, function_at_plus,), (-1.0, function_at_minus)
                ):
                    child_center = rect.center.copy()
                    child_center[current_dimension] = float(
                        np.clip(
                            rect.center[current_dimension] + sign * delta,
                            0.0, 1.0
                        )
                    )

                    if np.isfinite(function_at_side):
                        child = Hyperrectangle(
                            center=child_center,
                            half_widths=child_half_widths.copy(),
                            function_value=function_at_side,
                            id=self._get_next_id(),
                        )

                        new_rectangles.append(child)
                        self._update_best(child)

                # The surviving central piece shrinks in this dimension only.
                central_rectangle_half_widths[current_dimension] = (
                    central_rectangle_half_widths[current_dimension] / 3.0
                )

            # The parent center and function_value are unchanged as the
            # parent's center has not been re-evaluated.
            rect.set_half_widths(central_rectangle_half_widths)

            return new_rectangles



        def _find_potentially_optimal(self) -> List[Hyperrectangle]:
            """
            Identify potentially optimal rectangles using a lower
            convex-hull approach.

            Mathematical description
            -----------------------
            This function identifies the set of "potentially optimal"
            hyperrectangles in the DIRECT algorithm (DIviding RECTangles)
            using a geometric lower convex hull construction.

            ---

            Each hyperrectangle i in the search space is characterized by:

                -d_i = its diameter (a measure of size/uncertainty)
                -f_i = the objective function value at its center

            DIRECT balance exploitation, select rectangles with small f_i
            (promising) and exploration, select rectangles with large d_i
            (uncertain).

            A rectangle i is "potentially optimal" if there exists some
            Lipschitz constant K >= 0 such that [1]:

                f_i - K * d_i <= f_j - K * d_j    for all other rectangles j

            Meaning that:

            - f_i - K * d_i is a lower bound estimate for the minimum
            value that could be hidden inside rectangle i.

            - If K is small trust f_i more (exploitation favored),
            If K is large trust d_i more (exploration favored)

            - Rectangle i is potentially optimal if a K can be found in order
            to make its lower bound better than all others


            A geometric interpretation is to plot each rectangle as a point
            in 2D space, where:

            - x-axis: d_i (rectangle diameter)
            - y-axis: f_i (function value)

            For a fixed K, the expression f_i - K * d_i represents the
            y-intercept of a line with slope K and passing through (d_i, f_i),
            the equation of this line is:

                y = f_i - K * d_i + K * x

            When x = 0, y-intercept = f_i - K·d_i

            The following is a visualization of the convex-hull:

                f (function value)
                ^
                |           • (d_2, f_2)
                |                            Line with slope K
                |   • (d_1, f_1)            /
                |                       /
                |       (d_i, f_i) •/
                |               /
                |           /
                |       /
                |___/________________________> d (diameter)
                    ^
                    y-intercept = f_i - K·d_i

            Rectangle i is potentially optimal if one can rotate a line
            with slope K >= 0 such that point (d_i, f_i) gives the
            smallest y-intercept among all rectangles.

            ---

            The potentially optimal rectangles are exactly those on the lower
            convex hull of the point set (d_i, f_i) in 2D space, meaning:

            -Imagine sweeping a line with increasing slope K from
            K=0 (horizontal) to K=inf (vertical).

            -At each slope K, the point with the minimum y-intercept
            f_i - K * d_i touches the x-axis first.

            -As K increases, the touching point "moves along" the
            lower convex hull from left to right.

            -Points not on the lower hull are never touched for any
            K >= 0, so they cannot be potentially optimal.

            knowing this, the implementation:

            -First, sort points by two keys, primary by d_i in ascending
            order and secondary by f_i in ascending order; done this way so
            the ties are broken consistently.
            -Second, compute the lower convex hull of these points and return
            them

            ---

            In order to build the lower convex hull a stack-based (Monotone
            Chain) algorithm is used:

            For each new point i (from left to right):

            1. Look at the last two points on the stack: k and j
            2. Check if adding point i makes j non-convex
            3. If yes, remove j from the stack as it's not on the hull
            4. Repeat until the hull is convex, then add i

            The convexity is tested using a 2D Cross Product:

            Given three points k, j, i (in left-to-right order):

                Vector k->j: (x_j - x_k, y_j - y_k)
                Vector j->i: (x_i - x_j, y_i - y_j)

                Cross product: cross = (x_j - x_k) * (y_i - y_j) -
                                       (y_j - y_k) * (x_i - x_j)

            if cross > 0, the path k->j->i turns left (convex for lower
            hull) -> keep point j

            if cross <= 0, the path k->j->i turns right or is straight
            concave or collinear -> remove point j (not on lower hull)

            By removing all points that create concave turns, then the final
            hull is convex and forms the lower envelope.

            References
            ----------
            [1] Part 7.6 from Kochenderfer, M. J., & Wheeler, T. A. (2019).
                **Algorithms for Optimization**.
                MIT Press.
                ISBN: 978-0-262-03942-0
            """

            if not self.rectangles:
                return []

            diameters = np.array([r.diameter() for r in self.rectangles])
            function_values = np.array(
                [r.function_value for r in self.rectangles]
            )

            # Sort by diameter (primary key) and function_value (secondary)
            indexes = np.lexsort((function_values, diameters))
            diameters_sorted = diameters[indexes]
            function_values_sorted = function_values[indexes]

            rects_sorted = [self.rectangles[i] for i in indexes]

            # Build lower convex hull
            stack: List[int] = []

            # i -> first point index, j -> second point index,
            # k -> third point index
            for i in range(len(diameters_sorted)):
                while len(stack) >= 2:
                    j = stack[-1]
                    k = stack[-2]

                    # 2D cross product value
                    xk, yk = diameters_sorted[k], function_values_sorted[k]
                    xj, yj = diameters_sorted[j], function_values_sorted[j]
                    xi, yi = diameters_sorted[i], function_values_sorted[i]
                    cross = (xj - xk) * (yi - yj) - (yj - yk) * (xi - xj)


                    # Lower hull rule: drop point j only when the turn k->j->i
                    # is strictly right (cross < 0). Collinear triples
                    # (cross= 0) are kept: at that slope K every collinear
                    # point shares the same y-intercept f_i - K * d_i, so all
                    # of them are equally potentially optimal and must remain
                    # candidates.
                    if cross <= -1e-12:
                        stack.pop()

                    else:
                        break

                stack.append(i)

            # Convert stack indices back to original rectangle objects
            return [rects_sorted[j] for j in stack]



        def _check_termination(self) -> Tuple[bool, str]:
            """Return (should_stop, message)."""
            if self.function_evaluations >= self.maxfun:
                return True, (f"Maximum number of function evaluations "
                              f"({self.maxfun}) reached.")

            if self.iteration_number >= self.max_iters:
                return True, (f"Maximum number of iterations "
                              f"({self.max_iters}) reached.")

            if np.isfinite(self.f_min_user) and np.isfinite(self.best_f):
                f_target = self.f_min_user
                f_current = self.best_f
                if f_current <= f_target:
                    return True, (
                        f"Current function value ({f_current}) is less than or "
                        f"equal to the user-specified target ({f_target})."
                    )

            if self.best_x is not None:
                best_unit = self._original_to_unit(self.best_x)

                # Find the rectangle containing the nearest center
                # closest to the best point
                dists = [np.linalg.norm(r.center - best_unit) for r in
                         self.rectangles]

                best_rect = self.rectangles[int(np.argmin(dists))]

                # volume tolerance
                vol_ratio = best_rect.volume() # Total unit volume is 1
                if vol_ratio <= self.vol_tol:
                    return True, (f"Volume of best rectangle {vol_ratio:.3e} "
                                  f"<= vol_tol.")

                # length tolerance
                diag_half = 0.5 * best_rect.diameter()
                if diag_half <= self.len_tol:
                    return True, f"Half of diagonal {diag_half:.3e} <= len_tol."

            if len(self.rectangles) > self.max_rectangles:
                return True, "Too many rectangles generated (memory safety)."

            return False, ""



        # Single iteration
        def iterate(self) -> None:
            """
            One DIRECT iteration:
            - Identify potentially optimal rectangles (maybe multiple).
            - Split each and add new rectangles.
            """

            # find Pontentially Optimal candidate rectangles
            po_candidates = self._find_potentially_optimal()

            candidate_ids = {r.id for r in po_candidates}

            # Add any rectangles that are too large
            if np.isfinite(self.max_diameter):
                # The L2 diameter of the unit cube is sqrt(dim_num)
                unit_cube_diameter = np.sqrt(self.dim_num)
                diameter_threshold = self.max_diameter * unit_cube_diameter

                for current_rectangle in self.rectangles:
                    # don't add duplicates
                    if current_rectangle.id not in candidate_ids:
                        if current_rectangle.diameter() > diameter_threshold:
                            candidate_ids.add(current_rectangle.id)


            new_rects = []
            # For each candidate, split
            for current_rectangle in list(self.rectangles): # iterate over copy
                if current_rectangle.id in candidate_ids:
                    # _split_rectangle shrinks current_rectangle's half_widths
                    # in place
                    created = self._split_rectangle(current_rectangle)
                    new_rects.extend(created)


            self.rectangles.extend(new_rects)



        # Main run method
        def run(self) -> Dict[str,Any]:
            """Run until one termination condition is met."""
            termination_reason = ""

            stop, termination_reason = self._check_termination()

            if self.verbose:
                print(f"{'Iter':>5s} | Best cost | Best x")
                print("-" * 60)
                print(f"{self.iteration_number:5d} | {self.best_f:.6g} | "
                      f"{self.best_x}")
            if self.callback:
                self.callback(self.iteration_number, self.best_x, self.best_f)

            no_progress_counter=0
            while not stop:

                self.prev_best_f = self.best_f
                if not np.isfinite(self.best_f):
                    termination_reason = ("Objective function returned an "
                                          "invalid value (NaN or +-Inf).")
                    break

                self.prev_best_x = self.best_x.copy()

                self.iteration_number += 1
                self.iterate()

                if (self.verbose and
                    (self.iteration_number % max(1, self.verbose_freq) == 0 or
                     self.iteration_number == self.max_iters)
                ):
                    print(f"{self.iteration_number:5d} | {self.best_f:.6g} | "
                          f"{self.best_x}")
                if self.callback:
                    self.callback(
                        self.iteration_number, self.best_x, self.best_f
                    )

                # Stagnation check: best cost and best point are both unchanged
                # (within tolerance) compared to the previous iteration.
                best_f_unchanged = np.isclose(self.best_f, self.prev_best_f)
                best_x_unchanged = np.all(
                    np.isclose(self.best_x, self.prev_best_x)
                )

                if best_f_unchanged and best_x_unchanged:
                    no_progress_counter += 1
                    if (
                        no_progress_counter >=
                        self.max_iters_with_no_progress
                    ):
                        termination_reason=(
                            f"Current best point and cost function value "
                            f"didn't change from last "
                            f"{self.max_iters_with_no_progress} "
                            f"iterations while it is still above tolerance."
                        )
                        break
                else:
                    no_progress_counter=0

                stop, termination_reason = self._check_termination()
                if stop:
                    break



            converged_reasons = (
                "Current function value", # f_min target reached
                "Volume of best rectangle", # vol_tol reached
                "Half of diagonal", # len_tol reached
                "didn't change from last", # stagnation reached
            )
            reached_convergence = any(
                token in termination_reason for token in converged_reasons
            )
            # Success requires a finite best value and a termination reason
            # different from maximum iterations, maximum function evaluations
            # or algorithm stagnation.
            success = (
                self.best_x is not None
                and np.isfinite(self.best_f)
                and reached_convergence
            )

            return {
                "success": bool(success),
                "final_params": (
                    np.asarray(self.best_x) if self.best_x is not None else None
                ),
                "final_cost": self.best_f,
                "iteration_number": self.iteration_number,
                "termination_reason": termination_reason,
            }



    direct = Direct(
        func=function, bounds=bounds, maxfun=max_fun_evals,
        common_options=common_options,
        max_iters_with_no_progress=max_iters_with_no_progress,
        max_rectangles=max_rectangles, max_diameter=max_diameter,
        f_min=f_min, len_tol=len_tol, vol_tol=vol_tol, callback=callback
    )
    return direct.run()





def _annealing_optimizer(
    function: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    common_options: Dict[str, Any],
    callback: Callable[[int,float,np.ndarray,float,np.ndarray],None],
    inner_solver: Callable[[Callable,np.ndarray,np.ndarray],OptimizationResult],
    x0: Optional[Union[Iterable[float], np.ndarray]],
    seed: Optional[int],
    max_fun_evals: int,
    f_min: float,
    temp_init: float,
    no_local_search: bool,
    reanneal_interval: Union[str, int],
    visit_const: float,
    accept_const: float,
    cooling_power: float,
)-> Dict[str, Any]:
    """
    Dual Annealing optimizer.

    This function implements the core logic for dual annealing, a global
    optimization algorithm that combines Generalized Simulated Annealing (GSA)
    with a local search method.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        The objective function to be minimized. Must take a 1-D numpy array
        as input and return a scalar float.
    bounds : np.ndarray
        A array of (min, max) pairs for each dimension of the solution
        space. Must have shape (2, n_dimensions).
    common_options : Dict[str, Any]
        Configuration dict. Must contain:
        - 'max_iters': int
        - 'verbose': bool
        - 'verbose_freq': int
    callback : Optional[Callable]
        User callback invoked at the end of every iteration, after the
        acceptance test but before the local-search step. Signature:

            callback(iter_num, candidate_cost, candidate_x,
                    best_cost, best_x)

        where candidate_cost and candidate_x describe the point proposed this
        iteration (whether or not it was accepted), and best_cost / best_x
        describe the global best so far.
    inner_solver : Callable
        The local search function used for refinement steps. Must accept the
        signature solver(func, x0, bounds) and return an OptimizationResult.
        The annealing loop reads final_params, final_cost, func_evals_number,
        and grad_evals_number from it; the remaining attributes (success,
        termination_reason, elapsed_time, iteration_number) are ignored.
    x0 : Iterable[float], np.ndarray, optional
        Initial guess. If None, a random point within bounds is chosen.
    seed : int, optional
        Seed for the random number generator for reproducible results.
    max_fun_evals : int or 'auto'
        Total evaluation budget (annealing trials + local-search calls). When
        'auto', resolves to 3 * 10^6 * n_dimensions.
    f_min : float
        Target global minimum value. If the optimizer finds a cost <=
        f_min, it terminates successfully.
    temp_init : float
        The initial temperature for the annealing schedule.
    no_local_search : bool
        If True, skip all local search steps.
    reanneal_interval : int or 'auto'
        Number of iterations between scheduled local search calls.
        Also controls the stagnation window for forced reannealing.
        If 'auto', set to max(max_iters // 10, 1).
    visit_const : float
        Tsallis entropic index for the visiting distribution, q_v.
        Must be in the open interval (1, 3). Controls tail heaviness:

        - q_v -> 1: Gaussian steps (classical SA limit)
        - q_v = 2: Cauchy steps (heaviest stable tails)
        - q_v -> 3: Lévy-like super-diffusion

        The Student-t equivalence gives degrees of freedom
        v = (3 - q_v) / (q_v - 1).
    accept_const : float
        Tsallis entropic index for the acceptance criterion, q_a.
        Must be in the open interval (1, 3). Controls acceptance shape:

        - q_a -> 1: standard Metropolis exp(-Delta_f/T_a)
        - q_a > 1: power-law acceptance (heavier-tailed than Metropolis)
        - q_a -> 3: very permissive acceptance, accepts many bad steps
    cooling_power : float
        Exponent of the cooling schedule actually implemented:

            T_k = T_init *
                  exp(-cooling_power * (k_eff / max_iters) ** (1 / n))

        where:
        - k_eff is the iteration count since the last reanneal.
        - n is the problem dimension.

        cooling_power = 1.0 is the default, larger cooling_power cools more
        aggressively, smaller cooling_power more gently. The schedule
        asymptotes to T_init * exp(-cooling_power), so the algorithm always
        retains some exploration noise.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the optimization results:
        - 'success': bool, indicating if convergence criteria were met.
        - 'final_params': np.ndarray, the best parameters found.
        - 'final_cost': float, the function value at 'final_params'.
        - 'func_evals_number': int, total function calls.
        - 'iteration_number': int, total iterations.
        - 'termination_reason': str, why the loop stopped.

    Mathematical Description
    ------------------------
    In this code Dual Annealing (Generalized Simulated Annealing, GSA, with
    local search) is implemented, a stochastic global optimization method
    inspired by the annealing process that takes place in metallurgy, whereby
    annealing a molten metal causes it to achieve its global minimum in terms
    of thermodynamic energy.

    It is called "Dual" because, in addition to propose trial points based on
    a visiting distribution and accept them if certain criteria is met, it
    also uses a deterministic local search method in order to reach local
    minima faster when a new best point is found.

    ---

    The core part of Generalized Annealing is the visiting distribution that
    comes from the maximization of the Tsallis entropy (spread uncertainty as
    much as possible), arriving at a final distribution
    [1, eq.21 (setting D = 1)]:

        f(Delta_x, T_k; q_v) is proportional to

        beta(T_k) * [1 + (q_v - 1) * (Delta_x / beta(T))^2]^(-1/(q_v - 1))

    with:
    - Delta_x = is the difference between x_{k+1} - x_k
    - beta(T) = T_k^(1/(3-q_v)), and T_k the temperature at iteration k
    - q_v is a scalar that defines the distribution (entropic index), it must
    be 1 < q_v < 3, otherwise distribution is not normalizable.

    This distribution is called the Tsallis q-Gaussian, which generalizes the
    Gaussian (q_v -> 1) and Cauchy-Lorentz (q_v = 2) distributions and allows
    occasional Lévy-flight-like long jumps (q_v > 1), which greatly improves
    escape from local minima.

    This code implements the numpy's standard_t sampler directly because the
    q-Gaussian with index q_v is algebraically identical to a scaled
    Student-t distribution [4]. Specifically, if:

        T_i ~ Student-t(v),   v = (3 - q_v) / (q_v - 1)

    then:

        Delta_x_i = T_i / sqrt(3 - q_v)

    follows the q-Gaussian f_{q_v} exactly.

    ---

    The temperature T_k is defined to control the scale of the jumps and
    the acceptance probability. The code utilizes a generalized cooling
    schedule:

        T_k = T_0 * np.exp(-p * ((k/k_max) ** (1.0 / n)))

    where:
    -T _0 = initial temperature
    - k = current iteration
    - k_max = maximum iterations
    - p = cooling_power, p = 1.0: Linear cooling, p = 2.0: Quadratic
    cooling (faster), p > 1.0: Accelerated cooling, p < 1.0: Decelerated
    cooling
    - n = dimension number of the problem

    As k approaches k_max, T_k approaches 0 so the jumps are smmaller and
    worse solutions have less acceptance, the algorithm reduces the range
    of exploration (Levy flight length) and turning the search into a
    fine-tuning phase (exploitation).

    ---

    Once a candidate x_new is generated, it is accepted or rejected based
    on the change in function value Delta_f = f(x_{k+1}) - f(x_k):

    If Delta_f < 0 (improvement): The new state is always accepted.
    If Delta_f >= 0 (worse solution): The new state is accepted with a
    probability P given by the acceptance criterion that generalizes the
    Metropolis Criterion - q-exponential [1, Eq. 6] making T_a = T_k/k:

        P = max(0, 1 - (1-q_a) * k * Delta_f/T_k) ^ (1/(1-q_a))

    where:
    - q_a = is the Tsallis acceptance index
    - k = current iteration
    - Delta_f = f(x_{k+1}) - f(x_k)
    - T_k = Temperature at iteration k

    The standard Metropolis criterion can be derived as:

        q_a -> 1 : p -> exp(-Delta_f/T_a)      (standard Metropolis)
        q_a > 1 : power-law decay, heavier-tailed than Metropolis

    To refine the results found by the stochastic global search, a local
    optimizer is triggered periodically (by the reanneal_interval variable) or
    when a significant improvement is found. This hybrid approach ensures that
    the final result is not just a basin containing the global minimum, but the
    minimizer itself.

    References
    ----------
    [1] Tsallis, C. & Stariolo, D. A. (1996).
        **Generalized simulated annealing**.
        Physica A, 233(1-2), 395-406.
        https://doi.org/10.1016/S0378-4371(96)00271-3

    [2] Xiang, Y., Gubian, S. & Martin, F. (2017)
        **Generalized Simulated Annealing**.
        Computational Optimization in Engineering - Paradigms and applications.
        http://dx.doi.org/10.5772/66071

    [3] Chapter 8.3 from Kochenderfer, M. J., & Wheeler, T. A. (2019).
        **Algorithms for Optimization**.
        MIT Press.
        ISBN: 978-0-262-03942-0

    [4] Wikipedia contributors.
        **q-Gaussian distribution.**
        https://en.wikipedia.org/wiki/Q-Gaussian_distribution#Related_distributions
    """

    _validate_annealing_parameters(
        bounds,x0,seed,max_fun_evals,f_min,temp_init,no_local_search,
        reanneal_interval,visit_const,accept_const,cooling_power
    )

    max_iters=common_options['max_iters']
    verbose=common_options['verbose']
    verbose_freq=common_options['verbose_freq']

    lower_bound = bounds[0, :]
    upper_bound = bounds[1, :]
    bounds_diff = upper_bound - lower_bound

    dim_num = lower_bound.size


    # Heuristics
    _max_fun_evals = (
        3000000*dim_num if max_fun_evals == 'auto' else max_fun_evals
    )


    rng = np.random.default_rng(seed)


    if reanneal_interval == 'auto':
        _reanneal_interval = max(max_iters // 10, 1)

    else:
        _reanneal_interval = reanneal_interval


    _visit_const = float(visit_const)

    # q_v - 1 > 0
    visit_q_m1 = _visit_const - 1.0
    # Student-t v
    visit_degree_freedom = (3.0 - _visit_const) / visit_q_m1
    # 1/sqrt(3 - q_v)
    visit_scale = 1.0 / np.sqrt(3.0 - _visit_const)
    # 1/(3 - q_v)
    visit_temp_exp = 1.0 / (3.0 - _visit_const)


    _accept_const = float(accept_const)

    # (1 - q_a) < 0
    accept_one_m_qa = 1.0 - _accept_const
    # 1/(1 - q_a)
    accept_inv_exp = 1.0 / accept_one_m_qa


    if x0 is None:
        current_x = rng.random(dim_num) * (bounds_diff) + lower_bound
    else:
        current_x = np.clip(
            np.asarray(x0, dtype=float).ravel(), lower_bound, upper_bound
        )


    f_at_current_x = function(current_x)
    best_x = current_x.copy()
    best_f = f_at_current_x

    func_evals_local_search = 0
    func_evals_annealing = 1 # counts the initial evaluation
    grad_evals_local_search = 0

    last_improve_iter = 0
    last_reanneal_iter = 0


    def _is_f_min_reached(candidate_best_f: float) -> Tuple[bool, str]:
        """
        Test the user-supplied f_min target against the current best.

        Returns (True, message) if the user's target has been hit. Returns
        (False, "") otherwise.
        """
        if candidate_best_f <= f_min:
            return True, (
                f"The best function value found ({candidate_best_f}) "
                f"is less than or equal to the user-defined target "
                f"({f_min})."
            )

        return False, ""


    if verbose:
        print(f"{'Iter':>5s} | {'Current cost':>14s} | {'Current x':>15s} "
              f"| {'Best cost':>13s} | {'Best x':>15s}")
        print("-" * 70)


    termination_reason = ""

    # main algorithm
    for current_iter in range(1, max_iters + 1):
        if (func_evals_annealing + func_evals_local_search) >= _max_fun_evals:
            termination_reason = (
                f"Maximum number of function evaluations "
                f"({_max_fun_evals}) reached."
            )
            break

        # Iterations elapsed since the last reanneal. Equals current_iter
        # at iter 1 (no reanneal has happened yet) and resets to 1 on the
        # first iteration after a reanneal.
        iterations_since_last_reanneal = current_iter - last_reanneal_iter
        # Temperature schedule: exponential cooling
        temp = (
            temp_init * np.exp(-cooling_power *
            ((iterations_since_last_reanneal/max_iters) ** (1.0 / dim_num)))
        )


        # Visiting distribution: q-Gaussian via Student-t
        t_samples = rng.standard_t(visit_degree_freedom, size=dim_num)
        temp_scale = np.power(temp, visit_temp_exp)
        delta = temp_scale * visit_scale * t_samples

        candidate_x = np.clip(current_x + delta, lower_bound, upper_bound)
        function_at_candidate = function(candidate_x)
        func_evals_annealing += 1


        if verbose and (
            current_iter % max(1, verbose_freq) == 0
            or current_iter == max_iters
        ):
            print(f"{current_iter:>5} | {function_at_candidate:.2e} "
                  f"| {candidate_x} | {best_f:.2e} | {best_x}")
        if callback:
            callback(
                current_iter, function_at_candidate, candidate_x, best_f,
                best_x
            )


        if not np.isfinite(function_at_candidate):
            termination_reason = ("Objective function returned an invalid "
                                  "value (NaN or +-Inf).")
            break

        delta_f = function_at_candidate - f_at_current_x

        # Metropolis acceptance criterion
        if delta_f <= 0.0:
            accepted = True
        else:
            base = (
                1.0 - accept_one_m_qa * delta_f *
                iterations_since_last_reanneal / temp
            )
            # Limit base as when it goes negative it is undefined when raised
            # to a non-integer exponent.
            prob = np.power(base, accept_inv_exp)
            accepted = rng.random() < prob

        if accepted:
            current_x = candidate_x
            f_at_current_x = function_at_candidate


        # Test best function value
        improved = function_at_candidate < best_f
        if improved:
            best_x = candidate_x.copy()
            best_f = function_at_candidate
            last_improve_iter = current_iter

            stop, termination_reason = _is_f_min_reached(best_f)
            if stop: break


        # LOCAL SEARCH
        # Run local search at start, when scheduled and at last iter
        trigger_scheduled = (current_iter == 1 or
                             current_iter % _reanneal_interval == 0 or
                             current_iter == max_iters)

        # Any improvement on the global best, plus the scheduled (start, every
        # reanneal_interval iters, end). This is a memetic-algorithm:
        # heavy-tailed global explorer locates basins, the local solver
        # polishes them.
        if not no_local_search and (improved or trigger_scheduled):

            inner_solver_result = inner_solver(function, best_x, bounds)
            func_evals_local_search += inner_solver_result.func_evals_number
            grad_evals_local_search += inner_solver_result.grad_evals_number

            if (
                (inner_solver_result.final_cost is None
                or not np.isfinite(inner_solver_result.final_cost))
                or
                (inner_solver_result.final_params is None
                or not np.any(np.isfinite(inner_solver_result.final_params)))
            ):
                termination_reason = (
                    f"Inner solver produced an invalid result: "
                    f"{inner_solver_result.termination_reason}"
                )
                break

            if inner_solver_result.final_cost < best_f:
                best_f = inner_solver_result.final_cost
                best_x = inner_solver_result.final_params.copy()

                stop, termination_reason = _is_f_min_reached(best_f)
                if stop: break

                # when new best found, make it the current point
                current_x = best_x.copy()
                f_at_current_x = best_f

                last_improve_iter = current_iter

        # If no improvement, reset chain to best_x.
        # Force it back to the promising area.
        if (current_iter - last_improve_iter) > _reanneal_interval:

            current_x = best_x.copy()
            f_at_current_x = best_f

            last_improve_iter = current_iter
            last_reanneal_iter = current_iter


    if not termination_reason:
        termination_reason = (
            f"Maximum number of iterations ({max_iters}) reached."
        )


    success = np.isfinite(best_f) and best_x is not None

    return {
        "success": bool(success),
        "final_params": best_x,
        "final_cost": best_f,
        "func_evals_number": func_evals_annealing + func_evals_local_search,
        "grad_evals_number": grad_evals_local_search,
        "iteration_number": current_iter,
        "termination_reason": termination_reason,
    }
