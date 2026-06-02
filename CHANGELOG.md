# Changelog

All notable changes to **Ripples** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [0.1.0] - 2026-06

Initial public release.

### Added
- `ripples.differentiation` submodule.
  - `nth_numerical_derivative`: builds a callable that evaluates the
    n-th partial-derivative tensor of a scalar function at one or many
    points. Supports plain central finite differences,
    Richardson-extrapolated central differences, and the complex-step
    method, with automatic per-coordinate step-size selection.
  - `numerical_hessian_vector_product`: computes `H(point) @ vector`
    via a directional finite-difference stencil applied to the
    gradient, without ever forming the full Hessian.
  - `DifferentiationResult`: immutable, ndarray-like wrapper carrying
    the derivative, optional Romberg-style truncation-error estimate,
    and full configuration metadata.

- `ripples.optimization` submodule.
  - `minimizer`: single unified entry point that dispatches to
    Nesterov, Adam, nonlinear conjugate gradient, BFGS, L-BFGS,
    Newton-CG, trust-ncg, trust-lanczos, DIRECT, and dual annealing.
    Supports bounds and equality / inequality constraints through an
    Augmented Lagrangian wrapper.
  - `OptimizationResult`: immutable, ndarray-like wrapper carrying the
    final parameters, cost, evaluation counts, elapsed time,
    termination reason, and full configuration metadata. Exposes
    conversion helpers (`as_array`, `as_float`, `to_list`, `to_dict`),
    equality (`==`, `allclose`), and a `summary` method.

[0.1.0]: https://github.com/ripples-sci/ripples/releases/tag/v0.1.0
