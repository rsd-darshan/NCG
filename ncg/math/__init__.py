"""
NCG math utilities — convergence diagnostics for meta-parameters.

Under mild smoothness assumptions, the meta-parameter gradient flow converges
to a fixed point φ* where F(φ*) = 0. At this point, α* reflects learned
confidence calibration, β* reflects optimal complexity penalty, and λ* reflects
optimal regularisation strength. Monotonic decrease alone does not confirm
convergence — stabilisation above zero is required.
"""

from ncg.math.convergence import (
    compute_theoretical_fixed_point,
    diagnose_convergence,
    run_diagnostics,
)

__all__ = [
    "diagnose_convergence",
    "run_diagnostics",
    "compute_theoretical_fixed_point",
]
