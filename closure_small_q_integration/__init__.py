"""Small JAM/NNPDF closures using explicit retained-field q integration."""

from small_closure_variant import (
    q_integration_activated as activated,
    run_q_integration_one as run_one,
)

__all__ = ["activated", "run_one"]
