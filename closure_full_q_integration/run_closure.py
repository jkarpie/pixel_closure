"""CLI for full-scale explicit-q closure sampling.

Example::

    ../pixel/venv/bin/python -m closure_full_q_integration.run_closure \
        --truth JAM --data union --Q 2 --modes both --no-plots
"""

from __future__ import annotations

from .variant import main


if __name__ == "__main__":
    main()
