"""CLI for the explicit-q copy of both small closure suites.

Examples::

    ../pixel/venv/bin/python -m closure_small_q_integration.run_closure --Q 2
    ../pixel/venv/bin/python -m closure_small_q_integration.run_closure \
        --truth NNPDF --Q 2 --modes dy exp both --no-plots
"""

from __future__ import annotations

from small_closure_variant import run_q_integration_cli


def main(argv=None) -> None:
    run_q_integration_cli(argv)


if __name__ == "__main__":
    main()
