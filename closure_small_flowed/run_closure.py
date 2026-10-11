"""CLI for the T=0.2 flowed-thimble copy of both small closure suites.

Examples::

    ../pixel/venv/bin/python -m closure_small_flowed.run_closure --Q 2
    ../pixel/venv/bin/python -m closure_small_flowed.run_closure \
        --truth JAM --Q 2 --modes exp both --no-plots
"""

from __future__ import annotations

from small_closure_variant import FLOWED, run_cli


def main(argv=None) -> None:
    run_cli(FLOWED, argv)


if __name__ == "__main__":
    main()
