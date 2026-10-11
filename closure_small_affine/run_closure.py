"""CLI for the affine-contour copy of both small closure suites.

Examples::

    ../pixel/venv/bin/python -m closure_small_affine.run_closure --Q 2
    ../pixel/venv/bin/python -m closure_small_affine.run_closure \
        --truth NNPDF --Q 2 --modes exp both --no-plots
"""

from __future__ import annotations

from small_closure_variant import AFFINE, run_cli


def main(argv=None) -> None:
    run_cli(AFFINE, argv)


if __name__ == "__main__":
    main()
