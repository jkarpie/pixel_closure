"""Aggregate independent JAM L1 posterior-moment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pixel.diagnostics import aggregate_artifact_files
from pixel.diagnostics import save_posterior_moments

from . import config as cfg
from . import fit, generate


def _run_campaign(q_key, mode, seeds, output):
    artifacts = []
    fields = operators = None
    for seed in seeds:
        _, fields, operators = generate.generate_member(
            q_key,
            noise=True,
            seed=int(seed),
            fields=fields,
            operators=operators,
        )
        truth = fit.load_truth(q_key)
        result = fit.run_fit(q_key, mode, seed=int(seed))
        moments = result["posterior_moments"]
        field_by_name = {field.name: field for field in result["model"].fields}
        truth_nodes = np.asarray(truth["x_nodes"], dtype=float)
        nodes, truth_values = [], []
        for row in moments.q_layout:
            name = str(row["name"])
            x = np.asarray(field_by_name[name].nodes, dtype=float)
            curve = (truth.get("curves") or {}).get(name)
            nodes.append(x)
            truth_values.append(
                np.full(x.shape, np.nan)
                if curve is None
                else np.interp(x, truth_nodes, np.asarray(curve, dtype=float))
            )
        artifact = output.parent / f"{output.stem}_seed{int(seed)}.npz"
        save_posterior_moments(
            artifact,
            moments,
            truth=np.concatenate(truth_values),
            nodes=np.concatenate(nodes),
            metadata={
                "suite": "JAM", "q_key": q_key, "mode": mode,
                "l1_seed": int(seed), "chi2_fit": result["chi2"],
                "n_physical_data": result["n_data"] - sum(
                    data.n_data for data in result["model"].datasets
                    if data.name.startswith("cons_")
                ),
            },
        )
        artifacts.append(artifact)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--Q", dest="q_key", choices=list(cfg.TRUTH_Q_CHOICES))
    parser.add_argument("--mode", choices=list(cfg.TEST_MODES))
    parser.add_argument("--field", action="append", dest="fields")
    parser.add_argument("--pca-rcond", type=float, default=1.0e-10)
    parser.add_argument("--chi2-law", type=Path)
    parser.add_argument("--effective-dof", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts
    if args.seeds:
        if artifacts or args.q_key is None or args.mode is None:
            parser.error("--seeds requires --Q and --mode and cannot include artifacts")
        artifacts = _run_campaign(args.q_key, args.mode, args.seeds, args.output)
    if len(artifacts) < 2:
        parser.error("provide at least two artifacts or at least two --seeds")
    chi2_law = None
    if args.chi2_law is not None:
        chi2_law = json.loads(args.chi2_law.read_text())
    report = aggregate_artifact_files(
        artifacts,
        field_names=args.fields,
        pca_rcond=args.pca_rcond,
        chi2_law=chi2_law,
        effective_dof=args.effective_dof,
    )
    # Binomial standard error makes explicit why ten universes are a pilot,
    # not a 68%-coverage measurement.
    report["xi_binomial_standard_error"] = (
        report["xi_1sigma"] * (1.0 - report["xi_1sigma"])
        / report["n_universes"]
    ) ** 0.5
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
