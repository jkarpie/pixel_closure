"""Check the shaped-mean priors against beta-shape fits to the JAM truth members.

This is a prior-design check, not an inference fit, and it does not set the
priors: those are shared rounded values in ``closure_JAM_truth.config``. Each Q
has its own log normalization; the two exponents are shared across Q. The fit
is linear in log(x f). The figure draws the prior band of the mean function
against the three truth curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from closure_JAM_truth import config as cfg


ROOT = Path(__file__).resolve().parents[1]
Q_KEYS = ("mc", "2", "3")
FIELDS = ("t3", "v3", "v8", "v", "v15", "t8", "t15", "sigma", "g")
X0 = cfg.SHAPED_MEAN_X0
X_MIN = 1e-6
X_MAX = 0.9


def _data(q: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    source = ROOT / "closure_JAM_truth" / "data" / f"truthQ_{q}" / "truth.json"
    truth = json.loads(source.read_text())
    x = np.asarray(truth["x_nodes"], dtype=float)
    y = np.asarray(truth["curves"][field], dtype=float)
    keep = (x >= X_MIN) & (x <= X_MAX) & np.isfinite(y) & (y > 0)
    if keep.sum() < 4:
        raise ValueError(f"too few positive truth nodes for {field} at Q={q}")
    return x[keep], y[keep]


def _fit(field: str, prior: dict) -> dict:
    blocks = []
    values = []
    per_q = {}
    for index, q in enumerate(Q_KEYS):
        x, y = _data(q, field)
        small = np.log(x / X0)
        large = np.log((1 - x) / (1 - X0))
        local = np.column_stack((np.ones(x.size), small, large))
        estimate = np.linalg.lstsq(local, np.log(y), rcond=None)[0]
        per_q[q] = {
            "M_at_x0": float(np.exp(estimate[0])),
            "a": float(estimate[1]),
            "b": float(estimate[2]),
            "log_rms": float(np.sqrt(np.mean((local @ estimate - np.log(y)) ** 2))),
            "nodes_used": int(x.size),
        }
        design = np.zeros((x.size, len(Q_KEYS) + 2))
        design[:, index] = 1
        design[:, -2:] = local[:, 1:]
        blocks.append(design)
        values.append(np.log(y))
    matrix = np.concatenate(blocks)
    response = np.concatenate(values)
    pooled = np.linalg.lstsq(matrix, response, rcond=None)[0]
    return {
        "per_Q": per_q,
        "shared_a": float(pooled[-2]),
        "shared_b": float(pooled[-1]),
        "shared_log_rms": float(np.sqrt(np.mean((matrix @ pooled - response) ** 2))),
        "prior": prior,
        "pull_a": float((pooled[-2] - prior["a"][0]) / prior["a"][1]),
        "pull_b": float((pooled[-1] - prior["b"][0]) / prior["b"][1]),
    }


def _truth(q: str) -> dict:
    return json.loads((ROOT / "closure_JAM_truth" / "data" / f"truthQ_{q}"
                       / "truth.json").read_text())


def _mean_draws(prior: dict, x: np.ndarray, rng) -> np.ndarray:
    draws = {name: rng.normal(*prior[name], size=4000) for name in ("M", "a", "b")}
    return (draws["M"][:, None] * (x / X0) ** draws["a"][:, None]
            * ((1 - x) / (1 - X0)) ** draws["b"][:, None])


def _plot_percentile(report: dict, stem: Path) -> list[Path]:
    """Where each truth curve falls inside the prior-mean distribution, vs x."""
    from closure_JAM_truth.run_closure import hybrid_xscale, save_figure_both
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True,
                             sharey=True)
    for ax, field in zip(axes.flat, FIELDS):
        prior = report["fields"][field]["prior"]
        for q, color in zip(Q_KEYS, ("C0", "C1", "C2")):
            truth = _truth(q)
            x = np.asarray(truth["x_nodes"], dtype=float)
            keep = (x >= 1e-6) & (x < 1)
            x = x[keep]
            curves = _mean_draws(prior, x, rng)
            y = np.asarray(truth["curves"][field], dtype=float)[keep]
            ax.plot(x, 100 * np.mean(curves < y, axis=0), color=color,
                    lw=1.4, label=f"truth Q={q}")
        ax.axhspan(16, 84, color="0.85", zorder=0, label="68% of prior mean")
        ax.axhline(50, color="0.4", lw=0.8)
        hybrid_xscale(ax, x_min=1e-6)
        ax.set_ylim(0, 100)
        ax.set_title(f"{field}: a~N{tuple(prior['a'])}, b~N{tuple(prior['b'])}",
                     fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("truth percentile in prior mean [%]")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(report["title"] + ": truth percentile")
    paths = save_figure_both(fig, stem)
    plt.close(fig)
    return paths


def _plot(report: dict, stem: Path) -> list[Path]:
    from closure_JAM_truth.run_closure import (
        _prepare_matplotlib, hybrid_xscale, save_figure_both)

    _prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = np.logspace(-6, np.log10(0.99), 300)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for ax, field in zip(axes.flat, FIELDS):
        prior = report["fields"][field]["prior"]
        curves = _mean_draws(prior, x, rng)
        lo, mid, hi = np.percentile(curves, [16, 50, 84], axis=0)
        ax.fill_between(x, lo, hi, color="0.8", label="prior mean, 68%")
        ax.plot(x, mid, color="0.35", lw=1.2, label="prior mean, median")
        for q, color in zip(Q_KEYS, ("C0", "C1", "C2")):
            truth = _truth(q)
            ax.plot(truth["x_nodes"], truth["curves"][field], color=color,
                    lw=1.4, label=f"truth Q={q}")
        fit = report["fields"][field]
        ax.set_title(f"{field}: a~N{tuple(prior['a'])}, b~N{tuple(prior['b'])}\n"
                     f"truth fit a={fit['shared_a']:+.2f} ({fit['pull_a']:+.1f}σ), "
                     f"b={fit['shared_b']:.2f} ({fit['pull_b']:+.1f}σ)", fontsize=9)
        hybrid_xscale(ax, x_min=1e-6)
        top = max(hi.max(), *(max(_truth(q)["curves"][field]) for q in Q_KEYS))
        if field in cfg.VANISHING_PDF_FIELDS:
            ax.set_ylim(-0.05, 1.1 * top)
        else:
            # The divergent mean prior spans decades at small x.
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 2 * top)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel(f"x {field}(x)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(report["title"])
    paths = save_figure_both(fig, stem)
    plt.close(fig)
    return paths + _plot_percentile(report, stem.with_name(stem.name + "_percentile"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "campaign" / "shaped_mean_truth_fit.json")
    # Trial priors for plots only; the defaults are what the closure uses.
    for flag, default in (("--M", cfg.SHAPED_MEAN_M_PRIOR),
                          ("--convergent-a", cfg.SHAPED_MEAN_CONVERGENT_A_PRIOR),
                          ("--divergent-a", cfg.SHAPED_MEAN_DIVERGENT_A_PRIOR),
                          ("--quark-b", cfg.SHAPED_MEAN_QUARK_B_PRIOR),
                          ("--gluon-b", cfg.SHAPED_MEAN_GLUON_B_PRIOR)):
        parser.add_argument(flag, type=float, nargs=2, metavar=("CENTER", "WIDTH"),
                            default=list(default))
    args = parser.parse_args()

    def prior(field: str) -> dict:
        return {"M": args.M,
                "a": args.convergent_a if field in cfg.VANISHING_PDF_FIELDS
                else args.divergent_a,
                "b": args.gluon_b if field == "g" else args.quark_b}

    report = {
        "method": "OLS of log(x f) on log(x/0.1) and log((1-x)/0.9); shared a,b with Q-specific intercepts",
        "title": (f"Shaped-mean priors (M~N{tuple(args.M)}, x0={X0}) vs injected JAM truth"),
        "q_keys": list(Q_KEYS), "x0": X0, "x_range": [X_MIN, X_MAX],
        "fields": {field: _fit(field, prior(field)) for field in FIELDS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.out}")
    for path in _plot(report, args.out.with_suffix("")):
        print(f"Saved {path}")
    for field, result in report["fields"].items():
        prior = result["prior"]
        print(f"{field:>5}  a={result['shared_a']:+.3f} vs {prior['a']} ({result['pull_a']:+.2f} sigma)"
              f"  b={result['shared_b']:.3f} vs {prior['b']} ({result['pull_b']:+.2f} sigma)"
              f"  log-RMS={result['shared_log_rms']:.3f}")


if __name__ == "__main__":
    main()
