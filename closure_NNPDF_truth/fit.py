"""Build and run one closure fit ``(truth member Q, test mode)``.

Every field prior is log-RBF with a **per-field constant mean that is tied to its
own amplitude** (``cfg.gp_amplitude``, larger for singlet/gluon/charm), so the
prior is ``a +- a``, **and a fixed length = log(2)**.  With the GP hyperparameters frozen the posterior is
analytic
in the GP field values; PIXEL returns a fixed-parameter ``PosteriorResult`` for
this zero-dimensional inference problem.  The forward operators are identical
to generation (``Q0 = mc``, NLO, ``Nf = 4``, no systematic nuisances).  Success =
the posterior band covers the injected NNPDF truth.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

import pixel
import numpy as np
from pixel import priors, kernels
from pixel.core.model import Contribution, Dataset
from pixel.core.params import Parameter
from pixel.infer import DirectMCMCSamples, PosteriorResult
from pixel.infer import AutocorrelationWarning, autocorrelation_summary
from pixel.infer import run_mcmc as infer_mcmc
from pixel.infer import bilinear_posterior_moments, posterior_moments
from pixel.util.progress import ProgressReporter, Stopwatch, format_duration

from . import config as cfg
from . import datasets as dsets


# -- physics constraints -----------------------------------------------------


def constraint_datasets(fields, truth=None):
    """Near-hard constraints made consistent with the represented closure truth.

    * every field vanishes at ``x = 1`` (a Delta evaluation = 0);
    * the valence fields carry the quark-counting normalizations
      ``int V3 = 1``, ``int V8 = int V = int V15 = 3`` (a Mellin ``n=1`` moment).

    The nominal targets are physical sum rules.  For a closure test, however,
    the injected truth must lie in the support of every near-hard pseudo-datum.
    When ``truth`` is supplied, evaluate the saved curve with the *same fit basis
    and kernel* and use that represented value as the closure target.  The
    nominal physical target is retained in dataset metadata for auditing.
    """
    out = []
    ve = cfg.CONSTRAINT_ENDPOINT_SIGMA ** 2
    vn = cfg.CONSTRAINT_NORM_SIGMA ** 2
    vo = cfg.CONSTRAINT_ORIGIN_SIGMA ** 2
    vm = cfg.CONSTRAINT_MOMENTUM_SIGMA ** 2
    truth_x = None if truth is None else np.asarray(truth["x_nodes"], dtype=float)

    def represented_target(name, field, kernel, nu, nominal):
        if truth_x is None:
            return float(nominal)
        curve = np.interp(
            np.asarray(field.nodes, dtype=float),
            truth_x,
            np.asarray(truth["curves"][name], dtype=float),
        )
        matrix = np.asarray(kernel.matrix(np.asarray([nu]), field.basis), dtype=float)
        return float((matrix @ curve)[0])

    def represented_sum_target(names, kernel_for, nu, nominal):
        """Represented value of a constraint that SUMS several fields.

        The momentum sum rule is a property of ``Sigma + g`` jointly, so its target
        cannot be built field by field.  As with every other constraint here the
        target is the *represented* truth -- evaluated through the same kernels and
        bases the fit will use -- not the nominal 1.0, so the injected truth lies in
        the constraint's support by construction rather than by hope.

        This matters concretely: measured on the stored truth, ``int x (Sigma+g) dx``
        is 1.0005 at Q=2 but was **0.821** for NNPDF at ``mc``, an 18% violation
        caused by LHAPDF extrapolating below its 1.65 QMin.  (``mc`` was dropped from
        TRUTH_Q_CHOICES for that reason on 2026-08-15 and **restored on 2026-08-16**,
        to compare the two prior forms across every scale; the 18% is recorded at
        config.py's TRUTH_Q_CHOICES rather than avoided.)  A nominal 1.0 would
        have injected that 18% as a fresh bias.
        """
        if truth_x is None:
            return float(nominal)
        total = 0.0
        for name in names:
            field = fields[name]
            curve = np.interp(
                np.asarray(field.nodes, dtype=float),
                truth_x,
                np.asarray(truth["curves"][name], dtype=float),
            )
            matrix = np.asarray(
                kernel_for(name).matrix(np.asarray([nu]), field.basis), dtype=float
            )
            total += float((matrix @ curve)[0])
        return total

    for name in cfg.ALL_FIELDS:
        kernel = kernels.Delta()
        target = represented_target(
            name, fields[name], kernel, cfg.ENDPOINT_X, 0.0
        )
        out.append(Dataset(
            f"cons_endpoint_{name}", np.array([cfg.ENDPOINT_X]), np.array([target]),
            np.array([[ve]]), [Contribution(name, kernel)],
            extras={"nominal_target": 0.0, "closure_target": target},
        ))
    for name, nominal in cfg.VALENCE_NORMS.items():
        # Fields are momentum densities q=x*f, so the valence number int f dx =
        # int (q/x) dx is a Mellin n=1 moment with an alpha=-1 integrand weight.
        kernel = kernels.Mellin(
            alpha=-1.0, low_x_extension=cfg.low_x_completion(name)
        )
        target = represented_target(name, fields[name], kernel, 1.0, nominal)
        out.append(Dataset(
            f"cons_norm_{name}", np.array([1.0]), np.array([float(target)]),
            np.array([[vn]]),
            [Contribution(name, kernel)],
            extras={"nominal_target": float(nominal), "closure_target": target},
        ))
    # x*f(x) -> 0 at the origin, for the fields where theory predicts it
    # (cfg.vanishes_at_origin: everything but the singlet and gluon momentum
    # densities, which may tend to a nonzero constant).
    #
    # Only imposed when the grid actually reaches x = 0.  On a grid that starts
    # at x_min the field is simply not defined below it: kernels.Delta at x = 0
    # returns an all-**zero** row, which is a silent no-constraint rather than an
    # error, and the [0, x_min) behaviour is already carried by the low-x
    # completion (a straight line to zero for exactly these fields).
    if float(np.asarray(next(iter(fields.values())).nodes).min()) == 0.0:
        for name in cfg.ALL_FIELDS:
            if not cfg.vanishes_at_origin(name):
                continue
            kernel = kernels.Delta()
            target = represented_target(name, fields[name], kernel, 0.0, 0.0)
            out.append(Dataset(
                f"cons_origin_{name}", np.array([0.0]), np.array([target]),
                np.array([[vo]]), [Contribution(name, kernel)],
                extras={"nominal_target": 0.0, "closure_target": target},
            ))
    # <x^1>: the momentum sum rule  int x (Sigma + g) dx = 1.  Fields are momentum
    # densities q = x f, so <x^1> = int q dx is a Mellin n=1 moment at alpha = 0 --
    # the alpha weight multiplies the integrand (kernels/common.py:58).  One Dataset
    # summing two Contributions expresses Sigma + g directly (model.py:346).
    #
    # Added 2026-08-15.  Before it, ``sigma`` and ``g`` were the only fields carrying
    # no integral constraint at all, and ``g`` showed a deterministic -0.16
    # signed-pull bias (3.5 sigma over 24 noise replicas, 19/24 negative).
    def _momentum_kernel(name):
        return kernels.Mellin(alpha=0.0, low_x_extension=cfg.low_x_completion(name))

    momentum_fields = ("sigma", "g")
    momentum_target = represented_sum_target(
        momentum_fields, _momentum_kernel, 1.0, 1.0
    )
    out.append(Dataset(
        "cons_momentum", np.array([1.0]), np.array([momentum_target]),
        np.array([[vm]]),
        [Contribution(name, _momentum_kernel(name)) for name in momentum_fields],
        extras={"nominal_target": 1.0, "closure_target": momentum_target},
    ))
    return out


# -- build -------------------------------------------------------------------


def _gp_prior(field: str):
    # One amplitude for both: the mean is pinned to the covariance sigma by
    # analysis.tie in _build, so this prior is a +- a.  Passing the same number
    # twice here only sets the starting value -- the tie is what keeps them
    # equal, and it is what will keep them equal if the amplitude is ever thawed.
    value = cfg.gp_amplitude(field)
    # The amplitude carries the floor and the hyper-prior; both are inert while
    # cfg.GP_AMPLITUDE_FREE is False (bounds_array skips frozen leaves, and
    # gp_amplitude_prior returns None so the evidence is unshifted).
    amplitude = Parameter(
        value,
        bounds=(cfg.GP_AMPLITUDE_FLOOR, None),
        prior=cfg.gp_amplitude_prior(field),
        frozen=not cfg.GP_AMPLITUDE_FREE,
        tags=("gp_kernel",),
    )
    return priors.GaussianProcess(
        mean=priors.Const(N=value),
        # Read the SAME jitter knob the envelope prior reads.  Until 2026-08-16
        # this call passed no jitter at all and silently took LogRBF's class
        # default of 1e-10, so `cfg.GP_JITTER` did not reach this prior: a
        # 36-cell scan that varied it produced bit-identical results because
        # nothing was changing.  A knob that is not wired reads exactly like a
        # knob that does not matter.
        cov=priors.LogRBF(sigma=amplitude, length=cfg.GP_LENGTH_LOG,
                          jitter=getattr(cfg, "GP_JITTER", 1.0e-10)),
    )


def _envelope_prior(field: str):
    """Zero mean under a ``x^alpha (1-x)^beta`` envelope; see cfg.GP_ENVELOPE.

    Nothing is tied here and nothing can be: the mean is identically zero, so it
    carries no ``N`` leaf.  That is the point of this form rather than an
    oversight -- it is what lets the width stay wide where the data is absent
    without the mean asserting a matching non-zero value there.
    """
    env = cfg.gp_envelope(field)
    return priors.GaussianProcess(
        mean=priors.Zero(),
        cov=priors.BetaTaperedLogRBF(
            sigma=env["sigma"], length=cfg.GP_LENGTH_LOG,
            alpha=env["alpha"], beta=env["beta"],
            # NOT zero -- see cfg.GP_ENVELOPE_JITTER, which ships 1e-2.
            #
            # The argument this comment used to make ("the rcond cut already
            # regularises, so a jitter here is a second knob doing the same job")
            # is wrong, and the correction is recorded in
            # guides/numerical_regularization.md: rcond and the jitter act on
            # DIFFERENT matrices.  rcond cuts `W = C + B K B^T`; the jitter enters
            # `K`, and `H = K - K B^T W^-1 B K` is built from `K` and inverted with
            # a bare `jnp.linalg.inv` -- no cutoff on that path at all, so
            # truncating `W` cannot make `H` definite.  Measured on `both` with
            # this prior: 9/9 fits fail across rcond 1e-10..1e-6 at jitter <= 1e-10.
            #
            # The envelope needs far more than the constant prior (1e-2 vs 1e-10)
            # because the jitter is a RELATIVE floor, lambda/sigma^2, and enforcing
            # alpha >= 0 pushed sigma to 139 for t8.  That is arithmetic, not a
            # second pathology.
            jitter=getattr(cfg, "GP_ENVELOPE_JITTER", 1.0e-2),
        ),
    )


def gp_prior(field: str):
    """Dispatch on :data:`cfg.PRIOR_FORM`."""
    if cfg.PRIOR_FORM == "beta_envelope":
        return _envelope_prior(field)
    if cfg.PRIOR_FORM != "const_logrbf":
        raise ValueError(f"unknown PRIOR_FORM {cfg.PRIOR_FORM!r}")
    return _gp_prior(field)


def _nuisance_prior():
    """Zero-mean log-RBF GP for a lattice-systematic nuisance field (frozen)."""
    return priors.GaussianProcess(
        mean=priors.Zero(),
        cov=priors.LogRBF(sigma=cfg.NUISANCE_GP_SIGMA, length=cfg.GP_LENGTH_LOG),
    )


def nuisance_field_names() -> list:
    """Names of pseudo-ITD curves and independent Mellin scalar nuisances."""
    itd = [cfg.nuisance_field_name(f, k) for f, k in cfg.nuisance_specs()]
    moments = [
        cfg.moment_nuisance_field_name(f, k, order)
        for f, k, order in cfg.moment_nuisance_specs()
    ]
    return itd + moments


def build_analysis(q_key: str, mode: str, *, use_kernel_cache: bool = True,
                   t0=None, bilinear_layout: str = "block-sparse"):
    """Construct the Analysis for ``(q_key, mode)`` and freeze all GP cov params.

    When lattice data are present, pseudo-ITD systematics use shared x-space
    nuisance curves while each Mellin power uses an independent one-point scalar
    nuisance.  Their hyperparameters are frozen too, so the inference stays a
    zero-free-parameter closed-form posterior.

    ``t0`` maps a DIS dataset name to the current theory prediction for its rows;
    multiplicative uncertainties are then referenced to it rather than to the
    data (the NNPDF t0 prescription).  :func:`run_fit` iterates it.

    Returns ``(analysis, fields)``.
    """
    if mode not in cfg.TEST_MODES:
        raise ValueError(f"unknown test mode {mode!r}")
    include_lattice = mode in ("lattice", "both")
    include_exp = mode in ("dis", "exp", "both")
    include_dy = mode in ("dy", "exp", "both")     # Drell-Yan
    analysis = pixel.Analysis(f"closure_{cfg.truth_label(q_key)}_{mode}")
    fields = {}
    for name in cfg.ALL_FIELDS:
        fields[name] = analysis.field(
            name, grid=cfg.make_grid(), element_type=cfg.ELEMENT_TYPE,
            prior=gp_prior(name),
        )
    nuisance_names = []
    if include_lattice:
        for field, key in cfg.nuisance_specs():
            nname = cfg.nuisance_field_name(field, key)
            fields[nname] = analysis.field(
                nname, grid=cfg.make_grid(), element_type=cfg.ELEMENT_TYPE,
                prior=_nuisance_prior(),
            )
            nuisance_names.append(nname)
        for field, key, order in cfg.moment_nuisance_specs():
            nname = cfg.moment_nuisance_field_name(field, key, order)
            fields[nname] = analysis.field(
                nname,
                grid=cfg.make_moment_nuisance_grid(order),
                element_type="piecewise",
                prior=_nuisance_prior(),
            )
            nuisance_names.append(nname)

    for ds in dsets.build_datasets(
        q_key, fields,
        include_lattice=include_lattice,
        include_exp=include_exp,
        include_dy=include_dy,
        use_kernel_cache=use_kernel_cache,
        with_systematics=include_lattice,
        t0=t0,
    ):
        analysis.add(ds)

    # Physics constraints (endpoint + valence counting) apply in every mode.
    for ds in constraint_datasets(fields, load_truth(q_key)):
        analysis.add(ds)

    analysis.compile(rcond=cfg.RCOND, bilinear_layout=bilinear_layout)
    # Freeze the physical prior means + cov params and the nuisance cov params, so
    # the prior is fully specified and there are no free params.
    frozen = []
    for name in cfg.ALL_FIELDS:
        params = analysis.prior_params(fields[name])
        # Pin the constant prior mean to the covariance amplitude, so the prior
        # is a +- a by construction rather than by two constants agreeing.  The
        # target is frozen out of the free vector by the tie itself, which is
        # why mean.N is not in `frozen` below.
        #
        # The envelope form has a Zero mean, which carries no `N` leaf at all --
        # there is nothing to tie, and asking for one raises rather than silently
        # doing nothing.  Its exponents are frozen by the covariance itself.
        if cfg.PRIOR_FORM != "beta_envelope":
            analysis.tie(params.mean.N, to=params.cov.sigma)
        for attr in cfg.FROZEN_COV_PARAMS:
            # The amplitude is the one cov parameter that may be floated; it
            # carries its own frozen flag from _gp_prior, so re-freezing it here
            # would silently override cfg.GP_AMPLITUDE_FREE.
            if attr == "sigma" and cfg.GP_AMPLITUDE_FREE:
                continue
            frozen.append(getattr(params.cov, attr))
    for nname in nuisance_names:
        params = analysis.prior_params(fields[nname])
        for attr in cfg.FROZEN_COV_PARAMS:
            frozen.append(getattr(params.cov, attr))
    analysis.freeze(*frozen)
    return analysis, fields


# -- run: NUTS over ordinary and Hubbard-Stratonovich parameters --------------


def _closure_sampler(model) -> str:
    """Choose NUTS unless mixed ordinary/HS coordinates require joint HMC."""
    if bool(getattr(model, "has_nonlinear", False)):
        return "direct_nuts"
    if model.has_bilinear and model.contour_partition().n_ordinary:
        return "hmc"
    return "nuts"


def run_fit(q_key: str, mode: str, *, use_kernel_cache: bool = True,
            n_samples: int = cfg.MCMC_SAMPLES, seed: int = cfg.MCMC_SEED) -> dict:
    """Run inference and return posterior moments/coverage ingredients.

    Each stage is timed and the breakdown printed, because a full-scale fit runs
    long enough that "which stage is this hour going into" is the first question
    asked of it.  ``build_analysis`` is where the kernel matrices are actually
    assembled -- the later ``analysis.compile()`` reuses that cached model -- so
    its stage number is the kernel-assembly cost.
    """
    watch = Stopwatch()
    reporter = ProgressReporter("closure")
    reporter.emit(
        f"fit start q={q_key} mode={mode} n_samples={int(n_samples)}"
    )
    reporter.emit("building analysis and kernels")
    analysis, fields = build_analysis(q_key, mode, use_kernel_cache=use_kernel_cache)
    model = analysis.compile()
    reporter.emit(
        f"analysis ready n_datasets={len(model.datasets)} "
        f"n_free={int(model.free_vector().size)} "
        f"has_bilinear={bool(model.has_bilinear)}",
        lap=True,
    )
    # NNPDF t0: multiplicative uncertainties must be referenced to the *theory*,
    # not the data, or the fit is biased low (D'Agostini).  The reference is not
    # known before fitting, so it is iterated -- fit, take the predictions, refold
    # the covariance, refit -- until the t0 vectors stop moving.
    reporter.emit("checking DIS t0 covariance references")
    t0, t0_iterations = _iterate_t0(q_key, mode, model,
                                    use_kernel_cache=use_kernel_cache)
    reporter.emit(
        f"DIS t0 ready n_tables={len(t0)} iterations={int(t0_iterations)}",
        lap=True,
    )
    if t0:
        reporter.emit("rebuilding analysis with converged DIS t0 references")
        analysis, fields = build_analysis(q_key, mode,
                                          use_kernel_cache=use_kernel_cache, t0=t0)
        model = analysis.compile()
        reporter.emit("t0 analysis rebuild ready", lap=True)
    t_build = watch.lap()
    sampler = _closure_sampler(model)
    if sampler == "direct_nuts":
        sampler_options = {
            "n_warmup": cfg.DIRECT_NUTS_WARMUP,
            "step_size": cfg.DIRECT_NUTS_INITIAL_STEP_SIZE,
            "max_tree_depth": cfg.DIRECT_NUTS_MAX_TREE_DEPTH,
            "adapt_mass": cfg.DIRECT_NUTS_ADAPT_MASS,
            "prior_rcond": cfg.DIRECT_PRIOR_RCOND,
        }
        warmup = cfg.DIRECT_NUTS_WARMUP
    elif sampler == "hmc":
        sampler_options = {
            "n_warmup": cfg.HMC_WARMUP,
            "step_size": cfg.HMC_STEP_SIZE,
            "n_leapfrog": cfg.HMC_N_LEAPFROG,
            "integrator": cfg.HMC_INTEGRATOR,
            "use_hessian_mass": True,
            "saddle_tol": cfg.CONTOUR_SADDLE_TOL,
        }
        warmup = cfg.HMC_WARMUP
    else:
        sampler_options = {
            "n_warmup": cfg.NUTS_WARMUP,
            "step_size": cfg.NUTS_INITIAL_STEP_SIZE,
            "integrator": cfg.NUTS_INTEGRATOR,
            "max_tree_depth": cfg.NUTS_MAX_TREE_DEPTH,
            "adapt_mass": cfg.NUTS_ADAPT_MASS,
            "use_hessian_mass": True,
        }
        warmup = cfg.NUTS_WARMUP
        if model.has_bilinear:
            sampler_options["saddle_tol"] = cfg.CONTOUR_SADDLE_TOL
            cache_stem = f"{cfg.truth_label(q_key)}_{mode}"
            sampler_options["reference_cache_path"] = (
                cfg.INFERENCE_CACHE_ROOT / f"{cache_stem}_reference.npz"
            )
            sampler_options["hessian_cache_path"] = (
                cfg.INFERENCE_CACHE_ROOT / f"{cache_stem}_hessian_mass.npz"
            )
    reporter.emit(
        f"inference start sampler={sampler} n_warmup={int(warmup)}"
    )
    if sampler == "direct_nuts":
        samples = pixel.run_direct_mcmc(
            model, sampler="nuts", n_samples=n_samples, seed=seed,
            **sampler_options,
        )
    else:
        samples = infer_mcmc(
            model, sampler=sampler,
            auxiliary="sample" if model.has_bilinear else "auto",
            n_samples=n_samples, run_map=False, seed=seed, **sampler_options,
        )
    reporter.emit(f"inference done sampler={sampler}", lap=True)
    t_mcmc = watch.lap()
    reporter.emit("posterior marginalization start")
    full_moments = posterior_moments(model, samples)
    field_by_name = {field.name: field for field in model.fields}
    marginal = {}
    for layout in full_moments.q_layout:
        name = str(layout["name"])
        sl = slice(int(layout["start"]), int(layout["stop"]))
        covariance = full_moments.covariance[sl, sl]
        marginal[name] = (
            np.asarray(field_by_name[name].nodes, dtype=float),
            np.asarray(full_moments.mean[sl], dtype=float),
            np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        )
    reporter.emit("posterior marginalization done", lap=True)
    t_marginal = watch.lap()
    reporter.emit("posterior predictive chi2 start")
    vec_bar = _weighted_mean_vec(samples)
    chi2_components = {"linear": None, "bilinear": None}
    try:
        if isinstance(samples, DirectMCMCSamples):
            chi2 = 0.0
            chi2_components = {"linear": 0.0, "bilinear": 0.0, "nonlinear": 0.0}
            for layout in full_moments.prediction_layout:
                dataset = model.datasets[int(layout["dataset_index"])]
                section = slice(int(layout["start"]), int(layout["stop"]))
                residual = np.asarray(full_moments.predictive_mean[section]) - np.asarray(dataset.mean)
                value = float(residual @ np.linalg.pinv(np.asarray(dataset.cov), rcond=cfg.RCOND) @ residual)
                kind = str(layout["prediction_kind"])
                bucket = kind if kind in ("linear", "bilinear") else "nonlinear"
                chi2_components[bucket] += value
                chi2 += value
        elif model.has_bilinear and hasattr(samples, "mean"):
            predictive = model.posterior_predictive_from_moments(
                samples.mean, samples.covariance, vec_bar
            )
            chi2 = float(predictive.chi2)
            chi2_components = {
                "linear": float(predictive.linear_chi2),
                "bilinear": float(predictive.bilinear_chi2),
            }
        else:
            _, _, chi2 = model.posterior_predictive(vec_bar)
            chi2 = float(chi2)
            chi2_components = {"linear": chi2, "bilinear": 0.0}
    except Exception as exc:
        reporter.emit(f"posterior predictive chi2 failed: {exc}")
        chi2 = float("nan")
    reporter.emit(f"posterior predictive chi2 done value={chi2:.6g}", lap=True)
    t_chi2 = watch.lap()
    fit_total = watch.elapsed
    print(f"   [time] build+kernels {format_duration(t_build)}"
          f"  mcmc {format_duration(t_mcmc)}"
          f"  marginal {format_duration(t_marginal)}"
          f"  chi2 {format_duration(t_chi2)}"
          f"  fit total {format_duration(fit_total)}", flush=True)
    return {
        "analysis": analysis, "fields": fields, "model": model,
        "samples": samples, "marginal": marginal, "posterior_moments": full_moments,
        "vec_bar": vec_bar,
        "fitted_normalization_stats": fitted_normalization_posterior(model, samples),
        "chi2": chi2, "chi2_components": chi2_components,
        "n_data": int(sum(d.n_data for d in model.datasets)),
        "n_free": int(model.free_vector().size),
        "sampler": sampler,
        "t0_iterations": t0_iterations,
        "ess": _effective_sample_size(samples),
        "autocorr": _autocorrelation_ess(samples),
        "direct_diagnostics": (
            samples.diagnostics() if isinstance(samples, DirectMCMCSamples) else None
        ),
        "timing_seconds": {
            "build_and_kernels": t_build,
            "inference": t_mcmc,
            "marginalization": t_marginal,
            "predictive_chi2": t_chi2,
            "fit_total": fit_total,
        },
    }


def _dis_predictions(model) -> dict:
    """MAP-point theory prediction per *linear experimental* dataset.

    Only DIS tables carry multiplicative uncertainties here, and only those need
    a t0 reference; lattice and constraint datasets are skipped.
    """
    layouts = []
    for layout in model._layout.datasets:
        d = model.datasets[layout.dataset_index]
        if d.metadata.extras.get("correlated_systematics"):
            layouts.append(layout)
    # In particular, a DY-only fit has no DIS t0 references.  Do not send its
    # bilinear Hubbard-Stratonovich model through an expensive, untrustworthy MAP
    # solve merely to discover that the result would be unused.
    if not layouts:
        return {}
    vec = np.asarray(pixel.map(
        model, method=cfg.MAP_METHOD, options=cfg.MAP_OPTIONS).x)
    pred, _, _ = model.posterior_predictive(vec)
    pred = np.asarray(pred)
    out = {}
    for layout in layouts:
        d = model.datasets[layout.dataset_index]
        out[d.name] = pred[layout.data_slice].copy()
    return out


def _iterate_t0(q_key, mode, model, *, use_kernel_cache=True):
    """Iterate the NNPDF t0 reference to convergence.

    Returns ``(t0, n_iterations)``.  ``t0`` maps a DIS dataset name to the theory
    prediction used as the reference for its multiplicative uncertainties; the
    loop stops once successive predictions agree to
    :data:`config.T0_TOLERANCE` relatively, or after
    :data:`config.T0_MAX_ITERATIONS`.  An empty dict means no dataset in this
    mode carries multiplicative uncertainties, so no iteration is needed.
    """
    max_iter = getattr(cfg, "T0_MAX_ITERATIONS", 0)
    # t0 is a DIS covariance reference.  In mixed modes, derive it from the
    # linear DIS sub-fit so the bilinear HS sector is never sent through MAP.
    t0_mode = "dis" if mode in ("exp", "both") else mode
    if t0_mode != mode:
        analysis, _ = build_analysis(
            q_key, t0_mode, use_kernel_cache=use_kernel_cache
        )
        model = analysis.compile()
    t0 = _dis_predictions(model)
    if not t0 or max_iter <= 0:
        return ({} if not t0 else t0), 0
    for iteration in range(1, max_iter + 1):
        analysis, _ = build_analysis(q_key, t0_mode,
                                     use_kernel_cache=use_kernel_cache, t0=t0)
        nxt = _dis_predictions(analysis.compile())
        shift = max(
            float(np.max(np.abs(nxt[k] - t0[k]) / (np.abs(t0[k]) + 1e-300)))
            for k in t0 if k in nxt
        )
        t0 = nxt
        if shift <= cfg.T0_TOLERANCE:
            return t0, iteration
    return t0, max_iter


def _weights(samples) -> np.ndarray:
    if isinstance(samples, PosteriorResult):
        return np.ones(1, dtype=float)
    n = np.asarray(samples.samples).shape[0]
    for name in ("residual_weights", "phase_weights", "weights"):
        raw = getattr(samples, name, None)
        if raw is not None:
            w = np.asarray(raw).reshape(-1)
            break
    else:
        w = np.ones(n, dtype=float)
    if w.size != n:
        raise ValueError("sample weights and parameter samples have different lengths")
    total = np.sum(w)
    if not np.isfinite(total) or abs(total) <= np.finfo(float).eps * max(n, 1):
        raise RuntimeError("closure sample weights have a non-finite or zero sum")
    return w / total


def _weighted_mean_vec(samples) -> np.ndarray:
    if isinstance(samples, PosteriorResult):
        return np.asarray(samples.x, dtype=float)
    if isinstance(samples, DirectMCMCSamples):
        return np.asarray(
            np.average(samples.theta_samples, axis=0, weights=samples.weights),
            dtype=float,
        )
    S = np.asarray(samples.samples)
    # Contour samples and their residual weights are complex; the physical
    # parameter expectation is the real part of their phase-weighted ratio.
    return np.asarray(
        np.real((_weights(samples)[:, None] * S).sum(axis=0)), dtype=float
    )


def _effective_sample_size(samples) -> float:
    """Return weight/phase ESS for summaries (not an autocorrelation ESS)."""
    if isinstance(samples, PosteriorResult):
        return 1.0
    if hasattr(samples, "ess_frac"):
        return float(samples.n_samples) * float(samples.ess_frac)
    effective = getattr(samples, "effective_sample_size", None)
    if effective is not None:
        return float(effective() if callable(effective) else effective)
    weights = _weights(samples)
    return float(1.0 / np.sum(np.abs(weights) ** 2))


def _markov_chain(samples):
    """Return the object holding the real Markov history, or ``None``.

    Three cases must be told apart, because applying the Gamma method to the
    wrong one is silently meaningless rather than loudly broken:

    * Contour artifacts expose the **complex** flowed ``f`` as ``.samples``; the
      real trajectory that was actually integrated lives under ``.chain``.
    * Importance samples are independent draws from a proposal, not a Markov
      chain -- there is no autocorrelation to measure and the weight/phase ESS
      above is already the right number.
    * A fixed-parameter ``PosteriorResult`` has no chain at all.
    """
    if isinstance(samples, PosteriorResult):
        return None
    chain = getattr(samples, "chain", None)
    candidate = samples if chain is None else chain
    history = np.asarray(getattr(candidate, "samples", np.empty(0)))
    if history.ndim != 2 or history.shape[0] < 2 or np.iscomplexobj(history):
        return None
    weights = getattr(candidate, "weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.size and not np.allclose(weights, weights.flat[0]):
            return None
    return candidate


def _autocorrelation_ess(samples) -> dict:
    """Wolff Gamma-method mixing diagnostics (hep-lat/0306017) for one fit.

    This is the *sampling-statistics* ESS: how many effectively independent
    draws the chain produced, taken over its slowest-mixing direction because a
    chain is only as good as that.  It is a different failure mode from the
    weight/phase ESS above and neither substitutes for the other.

    ``tau_int`` follows Wolff, so ``1/2`` means uncorrelated draws and the ESS is
    ``N / (2 tau_int)``; a well-tuned chain can be anti-correlated and report an
    ESS above ``N``.  ``reliable`` is false when the chain is shorter than 50
    autocorrelation times, in which case the number is itself noise.
    """
    chain = _markov_chain(samples)
    if chain is None:
        return {"ess": None, "tau_int_max": None, "slowest": None, "reliable": None}
    with warnings.catch_warnings():
        # The report carries `reliable` explicitly; warning per parameter here
        # would bury the fit log.
        warnings.simplefilter("ignore", AutocorrelationWarning)
        analysis = autocorrelation_summary(chain)
    return {
        "ess": float(analysis.min_effective_sample_size),
        "tau_int_max": float(analysis.max_tau_int),
        "slowest": analysis.slowest_param,
        "reliable": bool(analysis.reliable),
    }


def _field_segments(model):
    """Yield ``(name, x_nodes, slice)`` over the concatenated GP grid, in order."""
    offset = 0
    for f in model.fields:
        n = int(f.n_grid)
        yield f.name, np.asarray(f.nodes, dtype=float).reshape(-1), slice(offset, offset + n)
        offset += n


def bilinear_marginal_posterior(model, samples) -> dict:
    """Per-field posterior from the **complex-phase H-S** aggregation (Drell-Yan).

    A bilinear DY dataset makes the physical posterior a full-line complex-phase
    integral over the auxiliary ``f`` -- a plain weighted average of the
    fixed-``f`` linear posteriors (as the linear branch does) is *wrong*.  Use the
    canonical phase-aware accumulators ``bilinear_posterior_mean`` /
    ``bilinear_posterior_covariance`` over generic samples, then split the
    concatenated GP-grid result per field.
    """
    if hasattr(samples, "ess_frac"):
        # Contour-HMC artifacts already contain the correctly residual-weighted
        # physical moments, including each flow Jacobian.
        qhat = np.asarray(samples.mean, dtype=float).reshape(-1)
        cov = np.asarray(samples.covariance, dtype=float)
    else:
        # One sweep for both moments: the covariance follows from the same
        # accumulated raw second moment as the mean (see
        # bilinear_posterior_moments), so a second pass over every sample is
        # pure duplicated work.
        qhat, cov = bilinear_posterior_moments(model, samples)
        qhat = np.asarray(qhat, dtype=float).reshape(-1)
        cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    out = {}
    for name, x, sl in _field_segments(model):
        out[name] = (x, qhat[sl], std[sl])
    return out


def marginal_posterior(model, samples) -> dict:
    """Hyperparameter-marginalized posterior per field: ``{name: (x, mean, std)}``.

    Mixture over weighted samples: ``mean = E[mu_s]`` and the reported variance
    is the total ``E[var_s] + Var[mu_s]`` (aleatoric + epistemic).  When the model
    carries bilinear (Drell-Yan / Hubbard-Stratonovich) data the field posterior is
    the complex-phase ``f``-integral, so dispatch to
    :func:`bilinear_marginal_posterior`.
    """
    if getattr(model, "has_bilinear", False) and not isinstance(samples, PosteriorResult):
        return bilinear_marginal_posterior(model, samples)

    if isinstance(samples, PosteriorResult):
        out = {}
        for name, p in samples.posterior().items():
            m = np.asarray(p["mean"], dtype=float).reshape(-1)
            v = np.clip(np.diag(np.asarray(p["cov"], dtype=float)), 0.0, None)
            x = np.asarray(p["x"], dtype=float).reshape(-1)
            out[name] = (x, m, np.sqrt(v))
        return out

    w = _weights(samples)
    S = np.asarray(samples.samples, dtype=float)
    # One dense posterior solve per sample: slow enough to want breadcrumbs, but
    # gated like the samplers so it stays silent unless PIXEL_PROGRESS is set.
    reporter = ProgressReporter("posterior", total=len(S))
    reporter.emit(f"marginal posterior start n_samples={len(S)}")
    from pixel.core.evidence import posterior

    acc_m, acc_m2, xref = {}, {}, {}
    acc_pred = None
    acc_pred2 = None
    for i, vec in enumerate(S):
        blocks = model._blocks(model._rebuild(vec))
        qbar, cov = posterior(blocks, model.rcond)
        qbar = np.asarray(qbar, dtype=float).reshape(-1)
        cov = np.asarray(cov, dtype=float)
        prediction = np.asarray(blocks.param_pred, dtype=float) + np.asarray(
            blocks.B, dtype=float
        ) @ qbar
        prediction_var = np.einsum(
            "ij,jk,ik->i", blocks.B, cov, blocks.B, optimize=True
        )
        if acc_pred is None:
            acc_pred = np.zeros_like(prediction)
            acc_pred2 = np.zeros_like(prediction)
        acc_pred += w[i] * prediction
        acc_pred2 += w[i] * (
            np.clip(prediction_var, 0.0, None) + prediction * prediction
        )
        reporter.report(i + 1, "marginal posterior samples")
        for name, x, sl in _field_segments(model):
            m = qbar[sl]
            v = np.clip(np.diag(cov[sl, sl]), 0.0, None)
            if name not in acc_m:
                acc_m[name] = np.zeros_like(m)
                acc_m2[name] = np.zeros_like(m)
                xref[name] = x
            acc_m[name] += w[i] * m
            acc_m2[name] += w[i] * (v + m * m)
    out = {}
    for name in acc_m:
        mean = acc_m[name]
        var = np.clip(acc_m2[name] - mean * mean, 0.0, None)
        out[name] = (xref[name], mean, np.sqrt(var))
    if acc_pred is not None:
        pred_var = np.clip(acc_pred2 - acc_pred * acc_pred, 0.0, None)
        samples._pixel_linear_predictive_moments = (
            acc_pred, np.sqrt(pred_var)
        )
    reporter.emit(f"marginal posterior done n_samples={len(S)}", lap=True)
    return out


# -- reporting ---------------------------------------------------------------


def _pull_stats(x, mean, std, tcurve, *, x_lo, x_hi):
    """Pointwise standardized residuals; fit-grid nodes are correlated."""
    bulk = (x >= x_lo) & (x <= x_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_pulls = ((mean - tcurve) / std)[bulk]
    signed_pulls = signed_pulls[np.isfinite(signed_pulls)]
    abs_pulls = np.abs(signed_pulls)
    n_points = int(abs_pulls.size)
    n_le_1 = int(np.sum(abs_pulls <= 1.0))
    n_le_2 = int(np.sum(abs_pulls <= 2.0))
    pull_chi2 = float(np.sum(signed_pulls * signed_pulls)) if n_points else None
    return {
        "n_points": n_points,
        "pull_chi2": pull_chi2,
        "pull_chi2_per_point": pull_chi2 / n_points if n_points else None,
        "mean_squared_standardized_residual":
            pull_chi2 / n_points if n_points else None,
        "points_are_correlated": True,
        "n_pull_le_1": n_le_1,
        "n_pull_le_2": n_le_2,
        "frac_pull_le_1": n_le_1 / n_points if n_points else None,
        "frac_pull_le_2": n_le_2 / n_points if n_points else None,
        "mean_abs_pull": float(np.mean(abs_pulls)) if n_points else None,
    }


def coverage_report(marginal, truth, *, x_lo=0.01, x_hi=0.9):
    """Pointwise standardized residuals; fit-grid nodes are correlated."""
    xn = np.asarray(truth["x_nodes"])
    report = {}
    for name, (x, mean, std) in marginal.items():
        tcurve = np.interp(x, xn, np.asarray(truth["curves"][name]))
        bulk = (x >= x_lo) & (x <= x_hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            signed_pulls = ((mean - tcurve) / std)[bulk]
        finite = np.isfinite(signed_pulls)
        # Mask x identically, so `signed_pull_bins` can align each pull with the
        # node it came from.  Dropping only the pulls would silently offset them.
        x_bulk_finite = np.asarray(x)[bulk][finite]
        signed_pulls = signed_pulls[finite]
        abs_pulls = np.abs(signed_pulls)
        n_points = int(abs_pulls.size)
        n_le_1 = int(np.sum(abs_pulls <= 1.0))
        n_le_2 = int(np.sum(abs_pulls <= 2.0))
        pull_chi2 = float(np.sum(signed_pulls * signed_pulls)) if n_points else None
        report[name] = {
            "n_points": n_points,
            "pull_chi2": pull_chi2,
            "pull_chi2_per_point": pull_chi2 / n_points if n_points else None,
            "mean_squared_standardized_residual":
                pull_chi2 / n_points if n_points else None,
            "points_are_correlated": True,
            "n_pull_le_1": n_le_1,
            "n_pull_le_2": n_le_2,
            "frac_pull_le_1": n_le_1 / n_points if n_points else None,
            "frac_pull_le_2": n_le_2 / n_points if n_points else None,
            "mean_abs_pull": float(np.mean(abs_pulls)) if n_points else None,
            # The DIRECTION of a miss.  Everything above is unsigned, so a
            # systematic over-estimate and a symmetric band that is merely too
            # narrow produce the identical `pull_chi2_per_point` -- the two were
            # indistinguishable from stored artifacts until 2026-08-16, and
            # telling them apart needed a bespoke probe outside the suite.
            #
            # It is the statistic that settled the `t3` question: over 24 noise
            # replicas `t3` averaged +0.04 +- 0.09 (consistent with zero, a draw
            # artifact) while `v15` sat at -0.545 with a scatter of only 0.060
            # across the same replicas -- a deterministic offset.  Same
            # `mean_abs_pull` story for both; opposite conclusions.
            "mean_signed_pull": float(np.mean(signed_pulls)) if n_points else None,
            # Per-x-bin signed means, so a sign FLIP in x is visible too: a field
            # can average to zero while being high at small x and low at large x,
            # which is what the GP correlation length does to a local pull.
            "signed_pull_bins": _signed_pull_bins(x_bulk_finite, signed_pulls),
        }
    return report


#: Bin edges for `signed_pull_bins`, inside the coverage bulk.
SIGNED_PULL_BIN_EDGES = (0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.9)


def _signed_pull_bins(x_bulk, signed_pulls) -> dict:
    """Mean signed pull per x bin; ``None`` where a bin holds no node."""
    x_bulk = np.asarray(x_bulk, dtype=float)
    signed = np.asarray(signed_pulls, dtype=float)
    if x_bulk.size != signed.size:          # the finite-mask dropped nodes
        return {}
    out = {}
    edges = SIGNED_PULL_BIN_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x_bulk >= lo) & (x_bulk < hi)
        key = f"{lo:g}-{hi:g}"
        out[key] = float(np.mean(signed[m])) if m.any() else None
    return out


def nuisance_coverage_report(marginal, truth, *, x_lo=0.05, x_hi=0.9):
    """Recovery of pseudo-ITD nuisance curves and Mellin scalar nuisances.

    Compares each recovered ``nuis_<field>_<key>`` posterior to the injected
    systematic curve saved in ``truth["systematic_curves"]``.  Empty when the fit
    had no lattice data (no nuisance fields) or the truth predates systematics.
    """
    sys_truth = truth.get("systematic_curves") or {}
    xn = np.asarray(truth["x_nodes"])
    report = {}
    for field, keys in sys_truth.items():
        for key, curve in keys.items():
            nname = cfg.nuisance_field_name(field, key)
            if nname not in marginal:
                continue
            x, mean, std = marginal[nname]
            tcurve = np.interp(x, xn, np.asarray(curve))
            report[nname] = _pull_stats(x, mean, std, tcurve, x_lo=x_lo, x_hi=x_hi)
    for nname, value in (truth.get("moment_systematic_values") or {}).items():
        if nname not in marginal:
            continue
        x, mean, std = marginal[nname]
        target = np.full(np.asarray(mean).shape, float(value), dtype=float)
        report[nname] = _pull_stats(
            x, mean, std, target, x_lo=-np.inf, x_hi=np.inf
        )
    return report


def fitted_normalizations(model, vec) -> dict:
    """Map dataset name -> fitted normalization amplitude ``A`` from ``vec``.

    With ``*_FIT_NORMALIZATION`` on, each table's normalization is a free
    parameter rather than a covariance direction, so it is read from the free
    vector (labelled ``"<dataset> normalization / amp"``) instead of from
    :meth:`pixel.core.model.Model.nuisance_pulls`, which reports only
    marginalized nuisances.
    """
    vec = model.free_vector() if vec is None else vec
    values = np.asarray(vec, dtype=float).reshape(-1)
    out = {}
    for i, label in enumerate(model.free_labels()):
        owner, _, rest = label.partition(" normalization / ")
        if rest == "amp" and i < values.size:
            out[owner] = float(values[i])
    return out


def fitted_normalization_posterior(model, samples) -> dict:
    """Posterior mean/std for every explicit fitted normalization amplitude."""
    pairs = []
    for i, label in enumerate(model.free_labels()):
        owner, _, rest = label.partition(" normalization / ")
        if rest == "amp":
            pairs.append((owner, i))
    if not pairs:
        return {}
    if isinstance(samples, PosteriorResult):
        values = np.asarray(samples.x, dtype=float).reshape(-1)
        return {owner: {"mean": float(values[i]), "std": 0.0}
                for owner, i in pairs}
    indices = [i for _, i in pairs]
    values = np.real(np.asarray(samples.samples)[:, indices])
    weights = _weights(samples)
    mean = np.real(np.sum(weights[:, None] * values, axis=0))
    second = np.real(np.sum(weights[:, None] * values * values, axis=0))
    std = np.sqrt(np.clip(second - mean * mean, 0.0, None))
    return {
        owner: {"mean": float(mu), "std": float(sd)}
        for (owner, _), mu, sd in zip(pairs, mean, std)
    }


def exp_nuisance_report(q_key, model, vec, truth, fitted_stats=None) -> dict:
    """Recovery of the injected DIS normalizations and correlated systematics.

    Generation drew one ``beta_k ~ N(0, 1)`` per correlated source and applied a
    configured offset to each table's normalization; the fit folded the identical
    directions into the covariance.  ``nuisance_pulls`` inverts that, so each
    ``beta_hat`` is comparable with the injected truth in ``truth.json``.

    Per-source rows are far too many to print (HERA alone injects 169 per table),
    so each table is summarized: the normalization is reported individually and
    the correlated block as a pull chi2 per source plus a coverage fraction.

    **Reading ``pull_z``.**  The quoted ``uncertainty`` is ``1 - v^T C^+ v``, the
    width of ``beta`` with the *theory held fixed*.  It does not include the
    freedom of the PDF to absorb an overall normalization -- and that degeneracy
    is real: with few points per table the fitted PDF soaks up much of the
    injected shift, leaving the residual only partly along ``v``.  ``pull_z`` is
    therefore a *conditional* z-score and a diagnostic, not a significance; a
    large value means the offset went into the PDF, not that the machinery is
    wrong.  ``tests/test_normalization_nuisances.py`` pins the fields to break
    that degeneracy and is where exact recovery is asserted.
    """
    injected = (truth or {}).get("exp_nuisances") or {}
    if not injected:
        return {}
    by_dataset: dict = {}
    for row in model.nuisance_pulls(vec):
        by_dataset.setdefault(row["dataset"], []).append(row)
    # A *fitted* normalization is a free parameter, not a marginalized pull, so
    # its recovered value is read from the free vector instead.
    fitted = fitted_normalizations(model, vec)
    fitted_stats = fitted_stats or {}

    report = {}
    for label, truth_entry in injected.items():
        rows = by_dataset.get(label, [])
        if not rows and label not in fitted:
            continue
        entry = {}
        norm_true = truth_entry.get("normalization")
        norm_row = next((r for r in rows if r["name"] == "normalization"), None)
        if norm_true is not None and label in fitted:
            # Fitted: the amplitude IS beta (the row factor is 1 + A*rel and the
            # injected shift was 1 + beta_true*rel), so they compare directly.
            stats = fitted_stats.get(label, {})
            mean = float(stats.get("mean", fitted[label]))
            std = float(stats.get("std", 0.0))
            entry["normalization"] = {
                "beta_true": float(norm_true),
                "fitted": True,
                "pull": mean,
                "uncertainty": std,
                "pull_minus_true": float(mean - norm_true),
                "pull_z": _finite_or_none((mean - norm_true) / std)
                if std > 0.0 else None,
            }
        elif norm_true is not None and norm_row is not None:
            unc = float(norm_row["uncertainty"] or 0.0)
            entry["normalization"] = {
                "beta_true": float(norm_true),
                "fitted": False,
                "pull": norm_row["pull"],
                "uncertainty": norm_row["uncertainty"],
                "pull_minus_true": float(norm_row["pull"] - norm_true),
                "pull_z": _finite_or_none((norm_row["pull"] - norm_true) / unc)
                if unc > 0.0 else None,
            }
        corr_true = truth_entry.get("correlated")
        if corr_true:
            names = {r["name"]: r for r in rows if r["name"] != "normalization"}
            betas, pulls, uncs = [], [], []
            for beta, (_, r) in zip(corr_true, sorted(names.items())):
                betas.append(float(beta))
                pulls.append(float(r["pull"]))
                uncs.append(float(r["uncertainty"] or np.nan))
            betas, pulls, uncs = map(np.asarray, (betas, pulls, uncs))
            good = np.isfinite(uncs) & (uncs > 0.0)
            z = (pulls[good] - betas[good]) / uncs[good]
            entry["correlated"] = {
                "n_sources": int(betas.size),
                "pull_chi2_per_source": float(np.mean(z * z)) if z.size else None,
                "frac_within_1sigma": float(np.mean(np.abs(z) <= 1.0)) if z.size else None,
                "frac_within_2sigma": float(np.mean(np.abs(z) <= 2.0)) if z.size else None,
                "median_uncertainty": float(np.median(uncs[good])) if z.size else None,
                "max_abs_pull": float(np.max(np.abs(pulls))) if pulls.size else None,
            }
        if entry:
            report[label] = entry
    return report


def dy_normalization_report(q_key, model, vec, fitted_stats=None) -> dict:
    """Recovery of the injected Drell-Yan normalization offsets.

    Generation multiplied each DY table by ``1 + beta_true * rel_norm``; the fit
    marginalized ``rel_norm`` into that table's covariance.  ``nuisance_pulls``
    reports the equivalent best-fit ``beta`` from the posterior-predictive
    residual, so ``pull`` should come back at ``beta_true`` (shrunk slightly by
    the marginalization, and biased by ``1 / (1 + beta_true rel_norm)`` because
    the covariance references the data rather than the unknown truth).  Empty
    when the fit had no DY data or the truth predates injected normalizations.
    """
    try:
        manifest = dsets.load_manifest(q_key)
    except FileNotFoundError:
        return {}
    truth_beta = {rec["name"]: rec.get("norm_beta") for rec in manifest.get("dy", [])}
    rel_norm = {rec["name"]: rec.get("rel_norm") for rec in manifest.get("dy", [])}
    report = {}
    fitted = fitted_normalizations(model, vec)
    fitted_stats = fitted_stats or {}
    for name, amplitude in fitted.items():
        beta, rel = truth_beta.get(name), rel_norm.get(name)
        if beta is None or rel is None:
            continue
        # Fitted: the amplitude is directly comparable to the injected beta.
        stats = fitted_stats.get(name, {})
        mean = float(stats.get("mean", amplitude))
        std = float(stats.get("std", 0.0))
        report[name] = {
            "beta_true": float(beta), "rel_norm": float(rel), "fitted": True,
            "pull": mean, "uncertainty": std,
            "expected_pull": float(beta),
            "pull_minus_expected": float(mean - beta),
            "pull_z": _finite_or_none((mean - beta) / std)
            if std > 0.0 else None,
            "shift_rms": None,
        }
    for row in model.nuisance_pulls(vec):
        if row["name"] != "normalization" or row["dataset"] not in truth_beta:
            continue
        beta = truth_beta[row["dataset"]]
        rel = rel_norm[row["dataset"]]
        if beta is None or rel is None:
            continue
        # The data-referenced covariance rescales the recoverable pull; quote the
        # residual mismatch against that expectation, in units of its uncertainty.
        expected = beta / (1.0 + beta * rel)
        unc = float(row["uncertainty"] or 0.0)
        z = (row["pull"] - expected) / unc if unc > 0.0 else None
        report[row["dataset"]] = {
            "beta_true": float(beta),
            "rel_norm": float(rel),
            "fitted": False,
            "pull": row["pull"],
            "uncertainty": row["uncertainty"],
            "expected_pull": float(expected),
            "pull_minus_expected": float(row["pull"] - expected),
            "pull_z": _finite_or_none(z) if z is not None else None,
            "shift_rms": row["shift_rms"],
        }
    return report


def _finite_or_none(value):
    """Return a JSON-friendly finite float, else ``None``."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def summarize(q_key, mode, result, truth) -> dict:
    """JSON-serializable summary of one closure fit."""
    n_data = result["n_data"]
    chi2 = result["chi2"]
    chi2_out = _finite_or_none(chi2)
    return {
        "q_key": q_key, "mode": mode,
        "sampler": result["sampler"],
        "n_data": n_data,
        "n_free_parameters": result["n_free"],
        "n_free_hyperparams": result["n_free"],
        "n_gp_latent_coefficients": sum(
            result["model"].fields[i].n_grid for i in result["model"].gp_indices
        ),
        "n_datasets": len(result["model"].datasets),
        "n_constraint_data": sum(
            d.n_data for d in result["model"].datasets if d.name.startswith("cons_")
        ),
        "n_physical_data": n_data - sum(
            d.n_data for d in result["model"].datasets if d.name.startswith("cons_")
        ),
        "chi2": chi2_out,
        "chi2_per_data": _finite_or_none(chi2 / n_data) if n_data else None,
        "chi2_components": {
            key: _finite_or_none(value)
            for key, value in result.get("chi2_components", {}).items()
        },
        # The headline ESS is the autocorrelation (Wolff) one: how many
        # effectively independent draws the chain actually produced.  The
        # weight/phase ESS is kept beside it under its own name because the two
        # catch different failures -- a contour fit can have a healthy phase ESS
        # and still be worthless because the chain never moved, and vice versa.
        "effective_sample_size": _finite_or_none(result["autocorr"]["ess"]),
        "phase_effective_sample_size": _finite_or_none(result["ess"]),
        "tau_int_max": _finite_or_none(result["autocorr"]["tau_int_max"]),
        "slowest_param": result["autocorr"]["slowest"],
        "tau_reliable": result["autocorr"]["reliable"],
        "t0_iterations": result.get("t0_iterations"),
        "direct_diagnostics": result.get("direct_diagnostics"),
        "timing_seconds": result.get("timing_seconds", {}),
        "coverage": coverage_report(result["marginal"], truth),
        "nuisance_coverage": nuisance_coverage_report(result["marginal"], truth),
        "dy_normalization": dy_normalization_report(
            q_key, result["model"], result.get("vec_bar"),
            result.get("fitted_normalization_stats")),
        "exp_nuisances": exp_nuisance_report(
            q_key, result["model"], result.get("vec_bar"), truth,
            result.get("fitted_normalization_stats")),
    }


def load_truth(q_key: str) -> dict:
    return json.loads((cfg.truth_dir(q_key) / "truth.json").read_text())
