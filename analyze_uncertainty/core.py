"""Core uncertainty analysis utilities extracted from ``analyze_uncertainty``."""

from __future__ import annotations

import collections

import numpy as np
from chex import dataclass

from dictionary_methods import dictbased_runner
from core import infer


@dataclass
class AnalyzeUQconfig:
    mt_sim_mode: str = "expm_bmmat"
    loc_dict_res: int = 100
    known_sigma: float | None = None
    known_gt: bool = False

au_config = AnalyzeUQconfig()


def extract_nn_estimate_of_posterior(tissue_param_est, shape, is_cest):
    """Recover NN posterior statistics in visualization units.

    Parameters
    ----------
    tissue_param_est : Mapping[str, np.ndarray]
        Dictionary produced by the NBMF networks that contains per-voxel
        covariance factorizations (``ucov``/``scov``) and mean estimates for the
        relevant pool (``fb_T``/``kc_T`` or ``fc_T``/``kc_T``).
    shape : tuple
        Spatial shape of the target volume, used to reshape the flattened
        covariance factors back into ``(H, Z, W, 2, 2)`` arrays.
    is_cest : bool
        Flag indicating whether we are processing a CEST/rNOE pool (``True``)
        or an MT pool (``False``). This controls which scalings are used.

    Returns
    -------
    tuple
        ``(f_est, k_est, cov_scaled, f_sigma, k_sigma, height, width, angle)``
        where the estimates are expressed in percent and 1/s, and
        ``(height, width, angle)`` parameterize the 95% Gaussian confidence
        ellipse for each voxel.

    Notes
    -----
    The covariance matrices predicted by the network are supplied via an SVD
    factorization and must be rescaled into the physical units expected by the
    downstream visualizations.
    """
    if is_cest:
        f_est = 100 * tissue_param_est["fb_T"]
        k_est = tissue_param_est["kb_T"]
    else:
        f_est = 100 * tissue_param_est["fc_T"]
        k_est = tissue_param_est["kc_T"]

    u = tissue_param_est["ucov"].reshape((*shape, 2, 2))
    s = tissue_param_est["scov"].reshape((*shape, 2, 2))

    scales = (
        np.array(
            (
                100 * infer.infer_config.fb_scale_fact,
                infer.infer_config.kb_scale_fact,
            )
        )
        if is_cest
        else np.array(
            (
                100 * infer.infer_config.fc_scale_fact,
                infer.infer_config.kc_scale_fact,
            )
        )
    )

    _cov = u @ s @ np.transpose(u, [0, 1, 2, 4, 3])
    cov_scaled = scales[None, :] * _cov * scales[:, None]

    f_sigma = np.sqrt(cov_scaled[..., 0, 0])
    k_sigma = np.sqrt(cov_scaled[..., 1, 1])

    s2_scaled, u_scaled = np.linalg.eigh(cov_scaled)
    s_scaled = np.sqrt(s2_scaled)
    d0 = s_scaled[..., 0, None] * u_scaled[..., 0]
    d1 = s_scaled[..., 1, None] * u_scaled[..., 1]

    angle = np.degrees(np.arctan2(d1[..., 1], d1[..., 0]))
    chisq = 2.45
    height = np.linalg.norm(d1, axis=-1) * 2 * chisq
    width = np.linalg.norm(d0, axis=-1) * 2 * chisq
    return f_est, k_est, cov_scaled, f_sigma, k_sigma, height, width, angle


def get_nrmse_grid(
    ax,
    _x,
    _y,
    sli,
    data_feed_mt=None,
    mt_tissue_param_est=None,
    data_feed_cest=None,
    mt_sim_mode=None,
    loc_dict_res=None,
    seq_name="mt",
    seq_df=None,
    max_f_mt=None,
    max_k_mt=None,
    max_f_cest=None,
    max_k_cest=None,
):
    """Compute the voxel-wise NRMSE grid across a local f-k dictionary.

    This is the core per-voxel likelihood-mapping routine.
    For a given voxel it:

    1. Builds a bespoke dictionary around the auxiliary parameters (T1, T2,
       B0, B1, and optionally MT estimates when fitting CEST).
    2. Normalizes both measured and synthetic signals to a common L2 scale.
    3. Evaluates the normalized root-mean-square error (NRMSE) across the
       Cartesian grid of f/k hypotheses.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Legacy axis argument retained for backwards compatibility. Plotting is
        now handled elsewhere so the value is ignored.
    _x, _y : int
        In-plane voxel coordinates.
    sli : int
        Slice index.
    data_feed_mt, data_feed_cest : SlicesFeed, optional
        Data feeds that expose the measured signals and auxiliary maps for MT
        or CEST acquisitions. Exactly one is expected depending on
        ``seq_name``.
    mt_tissue_param_est : dict, optional
        MT parameter estimates used as conditioning when fitting CEST pools.
    mt_sim_mode : str, optional
        Simulation mode for MT dictionaries. Defaults to ``au_config.mt_sim_mode``.
    loc_dict_res : int
        Resolution of the local dictionary grid along each axis.
    seq_name : {"mt", ...}
        Sequence identifier. Non-"mt" values trigger the CEST branch.
    seq_df : pandas.DataFrame, optional
        Sequence definition forwarded to ``dictbased_runner``.
    max_f_* / max_k_* : float, optional
        Axis limits used to derive the grid spacing ``_df``/``_dk`` when the
        defaults from ``infer.infer_config`` need overriding.

    Returns
    -------
    tuple
        ``(f_best, k_best, dict_signal, nrmse, _df, _dk)`` where ``f_best`` is
        reported in percent and ``_df``/``_dk`` are the step sizes in percent
        and 1/s respectively.

    """

    del ax  # legacy plotting axis retained for compatibility
    loc_dict_res = loc_dict_res or au_config.loc_dict_res
    mt_sim_mode = mt_sim_mode or au_config.mt_sim_mode

    max_f_mt = max_f_mt or infer.infer_config.fc_scale_fact
    max_k_mt = max_k_mt or infer.infer_config.kc_scale_fact
    max_f_cest = max_f_cest or infer.infer_config.fb_scale_fact
    max_k_cest = max_k_cest or infer.infer_config.kb_scale_fact

    if seq_name != "mt":
        signal = data_feed_cest.measured_normed_T[:, _x, sli, _y]
        _df, _dk = max_f_cest / loc_dict_res, max_k_cest / loc_dict_res
        local_dict_feed = dictbased_runner.get_dict(
            use_cartesian=True,
            seq_name=seq_name,
            seq_df=seq_df,
            parameter_values_od=collections.OrderedDict(
                {
                    "T1a_ms": np.array(1000 / data_feed_cest.R1a_V[_x, sli, _y]),
                    "T2a_ms": np.array(1000 / data_feed_cest.R2a_V[_x, sli, _y]),
                    "B0_shift_ppm_map": np.array(
                        data_feed_cest.B0_shift_ppm_map[_x, sli, _y]
                    ),
                    "B1_fix_factor_map": np.array(
                        data_feed_cest.B1_fix_factor_map[_x, sli, _y]
                    ),
                    "fc_gt_T": mt_tissue_param_est["fc_T"][_x, sli, _y],
                    "kc_gt_T": mt_tissue_param_est["kc_T"][_x, sli, _y],
                    "fb_gt_T": np.arange(_df, max_f_cest + _df, _df),
                    "kb_gt_T": np.arange(_dk, max_k_cest + _dk, _dk),
                }
            ),
            shape=[loc_dict_res, 1, loc_dict_res],
        )
    else:
        signal = data_feed_mt.measured_normed_T[:, _x, sli, _y]
        _df, _dk = max_f_mt / loc_dict_res, max_k_mt / loc_dict_res
        local_dict_feed = dictbased_runner.get_dict(
            use_cartesian=True,
            seq_name="mt",
            seq_df=seq_df,
            mt_sim_mode=mt_sim_mode,
            parameter_values_od=collections.OrderedDict(
                {
                    "T1a_ms": np.array(1000 / data_feed_mt.R1a_V[_x, sli, _y]),
                    "T2a_ms": np.array(1000 / data_feed_mt.R2a_V[_x, sli, _y]),
                    "B0_shift_ppm_map": np.array(
                        data_feed_mt.B0_shift_ppm_map[_x, sli, _y]
                    ),
                    "B1_fix_factor_map": np.array(
                        data_feed_mt.B1_fix_factor_map[_x, sli, _y]
                    ),
                    "fb_gt_T": [0.0],
                    "kb_gt_T": [0.0],
                    "fc_gt_T": np.arange(_df, max_f_mt + _df, _df),
                    "kc_gt_T": np.arange(_dk, max_k_mt + _dk, _dk),
                }
            ),
            shape=[loc_dict_res, 1, loc_dict_res],
        )

    dict_signal = local_dict_feed.normalize(
        local_dict_feed.measured_normed_T, "l2"
    )[:, :, 0, :]
    signal = local_dict_feed.normalize(signal, "l2")

    dot_prod = np.transpose(dict_signal, [1, 2, 0]) @ signal
    nrmse = np.sqrt(2 * (1 - dot_prod))

    f_best = (1 + np.nanargmin(nrmse) // nrmse.shape[1]) * _df * 100
    k_best = (1 + np.nanargmin(nrmse) % nrmse.shape[1]) * _dk

    return f_best, k_best, dict_signal, nrmse, _df, _dk


def nrmse_to_CIs(
    nrmse,
    df=None,
    dk=None,
    is_cest=False,
    loc_dict_res=None,
    n_meas=31,
    prior=1,
    gt=None,
    pdf=None
):
    """Convert an NRMSE grid into "exact" posterior statistics and credible intervals.

    Parameters
    ----------
    nrmse : np.ndarray
        2D grid generated by :func:`get_nrmse_grid` for a particular voxel.
    df, dk : float, optional
        Grid spacings along ``f`` (in ratio units) and ``k`` (1/s). If omitted
        they are inferred from ``loc_dict_res`` and the active infer config.
    is_cest : bool
        Selects the appropriate scale factors for MT vs. CEST pools.
    loc_dict_res : int, optional
        Resolution of the local dictionary used to infer ``df``/``dk``.
    n_meas : int
        Number of measurements contributing to the NRMSE computation; needed
        for the Gaussian-likelihood scaling.
    prior : float or np.ndarray
        Optional prior weight applied element-wise before normalising the PDF.

    Returns
    -------
    tuple
        ``(pdf, nrmse_95cr_th, CR_area, f_mean, f_025, f_975, k_mean, k_025, k_975, posterior_cov)``
        providing the posterior surface, 95% CR threshold, CR area (in pixels),
        posterior means, and central credible intervals for both parameters,
        along with the covariance of the exact posterior.
    Notes: 
        a. ``f_mean``, ``f_025``, and ``f_975`` are expressed in percentage units (0-100%)
            and covariance elements accordingly scaled (to f in percent).
        b. PDF is discreet-normalised such that sum(pdf) == 1 (not integral over f/k)
    """

    loc_dict_res = loc_dict_res or au_config.loc_dict_res
    df = df or (
        infer.infer_config.fb_scale_fact
        if is_cest
        else infer.infer_config.fc_scale_fact
    ) / loc_dict_res
    df_in_percent = 100 * df
    
    dk = dk or (
        infer.infer_config.kb_scale_fact
        if is_cest
        else infer.infer_config.kc_scale_fact
    ) / loc_dict_res
    
    if pdf is None: 
        # Estimate noise level from min NRMSE ("bootstrap" approach) - 
        #    unless known apriori (e.g., simulated data, or calibrated noise floor)
        sigma_est = au_config.known_sigma or np.min(nrmse) / np.sqrt(n_meas)
        # Compute likelihoods under Gaussian noise assumption, using the NRMSE as the error metric
        log_likelihood = - (nrmse**2 - np.min(nrmse) ** 2) / (2 * sigma_est**2)
        pdf = np.exp(log_likelihood) * prior
    pdf /= np.sum(pdf)
    if nrmse is None: # auxiliary monotone func just for the CDF-based CR derivation below
        nrmse = np.sqrt(-np.log(pdf + 1e-8))
    
    nrmse_over_min_thresholds = np.arange(1.005, 2, 0.005)
    posterior_cdf = [
        np.sum(pdf[nrmse < th * np.min(nrmse)]) for th in nrmse_over_min_thresholds
    ]

    nrmse_95cr_th_over_min = nrmse_over_min_thresholds[
        np.argmin(np.abs(np.array(posterior_cdf) - 0.95))
    ]
    nrmse_95cr_th = np.min(nrmse) * nrmse_95cr_th_over_min
    CR_area = np.sum(nrmse < nrmse_95cr_th)
    
    if gt is not None and au_config.known_gt:
        gt_f, gt_k = gt
        gt_f_idx = int(np.round(gt_f / df)) - 1
        gt_k_idx = int(np.round(gt_k / dk)) - 1
        nrmse_at_gt = nrmse[gt_f_idx, gt_k_idx]
        cdf_at_gt = posterior_cdf[np.argmin(np.abs(np.array(nrmse_over_min_thresholds) - nrmse_at_gt / np.min(nrmse)))]
    else:
        cdf_at_gt = None
        
    df_grid = df_in_percent * np.arange(0, nrmse.shape[0])[:, None]
    dk_grid = dk * np.arange(0, nrmse.shape[1])[None, :]

    marg_f = np.sum(pdf, axis=1)
    f_mean_posterior = np.sum(marg_f * df_grid[:, 0])
    marg_f_cdf = np.cumsum(marg_f)
    marg_k = np.sum(pdf, axis=0)
    k_mean_posterior = np.sum(marg_k * dk_grid[0, :])
    marg_k_cdf = np.cumsum(marg_k)
    f_975 = df_in_percent * np.searchsorted(marg_f_cdf, 0.975)
    f_025 = df_in_percent * np.searchsorted(marg_f_cdf, 0.025)
    k_975 = dk * np.searchsorted(marg_k_cdf, 0.975)
    k_025 = dk * np.searchsorted(marg_k_cdf, 0.025)
    
    delta = np.stack(
        (
            df_grid.repeat(loc_dict_res, axis=1) - f_mean_posterior,
            dk_grid.repeat(loc_dict_res, axis=0) - k_mean_posterior,
        ),
        axis=-1,
    )
    posterior_cov = (pdf[..., None, None] * delta[..., None] @ delta[..., None, :]).sum(axis=(0, 1))    
    
    return (
        pdf,
        nrmse_95cr_th,
        CR_area,
        f_mean_posterior,
        f_025,
        f_975,
        k_mean_posterior,
        k_025,
        k_975,
        posterior_cov,
        cdf_at_gt
    )

def pdf_to_marginals_and_CIs(pdf, df, dk):
    """Convert a joint PDF over the f-k grid into marginals and credible intervals."""
    df_in_percent = 100 * df
    df_grid = df_in_percent * np.arange(0, pdf.shape[0])[:, None]
    dk_grid = dk * np.arange(0, pdf.shape[1])[None, :]

    marg_f = np.sum(pdf, axis=1)
    f_mean_posterior = np.sum(marg_f * df_grid[:, 0])
    marg_f_cdf = np.cumsum(marg_f)
    marg_k = np.sum(pdf, axis=0)
    k_mean_posterior = np.sum(marg_k * dk_grid[0, :])
    marg_k_cdf = np.cumsum(marg_k)
    f_975 = df_in_percent * np.searchsorted(marg_f_cdf, 0.975)
    f_025 = df_in_percent * np.searchsorted(marg_f_cdf, 0.025)
    k_975 = dk * np.searchsorted(marg_k_cdf, 0.975)
    k_025 = dk * np.searchsorted(marg_k_cdf, 0.025)

    return f_mean_posterior, f_025, f_975, k_mean_posterior, k_025, k_975
    
def get_progressive_pdf(_signal, dict_signal):
    """Compute cumulative likelihood maps as the sequence grows.

    Parameters
    ----------
    _signal : np.ndarray
        Measured voxel signal (1D) that will be progressively analyzed.
    dict_signal : np.ndarray
        Local dictionary responses 
    Returns
    -------
    tuple
        ``(cumsum_sqerr, pdf)`` where ``cumsum_sqerr`` tracks the cumulative
        squared error for each truncation point and ``pdf`` stores the
        normalised posterior surface associated with each prefix length.
    """
    _signal /= np.linalg.norm(_signal, axis=0)

    residuals = dict_signal - _signal
    residuals_sq = residuals**2
    cumsum_sqerr = np.cumsum(residuals_sq, axis=0)
    n_meas = dict_signal.shape[0]
    nrmse = np.sqrt(cumsum_sqerr[-1])
    sigma_est = np.nanmin(nrmse)
    cumul_log_likelihood = -n_meas * cumsum_sqerr / (2 * sigma_est**2)
    cumul_log_likelihood -= np.nanmax(cumul_log_likelihood)
    pdf = np.exp(cumul_log_likelihood)
    pdf /= 1e-8 + np.sum(pdf, axis=(1, 2), keepdims=True)
    return cumsum_sqerr, pdf


__all__ = [
    "AnalyzeUQconfig",
    "au_config",
    "extract_nn_estimate_of_posterior",
    "get_nrmse_grid",
    "nrmse_to_CIs",
    "get_progressive_pdf",
]