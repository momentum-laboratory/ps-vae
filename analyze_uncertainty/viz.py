"""Visualization helpers extracted from ``analyze_uncertainty``."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.patches import Arrow, Ellipse
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from core import infer
from helpers import utils

from .core import au_config, extract_nn_estimate_of_posterior, get_nrmse_grid, nrmse_to_CIs


class Null:
    """No-op stand-in used when callers disable visualization."""

    def __getattr__(self, name):  # noqa: D401 - small helper
        return self

    def __call__(self, *args, **kwargs):
        return self


def cbarhist(img, data, ax, pad: float = 0.05, bins: int | np.ndarray = 50, old_style: bool = False):
    """Attach a colorbar that overlays the histogram of ``data``."""    
    if old_style:
        cax = make_axes_locatable(ax).append_axes("right", size="15%", pad=pad)
    else:
        # ! switched to inset_axes to have more control oveer height
        cax = inset_axes(ax, width="15%", height="90%", loc="center right", borderpad=-pad*15)    
    cbar = plt.colorbar(img, cax=cax)    
    hist_data = data.flatten()
    if isinstance(bins, int):
        hist_data = hist_data[~np.isnan(hist_data)]
    else:
        hist_data = np.clip(hist_data[~np.isnan(hist_data)], bins[0], bins[-1])  # ignore nans and outliers for histogram
    
    hist, bins = np.histogram(hist_data, bins=bins, density=True)
    hist = hist / np.max(hist) if np.size(hist) else hist
    hist = 0.9*hist + 0.1  # scale to fit nicely within the colorbar
    cbar.ax.plot(hist, bins[1:], color="white", linewidth=2)
    return cbar


def plot_map(ax, data, vmin, vmax, cmap, label, *, cbar_ticks=True, pad=0.2, fontsize=14, **kwargs):
    """Render ``data`` with a histogram colorbar for quick distribution readout."""

    img = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = cbarhist(img, data, ax, pad=pad, bins=np.linspace(vmin, vmax, 50), **kwargs)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.yaxis.set_ticks_position("left")
    if not cbar_ticks:
        cbar.ax.yaxis.set_ticks([])
    return img, cbar


def plot_CI_maps(
    f_est,
    f_sigma,
    k_est,
    k_sigma,
    is_cest,
    figsize=(17, 8),
    vmin_f=0,
    vmax_f=None,
    vmin_k=0,
    vmax_k=None,
    cmap_f="viridis",
    cmap_k="magma",
    fig=None,
    axes=None,
    ktop=False,
    ticks_for_all_cbars=False,
    kwargs_plot_map=None,
):
    """Plot mean ± 2σ confidence bounds for ``f`` and ``k`` estimates."""

    kwargs_plot_map = kwargs_plot_map or {}
    if axes is None:
        fig = fig or plt.figure(figsize=figsize)
        axes = fig.subplots(2, 3)
        
    f_label = r"$\hat{f}_{s}$" if is_cest else r"$\hat{f}_{ss}$"
    k_label = r"$\hat{k}_{sw}$" if is_cest else r"$\hat{k}_{ssw}$"
    vmax_f = vmax_f or 100 * (
        infer.infer_config.fb_scale_fact if is_cest else infer.infer_config.fc_scale_fact
    )
    vmax_k = vmax_k or (
        infer.infer_config.kb_scale_fact if is_cest else infer.infer_config.kc_scale_fact
    )

    frow = 1 if ktop else 0
    plot_map(
        axes[frow, 0],
        f_est - 2 * f_sigma,
        vmin_f,
        vmax_f,
        cmap_f,
        f_label + r"$ - 2\hat{\sigma}_f$  (%)",
        cbar_ticks=ticks_for_all_cbars,
        **kwargs_plot_map,
    )
    plot_map(
        axes[frow, 1],
        f_est,
        vmin_f,
        vmax_f,
        cmap_f,
        f_label + "       (%)",
        **kwargs_plot_map,
    )
    plot_map(
        axes[frow, 2],
        f_est + 2 * f_sigma,
        vmin_f,
        vmax_f,
        cmap_f,
        f_label + r" + 2$\hat{\sigma}_f$  (%)",
        cbar_ticks=ticks_for_all_cbars,
        **kwargs_plot_map,
    )
    plot_map(
        axes[1 - frow, 2],
        k_est - 2 * k_sigma,
        vmin_k,
        vmax_k,
        cmap_k,
        k_label + r" - 2$\hat{\sigma}_k  (s^{-1})$",
        cbar_ticks=ticks_for_all_cbars,
        **kwargs_plot_map,
    )
    plot_map(
        axes[1 - frow, 1],
        k_est,
        vmin_k,
        vmax_k,
        cmap_k,
        k_label + r"      $(s^{-1})$",
        **kwargs_plot_map,
    )
    plot_map(
        axes[1 - frow, 0],
        k_est + 2 * k_sigma,
        vmin_k,
        vmax_k,
        cmap_k,
        k_label + r" + 2$\hat{\sigma}_k  (s^{-1})$",
        cbar_ticks=ticks_for_all_cbars,
        **kwargs_plot_map,
    )
    for axis in axes.flatten():
        utils.remove_spines(axis)
    return axes


def viz_cov(
    f_sigma_slice,
    f_val_slice,
    k_sigma_slice,
    k_val_slice,
    df_c_0_slice,
    dk_ca_0_slice,
    df_c_1_slice,
    dk_ca_1_slice,
    f_total,
    k_total,
):
    """Summarize posterior covariance components as maps."""

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))

    img = axes[0, 0].imshow(
        f_sigma_slice / (1e-6 + f_val_slice), cmap="hot_r", vmin=0, vmax=0.25
    )
    cbar = cbarhist(img, f_sigma_slice / (1e-6 + f_val_slice), axes[0, 0])
    cbar.set_label(r"$\sigma / \mu\ [$f$_{ss}$]", fontsize=14)
    img = axes[1, 0].imshow(
        k_sigma_slice / k_val_slice, cmap="hot_r", vmin=0, vmax=0.25
    )
    cbar = cbarhist(img, k_sigma_slice / k_val_slice, axes[1, 0])
    cbar.set_label(r"$\sigma / \mu\ [$k$_{ssw}]$", fontsize=14)

    img = axes[0, 1].imshow(f_val_slice, vmin=0)
    cbar = cbarhist(img, f_val_slice, axes[0, 1])
    cbar.set_label(r"$\mu\ [$f$_{ss}$] (%)", fontsize=14)
    img = axes[1, 1].imshow(k_val_slice, vmin=0, cmap="magma")
    cbar = cbarhist(img, k_val_slice, axes[1, 1])
    cbar.set_label(r"$\mu\ [$k$_{ssw}]$ (s$^{-1})$", fontsize=14)

    img = axes[0, 2].imshow(f_sigma_slice, vmin=0, vmax=np.nanmax(f_val_slice) / 6)
    cbar = cbarhist(img, f_sigma_slice, axes[0, 2])
    cbar.set_label(r"$\sigma\ [$f$_{ss}$] (%)", fontsize=14)
    img = axes[1, 2].imshow(
        k_sigma_slice, vmin=0, vmax=np.nanmax(k_val_slice) / 6, cmap="magma"
    )
    cbar = cbarhist(img, k_sigma_slice, axes[1, 2])
    cbar.set_label(r"$\sigma\ [$k$_{ssw}]$ (s$^{-1})$", fontsize=14)

    img = axes[0, 3].imshow(
        100 * df_c_0_slice,
        vmin=-np.nanmax(100 * f_total),
        vmax=np.nanmax(100 * f_total),
        cmap="bwr",
    )
    cbar = cbarhist(img, 100 * df_c_0_slice, axes[0, 3])
    cbar.set_label(r"$\sigma_{PC0}\ [$f$_{ss}$] (%)", fontsize=14)
    img = axes[1, 3].imshow(
        dk_ca_0_slice,
        vmin=-np.nanmax(k_total),
        vmax=np.nanmax(k_total),
        cmap="bwr",
    )
    cbar = cbarhist(img, dk_ca_0_slice, axes[1, 3])
    cbar.set_label(r"$\sigma_{PC0}\ [$k$_{ssw}$] (%)", fontsize=14)

    img = axes[0, 4].imshow(
        100 * df_c_1_slice,
        vmin=-np.nanmax(100 * f_total),
        vmax=np.nanmax(100 * f_total),
        cmap="bwr",
    )
    cbar = cbarhist(img, 100 * df_c_1_slice, axes[0, 4])
    cbar.set_label(r"$\sigma_{PC1}\ [$f$_{ss}$] (%)", fontsize=14)
    img = axes[1, 4].imshow(
        dk_ca_1_slice,
        vmin=-np.nanmax(k_total),
        vmax=np.nanmax(k_total),
        cmap="bwr",
    )
    cbar = cbarhist(img, dk_ca_1_slice, axes[1, 4])
    cbar.set_label(r"$\sigma_{PC1}\ [$k$_{ssw}$] (%)", fontsize=14)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])


def plot_ellipse(
    ax,
    _x,
    _y,
    sli,
    f_val,
    k_val,
    df_c_0,
    dk_ca_0,
    df_c_1,
    dk_ca_1,
    cest=False,
    color="r",
    do_minor_ellipses=False,
    fontsize=14,
):
    """Visualize principal directions as ellipses for a single voxel."""

    pd0 = np.array([100 * df_c_0[_x, sli, _y], dk_ca_0[_x, sli, _y]])
    pd1 = np.array([100 * df_c_1[_x, sli, _y], dk_ca_1[_x, sli, _y]])
    if np.linalg.norm(pd0) > np.linalg.norm(pd1):
        pd0, pd1 = pd1, pd0
    angle = np.degrees(np.arctan2(pd1[1], pd1[0]))
    height = np.linalg.norm(pd1) * 4
    width = (
        np.linalg.norm(pd0 - (pd0.T @ pd1) * pd1 / np.linalg.norm(pd1) ** 2)
        * 2
        * 2.45
    )
    center = [f_val[_x, sli, _y], k_val[_x, sli, _y]]

    maj_ellipse = ax.add_patch(
        Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle - 90,
            edgecolor=color,
            facecolor="none",
            linewidth=2,
            zorder=2,
        )
    )
    if do_minor_ellipses:
        for scale in (0.5, 1.5):
            ax.add_patch(
                Ellipse(
                    xy=center,
                    width=width * scale,
                    height=height * scale,
                    angle=angle - 90,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=0.5,
                    zorder=2,
                )
            )

    ax.set_xlim(0, 1.2 if cest else 30)
    ax.set_ylim(0, 600 if cest else 100)
    ax.set_xlabel("f$_{s}$ (%)" if cest else "f$_{ss}$ (%)", fontsize=fontsize)
    ax.set_ylabel("k$_{sw}$ (s$^{-1}$)" if cest else "k$_{ssw}$ (s$^{-1}$)", fontsize=fontsize)
    return maj_ellipse


def simple_error_map(err_mt, err_cest, do_cest=True, sli=0, figsize=(9, 2)):
    """Display MT/CEST NRMSE maps using shared color mapping."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    log_bins = np.logspace(np.log10(1.0), np.log10(14), num=12) / 100
    norm = mcolors.BoundaryNorm(log_bins, ncolors=plt.get_cmap("hot_r").N, clip=True)

    cmap_nrmse = plt.cm.get_cmap("YlOrRd").copy()
    cmap_nrmse.set_bad("1.0")
    axes[0].set_title("MT")
    axes[1].set_title("CEST")
    img0 = axes[0].imshow(err_mt[:, sli, :], norm=norm, cmap=cmap_nrmse)
    cbar = cbarhist(img0, err_mt[:, sli, :], axes[0])
    cbar.set_ticks(log_bins)
    cbar.set_ticklabels([f"{v*100:.1f}%" for v in log_bins])
    cbar.set_label("NRMSE (%)")
    img1 = axes[1].imshow(
        err_cest[:, sli, :], norm=norm, cmap=cmap_nrmse
    ) if do_cest else axes[1].imshow(
        np.zeros_like(err_mt[:, sli, :]), norm=norm, cmap=cmap_nrmse
    )
    cbar = cbarhist(img1, err_cest[:, sli, :], axes[1])
    cbar.set_ticks(log_bins)
    cbar.set_ticklabels([f"{v*100:.1f}%" for v in log_bins])
    cbar.set_label("NRMSE (%)")


def demonstrate_ellipses(
    xy_points,
    f_est,
    k_est,
    width,
    height,
    angle,
    labels,
    is_mt=True,
    xlim=None,
    ylim=None,
    colord="cmykrgb",
    figsize=(7, 3),
):
    """Overlay NN uncertainty ellipses for a collection of voxels."""

    xlim = xlim or (
        0,
        100
        * (
            infer.infer_config.fc_scale_fact
            if is_mt
            else infer.infer_config.fb_scale_fact
        ),
    )
    ylim = ylim or (
        0,
        infer.infer_config.kc_scale_fact
        if is_mt
        else infer.infer_config.kb_scale_fact,
    )
    fig, ax = plt.subplots(figsize=figsize)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(r"$\hat{f}_{ss}$ (%)" if is_mt else r"$\hat{f}_{s}$   (%)")
    plt.ylabel(r"$\hat{k}_{ss}\ (s^{-1})$" if is_mt else r"$\hat{k}_{s}\ (s^{-1})$")

    idx_x = [p[0] for p in xy_points]
    idx_y = [p[1] for p in xy_points]
    for mu_f, mu_k, ew, eh, eangle, label in zip(
        f_est[idx_x, idx_y],
        k_est[idx_x, idx_y],
        width[idx_x, idx_y],
        height[idx_x, idx_y],
        angle[idx_x, idx_y],
        labels,
    ):
        ax.add_patch(
            Ellipse(
                xy=(mu_f, mu_k),
                width=ew,
                height=eh,
                angle=eangle - 90,
                edgecolor=colord[label],
                facecolor="none",
                linewidth=0.5,
                zorder=0,
                alpha=0.5,
            )
        )
        ax.add_patch(
            Ellipse(
                xy=(mu_f, mu_k),
                width=ew / 2,
                height=eh / 2,
                angle=eangle - 90,
                edgecolor="none",
                facecolor=colord[label],
                alpha=0.2,
            )
        )

def nrmse_to_CI_contour(nrmse, n_meas=31, sigma_est=None):
    """Convert NRMSE grid to a contour level corresponding to a confidence interval."""
    sigma_est = sigma_est or np.min(nrmse) / np.sqrt(n_meas)
    log_likelihood = - (nrmse**2 - np.min(nrmse) ** 2) / (2 * sigma_est**2)
    pdf = np.exp(log_likelihood) / np.sum(np.exp(log_likelihood))    
    nrmse_over_min = np.arange(1.005, 2, 0.005)
    posterior_cdf = [np.sum(pdf[nrmse < th * np.min(nrmse)]) for th in nrmse_over_min]
    nrmse_95_th = np.min(nrmse) * nrmse_over_min[np.argmin(np.abs(np.array(posterior_cdf) - 0.95))]
    print('nrmse_95_th', nrmse_95_th, 'cdf95', posterior_cdf[np.argmin(np.abs(np.array(posterior_cdf) - 0.95))])
    f_best_unscaled = (1 + np.nanargmin(nrmse) // nrmse.shape[1])
    k_best_unscaled = (1 + np.nanargmin(nrmse) % nrmse.shape[1])
    return pdf, nrmse_95_th, f_best_unscaled, k_best_unscaled


def viz_posteriors_two_nn(
    ax, nrmse, nn_est1, nn_est2, gt=None, # width1, height1, angle1, width2, height2, angle2, ax, 
    n_meas=31, _df=None, _dk=None, linewidth=1.5, markersize=5, is_cest=False, fontsize=14, do_heatmap=True,
    nn_colors = ("c", "g"), nn_est_list=None, do_legend=True, legend_loc="center right", kmax=None, 
    alt_NN_name=None, ellipse_linestyle='-', with_PSVAE=True
):
    loc_dict_res = nrmse.shape[0] 
    _df = _df or (infer.infer_config.fc_scale_fact if not is_cest else infer.infer_config.fb_scale_fact) / loc_dict_res
    _dk = _dk or (infer.infer_config.kc_scale_fact if not is_cest else infer.infer_config.kb_scale_fact) / loc_dict_res    
    
    if gt is not None and au_config.known_gt:                
        gt_handle = ax.plot(
            [gt[0], gt[0]],
            [gt[1], gt[1]],
            "*",
            color="blue",
            markersize=markersize+2,
            zorder=10
        )        
        handles, labels = [gt_handle[0]], [r"Ground Truth $\theta$"]
    else:
        handles, labels = [], []
                
    pdf, nrmse_95_th, f_best_unscaled, k_best_unscaled = nrmse_to_CI_contour(nrmse, n_meas=n_meas, sigma_est=None)
    f_best_dotprod, k_best_dotprod = f_best_unscaled * _df * 100, k_best_unscaled * _dk
    
    extent = [_df, _df * 100 * (loc_dict_res+1), _dk, _dk * (loc_dict_res+1)]
    ax.contour(
        nrmse.T,
        levels=[nrmse_95_th],
        colors=["magenta"],
        linewidths=[linewidth],
        extent=extent
    )    
    if do_heatmap:
        imsh = ax.imshow(
            pdf.T / np.max(pdf),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="hot",
        )
        cbar = plt.colorbar(imsh, ax=ax, fraction=0.20, pad=0.005)
        cbar.set_label("Ref. posterior PDF (a.u.)", fontsize=fontsize-4)
    ref_MAP_handle = ax.plot(
        [f_best_dotprod, f_best_dotprod],
        [k_best_dotprod, k_best_dotprod],
        "X",
        color="k",
        markersize=markersize,
    )
    ax.set_xlabel('f$_{s}$ (%)' if is_cest else 'f$_{ss}$ (%)', fontsize=fontsize)
    ax.set_ylabel('k$_{sw}$ (s$^{-1}$)' if is_cest else 'k$_{ssw}$ (s$^{-1}$)', fontsize=fontsize)    
    handles += [ref_MAP_handle[0], mpatches.Patch(color="m")]
    labels += [r"Ref. MAP $\hat{\theta}$", "Ref. 95% CR"]  #["Grid Bayesian MAP", "Grid Bayesian 95% CR"]
    
    nn_est_list = nn_est_list or [nn_est1, nn_est2]
    for jj, (c, nn_est) in enumerate(zip(nn_colors, nn_est_list)):
        ellipse = Ellipse(
            xy=(nn_est['f'], nn_est['k']),
            width=nn_est['width'],
            height=nn_est['height'],
            angle=nn_est['angle'] - 90,
            edgecolor=c,
            facecolor="none",
            linewidth=linewidth,
            linestyle=ellipse_linestyle,
        )
        ax.add_patch(ellipse)
        nn_point = ax.plot(
            [nn_est['f'], nn_est['f']],
            [nn_est['k'], nn_est['k']],
            "X",
            color=c,
            markersize=markersize,
        )    
        handles += [nn_point[0], ellipse]
    alt_NN_name = alt_NN_name or "alt. NN"
    if with_PSVAE:
        labels += [r"PS-VAE MAP $\hat{\theta}$", r"PS-VAE 95% CR", rf"{alt_NN_name} MAP $\hat{{\theta}}$", rf"{alt_NN_name} 95% CR"]                
    else:
        labels += [rf"{alt_NN_name} MAP $\hat{{\theta}}$", rf"{alt_NN_name} 95% CR"]
    if do_legend:
        ax.legend(handles, labels, loc=legend_loc, fontsize=fontsize-5)
    ax.set_ylim(0, kmax if kmax is not None else _dk * (loc_dict_res+1))
    ax.set_xlim(0, _df * 100 * (loc_dict_res+1))
    
    
def viz_posteriors(
    f_est_nn,
    k_est_nn,
    cov_nnpred_scaled,
    width,
    height,
    angle,
    f_best_dotprod,
    k_best_dotprod,
    nrmse,
    _df=None,
    _dk=None,
    n_meas=30,
    is_cest=False,
    visualize=True,
    ax=None,
    figsize=(6, 4),
    fontsize=14,
    do_legend=True,
    do_cbar=True,
    do_marginals=True,
    show_text=True,
    show_NN=True,
    loc_dict_res=None,
    xlim=None,
    ylim=None,
    imshow_alpha=1,
    fig=None,
    marg_color="gray",
    linewidth=1,
    markersize=5,
    gt=None,
    max_k_cest=None
):
    """Compare dictionary-based posteriors against NN estimates for a single voxel.

    This routine mirrors the legacy visualization that overlays the
    brute-force dictionary posterior with the network's Gaussian approximation.
    It optionally augments the plot with marginal distributions, colorbars, and
    descriptive annotations so notebooks can inspect calibration/coverage.

    Parameters
    ----------
    f_est_nn, k_est_nn : float
        NN point estimates (in percent and 1/s).
    cov_nnpred_scaled : np.ndarray, shape (2, 2)
        Scaled covariance predicted by the NN for the voxel of interest.
    width, height, angle : float
        Pre-computed ellipse parameters (95% CR) derived from
        ``cov_nnpred_scaled``.
    f_best_dotprod, k_best_dotprod : float
        MAP estimates obtained from the grid search (in percent and 1/s).
    nrmse : np.ndarray
        NRMSE landscape from ``get_nrmse_grid`` for the voxel.
    _df, _dk : float, optional
        Grid spacings along ``f`` and ``k``. If omitted they are inferred from
        ``loc_dict_res`` and the active ``infer_config`` scales.
    n_meas : int
        Number of measurements used to derive the NRMSE grid (affects the
        pseudo-likelihood scaling).
    is_cest : bool
        Selects label/scale conventions appropriate for MT or CEST pools.
    visualize : bool
        When ``False`` the function still computes statistics but returns
        placeholder axes to simplify scripting.
    ax, fig : matplotlib objects, optional
        Pre-existing axes/figures for reuse.
    do_legend, do_cbar, do_marginals, show_text, show_NN : bool
        Feature toggles that control the level of annotation displayed.

    Returns
    -------
    list
        ``[M_grid_to_nn, M_nn_to_grid, posterior_cov, CR_area, CI_area,
        (f_mean, k_mean)]`` summarizing Mahalanobis distances, posterior
        covariance, grid credible-region area, approximate rectangular CI area,
        and posterior means. 

    Notes
    -----
    The posterior is reconstructed directly from the NRMSE map assuming a
    Gaussian noise model. The Mahalanobis distance between NN and grid-based
    posteriors is reported via console ``print`` statements for quick checks.
    """

    loc_dict_res = loc_dict_res or au_config.loc_dict_res
    _df = _df or (
        infer.infer_config.fb_scale_fact
        if is_cest
        else infer.infer_config.fc_scale_fact
    ) / loc_dict_res
    _dk = _dk or (
        infer.infer_config.kb_scale_fact
        if is_cest
        else infer.infer_config.kc_scale_fact
    ) / loc_dict_res

    diff = np.array((f_est_nn - f_best_dotprod, k_est_nn - k_best_dotprod))
    mahalanobis_nn_to_grid = np.sqrt(diff.T @ np.linalg.inv(cov_nnpred_scaled.T) @ diff)    
    
    first_time = False
    if not visualize:
        fig = ax = ax_marg_x = ax_marg_y = Null()
    elif not do_marginals:
        fig = fig or plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = fig or plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(19, 20, top=1.03)
        existing_axes = fig.get_axes()
        if len(existing_axes) >= 3:
            ax = existing_axes[0]
            ax_marg_x = existing_axes[1]
            ax_marg_y = existing_axes[2]
        else:
            ax = fig.add_subplot(gs[2:, 2:-1])
            ax_marg_x = fig.add_subplot(gs[:2, 2:-1], sharex=ax)
            ax_marg_y = fig.add_subplot(gs[2:, :2], sharey=ax)
            first_time = True
    
    sigma_est = au_config.known_sigma or np.min(nrmse) / np.sqrt(n_meas)
    log_likelihood = - (nrmse**2 - np.min(nrmse) ** 2) / (2 * sigma_est**2)
    pdf = np.exp(log_likelihood) / np.sum(np.exp(log_likelihood))

    extent = [_df, _df * 100 * (loc_dict_res+1), _dk, _dk * (loc_dict_res+1)]
    imsh = ax.imshow(
        pdf.T / np.max(pdf),
        origin="lower",
        aspect="auto",
        alpha=imshow_alpha,        
        extent=extent,
        cmap="hot",
    )

    if do_cbar:
        if not do_marginals:
            cbar = plt.colorbar(imsh)
        else:
            ax_cbar = fig.add_subplot(gs[2:, 19] if visualize else None)
            cbar = fig.colorbar(imsh, cax=ax_cbar)
        cbar.set_label("Full-grid posterior PDF (a.u.)", fontsize=fontsize - 4)

    if show_NN:
        ellipse = Ellipse(
            xy=(f_est_nn, k_est_nn),
            width=width,
            height=height,
            angle=angle - 90,
            edgecolor="c",
            facecolor="none",
            linewidth=linewidth,
        )
        ax.add_patch(ellipse)
        nn_point = ax.plot(
            [f_est_nn, f_est_nn],
            [k_est_nn, k_est_nn],
            "X",
            color="c",
            markersize=markersize,
        )
    else:
        ellipse = None
        nn_point = (None,)

    ax.set_xlabel('f$_{s}$ (%)' if is_cest else 'f$_{ss}$ (%)', fontsize=fontsize)
    ax.set_ylabel('k$_{sw}$ (s$^{-1}$)' if is_cest else 'k$_{ssw}$ (s$^{-1}$)', fontsize=fontsize)
    ax.set_xlim(0, 100*infer.infer_config.fb_scale_fact if is_cest else 100*infer.infer_config.fc_scale_fact)
    ax.set_ylim(0, (max_k_cest or infer.infer_config.kb_scale_fact) if is_cest else infer.infer_config.kc_scale_fact)
    
    ref_point = ax.plot(
        [f_best_dotprod, f_best_dotprod],
        [k_best_dotprod, k_best_dotprod],
        "X",
        color="k",
        markersize=markersize,
    )
    if gt is not None and au_config.known_gt:
        gt_point = ax.plot(
            [gt[0], gt[0]],
            [gt[1], gt[1]],
            "H",
            color="blue",
            markersize=markersize,
        )
        ref_point = gt_point
    nrmse_over_min = np.arange(1.005, 2, 0.005)
    posterior_cdf = [np.sum(pdf[nrmse < th * np.min(nrmse)]) for th in nrmse_over_min]
    nrmse_95_th = np.min(nrmse) * nrmse_over_min[np.argmin(np.abs(np.array(posterior_cdf) - 0.95))]
    print('nrmse_95_th', nrmse_95_th, 'cdf95', posterior_cdf[np.argmin(np.abs(np.array(posterior_cdf) - 0.95))])
    ax.contour(
        nrmse.T,
        levels=[nrmse_95_th],
        colors=["magenta"],
        linewidths=[linewidth],
        extent=extent
    )
    CR_area = np.sum(nrmse < nrmse_95_th)

    if visualize and do_legend:
        handles = [ref_point[0], mpatches.Patch(color="m")]
        labels = [r"Ref. MAP $\hat{\theta}$", "Ref. 95% CR"]  #["Grid Bayesian MAP", "Grid Bayesian 95% CR"]
        if show_NN and ellipse is not None:
            handles += [nn_point[0], ellipse]
            labels += [r"NN MAP $\hat{\theta}$", r"NN 95% CR"]                       
            #[r"NN MAP $\hat{\mu}_\theta$", r"NN CR $\leftarrow\hat{\Sigma}_\theta$"]
        ax.legend(handles, labels, loc="center right", fontsize=8)

    df_grid = 100 * _df * np.arange(0, loc_dict_res)[:, None]
    dk_grid = _dk * np.arange(0, loc_dict_res)[None, :]
    marg_f = np.sum(pdf, axis=1)
    marg_k = np.sum(pdf, axis=0)
    f_mean = np.sum(marg_f * df_grid[:, 0])
    k_mean = np.sum(marg_k * dk_grid[0, :])
    marg_f_cdf = np.cumsum(marg_f)
    marg_k_cdf = np.cumsum(marg_k)
    f_975 = _df * 100 * np.searchsorted(marg_f_cdf, 0.975)
    f_025 = _df * 100 * np.searchsorted(marg_f_cdf, 0.025)
    k_975 = _dk * np.searchsorted(marg_k_cdf, 0.975)
    k_025 = _dk * np.searchsorted(marg_k_cdf, 0.025)
    CI_area = (k_975 - k_025) * (f_975 - f_025) / (100 * _df * _dk)

    if do_marginals and visualize:
        ax_marg_x.fill_between(
            x=100 * _df * np.arange(0, loc_dict_res),
            y1=0,
            y2=marg_f / np.max(marg_f),
            color=marg_color,
            alpha=0.5,
        )
        ax_marg_x.set_ylim(0, 1)
        marg_k_norm = marg_k / np.max(marg_k) if np.max(marg_k) else marg_k
        ax_marg_y.fill_betweenx(
            y=_dk * np.arange(0, loc_dict_res),
            x1=0,
            x2=marg_k_norm,
            color=marg_color,
            alpha=0.5,
        )
        if first_time:
            ax_marg_y.set_xlim(0, 1)
            ax_marg_y.invert_xaxis()
        ax_marg_y.axis("off")
        ax_marg_x.axis("off")
        ax.plot((f_025, f_025), (k_975, _dk*loc_dict_res), '--', color='pink', alpha=0.2, linewidth=0.7)
        ax.plot((f_975, f_975), (k_025, _dk*loc_dict_res), '--', color='pink', alpha=0.2, linewidth=0.7)
        ax.plot((0, f_975), (k_025, k_025), '--', color='pink', alpha=0.2, linewidth=0.7)
        ax.plot((0, f_025), (k_975, k_975), '--', color='pink', alpha=0.2, linewidth=0.7)   
        
    delta = np.stack(
        (
            df_grid.repeat(loc_dict_res, axis=1) - f_mean,
            dk_grid.repeat(loc_dict_res, axis=0) - k_mean,
        ),
        axis=-1,
    )
    posterior_cov = (pdf[..., None, None] * delta[..., None] @ delta[..., None, :]).sum(axis=(0, 1))
    mahal_grid_to_nn = np.sqrt(diff.T @ np.linalg.inv(posterior_cov) @ diff)

    if visualize and show_text:
        text = (
            "Mahalanobis distances:\n"
            + r"$M(\hat{\theta}_{NN}, P_{FG})$="
            + f"{mahal_grid_to_nn:.1f}"
            + "; "
            + r"$M(\hat{\theta}_{FG}, P_{NN})$="
            + f"{mahalanobis_nn_to_grid:.1f}"
        )
        ax.text(0.1, 5, text, color="white")
        if xlim and ylim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        plt.sca(ax)

    return [
        mahal_grid_to_nn,
        mahalanobis_nn_to_grid,
        posterior_cov,
        CR_area,
        CI_area,
        (f_mean, k_mean),
    ]


def plot_two_points(
    mt_tissue_params_est,
    data_feed_mt,
    points,
    mt_sim_mode,
    sli: int = 0,
    seq_df=None,
    seq_name: str = "mt",
):
    """Compare posterior heatmaps for a list of voxel coordinates."""

    (
        f_est,
        k_est,
        cov_nnpred_scale,
        _,
        _,
        height,
        width,
        angle,
    ) = extract_nn_estimate_of_posterior(mt_tissue_params_est, data_feed_mt.shape, False)
    ax = None
    for _x, _y in points:
        (
            f_best,
            k_best,
            dict_signal,
            nrmse,
            _df,
            _dk,
        ) = get_nrmse_grid(
            None,
            _x,
            _y,
            sli,
            data_feed_mt,
            mt_tissue_params_est,
            data_feed_cest=None,
            seq_name=seq_name,
            seq_df=seq_df,
            mt_sim_mode=mt_sim_mode,
            max_k_mt=70,
        )
        _, _, posterior_cov, _, _, (f_mean, k_mean) = viz_posteriors(
            f_est[_x, sli, _y],
            k_est[_x, sli, _y],
            cov_nnpred_scale[_x, sli, _y],
            width[_x, sli, _y],
            height[_x, sli, _y],
            angle[_x, sli, _y],
            f_best,
            k_best,
            nrmse,
            _df,
            _dk,
            is_cest=False,
            fontsize=14,
            do_marginals=False,
            figsize=[7, 5],
            show_text=False,
            show_NN=True,
            ax=ax,
            imshow_alpha=0.5,
            do_legend=False,
        )
        ax = plt.gca()
        from .metrics import get_cross_mahas  # local import avoids circularity

        cross_mahas1, cross_mahas2 = get_cross_mahas(
            f_est[_x, sli, _y],
            k_est[_x, sli, _y],
            cov_nnpred_scale[_x, sli, _y],
            f_mean,
            k_mean,
            posterior_cov,
            samples=30,
        )
        ax.text(
            f_mean * 1.2,
            k_mean,
            "Mahalanobis distances: \n"
            + r"$\mathcal{M}(\theta'\sim P_{NN}, P_{FG})$ = "
            + rf"{np.mean(cross_mahas1):.1f}$\pm${np.std(cross_mahas1):.1f}"
            + "\n"
            + r"$\mathcal{M}(\theta'\sim P_{FG}, P_{NN})$ = "
            + rf"{np.mean(cross_mahas2):.1f}$\pm${np.std(cross_mahas2):.1f}",
            color="w",
        )


def _compute_posterior_summary(nrmse, _df, _dk, is_cest):
    """Summarize posterior statistics derived from an NRMSE grid."""

    loc_dict_res = nrmse.shape[0]
    (
        pdf,
        nrmse_level,
        CR_area,
        f_mean,
        f_025,
        f_975,
        k_mean,
        k_025,
        k_975,
        _, __
    ) = nrmse_to_CIs(
        nrmse,
        df=_df,
        dk=_dk,
        is_cest=is_cest,
        loc_dict_res=loc_dict_res,
    )
    pdf_max = float(np.max(pdf))
    if pdf_max == 0:
        pdf_max = 1.0

    f_axis = 100 * _df * np.arange(loc_dict_res)
    k_axis = _dk * np.arange(nrmse.shape[1])
    marg_f = np.sum(pdf, axis=1)
    marg_k = np.sum(pdf, axis=0)
    marg_f_norm = marg_f / np.max(marg_f) if np.max(marg_f) else marg_f
    marg_k_norm = marg_k / np.max(marg_k) if np.max(marg_k) else marg_k

    delta_f = f_axis[:, None] - f_mean
    delta_k = k_axis[None, :] - k_mean
    cov_ff = np.sum(pdf * delta_f**2)
    cov_fk = np.sum(pdf * delta_f * delta_k)
    cov_kk = np.sum(pdf * delta_k**2)
    posterior_cov = np.array([[cov_ff, cov_fk], [cov_fk, cov_kk]])

    return {
        "pdf": pdf,
        "pdf_max": pdf_max,
        "nrmse": nrmse,
        "nrmse_level": nrmse_level,
        "CR_area": CR_area,
        "f_mean": f_mean,
        "k_mean": k_mean,
        "f_ci": (f_025, f_975),
        "k_ci": (k_025, k_975),
        "f_axis": f_axis,
        "k_axis": k_axis,
        "f_max": 100 * _df * loc_dict_res,
        "k_max": _dk * nrmse.shape[1],
        "marg_f_norm": marg_f_norm,
        "marg_k_norm": marg_k_norm,
        "posterior_cov": posterior_cov,
    }
    
    
def juxtapose_two_voxels_posteriors(
    data_feed_mt,
    tissue_params_est,
    shape,
    pointA,
    pointB,
    do_legend=True,
    sli: int = 0,
    is_cest: bool = False,
    data_feed_cest=None,
    seq_name=None,
    fig=None,
    fontsize: int = 12,
    max_k_cest=None,
    seq_df=None,
    annotate_cross_mahas=True,
    letters=['A', 'B'],
    colors=("gold", "white"), xytext=(0.63, 0.93)
):
    """Visualize two voxels side-by-side to compare posterior behaviour."""    
    seq_name = seq_name or ("mt" if not is_cest else "amide")
    fig = fig or plt.figure(figsize=(10, 5))
    (
        f_est,
        k_est,
        cov_nnpred_scale,
        _,
        _,
        height,
        width,
        angle,
    ) = extract_nn_estimate_of_posterior(tissue_params_est, shape, is_cest=is_cest)

    ax = None
    arrow_tips: list[tuple[float, float]] = []
    for jj, (_x, _y) in enumerate([pointA, pointB]):
        (
            f_best,
            k_best,
            dict_signal,
            nrmse,
            _df,
            _dk,
        ) = get_nrmse_grid(
            None,
            _x,
            _y,
            sli,
            data_feed_mt,
            tissue_params_est,
            max_k_cest=max_k_cest,
            data_feed_cest=data_feed_cest,
            seq_name=seq_name,
            seq_df=seq_df,
        )
        _, _, posterior_cov, _, _, (f_mean, k_mean) = viz_posteriors(
            f_est[_x, sli, _y],
            k_est[_x, sli, _y],
            cov_nnpred_scale[_x, sli, _y],
            width[_x, sli, _y],
            height[_x, sli, _y],
            angle[_x, sli, _y],
            f_best,
            k_best,
            nrmse,
            _df,
            _dk,
            is_cest=is_cest,
            fontsize=fontsize,
            do_cbar=(jj == 0),
            do_marginals=True,
            figsize=[7, 5],
            show_text=False,
            show_NN=True,
            imshow_alpha=0.5,
            do_legend=do_legend if jj == 0 else False,
            fig=fig,
            ax=ax,
            marg_color="xkcd:pinkish grey" if jj == 1 else "#C8C8A2",
            max_k_cest=max_k_cest
        )
        ax = fig.get_axes()[0]
        from .metrics import get_cross_mahas  # local import avoids circularity

        if annotate_cross_mahas:
            cross_mahas1, cross_mahas2 = get_cross_mahas(
                f_est[_x, sli, _y],
                k_est[_x, sli, _y],
                cov_nnpred_scale[_x, sli, _y],
                f_mean,
                k_mean,
                posterior_cov,
                samples=30,
            )
            text = (
                #r"$M_1=\mathcal{M}(\theta'\sim P_{NN}, P_{Ref})$="
                r"M$_1$="
                + rf"{np.mean(cross_mahas1):.1f}$\pm${np.std(cross_mahas1):.1f}"
                + "\n"
                #+ r"$M_2=\mathcal{M}(\theta'\sim P_{Ref}, P_{NN})$="
                + r"M$_2$="
                + rf"{np.mean(cross_mahas2):.1f}$\pm${np.std(cross_mahas2):.1f}"
            )
            va = "top" if jj else "bottom"
            arrow_tip = (f_est[_x, sli, _y] * 1.1, k_est[_x, sli, _y] * (0.7 + 0.4 * jj))            
            ax.annotate(
                letters[jj], # text,
                xy=arrow_tip,
                xycoords="data",
                xytext=xytext if jj else (xytext[0], 1-xytext[1]), # text_pos,
                textcoords="axes fraction",
                horizontalalignment="right",
                multialignment="left",
                verticalalignment=va,
                fontsize=fontsize+2, #-2, # - 3,
                color=colors[jj],
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3, rad=-0.2",
                    relpos=(0, 0.5), #(0.5, 0) if jj else (0, 0.5),
                    color="white",
                    shrinkB=5,
                    shrinkA=5,
                ),
            )            
            arrow_tips.append(arrow_tip)
            ax.text(*((1.0, 0.98) if jj else (1.0, 0.02)), text, color='white', fontsize=fontsize-2, 
                    transform=ax.transAxes, horizontalalignment="right", verticalalignment=va) #"bottom" if jj else "top"))
    return ax, arrow_tips


def single_voxel_detailed_viz(
    _x,
    _y,
    data_feed_mt,
    mt_tissue_params_est,
    sli: int = 0,
):
    """Detailed posterior inspection for a single voxel."""

    (
        f_est,
        k_est,
        cov_nnpred_scale,
        _,
        _,
        height,
        width,
        angle,
    ) = extract_nn_estimate_of_posterior(mt_tissue_params_est, data_feed_mt.shape, False)

    (
        f_best,
        k_best,
        dict_signal,
        nrmse,
        _df,
        _dk,
    ) = get_nrmse_grid(
        None,
        _x,
        _y,
        sli,
        data_feed_mt,
        mt_tissue_params_est,
        data_feed_cest=None,
        seq_name="mt",
        mt_sim_mode="expm_bmmat",
    )

    _, _, posterior_cov, _, _, (f_mean, k_mean) = viz_posteriors(
        f_est[_x, sli, _y],
        k_est[_x, sli, _y],
        cov_nnpred_scale[_x, sli, _y],
        width[_x, sli, _y],
        height[_x, sli, _y],
        angle[_x, sli, _y],
        f_best,
        k_best,
        nrmse,
        _df,
        _dk,
        is_cest=False,
        visualize=True,
        xlim=[0, 20],
        ylim=[0, 70],
        fontsize=14,
        do_marginals=True,
        figsize=[7, 5],
        show_text=True,
        show_NN=True,
    )

    posterior_mu = np.array([f_mean, k_mean])
    samples_from_exact = np.random.multivariate_normal(posterior_mu, posterior_cov, size=500)
    nn_mu = np.array([f_est[_x, sli, _y], k_est[_x, sli, _y]])
    samples_from_nn = np.random.multivariate_normal(nn_mu, cov_nnpred_scale[_x, sli, _y], size=500)

    plt.figure()
    plt.xlim(0, 20)
    plt.ylim(0, 70)
    plt.scatter(
        samples_from_exact[:, 0],
        samples_from_exact[:, 1],
        color="magenta",
        s=20,
        alpha=0.1,
        label="Samples from exact posterior",
    )
    plt.scatter(
        samples_from_nn[:, 0],
        samples_from_nn[:, 1],
        color="cyan",
        s=20,
        alpha=0.1,
        label="Samples from NN posterior",
    )

    self_mahas1 = np.sqrt(
        np.sum(
            (samples_from_exact.T - posterior_mu[:, None])
            * (np.linalg.inv(posterior_cov) @ (samples_from_exact.T - posterior_mu[:, None])),
            axis=0,
        )
    )
    self_mahas2 = np.sqrt(
        np.sum(
            (samples_from_nn.T - nn_mu[:, None])
            * (np.linalg.inv(cov_nnpred_scale[_x, sli, _y]) @ (samples_from_nn.T - nn_mu[:, None])),
            axis=0,
        )
    )
    from .metrics import get_cross_mahas  # local import avoids circularity

    cross_mahas1, cross_mahas2 = get_cross_mahas(
        f_est[_x, sli, _y],
        k_est[_x, sli, _y],
        cov_nnpred_scale[_x, sli, _y],
        f_mean,
        k_mean,
        posterior_cov,
        samples=500,
    )
    plt.figure()
    for data, label in (
        (self_mahas1, "Exact posterior samples TO SELF"),
        (self_mahas2, "NN posterior samples TO SELF"),
        (cross_mahas1, "Exact samples to NN posterior"),
        (cross_mahas2, "NN samples to exact posterior"),
    ):
        plt.hist(data, bins=np.arange(0, 5, 0.2), histtype="step", linewidth=2, label=label)
    plt.legend()


__all__ = [
    "cbarhist",
    "plot_map",
    "plot_CI_maps",
    "viz_cov",
    "plot_ellipse",
    "plot_progressive_pdfs",
    "simple_error_map",
    "demonstrate_ellipses",
    "viz_posteriors",
    "plot_two_points",
    "juxtapose_two_voxels_posteriors",
    "single_voxel_detailed_viz",
]
