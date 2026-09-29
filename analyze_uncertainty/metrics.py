"""Analysis utilities extracted from ``analyze_uncertainty``.

This module provides a direct copy of the supporting helpers so new code can
depend on ``_analyze_uncertainty.metrics`` without referencing the monolithic
legacy module.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from .core import extract_nn_estimate_of_posterior, get_nrmse_grid, au_config
from . import core
from .viz import viz_posteriors
from . import helpers


def get_cross_mahas(
    _f_est,
    _k_est,
    _cov_nnpred_scaled,
    f_mean_posterior,
    k_mean_posterior,
    posterior_cov,
    samples: int = 10,
):
    """Sample posteriors and compute cross Mahalanobis distance:
        1) from (samples of) NN-predicted posterior to brute-force posterior
        2) from (samples of) brute-force posterior to NN-predicted posterior
    """
    posterior_mu = np.array([f_mean_posterior, k_mean_posterior])
    samples_from_exact = np.random.multivariate_normal(
        posterior_mu, posterior_cov, size=samples
    )
    nn_mu = np.array([_f_est, _k_est])
    
    # Work around some rare numerical issues by "forcing absolute symmetry"
    _cov_nnpred_scaled = (_cov_nnpred_scaled + _cov_nnpred_scaled.T) / 2
    posterior_cov = (posterior_cov + posterior_cov.T) / 2
    
    samples_from_nn = np.random.multivariate_normal(
        nn_mu, _cov_nnpred_scaled, size=samples
    )
    cross_mahas1 = np.sqrt(
        np.sum(
            (samples_from_nn.T - posterior_mu[:, None])
            * (
                np.linalg.inv(posterior_cov)
                @ (samples_from_nn.T - posterior_mu[:, None])
            ),
            axis=0,
        )
    )
    cross_mahas2 = np.sqrt(
        np.sum(
            (samples_from_exact.T - nn_mu[:, None])
            * (
                np.linalg.inv(_cov_nnpred_scaled)
                @ (samples_from_exact.T - nn_mu[:, None])
            ),
            axis=0,
        )
    )
    return cross_mahas1, cross_mahas2


def summarize_mahas(
    mahas1,
    mahas2,
    good_xy_points=None,
    samples: int = 10,
    th: float = 4,
    bw: float = 0.5,
    average_per_pixel: bool = False,
    median_per_pixel: bool = True,
    fig=None,
    figname: str | None = None,
    do_legend: bool = True,
    do_ylabel: bool = True,
):
    """Summarize Mahalanobis distance distributions across voxels."""
    if mahas1 is None and mahas2 is None:
        raise ValueError("At least one of mahas1 or mahas2 must be provided")

    fig = fig or plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.sca(ax)

    if median_per_pixel:
        if good_xy_points is not None:
            final_mahas1 = (
                np.median(np.array(mahas1).reshape(len(good_xy_points), -1), axis=1)
                if mahas1 is not None
                else None
            )
            final_mahas2 = (
                np.median(np.array(mahas2).reshape(len(good_xy_points), -1), axis=1)
                if mahas2 is not None
                else None
            )
        else:
            final_mahas1 = (
                np.median(np.array(mahas1).reshape(-1, samples), axis=1)
                if mahas1 is not None
                else None
            )
            final_mahas2 = (
                np.median(np.array(mahas2).reshape(-1, samples), axis=1)
                if mahas2 is not None
                else None
            )
    elif average_per_pixel:
        if good_xy_points is not None:
            final_mahas1 = (
                np.array(mahas1).reshape(len(good_xy_points), -1).mean(axis=1)
                if mahas1 is not None
                else None
            )
            final_mahas2 = (
                np.array(mahas2).reshape(len(good_xy_points), -1).mean(axis=1)
                if mahas2 is not None
                else None
            )
        else:
            final_mahas1 = (
                np.array(mahas1).reshape(-1, samples).mean(axis=1)
                if mahas1 is not None
                else None
            )
            final_mahas2 = (
                np.array(mahas2).reshape(-1, samples).mean(axis=1)
                if mahas2 is not None
                else None
            )
    else:
        final_mahas1 = np.array(mahas1) if mahas1 is not None else None
        final_mahas2 = np.array(mahas2) if mahas2 is not None else None

    bins = np.arange(bw / 2, th + bw + bw / 2, bw)
    series = []
    if final_mahas1 is not None:
        series.append(
            (
                np.array(final_mahas1),
                "c",
                r"M$_1$", #: $\mathcal{M}(\theta'\sim P_{NN},P_{Ref})$",
                f"M$_1$ > {th}",
            )
        )
    if final_mahas2 is not None:
        series.append(
            (
                np.array(final_mahas2),
                "m",
                r"M$_2$",#: $\mathcal{M}(\theta'\sim P_{Ref},P_{NN})$",
                f"M$_2$ > {th}",
            )
        )

    if len(series) == 1:
        bin_offsets = [0.5 * bw]
        outlier_xs = [th + 2.25 * bw]
    else:
        bin_offsets = [0.27 * bw, 0.73 * bw]
        outlier_xs = [th + 2.5 * bw, th + 2 * bw]

    for idx, (values, color, label_main, label_outlier) in enumerate(series):
        total_points = len(values)
        counts, _ = np.histogram(values, bins=bins)
        counts = counts / total_points

        plt.bar(
            bins[:-1] + bin_offsets[idx],
            counts,
            width=0.45 * bw,
            color=color,
            label=label_main,
            alpha=0.5,
        )
        plt.bar(
            [outlier_xs[idx]],
            [np.sum(values > th) / total_points],
            edgecolor="k",
            color=color,
            label=label_outlier,
            width=0.45 * bw,
        )

    plt.gca().set_xticks(np.arange(0, th + 1, 1))
    plt.xlabel('Mahalanobis distance')
    if do_ylabel:
        plt.ylabel("Percentage of voxels in bin")
    _bins = np.arange(0, th + 0.1, 0.1)
    bin_centers = (_bins[:-1] + _bins[1:]) / 2
    r = bin_centers
    pdf = 2 * r * chi2.pdf(r**2, df=2)
    scale = (bw / 0.1) / np.sum(pdf)
    plt.plot(
        bin_centers,
        pdf * scale,
        color="g",
        linewidth=3,
        label=r"M$^{ideal}$" #: $\mathcal{M}(\theta'\sim P,P)$"
        # "Ideal curve:\n Mahalanobis distances of samples\n from distribution to itself",
    )
    if do_legend:
        plt.legend(fontsize=10, loc="upper right")
    plt.gca().set_yticklabels([f"{x:.0f}%" for x in plt.gca().get_yticks() * 100])
    plt.grid()
    if figname:
        plt.savefig(figname)
    return ax


def plot_CI_intersect(
    best_dotprod,
    nn_vals,
    grid_sigma,
    nn_sigma,
    pa_min,
    pa_max,
    dpa,
    ax,
    xlabel,
    ylabel,
):
    """Visualize CI overlap for one parameter against its reference."""

    order = np.argsort(best_dotprod)
    lookup = np.searchsorted(np.sort(best_dotprod), np.arange(pa_min, pa_max, dpa)) - 1
    best_ordered = np.array(best_dotprod)[order][lookup]
    nn_ordered = nn_vals[order][lookup]
    nn_sigma_ordered = nn_sigma[order][lookup]
    grid_sigma_ordered = grid_sigma[order][lookup]

    ax.plot(best_ordered, nn_ordered - best_ordered, "b.")
    span = pa_max - pa_min
    for jj in range(len(nn_ordered)):
        ax.plot(
            [best_ordered[jj], best_ordered[jj]],
            [
                nn_ordered[jj] - best_ordered[jj] - 2 * nn_sigma_ordered[jj],
                nn_ordered[jj] - best_ordered[jj] + 2 * nn_sigma_ordered[jj],
            ],
            "b-",
            alpha=0.4,
        )
        ax.plot(
            [best_ordered[jj] + 0.01 * span, best_ordered[jj] + 0.01 * span],
            [-2 * grid_sigma_ordered[jj], 2 * grid_sigma_ordered[jj]],
            "k-",
            alpha=0.4,
        )

    ax.legend(
        [
            r"$\hat{\theta}_x^{\mathrm{(NN)}} - \hat{\theta}_x^{\mathrm{(ref)}}$",
            r"$CI^{\mathrm{(NN)}}:\ \pm\,2\,\hat{\sigma}_{\theta_x}^{\mathrm{(NN)}}$",
            r"$CI^{\mathrm{(ref)}}:\ \pm\,2\,\hat{\sigma}_{\theta_x}^{\mathrm{(ref)}}$",
        ],
        fontsize=8,
        loc="center right",
    )
    ax.plot([pa_min, pa_max], [0, 0], "k-", linewidth=2, alpha=0.3)
    ax.set_ylim(-span / 2, span / 2)
    ax.set_xlim(pa_min, pa_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_fk_CI_intersect(
    f_best_dotprod,
    k_best_dotprod,
    cov_dotprod,
    f_nn,
    k_nn,
    cov_nn,
    is_mt,
    fig=None,
    fmin=None,
    fmax=None,
    kmin=None,
    kmax=None,
    df=None,
    dk=None,
):
    """Wrapper that draws CI intersection plots for f and k parameters."""

    fig = fig or plt.figure(figsize=(12, 5))
    axes = fig.subplots(2, 1)
    fmin = fmin or (2 if is_mt else 0.03)
    fmax = fmax or (27 if is_mt else 0.4)
    kmin = kmin or (4 if is_mt else 0)
    kmax = kmax or (80 if is_mt else 700)
    df = df or (0.5 if is_mt else 0.007)
    dk = dk or (0.5 if is_mt else 10)

    f_grid_sigma = np.array([np.sqrt(cov[0][0]) for cov in cov_dotprod])
    f_nn_sigma = np.array([np.sqrt(cov[0][0]) for cov in cov_nn])
    k_grid_sigma = np.array([np.sqrt(cov[1][1]) for cov in cov_dotprod])
    k_nn_sigma = np.array([np.sqrt(cov[1][1]) for cov in cov_nn])

    plot_CI_intersect(
        f_best_dotprod,
        f_nn,
        f_grid_sigma,
        f_nn_sigma,
        fmin,
        fmax,
        df,
        axes[0],
        xlabel=r"$\hat{f}_{ss}\ (\%)$" if is_mt else r"$\hat{f}_{s}\ (\%)$",
        ylabel=r"$ \Delta \hat{f}_{ss}\ (\%)$" if is_mt else r"$ \Delta \hat{f}_{s}\ (\%)$",
    )
    plot_CI_intersect(
        k_best_dotprod,
        k_nn,
        k_grid_sigma,
        k_nn_sigma,
        kmin,
        kmax,
        dk,
        axes[1],
        xlabel=r"$\hat{k}_{ssw}\ (s^{-1})$" if is_mt else r"$\hat{k}_{s}\ (s^{-1})$",
        ylabel=r"$ \Delta \hat{k}_{ssw}\ (s^{-1})$" if is_mt else r"$ \Delta \hat{k}_{s}\ (s^{-1})$",
    )


def analyze_CI_intersections(
    f_est,
    f_best_dotprod,
    k_est,
    k_best_dotprod,
    cov_nnpred_scaled,
    posterior_cov,
    good_xy_points,
    fmin=None,
    fmax=None,
    kmin=None,
    kmax=None,
    sli=0,
    is_mt=True,
    figsize=(10, 8),
    result_figure_path=None,
    df=0.4,
    dk=0.4,
    fig=None,
):
    """Quantify how NN and reference CIs overlap across voxels."""

    fig = fig or plt.figure(figsize=figsize)
    ax = fig.add_subplot(3, 1, 1)
    plt.sca(ax)
    f_nn = np.array([f_est[_x, sli, _y] for (_x, _y) in good_xy_points])
    f_nn_sigma = np.array([
        np.sqrt(cov_nnpred_scaled[_x, sli, _y][0][0]) for (_x, _y) in good_xy_points
    ])
    f_grid_sigma = np.array([np.sqrt(grid_cov[0][0]) for grid_cov in posterior_cov])
    f_best = np.array(f_best_dotprod, copy=True)

    for nsigmas in (1, 2, 3, 4):
        crosses_zero = np.logical_and(
            f_nn - f_best - nsigmas * f_nn_sigma < 0,
            f_nn - f_best + nsigmas * f_nn_sigma > 0,
        )
        if nsigmas == 2:
            f_perc_nn_covers = 100 * np.sum(crosses_zero) / len(crosses_zero)
        print(
            f"Fraction of points with {nsigmas}-sigma CI crossing zero: "
            f"{100 * np.sum(crosses_zero) / len(crosses_zero):.1f}% out of {len(crosses_zero)}"
        )
        overlaps = np.logical_and(
            f_nn - f_best - nsigmas * f_nn_sigma - nsigmas * f_grid_sigma < 0,
            f_nn - f_best + nsigmas * f_nn_sigma + nsigmas * f_grid_sigma > 0,
        )
        if nsigmas == 2:
            f_perc_intersect = 100 * np.sum(overlaps) / len(overlaps)
        print(
            f"Fraction of points with {nsigmas}-sigma have CI of the 2 methods intersecting: "
            f"{100 * np.sum(overlaps) / len(overlaps):.1f}% out of {len(overlaps)}"
        )

    order = np.argsort(f_best)
    fmin = fmin or (2 if is_mt else 0.01)
    fmax = fmax or (25 if is_mt else 0.5)
    lookup = np.searchsorted(np.sort(f_best), np.arange(fmin, fmax, df)) - 1
    f_best = np.array(f_best)[order][lookup]
    f_nn = f_nn[order][lookup]
    f_nn_sigma = f_nn_sigma[order][lookup]
    f_grid_sigma = f_grid_sigma[order][lookup]

    ax.plot(f_best, f_nn - f_best, "b.")
    span_f = fmax - fmin
    for jj in range(len(f_nn)):
        ax.plot(
            [f_best[jj], f_best[jj]],
            [f_nn[jj] - f_best[jj] - 2 * f_nn_sigma[jj], f_nn[jj] - f_best[jj] + 2 * f_nn_sigma[jj]],
            "b-",
            alpha=0.4,
        )
        ax.plot(
            [f_best[jj] + 0.01 * span_f, f_best[jj] + 0.01 * span_f],
            [-2 * f_grid_sigma[jj], 2 * f_grid_sigma[jj]],
            "k-",
            alpha=0.4,
        )
    ax.legend(["Estimates' discrepancy", "Confidence Intervals"])
    ax.plot([fmin, fmax], [0, 0], "k-", linewidth=2, alpha=0.3)
    ax.set_ylim(-span_f / 2, span_f / 2)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel(r"$\hat{f}_{ss}\ (\%)$" if is_mt else r"$\hat{f}_{s}\ (\%)$")
    ax.set_ylabel(r"$ \Delta \hat{f}_{ss}\ (\%)$" if is_mt else r"$ \Delta \hat{f}_{s}\ (\%)$")

    ax = fig.add_subplot(3, 1, 2)
    plt.sca(ax)
    k_nn = np.array([k_est[_x, sli, _y] for (_x, _y) in good_xy_points])
    k_nn_sigma = np.array([
        np.sqrt(cov_nnpred_scaled[_x, sli, _y][1][1]) for (_x, _y) in good_xy_points
    ])
    k_grid_sigma = np.array([np.sqrt(grid_cov[1][1]) for grid_cov in posterior_cov])
    k_best = np.array(k_best_dotprod, copy=True)

    for nsigmas in (1, 2, 3, 4):
        crosses_zero = np.logical_and(
            k_nn - k_best - nsigmas * k_nn_sigma < 0,
            k_nn - k_best + nsigmas * k_nn_sigma > 0,
        )
        if nsigmas == 2:
            k_perc_nn_covers = 100 * np.sum(crosses_zero) / len(crosses_zero)
        print(
            f"Fraction of points with {nsigmas}-sigma CI crossing zero: "
            f"{100 * np.sum(crosses_zero) / len(crosses_zero):.1f}% out of {len(crosses_zero)}"
        )
        overlaps = np.logical_and(
            k_nn - k_best - nsigmas * k_nn_sigma - nsigmas * k_grid_sigma < 0,
            k_nn - k_best + nsigmas * k_nn_sigma + nsigmas * k_grid_sigma > 0,
        )
        if nsigmas == 2:
            k_perc_intersect = 100 * np.sum(overlaps) / len(overlaps)
        print(
            f"Fraction of points with {nsigmas}-sigma have CI of the 2 methods overlapping: "
            f"{100 * np.sum(overlaps) / len(overlaps):.1f}% out of {len(overlaps)}"
        )

    order = np.argsort(k_best)
    kmin = kmin or (10 if is_mt else 100)
    kmax = kmax or (60 if is_mt else 600)
    lookup = np.searchsorted(np.sort(k_best), np.arange(kmin, kmax, dk)) - 1
    k_best = np.array(k_best)[order][lookup]
    k_nn = k_nn[order][lookup]
    k_nn_sigma = k_nn_sigma[order][lookup]
    k_grid_sigma = k_grid_sigma[order][lookup]

    ax.plot(k_best, k_nn - k_best, "b.")
    span_k = kmax - kmin
    for jj in range(len(k_nn)):
        ax.plot(
            [k_best[jj], k_best[jj]],
            [k_nn[jj] - k_best[jj] - 2 * k_nn_sigma[jj], k_nn[jj] - k_best[jj] + 2 * k_nn_sigma[jj]],
            "b-",
            alpha=0.4,
        )
        ax.plot(
            [k_best[jj] + 0.01 * span_k, k_best[jj] + 0.01 * span_k],
            [-2 * k_grid_sigma[jj], 2 * k_grid_sigma[jj]],
            "k-",
            alpha=0.4,
        )
    ax.plot([kmin, kmax], [0, 0], "k", linewidth=2, alpha=0.3)
    ax.set_ylim(-span_k, span_k)
    ax.set_xlim(kmin, kmax)
    ax.set_xlabel(r"$\hat{k}_{ssw}\ (s^{-1})$" if is_mt else r"$\hat{k}_{s}\ (s^{-1})$")
    ax.set_ylabel(r"$ \Delta \hat{k}_{ssw}\ (s^{-1})$" if is_mt else r"$ \Delta \hat{k}_{s}\ (s^{-1})$")

    plt.tight_layout()
    if result_figure_path is not None:
        plt.savefig(result_figure_path, bbox_inches="tight", dpi=300)

    return (
        f_perc_intersect,
        k_perc_intersect,
        f_perc_nn_covers,
        k_perc_nn_covers,
    )


def analyze_scatter(
    f_est_nn_l,
    f_best_dotprod_arr,
    k_est_nn_l,
    k_best_dotprod_arr,
    cov_nnpred_scaled_l,
    covs_of_exact_posterior,
    fmin=None,
    fmax=None,
    kmin=None,
    kmax=None,
    colorful_crosses: bool = False,
    to_plot: bool = True,
    is_mt: bool = False,
    figsize=(10, 5),
    result_fig=None,
    result_fig_name: str | None = None,
    alpha: float = 0.2,
):
    """Plot correlations between reference and NN estimates/CIs."""
    from scipy.stats import pearsonr

    result_fig = result_fig or plt.figure(figsize=figsize)
    ((ax1, ax2), (ax3, ax4)) = result_fig.subplots(2, 2)

    fmin = fmin or (2 if is_mt else 0.07)
    fmax = fmax or (25 if is_mt else 0.3)
    kmin = kmin or (10 if is_mt else 100)
    kmax = kmax or (60 if is_mt else 600)

    xx = np.array(f_best_dotprod_arr)
    yy = np.array(f_est_nn_l)
    pearsons_r_f, p_value = pearsonr(xx, yy)
    if to_plot:
        ax1.scatter(xx, yy, alpha=alpha, s=15)
        ax1.set_xlim(fmin, fmax)
        ax1.set_ylim(fmin, fmax)
        ax1.plot([fmin, fmax], [fmin, fmax], "k--")
        ax1.text(
            0.05,
            0.95,
            f"Pearson's r = {pearsons_r_f:.2f}, "
            + (f"p-value={p_value:.2e}" if p_value > 1e-6 else "p-value < 1e-6"),
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
        )

    xx = np.array(k_best_dotprod_arr)
    yy = np.array(k_est_nn_l)
    pearsons_r_k, p_value = pearsonr(xx, yy)
    if to_plot:
        ax2.scatter(xx, yy, alpha=alpha, s=15)
        ax2.set_xlim(kmin, kmax)
        ax2.set_ylim(kmin, kmax)
        ax2.plot([kmin, kmax], [kmin, kmax], "k--")
        ax2.text(
            0.05,
            0.95,
            f"Pearson's r = {pearsons_r_k:.2f}, "
            + (f"p-value={p_value:.2e}" if p_value > 1e-6 else "p-value < 1e-6"),
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
        )
        f_symbol = r"\hat{f}_{ss}" if is_mt else r"\hat{f}_{s}"
        k_symbol = r"\hat{k}_{ssw}" if is_mt else r"\hat{k}_{sw}"
        for ax in (ax1, ax3):
            ax.set_xlabel(f"${f_symbol}$ (%), reference")
            ax.set_ylabel(f"${f_symbol}$ (%), PdVAE")
        for ax in (ax2, ax4):
            ax.set_xlabel(f"${k_symbol}$ (s$^{{-1}}$), reference")
            ax.set_ylabel(f"${k_symbol}$ (s$^{{-1}}$), PdVAE")

    if colorful_crosses:
        for fa, fb, fva, fvb in list(
            zip(
                np.array(f_best_dotprod_arr),
                f_est_nn_l,
                [x[0][0] for x in covs_of_exact_posterior],
                [cov[0][0] for cov in cov_nnpred_scaled_l],
            )
        )[::20]:
            c = np.random.random(3)
            ax3.plot(
                [fa - 2 * np.sqrt(fva), fa + 2 * np.sqrt(fva)],
                [fb, fb],
                color=c,
                linewidth=2,
                alpha=0.2,
            )
            ax3.plot(
                [fa, fa],
                [fb - 2 * np.sqrt(fvb), fb + 2 * np.sqrt(fvb)],
                color=c,
                linewidth=2,
                alpha=0.2,
            )
            ax3.plot([fa, fa], [fb, fb], "s", color=c, markersize=4)
        ax3.set_xlim(fmin, fmax)
        ax3.set_ylim(fmin, fmax)
        ax3.plot([fmin, fmax], [fmin, fmax], "k--")

        for ka, kb, kva, kvb in list(
            zip(
                np.array(k_best_dotprod_arr),
                k_est_nn_l,
                [x[1][1] for x in covs_of_exact_posterior],
                [cov[1][1] for cov in cov_nnpred_scaled_l],
            )
        )[::20]:
            c = np.random.random(3)
            ax4.plot(
                [ka - 2 * np.sqrt(kva), ka + 2 * np.sqrt(kva)],
                [kb, kb],
                color=c,
                linewidth=2,
                alpha=0.4,
            )
            ax4.plot(
                [ka, ka],
                [kb - 2 * np.sqrt(kvb), kb + 2 * np.sqrt(kvb)],
                color=c,
                linewidth=2,
                alpha=0.4,
            )
            ax4.plot([ka, ka], [kb, kb], "s", color=c, markersize=4)
        ax4.set_xlim(kmin, kmax)
        ax4.set_ylim(kmin, kmax)
        ax4.plot([kmin, kmax], [kmin, kmax], "k--")
        pearsons_r_f_ci = np.nan
        pearsons_r_k_ci = np.nan
    else:
        xx = [4 * np.sqrt(x[0][0]) for x in covs_of_exact_posterior]
        yy = [4 * np.sqrt(cov[0][0]) for cov in cov_nnpred_scaled_l]
        pearsons_r_f_ci, p_value = pearsonr(xx, yy)
        if to_plot:
            ax3.scatter(xx, yy, alpha=alpha, s=15)
            ax3.set_xlim(0, fmax / 3)
            ax3.set_ylim(0, fmax / 3)
            ax3.text(
                0.05,
                0.95,
                f"Pearson's r = {pearsons_r_f_ci:.2f}, "
                + (f"p-value={p_value:.2e}" if p_value > 1e-6 else "p-value < 1e-6"),
                transform=ax3.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
            )
        xx = [4 * np.sqrt(x[1][1]) for x in covs_of_exact_posterior]
        yy = [4 * np.sqrt(cov[1][1]) for cov in cov_nnpred_scaled_l]
        pearsons_r_k_ci, p_value = pearsonr(xx, yy)
        if to_plot:
            ax4.scatter(xx, yy, alpha=alpha, s=15)
            ax4.set_xlim(0, kmax)
            ax4.set_ylim(0, kmax)
            ax4.text(
                0.05,
                0.95,
                f"Pearson's r = {pearsons_r_k_ci:.2f}, "
                + (f"p-value={p_value:.2e}" if p_value > 1e-6 else "p-value < 1e-6"),
                transform=ax4.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
            )
            ax3.plot([0, fmax], [0, fmax], "k--")
            ax4.plot([0, kmax], [0, kmax], "k--")
            ax3.set_xlabel(r"$4*\hat{\sigma}_f$ (%), reference")
            ax3.set_ylabel(r"$4*\hat{\sigma}_f$ (%), PdVAE"
            )
            ax4.set_xlabel(r"$4*\hat{\sigma}_k$ (s$^{-1}$), reference")
            ax4.set_ylabel(r"$4*\hat{\sigma}_k$ (s$^{-1}$), PdVAE")

    plt.tight_layout()
    if to_plot and result_fig_name is not None:
        plt.savefig(result_fig_name, bbox_inches="tight", dpi=300)

    return pearsons_r_f, pearsons_r_k, pearsons_r_f_ci, pearsons_r_k_ci


def compute_dist_distances(pdf_1, pdf_2, f_grid=None, k_grid=None):
    """Compute various distances between two 2D distributions.
        Assumes pdf_1 and pdf_2 are defined on the same f_grid and k_grid,
         and are properly normalized to sum(pdf * df * dk) = 1.
        (Default is 'abstract' / 'discreet-PDF' as df=dk=1 unit)
    """
    from scipy.stats import entropy
    # 1. Handle coordinate grids safely
    f_grid = f_grid if f_grid is not None else np.arange(pdf_1.shape[0])
    k_grid = k_grid if k_grid is not None else np.arange(pdf_1.shape[1])    
    df = f_grid[1] - f_grid[0]
    dk = k_grid[1] - k_grid[0]
    delta_area = df * dk

    # 2. Flatten for 1D vector processing
    pdf_1_flat = pdf_1.flatten() + 1e-12
    pdf_2_flat = pdf_2.flatten() + 1e-12

    # 3. CORRECT CONTINUOUS KL DIVERGENCE
    # Since scipy.stats.entropy automatically normalizes inputs internally, 
    # it returns: sum( P_discrete * log(P_discrete / Q_discrete) ).
    # Because pdf_discrete = pdf_continuous * delta_area, the delta_area factors 
    # cancel out completely inside the log and the sum. Do NOT multiply by delta_area!
    kl_1_2 = entropy(pdf_1_flat, pdf_2_flat)
    kl_2_1 = entropy(pdf_2_flat, pdf_1_flat)

    # 4. CORRECT JENSEN-SHANNON DIVERGENCE (Using the midpoint M)
    # Ensure you evaluate this in base-2 (scipy defaults to natural logs/nats)
    # Using base-2 bounds the JS value cleanly between 0 and 1.
    pdf_m_flat = 0.5 * (pdf_1_flat + pdf_2_flat)
    js_divergence = 0.5 * (entropy(pdf_1_flat, pdf_m_flat, base=2) + 
                        entropy(pdf_2_flat, pdf_m_flat, base=2))
    js_distance = np.sqrt(js_divergence)

    # 5. CORRECT HELLINGER DISTANCE
    # Hellinger requires delta_area because it integrates differences of density roots directly.
    # The 0.5 coefficient MUST remain inside the square root.
    hellinger_distance = np.sqrt(
        0.5 * np.sum((np.sqrt(pdf_1_flat) - np.sqrt(pdf_2_flat))**2) * delta_area
    )
    # Bhattacharyya distance
    bhattacharyya_coeff = np.sum(np.sqrt(pdf_1_flat * pdf_2_flat)) * delta_area
    bhattacharyya_coeff = np.clip(bhattacharyya_coeff, 1e-12, 1.0)
    bhattacharyya_distance = -np.log(bhattacharyya_coeff)
    
    return {
        "KL(P1||P2)": kl_1_2,
        "KL(P2||P1)": kl_2_1,
        "Jensen-Shannon Divergence": js_divergence,
        "Hellinger Distance": hellinger_distance,
        "Bhattacharyya Distance": bhattacharyya_distance,
    }


def compare_to_precomputed_exact_posteriors(
    data_feed_mt, good_xy_points, sli,
    f_est, k_est, cov_nnpred_scaled,
    posterior_mus, posterior_covs):
    mahas1, mahas2 = [], []
    for (_x, _y), posterior_mu, posterior_cov in zip(good_xy_points, posterior_mus, posterior_covs):
        f_mean_grid_posterior, k_mean_grid_posterior = posterior_mu
        cross_mahas1, cross_mahas2 = get_cross_mahas(
            f_est[_x, sli, _y],
            k_est[_x, sli, _y],
            cov_nnpred_scaled[_x, sli, _y],
            f_mean_grid_posterior,
            k_mean_grid_posterior,
            posterior_cov
        )  
        mahas1 += cross_mahas1.tolist()
        mahas2 += cross_mahas2.tolist()
    return mahas1, mahas2


def compare_posteriors(
    xy_points: Iterable[Sequence[int]],
    data_feed_mt,
    mt_tissue_params_est,
    cest_tissue_params_est=None,
    data_feed_cest=None,
    is_cest=False,
    mt_sim_mode="expm_bmmat",
    seq_name="mt",
    seq_df=None,
    voxels2sample: int = 10,
    visualize: bool = True,
    sli: int = 0,
    folder_name: str | None = None,
    max_k_cest=None,
    known_sigmas=None
):
    """Contrast NN and brute-force posteriors for selected voxels."""

    good_xy_points: list[list[int]] = []
    mahas1: list[float] = []
    mahas2: list[float] = []
    CR_areas, nnCR_areas = [], []
    timings = []
    exact_min_percs, nn_min_percs = [], []
    means_exact, modes_exact, covs_exact = [], [], []
    distrib_distances_dicts = []    
    cdfs_at_gt = []
    
    if folder_name and visualize:
        os.makedirs(folder_name, exist_ok=True)

    (
        f_est,
        k_est,
        cov_nnpred_scaled,
        _,
        _,
        height,
        width,
        angle,
    ) = extract_nn_estimate_of_posterior(
        cest_tissue_params_est if is_cest else mt_tissue_params_est,
        data_feed_mt.shape,
        is_cest=is_cest,
    )

    try:
        for _x, _y in xy_points:
            if len(good_xy_points) >= voxels2sample:
                break
            if np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y]):
                continue            

            time0 = time.time()
            (
                f_best,
                k_best,
                _,
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
                data_feed_cest=data_feed_cest,
                seq_name=seq_name,
                seq_df=seq_df,
                mt_sim_mode=mt_sim_mode,
                max_k_cest=max_k_cest,
            )
            # ! note NN output is already scaled to percentage units
            _df_in_perc = 100 * _df  
            timings.append(time.time() - time0)
            exact_min_percs.append(100 * np.min(nrmse))
            try:
                nn_min_percs.append(
                    100 * nrmse[
                        min(int(f_est[_x, sli, _y] / _df_in_perc), nrmse.shape[0] - 1),
                        min(int(k_est[_x, sli, _y] / _dk), nrmse.shape[1] - 1)
                    ]
                )
            except:
                import ipdb; ipdb.set_trace()

            print(f'min-nrmse (@MAP), exact posterior : {exact_min_percs[-1]:.2f}%')              
            print(f'min-nrmse (@MAP), nn-est. posterior: {nn_min_percs[-1]:.2f}%')    
            gt = data_feed_mt.fc_gt_T[_x, sli, _y], data_feed_mt.kc_gt_T[_x, sli, _y]  # only exists in simulated data 
            if known_sigmas is not None:
                au_config.known_sigma = known_sigmas[_x, sli, _y]
            if visualize:
                (
                    _,
                    _,
                    posterior_cov,
                    CR_area,
                    _,
                    (f_mean_grid_posterior, k_mean_grid_posterior),
                ) = viz_posteriors(
                    f_est[_x, sli, _y],
                    k_est[_x, sli, _y],
                    cov_nnpred_scaled[_x, sli, _y],
                    width[_x, sli, _y],
                    height[_x, sli, _y],
                    angle[_x, sli, _y],
                    f_best, 
                    k_best,
                    nrmse,
                    _df,
                    _dk,
                    is_cest=is_cest,
                    visualize=visualize,
                    fontsize=14,
                    do_marginals=True,
                    figsize=[7, 5],
                    show_text=True,
                    show_NN=True,
                    gt=(gt[0]*100, gt[1])
                )
                if folder_name and len(xy_points) < 10:
                    for fmt in ["svg", "png"]:
                        plt.savefig(
                            f"./{folder_name}/voxel_{_x}_{sli}_{_y}.{fmt}",
                            format=fmt,
                            dpi=300,
                        )            
            (
                pdf, nrmse_95cr_th, CR_area,
                f_mean_grid_posterior, f_025, f_975,
                k_mean_grid_posterior, k_025, k_975,
                posterior_cov, cdf_at_gt
            ) = core.nrmse_to_CIs(nrmse, is_cest=is_cest, df=_df, dk=_dk, gt=gt)
            cdfs_at_gt.append(cdf_at_gt)
            if visualize:
                print('cdf_at_gt: ', cdf_at_gt)
                plt.show()
            posterior_mu = np.array([f_mean_grid_posterior, k_mean_grid_posterior])                        
            cross_mahas1, cross_mahas2 = get_cross_mahas(
                f_est[_x, sli, _y],
                k_est[_x, sli, _y],
                cov_nnpred_scaled[_x, sli, _y],
                f_mean_grid_posterior,
                k_mean_grid_posterior,
                posterior_cov
            )  
            mahas1 += cross_mahas1.tolist()
            mahas2 += cross_mahas2.tolist()            

            pdf_NN = helpers.gaussian_pdf_2D(
                [f_est[_x, sli, _y], k_est[_x, sli, _y]],
                cov_nnpred_scaled[_x, sli, _y],
                fmax=_df_in_perc * nrmse.shape[0], df=_df_in_perc,
                kmax=_dk * nrmse.shape[1], dk=_dk
            )
            pdf_NN /= np.sum(pdf_NN)  # discreet-norm as the grid-based PDF
            distrib_distances_dict = compute_dist_distances(pdf, pdf_NN)
            # KL(P||Q) = \sum P(x) log(P(x)/Q(x)), so
            # distrib_distances["KL(P1||P2)"] is from exact to NN 
            # ("how much info is lost using NN to approximate exact")
            CR_areas.append(CR_area)
            nnCR_areas.append(width[_x, sli, _y] * height[_x, sli, _y] * np.pi / 4)
            means_exact.append(posterior_mu)
            modes_exact.append((f_best, k_best))
            covs_exact.append(posterior_cov)
            distrib_distances_dicts.append(distrib_distances_dict)
            good_xy_points.append([_x, _y])
            
    except KeyboardInterrupt:
        print("compare_posteriors interrupted; returning collected data.")

    try:
        extra_metrics_dict = {k: np.array([dd[k] for dd in distrib_distances_dicts]) for k in distrib_distances_dicts[0]}
        extra_metrics_dict["cdf_at_gt"] = np.array(cdfs_at_gt)
    except:
        extra_metrics_dict = {}
    
    return (
        mahas1,
        mahas2,
        CR_areas,
        nnCR_areas,
        good_xy_points,
        modes_exact,
        means_exact,
        covs_exact,
        timings,
        exact_min_percs,
        nn_min_percs,
        extra_metrics_dict
    )


__all__ = [
    "get_cross_mahas",
    "summarize_mahas",
    "plot_CI_intersect",
    "plot_fk_CI_intersect",
    "analyze_CI_intersections",
    "analyze_scatter",
    "compare_posteriors",
]
