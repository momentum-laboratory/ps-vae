"""High-level application demos refactored from ``analyze_uncertainty``."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .core import get_nrmse_grid
from .viz import cbarhist
from dictionary_methods import dictbased_runner
from core import infer


def plot_progressive_maps_calc_MAPEs(    
    dict_as_slicefeed, data_feed_mt, f_best, k_best, MAE_nmeas_list, fig=None, xy=None
    ):
    """ 
    Extend dictionary-matching for analyzing partial sequences,
    creating quantitative maps of parameters at partial sequence lengths,
    and calculating MAPE of the "preliminary estimate" at each position
    versus the full-sequence estimate.  
    """
    if fig is None:
        fig, axes = plt.subplots(2, 8, figsize=(20, 4))  # figsize=(2, 2) # (3, 3) for the full image
    else:
        axes = fig.subplots(2, 8)
    f_MAPEs, k_MAPEs, f_MAEs, k_MAEs = [], [], [], []
    for ii, nn in enumerate(MAE_nmeas_list): #[4, 12, 20, 28]): #,12,16,20,24,28]):
        
        _dict_as_slicefeed = copy.deepcopy(dict_as_slicefeed)
        _dict_as_slicefeed.measured_normed_T = np.copy(_dict_as_slicefeed.measured_normed_T[:nn])        
        _data_feed_mt = copy.deepcopy(data_feed_mt)
        _data_feed_mt.measured_normed_T = np.copy(_data_feed_mt.measured_normed_T[:nn])        
        _f_best, _k_best, best_match, err_3d_dict = \
        dictbased_runner.match_to_dict(_dict_as_slicefeed, _data_feed_mt, constrain_T1T2=False)
            
        # MAPE nuance:  ideally vs GT, here taking the final mM and k_best either voxelwise or vialwise        
        f_MAPE = 100 * np.nanmean(np.abs(_f_best - f_best) / (1e-6 + f_best))
        f_MAPEs.append(f_MAPE)
        f_MAEs.append(np.nanmean(np.abs(_f_best - f_best)))
        k_MAPE = 100 * np.nanmean(np.abs(_k_best - k_best) / (1e0 + k_best))
        k_MAPEs.append(k_MAPE)
        k_MAEs.append(np.nanmean(np.abs(_k_best - k_best) ))
        
        if ii % 4 != 0:
            continue  # don't plot odd    
        
        _data = _f_best.squeeze()
        _ax1, _ax2 = axes[:, int(ii/4)]
        _img = _ax1.imshow(_data, vmin=0, vmax=30) 
        if xy is not None:
            _ax1.plot([xy[1]], [xy[0]], marker='s', markerfacecolor='none', markeredgecolor='violet', markersize=7, markeredgewidth=3) #, linewidths=2)                
        _ax1.set_xticks([])  # remove x-ticks
        _ax1.set_yticks([])  # remove y-ticks
        cbar = cbarhist(_img, _data, _ax1, bins=20, old_style=True) # , bins=np.arange(0.1, 0.2, _df*100)); 
        if nn==MAE_nmeas_list[-1]:  # work around a bug on second row by aligning both
            cbar.set_label('f$_{ss}$ (%)')
        else:
            cbar.set_ticklabels([]) 

        _ax1.text(x=0.05, y=1.05, transform=_ax1.transAxes, s=f'MAPE={f_MAPE:.1f}%')#, fontweight='bold')
        _data = _k_best.squeeze() * data_feed_mt.roi_mask_nans[:,0,:]
        k_vmax=50 # 1200 larg
        _img = _ax2.imshow(_data, vmin=0, vmax=k_vmax, cmap='magma')         
        _ax2.set_xticks([])  # remove x-ticks
        _ax2.set_yticks([])  # remove y-ticks
        
        _ax2.text(x=0.05, y=-0.15, transform=_ax2.transAxes, s=f'n={nn}', fontweight='bold')
        _ax2.text(x=0.05, y=1.05, transform=_ax2.transAxes, s=f'MAPE={k_MAPE:.1f}%')#, fontweight='bold')
    
        cbar = cbarhist(_img, _data, _ax2, bins=25, old_style=True) #, bins=np.arange(vmin=0, vmax=1000, _dk)); 
        if nn==MAE_nmeas_list[-1]:  
            cbar.set_label('k$_{ssw}$ (s$^{-1}$)')
        else:
            cbar.set_ticklabels([]) 
        
    return f_MAPEs, k_MAPEs, f_MAEs, k_MAEs


def plot_progressive_pdfs(
    cumsum_sqerr,
    pdf,
    n_meas,
    _df,
    _dk,
    loc_dict_res,
    figsize=(15, 3),
    sample_voxel_CRs_indices=None,
    fig=None
):
    """Track posterior evolution as measurements accumulate.
    Plot posteriors at several sequence lengths for a sample voxel.
    """
    sample_voxel_CRs_indices = sample_voxel_CRs_indices or list(range(5, n_meas, 4))
    fig = fig or plt.figure(figsize=figsize)
    
    axes_top = [
        fig.add_subplot(1, len(sample_voxel_CRs_indices), ii + 1)
        for ii in range(len(sample_voxel_CRs_indices))
    ]

    f_prog_est, k_prog_est, CR_areas, f_CIs, k_CIs = [], [], [], [], []
    for ax, jj in zip(axes_top, sample_voxel_CRs_indices):
        plt.sca(ax)
        loc_nrmse = np.sqrt(n_meas * cumsum_sqerr[jj] / jj)
        f_best = np.nanargmin(loc_nrmse) // loc_nrmse.shape[1] * _df * 100
        k_best = np.nanargmin(loc_nrmse) % loc_nrmse.shape[1] * _dk
        f_prog_est.append(f_best)
        k_prog_est.append(k_best)

        pdf_jj = pdf[jj].T
        pdf_jj_01 = pdf_jj / np.max(pdf[jj])
        imsh = ax.imshow(
            pdf_jj_01,
            origin="lower",
            aspect="auto",
            extent=[0, _df * 100 * loc_dict_res, 0, _dk * loc_dict_res],
            cmap="hot",
        )
        if sample_voxel_CRs_indices[-1] == jj:
            plt.colorbar(imsh).set_label("Posterior probability density (pdf)")
        pdf01_th_arr = np.arange(0.0, 0.2, 0.004)
        cdf = [np.sum(pdf_jj[pdf_jj_01 > th]) for th in pdf01_th_arr]
        pdf01_th = pdf01_th_arr[np.argmin(np.abs(np.array(cdf) - 0.95))]
        ax.contour(
            pdf_jj_01,
            levels=[pdf01_th],
            colors=["magenta"],
            linewidths=[2],
            alpha=0.7,
            extent=[0, _df * 100 * loc_dict_res, 0, _dk * loc_dict_res],
        )
        CR_areas.append(np.sum(pdf_jj_01 > pdf01_th) / loc_dict_res**2)

        ax.set_xlabel(r"f$_{ss}$ (%)", fontsize=16)
        if ax is not axes_top[0]:
            ax.set_yticks([])
        else:
            ax.set_ylabel("k$_{ssw}$ (s$^{-1}$)", fontsize=16)
        marg_f = np.sum(pdf_jj, axis=0)
        marg_k = np.sum(pdf_jj, axis=1)
        marg_f_cdf = np.cumsum(marg_f)
        marg_k_cdf = np.cumsum(marg_k)
        f_975 = _df * 100 * np.searchsorted(marg_f_cdf, 0.975)
        f_025 = _df * 100 * np.searchsorted(marg_f_cdf, 0.025)
        k_975 = _dk * np.searchsorted(marg_k_cdf, 0.975)
        k_025 = _dk * np.searchsorted(marg_k_cdf, 0.025)
        for x in (f_025, f_975):
            ax.plot((x, x), (0, _dk * loc_dict_res), "w--", alpha=0.2)
        for y in (k_025, k_975):
            ax.plot((0, _df * 100 * loc_dict_res), (y, y), "w--", alpha=0.2)
        f_CIs.append(f_975 - f_025)
        k_CIs.append(k_975 - k_025)
        CR_area_perc = 100 * CR_areas[-1]        # % of all param space
        ax.text(
            s=f"CR area={CR_area_perc:.1f} a.u.\nf$_{{ss}}$={f_025:.1f}-{f_975:.1f} %\nk$_{{ssw}}$={k_025:.0f}-{k_975:.0f} s$^{{-1}}$",
            fontsize=16, color='w', x=0.98, y=0.75, transform=ax.transAxes, ha='right'
            
        )  
        ax.text(s=f'n={jj+1}', fontsize=16, color='w', fontweight='bold', x=0.05, y=0.05, transform=ax.transAxes)        
        
    return sample_voxel_CRs_indices, f_prog_est, k_prog_est, CR_areas, f_CIs, k_CIs


def plot_CRs_decrease_along_sequence(
    data_feed,
    sli: int = 0,
    voxels2sample: int = 10,
    seq_name: str = "larg",
    seq_df=None,
    mt_tissue_param_est=None,
    figsize=(10, 5),
    loc_dict_res: int = 100,
    to_plot: bool = False,
):
    """
    Track confidence-region shrinkage as measurements accrue.
    Repeat for several voxels randomly sampled in a slice, 
    register trends of metrics of interest (CR area, CIs, estimation error),
    across partial sequence lengths, for each voxel
    """

    _z = sli
    good_points_done = 0
    CR_area_trends, f_CI_trends, k_CI_trends = [], [], []
    f_error_trends, k_error_trends = [], []
    xy_points = np.int32(np.random.random((1000, 2)) * data_feed.shape[0::2])
    good_xy_points: list[list[int]] = []
    f_vals, k_vals = [], []

    for _x, _y in xy_points:
        if np.isnan(data_feed.roi_mask_nans[_x, sli, _y]):
            continue
        good_points_done += 1
        if good_points_done > voxels2sample:
            break
        good_xy_points.append([_x, _y])
        signal = np.copy(data_feed.measured_normed_T[:, _x, sli, _y])
        signal /= np.linalg.norm(signal, axis=0)
        n_meas = signal.shape[0]
        grid_kwargs = (
            dict(
                data_feed_mt=None,
                mt_tissue_param_est=mt_tissue_param_est,
                data_feed_cest=data_feed,
                seq_name=seq_name,
                seq_df=seq_df,
            )
            if seq_name != "mt"
            else dict(
                data_feed_mt=data_feed,
                mt_tissue_param_est=mt_tissue_param_est,
                seq_name=seq_name,
                seq_df=seq_df,
            )
        )
        f_best, k_best, dict_signal, nrmse, _df, _dk = get_nrmse_grid(
            None,
            _x,
            _y,
            _z,
            loc_dict_res=loc_dict_res,
            **grid_kwargs,
        )
        f_vals.append(f_best)
        k_vals.append(k_best)

        residuals_sq = (dict_signal - signal[:, None, None]) ** 2
        cumsum_sqerr = np.cumsum(residuals_sq, axis=0)
        sigma_est = np.sqrt(np.min(cumsum_sqerr[-1]) / n_meas)

        cumul_log_likelihood = -cumsum_sqerr / (2 * sigma_est**2)
        pdf_by_stop_points = np.exp(cumul_log_likelihood)
        pdf_by_stop_points /= np.sum(pdf_by_stop_points, axis=(1, 2), keepdims=True)

        CR_areas_local, f_CIs, k_CIs = [], [], []
        stop_points = list(range(2, n_meas))
        f_errors, k_errors = [], []

        for jj in stop_points:
            pdf_jj = pdf_by_stop_points[jj].T
            pdf_jj_01 = pdf_jj / np.max(pdf_jj)
            pdf01_th_arr = np.arange(0.0, 0.2, 0.004)
            cdf = [np.sum(pdf_jj[pdf_jj_01 > th]) for th in pdf01_th_arr]
            pdf01_th = pdf01_th_arr[np.argmin(np.abs(np.array(cdf) - 0.95))]
            CR_areas_local.append(np.sum(pdf_jj_01 > pdf01_th) / loc_dict_res**2)

            marg_f = np.sum(pdf_jj, axis=0)
            marg_f_cdf = np.cumsum(marg_f)
            marg_k = np.sum(pdf_jj, axis=1)
            marg_k_cdf = np.cumsum(marg_k)
            f_975 = _df * 100 * np.searchsorted(marg_f_cdf, 0.975)
            f_025 = _df * 100 * np.searchsorted(marg_f_cdf, 0.025)
            k_975 = _dk * np.searchsorted(marg_k_cdf, 0.975)
            k_025 = _dk * np.searchsorted(marg_k_cdf, 0.025)
            f_CIs.append(f_975 - f_025)
            k_CIs.append(k_975 - k_025)

            flat_idx = np.nanargmax(pdf_by_stop_points[jj])
            f_best_loc = flat_idx // pdf_by_stop_points[jj].shape[1] * _df * 100
            k_best_loc = flat_idx % pdf_by_stop_points[jj].shape[1] * _dk
            f_errors.append(f_best_loc - f_best)
            k_errors.append(k_best_loc - k_best)

        CR_area_trends.append(CR_areas_local)
        f_CI_trends.append(f_CIs)
        k_CI_trends.append(k_CIs)
        f_error_trends.append(f_errors)
        k_error_trends.append(k_errors)

        if to_plot:
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.ylabel("CR area / final")
            plt.xlabel("sequence length")
            plt.semilogy(
                stop_points,
                np.array(CR_areas_local) / CR_areas_local[-1],
                "o-",
                markerfacecolor="none",
                linewidth=0.5,
                alpha=0.4,
            )
            plt.ylim(0.8, 20)
            plt.grid("on")
            plt.subplot(1, 2, 2)
            plt.ylabel("CR area (% space)")
            plt.xlabel("sequence length")
            plt.plot(
                stop_points,
                100 * np.array(CR_areas_local),
                "o-",
                markerfacecolor="none",
                linewidth=0.5,
                alpha=0.4,
            )
            plt.grid("on")

    return (
        good_xy_points,
        stop_points,
        CR_area_trends,
        f_CI_trends,
        k_CI_trends,
        f_error_trends,
        k_error_trends,
        np.array(f_vals),
        np.array(k_vals),
    )


def plot_CR_area_vs_measurements(
        all_voxels_CRs_indices, sample_voxel_CRs_indices, CR_areas, CR_area_trends, f_CI_trends, k_CI_trends, data_feed_mt, 
        fig=None, CRagg='mean_pm_std' # otherwise median, Q2-Q3 (25_75)
    ):
    if fig is None:
        fig = plt.figure(figsize=(8, 4))     # 8, 6
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
    axes[0].plot(
        sample_voxel_CRs_indices+1, CR_areas, 'ms-', 
        markersize=5, markeredgewidth=1.5, markerfacecolor='none', linewidth=1,
        label='CR area, marked voxel' #, alpha=0.5)
    )
    CR_area_trends = np.array(CR_area_trends)
    
    rlines = axes[0].plot(
        all_voxels_CRs_indices+1, 
        CR_area_trends[np.random.choice(CR_area_trends.shape[0], 30, replace=False), :].T,
        'm-', linewidth=0.5, alpha=0.3,
        #label='CR area- random voxels' # if jj==0 else None
    )
    rlines[0].set_label('CR area, random voxels')
    
    xlim = (0.5, 31.5)    
    axes[0].spines['bottom'].set_visible(False)    
    axes[0].set_xticklabels([])  # Hide x-ticks for the top plot
    axes[0].set_ylim(0, 0.1)
    axes[0].set_xlim(*xlim)

    # Set major ticks (default) and minor ticks (twice as dense)
    axes[0].yaxis.set_minor_locator(MultipleLocator(axes[0].get_ylim()[1] / 20))
    axes[0].grid(which='both', linewidth=0.5, alpha=0.5)
    axes[0].spines['left'].set_color('m')
    axes[0].spines['left'].set_linewidth(2)
    axes[0].tick_params(axis='y', colors='m')
    axes[0].set_ylabel('CR area (a.u.)', color='m')   
    axes[0].legend(loc='center right')
    
    axes0b = axes[0].twinx()
    axes0b.tick_params(axis='y', colors='g')
    axes0b.spines['right'].set_color('g')
    axes0b.spines['right'].set_linewidth(2)
    axes0b.spines['left'].set_visible(False)
    axes0b.set_xlim(*xlim)
    axes0b.plot(
        all_voxels_CRs_indices+1, 
        100*(np.mean(f_CI_trends, axis=0) if CRagg=='mean_pm_std' else np.median(f_CI_trends, axis=0))/100/infer.infer_config.fc_scale_fact,
        'g^-', markersize=4, linewidth=0.5, label='f$_{ss}$ CI mean size' if CRagg=='mean_pm_std' else 'f$_{ss}$ CI (median over voxels)'
    )
    axes0b.plot(
        all_voxels_CRs_indices+1, 
        100*(np.mean(k_CI_trends, axis=0) if CRagg=='mean_pm_std' else np.median(k_CI_trends, axis=0))/infer.infer_config.kc_scale_fact, 
        'gv-', markersize=4, linewidth=0.5, label='k$_{ss}$ CI mean size' if CRagg=='mean_pm_std' else 'k$_{ss}$ CI (median over voxels)')
    axes0b.set_ylabel('% of parameter range', color='g')
    axes0b.legend(loc='upper right')
            
    # !! much commented-out code here removed on 2025-dec-04
    #########################################################

    l1, = axes[1].plot(range(1, 31+1), data_feed_mt.seq_df['B1_uT'], 'bx-', linewidth=0.5, label='Sat. RF strength') # 'B$_1$') 
    axes[1].set_ylabel('B$_1$ (uT)', color='b')
    axes1b = axes[1].twinx()
    l2, = axes1b.plot(range(2, 31+1), data_feed_mt.seq_df['ppm'][1:], 'r+-', markersize=7, linewidth=0.5, label='Sat. RF offset') 
    axes1b.set_ylabel(r'$\Delta\omega/\omega_0$ (ppm)', color='r')
    axes1b.legend(handles=[l1, l2], loc='upper right')
    axes1b.set_xlim(*xlim)
    axes[1].spines['top'].set_visible(False)
    axes1b.spines['top'].set_visible(False)

    axes1b.set_yticks([6, 10, 14])
    axes1b.tick_params(axis='y', colors='r')
    axes1b.spines['right'].set_color('r')
    axes1b.spines['right'].set_linewidth(2)

    axes[1].tick_params(axis='y', colors='b')
    axes[1].spines['left'].set_color('b')
    axes[1].spines['left'].set_linewidth(2)
    
    axes[1].set_xlabel('Number of scans acquired')
    axes[1].set_xlim(*xlim)
    axes[1].tick_params(axis='x', which='both', labelbottom=True)
    axes[1].set_xticklabels(axes[1].get_xticks().astype(int))

########################################
########################################


def demo_classification_app(
    f_est=None,
    k_est=None,
    cov_nnpred_scaled=None,
    all_masked_points=None,
    all_masked_labels=None,
    sli=None,
    ds: int = 1,
    is_mt: bool = True,
    psample: int = 30,
    result_fig_name: str | None = None,
    ff=None,
    kk=None,
    covcov=None,
    clsf_type: str = "logreg",
    fig=None, fontsize=8
):
    """Sample posterior clusters and draw a simple decision boundary."""

    import sklearn  # noqa: F401 - ensures dependency availability at runtime

    if fig is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 20 if is_mt else 0.5)
    ax.set_ylim(0, 70 if is_mt else 600)
    xlabel = r"$\hat{f}_{ss}$ (%)" if is_mt else r"$\hat{f}_{s}$   (%)"
    ylabel = r"$\hat{k}_{ssw}\ (s^{-1})$" if is_mt else r"$\hat{k}_{sw}\ (s^{-1})$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    all_samples, all_labels = [], []
    dark, colors = True, ["royalblue", "gold"]

    if ff is None or kk is None or covcov is None:
        assert (
            f_est is not None
            and k_est is not None
            and cov_nnpred_scaled is not None
            and all_masked_points is not None
            and sli is not None
        ), "Missing arguments!"
        idx_x = [p[0] for p in all_masked_points]
        idx_y = [p[1] for p in all_masked_points]
        ff = f_est[idx_x, sli, idx_y].astype(float)
        kk = k_est[idx_x, sli, idx_y].astype(float)
        covcov = cov_nnpred_scaled[idx_x, sli, idx_y]

    for jj, (_cov, mu_f, mu_k, label) in enumerate(
        zip(covcov[::ds], ff[::ds], kk[::ds], all_masked_labels[::ds])
    ):
        sample = np.random.multivariate_normal((mu_f, mu_k), _cov, size=psample)
        all_samples.extend(sample.tolist())
        all_labels += [label] * len(sample)

    if dark:
        ax.set_facecolor("black")
    scatter_handle = ax.scatter(
        np.array(all_samples)[:, 0],
        np.array(all_samples)[:, 1],
        marker=".",
        c=[colors[lb] for lb in all_labels],
        s=15,
        alpha=0.15,
        label=f"{psample} samples\nfrom posterior",
    )

    centers_handle = ax.scatter(
        ff,
        kk,
        c=[colors[lb] for lb in all_masked_labels],
        s=10,
        edgecolors="r",
        linewidths=0.5,
        label=f"MAP estimate",
    )

    X = np.array(all_samples)
    Y = np.array(all_labels)

    if clsf_type == "svm":
        from sklearn.svm import SVC

        clf = SVC(kernel="linear")
        clf.fit(X, Y)
        x = np.linspace(0, 30 if is_mt else 0.5, 100)
        y = -(clf.intercept_[0] + clf.coef_[0][0] * x) / clf.coef_[0][1]
        plt.plot(x, y, "--", color="pink", label="SVM boundary", linewidth=1)
    elif clsf_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit(X, Y)
        x = np.linspace(0, 30 if is_mt else 0.5, 100)
        y = -(clf.intercept_[0] + clf.coef_[0][0] * x) / clf.coef_[0][1]
        plt.plot(
            x,
            y,
            "--",
            color="pink",
            label="Classifier\nboundary",
            linewidth=1,
        )
    else:
        raise ValueError("Unknown classifier type!")

    plt.legend(
        handles=[centers_handle] + [scatter_handle],
        fontsize=fontsize,
        loc="upper right",# if is_mt else "lower right",
        facecolor="darkgray",
        framealpha=0.75,
        labelcolor="white",
    )
        
    if result_fig_name is not None:
        for fmt in ["svg", "png"]:
            plt.savefig(
                result_fig_name + "." + fmt,
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )

    zz = np.array(all_samples)[np.array(all_labels) == 1, :]
    mu, std = np.mean(zz, axis=0), np.std(zz, axis=0)
    fss_stats_tumor, kss_stats_tumor = (mu[0], std[0]), (mu[1], std[1])
    zz = np.array(all_samples)[np.array(all_labels) == 0, :]
    mu, std = np.mean(zz, axis=0), np.std(zz, axis=0)
    fss_stats, kss_stats = (mu[0], std[0]), (mu[1], std[1])

    fss_CUR = abs(fss_stats_tumor[0] - fss_stats[0]) / np.mean(
        [fss_stats[1], fss_stats_tumor[1]]
    )
    kss_CUR = abs(kss_stats_tumor[0] - kss_stats[0]) / np.mean(
        [kss_stats[1], kss_stats_tumor[1]]
    )

    return (
        fss_stats_tumor,
        kss_stats_tumor,
        fss_stats,
        kss_stats,
        fss_CUR,
        kss_CUR,
    )


def demo_posterior_sampling(
    ff,
    kk,
    covcov,
    widths,
    heights,
    angles,
    labels,
    psample: int = 30,
    is_mt: bool = True,
    fig=None, fontsize: int = 10
):
    """Illustrate Monte Carlo posterior sampling with ellipse overlays."""

    if fig is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        ax = fig.add_subplot(1, 1, 1)
    if is_mt:
        ax.set_xlim(2, 20)
        ax.set_ylim(5, 70)
    else:   
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 600)
    xlabel = r"$\hat{f}_{ss}$ (%)" if is_mt else r"$\hat{f}_{s}$ (%)"
    ylabel = r"$\hat{k}_{ssw}\ (s^{-1})$" if is_mt else r"$\hat{k}_{sw}\ (s^{-1})$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    dark, colors = True, ["royalblue", "gold"]
    ellipse_handles = {}
    all_samples, all_labels = [], []

    for jj, (mu_f, mu_k, cov, ew, eh, eangle, label) in enumerate(
        zip(ff, kk, covcov, widths, heights, angles, labels)
    ):
        ellipse = Ellipse(
            xy=(mu_f, mu_k),
            width=ew,
            height=eh,
            angle=eangle - 90,
            edgecolor=colors[label],
            facecolor="none",
            linewidth=2,
            zorder=0,
            alpha=0.5,
            label="95% (2$\sigma$) CR,"
            + ["\ncontralateral \nvoxel", "\ntumor voxel"][label],
        )
        ellipse_handles[label] = ellipse
        ax.add_patch(ellipse)
        sample = np.random.multivariate_normal((mu_f, mu_k), cov, size=psample)
        all_samples.extend(sample.tolist())
        all_labels += [label] * len(sample)

    if dark:
        ax.set_facecolor("black")
    centers_handle = plt.scatter(
        ff,
        kk,
        marker="o",
        c=[colors[lb] for lb in labels],
        s=30,
        edgecolors="red",
        linewidths=2,
        label="MAP estimate",
    )
    scatter_handle = plt.scatter(
        np.array(all_samples)[:, 0],
        np.array(all_samples)[:, 1],
        marker=".",
        c=[colors[lb] for lb in all_labels],
        s=10,
        alpha=0.5,
        label=f"{psample} samples\nfrom posterior",
    )
    plt.legend(
        handles=[centers_handle] + list(ellipse_handles.values()) + [scatter_handle],
        fontsize=fontsize,
        loc="upper right",# if is_mt else "lower right",
        facecolor="gray", #"darkgray",
        labelcolor="white",
        framealpha=0.75,
    )


def nn_vs_ref_est_vs_posterior_sample(
    f_est, k_est, cov_nnpred_scaled, 
    f_dict_est_map, k_dict_est_map, fk_cov_dict_est_map, 
    figNN=None, figRef=None, fmax=25, kmax=60, ss='ss'
    ):
        
    f_est_noised = f_est.squeeze().copy()
    k_est_noised = k_est.squeeze().copy()
    f_dict_est_noised = f_dict_est_map.squeeze().copy()
    k_dict_est_noised = k_dict_est_map.squeeze().copy()
    for ii in range(f_est_noised.shape[0]):
        for jj in range(f_est_noised.shape[1]):    
            if np.any(np.isnan(cov_nnpred_scaled[ii, jj])):
                continue   
            sample = np.random.multivariate_normal(
                (f_est[ii, jj], k_est[ii, jj]), cov_nnpred_scaled[ii, jj], 
                size=1
            )        
            f_est_noised[ii, jj] = sample[0, 0]
            k_est_noised[ii, jj] = sample[0, 1]
            sample_dictUQ = np.random.multivariate_normal(
                (f_dict_est_map[ii, jj], k_dict_est_map[ii, jj]), fk_cov_dict_est_map[ii, jj], 
                size=1
            )        
            f_dict_est_noised[ii, jj] = sample_dictUQ[0, 0]
            k_dict_est_noised[ii, jj] = sample_dictUQ[0, 1]

    figNN = figNN or plt.figure(figsize=(8, 6))
    axes = figNN.subplots(2, 2)
    for ax in axes.flatten():
        ax.axis('off')
    axes[0, 0].set_title(('MT' if ss=='ss' else 'APT') + ': MAP parameters (NN)')
    axes[0, 0].imshow(f_est, vmin=0, vmax=fmax, cmap='viridis')
    axes[1, 0].imshow(k_est, vmin=0, vmax=kmax, cmap='magma')          
    axes[0, 1].set_title(('MT' if ss=='ss' else 'APT') + ': posterior sample (NN)') 
    axes[0, 1].imshow(f_est_noised, vmin=0, vmax=fmax, cmap='viridis')     
    axes[1, 1].imshow(k_est_noised, vmin=0, vmax=kmax, cmap='magma')
    cax = inset_axes(axes[0,1], width="10%", height="90%", loc="center right", borderpad=-2)
    cbar = plt.colorbar(axes[0,1].images[0], cax=cax) 
    cbar.set_label(rf'f$_{{{ss}}}$ (%)')
    cax = inset_axes(axes[1,1], width="10%", height="90%", loc="center right", borderpad=-2)
    cbar = plt.colorbar(axes[1,1].images[0], cax=cax) 
    cbar.set_label(rf'k$_{{{ss}w}}$ (s$^{-1}$)')
    
    figRef = figRef or plt.figure(figsize=(8, 6))
    axes = figRef.subplots(2, 2)
    for ax in axes.flatten():
        ax.axis('off')
    axes[0, 0].set_title(('MT' if ss=='ss' else 'APT') + ': MAP parameters (Ref.)')
    axes[0, 0].imshow(f_dict_est_map, vmin=0, vmax=fmax, cmap='viridis')
    axes[1, 0].imshow(k_dict_est_map, vmin=0, vmax=kmax, cmap='magma')          
    axes[0, 1].set_title(('MT' if ss=='ss' else 'APT') + ': posterior sample (Ref.)') 
    axes[0, 1].imshow(f_dict_est_noised, vmin=0, vmax=fmax, cmap='viridis')     
    axes[1, 1].imshow(k_dict_est_noised, vmin=0, vmax=kmax, cmap='magma')
    cax = inset_axes(axes[0,1], width="10%", height="90%", loc="center right", borderpad=-2)
    cbar = plt.colorbar(axes[0,1].images[0], cax=cax) 
    cbar.set_label(rf'f$_{{{ss}}}$ (%)')
    cax = inset_axes(axes[1,1], width="10%", height="90%", loc="center right", borderpad=-2)
    cbar = plt.colorbar(axes[1,1].images[0], cax=cax) 
    cbar.set_label(rf'k$_{{{ss}w}}$ (s$^{-1}$)')


__all__ = [
    "plot_CRs_decrease_along_sequence",
    "demo_classification_app",
    "demo_posterior_sampling",
]
