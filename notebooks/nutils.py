from matplotlib import patches
import numpy as np, matplotlib.pyplot as plt
from importlib import reload
import xarray as xr
import os, time, shutil
from pathlib import Path
from scipy import stats

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "core").exists() and (REPO_ROOT.parent / "core").exists():
    os.chdir(REPO_ROOT.parent)
    REPO_ROOT = Path.cwd()
    
from core import data, config, pipelines, net, infer, train
import analyze_uncertainty as au 


ranges4synth = data.def_ranges4LHS
ranges4synth['T2a_ms'] = [50, 120] 
ranges4synth['B0_shift_ppm_map'] = [-0.3, 0.3] 
ranges4synth['B1_fix_factor_map'] = [0.9, 1.1] 
ranges4synth['fc_gt_T'] = [0.05, 0.25]
ranges4synth['kc_gt_T'] = [10, 60]
NOISE_LEVEL = list(np.arange(0.01, 0.04, 0.002))


def config4sim():    
    config.Config.configure_AMIDE_clinical_whole_brain() 
    train.train_config.hidden_layers = 4


def synthesize_data(
    mt_sim_mode='isar2_c', seq_name='mt', 
    data_shape=(100, 10, 100), noise_level=NOISE_LEVEL, plot_noise_hist=False,
    add_bias=False
    ):
    # Sample parameter space:
    synthetic_datafeed = data.SlicesFeed.make_lh_sample( 
        seq_name=seq_name, 
        tissue_parameter_ranges_od=ranges4synth,
        shape=data_shape
    )   
    # Synthesize the signals:
    sim_mode = mt_sim_mode if seq_name=='mt' else 'expm_bmmat'
    synthetic_datafeed.slw = 10 if seq_name=='mt' else 1
    _, signal_3p = infer.infer(
        synthetic_datafeed, 
        simulation_mode=sim_mode
    )    
    #signal_3p *= (1.0 + noise_level * np.random.randn(*signal_3p.shape))  # adding noise to synthetic signals
    if type(noise_level) == list:
        noise_level = np.random.choice(NOISE_LEVEL, size=[1, *signal_3p.shape[1:]])        
        
    noise = noise_level * np.random.randn(*signal_3p.shape) / np.sqrt(signal_3p.shape[0])
    noise_norm = np.linalg.norm(noise, axis=0)
    if plot_noise_hist:
        plt.hist(noise_norm.flatten(), bins=50)
    signal_3p += noise
    if add_bias:
        bias_level = np.random.choice(NOISE_LEVEL, size=[1, *signal_3p.shape[1:]])  # small bias of 1-3% of the signal amplitude
        signal_3p[10:20] += bias_level / np.sqrt(signal_3p.shape[0])
    # !! normalize the signals after adding noise and/or bias !!
    synthetic_datafeed.measured_normed_T = synthetic_datafeed.normalize(signal_3p)
    synthetic_datafeed.slw = 1
    return synthetic_datafeed, sim_mode, noise_norm


def get_lims(data_xa):
    xmin = list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=1) > 0).index(True) - 1
    xmax = 1 - list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=1) > 0)[::-1].index(True)
    ymin = list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=0) > 0).index(True) - 1
    ymax = 1 - list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=0) > 0)[::-1].index(True)
    return xmin, xmax, ymin, ymax

def prep_mouse(data_xa):
    xmin, xmax, ymin, ymax = get_lims(data_xa)
    data_xa_cutout = data_xa.isel(
        height=slice(xmin, xmax),
        width=slice(ymin, ymax) 
    )
    
    data_xa_cutout['B0_shift_ppm_map'] = (('height','slice','width'), np.zeros_like(data_xa_cutout['T1ms']))
    data_xa_cutout['B1_fix_factor_map'] = (('height','slice','width'), np.ones_like(data_xa_cutout['T1ms']))
    return data_xa_cutout


def infer_ckpt(
        ckpt_folder_mt, brain2test_mt, mt_sim_mode=None, force_diag=False, do_forward=False, 
        is_cest=False, ckpt_folder_cest=None, brain2test_cest=None
        ):
    """
        Infer tissue parameters' gaussian posterior estimate using a pre-trained checkpoint of PS-VAE, 
        and extract uncertainty-related data from the covariance of the NN-predicted posterior.
    """
    infer.infer_config.force_diag = force_diag
    predictor_mt = net.load_ckpt(folder=ckpt_folder_mt)[0]
    try:
        predictor_cest = net.load_ckpt(folder=ckpt_folder_cest)[0]
    except:
        predictor_cest = None
    transfer_res, err_mt, err_amide = pipelines.transfer_and_plot(
        brain2test_mt=brain2test_mt, brain2test_cest=brain2test_cest, predictor_mt=predictor_mt, predictor_cest=predictor_cest, 
        figsfolder=f'tmp', figsfx='mt', do_slice_plots=None, do_boxplots=None, mt_simulation_mode=mt_sim_mode, do_forward=do_forward,
    )    
    mt_tissue_params_est, amide_tissue_param_est = transfer_res[1], transfer_res[4]
    nn_pred_sim_signal_mt, nn_pred_sim_signal_amide = transfer_res[2], transfer_res[5]
    mt_tissue_params_est.update({'nn_pred_sim_signal_mt': nn_pred_sim_signal_mt, 'err_mt': err_mt})
    if is_cest:
        amide_tissue_param_est.update({'nn_pred_sim_signal_amide': nn_pred_sim_signal_amide, 'err_amide': err_amide})

    f_est, k_est, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
        au.extract_nn_estimate_of_posterior(mt_tissue_params_est if not is_cest else amide_tissue_param_est, brain2test_mt.shape, is_cest=is_cest)
    
    return (mt_tissue_params_est if not is_cest else amide_tissue_param_est), \
        {'width': width, 'height': height, 'angle': angle, 'f': f_est, 'k': k_est, 'cov_nnpred_scaled': cov_nnpred_scaled, 'f_sigma': f_sigma, 'k_sigma': k_sigma}
    
    
def get_stochastic_predictor(model_state=None, ckpt_folder=None, mt_sim_mode='isar2_c'):
    if ckpt_folder is not None:
        model_state = net.get_model_state(ckpt_folder)
    else:
        assert model_state is not None, "Either model_state or ens_ckpts_folder must be provided"
    
    pred_params = {'params': model_state.params, 'batch_stats': model_state.batch_stats}

    def stochastic_predictor(datafeed, key):
        _predictor = lambda batch: model_state.apply_fn(
            pred_params, batch, mutable=['batch_stats'],
            train=False, enable_dropout=True, rngs={'dropout': key} #, deterministic=False, # jax.random.PRNGKey(0)}
        )    
        mt_tissue_params_est, _mt_reconstructed_signal = infer.infer(
            datafeed, pool2predict='c',
            nn_predictor=_predictor, simulation_mode=mt_sim_mode, do_forward=False
        )
        f_est = 100 * mt_tissue_params_est["fc_T"]
        k_est = mt_tissue_params_est["kc_T"]
        return f_est, k_est
    
    return stochastic_predictor

import jax, jax.numpy as jnp

def extract_mc_dropout_uncertainty(stochastic_predictor, datafeed, num_samples=10, use_vmap=True):
    """MC Dropout inference - run multiple stochastic forward passes to get uncertainty estimates
    """            
    keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    
    print('use_vmap', use_vmap)
    if use_vmap: # acceleration; had some past issues
        print('Using vmap for stochastic forward passes')
        single_pass = lambda key: stochastic_predictor(datafeed, key)
        f_est_samples, k_est_samples = jax.vmap(single_pass)(keys)            
    else:
        print('Using loop for stochastic forward passes')
        f_est_samples = []
        k_est_samples = []
        for jj in range(num_samples):
            f_est, k_est = stochastic_predictor(datafeed, keys[jj])        
            f_est_samples.append(f_est)
            k_est_samples.append(k_est)

    mean_f_est = np.mean(f_est_samples, axis=0)
    mean_k_est = np.mean(k_est_samples, axis=0)
    
    #fk_cov = np.cov(f_est_samples.reshape(num_samples, -1), k_est_samples.reshape(num_samples, -1), rowvar=False)  # Fails on memory
    fk_diff = np.concatenate([(f_est_samples-mean_f_est)[:, None, ...], 
                            (k_est_samples-mean_k_est)[:, None, ...]], axis=1)
    fk_cov = np.einsum('ij...,ik...->...jk', fk_diff, fk_diff) / (num_samples - 1)  # unbiased covariance estimate
    return (mean_f_est, mean_k_est, fk_cov, fk_diff)


def extract_ensemble_uncertainty(predictors_ensemble, datafeed, mt_sim_mode='isar2_c'):
    f_est_samples = []
    k_est_samples = []
    for _predictor in predictors_ensemble:
        mt_tissue_params_est, _mt_reconstructed_signal = infer.infer(
            datafeed, pool2predict='c',
            nn_predictor=_predictor, simulation_mode=mt_sim_mode, do_forward=False
        )
        f_est = 100 * mt_tissue_params_est["fc_T"]
        k_est = mt_tissue_params_est["kc_T"]
        f_est_samples.append(f_est)
        k_est_samples.append(k_est)

    mean_f_est = np.mean(f_est_samples, axis=0)
    mean_k_est = np.mean(k_est_samples, axis=0)
    
    fk_diff = np.concatenate([(f_est_samples-mean_f_est)[:, None, ...], 
                            (k_est_samples-mean_k_est)[:, None, ...]], axis=1)
    fk_cov = np.einsum('ij...,ik...->...jk', fk_diff, fk_diff) / (len(predictors_ensemble) - 1)  # unbiased covariance estimate
    return (mean_f_est, mean_k_est, fk_cov, fk_diff)


# ======== visualization helpers ========= 

def cov2ellipse(cov_scaled, chisq=2.45):
    print('cov_scaled.shape', cov_scaled.shape)
    s2_scaled, u_scaled = np.linalg.eigh(cov_scaled)
    s_scaled = np.sqrt(s2_scaled)
    d0 = s_scaled[..., 0, None] * u_scaled[..., 0]
    d1 = s_scaled[..., 1, None] * u_scaled[..., 1]

    angle = np.degrees(np.arctan2(d1[..., 1], d1[..., 0]))
    chisq = 2.45  # Mahalanobis radius for 95% confidence in 2D
    height = np.linalg.norm(d1, axis=-1) * 2 * chisq
    width = np.linalg.norm(d0, axis=-1) * 2 * chisq
    return height, width, angle


def slap_label_on_container(container, label_text, loc=(-0.01, 1.04), color='black'):    
    ghost_ax = container.add_axes([0, 0, 1, 1], frameon=False)        
    ghost_ax.set_xticks([]); ghost_ax.set_yticks([])        
    ghost_ax.text(
        loc[0], loc[1], label_text, transform=ghost_ax.transAxes, 
        fontsize=18, fontweight='bold', va='top', ha='left', 
        color=color
    )  

def plot_fk_scatters(f_gt, k_gt, f_est_nn_l, k_est_nn_l, visualize=True):
    k_MAPE = np.mean(np.abs((k_gt - k_est_nn_l) / k_gt)) * 100
    f_MAPE = np.mean(np.abs((f_gt - f_est_nn_l) / f_gt)) * 100
    if visualize:
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 2, 1)
        plt.scatter(f_gt, f_est_nn_l, alpha=0.1)
        plt.plot([np.min(f_gt), np.max(f_gt)],[np.min(f_gt), np.max(f_gt)], 'k--')

        plt.title(f'MAPE: f={f_MAPE:.2f}%')
        plt.xlabel('Ground Truth f')
        plt.ylabel('NN estimate f')
        plt.subplot(1, 2, 2)
        plt.title(f'MAPE: k={k_MAPE:.2f}%')
        plt.scatter(k_gt, k_est_nn_l, alpha=0.1)
        plt.plot([np.min(k_gt), np.max(k_gt)],[np.min(k_gt), np.max(k_gt)], 'k--')  
        plt.xlabel('Ground Truth k')
        plt.ylabel('NN estimate k')
    return f_MAPE, k_MAPE

def get_mahalanobis_critical(coverage, dims):
    """
    Returns the Mahalanobis distance (M) for a specific coverage 
    level (e.g., 0.95) and dimension (k).
    """
    # M^2 follows a Chi-squared distribution with 'dims' degrees of freedom
    squared_m = stats.chi2.ppf(coverage, df=dims)
    return np.sqrt(squared_m)

def get_coverage(fk_est_nn, reference, cov_est_nn_arr, coverage_levels=None):
    if coverage_levels is None:
        coverage_levels = list(np.arange(0.2, 0.75, 0.1)) + list(np.arange(0.75, 1, 0.02))
    M_thresholds = [get_mahalanobis_critical(coverage, 2) for coverage in coverage_levels]
    coverage_pct = 100.0 * np.array(coverage_levels, dtype=float)
    # Mahalanobises of the GROUND TRUTH under the NN-predicted posterior
    gt_mahas = np.sqrt(((fk_est_nn - reference)[:, None, :] @ (np.linalg.inv(cov_est_nn_arr) @ (fk_est_nn - reference)[..., None])).squeeze())
    empirical_coverage = np.array([np.sum(gt_mahas < mth) / len(gt_mahas) for mth in M_thresholds], dtype=float)
    empirical_pct = 100.0 * empirical_coverage
    return dict(zip(np.int32(coverage_pct) , empirical_pct))


def plot_calibration_curve(coverage_pct, empirical_pct, highlight_idx=-3, miny_inset=91, ax=None, do_inset=True, fill_between=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(coverage_pct, empirical_pct, marker='o', label='Empirical\nCoverage')
    ax.plot([0, 100], [0, 100], 'k--', label='Ideal', zorder=4)
    if fill_between:    # green where empirical > ideal, red where empirical < ideal
        cp = np.array(coverage_pct, dtype=float)
        ep = np.array(empirical_pct, dtype=float)
        above = ep > cp
        below = ep < cp
        if above.any():
            ax.fill_between(cp, ep, cp, where=above, interpolate=True, color='#8fd694', alpha=0.2, zorder=2)
        if below.any():
            ax.fill_between(cp, ep, cp, where=below, interpolate=True, color='#f5a3a3', alpha=0.2, zorder=2)

    ax.set_xlim(20, 100)
    ax.set_ylim(20, 100)
    ax.set_xlabel('Coverage percentile (%)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.legend(loc='center left', frameon=False)
    ax.plot([coverage_pct[highlight_idx], coverage_pct[highlight_idx]],  [empirical_pct[highlight_idx], empirical_pct[highlight_idx]], 'o',  markerfacecolor='brown', markersize=8)

    # Inset: zoom into high-coverage region (90-100%)
    if do_inset:
        axins = ax.inset_axes([0.62, 0.12, 0.35, 0.35])
        axins.plot(coverage_pct, empirical_pct, marker='o')
        axins.plot([91, 100], [91, 100], 'k--')
        axins.plot([coverage_pct[highlight_idx], coverage_pct[highlight_idx]],  [empirical_pct[highlight_idx], empirical_pct[highlight_idx]], 'o', markerfacecolor='brown', markersize=11)
        axins.set_xlim(91, 100)
        axins.set_ylim(miny_inset, 100)
        axins.set_xticks(range(91, 100, 2))
        axins.set_yticks(range(miny_inset, 100, 4))
        axins.grid(alpha=0.25)
        ax.indicate_inset_zoom(axins, edgecolor='gray')
    return ax


def plot_nn_vs_ref_coverage(npz_filename=None, gt_reference=None, fk_est_nn=None, cov_est_nn_arr=None, est_method="PS-VAE estimation:\n",
                            coverage_levels=None, axes=[None, None], txtprefix="", fontsize=10, **kwargs):
    """
    Args: 
        npz_filename (_type_): as typically saved in xx_allpoints.npz (result of compare_posteriors() )
        coverage_levels (_type_, optional): _description_. Defaults to None.
        axes (list, optional): _description_. Defaults to [None, None].
        txtprefix (str, optional): _description_. Defaults to "".
        fontsize (int, optional): _description_. Defaults to 10.
    """
    if axes[0] is None and axes[1] is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
    coverage_levels = coverage_levels or list(np.arange(0.2, 0.75, 0.1)) + list(np.arange(0.75, 1, 0.02))
    coverage_pct = 100 * np.array(coverage_levels) 
    M_thresholds = [get_mahalanobis_critical(coverage, 2) for coverage in coverage_levels]
    
    if npz_filename:
        saved = np.load(npz_filename, allow_pickle=True)
        f_est_nn_l = [saved['f_est'][_x, 0, _y] for (_x, _y) in saved['good_xy_points']]
        k_est_nn_l = [saved['k_est'][_x, 0, _y] for (_x, _y) in saved['good_xy_points']]
        
        fk_est_nn = np.stack([f_est_nn_l, k_est_nn_l], axis=1)
        gt_reference = np.stack([saved['f_gt'], saved['k_gt']], axis=-1)  
        cov_est_nn_arr = np.array([saved['cov_nnpred_scaled'][_x, 0, _y] for (_x, _y) in saved['good_xy_points']])
    else:
        assert fk_est_nn is not None and gt_reference is not None and cov_est_nn_arr is not None, \
            "If npz_filename is not provided, fk_est_nn, gt_reference, and cov_est_nn_arr must be given."
    
    print(fk_est_nn.shape, gt_reference.shape, cov_est_nn_arr.shape)   
    # Mahalanobises of the GROUND TRUTH under the NN-predicted posterior
    gt_mahas = np.sqrt(((fk_est_nn - gt_reference)[:, None, :] @ (np.linalg.inv(cov_est_nn_arr) @ (fk_est_nn - gt_reference)[..., None])).squeeze())
    
    if axes[0] is not None:
        empirical_coverage = np.array([np.sum(gt_mahas < mth) / len(gt_mahas) for mth in M_thresholds], dtype=float)    
        ax = plot_calibration_curve(coverage_pct, 100.0 * empirical_coverage, miny_inset=81, ax=axes[0], **kwargs)   # 95
        nn_mape_f, nn_mape_k = plot_fk_scatters(gt_reference[:, 0], gt_reference[:, 1], fk_est_nn[:, 0], fk_est_nn[:, 1], visualize=False)
        ax.text(0.02, 0.98, txtprefix+est_method+f"MAPE = {nn_mape_f:.1f}% (f$_{{ss}}$), {nn_mape_k:.1f}% (k$_{{ssw}}$) ", 
                fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
    
    if axes[1] is not None:
        cdf_at_gt = saved['cdf_at_gt'] # ! free-form CRs from grid-likelihood
        empirical_coverage = np.array([np.sum(cdf_at_gt <= coverage) / len(cdf_at_gt) for coverage in coverage_levels], dtype=float)    
        ax = plot_calibration_curve(coverage_pct, 100.0 * empirical_coverage, miny_inset=81, ax=axes[1], **kwargs)
        ref_mape_f, ref_mape_k = plot_fk_scatters(
            saved['f_gt'], saved['k_gt'], saved['means_of_exact_posterior'][:,0], saved['means_of_exact_posterior'][:,1], visualize=False)
        ax.text(0.02, 0.98, txtprefix+f"Reference estimation:\nMAPE = {ref_mape_f:.1f}% (f$_{{ss}}$), {ref_mape_k:.1f}% (k$_{{ssw}}$) ", 
                fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')

def load_UQvsREF_results(data, is_mt=True, sli=0):            
    voxel_timings = data['timings']
    print(fr"Bayes per voxel: {np.mean(voxel_timings):.1f}$\pm{np.std(voxel_timings):.1f}$sec")
    
    print( data['nn_posterior_min_nrmse_percs'])
    try:
        nn_posterior_min_nrmse_percs = np.mean(data['nn_posterior_min_nrmse_percs']) if data['nn_posterior_min_nrmse_percs'] is not None else np.nan    
        exact_posterior_min_nrmse_percs = np.mean(data['exact_posterior_min_nrmse_percs']) if data['exact_posterior_min_nrmse_percs'] is not None else np.nan   
    except: 
        print('Error loading nrmse percs, setting to NaN')
        nn_posterior_min_nrmse_percs = np.nan
        exact_posterior_min_nrmse_percs = np.nan

    f_best_dotprod_arr, k_best_dotprod_arr = zip(*data['means_of_exact_posterior'])

    (f_perc_CIs_intersect, k_perc_CIs_intersect, f_perc_NN_CI_covers_exact, k_perc_NN_CI_covers_exact) = \
    au.analyze_CI_intersections(
        data['f_est'], np.array(f_best_dotprod_arr), 
        data['k_est'], np.array(k_best_dotprod_arr), 
        data['cov_nnpred_scaled'], np.array(data['covs_of_exact_posterior']), data['good_xy_points'], 
        sli=sli, is_mt=is_mt, figsize=(3, 3), #df=0.008, dk=15, fmin=0.07, fmax=0.5
    )
    plt.show()
        
    mahas1 = np.median(data['mahas1'].reshape(-1, 10), axis=1)
    mahas2 = np.median(data['mahas2'].reshape(-1, 10), axis=1)
    # after taking voxelwise typical, find outliers % and typical across voxels
    mahas1_percBAD = 100*np.sum(mahas1>4)/len(mahas1)
    mahas2_percBAD = 100*np.sum(mahas2>4)/len(mahas2)
    mahas1_typical = np.median(mahas1)
    mahas2_typical = np.median(mahas2)

    humanAPT_KLdist_median = np.median(data['KL(P1||P2)'])
    humanAPT_JS_median = np.median(data['Jensen-Shannon Divergence'])
    humanAPT_Hell_median = np.median(data['Hellinger Distance'])       
        
    return {'nn_posterior_min_nrmse_percs': nn_posterior_min_nrmse_percs,
            'exact_posterior_min_nrmse_percs': exact_posterior_min_nrmse_percs,
            'f_perc_CIs_intersect': f_perc_CIs_intersect,
            'k_perc_CIs_intersect': k_perc_CIs_intersect,
            'mahas1_percBAD': mahas1_percBAD,
            'mahas2_percBAD': mahas2_percBAD,
            'mahas1_typical': mahas1_typical,
            'mahas2_typical': mahas2_typical,
            'raw_mahas1': data['mahas1'],
            'raw_mahas2': data['mahas2'],
            'humanAPT_KLdist_median': humanAPT_KLdist_median,
            'humanAPT_JS_median': humanAPT_JS_median,
            'humanAPT_Hell_median': humanAPT_Hell_median,
            }


def meshgrid_xy_pairs(shape):
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    xy_pairs = np.stack([x, y], axis=-1).reshape(-1, 2)
    return xy_pairs.tolist()


def get_all_pdfs(data_feed_mt, sli, seq_name='mt', mt_sim_mode='isar2_c', **kwargs):
    #_x, _y, sli 
    good_xy_points = []
    allpdfs = []
    min_nrms_list, min_nrms_95cr_th_list, CR_areas = [], [], []
    xy_points = meshgrid_xy_pairs(data_feed_mt.shape[0::2])
    try: 
        for _x, _y in xy_points:
            if np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y]):
                continue   
            good_xy_points.append((_x, _y))
            (f_best,k_best, _, nrmse, _df, _dk,) = au.core.get_nrmse_grid(
                None, _x, _y, sli, data_feed_mt,
                seq_name=seq_name, mt_sim_mode=mt_sim_mode,
                **kwargs
                )
            (
                pdf, nrmse_95cr_th, CR_area,
                f_mean_grid_posterior, f_025, f_975,
                k_mean_grid_posterior, k_025, k_975,
                posterior_cov, cdf_at_gt
            ) = au.core.nrmse_to_CIs(nrmse, is_cest=(seq_name!='mt'), df=_df, dk=_dk, gt=None)
            allpdfs.append(pdf)
            min_nrms = np.min(nrmse)
            min_nrms_list.append(min_nrms)
            min_nrms_95cr_th_list.append(nrmse_95cr_th)
            CR_areas.append(CR_area)
    except KeyboardInterrupt as e:
        print("Interrupted by user..")
    print("Processed {} voxels.".format(len(good_xy_points)))
    
    return allpdfs, good_xy_points, min_nrms_list, min_nrms_95cr_th_list, CR_areas

    
# ============== In-vivo helpers ===============

def patient_subfigure(
        data_feed_mt, tissue_params_est, seq_df=None,
        is_cest=False, drop_first=True, mt_sim_mode='isar2_c', data_feed_cest=None,
        sli=27, fig=None, points_ABCD = None, #[(55, 57), (65, 60), (75, 63)],
        cropy1=18, cropy2=10, cropx=7, letters = ('a', 'b', 'c'), max_k_cest=None,
        annotate_cross_mahas=True, colors = ['gold', 'white']
    ):
    fig = fig or  plt.figure(figsize=(14, 4), constrained_layout=True)
    
    au.au_config.mt_sim_mode = mt_sim_mode
    
    shape = tissue_params_est['fc_T'].shape
    seq_df = seq_df or data.read_sequence('mt', drop_first=drop_first) if not is_cest else data.read_sequence('amide', drop_first=drop_first)
        
    points_tumor = [(55, 57), (65, 60), (75, 63)]
    points_contralateral = [(x, shape[2]-y+3) for x,y in  points_tumor]
    points_ABCD = points_ABCD or [points_tumor[0], points_contralateral[0], points_tumor[2], points_contralateral[2]]
    
    subfigs = fig.subfigures(1, 3, width_ratios=[3, 1.5, 1.5], wspace=0.1)
    
    f_est, k_est, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
        au.extract_nn_estimate_of_posterior(tissue_params_est, shape, is_cest=is_cest)        

    axes = au.plot_CI_maps(
        f_est[cropy1:-cropy2, sli, cropx:-cropx], f_sigma[cropy1:-cropy2, sli, cropx:-cropx], 
        k_est[cropy1:-cropy2, sli, cropx:-cropx], k_sigma[cropy1:-cropy2, sli, cropx:-cropx],
        is_cest=is_cest, vmax_k=max_k_cest if is_cest else None,
        fig=subfigs[0], ktop=True
    )    
    for point, label, letter in zip(points_ABCD, [0, 1, 0, 1], ['A', 'B', 'C', 'D']):
        point2plot = np.array(point)-(cropy1, cropx)
        square = patches.Rectangle((point2plot[1]-.6, point2plot[0]-.6), 4.5, 4.5, facecolor='none', edgecolor=colors[label], linewidth=2.5)
        axes[1, 1].add_patch(square)
        axes[1, 1].text(point2plot[1], point2plot[0]-1, letter, color=colors[label], fontsize=15, ha='center')

    slap_label_on_container(subfigs[0], f'({letters[0]})')

    ax, arrow_tips = au.juxtapose_two_voxels_posteriors(
        data_feed_mt, tissue_params_est, shape, 
        points_ABCD[0], points_ABCD[1], fig=subfigs[1], seq_name='mt' if not is_cest else 'amide',
        sli=sli, do_legend=False, seq_df=seq_df, is_cest=is_cest, data_feed_cest=data_feed_cest,
        max_k_cest=max_k_cest, annotate_cross_mahas=annotate_cross_mahas, colors=colors, xytext=(0.6, 0.95)
    )
    if False: # annotate_cross_mahas:
        for atip, letter, color in zip(arrow_tips, ['A', 'B'], colors[::-1]):
            ax.text(atip[0] + (1.5 if letter=='B' else -3), 23,
                    letter, fontweight='bold', color=color, fontsize=15, ha='left', va='center')
        
    slap_label_on_container(subfigs[1], f'({letters[1]})')

    ax, arrow_tips = au.juxtapose_two_voxels_posteriors(
        data_feed_mt, tissue_params_est, shape, 
        points_ABCD[2], points_ABCD[3], fig=subfigs[2],
        sli=sli, do_legend=True, seq_df=seq_df, is_cest=is_cest, data_feed_cest=data_feed_cest, 
        max_k_cest=max_k_cest, annotate_cross_mahas=annotate_cross_mahas,
        letters=['C', 'D'], colors=colors, xytext=(0.6, 0.95)
    )
    if False: # annotate_cross_mahas:
        for atip, letter, color in zip(arrow_tips, ['C', 'D'], colors[::-1]):
            ax.text(atip[0]+1.5, 23,
                    letter, fontweight='bold', color=color, fontsize=15, ha='left', va='center')
    slap_label_on_container(subfigs[2], f'({letters[2]})')


from scipy import ndimage

def find_interior_point(mask):
    """
    Given a binary mask, find an x,y point that's inside
    (removed at least 2 hops from pixels having zero mask value).
    """
    mask = np.copy(mask)
    mask[np.isnan(mask)] = 0  # treat NaNs as outside mask
        
    # Distance transform: each pixel gets distance to nearest 0 in mask
    dist = ndimage.distance_transform_edt(mask>0.8)
    
    # Find points at least 2 pixels away from boundary
    interior = dist >= 2
    
    if not np.any(interior):
        
        # Fallback: find point with maximum distance
        max_dist_idx = np.unravel_index(np.argmax(dist), dist.shape)
        
        plt.figure()
        plt.imshow(mask.astype(int), cmap='gray'); plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(dist, cmap='gray'); plt.colorbar()
        plt.show()
        
        assert 0, "No interior point found in mask!"
        return max_dist_idx
    
    # Get coordinates of interior points
    interior_coords = np.argwhere(interior)
    
    # Return the point with maximum distance (most interior)
    max_dist_in_interior = dist[interior].max()
    interior_point = np.argwhere(dist == max_dist_in_interior)[0]
    
    return tuple(interior_point)


def load_vol_detail(vol_name, seq_name='mt', human_vol_sli=20):
    ''' Load data feed and sample points for a given volunteer
    '''
    data_xa = xr.open_dataset(f'data/{vol_name}.nc' )
    
    gray_mask = data_xa['gray_mask'].to_numpy().copy()[:, human_vol_sli, :]
    white_mask = data_xa['white_mask'].to_numpy().copy()[:, human_vol_sli, :]

    data_feed = data.SlicesFeed.from_xarray(data_xa, seq_name=seq_name)    
    
    # Auto-create sample points as centroids
    gray_mask_top = gray_mask.copy()
    gray_mask_top[:gray_mask_top.shape[0]//2, :] = 0
    white_mask_top = white_mask.copy()
    white_mask_top[:white_mask_top.shape[0]//2, :] = 0
    gray_mask_bot = gray_mask.copy()
    gray_mask_bot[gray_mask_bot.shape[0]//2:, :] = 0
    white_mask_bot = white_mask.copy()
    white_mask_bot[white_mask_bot.shape[0]//2:, :] = 0
    
    points_grey = [find_interior_point(gray_mask_top), find_interior_point(gray_mask_bot)]
    points_white = [find_interior_point(white_mask_top), find_interior_point(white_mask_bot)]

    return data_feed, points_grey, points_white


def load_ms7t_data_feed(subdir='MS7T', filename='MT_data_feed.pkl'):
    ''' Load a SlicesFeed pickled under a top-level `data` module (`import data`)
    rather than `core.data`; alias it so unpickling resolves SlicesFeed correctly
    instead of hitting the `data/` namespace package shadowing it at REPO_ROOT.
    Some attributes aren't plain-pickle-safe, so this uses dill.
    '''
    import dill, sys
    sys.modules['data'] = data

    with open(REPO_ROOT / 'data' / subdir / filename, 'rb') as f:
        return dill.load(f)

    