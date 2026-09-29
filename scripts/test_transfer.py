

from pathlib import Path
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from importlib import reload

import jax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

from core import config, data, infer, net, pipelines
from helpers import utils
import analyze_uncertainty as au


FIGS_DIR = REPO_ROOT / "figs"
CKPTS_DIR = REPO_ROOT / "ckpts"
FIGS_DIR.mkdir(exist_ok=True)
CKPTS_DIR.mkdir(exist_ok=True)

config.Config.configure_AMIDE_clinical_whole_brain()    
mt_sim_mode = 'isar2_c'           # faster
data.SlicesFeed.norm_type = 'l2'  # consistent with dot-product

all_vol_IDs = [7,8,9,10]



def drawmyblandaltman(_nn_arr, _nn_sigma_arr, _grid_sigma_arr, _best_dotprod_arr, is_mt, is_f):
    
    for nsigmas in [2,3,4]:
        zz = np.logical_and(_nn_arr - _best_dotprod_arr - nsigmas*_nn_sigma_arr < 0, 
                            _nn_arr - _best_dotprod_arr + nsigmas*_nn_sigma_arr > 0)
        print(f'Fraction of points with {nsigmas}-sigma CI crossing zero: {100*np.sum(zz)/len(zz):.1f}% out of {len(zz)}')
        zz = np.logical_and(
            _nn_arr - _best_dotprod_arr - nsigmas*_nn_sigma_arr - nsigmas*_grid_sigma_arr < 0, 
            _nn_arr - _best_dotprod_arr + nsigmas*_nn_sigma_arr + nsigmas*_grid_sigma_arr > 0
        )
        print(f'Fraction of points with {nsigmas}-sigma have CI of the 2 methods overlapping: {100*np.sum(zz)/len(zz):.1f}% out of {len(zz)}')
    _sort_inds = np.argsort(_best_dotprod_arr)
    if is_mt:
        ranges = (2, 30, 0.6) if is_f else (20, 60, 0.5)
    else:
        ranges = (0.05, 1, 0.01) if is_f else (0, 500, 10)
    inds_into_sorted = np.searchsorted(np.sort(_best_dotprod_arr), np.arange(*ranges))-1
    _best_dotprod_arr = np.array(_best_dotprod_arr)[_sort_inds][inds_into_sorted]
    print(_best_dotprod_arr)

    _nn_arr = _nn_arr[_sort_inds][inds_into_sorted] # [inds]
    _nn_sigma_arr = _nn_sigma_arr[_sort_inds][inds_into_sorted] # [inds]
    _grid_sigma_arr = _grid_sigma_arr[_sort_inds][inds_into_sorted] # [inds]

    plt.plot(_best_dotprod_arr, _nn_arr - _best_dotprod_arr, 'b.') #, alpha=0.5) #, markersize=2)  #  label='NN-est - best dotprod',
    for jj in range(len(_nn_arr)):     
        plt.plot(
            [_best_dotprod_arr[jj], _best_dotprod_arr[jj]], 
            [_nn_arr[jj] - _best_dotprod_arr[jj] - 2*_nn_sigma_arr[jj], _nn_arr[jj] - _best_dotprod_arr[jj] + 2*_nn_sigma_arr[jj]],
            'b-', alpha=0.4, markersize=2)
        eps = (ranges[2])*0.2
        plt.plot(
            [_best_dotprod_arr[jj]+eps, _best_dotprod_arr[jj]+eps], 
            [ - 2*_grid_sigma_arr[jj], 0 + 2*_grid_sigma_arr[jj]],
            'k-', alpha=0.4, markersize=1)  
    
    if is_f:    
        plt.legend(['Estimates\' discrepancy', 'Confidence Intervals'])        
        plt.plot(([0, 25] if is_mt else [0, 1]), [0, 0], 'k-', linewidth=2, alpha=0.3, label='y=x')
        plt.ylim(*((-10, 10) if is_mt else (-0.5, 0.5)))
        plt.xlim(*((3.2, 18.8) if is_mt else (0.05, 0.55)))
        plt.xlabel(r'$\hat{f}_{ss}\ (\%)$' if is_mt else r'$\hat{f}_{s}\ (\%)$')
        plt.ylabel(r'$ \Delta \hat{f}_{ss} (\%)$' if is_mt else r'$ \Delta \hat{f}_{s} (\%)$')
    else:
        plt.plot(([0, 80] if is_mt else [0, 500]), [0, 0], 'k-', linewidth=2, alpha=0.3, label='y=x')
        plt.ylim(*((-40, 40) if is_mt else (-250, 250)))
        plt.xlim(*((1.5, 52.5) if is_mt else (0.0, 500)))
        plt.xlabel(r'$\hat{k}_{ssw}\ (s^{-1})$' if is_mt else r'$\hat{k}_{sw}\ (s^{-1})$')
        plt.ylabel(r'$ \Delta \hat{k}_{ssw} (s^{-1})$' if is_mt else r'$ \Delta \hat{k}_{sw} (s^{-1})$')
        

def analyze_CIs(
        data_feed_mt, f_est, k_est, mt_tissue_params_est, cov_nnpred_scaled, 
        figdir: Path, sli=15, is_mt=True, num_voxels2sample=5000
    ):
    mahas1, mahas2, mahas1_1D, mahas2_1D = [], [], [], []
    f_best_dotprod_arr, k_best_dotprod_arr, posterior_cov_arr = [], [], []
    f_mean_dotprod_arr, k_mean_dotprod_arr = [], []
    nn_nrmses, min_nrmses = [], []

    voxels2sample = num_voxels2sample # 500
    xy_points = np.int32(np.random.random((1000, 2)) * data_feed_mt.shape[0::2]) 
    good_xy_points = []
    good_points_done = 0    
    for [_x, _y] in xy_points:
        if np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y]):
            continue    
        good_points_done += 1
        if good_points_done > voxels2sample:
            break
        good_xy_points.append([_x, _y])
        try:
            f_best_dotprod, k_best_dotprod, dict_signal, nrmse, _df, _dk = au.get_nrmse_grid(
                None, _x, _y, sli, 
                data_feed_mt, mt_tissue_params_est,
                data_feed_cest=None, seq_name='mt',
                mt_sim_mode=mt_sim_mode, do_plot=False
            )        
        except Exception as e:
            print(e)
            continue
        
        nn_nrmses.append(100*nrmse[int(np.clip(f_est[_x, sli, _y]/100/_df, 0, 99)), int(np.clip(k_est[_x, sli, _y]/_dk, 0, 99))])
        min_nrmses.append(100*np.min(nrmse))
        print(f'np.min(nrmse): {100*np.min(nrmse):.2f}%')  
        print(f'nrmse[NN-est]: {nn_nrmses[-1]:.2f}%')          
        
        maha1, maha2, posterior_cov, CR_area, CIk_x_CIf, (f_mu_post, k_mu_post) = au.viz_posteriors(
            f_est[_x, sli, _y], k_est[_x, sli, _y], 
            cov_nnpred_scaled[_x, sli, _y],
            1,
            1,
            1,
            f_best_dotprod, k_best_dotprod, nrmse, _df, _dk,        
            is_cest=False, do_marginals=True, visualize=False        
            )    
        plt.tight_layout()
        mahas1.append(maha1)
        mahas2.append(maha2)        
        f_best_dotprod_arr.append(f_best_dotprod)
        k_best_dotprod_arr.append(k_best_dotprod)
        posterior_cov_arr.append(posterior_cov)
        f_mean_dotprod_arr.append(f_mu_post)
        k_mean_dotprod_arr.append(k_mu_post)    
        mahas1_1D.append([
            np.abs(f_est[_x, sli, _y] - f_best_dotprod)/np.sqrt(posterior_cov[0][0]),
            np.abs(k_est[_x, sli, _y] - k_best_dotprod)/np.sqrt(posterior_cov[1][1])
        ])
        mahas2_1D.append([
            np.abs(f_est[_x, sli, _y] - f_best_dotprod)/np.sqrt(cov_nnpred_scaled[_x, sli, _y][0][0]),
            np.abs(k_est[_x, sli, _y] - k_best_dotprod)/np.sqrt(cov_nnpred_scaled[_x, sli, _y][1][1])
        ])
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    _nn_arr = np.array([f_est[_x, sli, _y] for (_x, _y) in good_xy_points]) # if not np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y])], dtype=np.float32)
    _nn_sigma_arr = np.array([np.sqrt(cov_nnpred_scaled[_x, sli, _y][0][0]) for (_x, _y) in good_xy_points]) # if not np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y])], dtype=np.float32)
    _grid_sigma_arr = np.array([np.sqrt(grid_cov[0][0]) for grid_cov in posterior_cov_arr])
    _best_dotprod_arr = np.copy( np.array(f_best_dotprod_arr) )  # f_mean_dotprod_arr f_best_dotprod_arr
    drawmyblandaltman(_nn_arr, _nn_sigma_arr, _grid_sigma_arr, _best_dotprod_arr, False, True)

    plt.subplot(2,1,2)
    _nn_arr = np.array([k_est[_x, sli, _y] for (_x, _y) in good_xy_points]) # if not np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y])], dtype=np.float32)
    _nn_sigma_arr = np.array([np.sqrt(cov_nnpred_scaled[_x, sli, _y][1][1]) for (_x, _y) in good_xy_points]) # if not np.isnan(data_feed_mt.roi_mask_nans[_x, sli, _y])], dtype=np.float32)
    _grid_sigma_arr = np.array([np.sqrt(grid_cov[1][1]) for grid_cov in posterior_cov_arr])
    _best_dotprod_arr = np.copy( np.array(k_best_dotprod_arr) )  # f_mean_dotprod_arr f_best_dotprod_arr

    drawmyblandaltman(_nn_arr, _nn_sigma_arr, _grid_sigma_arr, _best_dotprod_arr, False, False)
    plt.savefig(figdir / 'bland_altman_mt.png', bbox_inches='tight', dpi=300)
    print('Voxels processed: ',good_points_done)
    
    
def main(load_pretrained=False):
    for test_vol_ID in all_vol_IDs:
        sfx = 'vol' + '-'.join([str(x) for x in all_vol_IDs if x!=test_vol_ID])
        xarr_path = REPO_ROOT / 'data' / f'{sfx}.nc'
        figdir = FIGS_DIR / 'leave1out' / str(test_vol_ID)
        figdir.mkdir(parents=True, exist_ok=True)
            
        # Load data
        data_xa = xr.open_dataset(xarr_path)
        ckpts = CKPTS_DIR / sfx
        data_feed_mt_train = data.SlicesFeed.from_xarray(data_xa, seq_name='mt')     
        data_xa_test = xr.open_dataset(REPO_ROOT / 'data' / f'vol{test_vol_ID}.nc') 
        data_feed_mt_test = data.SlicesFeed.from_xarray(data_xa_test, seq_name='mt') 
        
        # Training on all-except-one volumes
        
        if load_pretrained:
            predictor_mt = net.load_ckpt(folder=str(ckpts / 'mt'))[0]
        else:
            shutil.rmtree(ckpts / 'mt', ignore_errors=True)    
            predictor_mt, predictor_amide = pipelines.run_train(
                datafeed_mt=data_feed_mt_train, 
                datafeed_cest=None, 
                ckpt_folder=str(ckpts),     
                do_cest=False,
                mt_sim_mode = mt_sim_mode
            )

        # Inference on left-out test volume
        data_feed_mt_test.slw=5
        transfer_res, err_mt, err_amide = pipelines.transfer_and_plot(
            brain2test_mt=data_feed_mt_test, brain2test_cest=None, 
            predictor_mt=predictor_mt, predictor_cest=None, 
            figsfolder=str(figdir), figsfx='mt', slices=None,
            mask_gray=data_xa_test.gray_mask.values, mask_white=data_xa_test.gray_mask.values,
            mt_simulation_mode=mt_sim_mode
        )        
        mt_tissue_params_est = transfer_res[1]
        f_est, k_est, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
            au.extract_nn_estimate_of_posterior(mt_tissue_params_est, data_feed_mt_test.shape, is_cest=False)
    
        # Evaluation
        analyze_CIs(data_feed_mt_test, f_est, k_est, mt_tissue_params_est, cov_nnpred_scaled, figdir=figdir)
        
        
if __name__ == '__main__':
    main()