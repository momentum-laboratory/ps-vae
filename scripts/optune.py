from random import sample
import optuna

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import config, data, infer, pipelines, train
import analyze_uncertainty as au

CKPTS_DIR = REPO_ROOT / "ckpts"
DATA_DIR = REPO_ROOT / "data"

config.Config.configure_mt_amide_preclinical()
data.SlicesFeed.norm_type = 'l2'


def get_lims(data_xa):
    xmin = list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=1) > 0).index(True) - 1
    xmax = 1 - list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=1) > 0)[::-1].index(True)
    ymin = list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=0) > 0).index(True) - 1
    ymax = 1 - list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=0) > 0)[::-1].index(True)
    return xmin, xmax, ymin, ymax


def meshgrid_xy_pairs(shape):
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    xy_pairs = np.stack([x, y], axis=-1).reshape(-1, 2)
    return xy_pairs.tolist()


def dists_closeness(trial, subsampled_test=True):                    
    
    pipelines.pipeline_config.mt_lr =  10**trial.suggest_float('mt_lr_log', -3.5, -1.5)
    pipelines.pipeline_config.cest_lr =  10**trial.suggest_float('cest_lr_log', -3.5, -1.5)    
    train.train_config.sim_seq_mode = trial.suggest_categorical('sim_seq_mode', ['sequential', 'parallel']) 
    train.train_config.reglosstype = trial.suggest_categorical('reglosstype', ['L1', 'L2']) 
    train.train_config.std_up_fact = trial.suggest_float('std_up_fact', 0.05, 0.25)  
    print('---- TRIAL PARAMS: ', trial.params)
    
    # - Prepare data:
    mouse_name = 'ozohar_two_ears_20240404_091158' 

    data_xa = xr.open_dataset(DATA_DIR / f'{mouse_name}.nc')
    xmin, xmax, ymin, ymax = get_lims(data_xa)
    data_xa_cutout = data_xa.isel(
        height=slice(xmin, xmax),
        width=slice(ymin, ymax) 
    )    
    data_xa_cutout['B0_shift_ppm_map'] = (('height','slice','width'), np.zeros_like(data_xa_cutout['T1ms']))
    data_xa_cutout['B1_fix_factor_map'] = (('height','slice','width'), np.ones_like(data_xa_cutout['T1ms']))
    data_feed_mt = data.SlicesFeed.from_xarray(data_xa_cutout, seq_name='mt')
    data_feed_amide = data.SlicesFeed.from_xarray(data_xa_cutout, seq_name='amide')
    
    # - Run the fitting
    ckpt_folder = CKPTS_DIR / 'mouse'
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    predictor_mt, predictor_amide = pipelines.run_train(
        datafeed_mt=data_feed_mt, 
        ckpt_folder=str(ckpt_folder),     
        do_cest=True, datafeed_cest=data_feed_amide 
    ) 

    # - Inference
    data_feed_mt.ds = data_feed_amide.ds = 1    
    mt_tissue_params_est, MT_nn_pred_signal_normed_np = infer.infer(
        data_feed_mt, nn_predictor=predictor_mt, 
        do_forward=True, pool2predict='c'
        )
    amide_tissue_params_est, APT_nn_pred_signal_normed_np = infer.infer(
        data_feed_amide, nn_predictor=predictor_amide, 
        do_forward=True, pool2predict='b'
        )

    # - Compare to exact posteriors
    xy_points = meshgrid_xy_pairs(data_feed_mt.shape[0::2])    
    if subsampled_test:
        xy_points = xy_points[::3]  # subsample for speed

    mahas1, mahas2, CR_areas, nnCR_areas, good_xy_points, \
        modes_of_exact_posterior, means_of_exact_posterior, covs_of_exact_posterior, timings, \
        exact_posterior_min_nrmse_percs, nn_posterior_min_nrmse_percs, extra_metrics = \
        au.compare_posteriors(
            xy_points, data_feed_mt, mt_tissue_params_est, 
            voxels2sample=1000, visualize=False
        )
    mahas1 = np.median(np.reshape(mahas1, (-1,10)), axis=1)
    mahas2 = np.median(np.reshape(mahas2, (-1,10)), axis=1)
    mt_maha1_outlier_frac = sum(np.array(mahas1)>4)/len(np.array(mahas1))
    mt_maha2_outlier_frac = sum(np.array(mahas2)>4)/len(np.array(mahas2))
    
    print('now apt test')
    mahas1, mahas2, CR_areas, nnCR_areas, good_xy_points, \
        modes_of_exact_posterior, means_of_exact_posterior, covs_of_exact_posterior, timings, \
        exact_posterior_min_nrmse_percs, nn_posterior_min_nrmse_percs, extra_metrics = \
        au.compare_posteriors(
            xy_points, data_feed_mt, mt_tissue_params_est, 
            cest_tissue_params_est=amide_tissue_params_est, data_feed_cest=data_feed_amide, is_cest=True, seq_name='amide', 
            voxels2sample=1000, visualize=False
        )     
    mahas1 = np.median(np.reshape(mahas1, (-1,10)), axis=1)
    mahas2 = np.median(np.reshape(mahas2, (-1,10)), axis=1)    
    amide_maha1_outlier_frac = sum(np.array(mahas1)>4)/len(np.array(mahas1))
    amide_maha2_outlier_frac = sum(np.array(mahas2)>4)/len(np.array(mahas2))
    print('---- TRIAL PARAMS: ', trial.params)
    print(f'====>> mt1/2, apt1/2 maha>4 perc: {100*mt_maha1_outlier_frac:.1f}, {100*mt_maha2_outlier_frac:1f}, {100*amide_maha1_outlier_frac:.1f}, {100*amide_maha2_outlier_frac:.1f}')
    worst_frac = max(mt_maha1_outlier_frac, mt_maha2_outlier_frac, amide_maha1_outlier_frac, amide_maha2_outlier_frac)
    mean_frac = np.mean([ mt_maha1_outlier_frac, mt_maha2_outlier_frac, amide_maha1_outlier_frac, amide_maha2_outlier_frac])
    
    return 0.8*worst_frac + 0.2*mean_frac
    

def run_mouse():
    study = optuna.create_study(direction='minimize')
    study.enqueue_trial({
        "mt_lr_log": -2, 
        "cest_lr_log": -2,
        "sim_seq_mode": "sequential",
        "reglosstype": "L1",
        "std_up_fact": 0.1
    })
    study.optimize(dists_closeness, n_trials=50)

    study.best_params 
    
        
def humans_fitness(trial, subsampled_test=True):                    
    config.Config.configure_AMIDE_clinical_whole_brain()    
    
    pipelines.pipeline_config.mt_lr =  10**trial.suggest_float('mt_lr_log', -4, -2)
    train.train_config.std_up_fact = trial.suggest_float('std_up_fact', 0.05, 0.25)        
    infer.infer_config.kc_scale_fact = trial.suggest_float('kc_scale_fact', 60, 100)       
    infer.infer_config.fc_scale_fact = trial.suggest_float('fc_scale_fact', 0.2, 0.35)    
    pipelines.pipeline_config.add_noise_to_signal = trial.suggest_float('add_noise_to_signal', 0.003, 0.03)
    infer.infer_config.predict_k_k = trial.suggest_categorical('predict_k_k', [True, False])    
    train.train_config.weight_decay = trial.suggest_float('weight_decay', 1e-4, 5e-2)   
    train.train_config.sigmoid_shrink = trial.suggest_float('sigmoid_shrink', 1, 10.0)
    train.train_config.hidden_layers = trial.suggest_categorical('hidden_layers', [1,2,3,4])
    print('---- TRIAL PARAMS: ', trial.params)    
    
    TRAINSET = 'vol7-9-10'  
    TESTSET = 'vol8'           
    data_xa_train = xr.open_dataset(f'./data/{TRAINSET}.nc')
    data_feed_mt_train = data.SlicesFeed.from_xarray(data_xa_train, seq_name='mt')
    data_xa_test = xr.open_dataset(f'./data/{TESTSET}.nc')
    data_feed_mt_test = data.SlicesFeed.from_xarray(data_xa_test, seq_name='mt')    

    # - Run the fitting
    ckpt_folder = CKPTS_DIR / 'human_tmp'
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    predictor_mt, _ = pipelines.run_train(
        datafeed_mt=data_feed_mt_train, 
        ckpt_folder=str(ckpt_folder), do_cest=False
    ) 
    
    _a, mt_tissue_param_est, mt_reconstructed_signal, _d, _e, _f = pipelines.transfer(
        data_feed_mt_train, brain2test_cest=None, mt_reconstructor=predictor_mt,
    )
    err_3d = np.linalg.norm(mt_reconstructed_signal - data_feed_mt_train.measured_normed_T, axis=0, ord=2) * data_feed_mt_train.roi_mask_nans
    trainset_mean_nrmse = np.nanmean(err_3d) 
    
    _a, mt_tissue_param_est, mt_reconstructed_signal, _d, _e, _f = pipelines.transfer(
        data_feed_mt_test, brain2test_cest=None, mt_reconstructor=predictor_mt,
    )
    err_3d = np.linalg.norm(mt_reconstructed_signal - data_feed_mt_test.measured_normed_T, axis=0, ord=2) * data_feed_mt_test.roi_mask_nans
    testset_mean_nrmse = np.nanmean(err_3d) 
    print('trainset_mean_nrmse:', trainset_mean_nrmse, 'testset_mean_nrmse:', testset_mean_nrmse)
    
    npz_filename = 'BCK/figs-2026-Jan-25/humans/vol8_w_net_trained_on_vol7-9-10/intermediates/MT_UQ_stats_allpoints.npz'    
    saved = np.load(npz_filename, allow_pickle=True)
    mahas1, mahas2 = au.metrics.compare_to_precomputed_exact_posteriors(
        data_feed_mt_test, saved['good_xy_points'], -30,
        saved['f_est'], saved['k_est'], saved['cov_nnpred_scaled'],
        saved['means_of_exact_posterior'], saved['covs_of_exact_posterior']
    )  
    mahas1 = np.median(np.reshape(mahas1, (-1,10)), axis=1)
    mahas2 = np.median(np.reshape(mahas2, (-1,10)), axis=1)    
    mt_maha1_outlier_frac = sum(np.array(mahas1)>4)/len(np.array(mahas1))
    mt_maha2_outlier_frac = sum(np.array(mahas2)>4)/len(np.array(mahas2))
    
    print(f'====>> mt1/2 maha>4 perc: {100*mt_maha1_outlier_frac:.1f}, {100*mt_maha2_outlier_frac:1f}')
    worst_frac = max(mt_maha1_outlier_frac, mt_maha2_outlier_frac)
    mean_frac = np.mean([ mt_maha1_outlier_frac, mt_maha2_outlier_frac])    
    fitness = 0.8*worst_frac + 0.2*mean_frac  # 0.2*trainset_mean_nrmse + 0.8*testset_mean_nrmse
    print('fitness:', fitness)        
    return fitness


def run_human():
    study = optuna.create_study(direction='minimize')
    study.enqueue_trial({
        "mt_lr_log": -3, 
        #"cest_lr_log": -3,    
        "std_up_fact": 0.1,
        "kc_scale_fact": 70,
        "fc_scale_fact": 0.3,
        "add_noise_to_signal": 0.01,
        "predict_k_k": False,
        "weight_decay": 1e-2,
        "sigmoid_shrink": 3.0,
        "hidden_layers": 2
    })
    study.optimize(humans_fitness, n_trials=50)

    study.best_params
    
    
if __name__ == "__main__":
    run_human()
