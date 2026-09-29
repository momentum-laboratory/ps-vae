import numpy as np, jax
import matplotlib.pyplot as plt
import copy
from dataclasses import asdict, dataclass
import logging
import time
import os

from . import infer, net, train
from helpers import utils


@dataclass
class PipelineConfig:    
    add_noise_to_signal = 1e-3
    mt_lr = 1e-3       
    mt_steps = 2000
    mt_patience = 50    
    cest_lr = 1e-3
    cest_steps = 2000
    # for slow in-vivo cest we want to finish within 10-30min by default
    cest_patience = 1        
    cest_stdup_red = 0.5   # !! 07-08-2025 experiment    
    mt_sim_mode = 'isar2_c'
    
    # batch sizes (set according to available GPU memory)   
    mt_train_slw = 5    
    mt_test_slw = 5  
    cest_train_slw = 1
    cest_test_slw = 2


# To be set externally! make sure notebook cells don't accidentally use the default config by reloading    
pipeline_config = None  # PipelineConfig()


def run_train(
    datafeed_mt, datafeed_cest=None, 
    do_cest=True, mode='self_supervised', mt_sim_mode=None, # 'isar2_c', 
    ckpt_folder='ckpts', ckptsfx='', logger=None, mt_step_callback=None, cest_step_callback=None
    ):
    """ Main training function
    """
    logger = logger or logging.getLogger(__name__)    
    datafeed_mt.slw = min(pipeline_config.mt_train_slw, datafeed_mt.shape[1])
    mt_sim_mode = mt_sim_mode or pipeline_config.mt_sim_mode
    datafeed_mt.add_noise_to_signal = pipeline_config.add_noise_to_signal
    t0 = time.time()
    logger.info("..Starting MT stage training..")
    model_state, loss_trend_mt, net_kwargs = \
        train.train(datafeed_mt, model_state=None, pool2predict='c', logger=logger,
                    mode=mode, simulation_mode=mt_sim_mode,
                    steps=pipeline_config.mt_steps, lr=pipeline_config.mt_lr, patience=pipeline_config.mt_patience,
                    step_callback=mt_step_callback)
    datafeed_mt.add_noise_to_signal = 0.0  # no noise during inference
    t1 = time.time()
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    np.save(f'{ckpt_folder}/mt{ckptsfx}_loss_trend', np.array(loss_trend_mt))
    predictor_mt = net.state2predictor(model_state)
    tc_dict = asdict(train.train_config)
    tc_dict = {k: v for k, v in tc_dict.items() if type(v) != str}  # working around jax issue
    # Note - better to also persist the infer_config alongside the checkpoint

    net.save_ckpt(
        model_state,
        config={'train_cfg': tc_dict, 'net_cfg': net_kwargs},
        folder=f'{ckpt_folder}/mt{ckptsfx}/',
        step=666)  # better to extract the step, stub for now

    if do_cest:
        datafeed_mt.ds = 1        
        datafeed_mt.slw = min(pipeline_config.mt_test_slw, datafeed_mt.shape[1])
        datafeed_mt.add_noise_to_signal = 0.0  # no noise during inference
        mt_tissue_params_est, _ = infer.infer(
            datafeed_mt, nn_predictor=predictor_mt, simulation_mode=mt_sim_mode,
            do_forward=False, pool2predict='c'
        )
        
        # !! use previous stage estimation as next stage ground-truth (possibly used as input to network)
        datafeed_cest.fc_gt_T = mt_tissue_params_est['fc_T']
        datafeed_cest.kc_gt_T = mt_tissue_params_est['kc_T']
        
        datafeed_cest.slw = min(pipeline_config.cest_train_slw, datafeed_cest.shape[1])
        datafeed_cest.add_noise_to_signal = pipeline_config.add_noise_to_signal
        stdup_orig = train.train_config.std_up_fact
        train.train_config.std_up_fact *= pipeline_config.cest_stdup_red  # smaller std-up for CEST,
        
        t2 = time.time()        
        logger.info("..Starting CEST stage training..")
        model_state, loss_trend_cest, net_kwargs = \
            train.train(datafeed_cest, model_state=None, pool2predict='b',
                        mode=mode, simulation_mode='expm_bmmat', logger=logger,
                        steps=pipeline_config.cest_steps, lr=pipeline_config.cest_lr,
                        step_callback=cest_step_callback, patience=pipeline_config.cest_patience)
        
        datafeed_cest.add_noise_to_signal = 0 # no noise during inference
        train.train_config.std_up_fact = stdup_orig  # restore original
        
        t3 = time.time()
        np.save(f'{ckpt_folder}/cest{ckptsfx}_loss_trend',
                np.array(loss_trend_cest))
        tc_dict = asdict(train.train_config)
        tc_dict = {k: v for k, v in tc_dict.items() if type(v) != str}  # working around jax issue
        net.save_ckpt(
            model_state,
            config={'train_cfg': tc_dict, 'net_cfg': net_kwargs},
            folder=f'{ckpt_folder}/cest{ckptsfx}/',
            step=666  # better to extract the step, stub for now
        )
        logger.info(
            f"Timings: t(MTtrain)={t1-t0}sec, t(cesttrain)={t3-t2}sec, t(total-train)={t3-t2+t1-t0}sec, gross-total: {t3-t0}sec")
        predictor_cest = net.state2predictor(model_state)
    else:
        predictor_cest = None

    return predictor_mt, predictor_cest


def errstr(err):
    return  \
        f"L2(err): {np.linalg.norm(err)/np.sqrt(err.size):.3f} " + \
        f"L1(err): {np.linalg.norm(err.flatten(), ord=1)/err.size:.3f}" \
        if type(err) != type(None) else ''


def transfer_and_plot(
    brain2test_mt, brain2test_cest,
    predictor_mt, predictor_cest, 
    do_forward=True, slices=None, 
    do_boxplots=True, mask_gray=None, mask_white=None,
    figsfolder='figs', figsfx='', logger=None, mt_simulation_mode=None, do_slice_plots=True
):
    """ 
    main testing and plotting func
    """
    logger = logger or logging.getLogger(__name__)
    transfer_res = transfer(
        brain2test_mt, brain2test_cest, mt_reconstructor=predictor_mt,
        cest_reconstructor=predictor_cest, do_forward=do_forward,
        mt_simulation_mode=mt_simulation_mode
        )
    mt_tissue_param_est, cest_tissue_param_est = transfer_res[1], transfer_res[4]
    if not os.path.exists(figsfolder):
        os.makedirs(figsfolder)
        
    err_mt, err_cest = plot_slice_rows_wrapper(
        transfer_res, slices=slices,
        mt_fig_name=f'{figsfolder}/MTslices{figsfx}.png',
        cest_fig_name=f'{figsfolder}/cestslices{figsfx}.png',
        do_plot=do_slice_plots
    )    
        
    if do_boxplots:
        boxplots_wrapper(
            mask_gray, mask_white,
            mt_tissue_param_est,
            cest_tissue_param_est,
            f'{figsfolder}/mt_box{figsfx}.png',
            f'{figsfolder}/cest_box{figsfx}.png', mask_th=0.9
        )

    logger.info(f'MT error analysis: {errstr(err_mt)}')
    logger.info(f'cest error analysis: {errstr(err_cest)}')
    return transfer_res, err_mt, err_cest


def transfer(
        brain2test_mt, brain2test_cest, 
        mt_reconstructor, cest_reconstructor=None, 
        infer_config=None, do_forward=True, mt_simulation_mode=None # 'isar2_c'
    ):
    if type(infer_config) != type(None):
        infer.infer_config = infer_config  # = infer_config or pipeline_config.infer_config 
    brain2test_mt.ds = 1
    brain2test_mt.slw = pipeline_config.mt_test_slw
    mt_simulation_mode = mt_simulation_mode or pipeline_config.mt_sim_mode
    
    mt_tissue_param_est, _mt_reconstructed_signal = \
        infer.infer(
            brain2test_mt, pool2predict='c',
            nn_predictor=mt_reconstructor, simulation_mode=mt_simulation_mode,
            do_forward=do_forward
            )
    if cest_reconstructor is None:
        cest_tissue_param_est = _cest_reconstructed_signal = None
    else:
        brain2test_cest.ds = 1
        brain2test_cest.slw = pipeline_config.cest_test_slw  

        # ! Set the parameters of the MT pool to our estimation from fitting the MT protocol:
        brain2test_cest.fc_gt_T = mt_tissue_param_est['fc_T']
        brain2test_cest.kc_gt_T = mt_tissue_param_est['kc_T']

        cest_tissue_param_est, _cest_reconstructed_signal = \
            infer.infer(brain2test_cest, pool2predict='b',
                        nn_predictor=cest_reconstructor, simulation_mode='expm_bmmat',
                        do_forward=do_forward)

    return (brain2test_mt, mt_tissue_param_est, _mt_reconstructed_signal,
            brain2test_cest, cest_tissue_param_est, _cest_reconstructed_signal)


def plot_slice_rows_wrapper(transfer_res,
                            mt_fig_name, cest_fig_name='1',
                            # fss_lims=[0, 20], kss_lims=[0, 70], #
                            fss_lims=[0, 25], kss_lims=[0, 100],
                            fs_lims=[0, 0.7], ks_lims=[0, 500],  # [100, 400],
                            figsize=None, slices=None, do_err=True, do_plot=True):

    brain2test_mt, mt_tissue_param_est, mt_reconstructed_signal, \
        brain2test_cest, cest_tissue_param_est, cest_reconstructed_signal = transfer_res

    fss_pred = mt_tissue_param_est[f'fc_T'] * brain2test_mt.roi_mask_nans * 100
    kss_pred = mt_tissue_param_est[f'kc_T'] * brain2test_mt.roi_mask_nans
    err_3d = np.linalg.norm(mt_reconstructed_signal - brain2test_mt.measured_normed_T,
                            axis=0, ord=2) * brain2test_mt.roi_mask_nans
    # ! nontrivial if signal is normed-by-first
    err_3d /= np.linalg.norm(brain2test_mt.measured_normed_T, axis=0, ord=2)
    err_3d[np.isnan(err_3d)] = 0
    
    if do_plot:
        utils.slice_row_plot(
            fss_pred, kss_pred, err_3d,
            fss_lims=fss_lims, kss_lims=kss_lims,
            figsize=figsize, slices=slices, do_err=do_err
            )
        plt.savefig(mt_fig_name, bbox_inches='tight')
    err_3d_mt = err_3d

    if cest_tissue_param_est is not None:
        fss_pred = cest_tissue_param_est[f'fb_T'] * \
            brain2test_cest.roi_mask_nans * 100
        kss_pred = cest_tissue_param_est[f'kb_T'] * \
            brain2test_cest.roi_mask_nans
        err_3d = np.linalg.norm(cest_reconstructed_signal - brain2test_cest.measured_normed_T,
                                axis=0, ord=2) * brain2test_cest.roi_mask_nans
        # ! nontrivial if signal is normed-by-first
        err_3d /= np.linalg.norm(
            brain2test_cest.measured_normed_T,
            axis=0, ord=2
            )
        err_3d[np.isnan(err_3d)] = 0
        full_texts = [
            'cest volume fraction (%)', '$f_{s}$ (%)',
            'cest exchange rate (Hz)', '$k_{sw}$ $(s^{-1})$',
            'signal reconstruction fidelity', '$R^2_{fit}$',
        ]
        simple_texts = ['', '$f_{s}$ (%)',
                        '', '$k_{sw}$ $(s^{-1})$',
                        '', '$R^2_{fit}$',
                        ]
        if do_plot:
            utils.slice_row_plot(
                fss_pred, kss_pred, err_3d, fss_lims=fs_lims, kss_lims=ks_lims,
                figsize=figsize, slices=slices, do_err=do_err, texts=simple_texts
            )
            plt.savefig(cest_fig_name, bbox_inches='tight')
        # texts=)
        err_3d_cest = err_3d
    else:
        err_3d_cest = None

    return err_3d_mt, err_3d_cest,


def boxplots_wrapper(
    mask_gray, mask_white, mt_tissue_param_est, cest_tissue_param_est,
    mt_fig_name, cest_fig_name, mask_th=0.9
):
    """ Draw the boxplot comparing fitting results to literature
    """
    mask_gray[mask_gray == 0] = np.nan
    mask_white[mask_white == 0] = np.nan

    boxplot_mt = utils.boxplot_white_vs_gray(
        mask_gray, mask_white,
        mt_tissue_param_est['fc_T'],
        mt_tissue_param_est['kc_T'],
        pool='MT',
        lit_f_wm_gm=[[13.9, 5], [6.2, 3.4], [
            11.2, 6.3], [9.4, 4.2], [18.7, 12.4]],
        lit_k_wm_gm=[[23.0, 40.0], [67.5, 63.5], [
            29, 40], [14, 35.1], [33.9, 49.1]],
        lit_names=('Stanitz 2005', 'Liu 2013', 'Heo 2019', 'Perlman 2022', 'Weigand-Whittier 2022'))[0]
    plt.savefig(mt_fig_name, bbox_inches='tight')

    if cest_tissue_param_est is not None:
        boxplot_cest = utils.boxplot_white_vs_gray(
            mask_gray, mask_white,
            cest_tissue_param_est['fb_T'],
            cest_tissue_param_est['kb_T'],
            pool='cest',
            lit_f_wm_gm=[[.19, .24], [.22, .25], [.31, .32], [.1, .17]],
            lit_k_wm_gm=[[162, 365], [280, 280], [42.3, 35], [260, 130]],
            lit_names=('Heo 2019', 'Liu 2013', 'Perlman 2022', 'Carradus 2023')
            )[0]
        plt.savefig(cest_fig_name, bbox_inches='tight')
    

def get_logger(fname):    
    logging.basicConfig(        
        level=logging.INFO,        
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',  # Date format
        handlers=[
            logging.FileHandler(fname),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    return logging.getLogger(__name__)    


def simple_infer_wrap(ds, nn_predictor, simulation_mode='expm_bmmat', visualize=False, sloi=5):    
    tissue_params, nn_pred_signal_normed_np = infer.infer(
        ds, nn_predictor=nn_predictor, 
        do_forward=True, simulation_mode=simulation_mode
        )
    nsignal_error = np.linalg.norm(
        nn_pred_signal_normed_np - 
        ds.measured_normed_T, axis=0, ord=2
        ) * ds.roi_mask_nans
    nsignal_error /= np.linalg.norm(
        ds.measured_normed_T, axis=0, ord=2
        )  # norm by the original signal
    print(f'100*np.nanmean(err_2d): {100*np.nanmean(nsignal_error):.3f}')
    print(f'100*L2(err_2d): {100 * np.sqrt(np.nanmean(nsignal_error**2)):.3f}')
    print(f'100*median(err_2d): {100 * np.nanmedian(nsignal_error):.3f}')
    if visualize:
        plt.figure(figsize=(10,2))
        plt.suptitle("signal pred/meas RRMSE (%) - MAP  /  HISTOGRAM")
        plt.subplot(1,2,1)
        plt.imshow(nsignal_error[:,sloi,:]*100, vmin=0, vmax=5, cmap='hot')
        plt.colorbar()        
        plt.subplot(1,2,2)
        try:
            plt.hist(nsignal_error.flatten()*100, 32)        
            plt.xlabel('signal pred/meas RRMSE (%)')
        except:
            print("histogram plotting failed")
    return tissue_params, nn_pred_signal_normed_np, nsignal_error

    
def VBMF_MT_run(brain2train_mt, dirname='vbmf', mt_steps=None, slices2plot=None):
    run_dirname = os.path.join(dirname, time.strftime('%B%d_%H%M')) 
    if not os.path.exists(run_dirname):
        os.makedirs(run_dirname)
    logger = get_logger(f'{dirname}/log.log')        
        
    infer.infer_config = pipeline_config.infer_config
    train.train_config = pipeline_config.train_config
    train.train_config.tp_noise = False  
    train.train_config.use_shuffled_sampler = False
    train.train_config.patience = pipeline_config.mt_patience
    mt_steps = mt_steps or pipeline_config.mt_steps
    
    # Works arounds exotic Jax bug in the VBMF+CEST case
    jax.config.update("jax_enable_x64", True)
    
    mt_lr = 1e-2
    t0 = time.time()        
    
    model_state, loss_trend_mt, net_kwargs = \
        train.train(brain2train_mt, model_state=None, pool2predict='c', logger=None,                                
                    mode='bloch_fitting', simulation_mode='isar2_c', steps=mt_steps, lr=mt_lr)
    logger.info(f"T(VBMF)={time.time()-t0}sec")
    # To evaluate, we set the fitted values as "ground truth" on a copy dataset
    brain2test_mt = copy.deepcopy(brain2train_mt)    
    train.bmfit_set_gt_from_model_state(brain2test_mt, model_state)  
    mt_tissue_param_est, _mt_reconstructed_signal = infer.infer(
        brain2test_mt, pool2predict='bc', 
        nn_predictor=None, simulation_mode='isar2_c', do_forward=True
        )  

    brain2test_cest = None; cest_tissue_param_est = _cest_reconstructed_signal = None
    transfer_res = (brain2test_mt, mt_tissue_param_est, _mt_reconstructed_signal, \
                    brain2test_cest, cest_tissue_param_est, _cest_reconstructed_signal)
        
    err_mt, err_cest = plot_slice_rows_wrapper(
        transfer_res, slices=slices2plot, mt_fig_name=run_dirname+'/mt_VBMF.png'
        )
    np.savez_compressed(run_dirname+'/err', err_mt_train=err_mt, mt_tissue_param_est=mt_tissue_param_est)
    return mt_tissue_param_est, err_mt

if __name__ == "__main__":
    VBMF_MT_run()
