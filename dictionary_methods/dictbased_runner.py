import logging

import numpy as np

from core import data, infer, net, pipelines, train
import dictionary_methods.dict_matcher as dict_matcher

def get_dict(
    seq_name='mt', seq_df=None, mt_sim_mode='isar2_c', use_cartesian = False, lhs_size_x10k=40,
    import_NV=False, nv_mt_fname='mtd.npz', nv_amide_fname='aptd.npz', **kwargs
    ):
    """ Create dictionary out of given ranges/grids, either cartesian (product of grids)
        or Latin-Hypercube Sampling over product ranges. Alternatively, load external. 

    """
    if not import_NV:
        if use_cartesian:
            synthetic_datafeed = data.SlicesFeed.make_cartesian_sample(
                seq_name=seq_name, seq_df=seq_df, **kwargs
                ) 
        else:  # Latin Hypercube Sampling
            slices = lhs_size_x10k # if mt_or_amide=='mt' else lhs_size_x10k
            synthetic_datafeed = data.SlicesFeed.make_lh_sample( 
                seq_name=seq_name, seq_df=seq_df, 
                shape=(100, slices, 100), **kwargs
            )   
        # Synthesize the signals:
        synthetic_datafeed.slw = 20 if mt_sim_mode=='isar2_c' else 1
        _, signal_3p = infer.infer(
            synthetic_datafeed, 
            simulation_mode=mt_sim_mode if seq_name=='mt' else 'expm_bmmat'
        )
        synthetic_datafeed.measured_normed_T = synthetic_datafeed.normalize(signal_3p)
    
    else: # import from dictionary created with N.Vladimirov2025,NatureProtocols code
        """
        tested once by creating mtd.npz as follows:
        run python -m sequential_nn_example.mouse with some modifications.
        1. sequential_nn_example/mouse.py::write_seq_defs -
            add B1=0 reference prescan after which come 15s delay, set B0=3, set pulsed (13,0.1/0.1) 
        2. sequential_nn_example/configs.py  - change grid.
        3: breakpoint in sequential_nn_example/mouse.py, line 58: to then run in debugger:
            np.savez_compressed('/hosthome/alexf/nbmf/mtd', dictionary)        
            [older experiments-] inclusion of zero (or very small fs) leads to very bad results in training
        """
        nvdict = np.load(
            nv_mt_fname if seq_name=='mt' else nv_amide_fname, 
            fix_imports=True, allow_pickle=True
            )['arr_0'].item()
        shape = (88, 50, 90) # nvdict['sig'].reshape(31, *shape).shape
        synthetic_datafeed = data.SlicesFeed.from_args(
            seq_name=seq_name, seq_df=seq_df,
            shape=shape, 
            T1a_ms=nvdict['t1w'].reshape(shape)*1000,
            T2a_ms=nvdict['t2w'].reshape(shape)*1000,
            fb_gt_T=0, kb_gt_T=0,
            fc_gt_T=nvdict['fs_0'].reshape(shape),
            kc_gt_T=nvdict['ksw_0'].reshape(shape),
            signal=nvdict['sig'].T.reshape(31,*shape),
        )

    return synthetic_datafeed


def train_on_dict(bsf_lhsynth, pool2predict='c', epochs=50):
    infer.infer_config = pipelines.pipeline_config.infer_config
    train.train_config = pipelines.pipeline_config.train_config
    train.train_config.patience=100  
    train.train_config.tp_noise = False
    train.train_config.use_shuffled_sampler = True
    train.train_config.reglosstype = 'L2'
    bsf_lhsynth.slw = 1     
    bsf_lhsynth.add_noise_to_signal = 5e-3
    # the amp of loss is de-facto x sqrt(30), so roughly 2.5% 

    logger = logging.getLogger('train')
    steps = bsf_lhsynth.shape[1]*epochs
    model_state, loss_trend_mt, net_kwargs = \
            train.train(bsf_lhsynth, model_state=None, 
                        pool2predict=pool2predict, logger=logger,
                        mode='reference_supervised', steps=steps, lr=3e-3)    

    predictor = net.state2predictor(model_state)
    return predictor
    

def match_to_dict(sig_dict_datafeed, signal_under_test_datafeed, 
                  constrain_T1T2=True, constrain_B0B1=False, quicky_slicer=False,
                  pool2match='c'
    ):
    sig_len = signal_under_test_datafeed.measured_normed_T.shape[0]
    l2normed_d = sig_dict_datafeed.measured_normed_T.reshape(sig_len, -1) + 0.0
    l2normed_d /= np.linalg.norm(l2normed_d, axis=0)

    afdm = dict_matcher.DictMatcher(
        l2normed_d, 
        T1_dict=1000/sig_dict_datafeed.R1a_V.flatten(), 
        T2_dict=1000/sig_dict_datafeed.R2a_V.flatten(),
        B0_dict=sig_dict_datafeed.B0_shift_ppm_map.flatten(),
        B1_dict=sig_dict_datafeed.B1_fix_factor_map.flatten(),
    ) 
    
    if quicky_slicer:
        demo_slices = np.int32(np.linspace(5,40,10))[3:-2]
        for k, v in signal_under_test_datafeed.__dict__.items():
            if type(v) == np.ndarray:
                print(k, v.shape, signal_under_test_datafeed.shape, signal_under_test_datafeed.shape==list(v.shape))
                if v.shape == tuple(signal_under_test_datafeed.shape):
                    signal_under_test_datafeed.__dict__[k] = v[:, demo_slices, :]
                elif v.shape[1:] == tuple(signal_under_test_datafeed.shape):
                    signal_under_test_datafeed.__dict__[k] = v[:, :, demo_slices, :]
        signal_under_test_datafeed.shape = [signal_under_test_datafeed.shape[0], 5, signal_under_test_datafeed.shape[2]]
    signal_under_test_datafeed.measured_normed_T.shape
    
    l2normed_signal = signal_under_test_datafeed.measured_normed_T + 0.
    l2normed_signal /= np.linalg.norm(l2normed_signal, axis=0)

    B0constraint = signal_under_test_datafeed.B0_shift_ppm_map if constrain_B0B1 else None
    B1constraint = signal_under_test_datafeed.B1_fix_factor_map if constrain_B0B1 else None
    
    if pool2match == 'c':
        fs_grid=sig_dict_datafeed.fc_gt_T, 
        ks_grid=sig_dict_datafeed.kc_gt_T
    else:
        fs_grid=sig_dict_datafeed.fb_gt_T, 
        ks_grid=sig_dict_datafeed.kb_gt_T
        
    afdm_match_res = afdm.match(
        l2normed_signal, 
        fs_grid=fs_grid,
        ks_grid=ks_grid,
        T1constraint=1000/signal_under_test_datafeed.R1a_V if constrain_T1T2 else None,
        T2constraint=1000/signal_under_test_datafeed.R2a_V if constrain_T1T2 else None,
        B0constraint=B0constraint,
        B1constraint=B1constraint
        # When adapting for CEST, might similarly consider fss, kss constraints for finding fs, ks
    )
    
    fss_pred = afdm_match_res[0] * 100 * signal_under_test_datafeed.roi_mask_nans
    kss_pred = afdm_match_res[1] *       signal_under_test_datafeed.roi_mask_nans
    best_match = l2normed_d[:, afdm_match_res[2]] * signal_under_test_datafeed.roi_mask_nans
    best_match /= np.max(best_match, axis=0) 

    err_3d = np.linalg.norm(sig_dict_datafeed.normalize(best_match) - 
                            sig_dict_datafeed.normalize(signal_under_test_datafeed.measured_normed_T), axis=0, ord=2)
    err_3d /= np.linalg.norm(sig_dict_datafeed.normalize(signal_under_test_datafeed.measured_normed_T), axis=0, ord=2)
    err_3d[np.isnan(err_3d)] = 0
    return fss_pred, kss_pred, best_match, err_3d


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