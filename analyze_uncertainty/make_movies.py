import os, time, shutil
import numpy as np, matplotlib.pyplot as plt
import xarray as xr
import matplotlib.animation as animation

from core import data, infer, net, pipelines, train
from helpers import utils

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


class Maps_Collector():
    slice_in_batch = 1         
    def __init__(        
            self, pool, shape, ds=2, slw=1, steps=2000,
            slice_in_slab_to_follow=0, slab2follow=1
        ): # ds=2, epochs=125, num_slices=4):
        self.f_key = 'fb_T' if pool == 'b' else 'fc_T'
        self.k_key = 'kb_T' if pool == 'b' else 'kc_T'        
        self.pool = pool         
        self.slab2follow = slab2follow 
        self.slice_in_slab_to_follow = slice_in_slab_to_follow
        self.shape = shape
        self.ds = ds
        self.slw = slw
        self.pseudo_epoch_steps = shape[1] // slw
        self.steps_per_epoch = ds**2 * self.pseudo_epoch_steps # slice_feed.shape[1] // slice_feed.slw
        self.epochs = steps // self.steps_per_epoch
        print(f'{self.epochs} epochs')
        
        x_len, y_len = int(shape[0]), int(shape[2])
        self.f_mu = np.zeros((x_len, y_len, self.epochs))
        self.k_mu = np.zeros((x_len, y_len, self.epochs))
        self.f_noised = np.zeros((x_len, y_len, self.epochs))
        self.k_noised = np.zeros((x_len, y_len, self.epochs))
        self.noised_fk_nrmse_perc = np.zeros((x_len, y_len, self.epochs))
        self.clean_fk_nrmse_perc = np.zeros((x_len, y_len, self.epochs))
        
    def collect(self, tissue_params, step, xoyozo):                
        (xo, yo, zo) = xoyozo
        xo, yo, zo = xo.item(), yo.item(), zo.item()
        ds = self.ds        
        _zo = (step // ds**2) % self.pseudo_epoch_steps 
        assert _zo == zo, f"{zo}!={_zo}"
        if _zo != self.slab2follow:
            return  True
        epoch = step // self.steps_per_epoch
        jj = self.slice_in_slab_to_follow
        f_noised = tissue_params[self.f_key][:,jj,:]._value*100 #, cmap="viridis", vmin=0, vmax=30 if self.pool=='c' else 1);         
        k_noised = tissue_params[self.k_key][:,jj,:]._value     # , cmap="magma", vmin=0, vmax=100 if self.pool=='c' else 500)        
        f_mu = tissue_params[self.f_key.replace('_T', '_clean')][:,jj,:]._value*100
        k_mu = tissue_params[self.k_key.replace('_T', '_clean')][:,jj,:]._value        
        noised_fk_nrmse_perc = 100 * tissue_params['signal_est_diffnorm_map'][:,jj,:]._value
        clean_fk_nrmse_perc =  100 * tissue_params['clean_signal_est_diffnorm_map'][:,jj,:]._value # , cmap="hot_r", vmin=0, vmax=10) #0.1);     
        self.f_mu[xo::ds, yo::ds, epoch] = f_mu
        self.k_mu[xo::ds, yo::ds, epoch] = k_mu
        self.f_noised[xo::ds, yo::ds, epoch] = f_noised
        self.k_noised[xo::ds, yo::ds, epoch] = k_noised
        self.noised_fk_nrmse_perc[xo::ds, yo::ds, epoch] = noised_fk_nrmse_perc
        self.clean_fk_nrmse_perc[xo::ds, yo::ds, epoch] = clean_fk_nrmse_perc
        
        return True
    

class McAnimator():
    
    def __init__(self, mc):
        self.fig, self.axes = plt.subplots(2, 4, figsize=(24,10))    
        self.txt1, self.txt2, self.txt3 = None, None, None
        self.mc = mc
        self.artists = []
        
    def run(self, epochs):
        self.mc_plot_2x4(0, first_run=True) #, fig=None, axes=None):
        for ep in epochs: 
            artists = self.mc_plot_2x4(ep)
            self.artists.append(artists)

    def mc_plot_2x4(self, epoch, first_run=False): #, fig=None, axes=None):
        #if fig is None:
        if not first_run: # self.txt1 is not None:
            self.txt1.set_visible(False)
            self.txt2.set_visible(False)
            self.txt3.set_visible(False)
            
        axes = self.axes
        mc = self.mc 
        f_imsh_clean = axes[0,0].imshow(mc.f_mu[:,:,epoch], cmap="viridis", vmin=0, vmax=30 if mc.pool=='c' else 1);     
        _=plt.colorbar(f_imsh_clean) if first_run else None   
        k_imsh_clean = axes[0,1].imshow(mc.k_mu[:,:,epoch], cmap="magma", vmin=0, vmax=100 if mc.pool=='c' else 500)        
        _=plt.colorbar(k_imsh_clean) if first_run else None   
        nrmse_imsh_clean = axes[0,2].imshow(mc.clean_fk_nrmse_perc[:,:,epoch], cmap="hot_r", vmin=0, vmax=10) #0.1)
        _=plt.colorbar(nrmse_imsh_clean) if first_run else None     
        
        uT = 1
        w1 = uT * 42.58 * 2*np.pi 
        R2b = 1 / 40e-3
        kk = mc.k_mu[:,:,epoch]
        Rex = mc.f_mu[:,:,epoch]/100 * kk * w1**2 / (w1**2 + kk*(kk+R2b) )    
        imsh_rex_clean = axes[0, 3].imshow(Rex, vmin=0, vmax=1)                 
        _=plt.colorbar(imsh_rex_clean) if first_run else None  
                
        f_imsh = axes[1,0].imshow(mc.f_noised[:,:,epoch], cmap="viridis", vmin=0, vmax=30 if mc.pool=='c' else 1);            
        _=plt.colorbar(f_imsh) if first_run else None   
        k_imsh = axes[1,1].imshow(mc.k_noised[:,:,epoch], cmap="magma", vmin=0, vmax=100 if mc.pool=='c' else 500)                   
        _=plt.colorbar(k_imsh) if first_run else None     
        nrmse_imsh = axes[1,2].imshow(mc.noised_fk_nrmse_perc[:,:,epoch], cmap="hot_r", vmin=0, vmax=10) #0.1)
        _=plt.colorbar(nrmse_imsh) if first_run else None   
        
        kk = mc.k_noised[:,:,epoch]
        Rex_n = mc.f_noised[:,:,epoch]/100 * kk * w1**2 / (w1**2 + kk*(kk+R2b) )    
        imsh_rex = axes[1, 3].imshow(Rex_n, vmin=0, vmax=1)                 
        _=plt.colorbar(imsh_rex) if first_run else None  
        
        txt1 = f"Epoch: {epoch}"
        self.txt1 = axes[0,2].text(
            -0.05, 0.98, txt1, 
            transform=axes[0,2].transAxes, bbox=dict(facecolor='gray', alpha=0.5),
            verticalalignment='top', horizontalalignment='left', fontsize=16, color='red',            
        )      
        txt2 = f"<NRMSE>={np.nanmean(mc.clean_fk_nrmse_perc[:,:,epoch]):.2f}%"        
        self.txt2 = axes[0,2].text(
            -0.05, 0.05, txt2,
            transform=axes[0,2].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=16, color='blue'
        )
        txt3 = f"<NRMSE>={np.nanmean(mc.noised_fk_nrmse_perc[:,:,epoch]):.2f}%"        
        self.txt3 = axes[1,2].text(
            -0.05, 0.05, txt3,
            transform=axes[1,2].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=16, color='blue'
        )
        
        return [f_imsh_clean, k_imsh_clean, nrmse_imsh_clean, f_imsh, k_imsh, nrmse_imsh, self.txt1, self.txt2, self.txt3, imsh_rex, imsh_rex_clean]  # fig, axes, 

    
def main():
    sample = True  # uncomment to use a single-subject full-brain dataset 
    xarr_name = './data/xarr_sample.nc' if sample else './data/xarr_vol8.nc'
    data_xa = xr.open_dataset(xarr_name) #'../nbmf/brain2test_mt_ds.nc')

    data_feed_mt = data.SlicesFeed.from_xarray(data_xa, mt_or_cest='mt')
    data_feed_amide = data.SlicesFeed.from_xarray(data_xa, mt_or_cest='amide')

    sfx = 'sample' if sample else 'brain'

    ckpts = os.path.abspath(f'./ckpts/{sfx}')  # edit for custom flows..

    pipelines.pipeline_config.mt_train_slw = 4               # quicker compile with 1-2..?
    pipelines.pipeline_config.mt_patience = 1000 # * 100             # ensure convergence? 
    pipelines.pipeline_config.cest_patience = 20            # ensure convergence?

    pipelines.pipeline_config.infer_config.predict_k_k = False  # (!!) Let's try the more straightforward prediction...
    
    pipelines.pipeline_config.train_config.tp_noise = True   #  SEEMS LIKE THERE'S NO IMPACT! not smooth either way

    pipelines.pipeline_config.train_config.std_up_fact = 5  
    pipelines.pipeline_config.train_config.tpnoise_augmentation_burn_in = 50
    pipelines.pipeline_config.mt_steps = 2000                # 2K good, 500 too low, 10K doesn't change much, just slower converge as LR is reduced
    pipelines.pipeline_config.train_config.use_shuffled_sampler = False
    
    # === MT ===
    
    mc_mt = Maps_Collector('c', data_feed_mt.shape, ds=data_feed_mt.ds, slw=data_feed_mt.slw, slice_in_slab_to_follow=1, slab2follow=0)
        
    predictor_mt, _ = pipelines.run_train(
        datafeed_mt=data_feed_mt, 
        datafeed_cest=None, 
        ckpt_folder=ckpts,     
        do_cest=False,
        mt_step_callback=mc_mt.collect
    ) 
    mt_tissue_params_est, nn_pred_signal_normed_np = infer.infer(
        data_feed_mt, nn_predictor=predictor_mt, 
        do_forward=False, pool2predict='c'
    )
    mca = McAnimator(mc_mt)
    mca.run(range(2000))  
    
    ani = animation.ArtistAnimation(fig=mca.fig, artists=mca.artists, interval=100)
    ani.save('mt_convergence_2x4.mp4', writer='ffmpeg', fps=20)

    # === Amide ===     
    
    data_feed_amide.fc_gt_T = mt_tissue_params_est['fc_T']
    data_feed_amide.kc_gt_T = mt_tissue_params_est['kc_T']
            
    data_feed_amide.ds = 2
    data_feed_amide.slw = 1
    data_feed_amide.random_ds = False  # !! properly goind over subsampling grids, now BEFORE advancing slices.

    mc_amide = Maps_Collector('b', data_feed_amide.shape) #  # ! need to have ds,slw good before getting here

    train.train_config.patience = 500
    train.train_config.tpnoise_augmentation_burn_in = 64
        
    model_state, loss_trend_amide, net_kwargs = \
        train.train(data_feed_amide, model_state=None, pool2predict='b',
                    mode='self_supervised', simulation_mode='expm_bmmat', logger=None,
                    steps=2000, lr=1e-3, step_callback=mc_amide.collect)
    
    mca = McAnimator(mc_amide)
    mca.run(range(100))

    ani = animation.ArtistAnimation(fig=mca.fig, artists=mca.artists, interval=200)
    ani.save('amide_convergence_2x4.mp4', writer='ffmpeg', fps=5)


if __name__ == "__main__":
    main()