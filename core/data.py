# -*- coding: utf-8 -*-
""" Load data from various sources and prepare it for training.

    See main SlicesFeed class for the main data store and feeding class.
""" 

import torch, numpy as np, pandas as pd
import jax, jax.numpy as jnp
import scipy.io as sio
from glob import glob
from natsort import natsorted
from pydicom import dcmread 
import os
import collections
import pyDOE
from dataclasses import dataclass


gamma = 42.58 * 2*np.pi * 1e6   # \bar{\gamma} = 42.58 MHz/T
@dataclass
class DataConfig:
    B0_base = 3.    # (Tesla)
    wc_ppm = 0.     # MT resonance (anywhere from -2.5 to 0 depending on source...)
    wb_ppm = 3.5    # Amide resonance
    # Amide relaxation: most sources say 40ms
    T2b_ms = 40
    # MT relaxation: 10us w Superlorentzian shape approximated by 40us Lorentzian (Zaiss2022, Perlman2022)
    T2c_ms = 0.04  
    T1a_ms = 1500  # ms, default T1 for the main pool (water)
    T2a_ms = 100   # ms, default T2 for the main pool (water)
        
# To be set externally! make sure notebook cells don't accidentally use the default config by reloading
data_config = None  # DataConfig()

SEQ_FILE_NAMES = {
        #'mt': './data/MT_52.txt',
        'mt': './data/MT_6-14ppm_52.txt',
        'larg': './data/larg_3ppm_51.txt',
        'amide': './data/amide_3.5ppm_51.txt',
        'noe': './data/noe_-3.5ppm_51.txt'
    }


def_ranges4LHS = collections.OrderedDict({\
    'T1a_ms': [800, 1800],  # vol7stats
    'T2a_ms': [15,  300],  # vol7stats
    'B0_shift_ppm_map': [-1.2, 1.2],  # vol7stats
    'B1_fix_factor_map': [0.7, 1.3],  # vol7stats
    'fb_gt_T': [0.001,  0.01],
    'kb_gt_T': [40,  400],
    'fc_gt_T': [0.05, 0.25], #[0.01,  0.3],
    'kc_gt_T': [10, 60]  # [1, 80]
})    


def_grids4cartesian = collections.OrderedDict({
    'T1a_ms': np.arange(700, 3000+100, 100),
    'T2a_ms': np.concatenate((np.arange(30, 150+10, 10), np.arange(200, 1000+100, 100))),
    'B0_shift_ppm_map': [0.],  # np.arange(-1, 1+.2, 0.2),  # vol7stats
    'B1_fix_factor_map': [1.],
    'fb_gt_T': [0.],
    'kb_gt_T': [0.],
    'fc_gt_T': np.arange(0.01, 0.3+.01, 0.01),
    'kc_gt_T': np.arange(4, 100+4, 4)
})               


def get_w1_wrf(B1, wrf_diff_Hz, B0_base=None, verbose=False):
    """ Turn B0,B1,ppm into angular frequencies (w1 - x axis, wrf - z axis), 
        that enter Bloch equations in the rotating frame (up to computing delta w_rf - w_abc)
    """
    B0_base = B0_base or data_config.B0_base
    w1 = B1 * gamma    
    w0_base = gamma * B0_base    
    wrf_rad = w0_base - 2*np.pi*wrf_diff_Hz  # PPM is "to left" by convention. Use minus for MT/NOE
    wrf_ppm = (wrf_rad-w0_base)/w0_base*1e6    
    if verbose:
        print(f'{wrf_ppm:.2f} ppm') 
    return w1, wrf_rad


def get_w_abc(B0_base=None, B0_shift_ppm=0, wb_ppm=None, wc_ppm=None):     
    B0_base = B0_base or data_config.B0_base
    wb_ppm = wb_ppm or data_config.wb_ppm
    wc_ppm = wc_ppm or data_config.wc_ppm
    
    B0_corrected = B0_base * (1 + 1e-6 * B0_shift_ppm)  # negligible effect..?
    wa = w0_cli = gamma * B0_corrected         
    wb = wa * (1 - wb_ppm*1e-6)  ## ppm is "to left" by convention
    wc = wa * (1 - wc_ppm*1e-6) 

    return wa, wb, wc


def read_sequence(seq_name, fname=None, drop_first=False, B0=None): 
    """ Cleaner read from clean files, supporting 3T/7T
    """
    B0 = B0 or data_config.B0_base
    fname = fname or SEQ_FILE_NAMES[seq_name]

    seq_df = pd.read_csv(fname, sep=' ')
    if drop_first:
        seq_df = seq_df[1:]
        seq_df.reset_index(inplace=True, drop=True) 
        try:
            seq_df.drop(['Unnamed: 0'], axis=1, inplace=True)
        except:
            pass
    try:
        seq_df['dwRF_Hz'] = seq_df['ppm'] * (gamma/(2*np.pi) * B0) / 1e6
    except KeyError:
        print("Warning: no 'ppm' column in the sequence file, dwRF_Hz will be used as is if exists")
        pass
    return seq_df


def get_lh_sample(var2range_od, shape):
    """Latin Hypercube Sampling for a given shape and variable ranges.
    """
    n_samples = np.product(shape)
    n_dim = len(var2range_od)
    lhd01 = pyDOE.lhs(n_dim, samples=n_samples)
    lhd_d = {}
    for ii, (pname, prange) in enumerate(var2range_od.items()):
        if len(prange) == 2:  # interpret as min-max
            lhd_d[pname] = lhd01[:, ii] * (prange[1] - prange[0]) + prange[0]
        else:                 # interpret as array of possible values
            lhd_d[pname] = prange[ np.int32(np.floor(lhd01[:, ii] * len(prange))) ]
        lhd_d[pname] = np.reshape(lhd_d[pname], shape)
    return lhd_d


def decode_data_entry(data_entry, detorch=True):
    ''' fixing the rogue 1st dimension added by torch.data.dataset
    '''
    roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict = data_entry        
    roi_mask_nans_T = roi_mask_nans_T[0].detach().numpy() if detorch else roi_mask_nans_T[0]
    measured_normed_T = measured_normed_T[0].detach().numpy() if detorch else measured_normed_T[0]
    for di in [w_dict, R_dict, gt_dict]:
        di.update({k: (v[0].detach().numpy()*roi_mask_nans_T if detorch else v[0]) for k, v in di.items()})
        
    return roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict

    
class Folder2Data:
    
    def __init__(self):
        pass    

    def load_erl(       
        self, subject_folder, output_folder='mrf_output', raw_folder='', drop_first=False,
        crop_bot=45, crop_top=95
        ): 
        """
        Load the results of the matlab pipeline
        """
        self.subject_folder = subject_folder        
        self.T1ms = sio.loadmat(subject_folder+f'/{raw_folder}/T1map.mat')['T1map'][:, crop_bot: crop_top, :]        
        self.T2ms = sio.loadmat(subject_folder+f'/{raw_folder}/T2map.mat')['T2map'][:, crop_bot: crop_top, :] * 1000
        self.dB0ppm = sio.loadmat(subject_folder+f'/{raw_folder}/dB0map.mat')['dB0map'][:, crop_bot: crop_top, :]
        self.B1map = sio.loadmat(subject_folder+f'/{raw_folder}/B1map.mat')['B1map'][:, crop_bot: crop_top, :]
        self.MT_data_full = sio.loadmat(subject_folder+f'/{raw_folder}/MT.mat')['mt_mat'].transpose([3,0,1,2])        
        self.MT_data = self.MT_data_full[1:, :, crop_bot: crop_top, :] if drop_first \
                else self.MT_data_full[:, :, crop_bot: crop_top, :]                       
        self.AMIDE_data_full = sio.loadmat(subject_folder+f'/{raw_folder}/AMIDE.mat')['am_mat'].transpose([3,0,1,2])
        self.AMIDE_data = self.AMIDE_data_full[1:, :, crop_bot: crop_top, :] if drop_first \
                else self.AMIDE_data_full[:, :, crop_bot: crop_top, :]       
        try:
            self.mt_mrf_output = sio.loadmat(subject_folder+f'/{output_folder}/MT_maps.mat')
            self.amide_mrf_output = sio.loadmat(subject_folder+f'/{output_folder}/AMIDE_maps.mat')
            self.fss_gt = self.mt_mrf_output['M0ss'][:, crop_bot: crop_top, :] # /100            
            self.kss_gt = self.mt_mrf_output['Kssw'][:, crop_bot: crop_top, :]         
            self.f_amide_gt = self.amide_mrf_output['M0s'][:, crop_bot: crop_top, :] # /100
            self.k_amide_gt = self.amide_mrf_output['Ksw'][:, crop_bot: crop_top, :]         
        except:            
            print("No reference extracted fss/kssw, amide fs/ksw maps")
            self.fss_gt = self.kss_gt = self.f_amide_gt = self.k_amide_gt = np.zeros(self.T1ms.shape)            
        try:
            brainmask = sio.loadmat(subject_folder+f'/{raw_folder}/BRAINMASK.mat')
            self.white_and_gray = brainmask['c2_nii'][:, :, :] + brainmask['c1_nii'][:, :, :]     
            self.white_and_gray = self.white_and_gray[:, crop_bot: crop_top, :]
            
            self.WM_GM_CSF = brainmask['c2_nii'][:, :, :] + brainmask['c1_nii'][:, :, :] + brainmask['c3_nii'][:, :, :]        
            self.WM_GM_CSF = self.WM_GM_CSF[:, crop_bot: crop_top, :]

            self.gray = brainmask['c1_nii'][:,crop_bot:crop_top,:]
            self.white = brainmask['c2_nii'][:,crop_bot:crop_top,:]
            self.gray[self.gray<0.21] = np.nan
            self.white[self.white<0.2] = np.nan

        except:
            print("No masks " )
        self.create_mask()
        print(self)
        return self
    
    def __str__(self):
        res = ""
        for x,y in self.__dict__.items():
            try:
                res += f"{x}: {y.shape}\n"
            except:
                pass
        return res
    
    def create_mask(self, use_fss=True):
        if hasattr(self, 'WM_GM_CSF'):
            print('Best scenario: setting mask acccording to available gray+white+CSF segmentation..')            
            self.roi_mask_zeros = np.float32(self.WM_GM_CSF > 0.8)  
        elif use_fss: 
            print("setting mask using the Fss oracle - convenient way to remove CSF")
            self.roi_mask_zeros = self.fss_gt > 0.01
            self.roi_mask_zeros = np.float32(np.logical_and(self.roi_mask_zeros, ~np.isnan(self.B1map)))
        else:
            print("w.o. masks or ss-oracle, use T1, T2 brackets to get WM+GM")
            self.roi_mask_zeros = np.logical_and(self.T1ms<40000, self.T2ms<4000) # disable..
            self.roi_mask_zeros = np.logical_and(self.roi_mask_zeros, self.T1ms>1)
            self.roi_mask_zeros = np.logical_and(self.roi_mask_zeros, self.T2ms>1)
            self.roi_mask_zeros = np.float32(self.roi_mask_zeros)

        self.roi_mask_nans = np.copy(self.roi_mask_zeros)
        self.roi_mask_nans[self.roi_mask_nans==0] = np.nan   
        

def dicoms2data(data_path, folder, xcrop=[0, -1], ycrop=[0, -1], Nmin=0, Nmax=-1, drop_first=True):
    dicoms_paths = natsorted(glob(os.path.join(data_path, folder, '*.dcm')))
    dicoms = [dcmread(dcm) for dcm in dicoms_paths]

    def reshape_image(test):
        num_x, num_y = 10, 10
        dim_x, dim_y = test.shape[0]//num_x, test.shape[1]//num_y
        reshaped_images = []
        # Split the large image into smaller images
        for i in range(num_x):
            for j in range(num_y):
                small_image = test[i*dim_x:(i+1)*dim_x, j*dim_y:(j+1)*dim_y]
                reshaped_images.append(small_image)

        reshaped_images = np.array(reshaped_images)
        return reshaped_images

    reshaped_images = np.array(reshape_image(dicoms[1].pixel_array))
    
    print('overall shape: ', np.array([reshape_image(x.pixel_array) for x in dicoms])[1:,...].shape)

    ## !! ===== dicom to data  ====== !! 
    data = [reshape_image(x.pixel_array)[xcrop[0]: xcrop[1], Nmin: Nmax, ycrop[0]: ycrop[1]] for x in dicoms]  #newdata
    if drop_first:
        return np.array(data)[1:,...]
    else:
        return np.array(data)
    
    
class SlicesFeed(torch.utils.data.Dataset):

    norm_type = 'l2'  # alternatively: "first", 'l2'
    # ! the above is specifically referred to as class attribute so should be set as SlicesFeed.norm_type
    # all the below are used as object attributes - consider moving to init. 
    add_noise_to_signal = 0
    ds = 1             # downsampling ratio
    slw = 10           # slab width            
    downsample_or_slab = True  # otherwise, will cut slabs instead of downsampling. 
    random_ds = True           # otherwise, will scan slab/downsample offsets sequentially. 
    xo_state = 0
    yo_state = 0
    zo_state = 0
    
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def from_args(
            cls, shape, T1a_ms=None, T2a_ms=None, seq_name=None, seq_df=None,
            kb_gt_T=1e-6, fb_gt_T=1e-6, kc_gt_T=1e-6, fc_gt_T=1e-6,  
            b_ppm=None, T2b_ms=None, c_ppm=None, T2c_ms=None, same_T1=True, 
            B0=None, B0_shift_ppm_map=0, B1_fix_factor_map=1,
            roi_mask_nans=1, signal=np.nan, drop_first=False
        ):
        """
            All tissue-parameter arguments can be maps shaped as <shape> arg, or scalars.
            
            The "ground truth" parameters are only used in pure forward-simulation flows (no fitting/training), e.g., synthetic data generation
        """
        b_ppm = b_ppm or data_config.wb_ppm
        c_ppm = c_ppm or data_config.wc_ppm
        
        T2b_ms = T2b_ms if type(T2b_ms)!=type(None) else data_config.T2b_ms
        T2c_ms = T2c_ms if type(T2c_ms)!=type(None) else data_config.T2c_ms
        
        T1a_ms = T1a_ms if type(T1a_ms)!=type(None) else data_config.T1a_ms
        T2a_ms = T2a_ms if type(T2a_ms)!=type(None) else data_config.T2a_ms
        
        data4nbmf = cls()        
        data4nbmf.shape = shape
        data4nbmf.slw = min(data4nbmf.slw, data4nbmf.shape[1])    
        if type(seq_df) == type(None):
            seq_df = read_sequence(seq_name, drop_first=drop_first)
            data4nbmf.seq_name = seq_name
        else:
            data4nbmf.seq_name = 'custom'
        data4nbmf.seq_df = seq_df
        data4nbmf.seq_len = seq_len = len(data4nbmf.seq_df['B1_uT'])  # signal.shape[0]        
        
        data4nbmf.kb_gt_T = kb_gt_T * np.ones(shape) 
        data4nbmf.fb_gt_T = fb_gt_T * np.ones(shape)
        data4nbmf.kc_gt_T = kc_gt_T * np.ones(shape) 
        data4nbmf.fc_gt_T = fc_gt_T * np.ones(shape)            
        data4nbmf.B0_shift_ppm_map = B0_shift_ppm_map * np.ones(shape)
        data4nbmf.B1_fix_factor_map = B1_fix_factor_map * np.ones(shape)        
        
        data4nbmf.R1a_V = 1 / (1e-3 * T1a_ms + 1e-6) * np.ones(shape)        
        if same_T1:
            # Same spin-lattice relaxation for all pools
            data4nbmf.R1c_V = data4nbmf.R1b_V = data4nbmf.R1a_V 
        else:
            assert 0, "Not implemented"	
        data4nbmf.R2a_V = 1 / (1e-3 * T2a_ms + 1e-6) * np.ones(shape)
        data4nbmf.R2b_V = 1 / (1e-3 * T2b_ms + 1e-6) * np.ones(shape)
        data4nbmf.R2c_V = 1 / (1e-3 * T2c_ms + 1e-6) * np.ones(shape)

        w1_seq, wrf_seq = np.zeros(seq_len), np.zeros(seq_len)         
        for ii in range(seq_len):
            w1_seq[ii], wrf_seq[ii] = get_w1_wrf(\
                data4nbmf.seq_df['B1_uT'][ii] * 1e-6, 
                data4nbmf.seq_df['dwRF_Hz'][ii], B0_base=B0
                ) #, verbose=True)

        w_a, w_b, w_c = get_w_abc(B0_base=B0, B0_shift_ppm=B0_shift_ppm_map*np.ones(shape), wb_ppm=b_ppm, wc_ppm=c_ppm)       
        data4nbmf.wa_T = w_a[None, ...]
        data4nbmf.wb_T = w_b[None, ...]
        data4nbmf.wc_T = w_c[None, ...]

        data4nbmf.w1_T = w1_seq[:, None, None, None]
        data4nbmf.w1_T = data4nbmf.w1_T * B1_fix_factor_map * np.ones(shape)
        data4nbmf.w1_a0_mean = np.nanmean(data4nbmf.w1_T[0])  # ..to recover b1 map later..
        data4nbmf.wrf_T = wrf_seq[:, None, None, None] 
        data4nbmf.roi_mask_nans = roi_mask_nans * np.ones(shape)

        # !! EXP: denoise M0 image to try to keep those 3dB
        denoise_M0_image = False
        if denoise_M0_image:
            from scipy.ndimage import median_filter
            signal[0] = median_filter(signal[0], size=3)
        
        data4nbmf.measured_normed_T = \
            data4nbmf.normalize(signal * np.ones((seq_len, *shape)))

        return data4nbmf
    
    def __add__(self, other):
        def concat_axial(a, b):
            return np.concatenate((a, b), axis=1)
        assert self.shape[0] == other.shape[0] and self.shape[2] == other.shape[2]
        assert self.seq_name == other.seq_name and self.seq_df.equals(other.seq_df)
        if isinstance(other, SlicesFeed):
            return SlicesFeed.from_args(
                shape=(self.shape[0], self.shape[1]+other.shape[1], self.shape[2]),
                T1a_ms=concat_axial(self.R1a_V, other.R1a_V)**-1 * 1e3,
                T2a_ms=concat_axial(self.R2a_V, other.R2a_V)**-1 * 1e3,
                seq_name=self.seq_name,
                seq_df=self.seq_df,
                kb_gt_T=concat_axial(self.kb_gt_T, other.kb_gt_T),
                fb_gt_T=concat_axial(self.fb_gt_T, other.fb_gt_T),
                kc_gt_T=concat_axial(self.kc_gt_T, other.kc_gt_T),
                fc_gt_T=concat_axial(self.fc_gt_T, other.fc_gt_T),
                B0_shift_ppm_map=concat_axial(self.B0_shift_ppm_map, other.B0_shift_ppm_map),
                B1_fix_factor_map=concat_axial(self.B1_fix_factor_map, other.B1_fix_factor_map),
                roi_mask_nans=concat_axial(self.roi_mask_nans, other.roi_mask_nans),
                signal=np.concatenate((self.measured_normed_T, other.measured_normed_T), axis=2)
            )
        return NotImplemented
    
    @classmethod
    def make_lh_sample(
        cls, 
        seq_name,
        tissue_parameter_ranges_od=None,
        shape=[100,25,100],        
        **kwargs
        ):           
        
        if type(tissue_parameter_ranges_od) == type(None): 
            tissue_parameter_ranges_od = def_ranges4LHS           
        tp_d = get_lh_sample(tissue_parameter_ranges_od, shape)
        return cls.from_args(seq_name=seq_name, shape=shape, **tp_d, **kwargs)
    
    @classmethod
    def make_cartesian_sample(
        cls, 
        seq_name,
        parameter_values_od=None,
        shape=None, # [100,25,100],    
        slices=[0,500], # max cutoff
        **kwargs
        ):                           
        if type(parameter_values_od) == type(None): 
            shape = [99, 40, 100]
            parameter_values_od = def_grids4cartesian
        grids = np.meshgrid(*parameter_values_od.values(), indexing='ij')
        
        tp_d = {name: grid.ravel().reshape([shape[0], -1, shape[2]])[:, slices[0]:slices[1],:]    # [:, :shape[1],:] 
                for name, grid in zip(parameter_values_od.keys(), grids)
                }
        shard_shape = list(tp_d.values())[0].shape
        return cls.from_args(seq_name=seq_name, shape=shard_shape, **tp_d, **kwargs)        
        
    @classmethod
    def from_nv_dict(cls, nvdict_fname='mtd.npz', shape=(88, 50, 90)):
        """Loading dictionary created using Vladimirov2024's code
        """
        nvdict = np.load('mtd.npz', fix_imports=True, allow_pickle=True)['arr_0'].item()
        bsf_lhsynth_mt = cls.from_args(
            seq_name='mt',
            shape=shape, 
            T1a_ms=nvdict['t1w'].reshape(shape)*1000,
            T2a_ms=nvdict['t2w'].reshape(shape)*1000,
            fb_gt_T=0, kb_gt_T=0,
            fc_gt_T=nvdict['fs_0'].reshape(shape),
            kc_gt_T=nvdict['ksw_0'].reshape(shape),
            signal=nvdict['sig'].T.reshape(31,*shape),
        )
        return bsf_lhsynth_mt
    
    @classmethod
    def from_xarray(cls, brxarray, seq_name=None, seq_df=None, drop_first=None, mt_key='MT_data', cest_key='AMIDE_data', **kwargs):
        """ Load data from a xarray dataset aggregating registered maps
        """              
        signal = brxarray[mt_key].values if seq_name == 'mt' else brxarray[cest_key].values
        if drop_first:
            signal = signal[1:]
        return cls.from_args(
            seq_name=seq_name,
            shape=brxarray['T1ms'].values.shape,
            seq_df=seq_df,            
            signal=signal,
            roi_mask_nans=brxarray['roi_mask_nans'].values,
            T1a_ms=brxarray['T1ms'].values,
            T2a_ms=brxarray['T2ms'].values,
            B0_shift_ppm_map=brxarray['B0_shift_ppm_map'].values,
            B1_fix_factor_map=brxarray['B1_fix_factor_map'].values,
            drop_first=drop_first,
            **kwargs
        )
        
    @classmethod
    def from_cestmrf_folder(cls, subject_folder, mt_or_amide, crop_bot=45, **kwargs):
        """ Load data from a our matlab pipeline-processed folder
        """
        f2d = Folder2Data().load_erl(subject_folder, crop_bot=crop_bot)
        if mt_or_amide == 'mt':
            signal = f2d.MT_data            
        elif mt_or_amide == 'amide':
            signal = f2d.AMIDE_data            
        return cls.from_args(
            seq_name=mt_or_amide, 
            shape=f2d.T1ms.shape,
            signal=signal, 
            roi_mask_nans=f2d.roi_mask_nans,
            T1a_ms=f2d.T1ms, T2a_ms=f2d.T2ms,
            B0_shift_ppm_map=f2d.dB0ppm, B1_fix_factor_map=f2d.B1map, 
            **kwargs
        )
    
    @classmethod
    def normalize(cls, signal, norm_type=None):        
        norm_type = norm_type or cls.norm_type
        # Suppress the "all-NaN slice" warning - 
        #  we're ok with reducing all-NaN slices to NaN
        with np.errstate(invalid='ignore'):
            if norm_type == 'none':
                signal_pixelwise_norm = 1
            elif norm_type == 'l2': 
                signal_pixelwise_norm = np.linalg.norm(signal, axis=0) 
            elif norm_type == 'first':
                signal_pixelwise_norm = signal[0]
            elif norm_type == 'max':
                signal_pixelwise_norm = np.nanmax(signal, axis=0)        
            else:
                assert 0
            signal_normed = signal / (signal_pixelwise_norm + 1e-6)
            return signal_normed
    
    @classmethod
    def normalize_jax(cls, signal, norm_type=None):
        norm_type = norm_type or cls.norm_type
        if norm_type == 'none':
            signal_pixelwise_norm = 1
        elif norm_type == 'l2': 
            signal_pixelwise_norm = jnp.linalg.norm(signal, axis=0) 
        elif norm_type == 'first':
            signal_pixelwise_norm = signal[0]
        elif norm_type == 'max':
            signal_pixelwise_norm = jnp.nanmax(signal, axis=0)        
        else:
            assert 0
        signal_normed = signal / (signal_pixelwise_norm + 1e-6)
        return signal_normed
    
    def __len__(self):
        return self.R1a_V.shape[1]  # slices (no account for slw, ds here..)
    
    def __getitem__(self, idx):
        """ get a SLAB of <self.slw> slices starting at slice #idx, 
            downsampled (by sampling or cutout) by stride <self.ds> in each of X, Y
        """
        z0 = idx
        slw = self.slw
        ds = self.ds        
        if self.random_ds:  
            xo, yo, zo = np.random.choice(ds, 3)          
        else:
            # !! NOTE igoring input idx and of course any shuffle.             
            xo, yo, z0 = self.xo_state, self.yo_state, self.zo_state
            # rotate for next time                        
            self.xo_state += 1            
            if self.xo_state == ds:
                self.xo_state = 0                
                self.yo_state = (self.yo_state + 1)
                if self.yo_state == ds:
                    self.yo_state = 0
                    self.zo_state += slw  # (typically 1)
                    if self.zo_state >= self.shape[1]:
                        self.zo_state = 0
        
        if self.downsample_or_slab:  # downsample
            xe, ye = self.shape[0], self.shape[2]
        else:                        # cutout                            
            x_slabsize = (self.shape[0] // self.ds) 
            y_slabsize = (self.shape[2] // self.ds) 
            xo, xe = xo*x_slabsize, (xo+1)*x_slabsize
            yo, ye = yo*y_slabsize, (yo+1)*y_slabsize
            ds = 1 

        R_dict = {
            'R1a_T': self.R1a_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0,
            'R2a_T': self.R2a_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0,
            'R2b_T': self.R2b_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0,
            'R2c_T': self.R2c_V[xo:xe:ds, z0:z0+slw:, yo:ye:ds] + 0.0
            }
        w_dict = {
            'wa_T': self.wa_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'wb_T': self.wb_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'wc_T': self.wc_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'w1_T': self.w1_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds]
            }
        measured_normed_T = self.measured_normed_T[:, xo:xe:ds, z0:z0+slw:, yo:ye:ds]
                
        if self.add_noise_to_signal != 0:                        
            measured_normed_T = measured_normed_T * \
                (1 + self.add_noise_to_signal * np.random.randn(*measured_normed_T.shape))
            measured_normed_T = self.normalize(measured_normed_T)
            
        roi_mask_nans_T = self.roi_mask_nans[xo:xe:ds, z0:z0+slw:ds, yo:ye:ds]
        
        gt_dict = {
            'kb_gt_T': self.kb_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'fb_gt_T': self.fb_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'kc_gt_T': self.kc_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds],
            'fc_gt_T': self.fc_gt_T[xo:xe:ds, z0:z0+slw:, yo:ye:ds]
            }        
        return (roi_mask_nans_T, measured_normed_T, w_dict, R_dict, gt_dict, (xo,yo,z0))

    @property
    def batch_shape(self):
        shape = np.array(self.shape)
        shape[1] = self.slw
        return shape//self.ds
    
    