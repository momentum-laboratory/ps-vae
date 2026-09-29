from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def get_by_name(patient='erlangen'):
    return {'erlangen': load_erlangen,
            'ichilov_alz': load_ichilov_alz
        }[patient]


def load_erlangen():
    mt_signal = loadmat('/data/patients_put_aside/2021_patient_erlangen/raw/dataToMatchMT_3T_3D.mat')['dataToMatch']
    amide_signal = loadmat('/data/patients_put_aside/2021_patient_erlangen/raw/dataToMatchAmide_3T_3D.mat')['dataToMatch']
    mt_signal = mt_signal[:, :, 65:115, :]
    amide_signal = amide_signal[:, :, 65:115, :]

    aux_data = loadmat('/data/patients_put_aside/2021_patient_erlangen/raw/Ground_Truth_T1_T2_B0_B1.mat')
    T1ms = aux_data['GT_T1'][:, 65:115, :]
    T2ms = aux_data['GT_T2'][:, 65:115, :]
    B0ppm = aux_data['GT_B0_ppm'][:, 65:115, :]
    B1fac = aux_data['B1map'][:, 65:115, :]
    # data comes with ZEROS outside of the brain, so we can use that to create a mask for the region of interest (ROI)
    roi_mask_nans = np.float32(T1ms != 0)
    roi_mask_nans[roi_mask_nans==0] = np.nan

    return mt_signal, amide_signal, T1ms, T2ms, B0ppm, B1fac, roi_mask_nans


def load_ichilov_alz(subjdir='/home/alexf/alz/a/organized_eval_for_alex_AL3'):     
    mt_signal = loadmat(f"{subjdir}/MT.mat")['mt_mat'].transpose(3, 1, 2, 0)[:, ::-1, :, :]
    amide_signal = loadmat(f"{subjdir}/AMIDE.mat")['am_mat'].transpose(3, 1, 2, 0)[:, ::-1, :, :]
    T1ms = loadmat(f"{subjdir}/T1map.mat")['T1map'].transpose(1, 2, 0)[::-1, :, :]
    T2ms = loadmat(f"{subjdir}/T2map.mat")['T2map'].transpose(1, 2, 0)[::-1, :, :] * 1000
    B1fac = loadmat(f"{subjdir}/B1map.mat")['B1map'].transpose(1, 2, 0)[::-1, :, :]
    B0ppm = loadmat(f"{subjdir}/dB0map.mat")['dB0map'].transpose(1, 2, 0)[::-1, :, :]
    # data comes with NaNs outside of the brain, so we can use that to create a mask for the region of interest (ROI)
    roi_mask_nans = T1ms.copy()
    roi_mask_nans[~np.isnan(roi_mask_nans)] = 1

    return mt_signal, amide_signal, T1ms, T2ms, B0ppm, B1fac, roi_mask_nans


def viz(mt_signal, amide_signal, T1ms, T2ms, B0ppm, B1fac, roi_mask_nans, sli=27):
    plt.figure(figsize=(20, 2))
    plt.subplot(161); 
    plt.imshow(mt_signal[0, :, sli, :], cmap='bone'); plt.colorbar()
    plt.subplot(162); 
    plt.imshow(amide_signal[0, :, sli, :], cmap='bone'); plt.colorbar()
    plt.subplot(163); 
    plt.imshow(T1ms[:, sli, :], cmap='jet', vmax=2500); plt.colorbar(); plt.title('T1map')     
    plt.subplot(164); 
    plt.imshow(T2ms[:, sli, :], cmap='jet', vmax=200); plt.colorbar() ; plt.title('T2map')
    plt.subplot(165); 
    plt.imshow(B0ppm[:, sli, :], vmin=-0.5, vmax=0.5,cmap='bwr'); plt.colorbar(); plt.title('B0map')
    plt.subplot(166); 
    plt.imshow(B1fac[:, sli, :], vmin=0.5, vmax=1.5, cmap='bwr'); plt.colorbar(); plt.title('B1map')
                                            
