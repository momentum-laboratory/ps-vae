import os, numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.compat import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
FIGS_DIR = REPO_ROOT / "figs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import config
config.Config.configure_AMIDE_clinical_whole_brain()
import analyze_uncertainty as au
    
from notebooks import nutils


def meshgrid_xy_pairs(shape):
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    xy_pairs = np.stack([x, y], axis=-1).reshape(-1, 2)
    return xy_pairs.tolist()
mt_sim_mode = 'isar2_c'

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
            try:
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
                raise e
            except Exception as e:
                print(f"Error processing voxel ({_x}, {sli}, {_y}): {e}")
                continue
    except KeyboardInterrupt as e:
        print("Interrupted by user..")
    print("Processed {} voxels.".format(len(good_xy_points)))
        
    return allpdfs, good_xy_points, min_nrms_list, min_nrms_95cr_th_list, CR_areas

for testvolname in ['vol8', 'vol9', 'vol10','vol7']:
    vol_data_feed_mt, points_grey, points_white = nutils.load_vol_detail(testvolname, seq_name='mt')
    allpdfs, good_xy_points, min_nrms_list, min_nrms_95cr_th_list, CR_areas = get_all_pdfs(vol_data_feed_mt, sli=15, seq_name='mt', mt_sim_mode=mt_sim_mode)
    np.savez_compressed(
        f'{FIGS_DIR}/humans/vol_pdfs/{testvolname}.npz', 
        allpdfs=allpdfs, good_xy_points=good_xy_points, 
        min_nrms_list=min_nrms_list, min_nrms_95cr_th_list=min_nrms_95cr_th_list, CR_areas=CR_areas
    )