#!/usr/bin/env python
"""Reproduce Fig. 3 (in-vivo mouse MT + amide uncertainty maps and two-voxel posteriors).

For each mouse shown in the figure this trains the two-stage PS-VAE (MT, then amide) on
the mouse's own scan, infers the tissue parameters with their NN-predicted posterior, and
saves the estimates. The figure is then assembled from those estimates; the reference
(grid-likelihood) posteriors for the two sample voxels per mouse are computed on the fly.

This is the minimal path through notebooks/demo_mouse.ipynb (cells up to the
tissue-params .npz dump) and the "FINAL FIGURE (Fig. 3)" cell of notebooks/summary.ipynb.

Usage (from anywhere):
    python scripts/reproduce_fig3.py                 # train + infer + plot
    python scripts/reproduce_fig3.py --skip-train    # re-plot from saved estimates
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)  # data/, sequence files etc. are resolved relative to the repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import config, data, infer, pipelines
import analyze_uncertainty as au

# The three mice shown in Fig. 3, top to bottom
FIG3_MICE = [
    'ped_Control_2R_2025-01-09',
    'ped_Control_1L1R_2025-01-09',
    'ozohar_two_ears_20240404_091158',
]


def configure():
    """Fresh preclinical configuration (resets all module configs, as in a new kernel)."""
    config.Config.configure_mt_amide_preclinical()
    data.SlicesFeed.norm_type = 'l2'


def get_lims(data_xa):
    ''' Trim non-brain areas up to single line margin in each direction
    '''
    xmin = list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=1) > 0).index(True) - 1
    xmax = 1 - list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=1) > 0)[::-1].index(True)
    ymin = list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=0) > 0).index(True) - 1
    ymax = 1 - list(np.nansum(data_xa['roi_mask_nans'][:,0,:].to_numpy(), axis=0) > 0)[::-1].index(True)
    return xmin, xmax, ymin, ymax


def get_mask_centroid(mask):
    ''' Calculate centroid of a binary mask
    '''
    coords = np.argwhere(mask == 1)
    return np.int32(coords.mean(axis=0))


def load_mouse(mouse_name):
    ''' Load a mouse scan cropped to the brain, as data feeds for both protocols, plus the
        tumor/contralateral masks and a sample point (centroid) in each.
    '''
    data_xa = xr.open_dataset(REPO_ROOT / 'data' / f'{mouse_name}.nc')
    xmin, xmax, ymin, ymax = get_lims(data_xa)
    data_xa_cutout = data_xa.isel(height=slice(xmin, xmax), width=slice(ymin, ymax))

    tumor_mask = data_xa_cutout['tumor_mask'].to_numpy().copy().squeeze()
    contralateral_mask = data_xa_cutout['contralateral_mask'].to_numpy().copy().squeeze()

    # no B0/B1 maps for the mice - assume nominal fields
    data_xa_cutout['B0_shift_ppm_map'] = (('height','slice','width'), np.zeros_like(data_xa_cutout['T1ms']))
    data_xa_cutout['B1_fix_factor_map'] = (('height','slice','width'), np.ones_like(data_xa_cutout['T1ms']))

    data_feed_mt = data.SlicesFeed.from_xarray(data_xa_cutout, seq_name='mt')
    data_feed_amide = data.SlicesFeed.from_xarray(data_xa_cutout, seq_name='amide')

    return dict(
        data_feed_mt=data_feed_mt, data_feed_amide=data_feed_amide,
        tumor_mask=tumor_mask, contralateral_mask=contralateral_mask,
        points_tumor=[get_mask_centroid(tumor_mask)],
        points_contralateral=[get_mask_centroid(contralateral_mask)],
    )


def train_and_infer(mouse_name, out_dir: Path, ckpt_dir: Path):
    ''' Self-supervised training on the mouse's own scan (MT stage, then amide stage),
        followed by inference; saves the tissue parameter estimates + NN posterior factors.
    '''
    configure()
    mouse = load_mouse(mouse_name)
    data_feed_mt, data_feed_amide = mouse['data_feed_mt'], mouse['data_feed_amide']

    # orbax won't overwrite an existing checkpoint step, so start clean on re-runs
    shutil.rmtree(ckpt_dir / mouse_name, ignore_errors=True)
    t0 = time.time()
    predictor_mt, predictor_amide = pipelines.run_train(
        datafeed_mt=data_feed_mt,
        ckpt_folder=str(ckpt_dir / mouse_name),
        do_cest=True, datafeed_cest=data_feed_amide
    )
    train_time = time.time() - t0
    print(f"[{mouse_name}] total training time: {train_time/60:.1f} min")

    data_feed_mt.ds = 1
    mt_tissue_params_est, MT_nn_pred_signal_normed_np = infer.infer(
        data_feed_mt, nn_predictor=predictor_mt,
        do_forward=True, pool2predict='c'
    )
    data_feed_amide.ds = 1
    amide_tissue_params_est, APT_nn_pred_signal_normed_np = infer.infer(
        data_feed_amide, nn_predictor=predictor_amide,
        do_forward=True, pool2predict='b'
    )

    intermediates = out_dir / mouse_name / 'intermediates'
    intermediates.mkdir(parents=True, exist_ok=True)
    np.savez(
        intermediates / 'mt_tissue_params_est.npz',
        MT_nn_pred_signal_normed_np=MT_nn_pred_signal_normed_np,
        tumor_mask=mouse['tumor_mask'], contralateral_mask=mouse['contralateral_mask'],
        train_time=train_time,
        **mt_tissue_params_est
    )
    np.savez(
        intermediates / 'amide_tissue_params_est.npz',
        APT_nn_pred_signal_normed_np=APT_nn_pred_signal_normed_np,
        tumor_mask=mouse['tumor_mask'], contralateral_mask=mouse['contralateral_mask'],
        **amide_tissue_params_est
    )


# ======== figure ========

def slap_label_on_container(container, label_text, loc=(-0.01, 1.05), color='black', fontsize=18):
    ghost_ax = container.add_axes([0, 0, 1, 1], frameon=False)
    ghost_ax.set_xticks([]); ghost_ax.set_yticks([])
    ghost_ax.text(
        loc[0], loc[1], label_text, transform=ghost_ax.transAxes,
        fontsize=fontsize, fontweight='bold', va='top', ha='left',
        color=color
    )


def mark_points(ax, points, labels, colors):
    for jj, (label, point) in enumerate(zip(labels, points)):
        square = patches.Rectangle((point[1]-.6, point[0]-.6), 1.5, 1.5, facecolor='none', edgecolor=colors[label], linewidth=2.5)
        ax.add_patch(square)
        ax.text(point[1]-(3 if label else -3), point[0]+1, ['A', 'B'][jj], color=colors[label], fontsize=15, ha='center')


def plot_mouse_demo(mouse_name, out_dir: Path, fig, subfigs, letters=('a','b','c','d')):
    ''' Two rows per mouse: MT (top) and amide (bottom); each row has the parameter/uncertainty
        maps on the left and the NN vs. reference posteriors of a tumor and a contralateral voxel on the right.
    '''
    mouse = load_mouse(mouse_name)
    data_feed_mt, data_feed_amide = mouse['data_feed_mt'], mouse['data_feed_amide']
    points_tumor, points_contralateral = mouse['points_tumor'], mouse['points_contralateral']
    intermediates = out_dir / mouse_name / 'intermediates'

    mt_tissue_params_est = dict(np.load(intermediates / 'mt_tissue_params_est.npz'))
    amide_tissue_params_est = dict(np.load(intermediates / 'amide_tissue_params_est.npz'))
    shape = mt_tissue_params_est['fc_T'].shape

    colors = ['white', 'gold']
    points = points_tumor + points_contralateral
    labels = [1]*len(points_tumor) + [0]*len(points_contralateral)

    f_est, k_est, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
        au.extract_nn_estimate_of_posterior(mt_tissue_params_est, shape, is_cest=False)
    axes = au.plot_CI_maps(
        f_est[:,0,:], f_sigma[:,0,:],
        k_est[:,0,:], k_sigma[:,0,:],
        is_cest=False, vmax_k=70, vmax_f=30,
        fig=subfigs[0][0], ktop=True, ticks_for_all_cbars=True, kwargs_plot_map={'pad': 0.25}
    )
    mark_points(axes[1, 1], points, labels, colors)
    slap_label_on_container(subfigs[0][0], f'({letters[0]})', (0.05, 1.05))

    au.juxtapose_two_voxels_posteriors(
        data_feed_mt, mt_tissue_params_est, shape,
        points_tumor[0], points_contralateral[0], fig=subfigs[0][1]
    )
    slap_label_on_container(subfigs[0][1], f'({letters[1]})', (0.03, 1.05))

    f_est, k_est, cov_nnpred_scaled, f_sigma, k_sigma, height, width, angle = \
        au.extract_nn_estimate_of_posterior(amide_tissue_params_est, shape, is_cest=True)
    axes = au.plot_CI_maps(
        f_est[:,0,:], f_sigma[:,0,:],
        k_est[:,0,:], k_sigma[:,0,:],
        is_cest=True, vmax_k=700, vmax_f=0.6,
        fig=subfigs[1][0], ktop=True, ticks_for_all_cbars=True, kwargs_plot_map={'pad': 0.3}
    )
    mark_points(axes[1, 1], points, labels, colors)
    slap_label_on_container(subfigs[1][0], f'({letters[2]})', (0.05, 1.05))

    au.juxtapose_two_voxels_posteriors(
        data_feed_mt, amide_tissue_params_est, shape,
        points_tumor[0], points_contralateral[0], fig=subfigs[1][1],
        is_cest=True, data_feed_cest=data_feed_amide, max_k_cest=700,
    )
    slap_label_on_container(subfigs[1][1], f'({letters[3]})', (0.03, 1.05))


def make_figure(mice, out_dir: Path):
    configure()
    fig = plt.figure(figsize=(14, 6*len(mice)), constrained_layout=True)
    subfigs = fig.subfigures(2*len(mice), 2, width_ratios=[3, 1.2], wspace=0.05, hspace=0.1)
    for pi, mouse_name in enumerate(mice):
        letters = [chr(ord(let) + 4*pi) for let in ('a', 'b', 'c', 'd')]
        plot_mouse_demo(mouse_name, out_dir, fig=fig, subfigs=subfigs[pi*2:(pi*2+2), :], letters=letters)

    for fmt in ['png', 'svg']:
        plt.savefig(out_dir / f'fig3.{fmt}', dpi=300, bbox_inches='tight', format=fmt)
    plt.savefig(out_dir / 'fig3_LOWRES.png', dpi=100, bbox_inches='tight')
    print(f"Saved {out_dir / 'fig3.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mice', nargs='+', default=FIG3_MICE, help='mouse scan names (data/<name>.nc)')
    parser.add_argument('--out', default='figs/fig3', help='output folder for estimates and the figure')
    parser.add_argument('--ckpts', default='ckpts/fig3', help='folder for trained network checkpoints')
    parser.add_argument('--skip-train', action='store_true', help='re-plot from previously saved estimates')
    args = parser.parse_args()

    out_dir, ckpt_dir = REPO_ROOT / args.out, REPO_ROOT / args.ckpts
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        for mouse_name in args.mice:
            print(f"\n=== {mouse_name}: training + inference ===")
            train_and_infer(mouse_name, out_dir, ckpt_dir)

    print("\n=== Assembling Fig. 3 ===")
    make_figure(args.mice, out_dir)


if __name__ == '__main__':
    main()
