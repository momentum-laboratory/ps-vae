</div>
(https://arxiv.org/abs/2602.03317)
 <div>
【<a href='https://github.com/falex-aimri' target='_blank'>Alex Finkelstein </a> |
<a href='https://github.com/operlman' target='_blank'>Or Perlman </a>】
<div>
<a href='https://mri-ai.github.io/' target='_blank'>Momentum Lab, Tel Aviv University</a>
</div>

# Physics-Structured Variational AutoEncoder (PS-VAE)

This repository is an extension of [Neural Bloch-McConnell Fitting](https://github.com/momentum-laboratory/neural-fitting) towards rapid and rigorous Bayesian posterior modeling in quantitative molecular MRI, via the technique coined in the title.  The methods and results (using scans at 3T,7T,9.4T of phantom, mice, healthy-humans, cancer/AD patients) re reported in the paper [Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE)](https://arxiv.org/abs/2602.03317) available on Arxiv and accepted to IEEE Transactions on Medical Imaging (in-press).

## ⚡ Getting Started

### Setting up the environment

For the fast speeds reported, the workstation must have a GPU installed with appropriately updated drivers and CUDA middleware. Tested with NVIDIA GeForce RTX 3060, Driver Version: 550.120, CUDA Version: 12.4 .
The below assumes a Linux machine (we tested on Ubuntu22).

The environment can be created using conda with a single duplication command:
```bash
conda env create -f environment.yml -n psvae_env
```

Alternatively, create and activate an environment using any suitable manager, e.g.:

(a) conda:
```
conda create -n psvae_env --no-default-packages python=3.12
conda activate psvae_env
```
(b) virtualenv:
```
virtualenv .venv
source .venv/bin/activate
```

and then in the target environment run:
```bash
pip install -r requirements.txt
```

All notebooks and scripts below are run from the repository root (notebooks also work when launched from `notebooks/`); output folders (`figs/`, `ckpts/`) are created as needed.

## 🔬 Transparency and Reproducibility of the paper's Methods and Results

The results in the paper fall into two groups: those that can be regenerated from scratch with the data shipped in this repository (**A**), and those that depend on human data that cannot be shared, which are provided as executed notebooks showing each figure next to the code that produced it (**B**).

### A. Runnable demos

| | Entry point | Data (in repo) | Reproduces |
|---|---|---|---|
| A1 | `notebooks/demo_human.ipynb` | `data/xarr_sample.nc`: one slice (T1, T2, MT/CEST-MRF scans) from each of 4 healthy volunteers (3T) | Human MT/amide quantification with uncertainty maps and posterior validation (as in panels of Fig. 4) |
| A2 | `notebooks/demo_mouse.ipynb` | `data/<mouse>.nc`: single-slice scans of tumor-bearing mice (7T) | Full fitting and analysis of one mouse; Fig. 9; per-mouse results aggregated in Figs. 5, 8 |
| A3 | `scripts/reproduce_fig3.py` | the three mice shown in Fig. 3 | **Fig. 3, in one command, including training** |
| A4 | `notebooks/demo_phantom.ipynb` | `data/phantom9T_data.mat`: L-arginine phantom (9.4T), 3 vials of known concentration and pH | In-vitro quantification with uncertainty, and posterior shrinkage along the acquisition |
| A5 | `notebooks/simulative.ipynb` | none (synthetic data generated on the fly by the simulation) | Validations and ablations in a controlled, ground truth-based setup (Fig. 2) |

**A1: Human, pretrained network** (`notebooks/demo_human.ipynb`).
By default the notebook loads the provided checkpoints of the MT and amide Bayesian quantification networks, pretrained on full brain volumes of three healthy volunteers (`ckpts/healthy/vol7-9-10_2026-04-28_h22-m45`), and applies them to the public sample. It then shows the parameter maps, univariate confidence maps and per-voxel multivariate posteriors, and compares the network's posteriors against likelihood mapping on a reference grid. Set `do_train = True` in the parameters cell to retrain both networks on the sample itself.

**A2: Mouse, full pipeline** (`notebooks/demo_mouse.ipynb`).
Self-supervised training of the two-stage network (semisolid MT, then amide) on a single mouse scan, inference, and the complete uncertainty analysis: CI maps, tumor vs. contralateral posteriors, whole-slice comparison against the reference posterior, and the protocol/acquisition-length analysis of Fig. 9. Repeated runs created inputs for the aggregate all-mice results reported in Figs. 5 and 8.


**A3: Fig. 3 in one click** (`scripts/reproduce_fig3.py`).
```bash
python scripts/reproduce_fig3.py
```
For each of the three mice in Fig. 3, trains the network on the mouse's own scan, estimates the tissue parameters with their posterior, and assembles the figure, including the reference posteriors of the two marked voxels per mouse, into `figs/fig3/fig3.{png,svg}`. The result matches the paper up to training stochasticity.

**A4: Simulation study** (`notebooks/simulative.ipynb`).
Trains on synthetic signals with known ground truth and evaluates accuracy and calibration (coverage) of the posteriors, with optional model misspecification (`do_test_bias`). The hyperparameter sweeps behind the paper's figure are run via `scripts/run_papermill.py` (`run_simulative()`); the figure itself is assembled in `notebooks/revision_figs.ipynb`.

**A5: Phantom** (`notebooks/demo_phantom.ipynb`).
Trains on the phantom scan and compares the estimates and their posteriors against the known vial properties and against likelihood mapping.

### B. Executed notebooks (human data not shared)

Figures based on human-subject data (full volunteer volumes and patient scans) cannot be regenerated from scratch here, because that data cannot be shared. For full transparency these figures are provided in context: the notebooks below are committed **with their outputs**, so every figure, table and reported statistic appears directly below the code that produced it. A reader, human or AI agent, can trace each result back through the aggregation code to the per-subject runs (A1/A2 and the `scripts/run_papermill.py` drivers) and the core library, and can re-execute them given the original or compatible data in the same format.

| Notebook | Contents |
|---|---|
| `notebooks/summary.ipynb` | Aggregation of all per-subject runs into the paper's tables and statistics; final figure assembly for mice (Fig. 3), human volunteers (Fig. 4), overall performance statistics (Fig. 6) and the application figure (Fig. 8); Video 1 |
| `notebooks/revision_figs.ipynb` | Validations and ablations added in revision: simulation study (Fig. 2), diagonal-covariance, MC-dropout and ensemble ablations, patients (Fig. 5), Fig. 7, inference speed benchmarks |

Each figure is produced in a cell under a heading of the form `FINAL FIGURE (Fig. N)`, which makes it easy to search for - before tracing back the paper trail (or code-trail:).

## 🚀 Contributing
We believe in openly sharing information, data, code and ideas between research groups. Whether you have a question, suggestion or a bug to fix, please let us know. See our group website at: https://mri-ai.github.io/

## 📑 References
If you use this code for research or software development please reference the following publication:
``` # TO CHANGE
Finkelstein, Alex, Ron Moneta, Or Zohar, Michal Rivlin, Moritz Zaiss, Dinora Friedmann Morvinski, and Or Perlman. "Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE)." arXiv preprint arXiv:2602.03317 (2026)‏
```
