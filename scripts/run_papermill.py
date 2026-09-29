#!/usr/bin/env python
"""Utility helpers to execute packaged notebooks via Papermill."""

import json
from pathlib import Path
import subprocess
import os
import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
FIGS_DIR = REPO_ROOT / "figs"


def _run_papermill(notebook_name: str, output_path: Path, **params):
    cmd = [
        "papermill",
        str(NOTEBOOKS_DIR / notebook_name),
        str(output_path),
    ]
    for key, value in params.items():
        cmd.extend(["-p", key, str(value)])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_human_leave_one_out(output_dir: str | os.PathLike = "human_papermill_runs", volumes2do = [7, 8, 9, 10], **params):
    """Execute the human demo notebook over all leave-one-out combinations."""
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / FIGS_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_volumes = [7, 8, 9, 10]
    # Leave-one-out configuration: train on 3, test on 1    
    # Generate all leave-one-out splits
    splits = []
    for test_vol in volumes2do:
        train_vols = [v for v in all_volumes if v != test_vol]
        train_set = f"vol{train_vols[0]}-{train_vols[1]}-{train_vols[2]}"
        test_set = f"vol{test_vol}"
        splits.append((train_set, test_set))

    print(f"Running {len(splits)} leave-one-out experiments:")
    for trainset, testset in splits:
        print(f"  Train: {trainset}, Test: {testset}")

    # Run papermill for each split
    for trainset, testset in splits:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_h%H-m%M")
        output_notebook = output_dir / f"demo_human_{testset}_trained_on_{trainset}_{timestamp}.ipynb"
        
        print(f"\nRunning: Train on {trainset}, Test on {testset}")
        print(f"Output: {output_notebook}")

        try:
            _run_papermill(
                "demo_human.ipynb",
                output_notebook,
                TRAINSET=trainset,
                TESTSET=testset,
                **params,
            )
            print(f"✓ Completed: {testset}")
        except subprocess.CalledProcessError as exc:
            print(f"✗ Failed: {testset} - {exc}")

    print(f"\nAll runs completed. Results saved to {output_dir}/")


def run_mice(
    output_dir: str | os.PathLike = "mice_papermill_runs",
    mice_names = [
            'ped_C1_1L_20241113_1245', 
            'ozohar_right_20240404_125156',
            'ozohar_two_ears_20240404_091158',
            'ped_Control_None_2025-01-09',
            'ped_Control_1L1R_2025-01-09',
            'ped_Control_2L_2025-01-09',
            'ped_Control_2R_2025-01-09',                   
    ],
    **params    
    ):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / FIGS_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for mouse_name in mice_names:
        output_notebook = output_dir / f"demo_mouse_{mouse_name}.ipynb"        
        print(f"\nRunning: {mouse_name}")
        print(f"Output: {output_notebook}")        
        try:
            _run_papermill(
                "demo_mouse.ipynb",
                output_notebook,
                mouse_name=mouse_name,
                **params,
            )
            print(f"✓ Completed: {mouse_name}")
        except subprocess.CalledProcessError as exc:
            print(f"✗ Failed: {mouse_name} - {exc}")
    print(f"\nAll runs completed. Results saved to {output_dir}/")
    

def run_patients(    
    output_dir: str | os.PathLike = "patient_papermill_runs",    
    patient_runs: tuple=(
            ('erlangen', {'drop_first': True, 'sli': 27, 'do_amide': False}),
            ('ichilov_alz', {'drop_first': False, 'sli': 3, 'do_amide': False}),
    ),
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh") #%M%S")
    print(f"Timestamp for this run: {timestamp}")        
    output_dir += f"_{timestamp}"
    output_dir = Path(output_dir)    
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / FIGS_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json.dump(patient_runs, open(output_dir / "patient_runs.json", "w"))
    for patient_name, params in patient_runs:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh") #%M%S")
        output_notebook = output_dir / f"{patient_name}_{timestamp}.ipynb"   
        _run_papermill(
                    "demo_patient.ipynb",
                    output_notebook,
                    patient_name=patient_name,
                    **params,
                )
        
def run_simulative(output_dir: str | os.PathLike = "simulative_papermill_runs"):
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh") #%M%S")
    print(f"Timestamp for this run: {timestamp}")            
    output_dir += f"_{timestamp}"
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / FIGS_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SWEEP THE ALPHA HYPERPARAMETER:
    #for std_up_fact in [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]:
    #   kwargs = {'std_up_fact': std_up_fact}
    
    #for (do_test_bias, force_diag, num_test_voxels_x100) in [(False, True, 1), (False, False, 5), (True, False, 5)]:
    for (do_test_bias, force_diag, num_test_voxels_x100) in [(False, False, 1), (True, False, 1)]:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") #%S")
        output_notebook = output_dir / f"simulative_{timestamp}.ipynb"
        params = {
            'do_test_bias': do_test_bias,
            'force_diag': force_diag,
            'num_test_voxels_x100': num_test_voxels_x100
        }
        _run_papermill(
            "simulative.ipynb",            
            output_notebook,            
            output_dir=output_dir,
            **params
        )
    
    
if __name__ == "__main__":
    run_mice()
    run_human_leave_one_out()   
    run_patients()
    run_simulative()
        
    # ! see prior versions for alternative experiments logs.
    