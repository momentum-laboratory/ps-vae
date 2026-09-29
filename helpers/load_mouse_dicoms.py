
import pydicom, os
import pydicom as dcm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr


### --- for usage in online manual tools
def create_binary_mask(json_path='/home/alexf/Downloads/mouse(1).json', image_shape=(64, 64)):
    """ Create mask from json polygon file as created with online tool 
    """
    import json, cv2
    masks, polygons = [], []
    with open(json_path, 'r') as f:
        data = json.load(f)
    for ii, polygon_item in enumerate(data):
        mask = np.zeros(image_shape, dtype=np.uint8) # black background
        polygon = polygon_item['content']
        polygon_points = [(it['x'], it['y']) for it in polygon]
        polygon_points = np.array([polygon_points], dtype=np.int32)                
        cv2.fillPoly(mask, polygon_points, color=1)
        masks.append(mask)
        polygons.append(np.concatenate([polygon_points[0], polygon_points[0][:1]]))
    return masks, polygons

def get_paravision_map_ms(path_to_paravision_dicom_folder: Path):
    """
    :param path_to_paravision_dicom_folder: points to folder "2" in the paravision-reconstrcuted T1/T2 map folder
    :return: paravision_map_ms - rescaled to ms units
    """
    data = (dcm.dcmread(path_to_paravision_dicom_folder / Path('MRIm3.dcm')))
    paravision_map_int15 = data.pixel_array
    paravision_map_double = paravision_map_int15.astype(np.double)
    paravision_map_ms = paravision_map_double * data.RescaleSlope

    return paravision_map_ms  # map_ms

def get_ordered_DICOM_files(path: Path) -> list:
    """
    Get ordered DICOM files from a given path
    :param path: Path to the DICOM files
    :return: List of ordered DICOM files
    """
    dicom_path = Path(path) / Path("pdata/1/dicom")

    # Sort files by numerical order based on their suffix
    dicom_files = sorted(dicom_path.glob('MRIm*.dcm'), key=lambda x: int(x.stem.replace('MRIm', '')))
    # actual dicom processing to numpy arrays
    scans = [pydicom.dcmread(im).pixel_array.astype(float) for im in dicom_files] 
    return scans

def load_mrf_data(mrf_path: Path) -> xr.DataArray:
    """
    Load the MRF data (at least 31 images) and inflates it to be of shape (31, height, 4, width)
    :param mrf_path: Path to the MRF maps
    :return: DataArray of inflated MRF images
    """
    images = get_ordered_DICOM_files(mrf_path)
    if len(images) < 31:
        raise ValueError(
            f"Not enough images found for MRF data in {mrf_path}. "
            f"Found {len(images)}, expected at least 31.")
    # Keep only the last 31 images (removes Dummy images from start if needed)
    images = images[-31:]

    mrf_data = np.stack(images, axis=0)
    mrf_data = np.expand_dims(mrf_data, axis=2)  # Add a new axis to the array
    return xr.DataArray(mrf_data, dims=("MRF_cycles", "height", "slice", "width"))

def get_qmaps(t_map_path: Path) -> xr.DataArray:
    """
    Generates quantitative map for T1/T2. Folder number 4 should be usd, but if doesn't exist, folder number 2 is used.
    :param t_map_path: Path to the T1/T2 map folder
    :return: Quantitative map
    """
    folder_4 = t_map_path / Path('pdata/4')
    folder_2 = t_map_path / Path('pdata/2')

    if os.path.isdir(folder_4):
        qmap = get_paravision_map_ms(folder_4 / Path('dicom'))
    elif os.path.isdir(folder_2):
        qmap = get_paravision_map_ms(folder_2 / Path('dicom'))
    else:
        assert 0 , 'no folder 2 or 4 found in t_map_path'
    qmap = np.expand_dims(qmap, axis=1)  # Add a new axis to the array
    return xr.DataArray(qmap, dims=("height", "slice", "width"))


def load_and_concat_xarrays(base_path: Path, subfolder_dict: dict = None) -> tuple:
    """
    Load xarrays from specified subfolders and concatenate them into a single Dataset.
    Also loads mask files (*mask.npy) from base_path, squeezes and downsamples to 64x64.
    
    :param base_path: Base path containing subfolders and mask files
    :param subfolder_dict: Dict mapping data types to subfolder indices
        (default: {'t1': 5, 'amide': 6, 'mt': 7, 't2': 11})
    :return: xarray.Dataset with all data and masks as 2D (height, width) arrays
    """
    if subfolder_dict is None:
        subfolder_dict = {'t1': '5', 'amide': '6', 'mt': '7', 't2': '11'}
    
    base_path = Path(base_path)
    
    # Load data from specified subfolders
    t1_path = base_path / Path(subfolder_dict['t1'])
    t2_path = base_path / Path(subfolder_dict['t2'])
    amide_path = base_path / Path(subfolder_dict['amide'])
    mt_path = base_path / Path(subfolder_dict['mt'])
    
    t1 = get_qmaps(t1_path)
    t2 = get_qmaps(t2_path)
    amide = load_mrf_data(amide_path)
    mt = load_mrf_data(mt_path)
    
    ref_shape = (64, 64) #  t1.shape[:2]  # Target shape for masks: (64, 64)
    
    # Create base dataset
    dataset_vars = {
        'T1ms': t1,
        'T2ms': t2,
        'AMIDE_data': amide,
        'MT_data': mt,
    }
    
    # Load and process mask files
    mask_files = sorted(base_path.glob('*mask.npy'))
    
    for mask_file in mask_files:
        mask_name = mask_file.stem.replace('_mask', '').replace('mask', '')
        if mask_name == '':
            mask_name = 'mask'
        
        mask_data = np.load(mask_file)
        
        # Squeeze all singleton dimensions
        mask_data = np.squeeze(mask_data)
                        
        # Ensure 2D
        if mask_data.ndim != 2:
            continue  # Skip if can't reshape to 2D
            print('WARNING: mask is not 2D')
        
        # Downsample from 128x128 to 64x64 if needed
        if mask_data.shape == (128, 128) and ref_shape == (64, 64):
            mask_data = mask_data[::2, ::2]
        
        # Add to dataset as 2D
        dataset_vars[mask_file.stem] = (['height', 'width'], mask_data)
    
    combined_ds = xr.Dataset(dataset_vars)
    return combined_ds


def visualize_dataset(ds: xr.Dataset, figsize=(16,6)) -> None:
    """
    Visualize the loaded dataset with main data types in top row and masks in bottom row.
    
    :param ds: xarray Dataset containing T1ms, T2ms, AMIDE_data, MT_data, and optional masks
    """
    # Identify mask variables (anything not in main data types)
    main_vars = {'T1ms', 'T2ms', 'AMIDE_data', 'MT_data', 'roi_mask_nans'}
    mask_vars = [v for v in ds.data_vars if v not in main_vars]
    n_masks = len(mask_vars)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    # --- TOP ROW: Main data types ---
    # T1 map
    ax = axes[0, 0]
    t1_slice = ds['T1ms'].values[:, 0, :]
    im1 = ax.imshow(t1_slice, vmin=600, vmax=2500, cmap='viridis')
    ax.set_title('T1 Map (ms)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax)
    
    # T2 map
    ax = axes[0, 1]
    t2_slice = ds['T2ms'].values[:, 0, :]
    im2 = ax.imshow(t2_slice, vmax=150, cmap='viridis')
    ax.set_title('T2 Map (ms)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax)
    
    # AMIDE MRF
    ax = axes[0, 2]
    amide_slice = ds['AMIDE_data'].values[29, :, 0, :]
    im3 = ax.imshow(amide_slice, cmap='gray')
    ax.set_title('AMIDE MRF (cycle 29)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax)
    
    # MT MRF
    ax = axes[0, 3]
    mt_slice = ds['MT_data'].values[29, :, 0, :]
    im4 = ax.imshow(mt_slice, cmap='gray')
    ax.set_title('MT MRF (cycle 29)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax)
    
    # --- BOTTOM ROW: Masks ---
    for idx, mask_var in enumerate(mask_vars):
        ax = axes[1, idx]
        mask_slice = ds[mask_var].values
        im = ax.imshow(mask_slice, cmap='gray')
        ax.set_title(f'{mask_var}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)
    
    # Hide unused bottom row panels
    for idx in range(n_masks, 4):
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nDataset variables: {list(ds.data_vars)}")
    print(f"\nData shapes:")
    print(f"  T1ms: {ds['T1ms'].shape}")
    print(f"  T2ms: {ds['T2ms'].shape}")
    print(f"  AMIDE_data: {ds['AMIDE_data'].shape}")
    print(f"  MT_data: {ds['MT_data'].shape}")
    if mask_vars:
        print(f"\nMasks:")
        for mask_var in mask_vars:
            print(f"  {mask_var}: {ds[mask_var].shape}")

