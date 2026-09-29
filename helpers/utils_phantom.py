import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_vials_hough_seeded_watershed(data, visualize=False):
    """
    Segment vials in a 3D image using Hough Circle Transform and seeded watershed algorithm.
    
    Parameters:
    - data: 4D numpy array with shape (1, height, width, channels), where channels is 1 for grayscale images.
    
    Returns:
    - mask: 2D numpy array with the segmented mask of the vials.
    """
    
    if data.ndim != 4 or data.shape[3] != 1: # or data.shape[0] != 1 
        raise ValueError("Input data must be a 4D numpy array with shape (?, height, width, 1).")    

    # Load image in grayscale
    img = data[0, :, :, 0] / np.max(data[0, :, :, 0])  # Normalize to [0, 1]
    img = (img * 255).astype(np.uint8)  # Convert to uint8 for OpenCV

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Hough circle detection to get center seeds
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=20
    )

    # Prepare mask for flood-filled regions
    mask = np.zeros_like(img, dtype=np.uint8)
    rois_nan = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # Make a temporary mask for floodFill (must be 2 pixels larger)
            temp_mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

            # Only flood-fill bright regions inside the circle (inside vial)
            seed_val = int(img[y, x])
            lo_thresh = 25
            hi_thresh = 25

            # Flood-fill writes to img in-place unless you clone it
            flood_fill_flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

            cv2.floodFill(
                image=img.copy(),          # use a clone if you don't want to alter input
                mask=temp_mask,
                seedPoint=(x, y),
                newVal=0,                  # ignored since we use MASK_ONLY
                loDiff=lo_thresh,
                upDiff=hi_thresh,
                flags=flood_fill_flags
            )                        

            # Combine result (cut back to original shape)
            mask = cv2.bitwise_or(mask, temp_mask[1:-1, 1:-1] // 255)
            
            roi_nan_vial = np.float32(temp_mask[1:-1, 1:-1] // 255)
            roi_nan_vial[np.where(roi_nan_vial==0)] = np.nan 
            rois_nan.append(roi_nan_vial)
        
    mask_nan = np.where(mask == 1, 1.0, np.nan)
    inv_mask_nan = np.where(mask == 0, 1.0, np.nan)  # Inverted mask for visualization
    
    if visualize:
        plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.imshow(mask_nan, cmap="gray")
        plt.title("'in-vial' mask")
        plt.subplot(132)
        plt.imshow(data[1, :, :, 0]*mask_nan, vmin=0, vmax=np.max(data))
        plt.title("masked image")
        plt.subplot(133)
        plt.title("inverse masked image")
        plt.imshow(data[1, :, :, 0]*inv_mask_nan, vmin=0, vmax=np.max(data))
        print(circles)
        
    
    return [(x,y) for (x,y,r) in  circles], rois_nan, mask_nan #  mask_nan, rois_nan