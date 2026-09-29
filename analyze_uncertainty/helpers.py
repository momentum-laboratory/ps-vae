import os
import imageio
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import chi2
import matplotlib.patches as patches
import cv2
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf_2D(
        mu_fk, cov_fk, grid_fk=None, 
        fmax=None, df=None, kmax=None, dk=None
    ):
    """Compute 2D Gaussian PDF at points (x, y) with given mean and covariance.       
    Args:
        mu_fk: Mean vector (2, ) in f-k space.
        cov_fk: Covariance matrix (2, 2) in f-k space.
        grid_fk: Optional grid of points (M, N, 2) where to evaluate the PDF.
        fmax: Maximum f value for grid generation if grid_fk is None.
        df: Step size in f for grid generation if grid_fk is None.
        kmax: Maximum k value for grid generation if grid_fk is None.
        dk: Step size in k for grid generation if grid_fk is None. 
    """
    grid_fk = grid_fk or np.mgrid[0:fmax:df, 0:kmax:dk].transpose(1, 2, 0)    
    cov_inv = np.linalg.inv(cov_fk)
    denom = 2 * np.pi * np.sqrt(np.linalg.det(cov_fk))

    diff = grid_fk - mu_fk
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, cov_inv, diff)

    return np.exp(exponent) / denom


def farthest_point_sampling(f_val, k_val, num_samples: int = 40):
    """Select parameter pairs via farthest point sampling on the f-k grid.
        Handy for exploring performance at diverse parameter combinations.
    """
    coords = np.column_stack((f_val.flatten(), k_val.flatten()))
    valid_coords = coords[~np.isnan(coords).any(axis=1)]

    selected_points = [valid_coords[np.random.choice(len(valid_coords))]]

    for _ in range(num_samples - 1):
        _, min_distances = pairwise_distances_argmin_min(
            valid_coords, np.array(selected_points)
        )
        next_point = valid_coords[np.argmax(min_distances)]
        selected_points.append(next_point)

    selected_indices = [
        (
            np.where((f_val == point[0]) & (k_val == point[1]))[0][0],
            np.where((f_val == point[0]) & (k_val == point[1]))[1][0],
        )
        for point in selected_points
    ]
    return selected_indices


def gaussian_ellipse(
    points,
    ax=None,
    conf: float = 0.95,
    mahalanobis_radius: float | None = None,
    facecolor: str = "none",
    edgecolor: str = "r",
    **ellipse_kwargs,
):
    """Fit a 2D Gaussian and draw the confidence ellipse."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N,2) array-like")

    mu = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)

    if mahalanobis_radius is None:
        if chi2 is not None:
            r2 = chi2.ppf(conf, df=2)
        else:
            r2 = 5.991 if conf == 0.95 else 5.991
        mahalanobis_radius = float(np.sqrt(r2))

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 * mahalanobis_radius * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **ellipse_kwargs,
    )

    if ax is None:
        _, ax = plt.subplots()
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.6)
        ax.scatter([mu[0]], [mu[1]], c="k", marker="+")
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
    ax.add_patch(ellipse)

    return {
        "mean": mu,
        "cov": cov,
        "ellipse": ellipse,
        "mahalanobis_radius": mahalanobis_radius,
    }


def draw_fat_arrow(fig, start=(0.45, 0.5), end=(0.55, 0.5), label=None, color="gray", fontsize=12):
    """
    Draws a fat arrow with a label on a figure using relative coordinates (0-1).
    """
    # Create a ghost axis that covers the entire figure
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Create the arrow style
    # 'tail_width' controls the shaft thickness, 'head_width'/'head_length' the tip
    arrow_style = 'simple,head_width=15,head_length=15,tail_width=6'
    
    arrow = patches.FancyArrowPatch(
        start, end,
        arrowstyle=arrow_style,
        color=color,
        mutation_scale=1.0,  # Scales the head/tail definitions above
    )
    ax.add_patch(arrow)
    
    if label:
        # Position text slightly above the midpoint of the arrow
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        # Offset y coordinate for label clearance
        offset_y = 0.02 
        
        ax.text(
            mid_x, mid_y + offset_y, label, 
            ha="center", va="bottom", 
            fontsize=fontsize, color=color, fontweight='bold'
        )
    return ax


def create_frames(figure_func, num_frames=10, **kwargs):
    """
    Create frames from repeated figure creation with randomization.
    
    Parameters:
    -----------
    figure_func : callable
        Function that creates a figure. Should return a matplotlib figure object.
    num_frames : int
        Number of frames to generate for the animation
    **kwargs : dict
        Additional keyword arguments to pass to figure_func
    """
    frames = []
    
    for i in range(num_frames):
        print(f"Generating frame {i+1}/{num_frames}...")
        
        # Create figure
        fig = figure_func(**kwargs)
        
        # Save to temporary file
        temp_path = f'/tmp/frame_{i:03d}.png'
        fig.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Read and append frame
        frames.append(imageio.imread(temp_path))
    return frames


def create_animated_gif(figure_func, num_frames=10, output_path='animation.gif', duration=0.1, **kwargs):
    """
    Create an animated GIF from repeated figure creation with randomization.
    
    Parameters:
    -----------
    figure_func : callable
        Function that creates a figure. Should return a matplotlib figure object.
    num_frames : int
        Number of frames to generate for the animation
    output_path : str
        Path where the GIF will be saved
    duration : float
        Duration of each frame in seconds
    **kwargs : dict
        Additional keyword arguments to pass to figure_func
    """
    frames = create_frames(figure_func, num_frames=num_frames, **kwargs)
    
    # Create GIF
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
    print(f"GIF saved to {output_path}")
    
    # Clean up temporary files
    for i in range(num_frames):
        temp_path = f'/tmp/frame_{i:03d}.png'
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return output_path


def save_frames_to_mp4(fig_gen, output_path, fps=10):
    imgs = []
    # Consume generator
    for fig in fig_gen:
        # Capture figure to buffer with bbox_inches='tight' to include outer annotations
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        img = np.array(Image.open(buf))
        buf.close()
        
        # Drop Alpha to get RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3] 
            
        imgs.append(img)
        plt.close(fig) # Important to free memory
    
    if not imgs:
        raise ValueError("No frames were generated.")

    height, width, _ = imgs[0].shape

    # Ensure dimensions are even for H.264/yuv420p
    if height % 2 != 0 or width % 2 != 0:
        height = height if height % 2 == 0 else height - 1
        width = width if width % 2 == 0 else width - 1

    # Standardize image sizes
    standardized_imgs = []
    for img in imgs:
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        else:
            img = img.copy()
        standardized_imgs.append(img)
    
    # Try using imageio with ffmpeg for H.264 (best compatibility)
    success = False
    try:
        # imageio expects RGB
        imageio.mimsave(output_path, standardized_imgs, fps=fps, codec='libx264', pixelformat='yuv420p')
        print(f"Video saved to {output_path} (via imageio/libx264)")
        success = True
    except Exception as e:
        print(f"Warning: Failed to save with imageio/libx264 ({e})...")

    if not success:
        # Fallback to OpenCV
        print("Falling back to OpenCV VideoWriter...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img in standardized_imgs:
            # OpenCV expects BGR
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
        video.release()
        print(f"Video saved to {output_path} (via cv2/mp4v)")
