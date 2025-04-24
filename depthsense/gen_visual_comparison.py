import os
import shutil
import glob
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

target_dir = "visualizations"

# Define base paths
base_path = "/home/hice1/ylee904/scratch/depthsense/depthsense/output/diode"
da_path = os.path.join(base_path, 'depthanythingv2')
ds_path = os.path.join(base_path, 'depthsense')
gt_path = os.path.join(da_path, 'GT')  # GT can come from either model

# Create output directory for plots
os.makedirs(target_dir, exist_ok=True)

# Find all frame images from DepthAnythingV2
frame_files = sorted(glob.glob(os.path.join(gt_path, '*_frame.png')))

# Expected dimensions
img_width = 252
img_height = 196
dpi = 100
num_cols = 7
fig_width = num_cols * img_width / dpi
fig_height = img_height / dpi

for frame_path in tqdm(frame_files, desc="Generating comparisons"):
    index = os.path.basename(frame_path).split('_')[0]

    images = [
        os.path.join(gt_path, f"{index}_frame.png"),
        os.path.join(gt_path, f"{index}_depth_gt.png"),
        os.path.join(gt_path, f"{index}_normal_gt.png"),
        os.path.join(da_path, f"{index}_depth.png"),
        os.path.join(da_path, f"{index}_normal.png"),
        os.path.join(ds_path, f"{index}_depth.png"),
        os.path.join(ds_path, f"{index}_normal.png")
    ]

    try:
        imgs = [Image.open(img_path) for img_path in images]
    except FileNotFoundError as e:
        print(f"Skipping {index}: {e}")
        continue

    # Create black background canvas
    fig, axs = plt.subplots(1, num_cols, figsize=(fig_width, fig_height), dpi=dpi, facecolor='black')
    for ax, img in zip(axs, imgs):
        ax.imshow(img)
        ax.set_facecolor('black')  # ensures inner border is black
        ax.axis('off')

    # Adjust spacing to show black lines
    plt.subplots_adjust(wspace=0.02, hspace=0)

    plt.savefig(f'visualizations/{index}_comparison.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("Saved all visualizations in 'visualizations/' folder.")

if not os.path.exists(target_dir):
    raise FileNotFoundError(f"Directory '{target_dir}' not found.")

# Create ZIP archive
shutil.make_archive(target_dir, 'zip', target_dir)

print(f"Zipped '{target_dir}/' into '{target_dir}.zip'")