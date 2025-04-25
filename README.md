## DepthSense
CS7643 DL Final Project (2025-Spring)
* [Project Proposal](./CS7643__Group_Project_Proposal.pdf)
* [Final Report](https://www.overleaf.com/read/crtkktyccgcr#2d9638)

## Usage
### Installation (Google Colab + Google Drive)
1. Clone this repository in Google Drive (Locally synced)
```
git clone https://github.com/chillymango/depthsense.git
cd depthsense
pip install -r requirements.txt
```
2. (Optional) clone git submodules
```
git submodule init
git submodule update
```
3. Check DEV environment by running ``depthanythingv2-test.ipynb``

### Download DIODE dataset and run benchmark
For our model, download https://drive.google.com/file/d/1yUFoY7kNWWhJRctwy4fWaszTndfdhVRU/view?usp=sharing to depthsense/checkpoints/
```
cd depthsense/data
wget -qO- http://diode-dataset.s3.amazonaws.com/val.tar.gz | tar -xzv
wget -qO- http://diode-dataset.s3.amazonaws.com/val_normals.tar.gz | tar -xzv

cd ../
./benchmark.sh # Adjust DATA_DIR and model
python gen_visual_comparison.py
```

## Project Structure
```
depthsense/
├── benchmark_diode_depthanythingv2_results.json
├── benchmark_diode_depthsense_results.json
├── benchmark.py                                        # Benchmark scripts
├── benchmark.sh
├── checkpoints                                         # Pre-trained model/checkpoints
│   ├── checkpoint_epoch_124.pt
│   ├── checkpoint_epoch_7.pt
│   ├── depth_anything_v2_metric_hypersim_vitl.pth
│   ├── depth_anything_v2_vitl.pth
│   └── dinov2_vitg14_pretrain.pth
├── depthsense_head.py                                  # Dual-head decoder for depth and normals
├── dinov2_layers                                       # ViT model building blocks (unchanged from Depth-Anything-V2)
│   ├── attention.py
│   ├── block.py
│   ├── drop_path.py
│   ├── __init__.py
│   ├── layer_scale.py
│   ├── mlp.py
│   ├── patch_embed.py
│   └── swiglu_ffn.py
├── dinov2.py                                           # DINOv2 backbone wrapper
├── dpt.py                                              # Contains the DepthSense model class
├── fetch_and_transform_hypersim.py                     # Preprocessing Hypersim dataset
├── generate_pseudo_labels.py                           # Creates pseudo-labels for student model's training data
├── gen_visual_comparison.py                            # Generate visual comparison (DepthAnythingV2 vs. Depthsense)
├── student_training.py                                 # Student training script
├── training.py                                         # Configurable training script for teacher / student models
├── util
│   ├── blocks.py                                       # Feature fusion utilities
│   ├── common.py                                       # Common utilities
│   ├── loss.py                                         # Loss function
│   └── transform.py                                    # Image preprocessing transforms
├── visualizations.zip
└── visualize.ipynb
```

## LICENSE
DepthSense code and model are under the MIT license. See [LICENSE](./LICENSE) for additional details.
