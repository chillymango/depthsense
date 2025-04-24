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
```

## Project Structure
```
depthsense/
├── dinov2_layers/       # ViT model building blocks (unchanged from Depth-Anything-V2)
│   ├── attention.py
│   ├── block.py
│   ├── drop_path.py
│   ├── layer_scale.py
│   ├── mlp.py
│   ├── patch_embed.py
│   ├── swiglu_ffn.py
│   └── __init__.py
├── depthsense_head.py         # Dual-head decoder for depth and normals
├── dinov2.py                  # DINOv2 backbone wrapper
├── dpt.py                     # Contains the DepthSense model class
├── generate_pseudo_labels.py  # Creates pseudo-labels for student model's training data
├── geonet-code-map.py         # Explain how GeoNet's code.py is mapped to DepthSense
├── training.py                # Configurable training script for teacher / student models
├── util/
│   ├── blocks.py              # Feature fusion utilities
│   ├── loss.py                # Loss function
│   └── transform.py           # Image preprocessing transforms
```

## LICENSE
DepthSense code and model are under the MIT license. See [LICENSE](./LICENSE) for additional details.
