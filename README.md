# Challenge 2: Self-Supervised Learning for 3D Perception

This repository contains the solution for Challenge 2, focusing on implementing a self-supervised learning algorithm (Bootstrap Your Own Latent - BYOL) with the PointNext model on the ScanNet dataset. The goal is to train a self-supervised PointNext encoder and demonstrate how well its representations transfer to a downstream task, specifically semantic segmentation (simplified to scene classification in this demonstration).

## Project Structure

```
.
├── byol_framework.py
├── data_loader.py
├── download-scannet.py
├── evaluation_metrics.py
├── pointnext_model.py
├── pretrain_byol.py
├── scannet_data/             # Directory for downloaded ScanNet data
│   └── scans/
│       └── scene0000_00/
│           ├── scene0000_00_vh_clean_2.ply
│           ├── scene0000_00_vh_clean_2.labels.ply
│           ├── scene0000_00_vh_clean.aggregation.json
│           └── scene0000_00_vh_clean_2.0.010000.segs.json
├── semantic_segmentation_data.py
├── semantic_segmentation_model.py
├── solution_design_report.md
├── train_segmentation.py
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies

It is highly recommended to use a Python virtual environment.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning
pip install open3d
pip install scikit-learn # For evaluation metrics
```

### 3. Download ScanNet Data

Due to the large size of the full ScanNet dataset, this project uses a single scene (`scene0000_00`) for demonstration purposes. You will need to obtain access to the ScanNet dataset by filling out a terms of use agreement on the [ScanNet website](http://www.scan-net.org/ScanNet/) and sending it to `scannet@googlegroups.com`.

Once you have access, download the `download-scannet.py` script and use it to download the specific scene:

```bash
wget http://kaldir.vc.in.tum.de/scannet/download-scannet.py
chmod +x download-scannet.py
./download-scannet.py -o scannet_data --id scene0000_00
```

During the download process, you will be prompted to accept the terms of use. Type `y` and press Enter.

## Usage

### 1. BYOL Self-Supervised Pre-training

First, pre-train the PointNext encoder using the BYOL framework. This script uses a dummy dataset for demonstration. In a real-world scenario, you would replace this with a proper ScanNet dataset loader with robust data augmentation.

```bash
python3.11 pretrain_byol.py
```

This will save the pre-trained encoder weights to `pretrained_pointnext_encoder.pth`.

### 2. Semantic Segmentation (Scene Classification) Fine-tuning

Next, fine-tune a linear classification head on top of the frozen pre-trained PointNext encoder for the semantic segmentation task (simplified to scene classification).

```bash
python3.11 train_segmentation.py
```

This script will load the `scene0000_00` data, train the segmentation head, and report classification metrics.

## Report

A detailed report describing the implementation, design choices, challenges faced, results, and potential improvements can be found in `solution_design_report.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

*   ScanNet dataset: [http://www.scan-net.org/ScanNet/](http://www.scan-net.org/ScanNet/)
*   PointNext paper: [https://arxiv.org/abs/2206.04670](https://arxiv.org/abs/2206.04670)
*   BYOL paper: [https://arxiv.org/abs/2006.07733](https://arxiv.org/abs/2006.07733)


