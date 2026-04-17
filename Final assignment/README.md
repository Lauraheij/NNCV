# Evaluating DINOv2 Architectures for Semantic Segmentation: Trade-offs in Performance andEfficiency

## Overview
This repository contains all code for the final assignment of 5LSM0. The project approaches semantic segmentation of urban street scenes (Cityscapes) from two complementary angles:

- **Peak Performance**: Maximising segmentation quality using a DINOv2 ViT-B/14 backbone with two decoder heads (Linear and multi-scale DPT Fusion).
- **Efficiency**: Achieving the best possible quality-per-FLOP trade-off using a compact DINOv2 ViT-S/14 backbone with a lightweight EfficientFusionHead, reduced input resolution, and optional knowledge distillation from the peak model.

## Environment Setup
The environmental setup and data downloading is based on the `README-Installation.md` and `README-slurm.md`. 
Connect to Snellius via SSH key, then create a virtual environment and install the required packages:

``` pip install torch torchvision pillow torchmetrics timm wandb```

These dependencies are also included in the `Dockerfile` to ensure correct server submission.
To run DINOv2 without an internet connection, clone the repository locally and download the pretrained backbone weights:

```git clone https://github.com/facebookresearch/dinov2.git dinov2_hub```

Followed by downloading the DinoV2 ViT-B/14 and ViT-S/14 backbone weights:
``` wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

This makes the ViT-B/14 and ViT-S/14 weights available locally, so no internet access is required at runtime.

## Data Preproccesing
The Cityscapes dataset downloading is part of the `README-Installation.md` instructions, when these steps are followed the data can be found in the folder `data`. No further manual data preprocessing steps are required. The raw data is processed dynamically within the training scripts. The training, validation, and test splits are used as provided by the course instructors.

## Training
The different experiments where run using these settings (this is the main.sh file).

```bash
wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 20 \
    --lr 0.00005 \
    --num-workers 10 \
    --seed 11 \
    --experiment-id "experiment-name" \
```

Training the models was done via SLURM queue:
```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

During development, specific architectures (backbone type and decoder head) were selected by manually modifying the Model class in model.py and the corresponding logic in train.py. For exact reproduction of specific results, ensure the configuration in model.py matches the desired architecture before submitting the SLURM job. I apologize for this practical limitation; while this workflow supported rapid iteration during the development phase, I recognize it is not the most efficient approach for seamless reproducibility.

## Code Structure

### Efficiency Model

The lightweight efficiency model is built around the DINOv2 ViT-S/14 backbone. The relevant files are:

- `model_DinoV2_vits14.py` — model definition
- `train_vits14.py` — standard training for efficiency
- `train_vits14_teacher.py` — knowledge distillation training
- `predict_vits14.py` — inference (used by both training approaches)

### Peak Performance Models

The higher-capacity models use a DINOv2 ViT-B/14 backbone with either a DPT or linear decoder head. The relevant files are:

- `model_DinoV2_DPT.py` — DPT decoder model definition
- `model_DinoV2_linear.py` — linear decoder model definition
- `train_vitb14_DPT.py` — training for both model variants
- `predict_vitb14.py` — inference for both model variants

## WandB
The tracking for the different models was done via Weights & Biases. These also provided the segmentation images of the qualitative analysis. 
This can be enable through the `.env` file. Adding your correct credentials.


## Server Submissions
Here are my submission as also can be found on the server (sorry for the bad naming in the peak performance branch)
### Peak performance
| Username | Experiment |
| :--- | :--- |
| `Laura_V1` | Baseline U-net |
| `Laura_Dlinear` | DinoV2 ViT-B + linear |
| `Laura_Baseline` | DinoV2 ViT-B + DPT |

### Efficiency
| Username | Experiment |
| :--- | :--- |
| `Laura_Baseline` | Baseline U-net |
| `Laura_DE` | DinoV2 ViT-S |
| `Laura_DET` | DinoV2 ViT-S + distillation |

