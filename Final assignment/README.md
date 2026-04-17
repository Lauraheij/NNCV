# Evaluating DINOv2 Architectures for Semantic Segmentation: Trade-offs in Performance andEfficiency
Laura Heij
Department of Mathematics and Computer Science
Eindhoven University of Technology
l.a.f.heij@student.tue.nl

## Overview
This repository contains all code for the final assignment of 5LSM0. The project approaches semantic segmentation of urban street scenes (Cityscapes) from two complementary angles:

- Peak Performance: Maximising segmentation quality using a DINOv2 ViT-B/14 backbone with two decoder heads (Linear and multi-scale DPT Fusion).
- Efficiency: Achieving the best possible quality-per-FLOP trade-off using a compact DINOv2 ViT-S/14 backbone with a lightweight EfficientFusionHead, reduced input resolution, and optional knowledge distillation from the peak model.


## Environment Setup
The Environmental Setup and data downloading is based on the `README-Installation.md` and `README-slurm.md`. The approach I used was to connect to snellius via the SSH key, here I created a virtual environment downloading the extra required packages for my models:

``` pip install torch torchvision pillow torchmetrics timm wandb ```

These libraries are also added to the `Dockerfile`, for server submission correctness. 

To ensure that the DinoV2 would not be internet dependent I cloned the DINOv2:

```git clone https://github.com/facebookresearch/dinov2.git dinov2_hub```

Followed by downloading the DinoV2 ViT-B/14 and ViT-S/14 backbone weights:
```wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth```
```wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth```

## Data Preproccesing
There are no data preprocessing steps needed. The downloaded data is only preprocessed in the scripts itself and the training/validation and test split was already given by the course instructors.

## Training
Training the models was done via SLURM queue:
```sbatch jobscript_slurm.sh```

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

As it is cumbersome approach I changed the model I wanted to train to the `model.py` as well as the corresponding `train.py` and `predict.py`.
During the development this was no issue as I trained it and then properly saved it with a relevent name. For reprosobility I now see that this is not the most usefull approach (excuses).


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

