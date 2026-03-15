"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import time
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode
)

from torchvision.transforms import v2 
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO: change to mac?

    # Replace your current img_transform and target_transform with this:
    train_transforms = v2.Compose([
        v2.ToImage(),
        # Fixed aspect ratio (Height, Width)
        v2.Resize((252, 518), interpolation=v2.InterpolationMode.BILINEAR), #for Dino which is 14x14
        # Essential for Peak Performance:
        v2.RandomHorizontalFlip(p=0.5), 
        # Use ImageNet normalization (Standard for pre-trained models)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Validation doesn't get flips
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((252, 518), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
    train_dataset = Cityscapes(
    args.data_dir, 
    split="train", 
    mode="fine", 
    target_type="semantic", 
    transforms=train_transforms
    )

    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=val_transforms
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    wandb.log({"total_params": total_params, "trainable_params": trainable_params})

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer (freeze backbone)
    optimizer = AdamW([
        {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': args.lr * 0.1},
        {'params': model.head.parameters(), 'lr': args.lr}
    ])    
    # Make the learning rate dyamical
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=args.epochs, 
                                                           eta_min=1e-5) # The lr can't go lower
    
    # Metric IOU #TODO: why ignore_index 255?
    iou_metric = MulticlassJaccardIndex(
        num_classes=19, 
        ignore_index=255, 
        average=None # Returns per class instead of the mean
        ).to(device)
    
    # Metric DICE
    dice_metric = MulticlassF1Score(
        num_classes=19,
        ignore_index=255,
        average=None
        ).to(device)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    patience = 5
    epochs_without_improvement = 0
    best_miou = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        epoch_start = time.time()

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                iou_metric.update(preds, labels)
                dice_metric.update(preds, labels)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            per_class_iou = iou_metric.compute()
            per_class_dice = dice_metric.compute()
            epoch_miou = per_class_iou.mean()
            epoch_mdice = per_class_dice.mean()
            epoch_time = time.time() - epoch_start

            wandb.log({
                "valid_loss": valid_loss,
                "val_mIoU": epoch_miou.item(),
                "val_mDice": epoch_mdice.item(),
                # Per category IoU
                "IoU_flat": per_class_iou[0:2].mean().item(),
                "IoU_construction": per_class_iou[2:5].mean().item(),
                "IoU_object": per_class_iou[5:8].mean().item(),
                "IoU_nature": per_class_iou[8:10].mean().item(),
                "IoU_sky": per_class_iou[10].item(),
                "IoU_human": per_class_iou[11:13].mean().item(),
                "IoU_vehicle": per_class_iou[13:19].mean().item(),
                # Per category Dice
                "Dice_flat": per_class_dice[0:2].mean().item(),
                "Dice_construction": per_class_dice[2:5].mean().item(),
                "Dice_object": per_class_dice[5:8].mean().item(),
                "Dice_nature": per_class_dice[8:10].mean().item(),
                "Dice_sky": per_class_dice[10].item(),
                "Dice_human": per_class_dice[11:13].mean().item(),
                "Dice_vehicle": per_class_dice[13:19].mean().item(),
                # Training efficiency
                "epoch_time_seconds": epoch_time,
            }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            # Reset metric for the next epoch's validation
            iou_metric.reset()
            dice_metric.reset()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
        scheduler.step()
        
        if epoch_miou.item() > best_miou:
            best_miou = epoch_miou.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
